"""Endpoint definition in both examples: a threshold, a recorded class, or the measurement.

Each example owns its endpoint -- which column, which threshold, how many classes --
because those are the cohort's decisions. What is checked here is that every way of
defining one is either built correctly or refused before any data is read, and that
anything derived from labels comes from the training fold alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from examples.cardiovae_multimodal import data as cardiovae_data
from examples.cardiovae_multimodal import runner as cardiovae_runner
from examples.lsemvae_ecg import data as lsemvae_data
from examples.lsemvae_ecg import runner as lsemvae_runner
from kalecardiac.evaluate import SplitPredictions
from kalecardiac.model.predict import class_weight_from_labels
from kalecardiac.pipeline import ClassificationTask, RegressionTask
from tests.examples.test_example_workflows import cardiovae_cfg, lsemvae_cfg

#: The column a synthetic run records its class in, when one is asked for.
SUBTYPE = "subtype"

EXAMPLES = [
    pytest.param(cardiovae_cfg, cardiovae_data, cardiovae_runner, id="cardiovae"),
    pytest.param(lsemvae_cfg, lsemvae_data, lsemvae_runner, id="lsemvae"),
]


def three_class(make_cfg, tmp_path, **overrides):
    """A configuration whose endpoint is a three-class column."""
    return make_cfg(tmp_path, **{"ENDPOINT.LABEL_COLUMN": SUBTYPE, "ENDPOINT.NUM_CLASSES": 3, **overrides})


@pytest.mark.parametrize(("make_cfg", "data", "runner"), EXAMPLES)
class TestCheckEndpoint:
    def test_the_published_endpoint_passes(self, make_cfg, data, runner, tmp_path):
        data.check_endpoint(make_cfg(tmp_path))

    def test_a_three_class_column_passes(self, make_cfg, data, runner, tmp_path):
        data.check_endpoint(three_class(make_cfg, tmp_path))

    def test_fewer_than_two_classes_is_refused(self, make_cfg, data, runner, tmp_path):
        with pytest.raises(ValueError, match="at least 2"):
            data.check_endpoint(make_cfg(tmp_path, **{"ENDPOINT.NUM_CLASSES": 1}))

    def test_a_threshold_cannot_make_three_classes(self, make_cfg, data, runner, tmp_path):
        with pytest.raises(ValueError, match="defines two classes, not 3; set ENDPOINT.LABEL_COLUMN"):
            data.check_endpoint(make_cfg(tmp_path, **{"ENDPOINT.NUM_CLASSES": 3}))

    @pytest.mark.parametrize("overrides", [{"ENDPOINT.LABEL_COLUMN": SUBTYPE}, {"ENDPOINT.NUM_CLASSES": 3}])
    def test_a_regression_takes_no_classes(self, make_cfg, data, runner, tmp_path, overrides):
        with pytest.raises(ValueError, match="has no classes"):
            data.check_endpoint(make_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True, **overrides}))

    def test_a_positive_weight_is_refused_for_a_multiclass_endpoint(self, make_cfg, data, runner, tmp_path):
        with pytest.raises(ValueError, match="POS_WEIGHT weights the positive class"):
            data.check_endpoint(three_class(make_cfg, tmp_path, **{"OBJECTIVE.POS_WEIGHT": 2.0}))

    def test_disabling_the_weight_is_allowed_for_a_multiclass_endpoint(self, make_cfg, data, runner, tmp_path):
        data.check_endpoint(three_class(make_cfg, tmp_path, **{"OBJECTIVE.POS_WEIGHT": -1.0}))

    def test_a_bad_endpoint_fails_the_run_before_anything_is_written(self, make_cfg, data, runner, tmp_path):
        cfg = make_cfg(tmp_path, **{"ENDPOINT.NUM_CLASSES": 3})
        with pytest.raises(ValueError, match="defines two classes"):
            runner.run(cfg)
        assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(("make_cfg", "data", "runner"), EXAMPLES)
class TestDeriveEndpoint:
    def test_a_threshold_labels_what_lies_above_it(self, make_cfg, data, runner, tmp_path):
        cfg = make_cfg(tmp_path)
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], cfg.ENDPOINT.VALUE_COLUMN: [12.0, 20.0, 31.0]})
        derived = data.derive_endpoint(cfg, table)
        # Strictly above: a pressure at the threshold is not elevated.
        assert derived["label"].tolist() == [0, 0, 1]
        assert derived["value"].tolist() == [12.0, 20.0, 31.0]

    def test_a_recorded_class_is_read_as_it_is(self, make_cfg, data, runner, tmp_path):
        cfg = three_class(make_cfg, tmp_path)
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], SUBTYPE: [2, 0, 1]})
        derived = data.derive_endpoint(cfg, table)
        assert derived["label"].tolist() == [2, 0, 1]
        # No measurement column is needed when the class is recorded directly.
        assert "value" not in derived

    def test_the_measurement_is_kept_beside_a_recorded_class(self, make_cfg, data, runner, tmp_path):
        cfg = three_class(make_cfg, tmp_path)
        table = pd.DataFrame(
            {"subject_id": ["a", "b"], SUBTYPE: [1.0, 2.0], cfg.ENDPOINT.VALUE_COLUMN: [18.0, float("nan")]}
        )
        derived = data.derive_endpoint(cfg, table)
        assert derived["label"].tolist() == [1, 2]
        assert derived["value"].iloc[0] == 18.0

    def test_a_subject_with_no_class_is_dropped_not_assumed(self, make_cfg, data, runner, tmp_path):
        cfg = three_class(make_cfg, tmp_path)
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], SUBTYPE: [0, None, 2]})
        derived = data.derive_endpoint(cfg, table)
        assert derived["subject_id"].tolist() == ["a", "c"]

    @pytest.mark.parametrize("bad", [3, -1, 1.5])
    def test_a_value_that_is_not_a_class_index_is_named(self, make_cfg, data, runner, tmp_path, bad):
        cfg = three_class(make_cfg, tmp_path)
        table = pd.DataFrame({"subject_id": ["a", "b"], SUBTYPE: [0, bad]})
        with pytest.raises(ValueError, match=r"not class indices 0\.\.2"):
            data.derive_endpoint(cfg, table)

    def test_a_table_with_no_endpoint_at_all_is_refused(self, make_cfg, data, runner, tmp_path):
        cfg = three_class(make_cfg, tmp_path)
        table = pd.DataFrame({"subject_id": ["a", "b"], SUBTYPE: [None, None]})
        with pytest.raises(ValueError, match="no subject has a recorded subtype"):
            data.derive_endpoint(cfg, table)


@pytest.mark.parametrize(("make_cfg", "data", "runner"), EXAMPLES)
class TestMulticlassCohort:
    def test_the_synthetic_cohort_records_equally_populated_classes(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(three_class(make_cfg, tmp_path))
        counts = np.bincount(cohort.table["label"], minlength=3)
        assert cohort.num_classes == 3
        assert counts.sum() == len(cohort.labelled_ids)
        assert counts.max() - counts.min() <= 1

    def test_the_classes_follow_the_measurement(self, make_cfg, data, runner, tmp_path):
        # Binned from the same measurement the threshold reads, so the endpoint is
        # learnable rather than noise.
        table = data.load_cohort(three_class(make_cfg, tmp_path)).table
        means = table.groupby("label")["value"].mean()
        assert list(means.index) == [0, 1, 2]
        assert means.is_monotonic_increasing

    def test_the_record_reports_class_counts_not_a_positive_rate(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(three_class(make_cfg, tmp_path))
        record = data.summarise(cohort, "subtype")
        assert record["class_counts"] == np.bincount(cohort.table["label"], minlength=3).tolist()
        assert "n_positive" not in record and "positive_rate" not in record
        json.dumps(record)

    def test_the_log_line_names_the_classes(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(three_class(make_cfg, tmp_path))
        counts = "/".join(str(count) for count in np.bincount(cohort.table["label"], minlength=3))
        assert f"classes {counts}" in cohort.describe()

    def test_a_binary_record_keeps_its_positive_rate(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(make_cfg(tmp_path))
        record = data.summarise(cohort, "binary")
        assert record["n_positive"] == int(cohort.table["label"].sum())
        assert "class_counts" not in record
        assert f"{record['n_positive']} positive" in cohort.describe()


@pytest.mark.parametrize(("make_cfg", "data", "runner"), EXAMPLES)
class TestBuildTask:
    def test_a_multiclass_endpoint_weights_every_class_from_the_training_fold(self, make_cfg, data, runner, tmp_path):
        training = [0, 0, 0, 0, 1, 1, 2]
        task = runner.build_task(three_class(make_cfg, tmp_path), training)
        assert isinstance(task, ClassificationTask) and task.num_classes == 3
        expected = class_weight_from_labels(torch.tensor(training, dtype=torch.float32), 3)
        assert torch.allclose(task.class_weight, expected)
        # The rarest training class weighs most.
        assert int(task.class_weight.argmax()) == 2

    def test_a_negative_weight_disables_class_weighting(self, make_cfg, data, runner, tmp_path):
        cfg = three_class(make_cfg, tmp_path, **{"OBJECTIVE.POS_WEIGHT": -1.0})
        assert runner.build_task(cfg, [0, 1, 2]).class_weight is None

    def test_a_binary_weight_is_the_training_folds_imbalance(self, make_cfg, data, runner, tmp_path):
        task = runner.build_task(make_cfg(tmp_path), [0, 0, 0, 1])
        assert task.num_classes == 2
        assert float(task.pos_weight) == pytest.approx(3.0)

    def test_a_fixed_binary_weight_is_used_as_given(self, make_cfg, data, runner, tmp_path):
        task = runner.build_task(make_cfg(tmp_path, **{"OBJECTIVE.POS_WEIGHT": 5.0}), [0, 1])
        assert float(task.pos_weight) == pytest.approx(5.0)

    def test_a_negative_binary_weight_disables_it(self, make_cfg, data, runner, tmp_path):
        assert runner.build_task(make_cfg(tmp_path, **{"OBJECTIVE.POS_WEIGHT": -1.0}), [0, 1]).pos_weight is None

    def test_a_regression_endpoint_is_a_regression_task(self, make_cfg, data, runner, tmp_path):
        task = runner.build_task(make_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True}), [1.0, 2.0])
        assert isinstance(task, RegressionTask)


@pytest.mark.parametrize(("make_cfg", "data", "runner"), EXAMPLES)
class TestScoring:
    def test_a_multiclass_fold_is_scored_on_every_class(self, make_cfg, data, runner, tmp_path):
        labels = np.array([0, 1, 2, 0, 1, 2], dtype=float)
        logits = np.eye(3)[labels.astype(int)] * 4.0
        predictions = SplitPredictions("test", list("abcdef"), logits, targets={"label": labels})
        metrics = runner.score(three_class(make_cfg, tmp_path), predictions)
        assert metrics["roc_auc_ovr"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
        assert np.array(metrics["confusion"]).shape == (3, 3)

    def test_the_headline_follows_the_endpoint(self, make_cfg, data, runner, tmp_path):
        assert runner.headline_metric(make_cfg(tmp_path)) == "roc_auc"
        assert runner.headline_metric(three_class(make_cfg, tmp_path)) == "roc_auc_ovr"
        assert runner.headline_metric(make_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True})) == "rmse"
