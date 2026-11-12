"""The endpoint a configuration describes: a threshold, a recorded class, or the measurement.

Which column and which threshold are an experiment's; what is checked here is that every
way of defining an endpoint is either built correctly or refused before any data is read,
and that anything derived from labels comes from the labels it is given -- the training
fold's -- alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from kale.evaluate.classification_metrics import multiclass_roc_auc, roc_auc
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight

from kalecardiac.auto import (
    check_endpoint,
    derive_endpoint,
    describe_class_balance,
    format_headline,
    get_ablation_metric,
    get_class_balance,
    get_endpoint_column,
    get_endpoint_target,
    get_headline_metric,
    get_prediction_task,
    score_predictions,
)
from kalecardiac.config import get_cfg_defaults
from kalecardiac.evaluate import SplitPredictions
from kalecardiac.pipeline import ClassificationTask, RegressionTask

#: The column a test cohort records its measurement in, and its class when one is asked for.
PRESSURE = "pressure"
SUBTYPE = "subtype"


def endpoint_cfg(**overrides):
    """The package defaults with a thresholded measurement as the endpoint."""
    cfg = get_cfg_defaults()
    cfg.ENDPOINT.VALUE_COLUMN = PRESSURE
    cfg.ENDPOINT.THRESHOLD = 20.0
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    return cfg


def three_class(**overrides):
    """An endpoint read from a three-class column."""
    return endpoint_cfg(**{"ENDPOINT.LABEL_COLUMN": SUBTYPE, "ENDPOINT.NUM_CLASSES": 3, **overrides})


class TestCheckEndpoint:
    def test_a_threshold_passes(self):
        check_endpoint(endpoint_cfg())

    def test_a_three_class_column_passes(self):
        check_endpoint(three_class())

    def test_the_package_defaults_name_no_endpoint_and_say_which_key_to_set(self):
        # The schema declares the keys but names no column: that is the experiment's.
        with pytest.raises(ValueError, match="set ENDPOINT.VALUE_COLUMN"):
            check_endpoint(get_cfg_defaults())

    def test_a_class_column_alone_is_enough(self):
        check_endpoint(three_class(**{"ENDPOINT.VALUE_COLUMN": ""}))

    def test_a_regression_on_the_measurement_passes(self):
        check_endpoint(endpoint_cfg(**{"ENDPOINT.REGRESSION": True}))

    def test_fewer_than_two_classes_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            check_endpoint(endpoint_cfg(**{"ENDPOINT.NUM_CLASSES": 1}))

    def test_a_threshold_cannot_make_three_classes(self):
        with pytest.raises(ValueError, match="defines two classes, not 3; set ENDPOINT.LABEL_COLUMN"):
            check_endpoint(endpoint_cfg(**{"ENDPOINT.NUM_CLASSES": 3}))

    @pytest.mark.parametrize("overrides", [{"ENDPOINT.LABEL_COLUMN": SUBTYPE}, {"ENDPOINT.NUM_CLASSES": 3}])
    def test_a_regression_takes_no_classes(self, overrides):
        with pytest.raises(ValueError, match="has no classes"):
            check_endpoint(endpoint_cfg(**{"ENDPOINT.REGRESSION": True, **overrides}))

    def test_a_positive_weight_is_refused_for_a_multiclass_endpoint(self):
        with pytest.raises(ValueError, match="POS_WEIGHT weights the positive class"):
            check_endpoint(three_class(**{"OBJECTIVE.POS_WEIGHT": 2.0}))

    def test_disabling_the_weight_is_allowed_for_a_multiclass_endpoint(self):
        check_endpoint(three_class(**{"OBJECTIVE.POS_WEIGHT": -1.0}))


class TestDeriveEndpoint:
    def test_the_class_column_is_read_in_preference_to_the_measurement(self):
        assert get_endpoint_column(endpoint_cfg()) == PRESSURE
        assert get_endpoint_column(three_class()) == SUBTYPE

    def test_a_threshold_labels_what_lies_above_it(self):
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], PRESSURE: [12.0, 20.0, 31.0]})
        derived = derive_endpoint(endpoint_cfg(), table)
        # Strictly above: a pressure at the threshold is not elevated.
        assert derived["label"].tolist() == [0, 0, 1]
        assert derived["value"].tolist() == [12.0, 20.0, 31.0]

    def test_the_table_given_is_not_changed(self):
        table = pd.DataFrame({"subject_id": ["a"], PRESSURE: [30.0]})
        derive_endpoint(endpoint_cfg(), table)
        assert list(table.columns) == ["subject_id", PRESSURE]

    def test_a_recorded_class_is_read_as_it_is(self):
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], SUBTYPE: [2, 0, 1]})
        derived = derive_endpoint(three_class(), table)
        assert derived["label"].tolist() == [2, 0, 1]
        # No measurement column is needed when the class is recorded directly.
        assert "value" not in derived

    def test_the_measurement_is_kept_beside_a_recorded_class(self):
        table = pd.DataFrame({"subject_id": ["a", "b"], SUBTYPE: [1.0, 2.0], PRESSURE: [18.0, float("nan")]})
        derived = derive_endpoint(three_class(), table)
        assert derived["label"].tolist() == [1, 2]
        assert derived["value"].iloc[0] == 18.0

    def test_a_subject_with_no_endpoint_is_dropped_not_assumed(self):
        table = pd.DataFrame({"subject_id": ["a", "b", "c"], SUBTYPE: [0, None, 2]})
        assert derive_endpoint(three_class(), table)["subject_id"].tolist() == ["a", "c"]

    def test_a_missing_measurement_is_dropped_not_called_negative(self):
        table = pd.DataFrame({"subject_id": ["a", "b"], PRESSURE: [None, 31.0]})
        derived = derive_endpoint(endpoint_cfg(), table)
        assert derived["subject_id"].tolist() == ["b"]
        assert derived["label"].tolist() == [1]

    @pytest.mark.parametrize("bad", [3, -1, 1.5])
    def test_a_value_that_is_not_a_class_index_is_named(self, bad):
        table = pd.DataFrame({"subject_id": ["a", "b"], SUBTYPE: [0, bad]})
        with pytest.raises(ValueError, match=r"not class indices 0\.\.2"):
            derive_endpoint(three_class(), table)

    def test_a_table_with_no_endpoint_at_all_is_refused(self):
        table = pd.DataFrame({"subject_id": ["a", "b"], SUBTYPE: [None, None]})
        with pytest.raises(ValueError, match="no subject has a recorded subtype"):
            derive_endpoint(three_class(), table)


class TestEndpointTarget:
    def test_a_classification_is_supervised_by_the_label(self):
        cfg = endpoint_cfg()
        table = derive_endpoint(cfg, pd.DataFrame({"subject_id": ["a", "b"], PRESSURE: [12.0, 31.0]}))
        values = get_endpoint_target(cfg, table, "subject_id").values_for(["b", "a"])
        assert list(values) == ["label"]
        assert values["label"].tolist() == [1.0, 0.0]

    def test_a_regression_is_supervised_by_the_measurement(self):
        cfg = endpoint_cfg(**{"ENDPOINT.REGRESSION": True})
        table = derive_endpoint(cfg, pd.DataFrame({"subject_id": ["a", "b"], PRESSURE: [12.0, 31.0]}))
        values = get_endpoint_target(cfg, table, "subject_id").values_for(["a", "b"])
        assert list(values) == ["value"]
        assert values["value"].tolist() == [12.0, 31.0]


class TestClassBalance:
    def test_a_binary_balance_is_a_positive_count_and_rate(self):
        table = pd.DataFrame({"label": [0, 1, 1, 0]})
        assert get_class_balance(table, 2) == {"n_positive": 2, "positive_rate": 0.5}
        assert describe_class_balance(table, 2) == "2 positive"

    def test_a_multiclass_balance_counts_every_class_including_an_empty_one(self):
        table = pd.DataFrame({"label": [0, 0, 2]})
        balance = get_class_balance(table, 3)
        assert balance == {"class_counts": [2, 0, 1]}
        assert describe_class_balance(table, 3) == "classes 2/0/1"
        json.dumps(balance)


class TestGetPredictionTask:
    def test_a_multiclass_endpoint_weights_every_class_from_the_labels_given(self):
        training = [0, 0, 0, 0, 1, 1, 2]
        task = get_prediction_task(three_class(), training)
        assert isinstance(task, ClassificationTask) and task.num_classes == 3
        expected = compute_class_weight("balanced", classes=np.arange(3), y=np.array(training))
        assert torch.allclose(task.class_weight, torch.tensor(expected, dtype=torch.float32))
        # The rarest training class weighs most.
        assert int(task.class_weight.argmax()) == 2

    def test_a_class_absent_from_the_training_fold_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"class\(es\) \[2\] have no subject"):
            get_prediction_task(three_class(), [0, 1, 1])

    def test_a_negative_weight_disables_class_weighting(self):
        assert get_prediction_task(three_class(**{"OBJECTIVE.POS_WEIGHT": -1.0}), [0, 1, 2]).class_weight is None

    def test_a_binary_weight_is_the_training_folds_imbalance(self):
        task = get_prediction_task(endpoint_cfg(), [0, 0, 0, 1])
        assert task.num_classes == 2
        assert float(task.pos_weight) == pytest.approx(3.0)

    def test_a_tensor_of_labels_is_read_as_well_as_a_list(self):
        task = get_prediction_task(endpoint_cfg(), torch.tensor([0.0, 0.0, 0.0, 1.0]))
        assert float(task.pos_weight) == pytest.approx(3.0)

    def test_a_fixed_binary_weight_is_used_as_given(self):
        assert float(get_prediction_task(endpoint_cfg(**{"OBJECTIVE.POS_WEIGHT": 5.0}), [0, 1]).pos_weight) == 5.0

    def test_a_negative_binary_weight_disables_it(self):
        assert get_prediction_task(endpoint_cfg(**{"OBJECTIVE.POS_WEIGHT": -1.0}), [0, 1]).pos_weight is None

    def test_a_regression_endpoint_is_a_regression_task(self):
        assert isinstance(
            get_prediction_task(endpoint_cfg(**{"ENDPOINT.REGRESSION": True}), [1.0, 2.0]), RegressionTask
        )

    def test_the_head_is_the_one_model_describes(self):
        task = get_prediction_task(endpoint_cfg(**{"MODEL.HEAD_HIDDEN": [], "MODEL.DROPOUT": 0.0}), [0, 1])
        head = task.build_head(6)
        assert sum(parameter.numel() for parameter in head.parameters()) == 6 + 1


class TestScoring:
    def test_a_binary_split_is_scored_on_its_ranking(self):
        labels = np.array([0, 1, 0, 1], dtype=float)
        predictions = SplitPredictions("test", list("abcd"), np.array([-2.0, 2.0, -1.0, 1.0]), {"label": labels})
        assert score_predictions(endpoint_cfg(), predictions)["roc_auc"] == pytest.approx(1.0)

    def test_a_multiclass_split_is_scored_on_every_class(self):
        labels = np.array([0, 1, 2, 0, 1, 2], dtype=float)
        logits = np.eye(3)[labels.astype(int)] * 4.0
        metrics = score_predictions(three_class(), SplitPredictions("test", list("abcdef"), logits, {"label": labels}))
        assert metrics["roc_auc_ovr"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
        assert np.array(metrics["confusion"]).shape == (3, 3)

    def test_a_regression_is_scored_in_the_endpoints_unit(self):
        values = np.array([10.0, 20.0, 30.0])
        predictions = SplitPredictions("test", list("abc"), values + 1.0, {"value": values})
        metrics = score_predictions(endpoint_cfg(**{"ENDPOINT.REGRESSION": True, "ENDPOINT.UNIT": "kPa"}), predictions)
        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["unit"] == "kPa"

    def test_the_headline_follows_the_endpoint(self):
        assert get_headline_metric(endpoint_cfg()) == "roc_auc"
        assert get_headline_metric(three_class()) == "roc_auc_ovr"
        assert get_headline_metric(endpoint_cfg(**{"ENDPOINT.REGRESSION": True})) == "rmse"

    def test_the_headline_line_reads_one_fold_or_the_spread_over_folds(self):
        cfg = endpoint_cfg()
        assert format_headline(cfg, {"roc_auc": 0.8125}) == "roc_auc=0.812"
        assert format_headline(cfg, {"roc_auc": {"mean": 0.75, "std": 0.05}}, folded=True) == "roc_auc=0.750 +/- 0.050"
        # A metric a fold could not compute reads as nan rather than failing the log line.
        assert format_headline(cfg, {}) == "roc_auc=nan"


class TestAblationMetric:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, ("roc_auc", roc_auc, "label")),
            (
                {"ENDPOINT.LABEL_COLUMN": SUBTYPE, "ENDPOINT.NUM_CLASSES": 3},
                ("roc_auc_ovr", multiclass_roc_auc, "label"),
            ),
            ({"ENDPOINT.REGRESSION": True}, ("r2", r2_score, "value")),
        ],
        ids=["binary", "three-class", "regression"],
    )
    def test_each_endpoint_is_rescored_with_its_threshold_free_metric(self, overrides, expected):
        assert get_ablation_metric(endpoint_cfg(**overrides)) == expected
