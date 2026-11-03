"""Endpoint definition in both examples: what their cohorts record, and what a run refuses.

How a threshold or a recorded class becomes a label, a task and a score is the
library's, and is tested in ``tests/auto``. What is checked here is each example's side
of it: that its synthetic cohort records a learnable class when one is asked for, that
its record and log line report the classes, and that a bad endpoint stops a run before
anything is written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from examples.cardiovae_multimodal import data as cardiovae_data
from examples.cardiovae_multimodal import runner as cardiovae_runner
from examples.lsemvae_ecg import data as lsemvae_data
from examples.lsemvae_ecg import runner as lsemvae_runner
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
class TestEndpointChecks:
    def test_the_published_endpoint_passes(self, make_cfg, data, runner, tmp_path):
        data.load_cohort(make_cfg(tmp_path))

    def test_a_bad_endpoint_fails_the_run_before_anything_is_written(self, make_cfg, data, runner, tmp_path):
        cfg = make_cfg(tmp_path, **{"ENDPOINT.NUM_CLASSES": 3})
        with pytest.raises(ValueError, match="defines two classes"):
            runner.run(cfg)
        assert not any(tmp_path.iterdir())

    def test_a_bad_endpoint_fails_before_a_cohort_is_read(self, make_cfg, data, runner, tmp_path):
        with pytest.raises(ValueError, match="at least 2"):
            data.load_cohort(make_cfg(tmp_path, **{"ENDPOINT.NUM_CLASSES": 1}))


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
        record = data.summarize_cohort(cohort, "subtype")
        assert record["class_counts"] == np.bincount(cohort.table["label"], minlength=3).tolist()
        assert "n_positive" not in record and "positive_rate" not in record
        json.dumps(record)

    def test_the_log_line_names_the_classes(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(three_class(make_cfg, tmp_path))
        counts = "/".join(str(count) for count in np.bincount(cohort.table["label"], minlength=3))
        assert f"classes {counts}" in cohort.describe()

    def test_a_binary_record_keeps_its_positive_rate(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(make_cfg(tmp_path))
        record = data.summarize_cohort(cohort, "binary")
        assert record["n_positive"] == int(cohort.table["label"].sum())
        assert "class_counts" not in record
        assert f"{record['n_positive']} positive" in cohort.describe()

    def test_a_regression_cohort_is_supervised_by_the_measurement(self, make_cfg, data, runner, tmp_path):
        cohort = data.load_cohort(make_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True}))
        values = cohort.target.values_for(cohort.labelled_ids[:3])
        assert list(values) == ["value"]
        assert values["value"].tolist() == pytest.approx(cohort.table["value"].iloc[:3].tolist())
