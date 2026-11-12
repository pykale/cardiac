"""Contracts specific to Cardiac's PyKale evaluation adapters."""

from __future__ import annotations

import numpy as np
import pytest
from kale.evaluate.classification_metrics import MetricError
from kale.evaluate.cross_validation import summarize_folds

from kalecardiac.evaluate import SplitPredictions, bootstrap_ci, flatten_fold_summary, pool_folds


def test_flatten_fold_summary_makes_a_results_row():
    summary = summarize_folds([{"roc_auc": 0.5}, {"roc_auc": 1.0}])
    assert flatten_fold_summary(summary) == {"n_folds": 2, "roc_auc_mean": 0.75, "roc_auc_std": 0.25}


class TestBootstrapCi:
    def test_keeps_labels_first_signature_and_default_seed(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        def metric(sample_labels, sample_scores):
            return float(np.mean(sample_scores[sample_labels == 1]))

        default = bootstrap_ci(labels, scores, metric, n_resamples=20)
        explicit = bootstrap_ci(labels, scores, metric, n_resamples=20, seed=2026)
        assert default == explicit
        assert default["point"] == pytest.approx(0.85)

    def test_skips_value_error_from_a_resample(self):
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])

        def metric(sample_labels, sample_scores):
            if np.unique(sample_labels).size == 1:
                raise ValueError("one class")
            return float(sample_scores.mean())

        interval = bootstrap_ci(labels, scores, metric, n_resamples=20, seed=0)
        assert 0 < interval["n_used"] < 20

    def test_flattens_cardiac_inputs_before_delegating(self):
        labels = np.array([[0], [1]])
        scores = np.array([[0.1], [0.9]])
        interval = bootstrap_ci(labels, scores, lambda y, s: float(np.mean(s[y == 1])), n_resamples=20)
        assert interval["point"] == pytest.approx(0.9)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(MetricError, match="2 labels and 1 scores"):
            bootstrap_ci([0, 1], [0.5], lambda labels, scores: 1.0)


class TestPoolFolds:
    def test_concatenates_cardiac_prediction_records_in_order(self):
        folds = [
            SplitPredictions("test", ["a", "b"], np.array([0.1, 0.2]), {"label": np.array([0, 1])}, fold=0),
            SplitPredictions("test", ["c"], np.array([0.3]), {"label": np.array([1])}, fold=1),
        ]
        labels, scores = pool_folds(folds)
        np.testing.assert_array_equal(labels, [0, 1, 1])
        np.testing.assert_array_equal(scores, [0.1, 0.2, 0.3])

    def test_rejects_a_fold_without_targets(self):
        fold = SplitPredictions("test", ["a"], np.array([0.5]), fold=0)
        with pytest.raises(MetricError, match="carries no target"):
            pool_folds([fold])

    def test_rejects_mismatched_targets_and_scores(self):
        fold = SplitPredictions("test", ["a", "b"], np.array([0.5]), {"label": np.array([0, 1])}, fold=0)
        with pytest.raises(MetricError, match="2 labels but 1 scores"):
            pool_folds([fold])


class TestKaleCardiacImports:
    """The metric names KaleCardiac kept return PyKale's values."""

    LABELS = np.array([0, 1, 0, 1, 1, 0])
    SCORES = np.array([0.1, 0.8, 0.5, 0.4, 0.9, 0.3])

    @pytest.mark.parametrize(
        "name,module",
        [
            ("roc_auc", "classification_metrics"),
            ("binary_metrics", "classification_metrics"),
            ("sensitivity_specificity", "classification_metrics"),
            ("regression_metrics", "regression_metrics"),
        ],
    )
    def test_binary_and_regression_metrics_match_pykale(self, name, module):
        import importlib

        import kalecardiac.evaluate as cardiac

        core = importlib.import_module(f"kale.evaluate.{module}")
        assert getattr(cardiac, name)(self.LABELS, self.SCORES) == getattr(core, name)(self.LABELS, self.SCORES)

    def test_multiclass_metrics_match_pykale(self):
        import kale.evaluate.classification_metrics as core

        import kalecardiac.evaluate as cardiac

        labels = np.array([0, 1, 2, 0, 1, 2])
        logits = np.array(
            [[2.0, 0.1, 0.3], [0.2, 1.5, 0.1], [0.1, 0.4, 2.2], [1.1, 0.9, 0.2], [0.3, 0.2, 0.1], [0.5, 0.1, 1.9]]
        )
        assert cardiac.multiclass_roc_auc(labels, logits) == core.multiclass_roc_auc(labels, logits)
        assert str(cardiac.multiclass_metrics(labels, logits, ["a", "b", "c"])) == str(
            core.multiclass_metrics(labels, logits, ["a", "b", "c"])
        )

    def test_mean_roc_curve_and_fold_summary_match_pykale(self):
        import kale.evaluate.classification_metrics as core

        import kalecardiac.evaluate as cardiac

        runs = ([self.LABELS, self.LABELS[::-1]], [self.SCORES, self.SCORES])
        assert str(cardiac.mean_roc_curve(*runs, grid_size=20)) == str(core.mean_roc_curve(*runs, grid_size=20))
        folds = [{"auc": 0.7}, {"auc": 0.9}]
        assert cardiac.summarize_folds(folds) == summarize_folds(folds)

    def test_metric_error_is_pykale_and_still_a_value_error(self):
        from kalecardiac.evaluate import MetricError as cardiac_error

        assert cardiac_error is MetricError and issubclass(cardiac_error, ValueError)
