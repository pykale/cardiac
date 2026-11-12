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
