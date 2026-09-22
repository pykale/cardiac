"""Regression metrics, fold aggregation, and resampled intervals."""

from __future__ import annotations

import numpy as np
import pytest

from kalecardiac.evaluate import (
    MetricError,
    SplitPredictions,
    binary_metrics,
    bootstrap_ci,
    pool_folds,
    r2_score,
    regression_metrics,
    roc_auc,
    summarise_folds,
)


class TestR2:
    def test_a_perfect_prediction_scores_one(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(targets, targets) == pytest.approx(1.0)

    def test_predicting_the_mean_scores_zero(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(targets, np.full(4, targets.mean())) == pytest.approx(0.0)

    def test_a_worse_than_mean_prediction_is_negative(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(targets, np.array([4.0, 3.0, 2.0, 1.0])) < 0.0

    def test_a_constant_target_has_no_variation_to_explain(self):
        with pytest.raises(MetricError, match="no variation"):
            r2_score(np.full(5, 3.0), np.arange(5.0))

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(MetricError, match="targets and"):
            r2_score(np.zeros(4), np.zeros(5))

    def test_an_empty_cohort_is_rejected(self):
        with pytest.raises(MetricError, match="at least one subject"):
            r2_score(np.array([]), np.array([]))


class TestRegressionMetrics:
    def test_a_perfect_prediction_has_no_error(self):
        targets = np.array([10.0, 20.0, 30.0])
        summary = regression_metrics(targets, targets)
        assert summary["rmse"] == pytest.approx(0.0)
        assert summary["mae"] == pytest.approx(0.0)
        assert summary["r2"] == pytest.approx(1.0)

    def test_bias_is_signed_so_systematic_error_is_visible(self):
        targets = np.array([10.0, 20.0, 30.0])
        summary = regression_metrics(targets, targets + 5.0)
        assert summary["bias"] == pytest.approx(5.0)
        summary = regression_metrics(targets, targets - 5.0)
        assert summary["bias"] == pytest.approx(-5.0)

    def test_rmse_is_more_sensitive_to_an_outlier_than_mae(self):
        targets = np.zeros(10)
        predictions = np.zeros(10)
        predictions[0] = 10.0
        summary = regression_metrics(targets, predictions)
        assert summary["rmse"] > summary["mae"]

    def test_the_unit_is_a_label_and_changes_nothing(self):
        targets = np.array([10.0, 20.0])
        plain = regression_metrics(targets, targets + 1)
        labelled = regression_metrics(targets, targets + 1, unit="mmHg")
        assert plain["rmse"] == labelled["rmse"]
        assert labelled["unit"] == "mmHg"

    def test_a_constant_target_reports_no_r_squared_but_still_an_rmse(self):
        summary = regression_metrics(np.full(5, 3.0), np.full(5, 4.0))
        assert summary["r2"] is None
        assert summary["rmse"] == pytest.approx(1.0)


class TestSummariseFolds:
    @pytest.fixture
    def folds(self, binary_scores):
        labels, scores = binary_scores
        generator = np.random.default_rng(1)
        return [binary_metrics(labels, scores + generator.normal(size=len(scores)) * 0.3) for _ in range(5)]

    def test_reports_a_mean_and_spread_per_metric(self, folds):
        summary = summarise_folds(folds)
        assert summary["n_folds"] == 5
        assert set(summary["roc_auc"]) == {"mean", "std", "values"}
        assert len(summary["roc_auc"]["values"]) == 5

    def test_the_mean_matches_the_values(self, folds):
        summary = summarise_folds(folds)
        assert summary["roc_auc"]["mean"] == pytest.approx(float(np.mean(summary["roc_auc"]["values"])))

    def test_non_numeric_keys_are_skipped_without_being_listed(self, folds):
        # The confusion matrix is a nested dict and has no mean.
        assert "confusion" not in summarise_folds(folds)

    def test_named_keys_are_honoured(self, folds):
        summary = summarise_folds(folds, keys=["roc_auc"])
        assert set(summary) == {"n_folds", "roc_auc"}

    def test_no_fold_is_rejected(self):
        with pytest.raises(MetricError, match="at least one fold"):
            summarise_folds([])


class TestBootstrapCi:
    def test_the_interval_contains_the_point_estimate(self, binary_scores):
        labels, scores = binary_scores
        interval = bootstrap_ci(labels, scores, roc_auc, n_resamples=300, seed=0)
        assert interval["low"] <= interval["point"] <= interval["high"]

    def test_it_is_reproducible_for_a_given_seed(self, binary_scores):
        labels, scores = binary_scores
        first = bootstrap_ci(labels, scores, roc_auc, n_resamples=200, seed=7)
        second = bootstrap_ci(labels, scores, roc_auc, n_resamples=200, seed=7)
        assert first == second

    def test_a_wider_confidence_gives_a_wider_interval(self, binary_scores):
        labels, scores = binary_scores
        narrow = bootstrap_ci(labels, scores, roc_auc, n_resamples=300, confidence=0.5, seed=0)
        wide = bootstrap_ci(labels, scores, roc_auc, n_resamples=300, confidence=0.99, seed=0)
        assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])

    def test_the_number_of_usable_resamples_is_reported(self, binary_scores):
        labels, scores = binary_scores
        assert bootstrap_ci(labels, scores, roc_auc, n_resamples=100, seed=0)["n_used"] <= 100

    def test_a_cohort_no_resample_can_score_is_rejected_with_the_reason(self):
        # Every resample of a one-class split is also one class, so no bootstrap value
        # exists. Failing with the reason beats returning a silently meaningless zero.
        labels = np.zeros(20)
        scores = np.random.default_rng(0).normal(size=20)
        with pytest.raises(MetricError, match="almost every resample holds one class"):
            bootstrap_ci(labels, scores, roc_auc, n_resamples=20, seed=0)

    def test_a_severely_imbalanced_cohort_reports_how_many_resamples_survived(self):
        # One positive in twenty: roughly a third of resamples miss it entirely, and
        # the count kept is what makes that visible rather than a silently wide interval.
        labels = np.array([0] * 19 + [1])
        scores = np.concatenate([np.random.default_rng(0).normal(size=19), [3.0]])
        interval = bootstrap_ci(labels, scores, roc_auc, n_resamples=200, seed=0)
        assert 0 < interval["n_used"] < 200

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(MetricError, match="labels and"):
            bootstrap_ci(np.zeros(4), np.zeros(5), roc_auc)

    def test_an_invalid_confidence_is_rejected(self, binary_scores):
        labels, scores = binary_scores
        with pytest.raises(ValueError, match="confidence must lie"):
            bootstrap_ci(labels, scores, roc_auc, confidence=1.5)


class TestPoolFolds:
    def test_concatenates_every_fold(self):
        folds = [
            SplitPredictions(
                split="test",
                subject_id=[f"s{index}"],
                scores=np.array([0.5]),
                targets={"label": np.array([index % 2])},
                fold=index,
            )
            for index in range(4)
        ]
        labels, scores = pool_folds(folds)
        assert labels.shape == scores.shape == (4,)

    def test_a_fold_with_no_target_is_rejected(self):
        fold = SplitPredictions(split="test", subject_id=["a"], scores=np.array([0.5]), fold=0)
        with pytest.raises(MetricError, match="carries no target"):
            pool_folds([fold])

    def test_no_fold_is_rejected(self):
        with pytest.raises(MetricError, match="at least one fold"):
            pool_folds([])
