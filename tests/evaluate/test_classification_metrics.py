"""Classification metrics: known values, the clinical pair, and small-cohort failures."""

from __future__ import annotations

import numpy as np
import pytest

from kalecardiac.evaluate import (
    MetricError,
    binary_metrics,
    mean_roc_curve,
    multiclass_metrics,
    roc_auc,
    sensitivity_specificity,
)


class TestRocAuc:
    def test_a_perfect_ranking_scores_one(self):
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_a_reversed_ranking_scores_zero(self):
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)

    def test_it_is_invariant_to_a_sigmoid(self):
        labels = np.array([0, 1, 0, 1, 1])
        logits = np.array([-2.0, 1.0, -0.5, 0.3, 2.0])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        assert roc_auc(labels, logits) == pytest.approx(roc_auc(labels, probabilities))

    def test_a_one_class_split_is_rejected_with_the_usual_cause(self):
        with pytest.raises(MetricError, match="too small"):
            roc_auc(np.zeros(10), np.random.default_rng(0).normal(size=10))


class TestSensitivitySpecificity:
    def test_known_confusion_gives_known_rates(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        probabilities = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1])
        sensitivity, specificity = sensitivity_specificity(labels, probabilities)
        assert sensitivity == pytest.approx(0.75)
        assert specificity == pytest.approx(0.75)

    def test_calling_everybody_negative_has_no_sensitivity(self):
        # The reason accuracy alone is not reported: on an imbalanced cardiac cohort
        # this scores well on accuracy and is clinically useless.
        labels = np.array([0] * 8 + [1] * 2)
        summary = binary_metrics(labels, np.full(10, -10.0))
        assert summary["accuracy"] == pytest.approx(0.8)
        assert summary["sensitivity"] == pytest.approx(0.0)

    def test_the_threshold_moves_the_operating_point(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.2, 0.6, 0.4, 0.9])
        lenient = sensitivity_specificity(labels, probabilities, threshold=0.3)
        strict = sensitivity_specificity(labels, probabilities, threshold=0.8)
        assert lenient[0] > strict[0]
        assert lenient[1] < strict[1]


class TestBinaryMetrics:
    def test_reports_counts_and_metrics_together(self, binary_scores):
        labels, scores = binary_scores
        summary = binary_metrics(labels, scores)
        expected = {
            "n",
            "n_positive",
            "positive_rate",
            "threshold",
            "roc_auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
            "confusion",
        }
        assert expected == set(summary)
        assert summary["n"] == len(labels)

    def test_the_confusion_counts_sum_to_the_cohort(self, binary_scores):
        labels, scores = binary_scores
        confusion = binary_metrics(labels, scores)["confusion"]
        assert sum(confusion.values()) == len(labels)

    def test_it_accepts_logits_and_probabilities_alike(self, binary_scores):
        labels, scores = binary_scores
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        assert binary_metrics(labels, scores)["roc_auc"] == pytest.approx(
            binary_metrics(labels, probabilities)["roc_auc"]
        )

    def test_a_perfect_classifier_scores_one_everywhere(self):
        labels = np.array([0, 0, 1, 1])
        summary = binary_metrics(labels, np.array([0.01, 0.02, 0.98, 0.99]))
        for key in ("roc_auc", "accuracy", "sensitivity", "specificity", "f1"):
            assert summary[key] == pytest.approx(1.0)

    def test_the_output_is_json_serialisable(self, binary_scores):
        import json

        labels, scores = binary_scores
        json.dumps(binary_metrics(labels, scores))


class TestMulticlassMetrics:
    @pytest.fixture
    def three_class(self):
        generator = np.random.default_rng(0)
        labels = generator.integers(0, 3, size=60)
        probabilities = generator.dirichlet(np.ones(3), size=60)
        probabilities[np.arange(60), labels] += 1.0
        return labels, probabilities / probabilities.sum(axis=1, keepdims=True)

    def test_reports_accuracy_and_a_confusion_matrix(self, three_class):
        labels, probabilities = three_class
        summary = multiclass_metrics(labels, probabilities)
        assert summary["n"] == 60
        assert np.array(summary["confusion"]).shape == (3, 3)
        assert sum(sum(row) for row in summary["confusion"]) == 60

    def test_one_versus_rest_auc_is_reported_when_every_class_is_present(self, three_class):
        labels, probabilities = three_class
        assert "roc_auc_ovr" in multiclass_metrics(labels, probabilities)

    def test_a_missing_class_omits_the_auc_rather_than_failing(self):
        # Ordinary on a small cohort; report what can be computed.
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
        assert "roc_auc_ovr" not in multiclass_metrics(labels, probabilities)

    def test_class_names_are_carried_through(self, three_class):
        labels, probabilities = three_class
        summary = multiclass_metrics(labels, probabilities, class_names=["none", "pre", "post"])
        assert summary["classes"] == ["none", "pre", "post"]

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(MetricError, match="expected"):
            multiclass_metrics(np.array([0, 1]), np.zeros((3, 3)))


class TestMeanRocCurve:
    @pytest.fixture
    def folds(self):
        generator = np.random.default_rng(0)
        return [(generator.integers(0, 2, 40), generator.normal(size=40)) for _ in range(5)]

    def test_averages_curves_onto_a_shared_grid(self, folds):
        summary = mean_roc_curve([labels for labels, _ in folds], [scores for _, scores in folds])
        assert summary["fpr"].shape == summary["tpr_mean"].shape == (100,)
        assert summary["tpr_runs"].shape == (5, 100)
        assert len(summary["auc_runs"]) == 5

    def test_the_curve_starts_at_zero_and_ends_at_one(self, folds):
        summary = mean_roc_curve([labels for labels, _ in folds], [scores for _, scores in folds])
        assert summary["tpr_mean"][0] == pytest.approx(0.0)
        assert summary["tpr_mean"][-1] == pytest.approx(1.0)

    def test_the_mean_auc_matches_the_per_run_mean(self, folds):
        summary = mean_roc_curve([labels for labels, _ in folds], [scores for _, scores in folds])
        assert summary["auc_mean"] == pytest.approx(float(np.mean(summary["auc_runs"])))

    def test_no_run_is_rejected(self):
        with pytest.raises(MetricError, match="at least one run"):
            mean_roc_curve([], [])

    def test_mismatched_lists_are_rejected(self, folds):
        with pytest.raises(MetricError, match="exactly one of each"):
            mean_roc_curve([labels for labels, _ in folds], [folds[0][1]])
