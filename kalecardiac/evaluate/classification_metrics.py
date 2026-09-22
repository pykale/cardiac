"""Metrics for categorical endpoints.

What a cardiac classification is reported with, and no endpoint is named here:
"pulmonary hypertension detection" and "elevated wedge pressure" are both a binary
label, and a metric that knew which would be an experiment.

The set is chosen for a clinical reading rather than a leaderboard. ROC-AUC and average
precision rank; sensitivity and specificity say what the model does at a chosen
operating point, which is what a clinician asks and what a single accuracy figure hides
on an imbalanced cohort; and the confusion matrix underlies both.

Scores may be logits or probabilities throughout. Ranking metrics are invariant to the
sigmoid, so only the threshold metrics apply it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the values given."""


def _as_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    present = np.unique(labels)
    if present.size < 2:
        raise MetricError(
            f"a binary metric needs both classes, but every subject has label {present.tolist()}; on a small "
            f"cardiac cohort this usually means the split is too small or an exclusion removed one class"
        )
    return labels.astype(int)


def _as_probabilities(scores: np.ndarray) -> np.ndarray:
    """Read scores as probabilities, applying a sigmoid only if they are not already."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size and scores.min() >= 0.0 and scores.max() <= 1.0:
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` predicted scores, higher meaning more likely positive.

    Returns:
        The area: 0.5 for a random ranking, 1.0 for a perfect one.

    Raises:
        MetricError: If only one class is present.
    """
    return float(roc_auc_score(_as_binary_labels(labels), np.asarray(scores).reshape(-1)))


def sensitivity_specificity(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """Sensitivity and specificity at an operating point.

    The pair a clinical reader asks for, and the reason accuracy alone is not enough:
    on a cohort with 20% positives, calling everybody negative scores 80% accuracy and
    0% sensitivity.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` logits or probabilities.
        threshold: Probability above which a subject is called positive.

    Returns:
        ``(sensitivity, specificity)``, each in ``[0, 1]``.

    Raises:
        MetricError: If only one class is present.
    """
    labels = _as_binary_labels(labels)
    predicted = (_as_probabilities(scores) >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return float(sensitivity), float(specificity)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Summarise a binary prediction.

    ``roc_auc`` and ``average_precision`` rank the scores and ignore ``threshold``; the
    remaining metrics apply a sigmoid where needed and cut at it.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` logits or probabilities.
        threshold: Probability above which a subject is called positive.

    Returns:
        Counts and metrics, ready for JSON export. ``confusion`` is a nested
        ``{"tn", "fp", "fn", "tp"}`` so the numbers behind the rates stay visible.

    Raises:
        MetricError: If only one class is present.
    """
    labels = _as_binary_labels(labels)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    probabilities = _as_probabilities(scores)
    predicted = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    sensitivity, specificity = sensitivity_specificity(labels, scores, threshold)

    return {
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "confusion": {
            "tn": int(true_negative),
            "fp": int(false_positive),
            "fn": int(false_negative),
            "tp": int(true_positive),
        },
    }


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str] | None = None) -> dict:
    """Summarise a multiclass prediction.

    Args:
        labels: ``(N,)`` class indices.
        probabilities: ``(N, num_classes)`` predicted probabilities.
        class_names: Names for the confusion matrix's axes. Defaults to the indices.

    Returns:
        Accuracy, balanced accuracy, macro F1, one-versus-rest macro ROC-AUC where it
        is defined, and the confusion matrix as a nested list.

    Raises:
        MetricError: If the shapes disagree, or fewer than two classes are present.
    """
    labels = np.asarray(labels).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise MetricError(f"expected ({labels.size}, num_classes) probabilities, got {probabilities.shape}")
    if np.unique(labels).size < 2:
        raise MetricError("a multiclass metric needs at least two classes present")

    num_classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    summary = {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "confusion": confusion_matrix(labels, predicted, labels=list(range(num_classes))).tolist(),
        "classes": list(class_names) if class_names else [str(index) for index in range(num_classes)],
    }
    # One-versus-rest AUC is undefined when a class is absent from this split, which on
    # a small cohort is ordinary; report what can be computed rather than failing.
    if np.unique(labels).size == num_classes:
        summary["roc_auc_ovr"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    return summary


def mean_roc_curve(labels_per_run: list, scores_per_run: list, grid_size: int = 100) -> dict:
    """Average several runs' ROC curves onto one false-positive-rate grid.

    What a cross-validated cardiac result is plotted as. Runs differ in how many
    distinct thresholds they produce, so their curves cannot be averaged point by point;
    each is interpolated onto a shared grid first, which is what makes a mean curve and
    a variability band well defined.

    This is not the same as pooling every run's predictions and computing one curve:
    averaging interpolated curves weights each fold equally regardless of its size, and
    is the convention when repeats of the same experiment are summarised together.

    Args:
        labels_per_run: One ``(N,)`` label array per run.
        scores_per_run: One ``(N,)`` score array per run, aligned with the labels.
        grid_size: Number of false-positive-rate points to interpolate onto.

    Returns:
        ``"fpr"``, ``"tpr_mean"``, ``"tpr_std"``, ``"tpr_runs"``, ``"auc_mean"``,
        ``"auc_std"`` and ``"auc_runs"``.

    Raises:
        MetricError: If no run is given, the two lists disagree in length, or a run has
            only one class.
    """
    if not labels_per_run:
        raise MetricError("mean_roc_curve needs at least one run")
    if len(labels_per_run) != len(scores_per_run):
        raise MetricError(
            f"got {len(labels_per_run)} label arrays but {len(scores_per_run)} score arrays; each run "
            f"contributes exactly one of each"
        )

    grid = np.linspace(0.0, 1.0, grid_size)
    curves, areas = [], []
    for labels, scores in zip(labels_per_run, scores_per_run, strict=True):
        labels = _as_binary_labels(labels)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        interpolated = np.interp(grid, false_positive, true_positive)
        # Interpolating onto a fixed grid does not guarantee the endpoints, which are
        # (0, 0) and (1, 1) for every ROC curve by construction.
        interpolated[0] = 0.0
        curves.append(interpolated)
        areas.append(float(roc_auc_score(labels, scores)))

    stacked = np.vstack(curves)
    mean = stacked.mean(axis=0)
    mean[-1] = 1.0

    return {
        "fpr": grid,
        "tpr_mean": mean,
        "tpr_std": stacked.std(axis=0),
        "tpr_runs": stacked,
        "auc_mean": float(np.mean(areas)),
        "auc_std": float(np.std(areas)),
        "auc_runs": areas,
    }
