"""Classification metrics, computed by PyKale.

KaleCardiac keeps these names and signatures; the calculations are
:mod:`kale.evaluate.classification_metrics`, which documents each metric. Scores may be
logits or probabilities, as before.
"""

from __future__ import annotations

from collections.abc import Sequence

from kale.evaluate.classification_metrics import MetricError
from kale.evaluate.classification_metrics import binary_metrics as _kale_binary_metrics
from kale.evaluate.classification_metrics import mean_roc_curve as _kale_mean_roc_curve
from kale.evaluate.classification_metrics import multiclass_metrics as _kale_multiclass_metrics
from kale.evaluate.classification_metrics import multiclass_roc_auc as _kale_multiclass_roc_auc
from kale.evaluate.classification_metrics import roc_auc as _kale_roc_auc
from kale.evaluate.classification_metrics import sensitivity_specificity as _kale_sensitivity_specificity
from numpy.typing import ArrayLike

__all__ = [
    "MetricError",
    "binary_metrics",
    "mean_roc_curve",
    "multiclass_metrics",
    "multiclass_roc_auc",
    "roc_auc",
    "sensitivity_specificity",
]


def roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve; see :func:`kale.evaluate.classification_metrics.roc_auc`."""
    return _kale_roc_auc(labels, scores)


def sensitivity_specificity(labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5) -> tuple[float, float]:
    """Sensitivity and specificity at ``threshold``; see PyKale's ``sensitivity_specificity``."""
    return _kale_sensitivity_specificity(labels, scores, threshold)


def binary_metrics(labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5) -> dict:
    """Binary metrics at ``threshold``; see :func:`kale.evaluate.classification_metrics.binary_metrics`."""
    return _kale_binary_metrics(labels, scores, threshold)


def multiclass_roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Macro one-versus-rest AUROC; see PyKale's ``multiclass_roc_auc``."""
    return _kale_multiclass_roc_auc(labels, scores)


def multiclass_metrics(labels: ArrayLike, scores: ArrayLike, class_names: Sequence[str] | None = None) -> dict:
    """Multiclass metrics; see :func:`kale.evaluate.classification_metrics.multiclass_metrics`."""
    return _kale_multiclass_metrics(labels, scores, class_names)


def mean_roc_curve(labels_per_run: list, scores_per_run: list, grid_size: int = 100) -> dict:
    """ROC curve averaged over runs; see :func:`kale.evaluate.classification_metrics.mean_roc_curve`."""
    return _kale_mean_roc_curve(labels_per_run, scores_per_run, grid_size)
