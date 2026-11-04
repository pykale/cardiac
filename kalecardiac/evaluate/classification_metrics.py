"""Metrics for categorical endpoints.

What a cardiac classification is reported with, and no endpoint is named here:
"pulmonary hypertension detection" and "elevated wedge pressure" are both a binary
label, and a metric that knew which would be an experiment.

The set is chosen for a clinical reading rather than a leaderboard. ROC-AUC and average
precision rank; sensitivity and specificity say what the model does at a chosen
operating point, which is what a clinician asks and what a single accuracy figure hides
on an imbalanced cohort; and the confusion matrix underlies both.

Scores may be logits or probabilities throughout. Binary ranking metrics are invariant
to the sigmoid, so only the threshold metrics apply it. A multiclass one-versus-rest
ranking is *not* invariant to the softmax -- a class's probability depends on every
other class's logit -- so the multiclass metrics apply it to anything that is not
already a distribution over the classes.

The computations are PyKale's: each function here delegates to its namesake in
:mod:`kale.evaluate.classification_metrics`, which was adapted from this module, and keeps
this package's signature and clinical documentation. ``MetricError`` is PyKale's class, so
an ``except MetricError`` here also catches the core's. Delegating changes four things,
all for inputs the metrics were never defined on: a binary label other than exactly 0 or
1, or a multiclass label that is not exactly a class index, raises rather than being cast;
and a two-column multiclass score is scored as the binary area instead of failing.
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
    """Area under the ROC curve.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` predicted scores, higher meaning more likely positive.

    Returns:
        The area: 0.5 for a random ranking, 1.0 for a perfect one.

    Raises:
        MetricError: If a label is not exactly 0 or 1, or only one class is present.
    """
    return _kale_roc_auc(labels, scores)


def sensitivity_specificity(labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5) -> tuple[float, float]:
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
        MetricError: If a label is not exactly 0 or 1, or only one class is present.
    """
    return _kale_sensitivity_specificity(labels, scores, threshold)


def binary_metrics(labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5) -> dict:
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
        MetricError: If a label is not exactly 0 or 1, or only one class is present.
    """
    return _kale_binary_metrics(labels, scores, threshold)


def multiclass_roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Macro one-versus-rest area under the ROC curve.

    Each class against all the others, then averaged with equal weight per class -- so a
    rare phenotype counts as much as a common one, which is what a clinical multiclass
    endpoint usually wants and what a frequency-weighted average would hide.

    The multiclass counterpart of :func:`roc_auc`, and what
    :class:`~kalecardiac.pipeline.ClassificationTask` monitors for a multiclass
    endpoint. Reporting one class against the rest instead would make checkpoint
    selection quietly favour distinguishing that one class.

    Args:
        labels: ``(N,)`` class indices.
        scores: ``(N, num_classes)`` probabilities, or logits, which are passed through a
            softmax first.

    Returns:
        The macro area: 0.5 for a random ranking, 1.0 for a perfect one.

    Raises:
        MetricError: If the shapes disagree, a label is not a valid class index, or a
            class is absent -- its one-versus-rest curve is then undefined, which on a
            small validation split is ordinary rather than an error in the model.
    """
    return _kale_multiclass_roc_auc(labels, scores)


def multiclass_metrics(labels: ArrayLike, scores: ArrayLike, class_names: Sequence[str] | None = None) -> dict:
    """Summarise a multiclass prediction.

    Args:
        labels: ``(N,)`` class indices.
        scores: ``(N, num_classes)`` probabilities, or logits, which are passed through a
            softmax first.
        class_names: Names for the confusion matrix's axes. Defaults to the indices.

    Returns:
        Accuracy, balanced accuracy, macro F1, one-versus-rest macro ROC-AUC where it
        is defined, and the confusion matrix as a nested list.

    Raises:
        MetricError: If the shapes disagree, a label is not a valid class index, or fewer
            than two classes are present.
    """
    return _kale_multiclass_metrics(labels, scores, class_names)


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
    return _kale_mean_roc_curve(labels_per_run, scores_per_run, grid_size)
