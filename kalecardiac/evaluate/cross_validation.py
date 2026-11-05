"""Aggregating a result across folds, and putting an interval on it.

A cardiac cohort of a few hundred subjects is evaluated by cross-validation, so what
gets reported is not one number but a distribution over folds. Two distinctions matter.

**How folds are summarised.** The mean and standard deviation *over folds* is what both
cardiac studies report, and it is what :func:`kale.evaluate.cross_validation.summarize_folds` computes. It is not the
same as pooling every fold's predictions into one array and scoring that: the pooled
figure weights subjects equally and the fold-wise mean weights folds equally, and for
unequal folds they differ. Both are defensible; reporting one and describing the other
is not, so both are available and named for what they do.

**What the interval means.** A standard deviation over five folds is a description of
how the estimate varied, not a confidence interval for the population: the folds share
subjects between their training sets and so are not independent. :func:`bootstrap_ci`
resamples *subjects* within a split instead, which answers the different and usually
more useful question of how much the number would move on another cohort of the same
size.

Use :func:`kale.evaluate.cross_validation.summarize_folds` for a summary over folds.
The functions retained here adapt PyKale to Cardiac's data or reporting contract:
``bootstrap_ci`` keeps its labels-first call and seed of 2026, ``pool_folds`` accepts
``SplitPredictions``, and ``flatten_fold_summary`` makes a row for a results table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
from kale.evaluate.bootstrap import bootstrap_ci as _kale_bootstrap_ci
from kale.evaluate.classification_metrics import MetricError
from kale.evaluate.cross_validation import pool_folds as _kale_pool_folds
from numpy.typing import ArrayLike


def flatten_fold_summary(summary: Mapping) -> dict:
    """A PyKale fold summary as flat columns, for one row of a results table.

    Each metric's mean and spread become a column of their own, and the per-fold values
    stay in the summary::

        >>> from kale.evaluate.cross_validation import summarize_folds
        >>> flatten_fold_summary(summarize_folds([{"roc_auc": 0.5}, {"roc_auc": 1.0}]))
        {'n_folds': 2, 'roc_auc_mean': 0.75, 'roc_auc_std': 0.25}

    Args:
        summary: As :func:`kale.evaluate.cross_validation.summarize_folds` returns it.

    Returns:
        ``n_folds``, then ``<metric>_mean`` and ``<metric>_std`` for every metric.
    """
    flat = {"n_folds": summary.get("n_folds")}
    for key, entry in summary.items():
        if isinstance(entry, Mapping) and "mean" in entry:
            flat[f"{key}_mean"] = entry["mean"]
            flat[f"{key}_std"] = entry["std"]
    return flat


def bootstrap_ci(
    labels: ArrayLike,
    scores: ArrayLike,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict:
    """A percentile bootstrap interval for a metric, resampling subjects.

    Resamples the split's subjects with replacement and recomputes the metric, so the
    interval describes how much the number would move on another cohort of the same
    size drawn the same way. It does *not* account for the model having been fitted on
    a particular training split.

    A resample in which one class is absent cannot be scored and is skipped rather than
    counted, which on a small or imbalanced cardiac cohort can discard a noticeable
    share; the count kept is reported so that a very small ``n_used`` is visible rather
    than silently widening the interval.

    Args:
        labels: ``(N,)`` targets.
        scores: ``(N,)`` predictions.
        metric: Takes ``(labels, scores)`` and returns a float, e.g.
            :func:`kale.evaluate.classification_metrics.roc_auc`.
        n_resamples: Bootstrap resamples to draw.
        confidence: Interval width, e.g. 0.95 for a 95% interval.
        seed: Seed, making the interval reproducible.

    Returns:
        ``"point"``, ``"low"``, ``"high"``, ``"confidence"`` and ``"n_used"``.

    Raises:
        MetricError: If the arrays disagree in length, the metric is not finite on every
            subject, or no resample could be scored to a finite value.
        ValueError: If ``n_resamples`` is not positive, or ``confidence`` is outside ``(0, 1)``.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise MetricError(f"got {labels.size} labels and {scores.size} scores")
    # A scikit-learn metric raises a plain ValueError on a one-class resample, so it marks
    # a resample as undefined here as MetricError does.
    return _kale_bootstrap_ci(
        metric,
        labels,
        scores,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        undefined_errors=(MetricError, ValueError),
    )


def pool_folds(fold_predictions: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every fold's labels and scores into one pair of arrays.

    The input to a *pooled* metric, which weights subjects equally where
    :func:`kale.evaluate.cross_validation.summarize_folds` weights folds equally. In a cross-validation where every
    subject is tested exactly once, the pooled arrays cover the whole cohort, and a
    metric over them is the cohort-level figure. Pool scores from different fitted
    models only when they share a comparable scale, such as calibrated probabilities.
    Otherwise, score each fold and use :func:`kale.evaluate.cross_validation.summarize_folds`.

    Args:
        fold_predictions: :class:`~kalecardiac.evaluate.SplitPredictions` per fold,
            each carrying a target.

    Returns:
        ``(labels, scores)`` over every fold.

    Raises:
        MetricError: If no fold is given, a fold carries no target, or a fold's targets
            and scores differ in length.
    """
    pairs = []
    for prediction in fold_predictions:
        if not prediction.targets:
            raise MetricError(f"fold {prediction.fold} carries no target, so there is nothing to score against")
        pairs.append((next(iter(prediction.targets.values())), prediction.scores))
    return _kale_pool_folds(pairs)
