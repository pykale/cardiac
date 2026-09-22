"""Aggregating a result across folds, and putting an interval on it.

A cardiac cohort of a few hundred subjects is evaluated by cross-validation, so what
gets reported is not one number but a distribution over folds. Two things follow, and
both are here because both are routinely got wrong.

**How folds are summarised.** The mean and standard deviation *over folds* is what both
cardiac studies report, and it is what :func:`summarise_folds` computes. It is not the
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
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from kalecardiac.evaluate.classification_metrics import MetricError


def summarise_folds(fold_metrics: Sequence[Mapping[str, float]], keys: Sequence[str] | None = None) -> dict:
    """Mean and standard deviation of each metric across folds.

    Args:
        fold_metrics: One mapping of metric name to value per fold, as
            :func:`~kalecardiac.evaluate.binary_metrics` returns them.
        keys: Which metrics to summarise. ``None`` summarises every key whose value is
            numeric in every fold, which skips the nested confusion matrix and any
            string label without the caller having to list them.

    Returns:
        ``{metric: {"mean": ..., "std": ..., "values": [...]}}`` plus ``"n_folds"``.

    Raises:
        MetricError: If no fold is given.
    """
    folds = list(fold_metrics)
    if not folds:
        raise MetricError("summarise_folds needs at least one fold")

    if keys is None:
        keys = [
            key
            for key in folds[0]
            if all(isinstance(fold.get(key), int | float) and not isinstance(fold.get(key), bool) for fold in folds)
        ]

    summary: dict = {"n_folds": len(folds)}
    for key in keys:
        values = np.array([float(fold[key]) for fold in folds if key in fold], dtype=float)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            # Population standard deviation over the folds observed, matching what both
            # cardiac studies report. It describes the spread of these folds, not the
            # uncertainty of the mean.
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return summary


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
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
            :func:`~kalecardiac.evaluate.roc_auc`.
        n_resamples: Bootstrap resamples to draw.
        confidence: Interval width, e.g. 0.95 for a 95% interval.
        seed: Seed, making the interval reproducible.

    Returns:
        ``"point"``, ``"low"``, ``"high"``, ``"confidence"`` and ``"n_used"``.

    Raises:
        MetricError: If the arrays disagree in length, or no resample could be scored.
        ValueError: If ``confidence`` is outside ``(0, 1)``.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise MetricError(f"got {labels.size} labels and {scores.size} scores")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    generator = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        draw = generator.integers(0, labels.size, size=labels.size)
        try:
            values.append(float(metric(labels[draw], scores[draw])))
        except (MetricError, ValueError):
            continue

    if not values:
        raise MetricError(
            "no bootstrap resample could be scored; with this few subjects, or this severe an imbalance, "
            "almost every resample holds one class"
        )

    tail = (1.0 - confidence) / 2.0
    return {
        "point": float(metric(labels, scores)),
        "low": float(np.percentile(values, 100 * tail)),
        "high": float(np.percentile(values, 100 * (1.0 - tail))),
        "confidence": float(confidence),
        "n_used": len(values),
    }


def pool_folds(fold_predictions: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every fold's labels and scores into one pair of arrays.

    The input to a *pooled* metric, which weights subjects equally where
    :func:`summarise_folds` weights folds equally. In a cross-validation where every
    subject is tested exactly once, the pooled arrays cover the whole cohort, and a
    metric over them is the cohort-level figure.

    Args:
        fold_predictions: :class:`~kalecardiac.evaluate.SplitPredictions` per fold,
            each carrying a target.

    Returns:
        ``(labels, scores)`` over every fold.

    Raises:
        MetricError: If no fold is given, or a fold carries no target.
    """
    labels, scores = [], []
    for prediction in fold_predictions:
        if not prediction.targets:
            raise MetricError(f"fold {prediction.fold} carries no target, so there is nothing to score against")
        labels.append(next(iter(prediction.targets.values())))
        scores.append(prediction.scores)

    if not labels:
        raise MetricError("pool_folds needs at least one fold")
    return np.concatenate(labels), np.concatenate(scores)
