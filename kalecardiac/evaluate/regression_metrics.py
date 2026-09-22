"""Metrics for continuous endpoints.

A cardiac cohort's ground truth is often a measurement -- a mean pulmonary arterial
pressure, a wedge pressure, an ejection fraction -- that a study then thresholds into a
diagnosis. Predicting the measurement keeps what the threshold discards, and needs
metrics in the measurement's own units.

Three are reported together because each answers a different question. Root mean
squared error is in the endpoint's units and is what a clinician can compare against
measurement error; mean absolute error says the same thing but is not dominated by the
few subjects furthest out; and R-squared says how much of the cohort's variation the
model accounts for, which is the only one of the three that is comparable across
endpoints.
"""

from __future__ import annotations

import numpy as np

from kalecardiac.evaluate.classification_metrics import MetricError


def _as_pair(targets: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.size != predictions.size:
        raise MetricError(f"got {targets.size} targets and {predictions.size} predictions")
    if targets.size == 0:
        raise MetricError("a regression metric needs at least one subject")
    return targets, predictions


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Coefficient of determination.

    One minus the ratio of the residual sum of squares to the total sum of squares:
    0 means no better than predicting the cohort mean, and a negative value means
    worse than that, which is ordinary for a model that has not learned anything yet.

    Args:
        targets: ``(N,)`` measured values.
        predictions: ``(N,)`` predicted values.

    Returns:
        The coefficient.

    Raises:
        MetricError: If the arrays disagree in length, are empty, or every target is
            the same value -- in which case the total sum of squares is zero and the
            ratio undefined.
    """
    targets, predictions = _as_pair(targets, predictions)
    total = float(((targets - targets.mean()) ** 2).sum())
    if total == 0.0:
        raise MetricError(
            "every target has the same value, so there is no variation for R-squared to explain; report the "
            "root mean squared error instead"
        )
    residual = float(((targets - predictions) ** 2).sum())
    return 1.0 - residual / total


def regression_metrics(targets: np.ndarray, predictions: np.ndarray, unit: str | None = None) -> dict:
    """Summarise a continuous prediction.

    Args:
        targets: ``(N,)`` measured values.
        predictions: ``(N,)`` predicted values.
        unit: What one unit of the endpoint is. **A display label only** -- nothing is
            converted, so it cannot make a result wrong -- carried so that a reported
            RMSE says whether it is millimetres of mercury or percentage points.

    Returns:
        ``n``, ``rmse``, ``mae``, ``bias``, ``r2`` where it is defined, and ``unit``.

    Raises:
        MetricError: If the arrays disagree in length or are empty.
    """
    targets, predictions = _as_pair(targets, predictions)
    residual = predictions - targets

    summary = {
        "n": int(targets.size),
        "rmse": float(np.sqrt((residual**2).mean())),
        "mae": float(np.abs(residual).mean()),
        # Signed, not absolute: a model that systematically over-predicts a pressure is
        # wrong in a way a symmetric error cannot show.
        "bias": float(residual.mean()),
        "unit": unit,
    }
    try:
        summary["r2"] = r2_score(targets, predictions)
    except MetricError:
        summary["r2"] = None
    return summary
