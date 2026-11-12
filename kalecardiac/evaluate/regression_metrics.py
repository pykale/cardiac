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

Each is scikit-learn's. What this module adds is the clinical summary around them --
the signed bias, the unit, and an R-squared that is reported as missing, rather than as
a number, when the targets do not vary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kalecardiac.evaluate.classification_metrics import MetricError


def _as_pair(targets: ArrayLike, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.size != predictions.size:
        raise MetricError(f"got {targets.size} targets and {predictions.size} predictions")
    if targets.size == 0:
        raise MetricError("a regression metric needs at least one subject")
    return targets, predictions


def regression_metrics(targets: ArrayLike, predictions: ArrayLike, unit: str | None = None) -> dict:
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
    return {
        "n": int(targets.size),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        # Signed, not absolute: a model that systematically over-predicts a pressure is
        # wrong in a way a symmetric error cannot show.
        "bias": float(np.mean(predictions - targets)),
        "unit": unit,
        # Undefined when every target is the same value: scikit-learn reports 0 or 1
        # there, which a results table would show as a result.
        "r2": float(r2_score(targets, predictions)) if np.ptp(targets) > 0 else None,
    }
