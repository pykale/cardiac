"""Metrics for continuous endpoints.

A cardiac cohort's ground truth is often a measurement -- a mean pulmonary arterial
pressure, a wedge pressure, an ejection fraction -- that a study then thresholds into a
diagnosis. Predicting the measurement keeps what the threshold discards, and needs
metrics in the measurement's own units.
"""

from __future__ import annotations

from kale.evaluate.regression_metrics import regression_metrics as _kale_regression_metrics
from numpy.typing import ArrayLike

__all__ = ["regression_metrics"]


def regression_metrics(targets: ArrayLike, predictions: ArrayLike, unit: str | None = None) -> dict:
    """Summarise a continuous prediction.

    Root mean squared error is in the endpoint's units; mean absolute error is not
    dominated by the few subjects furthest out; and R-squared says how much of the
    cohort's variation the model accounts for, and is reported as missing when the
    targets do not vary.

    Args:
        targets: ``(N,)`` measured values.
        predictions: ``(N,)`` predicted values.
        unit: What one unit of the endpoint is. A display label only -- nothing is
            converted -- carried so that a reported RMSE says whether it is millimetres
            of mercury or percentage points.

    Returns:
        ``n``, ``rmse``, ``mae``, ``bias``, ``r2`` where it is defined, and ``unit``.

    Raises:
        MetricError: If the arrays disagree in length or are empty.
    """
    return _kale_regression_metrics(targets, predictions, unit)
