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

Each is scikit-learn's, computed by :func:`kale.evaluate.regression_metrics.regression_metrics`,
which was adapted from this module and adds the clinical summary around them -- the
signed bias, the unit, and an R-squared that is reported as missing, rather than as a
number, when the targets do not vary.
"""

from __future__ import annotations

from kale.evaluate.regression_metrics import regression_metrics as _kale_regression_metrics
from numpy.typing import ArrayLike


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
    return _kale_regression_metrics(targets, predictions, unit)
