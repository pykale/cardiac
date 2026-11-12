"""Regression metrics, computed by PyKale (:mod:`kale.evaluate.regression_metrics`)."""

from __future__ import annotations

from kale.evaluate.regression_metrics import regression_metrics as _kale_regression_metrics
from numpy.typing import ArrayLike

__all__ = ["regression_metrics"]


def regression_metrics(targets: ArrayLike, predictions: ArrayLike, unit: str | None = None) -> dict:
    """RMSE, MAE, bias and R-squared; see :func:`kale.evaluate.regression_metrics.regression_metrics`."""
    return _kale_regression_metrics(targets, predictions, unit)
