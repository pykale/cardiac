"""Scoring predictions, by the task they came from.

Four modules, named for the task they score and the thing they produce:

.. code-block:: text

    classification_metrics.py  ROC-AUC, sensitivity, specificity, confusion, mean ROC
    regression_metrics.py      RMSE, MAE, bias, R-squared
    predictions.py             running a model over a loader, and keeping the result
    cross_validation.py        summarising folds, and resampled intervals

**No endpoint is named here.** Pulmonary hypertension detection and elevated wedge
pressure are both a binary label; a mean pulmonary arterial pressure is a number. Which
endpoint a cohort defines, and what a clinically useful operating point on it is, belong
to the experiment -- the library supplies the metrics that decision is expressed in.
"""

from kalecardiac.evaluate.classification_metrics import (
    MetricError,
    binary_metrics,
    mean_roc_curve,
    multiclass_metrics,
    roc_auc,
    sensitivity_specificity,
)
from kalecardiac.evaluate.cross_validation import bootstrap_ci, pool_folds, summarise_folds
from kalecardiac.evaluate.predictions import SplitPredictions, predict_split, save_predictions
from kalecardiac.evaluate.regression_metrics import r2_score, regression_metrics

__all__ = [
    "MetricError",
    "SplitPredictions",
    "binary_metrics",
    "bootstrap_ci",
    "mean_roc_curve",
    "multiclass_metrics",
    "pool_folds",
    "predict_split",
    "r2_score",
    "regression_metrics",
    "roc_auc",
    "save_predictions",
    "sensitivity_specificity",
    "summarise_folds",
]


def __dir__():
    return sorted(__all__)
