"""Scoring predictions, by the task they came from.

Two modules retain the Cardiac-specific prediction and reporting APIs:

.. code-block:: text

    predictions.py             running a model over a loader, and keeping the result
    cross_validation.py        adapting resampled intervals and fold predictions

**No endpoint is named here.** Pulmonary hypertension detection and elevated wedge
pressure are both a binary label; a mean pulmonary arterial pressure is a number. Which
endpoint a cohort defines, and what a clinically useful operating point on it is, belong
to the experiment. General classification, regression, and fold-summary functions are
imported directly from :mod:`kale.evaluate`.
"""

from typing import TYPE_CHECKING

from kalecardiac.evaluate.cross_validation import bootstrap_ci, flatten_fold_summary, pool_folds
from kalecardiac.evaluate.predictions import SplitPredictions, predict_split, save_predictions
from kalecardiac.utils.renames import renamed

__all__ = [
    "SplitPredictions",
    "bootstrap_ci",
    "flatten_fold_summary",
    "pool_folds",
    "predict_split",
    "save_predictions",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
