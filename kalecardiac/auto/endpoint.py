"""The endpoint a configuration describes: checked, derived from a cohort table, trained and scored.

``cfg.ENDPOINT`` names a column of the cohort's table and what to make of it -- a
threshold on a measurement, a class the table records directly, or the measurement
itself. Which column and which threshold are the experiment's decisions. How a threshold
labels a subject, which task a class count implies and which metrics score it are the
same for every cohort, so they are made once, here::

    >>> import pandas as pd
    >>> from kalecardiac.auto import derive_endpoint, get_prediction_task
    >>> from kalecardiac.config import get_cfg_defaults
    >>> cfg = get_cfg_defaults()
    >>> cfg.ENDPOINT.VALUE_COLUMN, cfg.ENDPOINT.THRESHOLD = "pressure", 20.0
    >>> table = derive_endpoint(cfg, pd.DataFrame({"subject_id": ["a", "b", "c"], "pressure": [12.0, 20.0, 31.0]}))
    >>> table["label"].tolist()
    [0, 0, 1]
    >>> type(get_prediction_task(cfg, table["label"])).__name__
    'ClassificationTask'
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight
from yacs.config import CfgNode

from kalecardiac.evaluate import (
    SplitPredictions,
    binary_metrics,
    multiclass_metrics,
    multiclass_roc_auc,
    regression_metrics,
    roc_auc,
)
from kalecardiac.loaddata import LABEL_KEY, VALUE_KEY, ColumnTarget
from kalecardiac.pipeline import ClassificationTask, PredictionTask, RegressionTask

logger = logging.getLogger(__name__)


def check_endpoint(cfg: CfgNode) -> None:
    """Reject an endpoint that cannot be built, before any data is read.

    Args:
        cfg: The run configuration.

    Raises:
        ValueError: If no column is named, or the class count, where the class is read
            from and the task disagree -- naming the setting to change.
    """
    if not get_endpoint_column(cfg):
        raise ValueError(
            "no endpoint column: set ENDPOINT.VALUE_COLUMN to the cohort table's column the endpoint is measured "
            "in, or ENDPOINT.LABEL_COLUMN to one recording the class directly"
        )
    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    if num_classes < 2:
        raise ValueError(f"ENDPOINT.NUM_CLASSES must be at least 2, got {num_classes}")
    if cfg.ENDPOINT.REGRESSION:
        if cfg.ENDPOINT.LABEL_COLUMN or num_classes != 2:
            raise ValueError(
                "ENDPOINT.REGRESSION predicts ENDPOINT.VALUE_COLUMN itself and has no classes; leave "
                "ENDPOINT.LABEL_COLUMN empty and ENDPOINT.NUM_CLASSES at 2, or set ENDPOINT.REGRESSION False"
            )
        return
    if num_classes > 2 and not cfg.ENDPOINT.LABEL_COLUMN:
        raise ValueError(
            f"a threshold on {cfg.ENDPOINT.VALUE_COLUMN} defines two classes, not {num_classes}; set "
            f"ENDPOINT.LABEL_COLUMN to a column of class indices 0..{num_classes - 1}"
        )
    if num_classes > 2 and cfg.OBJECTIVE.POS_WEIGHT > 0:
        raise ValueError(
            f"OBJECTIVE.POS_WEIGHT weights the positive class of a binary endpoint, and this one has {num_classes} "
            f"classes; leave it 0 to weight every class from the training fold, or set it negative for no weighting"
        )


def get_endpoint_column(cfg: CfgNode) -> str:
    """The column of the cohort table the endpoint is read from: the class column if one is named."""
    return cfg.ENDPOINT.LABEL_COLUMN or cfg.ENDPOINT.VALUE_COLUMN


def derive_endpoint(cfg: CfgNode, table: pd.DataFrame) -> pd.DataFrame:
    """Add the ``label`` and ``value`` columns the endpoint is supervised and stratified on.

    The label is a threshold on ``ENDPOINT.VALUE_COLUMN`` -- strictly above it is
    positive -- or is read from ``ENDPOINT.LABEL_COLUMN`` where the cohort records the
    class directly. A subject with no recorded endpoint is dropped rather than assumed
    negative, and the run logs how many were.

    Args:
        cfg: The run configuration.
        table: One row per subject, holding the endpoint's column.

    Returns:
        A copy of the subjects that have the endpoint, with ``label`` added -- and
        ``value``, wherever the table records the measurement.

    Raises:
        ValueError: If no subject has the endpoint, or a recorded class is not one of
            ``0 .. ENDPOINT.NUM_CLASSES - 1``.
    """
    column = get_endpoint_column(cfg)
    recorded = pd.to_numeric(table[column], errors="coerce")
    known = recorded.notna()
    if not bool(known.all()):
        logger.info("dropping %d subjects with no recorded %s", int((~known).sum()), column)
        table = table[known].reset_index(drop=True)
        recorded = recorded[known].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"no subject has a recorded {column}, so there is no endpoint to fine-tune on")

    table = table.copy()
    if not cfg.ENDPOINT.LABEL_COLUMN:
        table[VALUE_KEY] = recorded.astype(float)
        table[LABEL_KEY] = (recorded > cfg.ENDPOINT.THRESHOLD).astype(int)
        return table

    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    invalid = sorted(set(recorded[(recorded % 1 != 0) | (recorded < 0) | (recorded >= num_classes)].tolist()))
    if invalid:
        raise ValueError(
            f"{column} records {invalid[:5]}, which are not class indices 0..{num_classes - 1}; encode the classes "
            f"as integers from 0, or set ENDPOINT.NUM_CLASSES to how many there are"
        )
    table[LABEL_KEY] = recorded.astype(int)
    # The measurement, where the table has it, is kept for the record; it does not
    # supervise a classification read from a column.
    if cfg.ENDPOINT.VALUE_COLUMN in table:
        table[VALUE_KEY] = pd.to_numeric(table[cfg.ENDPOINT.VALUE_COLUMN], errors="coerce").astype(float)
    return table


def get_endpoint_target(cfg: CfgNode, table: pd.DataFrame, id_column: str) -> ColumnTarget:
    """The supervision a derived table carries: its labels, or its measurements for a regression.

    Args:
        cfg: The run configuration.
        table: A table :func:`derive_endpoint` has derived.
        id_column: Column holding the subject identifiers.

    Returns:
        The target a :class:`~kalecardiac.loaddata.MultimodalDataset` hands its batches.
    """
    key = VALUE_KEY if cfg.ENDPOINT.REGRESSION else LABEL_KEY
    return ColumnTarget(table, columns={key: key}, id_column=id_column)


def get_class_balance(table: pd.DataFrame, num_classes: int) -> dict:
    """How a derived table's subjects divide between the classes, for a record or a log line.

    Args:
        table: A table :func:`derive_endpoint` has derived.
        num_classes: Classes of the endpoint.

    Returns:
        ``{"class_counts": [...]}`` for more than two classes, else ``{"n_positive",
        "positive_rate"}``.
    """
    labels = table[LABEL_KEY].to_numpy(dtype=int)
    if num_classes > 2:
        return {"class_counts": np.bincount(labels, minlength=num_classes).tolist()}
    return {"n_positive": int(labels.sum()), "positive_rate": float(labels.mean())}


def describe_class_balance(table: pd.DataFrame, num_classes: int) -> str:
    """The class balance as a phrase: ``"12 positive"`` or ``"classes 8/8/8"``."""
    balance = get_class_balance(table, num_classes)
    if "class_counts" in balance:
        return "classes " + "/".join(str(count) for count in balance["class_counts"])
    return f"{balance['n_positive']} positive"


def get_prediction_task(cfg: CfgNode, labels: ArrayLike) -> PredictionTask:
    """The task the endpoint implies: a class of it, or the measurement itself.

    Class balance is taken from ``labels``, which must be one fold's *training* labels:
    a balance taken over the whole cohort carries the test split's composition into the
    loss. ``OBJECTIVE.POS_WEIGHT`` 0 derives it as scikit-learn's ``"balanced"`` weights,
    a negative value disables it, and a positive one fixes a binary endpoint's positive
    weight.

    Args:
        cfg: The run configuration.
        labels: The training fold's labels; not read for a regression.

    Returns:
        A :class:`~kalecardiac.pipeline.RegressionTask` or a
        :class:`~kalecardiac.pipeline.ClassificationTask`, with the head ``MODEL``
        describes.

    Raises:
        ValueError: If a class has no subject among ``labels``, so nothing weights it.
    """
    common = {"hidden_dims": list(cfg.MODEL.HEAD_HIDDEN), "dropout": cfg.MODEL.DROPOUT}
    if cfg.ENDPOINT.REGRESSION:
        return RegressionTask(**common)

    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    if cfg.OBJECTIVE.POS_WEIGHT != 0:
        # Negative disables the weighting; positive fixes a binary endpoint's weight.
        return ClassificationTask(num_classes=num_classes, pos_weight=max(cfg.OBJECTIVE.POS_WEIGHT, 0.0), **common)

    training = np.asarray(labels).astype(int).reshape(-1)
    absent = sorted(set(range(num_classes)) - set(training.tolist()))
    if absent:
        raise ValueError(
            f"class(es) {absent} have no subject in this training fold, so there is nothing to weight them by; "
            f"use fewer folds, or set OBJECTIVE.POS_WEIGHT negative to train unweighted"
        )
    # scikit-learn's "balanced" weights, n / (classes * count). For a binary endpoint the
    # ratio of the two is negatives over positives, the usual positive weight.
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=training)
    if num_classes > 2:
        return ClassificationTask(num_classes=num_classes, class_weight=weights.tolist(), **common)
    return ClassificationTask(pos_weight=float(weights[1] / weights[0]), **common)


def score_predictions(cfg: CfgNode, predictions: SplitPredictions) -> dict:
    """Metrics for one split's predictions, chosen by what the endpoint is.

    Args:
        cfg: The run configuration.
        predictions: A split's scores, with the targets they are scored against.

    Returns:
        :func:`~kalecardiac.evaluate.regression_metrics`,
        :func:`~kalecardiac.evaluate.multiclass_metrics` or
        :func:`~kalecardiac.evaluate.binary_metrics`.
    """
    if cfg.ENDPOINT.REGRESSION:
        return regression_metrics(predictions.targets[VALUE_KEY], predictions.scores, unit=cfg.ENDPOINT.UNIT)
    if int(cfg.ENDPOINT.NUM_CLASSES) > 2:
        # The scores are logits, one column per class; the metric applies the softmax.
        return multiclass_metrics(predictions.targets[LABEL_KEY], predictions.scores)
    return binary_metrics(predictions.targets[LABEL_KEY], predictions.scores)


def get_headline_metric(cfg: CfgNode) -> str:
    """The metric a run is summarized by: the error for a measurement, else the threshold-free ranking."""
    if cfg.ENDPOINT.REGRESSION:
        return "rmse"
    return "roc_auc_ovr" if int(cfg.ENDPOINT.NUM_CLASSES) > 2 else "roc_auc"


def get_ablation_metric(cfg: CfgNode) -> tuple[str, Callable, str]:
    """What a rescoring with modalities withheld is measured in: ``(name, metric, target key)``.

    The threshold-free metric of each endpoint, so that a score which moves as a modality
    is withheld is not also moving with an operating point: ROC-AUC, the macro
    one-versus-rest ROC-AUC, or R² for a measurement.
    """
    if cfg.ENDPOINT.REGRESSION:
        return "r2", r2_score, VALUE_KEY
    if int(cfg.ENDPOINT.NUM_CLASSES) > 2:
        return "roc_auc_ovr", multiclass_roc_auc, LABEL_KEY
    return "roc_auc", roc_auc, LABEL_KEY


def format_headline(cfg: CfgNode, metrics: Mapping, folded: bool = False) -> str:
    """One line naming the metric that matters for this endpoint, for a log or a table.

    Args:
        cfg: The run configuration.
        metrics: One fold's metrics, or with ``folded`` a
            :func:`~kalecardiac.evaluate.summarize_folds` summary.
        folded: Report the mean and spread across folds.

    Returns:
        ``"roc_auc=0.812"``, or ``"roc_auc=0.812 +/- 0.040"`` across folds.
    """
    key = get_headline_metric(cfg)
    if not folded:
        return f"{key}={metrics.get(key, float('nan')):.3f}"
    entry = metrics.get(key, {})
    return f"{key}={entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}"
