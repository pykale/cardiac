"""Workflows built from a configuration alone.

The stages below this one take explicit arguments; this one reads them from a YACS
``CfgNode``. What it holds is what every cardiac experiment does with the same settings
the same way -- the endpoint a table defines, the task and metrics it implies, and
pretraining, fine-tuning over folds and interpretation -- so that an example keeps only
its own decisions: where its cohort is, what its columns mean, and which model it builds.

``endpoint`` turns ``cfg.ENDPOINT`` into labels, a task and scores; ``workflow`` runs
pretraining and fine-tuning as ``cfg`` describes them.
"""

from kalecardiac.auto.endpoint import (
    check_endpoint,
    derive_endpoint,
    describe_class_balance,
    format_headline,
    get_ablation_metric,
    get_class_balance,
    get_endpoint_column,
    get_endpoint_target,
    get_headline_metric,
    get_prediction_task,
    score_predictions,
)
from kalecardiac.auto.workflow import (
    Fold,
    FoldResult,
    check_model_name,
    cross_validate,
    fine_tune,
    get_dataloader,
    get_ecg_transform_from_cfg,
    get_folds,
    get_source_sampling_rate,
    interpret_model,
    pretrain,
)

__all__ = [
    "Fold",
    "FoldResult",
    "check_endpoint",
    "check_model_name",
    "cross_validate",
    "derive_endpoint",
    "describe_class_balance",
    "fine_tune",
    "format_headline",
    "get_ablation_metric",
    "get_class_balance",
    "get_dataloader",
    "get_ecg_transform_from_cfg",
    "get_endpoint_column",
    "get_endpoint_target",
    "get_folds",
    "get_headline_metric",
    "get_prediction_task",
    "get_source_sampling_rate",
    "interpret_model",
    "pretrain",
    "score_predictions",
]


def __dir__():
    return sorted(__all__)
