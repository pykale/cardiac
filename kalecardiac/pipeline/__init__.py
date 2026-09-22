"""Training, as two independent choices.

:class:`CardiacTrainer` is the only trainer. What it trains is a *model*; what it
trains it *for* is a :class:`PredictionTask`. Neither constrains the other, so the same
class covers both halves of a cardiac workflow::

    CardiacTrainer(LSEMVAE(leads=STANDARD_12_LEAD), task=ReconstructionTask(alignment_weight=0.1))
    CardiacTrainer(MultimodalPredictor(embedders, task.build_head), task=ClassificationTask(pos_weight=3.0))
    CardiacTrainer(that_same_predictor, task=RegressionTask())

A task decides exactly four things -- the head, the loss, what the model is asked to
produce, and the epoch metric -- which is why adding an endpoint means adding a task
rather than another trainer, and why *pretraining is a task* rather than a second
trainer.

Orchestration is deliberately absent: how a cohort is assembled, split and reported on
belongs to an experiment, so it lives in ``examples/`` rather than here.
"""

from kalecardiac.pipeline.task import ClassificationTask, PredictionTask, ReconstructionTask, RegressionTask
from kalecardiac.pipeline.trainer import CardiacTrainer

__all__ = [
    "CardiacTrainer",
    "ClassificationTask",
    "PredictionTask",
    "ReconstructionTask",
    "RegressionTask",
]


def __dir__():
    return sorted(__all__)
