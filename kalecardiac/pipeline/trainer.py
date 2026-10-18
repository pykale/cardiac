"""The trainer.

One trainer serves every experiment in this package. It takes a *model* and a
:class:`~kalecardiac.pipeline.task.PredictionTask`, and neither axis constrains the
other: pretraining a twelve-lead VAE on unlabelled recordings and fine-tuning a
six-lead classifier on a clinical endpoint are the same class with different arguments.
There is nothing left for a ``CardioVAETrainer`` or an ``LSEMVAETrainer`` to add.

That works because both variable parts were pushed out of the training loop. The task
supplies the head, the loss, what the model is asked to produce, and the epoch metric.
The model is anything that maps a batch's modalities and presence flags to an output
the task understands -- a :class:`~kalecardiac.model.embed.MultimodalVAE`, a
:class:`~kalecardiac.model.embed.MultimodalPredictor`, or something a user wrote.
Nothing here names a modality, a lead, a dataset or an endpoint.

**Why this takes a model where KaleCancer's trainer builds one.** KaleCancer's
``CohortTrainer`` constructs a fusion model from a set of embedders, because every
experiment there is discriminative. A cardiac workflow is two-stage: the generative
model that pretrains a representation and the discriminative one that reads it are
different models, trained in different runs, over cohorts of different sizes. A trainer
that could construct only one of them could not pretrain, so the model is an argument.

Epoch metrics are accumulated over the whole split rather than averaged over batches.
ROC-AUC and R-squared are properties of a cohort, and a mean of per-batch values is not
the value of the cohort: with a batch of 16 it would average 16-subject orderings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import Tensor, nn

from kalecardiac.evaluate.classification_metrics import MetricError
from kalecardiac.loaddata.multimodal_access import SubjectBatch
from kalecardiac.pipeline.task import PredictionTask

logger = logging.getLogger(__name__)


class CardiacTrainer(BaseNNTrainer):
    """Train a model over a cohort of subjects, for whatever task is given.

    Args:
        model: Maps ``(modalities, present)`` to an output the task understands.
            Built by the caller, so this trainer stays independent of any particular
            architecture.
        task: What is being trained for, e.g.
            :class:`~kalecardiac.pipeline.task.ReconstructionTask` to pretrain or
            :class:`~kalecardiac.pipeline.task.ClassificationTask` to fine-tune.
        optimizer: PyKale optimizer spec, e.g.
            ``{"type": "AdamW", "optim_params": {"weight_decay": 1e-5}}``. ``None``
            uses Adam at ``init_lr``.
        max_epochs: Maximum training epochs, passed on to PyKale's scheduler.
        init_lr: Initial learning rate.
        grad_clip: Clip the gradient norm to this value. 0 disables it. Worth setting
            for a VAE, whose reconstruction term can produce a large gradient in the
            first epochs before the KL has annealed in.

    Raises:
        ValueError: If ``grad_clip`` is negative.
    """

    def __init__(
        self,
        model: nn.Module,
        task: PredictionTask,
        optimizer: dict | None = None,
        max_epochs: int = 50,
        init_lr: float = 1e-3,
        grad_clip: float = 0.0,
    ) -> None:
        super().__init__(optimizer=optimizer, max_epochs=max_epochs, init_lr=init_lr)
        if grad_clip < 0:
            raise ValueError(f"grad_clip must be non-negative, got {grad_clip}")
        self.model = model
        self.task = task
        self.grad_clip = float(grad_clip)
        # Scores and targets are held apart rather than stacked into one tensor: a
        # multiclass task scores each subject with a row of probabilities, which cannot
        # share a stack with one target value per subject.
        self._epoch_scores: dict[str, list[Tensor]] = {}
        self._epoch_targets: dict[str, dict[str, list[Tensor]]] = {}

    def forward(self, batch: SubjectBatch):
        """Run the model over every modality the batch carries."""
        modalities = {name: value.to(self.device) for name, value in batch.modalities.items()}
        present = {name: value.to(self.device) for name, value in batch.present.items()}
        return self.model(modalities, present, **self.task.forward_kwargs)

    def predict(self, batch: SubjectBatch):
        """Model output for a batch, without tracking gradients.

        The training/evaluation mode is restored on return, so calling this during
        training does not silently disable dropout for subsequent steps -- which for a
        variational model would also stop the posterior being sampled.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch)
        finally:
            self.train(was_training)

    def compute_loss(self, batch: SubjectBatch, split_name: str = "valid") -> tuple[Tensor, dict]:
        """Loss for one batch, accumulating predictions on the evaluation splits.

        Args:
            batch: One batch.
            split_name: ``"train"``, ``"valid"`` or ``"test"``; names the logged
                metrics and decides whether predictions are accumulated.

        Returns:
            ``(loss, log_metrics)``, the second keyed ``{split}_{term}``.
        """
        batch = self._on_device(batch)
        output = self.forward(batch)
        loss, parts = self.task.loss(output, batch, epoch=self.current_epoch)

        if split_name != "train":
            scores = self.task.scores(output)
            if scores is not None:
                self._accumulate(split_name, scores, batch)

        return loss, {f"{split_name}_{name}": value for name, value in parts.items()}

    def training_step(self, batch: SubjectBatch, batch_idx: int) -> Tensor:
        loss, log_metrics = self.compute_loss(batch, split_name="train")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: SubjectBatch, batch_idx: int) -> None:
        loss, log_metrics = self.compute_loss(batch, split_name="valid")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch), prog_bar=False)

    def test_step(self, batch: SubjectBatch, batch_idx: int) -> None:
        loss, log_metrics = self.compute_loss(batch, split_name="test")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metric("valid")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metric("test")

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None) -> None:
        """Clip gradients, preferring this trainer's setting over Lightning's."""
        value = self.grad_clip or gradient_clip_val
        if value:
            self.clip_gradients(optimizer, gradient_clip_val=value, gradient_clip_algorithm="norm")

    def _on_device(self, batch: SubjectBatch) -> SubjectBatch:
        """Move a batch's tensors to the trainer's device."""
        return batch.to(self.device)

    def _accumulate(self, split_name: str, scores: Tensor, batch: SubjectBatch) -> None:
        """Hold one batch's predictions and targets until the epoch ends.

        ``scores`` is ``(B,)`` for a one-score task and ``(B, K)`` for one that scores
        each subject per class; either way the leading axis is the subject.
        """
        if scores.shape[0] != len(batch):
            raise ValueError(
                f"the task scored {scores.shape[0]} subjects in a batch of {len(batch)}; a task's scores() "
                f"must return one row per subject"
            )
        self._epoch_scores.setdefault(split_name, []).append(scores.detach().float().cpu())
        targets = self._epoch_targets.setdefault(split_name, {})
        for key in self.task.target_keys:
            targets.setdefault(key, []).append(batch.target[key].detach().float().reshape(-1).cpu())

    def _log_epoch_metric(self, split_name: str) -> None:
        """Compute the task's metric over everything accumulated this epoch."""
        batches = self._epoch_scores.pop(split_name, [])
        target_batches = self._epoch_targets.pop(split_name, {})
        if not batches:
            return

        scores = torch.cat(batches, dim=0)
        targets: Mapping[str, Tensor] = {key: torch.cat(values) for key, values in target_batches.items()}

        try:
            value = self.task.metric(scores, targets)
        except MetricError as error:
            # Ordinary on a small cardiac cohort: a validation split of eight subjects
            # can easily hold one class, and that is a reason to skip the metric rather
            # than to fail the run.
            logger.warning("%s %s unavailable: %s", split_name, self.task.metric_name, error)
            return
        self.log(f"{split_name}_{self.task.metric_name}", value, prog_bar=True)

    @property
    def monitor(self) -> str:
        """The metric a checkpoint or early-stopping callback should watch.

        The task's epoch metric where it has one, and the validation loss otherwise,
        which is the honest choice for a generative task. Pair it with
        :attr:`monitor_mode`.
        """
        return "valid_loss" if self.task.metric_name == "loss" else f"valid_{self.task.metric_name}"

    @property
    def monitor_mode(self) -> str:
        """``"max"`` or ``"min"``, matching :attr:`monitor`."""
        return self.task.metric_mode
