"""What distinguishes one training objective from another.

A trainer's structure is the same whether it is pretraining a representation on
unlabelled recordings or fine-tuning one onto a clinical endpoint: run the batch
forward, take a loss, accumulate predictions, summarise the epoch. Only four things
change with the task -- the head, the loss, what the model is asked to produce, and
which metric summarises an epoch -- and those four live here.

That is what lets one trainer serve both halves of a cardiac workflow. A
:class:`PredictionTask` is passed to a trainer rather than baked into it, so
*pretraining is a task*, not a second trainer:

.. code-block:: text

    CardiacTrainer(vae,       task=ReconstructionTask(...))   # pretrain, no labels
    CardiacTrainer(predictor, task=ClassificationTask(...))   # fine-tune on an endpoint
    CardiacTrainer(predictor, task=RegressionTask())          # or on a measured value

Three tasks cover what a cardiac cohort asks for: a binary or multiclass diagnosis, a
continuous haemodynamic measurement, and the generative objective that pretrains the
representation both are read from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from kalecardiac.evaluate.classification_metrics import MetricError, roc_auc
from kalecardiac.evaluate.regression_metrics import r2_score
from kalecardiac.loaddata.multimodal_access import LABEL_KEY, VALUE_KEY, SubjectBatch
from kalecardiac.model.predict.heads import LinearHead, MLPHead
from kalecardiac.model.predict.losses import (
    binary_cross_entropy,
    cross_entropy,
    elbo_loss,
    latent_alignment_loss,
    mean_squared_error,
)


class PredictionTask(nn.Module, ABC):
    """The task-specific parts of training: head, loss, forward options, epoch metric.

    Subclasses are :class:`torch.nn.Module` so a task may own tensors -- a class weight
    is the case here -- and have them follow the trainer onto its device without the
    trainer knowing they exist.

    Attributes:
        target_keys: Batch target keys this task supervises on. Empty for an
            unsupervised task, which is what lets a cohort with no labels train.
        metric_name: Suffix of the epoch-level metric, logged as
            ``{split}_{metric_name}``.
        metric_mode: ``"max"`` or ``"min"``, so a caller configuring early stopping or
            checkpointing does not have to know which way the metric runs.
    """

    target_keys: tuple[str, ...] = ()
    metric_name: str = "loss"
    metric_mode: str = "min"

    @property
    def forward_kwargs(self) -> dict:
        """Extra keyword arguments the trainer passes to the model's forward pass.

        Empty for a task that only needs the ordinary pass. It exists because what a
        model is asked to *produce* is part of the objective rather than of the model:
        a tri-stream ELBO needs the per-modality passes, and nothing else does.
        """
        return {}

    @abstractmethod
    def build_head(self, in_features: int) -> nn.Module:
        """Build a prediction head for representations of width ``in_features``.

        Passed to :class:`~kalecardiac.model.embed.MultimodalPredictor` as its head
        factory.
        """

    @abstractmethod
    def loss(self, output, batch: SubjectBatch, epoch: int = 0) -> tuple[Tensor, dict[str, float]]:
        """Training objective for one batch.

        Takes the whole batch rather than just its targets, because a generative task
        is supervised by the *inputs*: its targets are the modalities themselves.

        Args:
            output: What the model returned for the batch.
            batch: The batch, carrying modalities, presence flags and any targets.
            epoch: Current epoch, for a schedule such as KL annealing. Ignored by a
                task with no schedule.

        Returns:
            ``(loss, parts)`` -- the scalar objective, and its terms for logging.
        """

    def scores(self, output) -> Tensor | None:
        """One score per subject, to accumulate for the epoch metric.

        ``None`` means there is nothing per-subject to accumulate, and the trainer
        then reports only the loss for that split. That is the honest answer for a
        generative task: its quality is the objective itself.
        """
        return None

    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        """Summarise one epoch's accumulated predictions.

        Args:
            scores: ``(N,)`` predictions for the epoch, in accumulation order.
            targets: Supervision for the same subjects, keyed by :attr:`target_keys`.

        Raises:
            MetricError: If this epoch's targets cannot support the metric. A trainer
                treats that as "skip this epoch's metric", not as a failure.
        """
        raise MetricError(f"{type(self).__name__} reports no epoch metric")


class ClassificationTask(PredictionTask):
    """Predict a categorical endpoint -- a diagnosis, a phenotype, a severity grade.

    Binary by default, which is what both cardiac use cases predict. With
    ``num_classes > 2`` the head widens and the loss becomes softmax cross-entropy;
    everything else, including the trainer, is unchanged.

    Heads emit logits rather than probabilities, because the fused sigmoid-and-log form
    is the numerically stable one and ROC-AUC is unchanged by the sigmoid.

    Args:
        num_classes: How many classes. 2 gives a single logit and binary
            cross-entropy.
        pos_weight: Weight on the positive class, binary only. A cardiac endpoint is
            usually imbalanced, and
            :func:`~kalecardiac.model.predict.positive_weight_from_labels` computes
            this from the *training* split. 0 disables it.
        class_weight: ``(num_classes,)`` weights, multiclass only; see
            :func:`~kalecardiac.model.predict.class_weight_from_labels`.
        hidden_dims: Hidden widths of the head. Empty gives a linear head.
        dropout: Dropout inside the head.

    Raises:
        ValueError: If ``num_classes`` is below 2, or a weight does not match it.
    """

    target_keys = (LABEL_KEY,)
    metric_name = "auc"
    metric_mode = "max"

    #: Declared for the type checker: a registered buffer is otherwise seen through
    #: ``nn.Module.__getattr__`` and typed as ``Tensor | Module``.
    pos_weight: Tensor | None
    class_weight: Tensor | None

    def __init__(
        self,
        num_classes: int = 2,
        pos_weight: float = 0.0,
        class_weight: Sequence[float] | Tensor | None = None,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if class_weight is not None and len(class_weight) != num_classes:
            raise ValueError(f"class_weight must have {num_classes} entries, got {len(class_weight)}")

        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        # Buffers rather than plain attributes so they follow the trainer to its
        # device, and non-persistent so they stay out of the checkpointed state --
        # they belong to a fold's class balance, not to the model.
        self.register_buffer(
            "pos_weight", torch.tensor(float(pos_weight)) if pos_weight > 0 else None, persistent=False
        )
        self.register_buffer(
            "class_weight",
            torch.as_tensor(class_weight, dtype=torch.float32) if class_weight is not None else None,
            persistent=False,
        )

    @property
    def out_features(self) -> int:
        """Scores per subject: one logit for a binary endpoint, one per class otherwise."""
        return 1 if self.num_classes == 2 else self.num_classes

    def build_head(self, in_features: int) -> nn.Module:
        if not self.hidden_dims:
            return LinearHead(in_features, self.out_features)
        return MLPHead(in_features, self.out_features, hidden_dims=self.hidden_dims, dropout=self.dropout)

    def loss(self, output, batch: SubjectBatch, epoch: int = 0) -> tuple[Tensor, dict[str, float]]:
        targets = batch.target[LABEL_KEY]
        if self.num_classes == 2:
            value = binary_cross_entropy(output.prediction, targets, pos_weight=self.pos_weight)
        else:
            value = cross_entropy(output.prediction, targets, class_weight=self.class_weight)
        return value, {"loss": float(value.detach())}

    def scores(self, output) -> Tensor:
        """The positive-class score: the logit when binary, else the positive column.

        One number per subject whichever it is, because a ranking metric needs an
        ordering and not a distribution. For a multiclass endpoint this reports the
        one-versus-rest AUC of class 1, which is a partial summary -- read the full
        confusion matrix from :mod:`kalecardiac.evaluate` rather than this.
        """
        prediction = output.prediction
        if prediction.dim() == 2 and prediction.shape[1] > 1:
            return torch.softmax(prediction, dim=1)[:, 1]
        return prediction.reshape(-1)

    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        labels = targets[LABEL_KEY].numpy()
        if self.num_classes > 2:
            labels = (labels == 1).astype(float)
        return roc_auc(labels, scores.double().numpy())


class RegressionTask(PredictionTask):
    """Predict a continuous endpoint -- a pressure, a volume, an ejection fraction.

    The counterpart of :class:`ClassificationTask` for a measurement rather than a
    label. Worth having explicitly: a cardiac cohort's invasive measurement is usually
    thresholded into a diagnosis, and predicting the measurement keeps the information
    that thresholding throws away.

    Args:
        hidden_dims: Hidden widths of the head.
        dropout: Dropout inside the head.
    """

    target_keys = (VALUE_KEY,)
    metric_name = "r2"
    metric_mode = "max"

    def __init__(self, hidden_dims: Sequence[int] = (128,), dropout: float = 0.5) -> None:
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)

    def build_head(self, in_features: int) -> nn.Module:
        if not self.hidden_dims:
            return LinearHead(in_features, 1)
        return MLPHead(in_features, 1, hidden_dims=self.hidden_dims, dropout=self.dropout)

    def loss(self, output, batch: SubjectBatch, epoch: int = 0) -> tuple[Tensor, dict[str, float]]:
        value = mean_squared_error(output.prediction, batch.target[VALUE_KEY])
        return value, {"loss": float(value.detach())}

    def scores(self, output) -> Tensor:
        return output.prediction.reshape(-1)

    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        return r2_score(targets[VALUE_KEY].double().numpy(), scores.double().numpy())


class ReconstructionTask(PredictionTask):
    """Pretrain a multimodal representation with a multimodal ELBO.

    The unlabelled half of a cardiac workflow: a large collection of recordings with no
    endpoint, from which a representation is learned that a small labelled cohort is
    then fine-tuned on. It supervises a
    :class:`~kalecardiac.model.embed.MultimodalVAE`, and it takes no targets at all --
    the batch's modalities are both the input and the supervision.

    Three settings cover the published cardiac objectives, and each is documented on
    the argument:

    * CardioVAE: ``unimodal_streams=True``, per-modality ``weights``, ``scale_factor``,
      KL annealing, no alignment.
    * LS-EMVAE: ``alignment_weight > 0``, ``scale_factor``, KL annealing, no unimodal
      streams.

    Args:
        weights: Reconstruction weight per modality; 1.0 where unnamed. CardioVAE
            weights its ECG ten times its radiograph, because a sum of squares over
            60,000 samples and one over 50,176 pixels are not comparable quantities.
        scale_factor: Applied to the summed reconstruction term, bringing it into range
            of the KL term.
        annealing_epochs: Epochs over which the KL weight rises from 0 to 1. 0 applies
            full weight from the first step, which on a strong decoder collapses the
            posterior to the prior before the encoder has learned anything.
        alignment_weight: Weight on
            :func:`~kalecardiac.model.predict.latent_alignment_loss`, which pulls each
            modality's posterior mean towards the joint one. 0 disables it.
        unimodal_streams: Also take an ELBO through each modality on its own, so each
            encoder must explain its own modality unaided. Costs one extra forward pass
            per modality.
        kinds: ``"mse"`` or ``"bce"`` per modality. ``"bce"`` suits a modality scaled
            to the unit interval whose decoder ends in a sigmoid -- an image, usually;
            ``"mse"`` suits a standardised signal, and is the default.
        free_bits: Nats per latent dimension exempt from the KL penalty.

    Raises:
        ValueError: If ``annealing_epochs`` or ``alignment_weight`` is negative.
    """

    target_keys = ()
    metric_name = "loss"
    metric_mode = "min"

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        scale_factor: float = 1.0,
        annealing_epochs: int = 0,
        alignment_weight: float = 0.0,
        unimodal_streams: bool = False,
        kinds: Mapping[str, str] | None = None,
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        if annealing_epochs < 0:
            raise ValueError(f"annealing_epochs must be non-negative, got {annealing_epochs}")
        if alignment_weight < 0:
            raise ValueError(f"alignment_weight must be non-negative, got {alignment_weight}")

        self.weights = dict(weights or {})
        self.kinds = dict(kinds or {})
        self.scale_factor = float(scale_factor)
        self.annealing_epochs = int(annealing_epochs)
        self.alignment_weight = float(alignment_weight)
        self.unimodal_streams = bool(unimodal_streams)
        self.free_bits = float(free_bits)

    @property
    def forward_kwargs(self) -> dict:
        return {"unimodal_streams": self.unimodal_streams}

    def build_head(self, in_features: int) -> nn.Module:
        raise NotImplementedError(
            "a reconstruction task has no prediction head: a multimodal VAE's output is its decoders' "
            "reconstructions, not a score. Fine-tune with a ClassificationTask or a RegressionTask to get one."
        )

    def annealing_factor(self, epoch: int) -> float:
        """Weight on the KL term at ``epoch``, rising linearly to 1."""
        if self.annealing_epochs <= 0:
            return 1.0
        return min(max(epoch, 0) / self.annealing_epochs, 1.0)

    def loss(self, output, batch: SubjectBatch, epoch: int = 0) -> tuple[Tensor, dict[str, float]]:
        annealing = self.annealing_factor(epoch)
        value, parts = elbo_loss(
            output.joint.reconstructions,
            batch.modalities,
            output.posterior,
            weights=self.weights,
            present=batch.present,
            kinds=self.kinds,
            annealing_factor=annealing,
            scale_factor=self.scale_factor,
            free_bits=self.free_bits,
        )
        parts["annealing"] = annealing

        # One ELBO per unimodal stream, so each encoder must stand on its own. Summed
        # rather than averaged, matching the tri-stream objective this generalises.
        for name, stream in output.unimodal.items():
            stream_loss, stream_parts = elbo_loss(
                stream.reconstructions,
                batch.modalities,
                stream.posterior,
                weights=self.weights,
                present=batch.present,
                kinds=self.kinds,
                annealing_factor=annealing,
                scale_factor=self.scale_factor,
                free_bits=self.free_bits,
            )
            value = value + stream_loss
            parts[f"elbo_{name}_only"] = stream_parts["recon"]

        if self.alignment_weight > 0:
            alignment = latent_alignment_loss(output.modality_posteriors, output.posterior, present=batch.present)
            value = value + self.alignment_weight * alignment
            parts["alignment"] = float(alignment.detach())

        parts["loss"] = float(value.detach())
        return value, parts
