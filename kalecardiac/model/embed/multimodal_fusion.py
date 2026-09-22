"""Combining modality embeddings into one prediction.

The discriminative counterpart of :mod:`kalecardiac.model.embed.multimodal_vae`. Where
that fuses *distributions* to learn a representation, this fuses *vectors* to make a
prediction, and it is the stage a pretrained cardiac model is fine-tuned through:

.. code-block:: text

    embedders  ->  optional projection  ->  fusion  ->  head

Three fusions, differing in what they assume about the modalities:

``concat``
    Keep every modality's vector and let the head weigh them. Assumes nothing, costs a
    head whose width grows with the number of modalities, and is what both cardiac use
    cases fine-tune with -- twelve leads concatenated into one classifier.
``mean``
    Average them. Assumes the modalities are interchangeable views, which for the leads
    of one recording is nearly true, and gives a head whose width does not grow.
``attention``
    Learn a weight per modality per subject, then average. The middle case: the width
    does not grow, and the weights say which lead or which scan the model leaned on --
    which is a per-subject interpretation that ablation can only give per cohort.

A modality the batch marks absent is replaced by a learned placeholder rather than by
zeros, because an all-zero vector is indistinguishable from a genuine one, and it is
excluded from an average or an attention weighting entirely.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class PredictionOutput:
    """What a :class:`MultimodalPredictor` returns.

    Attributes:
        prediction (Tensor): ``(B, out_features)`` from the head.
        representation (Tensor): ``(B, fused_dim)`` fused vector, kept so a prediction
            can be projected, clustered or inspected without a second forward pass.
        embeddings (dict[str, Tensor]): Each modality's vector after projection, for
            per-modality interpretation.
        weights (Tensor | None): ``(B, num_modalities)`` fusion weights, for the
            attention fusion only.
    """

    prediction: Tensor
    representation: Tensor
    embeddings: dict[str, Tensor] = field(default_factory=dict)
    weights: Tensor | None = None


def modality_dropout(mask: Tensor, probability: float, generator: torch.Generator | None = None) -> Tensor:
    """Randomly mark present modalities absent, simulating missing data.

    Training a cardiac model this way is what makes it survive the deployment it is
    built for: a reduced-lead recorder, a subject with no recent radiograph. At least
    one modality is always kept, so no subject is left with nothing to predict from.

    Args:
        mask: ``(batch, num_modalities)`` indicator, one where present.
        probability: Chance of dropping each present modality.
        generator: RNG, for reproducibility.

    Returns:
        A new mask; the input is not modified.
    """
    if probability <= 0:
        return mask

    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) >= probability).to(mask.dtype)
    dropped = mask * keep
    empty = dropped.sum(dim=1) == 0
    if bool(empty.any()):
        dropped[empty, mask[empty].argmax(dim=1)] = 1.0
    return dropped


class FusionBlock(nn.Module):
    """Base class for combining modality embeddings.

    Args:
        input_dims: Width of each modality's embedding, in a fixed order.

    Raises:
        ValueError: If no modality is given.
    """

    def __init__(self, input_dims: Sequence[int]) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("fusion needs at least one modality")
        self.input_dims = [int(dim) for dim in input_dims]
        self.output_dim = self.input_dims[0]
        # A learned stand-in per modality, because zero-padding an absent modality is
        # indistinguishable from a genuine all-zero embedding.
        self.placeholders = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for dim in self.input_dims])

    def substitute(self, embeddings: list[Tensor], mask: Tensor) -> list[Tensor]:
        """Replace each absent modality's embedding with its learned placeholder."""
        present = mask.unsqueeze(-1)
        return [
            embedding * present[:, index] + placeholder * (1.0 - present[:, index])
            for index, (embedding, placeholder) in enumerate(zip(embeddings, self.placeholders, strict=True))
        ]

    def check(self, embeddings: list[Tensor], mask: Tensor | None) -> Tensor:
        """Validate the embeddings and return a usable ``(batch, num_modalities)`` mask.

        Raises:
            ValueError: If the count or a width disagrees with what was declared, or
                the mask has the wrong shape.
        """
        if len(embeddings) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} modalities, got {len(embeddings)}")
        for index, (embedding, dim) in enumerate(zip(embeddings, self.input_dims, strict=True)):
            if embedding.dim() != 2 or embedding.shape[-1] != dim:
                raise ValueError(f"modality {index} should be (batch, {dim}), got {tuple(embedding.shape)}")

        batch_size = embeddings[0].shape[0]
        if mask is None:
            return torch.ones(batch_size, len(embeddings), device=embeddings[0].device, dtype=embeddings[0].dtype)
        if mask.shape != (batch_size, len(embeddings)):
            raise ValueError(f"mask must be ({batch_size}, {len(embeddings)}), got {tuple(mask.shape)}")
        return mask.to(device=embeddings[0].device, dtype=embeddings[0].dtype)

    def forward(self, embeddings: list[Tensor], mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Fuse embeddings into ``(batch, output_dim)``, and any per-modality weights."""
        raise NotImplementedError


class ConcatFusion(FusionBlock):
    """Concatenate every modality's embedding.

    Nothing is assumed about how the modalities relate, and nothing is discarded; the
    head decides what to weigh. The cost is a head whose width is the sum of the
    modality widths, which for twelve leads at 256 dimensions is a 3072-wide input --
    ordinary for the cohorts here, and worth watching as modalities are added.
    """

    def __init__(self, input_dims: Sequence[int]) -> None:
        super().__init__(input_dims)
        self.output_dim = sum(self.input_dims)

    def forward(self, embeddings: list[Tensor], mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mask = self.check(embeddings, mask)
        return torch.cat(self.substitute(embeddings, mask), dim=1), None


class MeanFusion(FusionBlock):
    """Average the modality embeddings, ignoring absent ones.

    Requires every modality to share a width, which means a projection. In exchange the
    fused width is constant, so the same head serves a six-lead and a twelve-lead model
    -- the practical reason to prefer it when the lead configuration varies between
    cohorts.

    Raises:
        ValueError: If the modality widths differ.
    """

    def __init__(self, input_dims: Sequence[int]) -> None:
        super().__init__(input_dims)
        if len(set(self.input_dims)) != 1:
            raise ValueError(
                f"averaging needs every modality at the same width, got {self.input_dims}; project them first "
                f"by setting fusion_dim"
            )

    def forward(self, embeddings: list[Tensor], mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mask = self.check(embeddings, mask)
        stacked = torch.stack(embeddings, dim=1)
        share = mask / mask.sum(dim=1, keepdim=True).clamp_min(torch.finfo(mask.dtype).eps)
        return (stacked * share.unsqueeze(-1)).sum(dim=1), share


class AttentionFusion(FusionBlock):
    """Weight each modality per subject, then average.

    A scoring network reads each modality's embedding and produces one logit; the
    logits are softmaxed over the present modalities and used as averaging weights. The
    weights come back on the output, so "which lead did this subject's prediction rest
    on" is answerable per subject rather than only per cohort -- the per-subject
    counterpart of the ablation in :mod:`kalecardiac.interpret`.

    Args:
        input_dims: Width of each modality's embedding; all must be equal.
        attention_dim: Width of the scoring network's hidden layer.

    Raises:
        ValueError: If the modality widths differ.
    """

    def __init__(self, input_dims: Sequence[int], attention_dim: int = 64) -> None:
        super().__init__(input_dims)
        if len(set(self.input_dims)) != 1:
            raise ValueError(
                f"attention fusion needs every modality at the same width, got {self.input_dims}; project them "
                f"first by setting fusion_dim"
            )
        self.score = nn.Sequential(
            nn.Linear(self.input_dims[0], attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, embeddings: list[Tensor], mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mask = self.check(embeddings, mask)
        stacked = torch.stack(embeddings, dim=1)
        logits = self.score(stacked).squeeze(-1)
        # Absent modalities are pushed to zero weight through the softmax rather than
        # zeroed after it, so the present ones still sum to one.
        logits = logits.masked_fill(mask == 0, float("-inf"))
        # A subject with nothing present would softmax a row of -inf into NaN; falling
        # back to uniform keeps the forward pass finite, and the mask already says the
        # result is not evidence.
        empty = mask.sum(dim=1) == 0
        logits = torch.where(empty.unsqueeze(1), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights


#: Embedding fusions, by the name a configuration selects them with.
FUSION_METHODS: dict[str, type[FusionBlock]] = {
    "concat": ConcatFusion,
    "mean": MeanFusion,
    "attention": AttentionFusion,
}


def build_fusion(method: str, input_dims: Sequence[int], **kwargs) -> FusionBlock:
    """Construct a fusion block by name.

    Args:
        method: One of :data:`FUSION_METHODS`.
        input_dims: Width of each modality's embedding.
        **kwargs: Method-specific options, e.g. ``attention_dim``.

    Returns:
        The fusion block, exposing ``output_dim``.

    Raises:
        KeyError: If ``method`` is not registered.
    """
    if method not in FUSION_METHODS:
        raise KeyError(f"unknown fusion method {method!r}; available: {sorted(FUSION_METHODS)}")
    return FUSION_METHODS[method](input_dims, **kwargs)


def _embedder_dim(name: str, embedder: nn.Module) -> int:
    dim = getattr(embedder, "out_dim", None)
    if dim is None:
        raise AttributeError(
            f"embedder {name!r} must expose 'out_dim'; see kalecardiac.model.embed.LatentEmbedder and "
            f"FeatureEmbedder, which add it"
        )
    return int(dim)


class MultimodalPredictor(nn.Module):
    """Embed each modality, fuse, and predict.

    The downstream half of every workflow here. Its modalities may be leads, a
    recording and a radiograph, or a pretrained encoder bank and a clinical vector; it
    does not distinguish them, which is what lets one class serve a unimodal baseline,
    a two-modality fine-tune and a twelve-lead one.

    Args:
        embedders: One module per modality, each exposing ``out_dim`` and mapping its
            input to ``(batch, out_dim)``. Built by the caller -- from
            :meth:`~kalecardiac.model.embed.MultimodalVAE.latent_embedders` after
            pretraining, or from
            :class:`~kalecardiac.model.embed.FeatureEmbedder` to train from scratch.
        head_factory: Called with a width to build the task head, e.g.
            :meth:`~kalecardiac.pipeline.PredictionTask.build_head`. Injected so this
            class stays independent of what is being predicted.
        method: One of :data:`FUSION_METHODS`.
        fusion_dim: Width every modality is projected to first. ``None`` uses each
            embedder's own width and applies no projection, which is what
            concatenating pretrained encoder means directly amounts to and what the
            cardiac reference implementations do. An integer is needed for the
            averaging and attention fusions, which require equal widths.
        modality_dropout: Chance of marking each present modality absent during
            training.
        **fusion_kwargs: Passed to the fusion block, e.g. ``attention_dim``.

    Raises:
        ValueError: If no embedder is given.
        AttributeError: If an embedder does not declare ``out_dim``.
        KeyError: If ``method`` is not registered.
    """

    def __init__(
        self,
        embedders: Mapping[str, nn.Module],
        head_factory: Callable[[int], nn.Module],
        method: str = "concat",
        fusion_dim: int | None = None,
        modality_dropout: float = 0.0,
        **fusion_kwargs,
    ) -> None:
        super().__init__()
        if not embedders:
            raise ValueError("a predictor needs at least one modality")

        self.modalities = tuple(embedders)
        self.method = method
        self.modality_dropout = float(modality_dropout)
        self.embedders = nn.ModuleDict(dict(embedders))

        widths = [_embedder_dim(name, embedders[name]) for name in self.modalities]
        # Declared as a ModuleDict up front: an Identity per modality and a Linear per
        # modality are the same container, and typing it as nn.Module would make every
        # lookup below untypable.
        self.projections: nn.ModuleDict
        if fusion_dim is None:
            self.projections = nn.ModuleDict({name: nn.Identity() for name in self.modalities})
            fused_widths = widths
        else:
            self.projections = nn.ModuleDict(
                {name: nn.Linear(width, int(fusion_dim)) for name, width in zip(self.modalities, widths, strict=True)}
            )
            fused_widths = [int(fusion_dim)] * len(self.modalities)

        self.fusion = build_fusion(method, fused_widths, **fusion_kwargs)
        self.head = head_factory(self.fusion.output_dim)

    @property
    def fused_dim(self) -> int:
        """Width of the fused representation the head reads."""
        return self.fusion.output_dim

    def forward(
        self,
        modalities: Mapping[str, Tensor],
        present: Mapping[str, Tensor] | None = None,
    ) -> PredictionOutput:
        """Embed, fuse and predict.

        Args:
            modalities: One entry per modality, keyed by name; each is whatever that
                modality's embedder accepts.
            present: ``(B,)`` boolean per modality, as a
                :class:`~kalecardiac.loaddata.SubjectBatch` carries it.

        Returns:
            The prediction, the fused representation, the per-modality embeddings and
            any fusion weights.

        Raises:
            KeyError: If a configured modality is absent from ``modalities``.
        """
        missing = set(self.modalities) - set(modalities)
        if missing:
            raise KeyError(f"missing input for modalities {sorted(missing)}")

        embeddings = {name: self.projections[name](self.embedders[name](modalities[name])) for name in self.modalities}
        ordered = [embeddings[name] for name in self.modalities]
        mask = self._resolve_mask(present, ordered[0])
        representation, weights = self.fusion(ordered, mask)
        return PredictionOutput(
            prediction=self.head(representation),
            representation=representation,
            embeddings=embeddings,
            weights=weights,
        )

    def _resolve_mask(self, present: Mapping[str, Tensor] | None, reference: Tensor) -> Tensor:
        if present is None:
            mask = torch.ones(reference.shape[0], len(self.modalities), device=reference.device, dtype=reference.dtype)
        else:
            mask = torch.stack(
                [present[name].to(device=reference.device, dtype=reference.dtype) for name in self.modalities], dim=1
            )
        return modality_dropout(mask, self.modality_dropout) if self.training else mask

    def __repr__(self) -> str:
        return (
            f"MultimodalPredictor({len(self.modalities)} modalities | method={self.method} | "
            f"fused_dim={self.fused_dim})"
        )
