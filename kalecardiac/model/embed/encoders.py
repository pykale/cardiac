"""Backbones adapted to the two contracts the rest of the package relies on.

A *backbone* is a block from :mod:`kalecardiac.model.layers`: it exposes ``out_dim``
and maps ``(batch, ...) -> (batch, out_dim)``. Two wrappers turn any backbone into
something the models here consume:

* :class:`VariationalEncoder` adds a Gaussian head, so the backbone emits a posterior
  and can be a modality of a multimodal VAE;
* :class:`FeatureEmbedder` adds an optional projection, so the backbone emits one
  vector per subject and can be a modality of a predictor.

and :class:`LatentEmbedder` bridges them: it takes a *pretrained* variational encoder
and presents its posterior mean as an embedding, which is exactly what fine-tuning a
representation-learning model onto a clinical endpoint means. Both cardiac use cases
do this -- freeze the encoders, take the means, concatenate, classify -- and it is one
class here rather than a rewritten classifier per model.

Keeping the backbone separate from the wrapper is what lets an experiment change
``ECG.ENCODER`` from ``conv`` to ``transformer`` without touching the VAE, the trainer
or the fine-tuning code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import Tensor, nn

from kalecardiac.model.layers.conv1d import Conv1dEncoder
from kalecardiac.model.layers.conv2d import Conv2dEncoder
from kalecardiac.model.layers.gaussian import GaussianHead, GaussianPosterior
from kalecardiac.model.layers.mlp import MLP
from kalecardiac.model.layers.residual1d import ResidualConv1dEncoder
from kalecardiac.model.layers.transformer import TransformerSignalEncoder

#: Signal backbones, by the name a configuration selects them with. Each takes
#: ``in_channels`` and ``length`` and exposes ``out_dim``.
SIGNAL_ENCODERS: dict[str, Callable[..., nn.Module]] = {
    "conv": Conv1dEncoder,
    "residual": ResidualConv1dEncoder,
    "transformer": TransformerSignalEncoder,
}


def build_signal_encoder(name: str, in_channels: int = 1, length: int = 5000, **kwargs) -> nn.Module:
    """Construct a signal backbone by name.

    ``length`` is passed only to backbones that need it. The convolutional encoder
    flattens, so its output width depends on the length; the residual and transformer
    encoders pool, so theirs does not, and handing them a length they would ignore
    would suggest it changed something.

    Args:
        name: One of :data:`SIGNAL_ENCODERS`.
        in_channels: Channels the signal has -- one per lead-as-modality, or all of
            them for a recording encoded together.
        length: Samples per channel.
        **kwargs: Backbone-specific options, e.g. ``channels`` or ``patch_size``.

    Returns:
        The backbone, exposing ``out_dim``.

    Raises:
        KeyError: If ``name`` is not registered.
        TypeError: If an option does not apply to that backbone.
    """
    if name not in SIGNAL_ENCODERS:
        raise KeyError(f"unknown signal encoder {name!r}; available: {sorted(SIGNAL_ENCODERS)}")
    if name == "conv":
        return SIGNAL_ENCODERS[name](in_channels=in_channels, length=length, **kwargs)
    return SIGNAL_ENCODERS[name](in_channels=in_channels, **kwargs)


def build_image_encoder(in_channels: int = 1, size: Sequence[int] = (224, 224), **kwargs) -> nn.Module:
    """Construct an image backbone.

    One implementation and so no registry: a convolutional encoder is what every
    cardiac imaging modality here uses, and a name-to-class map with a single entry
    would only suggest there were alternatives.

    Args:
        in_channels: Channels the image has.
        size: Input ``(height, width)``.
        **kwargs: Passed to :class:`~kalecardiac.model.layers.Conv2dEncoder`.

    Returns:
        The backbone, exposing ``out_dim``.
    """
    return Conv2dEncoder(in_channels=in_channels, size=size, **kwargs)


class VariationalEncoder(nn.Module):
    """Any backbone, plus a Gaussian head.

    One modality in, one posterior out. What a modality of a
    :class:`~kalecardiac.model.embed.MultimodalVAE` is, whatever the modality happens
    to be::

        >>> VariationalEncoder(build_signal_encoder("conv", 1, 5000), latent_dim=256)
        >>> VariationalEncoder(build_image_encoder(1, (224, 224)), latent_dim=256)

    Args:
        backbone: A block exposing ``out_dim`` and mapping ``(batch, ...)`` to
            ``(batch, out_dim)``.
        latent_dim: Width of the latent space.

    Raises:
        AttributeError: If the backbone does not declare ``out_dim``.
    """

    def __init__(self, backbone: nn.Module, latent_dim: int = 256) -> None:
        super().__init__()
        out_dim = getattr(backbone, "out_dim", None)
        if out_dim is None:
            raise AttributeError(
                f"backbone {type(backbone).__name__} must expose 'out_dim'; see kalecardiac.model.layers for "
                f"blocks that do"
            )
        self.backbone = backbone
        self.latent_dim = int(latent_dim)
        self.head = GaussianHead(int(out_dim), self.latent_dim)

    def forward(self, x: Tensor) -> GaussianPosterior:
        """Encode one modality into a ``(batch, latent_dim)`` posterior."""
        return self.head(self.backbone(x))

    def __repr__(self) -> str:
        return f"VariationalEncoder({self.backbone!r}, latent_dim={self.latent_dim})"


def freeze(module: nn.Module, frozen: bool = True) -> nn.Module:
    """Stop (or resume) a module accumulating gradients.

    Args:
        module: Module to freeze.
        frozen: ``True`` freezes, ``False`` unfreezes.

    Returns:
        The same module, for chaining.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)
    return module


class LatentEmbedder(nn.Module):
    """A pretrained variational encoder, presented as an embedder.

    The bridge between pretraining and fine-tuning. A
    :class:`~kalecardiac.model.embed.MultimodalVAE` learns one variational encoder per
    modality; wrapping each in this gives a set of embedders a
    :class:`~kalecardiac.model.embed.MultimodalPredictor` can fuse and put a clinical
    head on. That is what both cardiac use cases do, and doing it this way means the
    downstream model is the same class whether its encoders were pretrained or not.

    The **mean** is the embedding, not a sample. A downstream classifier wants a
    deterministic representation: sampling would make the same subject score
    differently on two passes, and the posterior mean is the encoder's best estimate.
    Pass ``sample=True`` only to use the posterior's noise as augmentation, and expect
    evaluation to become stochastic.

    A frozen encoder is also held in evaluation mode, because a frozen module with
    batch normalisation still updates its running statistics from whatever batches pass
    through it, which drifts a "frozen" encoder over an epoch of fine-tuning.

    Args:
        encoder: The variational encoder, exposing ``latent_dim`` and returning a
            :class:`~kalecardiac.model.layers.GaussianPosterior`.
        frozen: Stop the encoder accumulating gradients, as both cardiac use cases do
            when fine-tuning a small labelled cohort onto a large pretrained model.
        sample: Return a posterior sample rather than the mean.

    Raises:
        AttributeError: If the encoder does not declare ``latent_dim``.
    """

    def __init__(self, encoder: nn.Module, frozen: bool = True, sample: bool = False) -> None:
        super().__init__()
        latent_dim = getattr(encoder, "latent_dim", None)
        if latent_dim is None:
            raise AttributeError(
                f"{type(encoder).__name__} must expose 'latent_dim'; wrap a backbone in VariationalEncoder first"
            )
        self.encoder = freeze(encoder, frozen)
        self.frozen = frozen
        self.sample = sample
        self.out_dim = int(latent_dim)

    def train(self, mode: bool = True) -> LatentEmbedder:
        """Set training mode, keeping a frozen encoder in evaluation mode."""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Embed one modality into ``(batch, out_dim)``."""
        posterior = self.encoder(x)
        return posterior.sample() if self.sample else posterior.mean

    def __repr__(self) -> str:
        return f"LatentEmbedder({self.encoder!r}, frozen={self.frozen}, sample={self.sample})"


class FeatureEmbedder(nn.Module):
    """Any backbone, plus an optional projection: one modality in, one vector out.

    The non-variational counterpart of :class:`LatentEmbedder`, and what a supervised
    baseline trained from scratch is made of -- the convolutional classifier both
    cardiac studies compare against is this embedder plus a
    :class:`~kalecardiac.model.predict.MLPHead`. Having it here is what makes
    "pretrained or not" a choice of embedder rather than a second model.

    Args:
        backbone: A block exposing ``out_dim``.
        out_dim: Width to project to. ``None`` keeps the backbone's own width, which
            for a flattening convolutional encoder is very wide -- tens of thousands
            of features -- so a projection is usually wanted.
        hidden_dims: Hidden widths of the projection.
        dropout: Dropout inside the projection.

    Raises:
        AttributeError: If the backbone does not declare ``out_dim``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int | None = None,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        backbone_dim = getattr(backbone, "out_dim", None)
        if backbone_dim is None:
            raise AttributeError(f"backbone {type(backbone).__name__} must expose 'out_dim'")

        self.backbone = backbone
        if out_dim is None:
            self.projection: nn.Module = nn.Identity()
            self.out_dim = int(backbone_dim)
        else:
            self.projection = MLP(int(backbone_dim), int(out_dim), hidden_dims=hidden_dims, dropout=dropout)
            self.out_dim = int(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Embed one modality into ``(batch, out_dim)``."""
        return self.projection(self.backbone(x))

    def __repr__(self) -> str:
        return f"FeatureEmbedder({self.backbone!r}, out_dim={self.out_dim})"
