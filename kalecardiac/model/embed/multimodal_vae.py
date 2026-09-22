"""The multimodal variational autoencoder every cardiac model here is a case of.

One encoder per modality proposes a Gaussian over a shared latent space; a latent
fusion combines those proposals into a joint posterior; a sample from it is decoded
back into every modality. That is the whole model. What distinguishes the published
cardiac architectures from one another, and from the multimodal-VAE literature they
come from, is **which fusion** and **which decoders**:

.. code-block:: text

    fusion       decoders     the model this gives
    ---------------------------------------------------------------
    poe          per modality  MVAE (Wu and Goodman, 2018); CardioVAE
    mean         per modality  the parameter-averaging MMVAE+ variant
    barycenter   per modality  a Wasserstein-barycenter multimodal VAE
    hime         per modality  a grouped mixture-of-products VAE
    hime         shared        LS-EMVAE

Two arguments, five published models. That is why there is one class here and not five:
an ablation over fusion mechanisms is a configuration sweep, and a model "without the
mixture step" is ``fusion="poe"`` rather than a second file that has to be kept in step
with the first.

**Missing modalities are first class.** A batch carries a boolean per modality, the
fusion honours it, and the reconstruction term skips what was never there. A model
pretrained on twelve leads therefore runs on six without retraining -- which is the
transfer both cardiac studies rely on.

**Streams.** Trained on the joint posterior alone, an encoder is free to depend on its
siblings and becomes useless when they are absent. Passing ``unimodal_streams=True``
takes an additional forward pass through each modality on its own, so that each
encoder must also explain its own modality unaided. That is CardioVAE's tri-stream
pretraining, generalised from two modalities to any number, and the objective that
consumes it is :class:`~kalecardiac.pipeline.ReconstructionTask`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from kalecardiac.model.layers.gaussian import GaussianPosterior, build_expert_fusion, prior_expert


@dataclass
class VAEStream:
    """One forward pass through the model, on one subset of the modalities.

    Attributes:
        posterior (GaussianPosterior): The ``(B, latent_dim)`` joint posterior this
            pass inferred.
        latent (Tensor): ``(B, latent_dim)`` sample drawn from it.
        reconstructions (dict[str, Tensor]): One reconstruction per modality the model
            decodes, in that modality's own shape.
        modalities (tuple[str, ...]): Which modalities this pass was allowed to see.
    """

    posterior: GaussianPosterior
    latent: Tensor
    reconstructions: dict[str, Tensor]
    modalities: tuple[str, ...]


@dataclass
class VAEOutput:
    """What a :class:`MultimodalVAE` returns.

    Attributes:
        joint (VAEStream): The pass that saw every present modality. Always produced.
        unimodal (dict[str, VAEStream]): One pass per modality on its own, keyed by
            modality. Empty unless ``unimodal_streams`` was requested.
        modality_posteriors (dict[str, GaussianPosterior]): Each encoder's own
            proposal, before fusion. Read by the latent alignment objective, and the
            per-modality representation a downstream model embeds with.
        present (dict[str, Tensor]): ``(B,)`` bool per modality, as the batch carried
            it, so a loss can skip what was never there.
    """

    joint: VAEStream
    unimodal: dict[str, VAEStream] = field(default_factory=dict)
    modality_posteriors: dict[str, GaussianPosterior] = field(default_factory=dict)
    present: dict[str, Tensor] = field(default_factory=dict)

    @property
    def posterior(self) -> GaussianPosterior:
        """The joint posterior: shorthand for ``output.joint.posterior``."""
        return self.joint.posterior

    @property
    def latent(self) -> Tensor:
        """The joint latent sample: shorthand for ``output.joint.latent``."""
        return self.joint.latent


class MultimodalVAE(nn.Module):
    """A variational autoencoder over any number of named modalities.

    Args:
        encoders: One variational encoder per modality, keyed by the name the modality
            carries in a :class:`~kalecardiac.loaddata.SubjectBatch`. Each must expose
            ``latent_dim`` and return a
            :class:`~kalecardiac.model.layers.GaussianPosterior`.
        decoders: One decoder per modality, mapping ``(batch, latent_dim)`` to that
            modality's shape. **Passing the same module object under several keys
            shares it**, which is how a decoder shared across leads is expressed --
            ``nn.ModuleDict`` registers the parameters once, so the optimiser sees one
            decoder and the gradients from every lead reach it. Modalities absent from
            this mapping are encoded but not reconstructed, which is what an
            encode-only modality (a clinical vector, say) wants.
        latent_dim: Width of the shared latent space. Every encoder must agree with it.
        fusion: Latent fusion, by name; one of
            :data:`~kalecardiac.model.layers.EXPERT_FUSIONS`.
        use_prior: Include the standard-normal expert in the fusion. Applies to
            ``"poe"`` and ``"hime"``, where it keeps a single-modality posterior proper
            and gives a subject with nothing present a defined result. A mixture would
            merely be pulled towards zero by it, so it is ignored there.
        **fusion_kwargs: Options for the fusion, e.g. ``num_groups`` for ``"hime"``.

    Raises:
        ValueError: If no encoder is given, an encoder's ``latent_dim`` disagrees with
            the model's, or a decoder names a modality with no encoder.
        KeyError: If ``fusion`` is not a registered mechanism.
    """

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        decoders: Mapping[str, nn.Module] | None = None,
        latent_dim: int = 256,
        fusion: str = "poe",
        use_prior: bool = True,
        **fusion_kwargs,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("a multimodal VAE needs at least one modality encoder")

        decoders = dict(decoders or {})
        unknown = sorted(set(decoders) - set(encoders))
        if unknown:
            raise ValueError(f"decoders name modalities with no encoder: {unknown}")

        for name, encoder in encoders.items():
            encoder_dim = getattr(encoder, "latent_dim", None)
            if encoder_dim is None:
                raise ValueError(f"encoder {name!r} must expose 'latent_dim'; wrap its backbone in VariationalEncoder")
            if int(encoder_dim) != int(latent_dim):
                raise ValueError(
                    f"encoder {name!r} has latent_dim={encoder_dim} but the model declares {latent_dim}; every "
                    f"modality must propose a distribution over the same latent space for a fusion to combine them"
                )

        self.modalities = tuple(encoders)
        self.latent_dim = int(latent_dim)
        self.fusion_name = fusion
        self.use_prior = bool(use_prior)
        self.encoders = nn.ModuleDict(dict(encoders))
        self.decoders = nn.ModuleDict(decoders)
        self._fuse = build_expert_fusion(fusion, **fusion_kwargs)
        # A prior expert in a mixture only drags the joint towards zero without adding
        # the identifiability it gives a product, so it applies to products only.
        self._prior_applies = fusion in {"poe", "hime"}

    @property
    def reconstructed(self) -> tuple[str, ...]:
        """Modalities this model decodes, in encoder order."""
        return tuple(name for name in self.modalities if name in self.decoders)

    def encode(
        self,
        modalities: Mapping[str, Tensor],
        present: Mapping[str, Tensor] | None = None,
        only: str | None = None,
    ) -> tuple[GaussianPosterior, dict[str, GaussianPosterior]]:
        """Infer the joint posterior from whichever modalities are given.

        Args:
            modalities: One tensor per modality, keyed by name. A modality the model
                knows but the mapping omits is treated as absent.
            present: ``(B,)`` boolean per modality, as a batch carries it. ``None``
                treats every supplied modality as present.
            only: Restrict inference to this one modality, for a unimodal stream.

        Returns:
            ``(joint, per_modality)`` -- the fused posterior, and each encoder's own
            proposal before fusion.

        Raises:
            ValueError: If no modality is supplied, or ``only`` names one that is not.
        """
        supplied = [name for name in self.modalities if name in modalities]
        if only is not None:
            if only not in supplied:
                raise ValueError(f"only={only!r} was not supplied; available: {supplied}")
            supplied = [only]
        if not supplied:
            raise ValueError(f"no modality supplied; this model encodes {list(self.modalities)}")

        posteriors = {name: self.encoders[name](modalities[name]) for name in supplied}
        reference = posteriors[supplied[0]].mean
        batch_size = reference.shape[0]

        means = [posteriors[name].mean for name in supplied]
        log_vars = [posteriors[name].log_var for name in supplied]
        masks = [
            torch.ones(batch_size, device=reference.device, dtype=reference.dtype)
            if present is None or name not in present
            else present[name].to(device=reference.device, dtype=reference.dtype)
            for name in supplied
        ]

        if self.use_prior and self._prior_applies:
            prior_mean, prior_log_var = prior_expert(
                (1, batch_size, self.latent_dim), device=reference.device, dtype=reference.dtype
            )
            # Prepended, so the prior joins the first group of a hierarchical fusion
            # rather than trailing behind the modalities and changing the grouping.
            means.insert(0, prior_mean[0])
            log_vars.insert(0, prior_log_var[0])
            masks.insert(0, torch.ones(batch_size, device=reference.device, dtype=reference.dtype))

        joint = self._fuse(torch.stack(means), torch.stack(log_vars), torch.stack(masks))
        return joint, posteriors

    def decode(self, latent: Tensor) -> dict[str, Tensor]:
        """Reconstruct every decoded modality from a latent sample.

        Args:
            latent: ``(batch, latent_dim)`` sample.

        Returns:
            One reconstruction per modality in :attr:`reconstructed`.
        """
        return {name: self.decoders[name](latent) for name in self.reconstructed}

    def stream(
        self,
        modalities: Mapping[str, Tensor],
        present: Mapping[str, Tensor] | None = None,
        only: str | None = None,
    ) -> tuple[VAEStream, dict[str, GaussianPosterior]]:
        """One encode-sample-decode pass.

        Args:
            modalities: One tensor per modality.
            present: ``(B,)`` boolean per modality.
            only: Restrict the pass to one modality.

        Returns:
            ``(stream, per_modality_posteriors)``.
        """
        joint, posteriors = self.encode(modalities, present, only=only)
        latent = joint.sample()
        return (
            VAEStream(
                posterior=joint,
                latent=latent,
                reconstructions=self.decode(latent),
                modalities=tuple(posteriors),
            ),
            posteriors,
        )

    def forward(
        self,
        modalities: Mapping[str, Tensor],
        present: Mapping[str, Tensor] | None = None,
        unimodal_streams: bool = False,
    ) -> VAEOutput:
        """Encode, fuse, sample and reconstruct.

        Args:
            modalities: One tensor per modality, keyed by name.
            present: ``(B,)`` boolean per modality, as a
                :class:`~kalecardiac.loaddata.SubjectBatch` carries it.
            unimodal_streams: Also take a pass through each modality on its own, so
                each encoder must explain its own modality unaided. This is what makes
                a pretrained encoder usable when its siblings are missing, and it costs
                one extra forward pass per modality.

        Returns:
            The joint stream, any unimodal streams, and each encoder's own posterior.

        Raises:
            ValueError: If no known modality is supplied.
        """
        joint, posteriors = self.stream(modalities, present)

        unimodal: dict[str, VAEStream] = {}
        if unimodal_streams:
            for name in posteriors:
                unimodal[name], _ = self.stream(modalities, present, only=name)

        supplied = {name: value for name, value in (present or {}).items() if name in posteriors}
        return VAEOutput(joint=joint, unimodal=unimodal, modality_posteriors=posteriors, present=supplied)

    def latent_embedders(self, modalities: Mapping[str, Tensor] | None = None, frozen: bool = True) -> dict:
        """Wrap this model's encoders as embedders, for fine-tuning onto an endpoint.

        The step between pretraining and prediction, in one call::

            >>> predictor = MultimodalPredictor(vae.latent_embedders(), task.build_head)

        Args:
            modalities: Which modalities to expose, keyed as the model names them; the
                values are ignored, so a subset may be given as ``{"I": None}`` or as a
                list. ``None`` exposes every modality -- naming a subset is how a model
                pretrained on twelve leads is fine-tuned on six.
            frozen: Freeze the encoders, as both cardiac use cases do.

        Returns:
            One :class:`~kalecardiac.model.embed.LatentEmbedder` per modality.

        Raises:
            KeyError: If a named modality is not one of this model's.
        """
        from kalecardiac.model.embed.encoders import LatentEmbedder

        names = list(self.modalities if modalities is None else modalities)
        unknown = sorted(set(names) - set(self.modalities))
        if unknown:
            raise KeyError(f"this model has no modality {unknown}; it encodes {list(self.modalities)}")
        return {name: LatentEmbedder(self.encoders[name], frozen=frozen) for name in names}

    def __repr__(self) -> str:
        shared = len({id(self.decoders[name]) for name in self.reconstructed}) == 1 and len(self.reconstructed) > 1
        decoders = "shared decoder" if shared else f"{len(self.reconstructed)} decoders"
        return (
            f"{type(self).__name__}({len(self.modalities)} modalities | latent_dim={self.latent_dim} | "
            f"fusion={self.fusion_name} | {decoders})"
        )
