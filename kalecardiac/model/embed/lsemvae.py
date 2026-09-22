"""LS-EMVAE: a lead-specific multimodal VAE over the leads of one ECG.

The model of Suvon et al., *Multimodal Latent Fusion of ECG Leads for Early Assessment
of Pulmonary Hypertension* (`arXiv:2503.13470 <https://arxiv.org/abs/2503.13470>`_),
whose reference implementation is at https://github.com/Shef-AIRE/LS-EMVAE.

Its contribution is a reframing. A twelve-lead ECG is usually treated as one
twelve-channel signal, which asserts that the leads are commensurable channels of a
single view. They are not: each lead is a projection of the heart's electrical activity
onto a different axis, so they are better modelled as *different views of the same
underlying state* -- which is precisely what a multimodal VAE is for. Treating each lead
as its own modality then buys two things the channel view cannot: an encoder per lead
that keeps working when the other leads are absent, and a principled way to fuse lead
evidence that respects how much each lead actually saw.

Three pieces carry that:

**Lead-specific encoders.** One variational encoder per lead. The generic mechanism is
in :mod:`kalecardiac.loaddata.ecg_access` -- :func:`~kalecardiac.loaddata.lead_sources`
splits a recording into named per-lead modalities -- and after it, nothing in the model
knows that the modalities happen to be leads.

**Hierarchical modality expert fusion (HiME).** A product within groups of leads, then a
mixture across the groups. The product sharpens agreement between leads that see related
territory; the mixture keeps one over-confident group from determining the latent alone.
It is :func:`~kalecardiac.model.layers.hierarchical_experts`, and the ablations from the
paper are ``fusion="poe"`` and ``fusion="moe"`` rather than separate models.

**A shared decoder.** Every lead is reconstructed by the *same* decoder from the shared
latent. That is what forces the latent to carry lead-agnostic cardiac state rather than
letting each lead hide its own information in a private decoder, and it is the one
architectural difference from the grouped-fusion baseline the paper compares against.

The **latent alignment** term that completes the objective lives with the other losses,
as :func:`~kalecardiac.model.predict.latent_alignment_loss`, because it is a property of
the objective rather than of the architecture.

Two documented departures from the reference implementation
---------------------------------------------------------

**The prior expert is no longer discarded.** Its grouping computed
``group_size = num_experts // num_groups`` and then took ``num_groups`` slices of that
width. With twelve leads plus the prior and four groups, that is four groups of three,
covering experts 0 to 11 -- the thirteenth expert, the prior, was silently dropped.
:func:`~kalecardiac.model.layers.expert_groups` distributes the remainder instead, so
every expert is used. Pass ``use_prior=False`` to fuse the leads alone.

**The KL term is taken against the posterior that was actually sampled from.** Its
training loop drew ``z`` from the HiME posterior but evaluated the KL divergence against
the *arithmetic mean* of the per-lead posteriors, so the objective regularised a
distribution the model never sampled. Here one posterior serves both. A reader wanting
the original's KL term exactly can set ``fusion="mean"``, which makes the fused posterior
the arithmetic mean and the two agree by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from kalecardiac.loaddata.ecg_access import canonical_leads
from kalecardiac.model.embed.encoders import VariationalEncoder, build_signal_encoder
from kalecardiac.model.embed.multimodal_vae import MultimodalVAE
from kalecardiac.model.layers.conv1d import Conv1dDecoder


class LSEMVAE(MultimodalVAE):
    """A hierarchical-expert VAE with one encoder per lead and a shared decoder.

    Args:
        leads: Lead names, in the order they are stacked into experts. The order
            decides which leads a hierarchical fusion groups together, so it is a
            modelling choice: the conventional order puts the limb leads in the early
            groups and the precordial leads in the later ones.
        length: Samples per lead.
        latent_dim: Width of the shared latent space.
        channels: Encoder channel widths; the decoder reverses them.
        encoder: Signal backbone for each lead, by name; see
            :data:`~kalecardiac.model.embed.SIGNAL_ENCODERS`.
        fusion: Latent fusion. ``"hime"`` is the published model; ``"poe"`` and
            ``"moe"`` are its two ablations.
        num_groups: Groups a hierarchical fusion divides the experts into.
        shared_decoder: Reconstruct every lead with one decoder. ``False`` gives each
            lead its own, which is the grouped-fusion baseline the paper compares
            against.
        use_prior: Include the standard-normal expert.

    Raises:
        ValueError: If no lead is given, or a lead name repeats.
    """

    def __init__(
        self,
        leads: Sequence[str],
        length: int = 5000,
        latent_dim: int = 256,
        channels: Sequence[int] = (16, 32, 64),
        encoder: str = "conv",
        fusion: str = "hime",
        num_groups: int = 4,
        shared_decoder: bool = True,
        use_prior: bool = True,
    ) -> None:
        names = canonical_leads(leads)
        decoder_channels = tuple(reversed(tuple(channels)))

        encoders: dict[str, nn.Module] = {}
        for name in names:
            backbone = (
                build_signal_encoder(encoder, in_channels=1, length=length, channels=channels)
                if encoder == "conv"
                else build_signal_encoder(encoder, in_channels=1, length=length)
            )
            encoders[name] = VariationalEncoder(backbone, latent_dim=latent_dim)

        def make_decoder() -> nn.Module:
            return Conv1dDecoder(
                latent_dim=latent_dim,
                out_channels=1,
                length=length,
                channels=decoder_channels,
                output_activation=None,
            )

        # One module object under every key when shared: nn.ModuleDict registers the
        # parameters once, so the optimiser sees a single decoder and every lead's
        # gradient reaches it. Rebinding the same instance is the whole mechanism.
        if shared_decoder:
            decoder = make_decoder()
            decoders: dict[str, nn.Module] = dict.fromkeys(names, decoder)
        else:
            decoders = {name: make_decoder() for name in names}

        fusion_kwargs = {"num_groups": num_groups} if fusion == "hime" else {}
        super().__init__(
            encoders=encoders,
            decoders=decoders,
            latent_dim=latent_dim,
            fusion=fusion,
            use_prior=use_prior,
            **fusion_kwargs,
        )
        self.leads = names
        self.length = int(length)
        self.shared_decoder = bool(shared_decoder)
