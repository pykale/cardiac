"""CardioVAE: a signal-and-image multimodal VAE for cardiac phenotyping.

The model of Suvon et al., *Multimodal Variational Autoencoder for Low-cost Cardiac
Hemodynamics Instability Detection*, MICCAI 2024 (`arXiv:2403.13658
<https://arxiv.org/abs/2403.13658>`_), whose reference implementation is at
https://github.com/Shef-AIRE/AI4Cardiothoracic-CardioVAE.

The idea it contributes is clinical rather than architectural: the two cheapest,
most widely available cardiac investigations -- a chest radiograph and a resting ECG --
carry enough shared signal that a representation learned from large *unlabelled*
collections of them transfers to a small labelled cohort with an invasively measured
endpoint. The architecture that carries the idea is a product-of-experts multimodal
VAE, trained with a tri-stream objective so that each encoder remains usable when the
other modality is absent.

**What this class is.** A :class:`~kalecardiac.model.embed.MultimodalVAE` configured
with a signal encoder and decoder, an image encoder and decoder, and a product-of-experts
fusion. Nothing here reimplements the fusion, the sampling or the objective: they are the
generic ones, and this class exists because ``CardioVAE`` is the name the literature uses
for that configuration, and because a reader looking for it should find it.

**Relationship to PyKale.** PyKale carries an earlier, fixed-shape form of this model as
:class:`kale.embed.multimodal_encoder.SignalImageVAE`, with
:class:`kale.pipeline.multimodal_trainer.SignalImageTriStreamVAETrainer` for its
objective. This class differs in three ways, all of them consequences of being a library
component rather than one paper's model: the signal length, the image size and the
channel widths are arguments; the modalities are named, so a third one is a dictionary
entry rather than a new class; and missing modalities are carried per subject by a mask
rather than by passing ``None`` for a whole batch. Built with the default arguments
it is the same architecture as ``SignalImageVAE()``, parameter for parameter, and
:meth:`CardioVAE.load_pykale_state_dict` loads the weights PyKale -- or the reference
implementation -- saved.

**One deliberate difference from the reference implementation.** It samples from the
posterior during training and returns the mean during evaluation, from inside the
reparameterisation. Here sampling is always sampling, and a caller who wants the mean
asks for it -- see :func:`~kalecardiac.model.layers.reparameterize` for why.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from kale.utils.remap_model_parameters import remap_state_dict_keys
from torch import Tensor, nn

from kalecardiac.model.embed.encoders import VariationalEncoder, build_image_encoder, build_signal_encoder
from kalecardiac.model.embed.multimodal_vae import MultimodalVAE
from kalecardiac.model.layers.conv1d import Conv1dDecoder
from kalecardiac.model.layers.conv2d import Conv2dDecoder
from kalecardiac.utils.io import strip_state_dict

#: Modality names this model uses by default. Generic rather than ``"ecg"`` and
#: ``"cxr"``: the architecture is a 1-D modality and a 2-D one, and a cohort pairing an
#: ECG with a cardiac MRI slice uses the identical model.
SIGNAL = "signal"
IMAGE = "image"


class CardioVAE(MultimodalVAE):
    """A multimodal VAE over one signal modality and one image modality.

    A product of experts by default, which is the published model. The other latent
    fusions are accepted as ablations -- "CardioVAE with a mixture rather than a
    product" is a constructor argument, not a second class, exactly as
    :class:`~kalecardiac.model.embed.LSEMVAE` takes its own ablations.

    Args:
        signal_channels: Channels the signal has. One for a rhythm strip or a
            flattened multi-lead recording, twelve for a recording encoded lead-wise
            as channels.
        signal_length: Samples per channel.
        image_channels: Channels the image has.
        image_size: Image ``(height, width)``.
        latent_dim: Width of the shared latent space.
        channels: Encoder channel widths; the decoders reverse them.
        kernel_size: Kernel width of every convolution, encoders and decoders, signal
            and image alike. 3 is the published model; any width reconstructs the input
            size exactly. A non-convolutional ``signal_encoder`` ignores it.
        signal_encoder: Signal backbone, by name; see
            :data:`~kalecardiac.model.embed.SIGNAL_ENCODERS`.
        image_output_activation: Applied to the image reconstruction. ``"sigmoid"``
            matches an image preprocessed to the unit interval, which is what
            :func:`~kalecardiac.prepdata.get_image_transform` produces by default;
            pass ``None`` for a standardised image.
        signal_name: Name the signal modality carries in a batch.
        image_name: Name the image modality carries in a batch.
        fusion: Latent fusion; ``"poe"`` is the published model. See
            :data:`~kalecardiac.model.layers.EXPERT_FUSIONS` for the ablations.
        use_prior: Include the standard-normal expert in the product.
        num_groups: Groups a hierarchical fusion forms; used by ``"hime"`` only.

    Raises:
        ValueError: If a width or size is not positive.
        KeyError: If ``fusion`` is not a registered mechanism.
    """

    def __init__(
        self,
        signal_channels: int = 1,
        signal_length: int = 60000,
        image_channels: int = 1,
        image_size: Sequence[int] = (224, 224),
        latent_dim: int = 256,
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        signal_encoder: str = "conv",
        image_output_activation: str | None = "sigmoid",
        signal_name: str = SIGNAL,
        image_name: str = IMAGE,
        fusion: str = "poe",
        use_prior: bool = True,
        num_groups: int = 4,
    ) -> None:
        decoder_channels = tuple(reversed(tuple(channels)))
        encoders: dict[str, nn.Module] = {
            signal_name: VariationalEncoder(
                build_signal_encoder(
                    signal_encoder,
                    in_channels=signal_channels,
                    length=signal_length,
                    channels=channels,
                    kernel_size=kernel_size,
                )
                if signal_encoder == "conv"
                else build_signal_encoder(signal_encoder, in_channels=signal_channels, length=signal_length),
                latent_dim=latent_dim,
            ),
            image_name: VariationalEncoder(
                build_image_encoder(
                    in_channels=image_channels, size=image_size, channels=channels, kernel_size=kernel_size
                ),
                latent_dim=latent_dim,
            ),
        }
        decoders: dict[str, nn.Module] = {
            signal_name: Conv1dDecoder(
                latent_dim=latent_dim,
                out_channels=signal_channels,
                length=signal_length,
                channels=decoder_channels,
                kernel_size=kernel_size,
                # Linear: a standardised recording is centred on zero and takes both
                # signs, which no saturating activation can produce.
                output_activation=None,
            ),
            image_name: Conv2dDecoder(
                latent_dim=latent_dim,
                out_channels=image_channels,
                size=image_size,
                channels=decoder_channels,
                kernel_size=kernel_size,
                output_activation=image_output_activation,
            ),
        }

        super().__init__(
            encoders=encoders,
            decoders=decoders,
            latent_dim=latent_dim,
            fusion=fusion,
            use_prior=use_prior,
            **({"num_groups": num_groups} if fusion == "hime" else {}),
        )
        self.kernel_size = int(kernel_size)
        self.signal_name = signal_name
        self.image_name = image_name
        self.signal_length = int(signal_length)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def load_pykale_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True):
        """Load weights saved from PyKale's ``SignalImageVAE`` or the reference CardioVAE.

        The published CardioVAE checkpoint -- the one PyKale's cardiac tutorial
        fine-tunes from -- names its parameters as the reference implementation did.
        :func:`remap_pykale_state_dict` translates them, so the model built with the
        same sizes loads them unchanged::

            model = CardioVAE(latent_dim=256)
            model.load_pykale_state_dict(torch.load("CardioVAE.pth", map_location="cpu"))

        Args:
            state_dict: Parameter tensors keyed as PyKale or the reference saved them.
            strict: Require every parameter of this model to be loaded.

        Returns:
            ``load_state_dict``'s report of missing and unexpected keys.

        Raises:
            KeyError: If a key is not a ``SignalImageVAE`` parameter.
            RuntimeError: If ``strict`` and the sizes disagree -- a checkpoint saved
                with another latent width or signal length, say.
        """
        return self.load_state_dict(
            remap_pykale_state_dict(state_dict, self.signal_name, self.image_name), strict=strict
        )


#: A ``SignalImageVAE`` parameter name, after PyKale's own remapping.
_PYKALE_KEY = re.compile(r"(signal|image)_(encoder|decoder)\.(\w+?)(?:\.(\d+))?\.(weight|bias)")


def remap_pykale_state_dict(
    state_dict: Mapping[str, Tensor], signal_name: str = SIGNAL, image_name: str = IMAGE
) -> dict[str, Tensor]:
    """Rename a ``SignalImageVAE`` state dict for :class:`CardioVAE`.

    PyKale's :func:`kale.utils.remap_model_parameters.remap_state_dict_keys` is applied
    first, which turns the reference implementation's names (``ecg_encoder``,
    ``fc_logvar``) into PyKale's; this then maps PyKale's names onto this package's.
    A Lightning checkpoint is unwrapped first by
    :func:`~kalecardiac.utils.strip_state_dict`.

    Args:
        state_dict: Parameter tensors keyed as PyKale or the reference saved them.
        signal_name: The signal modality's name in the target model.
        image_name: The image modality's name in the target model.

    Returns:
        The same tensors, keyed as :class:`CardioVAE` names them.

    Raises:
        KeyError: If a key is not a ``SignalImageVAE`` parameter, which means the
            checkpoint is of another model and loading it would be a mistake.
    """
    # A checkpoint saved through a Lightning module holds the weights under
    # "state_dict", each prefixed "model."; strip_state_dict removes what wraps them.
    nested = state_dict.get("state_dict")
    state = strip_state_dict(nested if isinstance(nested, Mapping) else state_dict)
    names = {"signal": signal_name, "image": image_name}
    return {_cardiovae_key(key, names): value for key, value in remap_state_dict_keys(state).items()}


def _cardiovae_key(key: str, names: Mapping[str, str]) -> str:
    """Where one ``SignalImageVAE`` parameter lives in a :class:`CardioVAE`."""
    match = _PYKALE_KEY.fullmatch(key)
    if match is not None:
        modality, part, layer, index, kind = match.groups()
        name = names[modality]
        if part == "encoder":
            if layer == "conv_layers" and index is not None:
                return f"encoders.{name}.backbone.conv_layers.{index}.{kind}"
            if re.fullmatch(r"conv\d", layer):
                # The reference implementation's name for the same layer; PyKale keeps
                # both, so a PyKale state dict carries each convolution twice.
                return f"encoders.{name}.backbone.conv_layers.{int(layer[-1]) - 1}.{kind}"
            if layer in ("fc_mu", "fc_log_var"):
                return f"encoders.{name}.head.{'to_mean' if layer == 'fc_mu' else 'to_log_var'}.{kind}"
        elif layer == "fc":
            return f"decoders.{name}.project.{kind}"
        elif re.fullmatch(r"convtrans\d", layer):
            # Transposed convolutions alternate with activations in the decoder's stack.
            return f"decoders.{name}.deconv.{2 * (int(layer[-1]) - 1)}.{kind}"
    raise KeyError(f"{key!r} is not a SignalImageVAE parameter; this checkpoint is of another model")
