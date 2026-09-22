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
rather than by passing ``None`` for a whole batch. Built with the default arguments,
``CardioVAE(signal_length=60000, image_size=(224, 224), latent_dim=256)`` is the same
architecture, parameter for parameter, as ``SignalImageVAE``.

**One deliberate difference from the reference implementation.** It samples from the
posterior during training and returns the mean during evaluation, from inside the
reparameterisation. Here sampling is always sampling, and a caller who wants the mean
asks for it -- see :func:`~kalecardiac.model.layers.reparameterise` for why.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from kalecardiac.model.embed.encoders import VariationalEncoder, build_image_encoder, build_signal_encoder
from kalecardiac.model.embed.multimodal_vae import MultimodalVAE
from kalecardiac.model.layers.conv1d import Conv1dDecoder
from kalecardiac.model.layers.conv2d import Conv2dDecoder

#: Modality names this model uses by default. Generic rather than ``"ecg"`` and
#: ``"cxr"``: the architecture is a 1-D modality and a 2-D one, and a cohort pairing an
#: ECG with a cardiac MRI slice uses the identical model.
SIGNAL = "signal"
IMAGE = "image"


class CardioVAE(MultimodalVAE):
    """A product-of-experts VAE over one signal modality and one image modality.

    Args:
        signal_channels: Channels the signal has. One for a rhythm strip or a
            flattened multi-lead recording, twelve for a recording encoded lead-wise
            as channels.
        signal_length: Samples per channel.
        image_channels: Channels the image has.
        image_size: Image ``(height, width)``.
        latent_dim: Width of the shared latent space.
        channels: Encoder channel widths; the decoders reverse them.
        signal_encoder: Signal backbone, by name; see
            :data:`~kalecardiac.model.embed.SIGNAL_ENCODERS`.
        image_output_activation: Applied to the image reconstruction. ``"sigmoid"``
            matches an image preprocessed to the unit interval, which is what
            :func:`~kalecardiac.prepdata.build_image_pipeline` produces by default;
            pass ``None`` for a standardised image.
        signal_name: Name the signal modality carries in a batch.
        image_name: Name the image modality carries in a batch.
        use_prior: Include the standard-normal expert in the product.

    Raises:
        ValueError: If a width or size is not positive.
    """

    def __init__(
        self,
        signal_channels: int = 1,
        signal_length: int = 5000,
        image_channels: int = 1,
        image_size: Sequence[int] = (224, 224),
        latent_dim: int = 256,
        channels: Sequence[int] = (16, 32, 64),
        signal_encoder: str = "conv",
        image_output_activation: str | None = "sigmoid",
        signal_name: str = SIGNAL,
        image_name: str = IMAGE,
        use_prior: bool = True,
    ) -> None:
        decoder_channels = tuple(reversed(tuple(channels)))
        encoders: dict[str, nn.Module] = {
            signal_name: VariationalEncoder(
                build_signal_encoder(
                    signal_encoder, in_channels=signal_channels, length=signal_length, channels=channels
                )
                if signal_encoder == "conv"
                else build_signal_encoder(signal_encoder, in_channels=signal_channels, length=signal_length),
                latent_dim=latent_dim,
            ),
            image_name: VariationalEncoder(
                build_image_encoder(in_channels=image_channels, size=image_size, channels=channels),
                latent_dim=latent_dim,
            ),
        }
        decoders: dict[str, nn.Module] = {
            signal_name: Conv1dDecoder(
                latent_dim=latent_dim,
                out_channels=signal_channels,
                length=signal_length,
                channels=decoder_channels,
                # Linear: a standardised recording is centred on zero and takes both
                # signs, which no saturating activation can produce.
                output_activation=None,
            ),
            image_name: Conv2dDecoder(
                latent_dim=latent_dim,
                out_channels=image_channels,
                size=image_size,
                channels=decoder_channels,
                output_activation=image_output_activation,
            ),
        }

        super().__init__(
            encoders=encoders,
            decoders=decoders,
            latent_dim=latent_dim,
            fusion="poe",
            use_prior=use_prior,
        )
        self.signal_name = signal_name
        self.image_name = image_name
        self.signal_length = int(signal_length)
        self.image_size = (int(image_size[0]), int(image_size[1]))
