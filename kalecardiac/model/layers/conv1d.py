"""Convolutional blocks over one-dimensional signals.

An encoder reduces a ``(batch, channels, length)`` recording to a flat feature vector,
and a decoder reconstructs one from a latent vector. Neither knows what the channels
are: a single lead, six leads, a pressure trace and a flattened twelve-lead strip are
the same tensor with a different ``in_channels``.

Both are built on :class:`kale.embed.base_cnn.BaseCNN`, PyKale's shared convolution-
block machinery, and both differ from PyKale's ``SignalVAEEncoder`` and
``SignalVAEDecoder`` in one respect that matters for a library: the channel widths and
the input length are arguments, and the flattened width is *measured* rather than
derived from a formula that assumes three stride-2 layers. PyKale's classes fix both,
so a cohort sampled at another rate or a model with another depth cannot use them; a
:class:`Conv1dEncoder` with the default arguments and ``length`` divisible by eight is
the same architecture, parameter for parameter.

The decoder's transposed convolutions use ``kernel_size=4, stride=2, padding=1``, which
doubles the length exactly. The equivalent ``kernel_size=3`` with ``output_padding=1``
also doubles it, and is what CardioVAE's decoder uses; the two differ only in the
kernel's width, and :func:`Conv1dDecoder` takes whichever is asked for.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from kale.embed.base_cnn import BaseCNN
from torch import Tensor, nn


def conv_output_length(length: int, channels: Sequence[int], kernel_size: int, stride: int, padding: int) -> int:
    """Length of a signal after a stack of convolutions, by the standard formula.

    Args:
        length: Input length.
        channels: One entry per convolution; only the count matters here.
        kernel_size: Kernel width of each convolution.
        stride: Stride of each convolution.
        padding: Padding of each convolution.

    Returns:
        The output length.

    Raises:
        ValueError: If the signal is reduced to nothing, which means the stack is too
            deep or too aggressive for the input.
    """
    original = length
    for _ in channels:
        length = (length + 2 * padding - kernel_size) // stride + 1
        if length < 1:
            raise ValueError(
                f"a stack of {len(channels)} convolutions with kernel {kernel_size} and stride {stride} reduces "
                f"a signal of {original} samples to nothing; use fewer layers, a smaller stride, or a longer input"
            )
    return length


class Conv1dEncoder(BaseCNN):
    """Encode a ``(batch, channels, length)`` signal into a flat feature vector.

    Args:
        in_channels: Channels the signal has. One for a single lead, twelve for a
            whole recording encoded together.
        length: Samples per channel. Fixed because the encoder ends in a flatten, and
            a flatten of a variable length cannot feed a linear layer.
        channels: Output channels of each convolution, one entry per layer.
        kernel_size: Kernel width of every convolution.
        stride: Stride of every convolution. 2 halves the length per layer.
        padding: Padding of every convolution.
        batch_norm: Insert batch normalisation after each convolution. Off by default,
            matching the cardiac reference implementations; worth turning on for a
            deeper stack, where it is what keeps the activations from drifting.
        activation: Non-linearity after each convolution, by PyKale's name.

    Raises:
        ValueError: If a width is not positive, or the stack reduces the signal to
            nothing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        length: int = 5000,
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_channels < 1 or length < 1 or not channels:
            raise ValueError(
                f"need positive in_channels, length and at least one layer, got {in_channels}, {length}, {list(channels)}"
            )

        self.in_channels = int(in_channels)
        self.length = int(length)
        self.channels = tuple(int(width) for width in channels)
        self.activation = activation

        self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
            in_channels=self.in_channels,
            out_channels_size_list=list(self.channels),
            kernel_sizes=int(kernel_size),
            conv_type="1d",
            strides=int(stride),
            conv_padding=int(padding),
            use_batch_norm=batch_norm,
            bias=not batch_norm,
        )
        self.use_batch_norm = batch_norm
        self.output_length = conv_output_length(self.length, self.channels, kernel_size, stride, padding)
        self.out_dim = self.channels[-1] * self.output_length

    def forward(self, x: Tensor) -> Tensor:
        """Encode a signal.

        Args:
            x: ``(batch, in_channels, length)`` signal.

        Returns:
            ``(batch, out_dim)`` features.

        Raises:
            ValueError: If ``x`` does not have the declared channels and length.
        """
        if x.dim() != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.length:
            raise ValueError(f"expected (batch, {self.in_channels}, {self.length}) signal, got {tuple(x.shape)}")
        for conv, norm in zip(self.conv_layers, self.batch_norms, strict=True):
            x = self._apply_activation(norm(conv(x)) if self.use_batch_norm else conv(x), self.activation)
        return self._flatten_features(x)

    def __repr__(self) -> str:
        return (
            f"Conv1dEncoder(in_channels={self.in_channels}, length={self.length}, "
            f"channels={list(self.channels)}, out_dim={self.out_dim})"
        )


class Conv1dDecoder(nn.Module):
    """Reconstruct a ``(batch, out_channels, length)`` signal from a latent vector.

    A linear projection to a feature map, then transposed convolutions that widen it
    back to ``length``. The last layer is linear, which is right for a standardised
    recording: a signal centred on zero with both signs is not something a sigmoid or
    a ReLU can produce.

    The transposed stack multiplies the length by ``stride`` per layer, so the
    projected map is ``length / stride**layers`` wide. Where that is not an integer the
    reconstruction is cropped or zero-padded to ``length`` at the end, so any length
    works -- but a length divisible by ``stride**layers`` avoids the seam entirely and
    is worth choosing.

    Args:
        latent_dim: Width of the latent vector.
        out_channels: Channels to reconstruct.
        length: Samples per channel to reconstruct.
        channels: Channels of each transposed convolution, from the widest inwards.
            The reverse of an encoder's ``channels`` by convention.
        kernel_size: Kernel width. 4 with ``padding=1`` doubles the length exactly;
            3 with ``output_padding=1`` does too.
        stride: Stride of each transposed convolution.
        activation: Non-linearity between layers.
        output_activation: Applied to the reconstruction. ``None`` leaves it linear,
            which a standardised signal needs. Pass ``"sigmoid"`` for a modality scaled
            to the unit interval.

    Raises:
        ValueError: If a width is not positive, or ``output_activation`` is unknown.
    """

    _OUTPUT_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 1,
        length: int = 5000,
        channels: Sequence[int] = (64, 32, 16),
        kernel_size: int = 4,
        stride: int = 2,
        activation: str = "relu",
        output_activation: str | None = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or out_channels < 1 or length < 1 or not channels:
            raise ValueError(
                f"need positive latent_dim, out_channels, length and at least one layer, "
                f"got {latent_dim}, {out_channels}, {length}, {list(channels)}"
            )

        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.length = int(length)
        self.channels = tuple(int(width) for width in channels)

        upsample = stride ** len(self.channels)
        # At least one sample to widen from, however short the target.
        self.start_length = max(self.length // upsample, 1)
        self.project = nn.Linear(self.latent_dim, self.channels[0] * self.start_length)

        padding = (kernel_size - stride) // 2
        output_padding = stride - kernel_size + 2 * padding
        layers: list[nn.Module] = []
        widths = [*self.channels, self.out_channels]
        for index in range(len(widths) - 1):
            layers.append(
                nn.ConvTranspose1d(
                    widths[index],
                    widths[index + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if index < len(widths) - 2:
                layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU())
        self.deconv = nn.Sequential(*layers)

        if output_activation is None:
            self.output_activation: nn.Module = nn.Identity()
        elif output_activation in self._OUTPUT_ACTIVATIONS:
            self.output_activation = self._OUTPUT_ACTIVATIONS[output_activation]()
        else:
            raise ValueError(
                f"unknown output_activation {output_activation!r}; available: {sorted(self._OUTPUT_ACTIVATIONS)} or None"
            )

    def forward(self, z: Tensor) -> Tensor:
        """Reconstruct a signal.

        Args:
            z: ``(batch, latent_dim)`` latent vector.

        Returns:
            ``(batch, out_channels, length)`` reconstruction.

        Raises:
            ValueError: If ``z`` is not ``(batch, latent_dim)``.
        """
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected (batch, {self.latent_dim}) latent, got {tuple(z.shape)}")

        features = self.project(z).view(z.shape[0], self.channels[0], self.start_length)
        signal = self.output_activation(self.deconv(features))
        return _fit_length(signal, self.length)

    def __repr__(self) -> str:
        return (
            f"Conv1dDecoder(latent_dim={self.latent_dim}, out_channels={self.out_channels}, "
            f"length={self.length}, channels={list(self.channels)})"
        )


def _fit_length(signal: Tensor, length: int) -> Tensor:
    """Crop or zero-pad the last axis to exactly ``length``.

    A transposed stack produces a multiple of its stride, which need not be the target.
    Adjusting here rather than choosing ``output_padding`` per layer keeps the decoder
    usable for any length, at the cost of a seam at the end for lengths that are not a
    multiple of the total stride.
    """
    produced = signal.shape[-1]
    if produced == length:
        return signal
    if produced > length:
        return signal[..., :length]
    padding = signal.new_zeros((*signal.shape[:-1], length - produced))
    return torch.cat([signal, padding], dim=-1)
