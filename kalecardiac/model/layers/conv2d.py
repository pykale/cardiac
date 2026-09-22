"""Convolutional blocks over two-dimensional images.

The image counterpart of :mod:`kalecardiac.model.layers.conv1d`, and the same argument
applies: the spatial size is an argument and the flattened width is measured, so a
cohort that resized its radiographs to 256 pixels, or a cardiac MRI slice that is not
square, works without editing a linear layer.

PyKale's :class:`kale.embed.image_cnn.ImageVAEEncoder` is the fixed-size version of
this -- three stride-2 convolutions of widths 16, 32 and 64 over a 224-pixel square --
and a :class:`Conv2dEncoder` built with the defaults and ``size=(224, 224)`` is the
same architecture. It is reimplemented rather than wrapped because its flattened width
is the literal ``64 * 28 * 28``, which is wrong for every other input size.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from kale.embed.base_cnn import BaseCNN
from torch import Tensor, nn


def conv_output_size(size: Sequence[int], layers: int, kernel_size: int, stride: int, padding: int) -> tuple[int, int]:
    """Spatial size after a stack of convolutions, by the standard formula.

    Args:
        size: Input ``(height, width)``.
        layers: How many convolutions.
        kernel_size: Kernel extent of each.
        stride: Stride of each.
        padding: Padding of each.

    Returns:
        The output ``(height, width)``.

    Raises:
        ValueError: If the image is reduced to nothing.
    """
    height, width = int(size[0]), int(size[1])
    for _ in range(layers):
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1
        if height < 1 or width < 1:
            raise ValueError(
                f"a stack of {layers} convolutions with kernel {kernel_size} and stride {stride} reduces an "
                f"image of {tuple(size)} to nothing; use fewer layers, a smaller stride, or a larger input"
            )
    return height, width


class Conv2dEncoder(BaseCNN):
    """Encode a ``(batch, channels, height, width)`` image into a flat feature vector.

    Args:
        in_channels: Channels the image has. One for a grayscale radiograph.
        size: Input ``(height, width)``. Fixed, because the encoder ends in a flatten.
        channels: Output channels of each convolution, one entry per layer.
        kernel_size: Kernel extent of every convolution.
        stride: Stride of every convolution.
        padding: Padding of every convolution.
        batch_norm: Insert batch normalisation after each convolution.
        activation: Non-linearity after each convolution, by PyKale's name.

    Raises:
        ValueError: If a width is not positive, or the stack reduces the image to
            nothing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        size: Sequence[int] = (224, 224),
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_channels < 1 or not channels or len(size) != 2:
            raise ValueError(
                f"need positive in_channels, a (height, width) size and at least one layer, "
                f"got {in_channels}, {tuple(size)}, {list(channels)}"
            )

        self.in_channels = int(in_channels)
        self.size = (int(size[0]), int(size[1]))
        self.channels = tuple(int(width) for width in channels)
        self.activation = activation

        self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
            in_channels=self.in_channels,
            out_channels_size_list=list(self.channels),
            kernel_sizes=int(kernel_size),
            conv_type="2d",
            strides=int(stride),
            conv_padding=int(padding),
            use_batch_norm=batch_norm,
            bias=not batch_norm,
        )
        self.use_batch_norm = batch_norm
        self.output_size = conv_output_size(self.size, len(self.channels), kernel_size, stride, padding)
        self.out_dim = self.channels[-1] * self.output_size[0] * self.output_size[1]

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image.

        Args:
            x: ``(batch, in_channels, height, width)`` image.

        Returns:
            ``(batch, out_dim)`` features.

        Raises:
            ValueError: If ``x`` does not have the declared channels and size.
        """
        if x.dim() != 4 or x.shape[1] != self.in_channels or tuple(x.shape[2:]) != self.size:
            raise ValueError(
                f"expected (batch, {self.in_channels}, {self.size[0]}, {self.size[1]}) image, got {tuple(x.shape)}"
            )
        for conv, norm in zip(self.conv_layers, self.batch_norms, strict=True):
            x = self._apply_activation(norm(conv(x)) if self.use_batch_norm else conv(x), self.activation)
        return self._flatten_features(x)

    def __repr__(self) -> str:
        return (
            f"Conv2dEncoder(in_channels={self.in_channels}, size={self.size}, "
            f"channels={list(self.channels)}, out_dim={self.out_dim})"
        )


class Conv2dDecoder(nn.Module):
    """Reconstruct a ``(batch, out_channels, height, width)`` image from a latent vector.

    The image counterpart of :class:`~kalecardiac.model.layers.Conv1dDecoder`, and the
    same cropping applies where the target size is not a multiple of the total stride.

    ``output_activation="sigmoid"`` is the usual choice here, because an image
    preprocessed to the unit interval is exactly what a sigmoid can produce; a
    standardised image needs ``None`` instead. The two must agree with how the image
    source normalised, or the reconstruction term is minimised by a constant.

    Args:
        latent_dim: Width of the latent vector.
        out_channels: Channels to reconstruct.
        size: ``(height, width)`` to reconstruct.
        channels: Channels of each transposed convolution, from the widest inwards.
        kernel_size: Kernel extent.
        stride: Stride of each transposed convolution.
        activation: Non-linearity between layers.
        output_activation: Applied to the reconstruction; ``"sigmoid"``, ``"tanh"`` or
            ``None``.

    Raises:
        ValueError: If a width is not positive, or ``output_activation`` is unknown.
    """

    _OUTPUT_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 1,
        size: Sequence[int] = (224, 224),
        channels: Sequence[int] = (64, 32, 16),
        kernel_size: int = 4,
        stride: int = 2,
        activation: str = "relu",
        output_activation: str | None = "sigmoid",
    ) -> None:
        super().__init__()
        if latent_dim < 1 or out_channels < 1 or not channels or len(size) != 2:
            raise ValueError(
                f"need positive latent_dim and out_channels, a (height, width) size and at least one layer, "
                f"got {latent_dim}, {out_channels}, {tuple(size)}, {list(channels)}"
            )

        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.size = (int(size[0]), int(size[1]))
        self.channels = tuple(int(width) for width in channels)

        upsample = stride ** len(self.channels)
        self.start_size = (max(self.size[0] // upsample, 1), max(self.size[1] // upsample, 1))
        self.project = nn.Linear(self.latent_dim, self.channels[0] * self.start_size[0] * self.start_size[1])

        padding = (kernel_size - stride) // 2
        output_padding = stride - kernel_size + 2 * padding
        layers: list[nn.Module] = []
        widths = [*self.channels, self.out_channels]
        for index in range(len(widths) - 1):
            layers.append(
                nn.ConvTranspose2d(
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
        """Reconstruct an image.

        Args:
            z: ``(batch, latent_dim)`` latent vector.

        Returns:
            ``(batch, out_channels, height, width)`` reconstruction.

        Raises:
            ValueError: If ``z`` is not ``(batch, latent_dim)``.
        """
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected (batch, {self.latent_dim}) latent, got {tuple(z.shape)}")

        features = self.project(z).view(z.shape[0], self.channels[0], *self.start_size)
        image = self.output_activation(self.deconv(features))
        return _fit_size(image, self.size)

    def __repr__(self) -> str:
        return (
            f"Conv2dDecoder(latent_dim={self.latent_dim}, out_channels={self.out_channels}, "
            f"size={self.size}, channels={list(self.channels)})"
        )


def _fit_size(image: Tensor, size: tuple[int, int]) -> Tensor:
    """Crop or zero-pad the two trailing axes to exactly ``size``."""
    height, width = image.shape[-2], image.shape[-1]
    if (height, width) == size:
        return image
    image = image[..., : size[0], : size[1]]
    pad_height, pad_width = size[0] - image.shape[-2], size[1] - image.shape[-1]
    if pad_height or pad_width:
        image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height))
    return image
