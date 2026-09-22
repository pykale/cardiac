"""A residual convolutional encoder for one-dimensional signals.

The third backbone family, beside the plain convolutional stack in
:mod:`~kalecardiac.model.layers.conv1d` and the transformer in
:mod:`~kalecardiac.model.layers.transformer`. It exists because a plain three-layer
stack is shallow for a ten-second recording at 500 Hz: it sees a receptive field of a
few tens of milliseconds, which covers a QRS complex but not the interval between two.
Residual blocks make a deeper stack trainable, and a deeper stack reaches across beats.

It is a clean implementation on ``torch`` primitives rather than a copy of the
widely-vendored ``ResNet1D``. That code has its own provenance and licence and is
available from its author; reproducing it here would add a dependency on someone else's
research code to a library, which is exactly what this package exists to avoid.

Every backbone here satisfies the same contract -- ``out_dim``, and
``(batch, channels, length) -> (batch, out_dim)`` -- so
:class:`~kalecardiac.model.embed.VariationalEncoder` turns any of them into a
variational encoder, and switching between them is a configuration change.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ResidualBlock1d(nn.Module):
    """Two convolutions with a skip connection.

    The skip is projected by a 1x1 convolution when the block changes width or length,
    which is what lets a block downsample without breaking the identity path.

    Args:
        in_channels: Channels entering the block.
        out_channels: Channels leaving it.
        kernel_size: Kernel width of both convolutions. Odd, so that ``padding`` keeps
            the length unchanged within the block.
        stride: Stride of the first convolution; 2 halves the length.
        dropout: Dropout between the two convolutions.

    Raises:
        ValueError: If a width is not positive, or ``kernel_size`` is even.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(f"channel widths must be positive, got {in_channels} and {out_channels}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd so the block preserves length, got {kernel_size}")

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.project = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(batch, in_channels, length)`` into ``(batch, out_channels, length // stride)``."""
        residual = self.project(x)
        out = self.dropout(self.activation(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class ResidualConv1dEncoder(nn.Module):
    """Encode a signal with a stem and a stack of residual blocks.

    Ends in global average pooling rather than a flatten, so ``out_dim`` is the last
    block's width and does not depend on the input length. That is the practical
    difference from :class:`~kalecardiac.model.layers.Conv1dEncoder`: the same encoder
    accepts a five-second and a ten-second recording, at the cost of discarding where
    in the recording a feature occurred.

    Args:
        in_channels: Channels the signal has.
        channels: Width of each residual stage.
        kernel_size: Kernel width inside each block.
        stem_channels: Width of the initial convolution, before the first block.
        stride: Stride of each stage after the first; the first keeps the length so the
            stem's output is not halved twice.
        dropout: Dropout inside each block.

    Raises:
        ValueError: If a width is not positive, or no stage is given.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 7,
        stem_channels: int = 32,
        stride: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or stem_channels < 1 or not channels:
            raise ValueError(
                f"need positive in_channels and stem_channels and at least one stage, "
                f"got {in_channels}, {stem_channels}, {list(channels)}"
            )

        self.in_channels = int(in_channels)
        self.channels = tuple(int(width) for width in channels)

        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, int(stem_channels), kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(int(stem_channels)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        blocks: list[nn.Module] = []
        width = int(stem_channels)
        for index, stage in enumerate(self.channels):
            blocks.append(
                ResidualBlock1d(
                    width, stage, kernel_size=kernel_size, stride=1 if index == 0 else stride, dropout=dropout
                )
            )
            width = stage
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = self.channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Encode a signal.

        Args:
            x: ``(batch, in_channels, length)`` signal.

        Returns:
            ``(batch, out_dim)`` features.

        Raises:
            ValueError: If ``x`` is not 3-D with the declared channels.
        """
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (batch, {self.in_channels}, length) signal, got {tuple(x.shape)}")
        return self.pool(self.blocks(self.stem(x))).squeeze(-1)

    def __repr__(self) -> str:
        return f"ResidualConv1dEncoder(in_channels={self.in_channels}, channels={list(self.channels)}, out_dim={self.out_dim})"
