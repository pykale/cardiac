"""A transformer encoder over one-dimensional signals.

The attention-based backbone, refactored from the vanilla-transformer baseline of the
LS-EMVAE study. A raw cardiac recording is far too long to attend over sample by
sample -- ten seconds at 500 Hz is five thousand tokens, and attention is quadratic --
so the signal is first divided into fixed-length patches and each patch is projected to
a token. That is the standard treatment for long physiological signals, and it is what
makes the sequence length a hyperparameter rather than a property of the sampling rate.

Like the other backbones here it exposes ``out_dim`` and maps
``(batch, channels, length) -> (batch, out_dim)``, so
:class:`~kalecardiac.model.embed.VariationalEncoder` turns it into a variational
encoder and ``ECG.ENCODER: transformer`` selects it.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position information to a sequence of tokens.

    Fixed rather than learned: a cardiac recording has no canonical starting phase, so
    there is nothing for a learned table to memorise about position ten that would not
    also be true of position eleven, and a fixed encoding extrapolates to a longer
    recording than it was trained on.

    Args:
        d_model: Token width.
        max_len: Longest sequence the table covers.

    Raises:
        ValueError: If ``d_model`` is not a positive even number.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model < 2 or d_model % 2:
            raise ValueError(f"d_model must be a positive even number, got {d_model}")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        # A buffer, not a parameter: it is a constant that must follow the module onto
        # its device, and persistent so a checkpoint round-trips without recomputing it.
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, tokens: Tensor) -> Tensor:
        """Add positions to ``(batch, length, d_model)`` tokens.

        Raises:
            ValueError: If the sequence is longer than the table.
        """
        length = tokens.shape[1]
        if length > self.encoding.shape[1]:
            raise ValueError(f"sequence of {length} tokens exceeds max_len={self.encoding.shape[1]}")
        return tokens + self.encoding[:, :length]


class TransformerSignalEncoder(nn.Module):
    """Encode a signal by attending over fixed-length patches of it.

    Args:
        in_channels: Channels the signal has. Every channel of a patch is projected
            together, so a twelve-lead patch becomes one token carrying all twelve.
        patch_size: Samples per token. Larger patches mean fewer, coarser tokens: at
            500 Hz a patch of 50 is 100 ms, about the width of a QRS complex.
        d_model: Token width.
        nhead: Attention heads per layer.
        num_layers: Transformer encoder layers.
        dim_feedforward: Width of each layer's feed-forward block.
        dropout: Dropout inside the transformer.
        max_len: Longest token sequence supported.
        pooling: How tokens become one vector: ``"mean"`` averages them, ``"last"``
            takes the final token.

    Raises:
        ValueError: If a width is not positive, or ``pooling`` is unknown.
    """

    _POOLING = ("mean", "last")

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 50,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if in_channels < 1 or patch_size < 1:
            raise ValueError(f"in_channels and patch_size must be positive, got {in_channels} and {patch_size}")
        if pooling not in self._POOLING:
            raise ValueError(f"unknown pooling {pooling!r}; available: {list(self._POOLING)}")

        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.pooling = pooling

        self.to_tokens = nn.Linear(self.in_channels * self.patch_size, d_model)
        self.positions = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = int(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a signal.

        A recording whose length is not a multiple of ``patch_size`` has its trailing
        remainder dropped, which is at most one patch and avoids a partial token whose
        padded samples the attention would treat as signal.

        Args:
            x: ``(batch, in_channels, length)`` signal.

        Returns:
            ``(batch, out_dim)`` features.

        Raises:
            ValueError: If ``x`` is not 3-D with the declared channels, or is shorter
                than one patch.
        """
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (batch, {self.in_channels}, length) signal, got {tuple(x.shape)}")

        batch, _, length = x.shape
        num_patches = length // self.patch_size
        if num_patches < 1:
            raise ValueError(f"a signal of {length} samples is shorter than one patch of {self.patch_size}")

        patched = x[..., : num_patches * self.patch_size]
        patched = patched.reshape(batch, self.in_channels, num_patches, self.patch_size)
        tokens = patched.permute(0, 2, 1, 3).reshape(batch, num_patches, self.in_channels * self.patch_size)

        encoded = self.transformer(self.positions(self.to_tokens(tokens)))
        pooled = encoded.mean(dim=1) if self.pooling == "mean" else encoded[:, -1]
        return self.norm(pooled)

    def __repr__(self) -> str:
        return (
            f"TransformerSignalEncoder(in_channels={self.in_channels}, patch_size={self.patch_size}, "
            f"out_dim={self.out_dim})"
        )
