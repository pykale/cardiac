"""Prediction heads: a fixed-width representation in, a score per subject out.

A head is the last thing a model does and the only part that knows what is being
predicted, which is why it is chosen by a
:class:`~kalecardiac.pipeline.task.PredictionTask` rather than fixed by a model.

Two heads, and the difference is capacity rather than endpoint. :class:`LinearHead` is
one linear map, which is the right model when the representation was learned by a large
pretrained encoder and the labelled cohort is small -- the usual cardiac situation.
:class:`MLPHead` adds a hidden layer and dropout, which is what both cardiac reference
implementations put on their frozen encoders; with a few hundred labelled subjects the
dropout is doing as much work as the hidden layer.

**On reusing PyKale's head.** ``kale.predict.decode.LinearClassifier`` is the same
layer, and reusing it would be the convention here. It is not imported because
``kale.predict.decode`` pulls in ``GripNet`` and therefore ``torch_geometric``, which
this package does not depend on -- a graph library is a large thing to require for a
single linear layer. The initialisers it applies come from ``kale.utils``, which imports
cleanly, so those are reused instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from kale.utils.initialize_nn import bias_init, xavier_init
from torch import Tensor, nn

from kalecardiac.model.layers.mlp import MLP


class LinearHead(nn.Module):
    """A linear map from a representation to one or more scores.

    Xavier-normal weights and a zeroed bias, matching what PyKale gives its own linear
    heads: a head reads a fused representation whose scale depends on how many
    modalities were concatenated, so setting the gain from the fan-in beats a fixed
    range.

    Emits a raw score, not a probability. For a binary endpoint that is a logit -- the
    fused sigmoid-and-log form of the loss is the numerically stable one, and ROC-AUC
    is unchanged by the sigmoid.

    Args:
        in_features: Width of the representation the head reads.
        out_features: Scores per subject. 1 for a binary endpoint or a regression,
            ``num_classes`` for a multiclass one.

    Raises:
        ValueError: If either width is not positive.
    """

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(f"widths must be positive, got {in_features} and {out_features}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.linear.apply(xavier_init)
        self.linear.apply(bias_init)

    def forward(self, z: Tensor) -> Tensor:
        """Score representations.

        Args:
            z: ``(B, in_features)`` representations.

        Returns:
            ``(B, out_features)`` scores.

        Raises:
            ValueError: If ``z`` is not 2-D or its width does not match.
        """
        if z.dim() != 2 or z.shape[1] != self.in_features:
            raise ValueError(f"expected (batch, {self.in_features}) representation, got {tuple(z.shape)}")
        return self.linear(z)


class MLPHead(nn.Module):
    """A small feed-forward head with dropout.

    What both cardiac reference implementations put on their frozen encoders: one
    hidden layer and heavy dropout, which on a few hundred labelled subjects is as much
    a regulariser as a model.

    Args:
        in_features: Width of the representation the head reads.
        out_features: Scores per subject.
        hidden_dims: Hidden widths. Empty makes this a linear head with dropout in
            front of nothing, so prefer :class:`LinearHead` for that.
        dropout: Dropout after each hidden activation.

    Raises:
        ValueError: If a width is not positive.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.net = MLP(self.in_features, self.out_features, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, z: Tensor) -> Tensor:
        """Score representations.

        Args:
            z: ``(B, in_features)`` representations.

        Returns:
            ``(B, out_features)`` scores.

        Raises:
            ValueError: If ``z`` is not 2-D or its width does not match.
        """
        if z.dim() != 2 or z.shape[1] != self.in_features:
            raise ValueError(f"expected (batch, {self.in_features}) representation, got {tuple(z.shape)}")
        return self.net(z)
