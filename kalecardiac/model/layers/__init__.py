"""Shared neural network building blocks.

Pieces small enough to be reused by more than one model, and independent enough to be
tested with ``torch.randn``. A block here transforms tensors and knows nothing about
modalities, leads or cohorts -- which is what separates it from
:mod:`kalecardiac.model.embed`, where a block is adapted to a contract.

.. code-block:: text

    gaussian.py     combining Gaussian experts, and the sampling that follows
    conv1d.py       convolutional encoder and decoder for signals
    conv2d.py       convolutional encoder and decoder for images
    residual1d.py   a deeper residual encoder for signals
    transformer.py  an attention encoder over signal patches
    mlp.py          a feed-forward stack

:mod:`gaussian` is the one to read first: every multimodal VAE in this package differs
from the others only in which function there it fuses its experts with.
"""

from kalecardiac.model.layers.conv1d import Conv1dDecoder, Conv1dEncoder, conv_output_length
from kalecardiac.model.layers.conv2d import Conv2dDecoder, Conv2dEncoder, conv_output_size
from kalecardiac.model.layers.gaussian import (
    ABSENT_LOG_VAR,
    EXPERT_FUSIONS,
    GaussianHead,
    GaussianPosterior,
    build_expert_fusion,
    expert_groups,
    hierarchical_experts,
    mean_of_experts,
    mixture_of_experts,
    prior_expert,
    product_of_experts,
    reparameterise,
    wasserstein_barycenter,
)
from kalecardiac.model.layers.mlp import MLP
from kalecardiac.model.layers.residual1d import ResidualBlock1d, ResidualConv1dEncoder
from kalecardiac.model.layers.transformer import SinusoidalPositionalEncoding, TransformerSignalEncoder

__all__ = [
    "ABSENT_LOG_VAR",
    "EXPERT_FUSIONS",
    "MLP",
    "Conv1dDecoder",
    "Conv1dEncoder",
    "Conv2dDecoder",
    "Conv2dEncoder",
    "GaussianHead",
    "GaussianPosterior",
    "ResidualBlock1d",
    "ResidualConv1dEncoder",
    "SinusoidalPositionalEncoding",
    "TransformerSignalEncoder",
    "build_expert_fusion",
    "conv_output_length",
    "conv_output_size",
    "expert_groups",
    "hierarchical_experts",
    "mean_of_experts",
    "mixture_of_experts",
    "prior_expert",
    "product_of_experts",
    "reparameterise",
    "wasserstein_barycenter",
]


def __dir__():
    return sorted(__all__)
