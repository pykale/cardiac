"""The model stage, in three parts.

``layers`` holds blocks that transform tensors, ``embed`` adapts them to the contracts
the models rely on and combines several modalities into one representation, and
``predict`` turns that representation into a score with the objective that scores it.

A class belongs to exactly one: a block that knows no modality is a layer, a thing with
``out_dim`` or ``latent_dim`` is an embedder, a thing emitting one score per subject is
a head.

The two published cardiac models are exported from here because that is where a reader
will look for them, but they are configurations of :class:`MultimodalVAE` rather than
architectures of their own -- see :mod:`kalecardiac.model.embed`.
"""

from kalecardiac.model.embed import (
    FUSION_METHODS,
    LSEMVAE,
    SIGNAL_ENCODERS,
    CardioVAE,
    FeatureEmbedder,
    LatentEmbedder,
    MultimodalPredictor,
    MultimodalVAE,
    PredictionOutput,
    VAEOutput,
    VariationalEncoder,
)
from kalecardiac.model.layers import (
    EXPERT_FUSIONS,
    Conv1dDecoder,
    Conv1dEncoder,
    Conv2dDecoder,
    Conv2dEncoder,
    GaussianPosterior,
    ResidualConv1dEncoder,
    TransformerSignalEncoder,
)
from kalecardiac.model.predict import LinearHead, MLPHead

__all__ = [
    "EXPERT_FUSIONS",
    "FUSION_METHODS",
    "SIGNAL_ENCODERS",
    "CardioVAE",
    "Conv1dDecoder",
    "Conv1dEncoder",
    "Conv2dDecoder",
    "Conv2dEncoder",
    "FeatureEmbedder",
    "GaussianPosterior",
    "LSEMVAE",
    "LatentEmbedder",
    "LinearHead",
    "MLPHead",
    "MultimodalPredictor",
    "MultimodalVAE",
    "PredictionOutput",
    "ResidualConv1dEncoder",
    "TransformerSignalEncoder",
    "VAEOutput",
    "VariationalEncoder",
]


def __dir__():
    return sorted(__all__)
