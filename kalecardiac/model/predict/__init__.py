"""Task heads and the objectives they are trained with.

A head turns a fixed-width representation into a score; a loss says whether that score
was right, or -- for a generative objective -- whether a reconstruction was. They live
together because neither is meaningful alone, and because which pair a model gets is
decided by its :class:`~kalecardiac.pipeline.task.PredictionTask`: a new endpoint means
a new task, not a new trainer.

The generative objectives are here for the same reason. An ELBO is what trains a
multimodal VAE, and a multimodal VAE's "head" is its decoders, so putting the ELBO
beside the classification losses keeps every objective in one place.
"""

from kalecardiac.model.predict.heads import LinearHead, MLPHead
from kalecardiac.model.predict.losses import (
    LOG_VAR_LIMIT,
    binary_cross_entropy,
    class_weight_from_labels,
    cross_entropy,
    elbo_loss,
    gaussian_kl_divergence,
    latent_alignment_loss,
    mean_squared_error,
    positive_weight_from_labels,
    reconstruction_loss,
)

__all__ = [
    "LOG_VAR_LIMIT",
    "LinearHead",
    "MLPHead",
    "binary_cross_entropy",
    "class_weight_from_labels",
    "cross_entropy",
    "elbo_loss",
    "gaussian_kl_divergence",
    "latent_alignment_loss",
    "mean_squared_error",
    "positive_weight_from_labels",
    "reconstruction_loss",
]


def __dir__():
    return sorted(__all__)
