"""Task heads and the objectives they are trained with.

A head turns a fixed-width representation into a score; a loss says whether that score
was right, or -- for a generative objective -- whether a reconstruction was. They live
together because neither is meaningful alone, and because which pair a model gets is
decided by its :class:`~kalecardiac.pipeline.task.PredictionTask`: a new endpoint means
a new task, not a new trainer.

The generative objectives are here for the same reason. An ELBO is what trains a
multimodal VAE, and a multimodal VAE's "head" is its decoders, so putting the ELBO
here keeps the objectives this package defines in one place; the classification and
regression losses are PyTorch's, and a task calls them directly.
"""

from typing import TYPE_CHECKING

from kalecardiac.model.predict.heads import LinearHead, MLPHead
from kalecardiac.model.predict.losses import (
    LOG_VAR_LIMIT,
    elbo_loss,
    gaussian_kl_divergence,
    latent_alignment_loss,
    reconstruction_loss,
)
from kalecardiac.utils.renames import renamed

__all__ = [
    "LOG_VAR_LIMIT",
    "LinearHead",
    "MLPHead",
    "elbo_loss",
    "gaussian_kl_divergence",
    "latent_alignment_loss",
    "reconstruction_loss",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
