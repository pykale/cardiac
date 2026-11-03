"""Modality encoders, the models that fuse them, and the predictor that scores them.

Two families, and the bridge between them:

.. code-block:: text

    encoders.py            backbones adapted to a contract, and the frozen-encoder bridge
    multimodal_vae.py      the generic multimodal VAE: encode, fuse, sample, reconstruct
    cardiovae.py           CardioVAE   -- signal + image, product of experts
    lsemvae.py             LS-EMVAE    -- one encoder per lead, hierarchical fusion, shared decoder
    multimodal_fusion.py   the predictor: embed, fuse, predict

The two named models are *configurations* of :class:`MultimodalVAE`, not separate
architectures, which is what makes an ablation a configuration change. Read
:mod:`multimodal_vae` first; the named classes are short, and their docstrings say what
their paper contributed and where this implementation departs from it.

The bridge is :class:`LatentEmbedder`. A VAE learns one encoder per modality; wrapping
each in it gives a set of embedders that :class:`MultimodalPredictor` fuses and puts a
clinical head on. Pretraining and fine-tuning therefore share the same encoders and
differ only in the task handed to the trainer.
"""

from typing import TYPE_CHECKING

from kalecardiac.model.embed.cardiovae import IMAGE, SIGNAL, CardioVAE, remap_pykale_state_dict
from kalecardiac.model.embed.encoders import (
    SIGNAL_ENCODERS,
    FeatureEmbedder,
    LatentEmbedder,
    VariationalEncoder,
    build_image_encoder,
    build_signal_encoder,
)
from kalecardiac.model.embed.lsemvae import LSEMVAE
from kalecardiac.model.embed.multimodal_fusion import (
    FUSION_METHODS,
    AttentionFusion,
    ConcatFusion,
    FusionBlock,
    MeanFusion,
    MultimodalPredictor,
    PredictionOutput,
    build_fusion,
    modality_dropout,
)
from kalecardiac.model.embed.multimodal_vae import MultimodalVAE, VAEOutput, VAEStream
from kalecardiac.utils.renames import renamed

__all__ = [
    "AttentionFusion",
    "CardioVAE",
    "ConcatFusion",
    "FUSION_METHODS",
    "FeatureEmbedder",
    "FusionBlock",
    "IMAGE",
    "LSEMVAE",
    "LatentEmbedder",
    "MeanFusion",
    "MultimodalPredictor",
    "MultimodalVAE",
    "PredictionOutput",
    "SIGNAL",
    "SIGNAL_ENCODERS",
    "VAEOutput",
    "VAEStream",
    "VariationalEncoder",
    "build_fusion",
    "build_image_encoder",
    "build_signal_encoder",
    "modality_dropout",
    "remap_pykale_state_dict",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
