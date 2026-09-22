"""Modality encoders, the models that fuse them, and the predictor that scores them.

Two families, and the bridge between them:

.. code-block:: text

    encoders.py            backbones adapted to a contract, and the freeze-and-embed bridge
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

from kalecardiac.model.embed.cardiovae import IMAGE, SIGNAL, CardioVAE
from kalecardiac.model.embed.encoders import (
    SIGNAL_ENCODERS,
    FeatureEmbedder,
    LatentEmbedder,
    VariationalEncoder,
    build_image_encoder,
    build_signal_encoder,
    freeze,
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

__all__ = [
    "FUSION_METHODS",
    "IMAGE",
    "SIGNAL",
    "SIGNAL_ENCODERS",
    "AttentionFusion",
    "CardioVAE",
    "ConcatFusion",
    "FeatureEmbedder",
    "FusionBlock",
    "LSEMVAE",
    "LatentEmbedder",
    "MeanFusion",
    "MultimodalPredictor",
    "MultimodalVAE",
    "PredictionOutput",
    "VAEOutput",
    "VAEStream",
    "VariationalEncoder",
    "build_fusion",
    "build_image_encoder",
    "build_signal_encoder",
    "freeze",
    "modality_dropout",
]


def __dir__():
    return sorted(__all__)
