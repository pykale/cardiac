"""Post-prediction interpretation: what the model used, and what it learned.

.. code-block:: text

    attribution.py  gradient attribution, lead and wave importance, modality ablation
    latent.py       collecting and projecting a learned latent space

Two families, answering questions at different levels. Attribution and ablation ask
what a *trained predictor* used; the latent module asks what a *pretrained
representation* contains, which is the only question available before any labels exist.

**Mechanisms belong here; figures do not.** Everything returns arrays and dictionaries.
A publication figure names a cohort, a colour scheme and a claim, so it is built in
``examples/`` from these numbers.
"""

from kalecardiac.interpret.attribution import (
    attribution_ratio,
    attribution_ratios,
    ecg_wave_segments,
    modality_ablation,
    modality_attributions,
    normalise_attribution,
    segment_attribution_ratios,
)
from kalecardiac.interpret.latent import collect_latents, latent_embedding

__all__ = [
    "attribution_ratio",
    "attribution_ratios",
    "collect_latents",
    "ecg_wave_segments",
    "latent_embedding",
    "modality_ablation",
    "modality_attributions",
    "normalise_attribution",
    "segment_attribution_ratios",
]


def __dir__():
    return sorted(__all__)
