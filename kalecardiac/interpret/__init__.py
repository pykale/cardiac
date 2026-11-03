"""Post-prediction interpretation: what the model used, and what it learned.

.. code-block:: text

    attribution.py  gradient attribution, lead and wave importance, modality ablation
                    one at a time and cumulatively
    latent.py       collecting and projecting a learned latent space

Two families, answering questions at different levels. Attribution and ablation ask
what a *trained predictor* used; the latent module asks what a *pretrained
representation* contains, which is the only question available before any labels exist.

**Mechanisms belong here; figures do not.** Everything returns arrays and dictionaries.
A publication figure names a cohort, a colour scheme and a claim, so it is built in
``examples/`` from these numbers.
"""

from typing import TYPE_CHECKING

from kalecardiac.interpret.attribution import (
    attribution_ratio,
    attribution_ratios,
    delineate_ecg_waves,
    modality_ablation,
    modality_attributions,
    normalize_attribution,
    progressive_ablation,
    segment_attribution_ratios,
)
from kalecardiac.interpret.latent import collect_latents, project_latents
from kalecardiac.utils.renames import renamed

__all__ = [
    "attribution_ratio",
    "attribution_ratios",
    "collect_latents",
    "delineate_ecg_waves",
    "project_latents",
    "modality_ablation",
    "modality_attributions",
    "normalize_attribution",
    "progressive_ablation",
    "segment_attribution_ratios",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
