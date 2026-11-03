"""Transforms, by the kind of data they transform.

.. code-block:: text

    signal_transform.py  resampling, cropping, amplitude normalisation -- any waveform
    ecg_transform.py     the part that needs to know a lead from a channel
    image_transform.py   resizing and intensity normalisation -- any 2-D modality

**Nothing here is fitted.** Every transform is a function of one recording or one
image, so a source may apply it without knowing which split the subject landed in.
That is what keeps preprocessing out of the leakage discussion entirely: there is no
state to fit on the wrong rows. A quantity estimated *across* subjects would be
fold-local state and would belong with the fold, not here.

Splitting is in :mod:`kalecardiac.loaddata`, not here, for the mirror-image reason: a
splitter decides which subjects load into which loader, and the thing that creates the
folds cannot be state belonging to one. scikit-learn draws the same line between
``model_selection`` and ``preprocessing``.
"""

from typing import TYPE_CHECKING

from kalecardiac.prepdata.ecg_transform import (
    DERIVABLE_LIMB_LEADS,
    derive_limb_leads,
    get_ecg_transform,
    get_lead_selector,
    interleave_leads,
    select_leads,
)
from kalecardiac.prepdata.image_transform import (
    check_image,
    get_image_transform,
    normalize_image,
    resize_image,
    scale_image_intensity,
)
from kalecardiac.prepdata.signal_transform import (
    MIN_SCALE,
    check_signal,
    clip_signal,
    crop_or_pad_signal,
    interpolate_signal,
    normalize_signal,
    resample_signal,
    scale_signal,
)
from kalecardiac.utils.renames import renamed

__all__ = [
    "DERIVABLE_LIMB_LEADS",
    "MIN_SCALE",
    "check_image",
    "check_signal",
    "clip_signal",
    "crop_or_pad_signal",
    "derive_limb_leads",
    "get_ecg_transform",
    "get_image_transform",
    "get_lead_selector",
    "interleave_leads",
    "interpolate_signal",
    "normalize_image",
    "normalize_signal",
    "resample_signal",
    "resize_image",
    "scale_image_intensity",
    "scale_signal",
    "select_leads",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
