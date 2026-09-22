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

from kalecardiac.prepdata.ecg_transform import (
    DERIVABLE_LIMB_LEADS,
    build_ecg_pipeline,
    derive_limb_leads,
    lead_selector,
    select_leads,
)
from kalecardiac.prepdata.image_transform import (
    ImagePipeline,
    as_image,
    build_image_pipeline,
    resize_image,
    scale_image,
    standardise_image,
)
from kalecardiac.prepdata.signal_transform import (
    MIN_SCALE,
    SignalPipeline,
    as_signal,
    clip_amplitude,
    crop_or_pad,
    interpolate_missing,
    resample,
    scale_amplitude,
    standardise,
    step_name,
)

__all__ = [
    "DERIVABLE_LIMB_LEADS",
    "MIN_SCALE",
    "ImagePipeline",
    "SignalPipeline",
    "as_image",
    "as_signal",
    "build_ecg_pipeline",
    "build_image_pipeline",
    "clip_amplitude",
    "crop_or_pad",
    "derive_limb_leads",
    "interpolate_missing",
    "lead_selector",
    "resample",
    "resize_image",
    "scale_amplitude",
    "scale_image",
    "select_leads",
    "standardise",
    "standardise_image",
    "step_name",
]


def __dir__():
    return sorted(__all__)
