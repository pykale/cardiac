"""Loading data, by modality, plus the splitting that keeps it honest.

Three access APIs, none of which knows a dataset:

* :mod:`~kalecardiac.loaddata.signal_access` -- electrocardiograms, by named lead, from
  memory or from one file per subject, plus the lead-splitting that turns a recording
  into a modality per lead.
* :mod:`~kalecardiac.loaddata.image_access` -- cardiovascular images: a radiograph, a
  cardiac MRI slice, an echocardiography frame.
* :mod:`~kalecardiac.loaddata.multimodal_access` -- any combination of those, or of
  anything else, joined by subject identifier.

and :mod:`~kalecardiac.loaddata.splitting`, whose splitter objects follow
scikit-learn's shape.

**Dataset-specific preparation does not live here.** Which column defines an endpoint,
how a filename encodes its subject, which subjects a study assigned to its validation
cohort -- all of that belongs to the dataset that defines it, in ``examples/``. This
package supplies the mechanisms those decisions are expressed with.

``release_workers`` is exported for the same reason it exists: a cross-validated run
builds loaders per fold, and their worker pools must be closed deliberately rather
than raced against interpreter shutdown.
"""

from typing import TYPE_CHECKING

from kalecardiac.loaddata.image_access import ImageArraySource, ImageFileSource, ImageFormatError, check_image_array
from kalecardiac.loaddata.multimodal_access import (
    LABEL_KEY,
    VALUE_KEY,
    ArraySource,
    ColumnTarget,
    ModalitySource,
    MultimodalDataset,
    SubjectBatch,
    SubjectSample,
    Target,
    check_target,
    collate_subjects,
    release_workers,
)
from kalecardiac.loaddata.signal_access import (
    LIMB_LEADS,
    PRECORDIAL_LEADS,
    STANDARD_12_LEAD,
    ECGArraySource,
    ECGFileSource,
    ECGFormatError,
    LeadSource,
    check_ecg_array,
    get_lead_indices,
    get_lead_sources,
    normalize_lead_name,
    normalize_lead_names,
)
from kalecardiac.loaddata.splitting import (
    SPLIT_NAMES,
    BaseSplitter,
    CrossValidationSplit,
    HoldOutSplit,
    PredefinedSplit,
    Split,
    SplitError,
    get_stratification_labels,
)
from kalecardiac.utils.renames import renamed

__all__ = [
    "ArraySource",
    "BaseSplitter",
    "ColumnTarget",
    "CrossValidationSplit",
    "ECGArraySource",
    "ECGFileSource",
    "ECGFormatError",
    "HoldOutSplit",
    "ImageArraySource",
    "ImageFileSource",
    "ImageFormatError",
    "LABEL_KEY",
    "LIMB_LEADS",
    "LeadSource",
    "ModalitySource",
    "MultimodalDataset",
    "PRECORDIAL_LEADS",
    "PredefinedSplit",
    "SPLIT_NAMES",
    "STANDARD_12_LEAD",
    "Split",
    "SplitError",
    "SubjectBatch",
    "SubjectSample",
    "Target",
    "VALUE_KEY",
    "check_ecg_array",
    "check_image_array",
    "check_target",
    "collate_subjects",
    "get_lead_indices",
    "get_lead_sources",
    "get_stratification_labels",
    "normalize_lead_name",
    "normalize_lead_names",
    "release_workers",
]


def __dir__():
    return sorted(__all__)


if not TYPE_CHECKING:
    # At runtime only: a module-level __getattr__ that a type checker saw would make
    # every misspelt import type-check.
    def __getattr__(name: str):
        raise renamed(__name__, name)
