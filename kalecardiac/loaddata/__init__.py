"""Loading data, by modality, plus the splitting that keeps it honest.

Three access APIs, none of which knows a dataset:

* :mod:`~kalecardiac.loaddata.ecg_access` -- electrocardiograms, by named lead, from
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

from kalecardiac.loaddata.ecg_access import (
    LIMB_LEADS,
    PRECORDIAL_LEADS,
    STANDARD_12_LEAD,
    ECGArraySource,
    ECGFileSource,
    ECGFormatError,
    LeadSource,
    as_lead_array,
    canonical_lead,
    canonical_leads,
    lead_indices,
    lead_sources,
)
from kalecardiac.loaddata.image_access import ImageArraySource, ImageFileSource, ImageFormatError, as_image_array
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
from kalecardiac.loaddata.splitting import (
    SPLIT_NAMES,
    CohortSplitter,
    CrossValidation,
    HoldOut,
    Predefined,
    Split,
    SplitError,
    composite_labels,
    subject_frame,
    train_test_split,
)

__all__ = [
    "ArraySource",
    "CohortSplitter",
    "ColumnTarget",
    "CrossValidation",
    "ECGArraySource",
    "ECGFileSource",
    "ECGFormatError",
    "HoldOut",
    "ImageArraySource",
    "ImageFileSource",
    "ImageFormatError",
    "LABEL_KEY",
    "LIMB_LEADS",
    "LeadSource",
    "ModalitySource",
    "MultimodalDataset",
    "PRECORDIAL_LEADS",
    "Predefined",
    "SPLIT_NAMES",
    "STANDARD_12_LEAD",
    "Split",
    "SplitError",
    "SubjectBatch",
    "SubjectSample",
    "Target",
    "VALUE_KEY",
    "as_image_array",
    "as_lead_array",
    "canonical_lead",
    "canonical_leads",
    "check_target",
    "collate_subjects",
    "composite_labels",
    "lead_indices",
    "lead_sources",
    "release_workers",
    "subject_frame",
    "train_test_split",
]


def __dir__():
    return sorted(__all__)
