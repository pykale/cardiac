"""Synthetic fixtures, so tests never need the private cardiac cohorts.

Everything here comes from :mod:`examples.synthetic_data`. The cohorts both supported
use cases were developed on are private clinical data and cannot be redistributed, so
the test suite is built to need no data, no credentials and no network -- and to fail
loudly if anything reaches for them.

The cohorts are deliberately small and short: a few dozen subjects of a few hundred
samples, which exercises every code path while keeping the whole suite to a couple of
minutes on a CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.loaddata import (
    ColumnTarget,
    ECGArraySource,
    ImageArraySource,
    MultimodalDataset,
    collate_subjects,
    lead_sources,
)

#: Kept small enough that a five-fold split still leaves both classes in every fold.
NUM_SUBJECTS = 40

#: Short recordings: 256 samples is divisible by eight, so a three-layer stride-2
#: encoder and its decoder round-trip without a cropping seam.
NUM_SAMPLES = 256

#: Small images, for the same reason.
IMAGE_SIZE = (32, 32)


@pytest.fixture(scope="session")
def cohort():
    """A paired ECG-and-image cohort with a learnable endpoint."""
    return make_synthetic_multimodal_data(
        num_subjects=NUM_SUBJECTS,
        num_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
        seed=0,
    )


@pytest.fixture
def ecg_source(cohort):
    """Every lead of every subject, as one recording source."""
    return ECGArraySource(cohort.ecg, leads=cohort.leads, sampling_rate=cohort.sampling_rate)


@pytest.fixture
def image_source(cohort):
    """The image modality."""
    return ImageArraySource(cohort.image, channels=1, size=IMAGE_SIZE)


@pytest.fixture
def label_target(cohort):
    """The binary endpoint."""
    return ColumnTarget.from_mapping({"label": cohort.labels})


@pytest.fixture
def lead_dataset(cohort, ecg_source, label_target):
    """One modality per lead, which is what a lead-specific model consumes."""
    return MultimodalDataset(cohort.subject_id, lead_sources(ecg_source), target=label_target)


@pytest.fixture
def multimodal_dataset(cohort, ecg_source, image_source, label_target):
    """The whole recording and the image, as two modalities."""
    return MultimodalDataset(cohort.subject_id, {"ecg": ecg_source, "image": image_source}, target=label_target)


@pytest.fixture
def loader_factory():
    """Build a deterministic DataLoader over any dataset.

    ``num_workers=0`` throughout: the suite must not depend on process spawning, which
    behaves differently on each platform and is not what any of these tests is about.
    """

    def build(dataset, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_subjects)

    return build


@pytest.fixture
def experts():
    """A stack of Gaussian experts: ``(mean, log_var)`` of shape ``(5, 4, 3)``."""
    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(5, 4, 3, generator=generator)
    log_var = torch.randn(5, 4, 3, generator=generator) * 0.5
    return mean, log_var


@pytest.fixture
def binary_scores():
    """Labels and scores with a real but imperfect ranking, for metric tests."""
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, size=60)
    scores = generator.normal(size=60) + labels
    return labels, scores


@pytest.fixture(autouse=True)
def deterministic():
    """Seed every test, so a failure is reproducible rather than intermittent."""
    torch.manual_seed(2026)
    np.random.seed(2026)
