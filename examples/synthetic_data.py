"""Synthetic cardiac cohorts, for demonstrations and tests.

Lives in ``examples/`` rather than in ``kalecardiac``: generating data with a known
answer is how you exercise a pipeline, not something the pipeline provides. Nothing in
the library imports it.

It exists because the cohorts both supported use cases were developed on are private
clinical data -- an NHS pulmonary hypertension registry and a biobank -- which cannot
be redistributed. Everything here is drawn from a generator, so the tests and the
quick-start runs need no data, no credentials and no network.

**These are not physiologically accurate ECGs.** The waveforms are a sum of sinusoids
with a beat-like envelope: they have the right shape, the right dimensions and a real
dependence between signal and label, which is what a pipeline test needs. They will not
support a claim about QRS morphology, and :func:`make_synthetic_ecg` is deliberately
crude so that nobody mistakes an output of it for a recording.

The label is generated from the signal, so a model that learns nothing scores about
0.5 and a working pipeline scores clearly above it. That is the point: a smoke test
that passes on noise tests nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

#: Lead names used when a caller does not supply their own. The six limb leads, because
#: that is the reduced configuration a low-cost recorder captures and the one both
#: cardiac use cases fine-tune on.
DEFAULT_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")


class SyntheticCohort(NamedTuple):
    """One synthetic cohort, ready to build sources from.

    Attributes:
        subject_id: Identifiers, in generation order.
        ecg: ``(num_leads, num_samples)`` recording per subject, keyed by identifier.
        image: ``(channels, height, width)`` image per subject. Empty unless asked for.
        labels: Binary label per subject.
        values: Continuous endpoint per subject -- the measurement the label thresholds.
        leads: Lead names the recordings are stored in.
        sampling_rate: Samples per second of the recordings.
    """

    subject_id: list[str]
    ecg: dict[str, np.ndarray]
    image: dict[str, np.ndarray]
    labels: dict[str, int]
    values: dict[str, float]
    leads: tuple[str, ...]
    sampling_rate: float


def make_synthetic_ecg(
    num_subjects: int = 32,
    leads: Sequence[str] = DEFAULT_LEADS,
    num_samples: int = 500,
    sampling_rate: float = 500.0,
    heart_rate: float = 70.0,
    effect: float = 1.0,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, float], tuple[str, ...]]:
    """Generate beat-like multi-lead recordings with a latent severity per subject.

    Each subject is given a severity drawn from a standard normal. Severity raises the
    heart rate, sharpens the beat and shifts the lead amplitudes, so a model can recover
    it -- which is what makes a downstream endpoint learnable rather than noise.

    Args:
        num_subjects: How many subjects.
        leads: Lead names; their count is the number of channels.
        num_samples: Samples per lead.
        sampling_rate: Samples per second, used to place the beats in time.
        heart_rate: Beats per minute at severity zero.
        effect: How strongly severity changes the waveform. 0 makes the signal
            independent of the endpoint, which is the case to generate when testing
            that a pipeline does *not* find structure that is not there.
        noise: Standard deviation of the additive noise.
        seed: Seed, making the cohort reproducible.

    Returns:
        ``(subject_ids, recordings, severity, leads)``.

    Raises:
        ValueError: If a count is not positive, or no lead is given.
    """
    if num_subjects < 1 or num_samples < 1 or not leads:
        raise ValueError(
            f"need at least one subject, one sample and one lead, got {num_subjects}, {num_samples}, {list(leads)}"
        )

    generator = np.random.default_rng(seed)
    leads = tuple(leads)
    # A fixed per-lead gain and phase, so the leads differ from each other in a way
    # that is consistent across subjects -- as real leads do, being projections of one
    # signal onto different axes.
    lead_gain = np.linspace(0.6, 1.4, len(leads))
    lead_phase = np.linspace(0.0, np.pi, len(leads), endpoint=False)

    subject_ids = [f"S{index:04d}" for index in range(num_subjects)]
    time = np.arange(num_samples) / sampling_rate
    recordings: dict[str, np.ndarray] = {}
    severity: dict[str, float] = {}

    for identifier in subject_ids:
        score = float(generator.normal())
        beats_per_second = (heart_rate + effect * 8.0 * score) / 60.0
        phase = 2 * np.pi * beats_per_second * time

        # A narrow positive spike per beat for the QRS, a broad bump after it for the T
        # wave, and a slow baseline. Crude on purpose; see the module docstring.
        qrs = np.exp(4.0 * (np.cos(phase) - 1.0)) * (1.0 + 0.3 * effect * score)
        t_wave = 0.25 * np.exp(2.0 * (np.cos(phase - 0.9) - 1.0))
        baseline = 0.05 * np.sin(2 * np.pi * 0.25 * time)

        signal = np.stack(
            [
                gain * (qrs + t_wave) + baseline + 0.1 * np.sin(phase + shift)
                for gain, shift in zip(lead_gain, lead_phase, strict=True)
            ]
        )
        signal = signal + generator.normal(scale=noise, size=signal.shape)
        recordings[identifier] = signal.astype(np.float32)
        severity[identifier] = score

    return subject_ids, recordings, severity, leads


def make_synthetic_image_data(
    subject_ids: Sequence[str],
    severity: dict[str, float] | None = None,
    size: Sequence[int] = (32, 32),
    channels: int = 1,
    effect: float = 1.0,
    noise: float = 0.1,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Generate images whose central intensity tracks each subject's severity.

    A stand-in for a chest radiograph, in which a raised pressure enlarges the cardiac
    silhouette. Here that is a bright central blob whose radius grows with severity --
    enough dependence for a model to find, and nothing that resembles anatomy.

    Values are in the unit interval, matching what
    :func:`~kalecardiac.prepdata.build_image_pipeline` produces and what a
    sigmoid image decoder can reconstruct.

    Args:
        subject_ids: Subjects to generate for, typically from
            :func:`make_synthetic_ecg` so the two modalities share their severity.
        severity: Severity per subject. ``None`` draws an independent one, which makes
            the image uninformative about an ECG-derived endpoint.
        size: ``(height, width)``.
        channels: Channels per image.
        effect: How strongly severity changes the image.
        noise: Standard deviation of the additive noise.
        seed: Seed.

    Returns:
        One ``(channels, height, width)`` array per subject.

    Raises:
        ValueError: If a dimension is not positive.
    """
    if channels < 1 or len(size) != 2 or any(int(extent) < 1 for extent in size):
        raise ValueError(f"need positive channels and a (height, width) size, got {channels}, {tuple(size)}")

    generator = np.random.default_rng(seed)
    height, width = int(size[0]), int(size[1])
    rows = np.linspace(-1.0, 1.0, height)[:, None]
    columns = np.linspace(-1.0, 1.0, width)[None, :]
    radius = np.sqrt(rows**2 + columns**2)

    images: dict[str, np.ndarray] = {}
    for identifier in subject_ids:
        score = float(generator.normal()) if severity is None else severity[identifier]
        extent = 0.45 + 0.12 * effect * score
        blob = np.exp(-((radius / max(extent, 0.05)) ** 2))
        image = np.repeat(blob[None, :, :], channels, axis=0)
        image = image + generator.normal(scale=noise, size=image.shape)
        images[identifier] = np.clip(image, 0.0, 1.0).astype(np.float32)
    return images


def make_synthetic_multimodal_data(
    num_subjects: int = 32,
    leads: Sequence[str] = DEFAULT_LEADS,
    num_samples: int = 500,
    sampling_rate: float = 500.0,
    with_images: bool = True,
    image_size: Sequence[int] = (32, 32),
    image_channels: int = 1,
    effect: float = 1.0,
    noise: float = 0.2,
    positive_rate: float = 0.4,
    seed: int = 0,
) -> SyntheticCohort:
    """Generate a paired ECG-and-image cohort with a shared latent severity.

    The endpoint is derived from the same severity that shapes both modalities, so a
    multimodal model has something real to fuse and a unimodal one has less of it. Both
    a continuous endpoint and a thresholded binary one are returned, mirroring how a
    cardiac cohort is actually labelled: a catheter measures a pressure, and a clinical
    threshold turns it into a diagnosis.

    Args:
        num_subjects: How many subjects.
        leads: Lead names.
        num_samples: Samples per lead.
        sampling_rate: Samples per second.
        with_images: Also generate an image modality.
        image_size: ``(height, width)`` of the images.
        image_channels: Channels per image.
        effect: How strongly severity shapes the modalities. 0 makes the endpoint
            unlearnable, which is the cohort to generate when testing that a metric
            reports chance.
        noise: Standard deviation of the additive noise.
        positive_rate: Share of subjects above the endpoint threshold. The threshold is
            the corresponding quantile of the generated measurement, so the rate is
            met exactly rather than in expectation.
        seed: Seed, making the cohort reproducible.

    Returns:
        The cohort.

    Raises:
        ValueError: If ``positive_rate`` is outside ``(0, 1)``.
    """
    if not 0 < positive_rate < 1:
        raise ValueError(f"positive_rate must lie in (0, 1), got {positive_rate}")

    subject_ids, recordings, severity, leads = make_synthetic_ecg(
        num_subjects=num_subjects,
        leads=leads,
        num_samples=num_samples,
        sampling_rate=sampling_rate,
        effect=effect,
        noise=noise,
        seed=seed,
    )
    images = (
        make_synthetic_image_data(
            subject_ids,
            severity=severity,
            size=image_size,
            channels=image_channels,
            effect=effect,
            noise=noise / 2,
            seed=seed + 1,
        )
        if with_images
        else {}
    )

    # A pressure-like measurement in plausible units, so a regression test has an
    # endpoint with a scale rather than a standard normal.
    generator = np.random.default_rng(seed + 2)
    values = {
        identifier: float(20.0 + 6.0 * severity[identifier] + generator.normal(scale=1.0)) for identifier in subject_ids
    }
    # Thresholded at a quantile rather than at a fixed value, so the class balance is
    # exactly what was asked for however the measurement happened to fall.
    threshold = float(np.quantile(list(values.values()), 1.0 - positive_rate))
    labels = {identifier: int(value > threshold) for identifier, value in values.items()}

    return SyntheticCohort(
        subject_id=subject_ids,
        ecg=recordings,
        image=images,
        labels=labels,
        values=values,
        leads=leads,
        sampling_rate=sampling_rate,
    )
