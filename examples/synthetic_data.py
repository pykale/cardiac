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

A cohort can also be written to disk as DICOM -- :func:`write_synthetic_dicom_cohort`,
one folder per subject, as a hospital exports one -- so that reading raw files is tested
without any raw file being real. That needs pydicom, from the ``imaging`` extra.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from kalecardiac.loaddata import STANDARD_12_LEAD

#: Lead names used when a caller does not supply their own. The six limb leads, because
#: that is the reduced configuration a low-cost recorder captures and the one both
#: cardiac use cases fine-tune on.
DEFAULT_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")

#: The SOP classes the DICOM writers produce: a twelve-lead ECG waveform, and a computed
#: radiograph.
ECG_WAVEFORM_STORAGE = "1.2.840.10008.5.1.4.1.1.9.1.1"
RADIOGRAPH_STORAGE = "1.2.840.10008.5.1.4.1.1.1"

#: How an ECG cart codes each standard lead (SCP-ECG), so a written file names its leads
#: as a real export does.
SCPECG_LEAD_CODES: dict[str, str] = {
    "I": "5.6.3-9-1",
    "II": "5.6.3-9-2",
    "III": "5.6.3-9-61",
    "aVR": "5.6.3-9-62",
    "aVL": "5.6.3-9-63",
    "aVF": "5.6.3-9-64",
    **{f"V{index}": f"5.6.3-9-{index + 2}" for index in range(1, 7)},
}

#: Millivolts per unit of the voltages a written channel may be stated in, by UCUM code.
MILLIVOLTS_PER_UNIT: dict[str, float] = {"uV": 1e-3, "mV": 1.0, "V": 1e3}


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
    :func:`~kalecardiac.prepdata.get_image_transform` produces and what a
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


def _require_pydicom():
    """Import pydicom's dataset and UID helpers, explaining the install if it is absent."""
    try:
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError('writing DICOM needs pydicom: pip install "kalecardiac[imaging]"') from error
    return Dataset, FileMetaDataset, ExplicitVRLittleEndian, generate_uid


def _dicom_file(sop_class: str, modality: str, patient_id: str):
    """An empty DICOM file of one SOP class, with the header every reader expects."""
    Dataset, FileMetaDataset, ExplicitVRLittleEndian, generate_uid = _require_pydicom()
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = sop_class
    dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.PatientID = patient_id
    dataset.InstanceNumber = 1
    return dataset


def _save(dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(path, enforce_file_format=True)
    return path


def _waveform_group(
    recording: np.ndarray,
    leads: Sequence[str],
    sampling_rate: float,
    unit: str,
    sensitivity: float,
    label: str,
    originality: str,
):
    """One multiplex group: a channel per lead, each stored as whole multiples of its sensitivity."""
    Dataset = _require_pydicom()[0]
    counts = np.round(np.asarray(recording, dtype=np.float64) / (sensitivity * MILLIVOLTS_PER_UNIT.get(unit, 1.0)))
    counts = np.clip(counts, -32768, 32767).astype("<i2")

    channels = []
    for lead in leads:
        source = Dataset()
        source.CodeValue = SCPECG_LEAD_CODES.get(lead, lead)
        source.CodingSchemeDesignator = "SCPECG"
        source.CodeMeaning = "Lead I (Einthoven)" if lead == "I" else f"Lead {lead}"
        units = Dataset()
        units.CodeValue = unit
        units.CodingSchemeDesignator = "UCUM"
        units.CodeMeaning = unit
        channel = Dataset()
        channel.ChannelSourceSequence = [source]
        channel.ChannelSensitivity = sensitivity
        channel.ChannelSensitivityUnitsSequence = [units]
        channel.ChannelSensitivityCorrectionFactor = 1
        channel.ChannelBaseline = 0
        channel.WaveformBitsStored = 16
        channels.append(channel)

    group = Dataset()
    group.MultiplexGroupTimeOffset = 0
    group.WaveformOriginality = originality
    group.NumberOfWaveformChannels = counts.shape[0]
    group.NumberOfWaveformSamples = counts.shape[1]
    group.SamplingFrequency = sampling_rate
    group.MultiplexGroupLabel = label
    group.ChannelDefinitionSequence = channels
    group.WaveformBitsAllocated = 16
    group.WaveformSampleInterpretation = "SS"
    # Samples are stored time-major: every channel's first sample, then every second.
    group.add_new(0x54001010, "OW", np.ascontiguousarray(counts.T).tobytes())
    return group


def write_dicom_ecg(
    path: str | Path,
    recording: np.ndarray,
    leads: Sequence[str],
    sampling_rate: float = 500.0,
    unit: str = "uV",
    sensitivity: float = 5.0,
    median_beat: bool = False,
    patient_id: str = "SYNTHETIC",
) -> Path:
    """Write a recording as a DICOM twelve-lead ECG waveform, as an ECG cart exports one.

    Each channel names its lead as such an export does -- ``"Lead I (Einthoven)"``,
    ``"Lead aVR"`` -- states its sensitivity in ``unit``, and stores whole multiples of it,
    so what is read back is the recording to within half a sensitivity step.

    Args:
        path: Where to write, folders created as needed.
        recording: ``(num_leads, num_samples)``, in millivolts.
        leads: Lead names, in the order the rows hold them.
        sampling_rate: Samples per second.
        unit: UCUM code of the unit each channel's sensitivity is stated in.
        sensitivity: How many ``unit`` one stored step is.
        median_beat: Also write a derived group -- the median beat a cart computes --
            ahead of the recorded one, so a reader that took the first group would read
            the wrong one.
        patient_id: The header's patient identifier.

    Returns:
        ``path``.
    """
    recording = np.asarray(recording, dtype=np.float64)
    groups = [_waveform_group(recording, leads, sampling_rate, unit, sensitivity, "RHYTHM", "ORIGINAL")]
    if median_beat:
        beat = recording[:, : max(1, recording.shape[1] // 10)]
        groups.insert(0, _waveform_group(beat, leads, sampling_rate, unit, sensitivity, "MEDIAN BEAT", "DERIVED"))
    dataset = _dicom_file(ECG_WAVEFORM_STORAGE, "ECG", patient_id)
    dataset.WaveformSequence = groups
    return _save(dataset, path)


def write_dicom_image(
    path: str | Path,
    image: np.ndarray,
    photometric: str = "MONOCHROME2",
    bits_stored: int = 12,
    window: tuple[float, float] | None = None,
    rescale: tuple[float, float] | None = None,
    patient_id: str = "SYNTHETIC",
) -> Path:
    """Write a 2-D image as a DICOM radiograph.

    Args:
        path: Where to write, folders created as needed.
        image: ``(height, width)`` stored values, rounded and clipped to ``bits_stored``.
        photometric: ``"MONOCHROME2"``, where a higher value is brighter, or
            ``"MONOCHROME1"``, where it is darker.
        bits_stored: Bits each stored value uses.
        window: ``(center, width)`` to display it with, if any.
        rescale: ``(slope, intercept)`` mapping stored values to the modality's units,
            if any.
        patient_id: The header's patient identifier.

    Returns:
        ``path``.

    Raises:
        ValueError: If ``image`` is not 2-D.
    """
    pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim != 2:
        raise ValueError(f"a radiograph is 2-D, got shape {pixels.shape}")
    stored = np.clip(np.round(pixels), 0, 2**bits_stored - 1).astype("<u2")

    dataset = _dicom_file(RADIOGRAPH_STORAGE, "CR", patient_id)
    dataset.Rows, dataset.Columns = stored.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = bits_stored
    dataset.HighBit = bits_stored - 1
    dataset.PixelRepresentation = 0
    if window is not None:
        dataset.WindowCenter, dataset.WindowWidth = window
    if rescale is not None:
        dataset.RescaleSlope, dataset.RescaleIntercept = rescale
    dataset.PixelData = stored.tobytes()
    return _save(dataset, path)


def write_synthetic_dicom_cohort(
    root: str | Path,
    num_subjects: int = 20,
    num_samples: int = 5000,
    sampling_rate: float = 500.0,
    image_size: Sequence[int] = (64, 64),
    with_labels: bool = True,
    seed: int = 0,
) -> SyntheticCohort:
    """Write a paired twelve-lead ECG and radiograph cohort as DICOM, laid out as a hospital exports one.

    .. code-block:: text

        <root>/ECG/<subject>/1/DICOM/00000000.dcm   a twelve-lead ECG waveform
        <root>/CXR/<subject>/1/DICOM/00000000.dcm   a 12-bit radiograph
        <root>/labels.csv                           subject_id,label  (with_labels)

    The recordings and images are :func:`make_synthetic_multimodal_data`'s, so the label
    is learnable from both, and the cohort is returned so a test can compare what a
    reader gives back with what was written.

    Args:
        root: The folder to write the cohort into.
        num_subjects: How many subjects.
        num_samples: Samples per lead.
        sampling_rate: Samples per second.
        image_size: ``(height, width)`` of the radiographs.
        with_labels: Also write the binary labels, as a fine-tuning cohort carries them.
        seed: Seed, making the cohort reproducible.

    Returns:
        The cohort that was written, its recordings in millivolts.
    """
    cohort = make_synthetic_multimodal_data(
        num_subjects=num_subjects,
        leads=STANDARD_12_LEAD,
        num_samples=num_samples,
        sampling_rate=sampling_rate,
        image_size=image_size,
        seed=seed,
    )
    root = Path(root)
    for subject in cohort.subject_id:
        write_dicom_ecg(
            root / "ECG" / subject / "1" / "DICOM" / "00000000.dcm",
            cohort.ecg[subject],
            cohort.leads,
            sampling_rate=sampling_rate,
            patient_id=subject,
        )
        # The generator's [0, 1] images, stored as a 12-bit detector would store them.
        write_dicom_image(
            root / "CXR" / subject / "1" / "DICOM" / "00000000.dcm",
            cohort.image[subject][0] * 4095,
            window=(2047.5, 4096),
            patient_id=subject,
        )
    if with_labels:
        with open(root / "labels.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject_id", "label"])
            writer.writerows((subject, cohort.labels[subject]) for subject in cohort.subject_id)
    return cohort
