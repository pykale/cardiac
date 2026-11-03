"""Building the CardioVAE cohort: where its files are, and what its columns mean.

This is the dataset-specific half of the example, and everything in it is a decision
the cohort forces rather than a mechanism the library supplies. The expected layout is
in this example's README; nothing here reads a cohort this package ships, because the
cohort the paper used is a private clinical registry.

``DATASET.ROOT`` empty falls back to :mod:`examples.synthetic_data`, so the whole
workflow runs end to end with no data at all. A synthetic run verifies the pipeline; it
is not a result, and the log says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision.transforms import Compose

from examples.cardiovae_multimodal.config import CXR, ECG
from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.auto import (
    check_endpoint,
    derive_endpoint,
    describe_class_balance,
    get_class_balance,
    get_ecg_transform_from_cfg,
    get_endpoint_column,
    get_endpoint_target,
    get_source_sampling_rate,
)
from kalecardiac.loaddata import (
    ColumnTarget,
    ECGArraySource,
    ECGFileSource,
    ImageArraySource,
    ImageFileSource,
    ModalitySource,
)
from kalecardiac.prepdata import get_image_transform

logger = logging.getLogger("cardiovae_multimodal")

#: Column names this example expects in the cohort's table. Declared here, in the
#: experiment, because they are the registry's vocabulary rather than the library's.
SUBJECT_COLUMN = "subject_id"


@dataclass
class CardioVAECohort:
    """One assembled cohort, ready to split and train on.

    Attributes:
        sources: One :class:`~kalecardiac.loaddata.ModalitySource` per modality.
        target: Supervision for the labelled subjects.
        labelled_ids: Subjects with an endpoint, which the fine-tuning stage uses.
        pretrain_ids: Subjects with the modalities, labelled or not, which the
            representation-learning stage uses. Usually the larger set -- that is the
            point of pretraining.
        table: One row per labelled subject, carrying the endpoint for stratification.
        ecg_channels: Channels the recordings are presented with.
        ecg_length: Samples per channel after preprocessing.
        image_size: ``(height, width)`` after preprocessing.
        synthetic: Whether this cohort was generated rather than read.
        num_classes: Classes of the endpoint; 2 for a threshold.
    """

    sources: dict[str, ModalitySource]
    target: ColumnTarget
    labelled_ids: list[str]
    pretrain_ids: list[str]
    table: pd.DataFrame
    ecg_channels: int
    ecg_length: int
    image_size: tuple[int, int]
    synthetic: bool
    num_classes: int = 2

    def describe(self) -> str:
        """One line summarising what was assembled."""
        origin = "synthetic" if self.synthetic else "local"
        return (
            f"{origin}: {len(self.pretrain_ids)} subjects for pretraining, {len(self.labelled_ids)} labelled "
            f"({describe_class_balance(self.table, self.num_classes)}) | ecg {self.ecg_channels}x{self.ecg_length} | "
            f"cxr {self.image_size}"
        )


def ecg_pipeline(cfg, flatten: bool, num_leads: int) -> Compose:
    """Preprocessing for the recordings, in the order the reference implementation applies it.

    Interpolate dropped samples, resample to the model's rate, crop or pad each lead to a
    fixed length, normalize each lead, then -- if the model reads the leads as one signal
    -- interleave them into one channel.

    Every step before the interleaving is per lead, and has to be: resampling or
    interpolating a flattened strip would mix one lead into its neighbour. The
    interleaving is the reference implementation's -- it reads a ``(samples, leads)``
    recording and flattens it row by row, so the leads of each time step sit together --
    and it is what the published checkpoint expects.

    Args:
        cfg: The run configuration.
        flatten: Interleave the leads into one channel. ``ECG.LENGTH`` is then the
            leads times the samples per lead, and each lead is cropped to its share.
        num_leads: Leads each recording holds.

    Returns:
        A :class:`~torchvision.transforms.Compose` to hand to an ECG source.

    Raises:
        ValueError: If ``ECG.LENGTH`` is not a whole number of samples per lead.
    """
    if not flatten:
        return get_ecg_transform_from_cfg(cfg)
    if cfg.ECG.LENGTH % num_leads:
        raise ValueError(
            f"ECG.LENGTH {cfg.ECG.LENGTH} is not a whole number of samples for each of {num_leads} leads; set it "
            f"to the leads times the samples per lead, e.g. {num_leads * 5000} for ten seconds at 500 Hz"
        )
    per_lead = get_ecg_transform_from_cfg(cfg, length=cfg.ECG.LENGTH // num_leads)
    return Compose([*per_lead.transforms, _interleave_leads])


def _interleave_leads(signal: np.ndarray) -> np.ndarray:
    """One channel, the leads interleaved sample by sample, as the reference implementation flattens them."""
    return np.ascontiguousarray(signal.T).reshape(1, -1)


def load_local(cfg) -> CardioVAECohort:
    """Read a cohort from ``DATASET.ROOT``.

    Expected layout, which the README documents in full:

    .. code-block:: text

        <ROOT>/
        |-- cohort.csv          subject_id, the endpoint column, any covariates
        |-- ecg/<subject_id>.npy    (num_leads, num_samples) or (num_samples,)
        `-- cxr/<subject_id>.npy    (height, width) or (channels, height, width)

    Raises:
        FileNotFoundError: If the table or a modality directory is absent.
        ValueError: If the table lacks the identifier or endpoint column.
    """
    root = Path(cfg.DATASET.ROOT)
    table_path = root / "cohort.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"no cohort table at {table_path}. See the example README for the expected layout; the registry "
            f"this example was developed on is private and is not distributed."
        )

    table = pd.read_csv(table_path)
    for column in (SUBJECT_COLUMN, get_endpoint_column(cfg)):
        if column not in table.columns:
            raise ValueError(f"{table_path} has no column {column!r}; available: {list(table.columns)}")
    table[SUBJECT_COLUMN] = table[SUBJECT_COLUMN].astype(str)

    def files(directory: str) -> dict[str, Path]:
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"no {directory}/ directory under {root}")
        return {path.stem: path for path in sorted(folder.glob("*.npy"))}

    ecg_files, cxr_files = files("ecg"), files("cxr")
    leads = list(cfg.ECG.LEADS)
    if not leads:
        raise ValueError("ECG.LEADS must name the leads each recording file stores, in the order it stores them")
    pipeline = ecg_pipeline(cfg, flatten=int(cfg.ECG.NUM_CHANNELS) == 1, num_leads=len(leads))

    ecg_source = ECGFileSource(
        ecg_files,
        leads=leads,
        sampling_rate=cfg.ECG.SAMPLING_RATE,
        transform=pipeline,
        length=cfg.ECG.LENGTH,
    )
    image_source = ImageFileSource(
        cxr_files,
        channels=cfg.IMAGE.CHANNELS,
        size=tuple(cfg.IMAGE.SIZE),
        transform=get_image_transform(size=tuple(cfg.IMAGE.SIZE), scale=cfg.IMAGE.SCALE),
    )

    return _assemble(cfg, table, {ECG: ecg_source, CXR: image_source}, set(ecg_files) & set(cxr_files), synthetic=False)


def load_synthetic(cfg) -> CardioVAECohort:
    """Generate a cohort, so the example runs with no data.

    A synthetic run exercises every code path and produces a number. The number is not
    a result, and nothing downstream should read it as one.
    """
    generated = make_synthetic_multimodal_data(
        num_subjects=cfg.SYNTHETIC.NUM_SUBJECTS,
        num_samples=cfg.SYNTHETIC.NUM_SAMPLES,
        # Generated at the rate a real cohort would arrive at, so that a configured
        # resample is exercised rather than skipped.
        sampling_rate=get_source_sampling_rate(cfg),
        image_size=tuple(cfg.SYNTHETIC.IMAGE_SIZE),
        image_channels=cfg.IMAGE.CHANNELS,
        seed=cfg.SOLVER.SEED,
    )

    pipeline = ecg_pipeline(cfg, flatten=int(cfg.ECG.NUM_CHANNELS) == 1, num_leads=len(generated.leads))

    ecg_source = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        # The rate of what the source returns, after resampling -- which is what an
        # attribution or a delineation needs to place a sample in time.
        sampling_rate=cfg.ECG.SAMPLING_RATE,
        transform=pipeline,
        length=cfg.ECG.LENGTH,
    )
    image_source = ImageArraySource(
        generated.image,
        channels=cfg.IMAGE.CHANNELS,
        size=tuple(cfg.IMAGE.SIZE),
        transform=get_image_transform(size=tuple(cfg.IMAGE.SIZE), scale=cfg.IMAGE.SCALE),
    )

    table = pd.DataFrame(
        {
            SUBJECT_COLUMN: generated.subject_id,
            cfg.ENDPOINT.VALUE_COLUMN: [generated.values[name] for name in generated.subject_id],
        }
    )
    if cfg.ENDPOINT.LABEL_COLUMN:
        # A recorded class, made learnable by binning the same measurement the threshold
        # would use into equally populated classes.
        table[cfg.ENDPOINT.LABEL_COLUMN] = pd.qcut(
            table[cfg.ENDPOINT.VALUE_COLUMN], int(cfg.ENDPOINT.NUM_CLASSES), labels=False
        )
    return _assemble(cfg, table, {ECG: ecg_source, CXR: image_source}, set(generated.subject_id), synthetic=True)


def _assemble(
    cfg,
    table: pd.DataFrame,
    sources: dict[str, ModalitySource],
    available: set[str],
    synthetic: bool,
) -> CardioVAECohort:
    """Join the table to the modalities and derive the endpoint."""
    required = list(cfg.DATASET.REQUIRE_MODALITIES) or []
    usable = set(available)
    if required:
        logger.info("restricting the cohort to subjects with %s", required)

    table = table[table[SUBJECT_COLUMN].isin(usable)].reset_index(drop=True)
    if table.empty:
        raise ValueError("no subject survived the join between the cohort table and the modality files")
    # The endpoint: a class of it, or the measurement itself. Which column and which
    # threshold are the registry's, which is why they are configured rather than coded.
    table = derive_endpoint(cfg, table)

    # The source measured its own post-transform shape at construction, so the model
    # can be built from it without reading a recording here.
    ecg_source = sources[ECG]
    return CardioVAECohort(
        sources=sources,
        target=get_endpoint_target(cfg, table, SUBJECT_COLUMN),
        labelled_ids=list(table[SUBJECT_COLUMN]),
        # Every subject with the modalities takes part in pretraining, labelled or not.
        pretrain_ids=sorted(usable),
        table=table,
        ecg_channels=ecg_source.channels,
        ecg_length=ecg_source.length,
        image_size=tuple(sources[CXR].size),
        synthetic=synthetic,
        num_classes=int(cfg.ENDPOINT.NUM_CLASSES),
    )


def load_cohort(cfg) -> CardioVAECohort:
    """Read the cohort from disk, or generate one when no root is configured."""
    check_endpoint(cfg)
    if cfg.DATASET.ROOT:
        return load_local(cfg)
    logger.warning(
        "DATASET.ROOT is empty, so this run uses synthetic data. It verifies that the pipeline executes; "
        "it is not a result."
    )
    return load_synthetic(cfg)


def summarize_cohort(cohort: CardioVAECohort, endpoint: str) -> dict:
    """A record of what the cohort was, for the run's output directory."""
    return {
        "endpoint": endpoint,
        "synthetic": cohort.synthetic,
        "n_pretrain": len(cohort.pretrain_ids),
        "n_labelled": len(cohort.labelled_ids),
        **get_class_balance(cohort.table, cohort.num_classes),
        "ecg_shape": [cohort.ecg_channels, cohort.ecg_length],
        "image_size": list(cohort.image_size),
    }


def modality_names() -> Sequence[str]:
    """The modalities this example fuses."""
    return (ECG, CXR)
