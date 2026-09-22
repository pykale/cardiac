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
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from examples.cardiovae_multimodal.config import CXR, ECG
from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.loaddata import (
    ColumnTarget,
    ECGArraySource,
    ECGFileSource,
    ImageArraySource,
    ImageFileSource,
    ModalitySource,
)
from kalecardiac.prepdata import SignalPipeline, build_image_pipeline, crop_or_pad, interpolate_missing, standardise

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

    def frame(self) -> pd.DataFrame:
        """The table a splitter splits."""
        return self.table

    def describe(self) -> str:
        """One line summarising what was assembled."""
        positives = int(self.table["label"].sum()) if "label" in self.table else 0
        origin = "synthetic" if self.synthetic else "local"
        return (
            f"{origin}: {len(self.pretrain_ids)} subjects for pretraining, {len(self.labelled_ids)} labelled "
            f"({positives} positive) | ecg {self.ecg_channels}x{self.ecg_length} | cxr {self.image_size}"
        )


def ecg_pipeline(cfg, channels: int) -> SignalPipeline:
    """Preprocessing for the recordings.

    Interpolate dropped samples, force a fixed length, then standardise. The order is
    not interchangeable: a NaN spreads through everything after it, and statistics taken
    over samples that are then cropped away describe a recording the model never sees.
    """
    steps = [interpolate_missing, partial(crop_or_pad, length=cfg.ECG.LENGTH)]
    if cfg.ECG.STANDARDISE:
        # Per channel when the leads are channels; over the whole recording when they
        # have been flattened into one, where per-channel would standardise the
        # concatenation as a unit anyway.
        steps.append(partial(standardise, per_channel=channels > 1))
    return SignalPipeline(steps)


def _flatten_leads(signal: np.ndarray) -> np.ndarray:
    """Concatenate every lead into one channel, as the reference implementation does."""
    return signal.reshape(1, -1)


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
    for column in (SUBJECT_COLUMN, cfg.ENDPOINT.VALUE_COLUMN):
        if column not in table.columns:
            raise ValueError(f"{table_path} has no column {column!r}; available: {list(table.columns)}")
    table[SUBJECT_COLUMN] = table[SUBJECT_COLUMN].astype(str)

    def files(directory: str) -> dict[str, Path]:
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"no {directory}/ directory under {root}")
        return {path.stem: path for path in sorted(folder.glob("*.npy"))}

    ecg_files, cxr_files = files("ecg"), files("cxr")
    channels = int(cfg.ECG.NUM_CHANNELS)
    pipeline = ecg_pipeline(cfg, channels)
    if channels == 1:
        pipeline = SignalPipeline([_flatten_leads, *pipeline.steps])

    leads = list(cfg.ECG.LEADS) or ["I"]
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
        transform=build_image_pipeline(size=tuple(cfg.IMAGE.SIZE), scale=cfg.IMAGE.SCALE),
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
        image_size=tuple(cfg.SYNTHETIC.IMAGE_SIZE),
        image_channels=cfg.IMAGE.CHANNELS,
        seed=cfg.SOLVER.SEED,
    )

    channels = int(cfg.ECG.NUM_CHANNELS)
    pipeline = ecg_pipeline(cfg, channels)
    if channels == 1:
        pipeline = SignalPipeline([_flatten_leads, *pipeline.steps])

    ecg_source = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        sampling_rate=generated.sampling_rate,
        transform=pipeline,
        length=cfg.ECG.LENGTH,
    )
    image_source = ImageArraySource(
        generated.image,
        channels=cfg.IMAGE.CHANNELS,
        size=tuple(cfg.IMAGE.SIZE),
        transform=build_image_pipeline(size=tuple(cfg.IMAGE.SIZE), scale=cfg.IMAGE.SCALE),
    )

    table = pd.DataFrame(
        {
            SUBJECT_COLUMN: generated.subject_id,
            cfg.ENDPOINT.VALUE_COLUMN: [generated.values[name] for name in generated.subject_id],
        }
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

    # The endpoint: a threshold on the measurement, or the measurement itself. Which
    # column and which threshold are the registry's, which is why they are configured
    # here rather than in the library.
    values = pd.to_numeric(table[cfg.ENDPOINT.VALUE_COLUMN], errors="coerce")
    known = values.notna()
    if not bool(known.all()):
        logger.info("dropping %d subjects with no recorded %s", int((~known).sum()), cfg.ENDPOINT.VALUE_COLUMN)
        table = table[known].reset_index(drop=True)
        values = values[known].reset_index(drop=True)

    table["value"] = values.astype(float)
    table["label"] = (values > cfg.ENDPOINT.THRESHOLD).astype(int)
    columns = {"value": "value"} if cfg.ENDPOINT.REGRESSION else {"label": "label"}
    target = ColumnTarget(table, columns=columns, id_column=SUBJECT_COLUMN)

    labelled = list(table[SUBJECT_COLUMN])
    # The source measured its own post-transform shape at construction, so the model
    # can be built from it without reading a recording here.
    ecg_source = sources[ECG]

    return CardioVAECohort(
        sources=sources,
        target=target,
        labelled_ids=labelled,
        # Every subject with the modalities takes part in pretraining, labelled or not.
        pretrain_ids=sorted(usable),
        table=table,
        ecg_channels=ecg_source.channels,
        ecg_length=ecg_source.length,
        image_size=tuple(sources[CXR].size),
        synthetic=synthetic,
    )


def load_cohort(cfg) -> CardioVAECohort:
    """Read the cohort from disk, or generate one when no root is configured."""
    if cfg.DATASET.ROOT:
        return load_local(cfg)
    logger.warning(
        "DATASET.ROOT is empty, so this run uses synthetic data. It verifies that the pipeline executes; "
        "it is not a result."
    )
    return load_synthetic(cfg)


def summarise(cohort: CardioVAECohort, endpoint: str) -> dict:
    """A record of what the cohort was, for the run's output directory."""
    return {
        "endpoint": endpoint,
        "synthetic": cohort.synthetic,
        "n_pretrain": len(cohort.pretrain_ids),
        "n_labelled": len(cohort.labelled_ids),
        "n_positive": int(cohort.table["label"].sum()),
        "positive_rate": float(cohort.table["label"].mean()),
        "ecg_shape": [cohort.ecg_channels, cohort.ecg_length],
        "image_size": list(cohort.image_size),
    }


def modality_names() -> Sequence[str]:
    """The modalities this example fuses."""
    return (ECG, CXR)
