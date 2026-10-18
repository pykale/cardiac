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
from kalecardiac.prepdata import (
    SignalPipeline,
    build_image_pipeline,
    crop_or_pad,
    interpolate_missing,
    resample,
    standardise,
)

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

    def frame(self) -> pd.DataFrame:
        """The table a splitter splits."""
        return self.table

    def describe(self) -> str:
        """One line summarising what was assembled."""
        origin = "synthetic" if self.synthetic else "local"
        return (
            f"{origin}: {len(self.pretrain_ids)} subjects for pretraining, {len(self.labelled_ids)} labelled "
            f"({_balance_text(self.table, self.num_classes)}) | ecg {self.ecg_channels}x{self.ecg_length} | "
            f"cxr {self.image_size}"
        )


def source_rate(cfg) -> float:
    """Rate the recordings arrive at: ``ECG.SOURCE_SAMPLING_RATE``, or the model's own."""
    return float(cfg.ECG.SOURCE_SAMPLING_RATE or cfg.ECG.SAMPLING_RATE)


def ecg_pipeline(cfg, flatten: bool) -> SignalPipeline:
    """Preprocessing for the recordings, in the only order that is correct.

    Interpolate dropped samples, resample to the model's rate, flatten the leads if the
    model reads them as one signal, force a fixed length, then standardise.

    Every step before the flatten is per lead, and has to be. Resampling a flattened
    strip would interpolate across the join between the end of one lead and the start of
    the next, inventing samples that belong to neither; interpolating a dropped sample
    there would fill it from the wrong lead. Statistics go last, because statistics taken
    over samples that are then cropped away describe a recording the model never sees.

    Args:
        cfg: The run configuration.
        flatten: Concatenate the leads into one channel, as the reference
            implementation does. ``ECG.LENGTH`` is then leads times samples per lead.

    Returns:
        A pipeline to hand to an ECG source's ``transform``.
    """
    steps = [interpolate_missing]
    if source_rate(cfg) != cfg.ECG.SAMPLING_RATE:
        steps.append(partial(resample, source_rate=source_rate(cfg), target_rate=cfg.ECG.SAMPLING_RATE))
    if flatten:
        steps.append(_flatten_leads)
    steps.append(partial(crop_or_pad, length=cfg.ECG.LENGTH))
    if cfg.ECG.STANDARDISE:
        # Per lead while the leads are channels; over the whole strip once flattened,
        # where there is only one channel left to standardise.
        steps.append(partial(standardise, per_channel=not flatten))
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
    for column in (SUBJECT_COLUMN, endpoint_column(cfg)):
        if column not in table.columns:
            raise ValueError(f"{table_path} has no column {column!r}; available: {list(table.columns)}")
    table[SUBJECT_COLUMN] = table[SUBJECT_COLUMN].astype(str)

    def files(directory: str) -> dict[str, Path]:
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"no {directory}/ directory under {root}")
        return {path.stem: path for path in sorted(folder.glob("*.npy"))}

    ecg_files, cxr_files = files("ecg"), files("cxr")
    pipeline = ecg_pipeline(cfg, flatten=int(cfg.ECG.NUM_CHANNELS) == 1)

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
        # Generated at the rate a real cohort would arrive at, so that a configured
        # resample is exercised rather than skipped.
        sampling_rate=source_rate(cfg),
        image_size=tuple(cfg.SYNTHETIC.IMAGE_SIZE),
        image_channels=cfg.IMAGE.CHANNELS,
        seed=cfg.SOLVER.SEED,
    )

    pipeline = ecg_pipeline(cfg, flatten=int(cfg.ECG.NUM_CHANNELS) == 1)

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
        transform=build_image_pipeline(size=tuple(cfg.IMAGE.SIZE), scale=cfg.IMAGE.SCALE),
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
    # threshold are the registry's, which is why they are configured here rather than
    # in the library.
    table = derive_endpoint(cfg, table)
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
        num_classes=int(cfg.ENDPOINT.NUM_CLASSES),
    )


def check_endpoint(cfg) -> None:
    """Reject an endpoint that cannot be built, before any data is read.

    Raises:
        ValueError: If the class count, where the class is read from and the task
            disagree, naming the setting to change.
    """
    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    if num_classes < 2:
        raise ValueError(f"ENDPOINT.NUM_CLASSES must be at least 2, got {num_classes}")
    if cfg.ENDPOINT.REGRESSION:
        if cfg.ENDPOINT.LABEL_COLUMN or num_classes != 2:
            raise ValueError(
                "ENDPOINT.REGRESSION predicts ENDPOINT.VALUE_COLUMN itself and has no classes; leave "
                "ENDPOINT.LABEL_COLUMN empty and ENDPOINT.NUM_CLASSES at 2, or set ENDPOINT.REGRESSION False"
            )
        return
    if num_classes > 2 and not cfg.ENDPOINT.LABEL_COLUMN:
        raise ValueError(
            f"a threshold on {cfg.ENDPOINT.VALUE_COLUMN} defines two classes, not {num_classes}; set "
            f"ENDPOINT.LABEL_COLUMN to a column of class indices 0..{num_classes - 1}"
        )
    if num_classes > 2 and cfg.OBJECTIVE.POS_WEIGHT > 0:
        raise ValueError(
            f"OBJECTIVE.POS_WEIGHT weights the positive class of a binary endpoint, and this one has {num_classes} "
            f"classes; leave it 0 to weight every class from the training fold, or set it negative for no weighting"
        )


def endpoint_column(cfg) -> str:
    """The column of the cohort table the endpoint is read from."""
    return cfg.ENDPOINT.LABEL_COLUMN or cfg.ENDPOINT.VALUE_COLUMN


def derive_endpoint(cfg, table: pd.DataFrame) -> pd.DataFrame:
    """Add the ``label`` and ``value`` columns the endpoint is supervised and stratified on.

    The label is a threshold on ``ENDPOINT.VALUE_COLUMN``, or is read from
    ``ENDPOINT.LABEL_COLUMN`` where the cohort records the class directly. A subject with
    no recorded endpoint is dropped rather than assumed negative, and the run logs how
    many were.

    Raises:
        ValueError: If no subject has the endpoint, or a recorded class is not one of
            ``0 .. ENDPOINT.NUM_CLASSES - 1``.
    """
    column = endpoint_column(cfg)
    recorded = pd.to_numeric(table[column], errors="coerce")
    known = recorded.notna()
    if not bool(known.all()):
        logger.info("dropping %d subjects with no recorded %s", int((~known).sum()), column)
        table = table[known].reset_index(drop=True)
        recorded = recorded[known].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"no subject has a recorded {column}, so there is no endpoint to fine-tune on")

    table = table.copy()
    if not cfg.ENDPOINT.LABEL_COLUMN:
        table["value"] = recorded.astype(float)
        table["label"] = (recorded > cfg.ENDPOINT.THRESHOLD).astype(int)
        return table

    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    invalid = sorted(set(recorded[(recorded % 1 != 0) | (recorded < 0) | (recorded >= num_classes)].tolist()))
    if invalid:
        raise ValueError(
            f"{column} records {invalid[:5]}, which are not class indices 0..{num_classes - 1}; encode the classes "
            f"as integers from 0, or set ENDPOINT.NUM_CLASSES to how many there are"
        )
    table["label"] = recorded.astype(int)
    # The measurement, where the table has it, is kept for the record; it does not
    # supervise a classification read from a column.
    if cfg.ENDPOINT.VALUE_COLUMN in table:
        table["value"] = pd.to_numeric(table[cfg.ENDPOINT.VALUE_COLUMN], errors="coerce").astype(float)
    return table


def class_balance(table: pd.DataFrame, num_classes: int) -> dict:
    """How the labelled subjects divide between the classes, for a record or a log line."""
    labels = table["label"].to_numpy(dtype=int)
    if num_classes > 2:
        return {"class_counts": np.bincount(labels, minlength=num_classes).tolist()}
    return {"n_positive": int(labels.sum()), "positive_rate": float(labels.mean())}


def _balance_text(table: pd.DataFrame, num_classes: int) -> str:
    """The class balance as a phrase: ``"12 positive"`` or ``"classes 8/8/8"``."""
    balance = class_balance(table, num_classes)
    if "class_counts" in balance:
        return "classes " + "/".join(str(count) for count in balance["class_counts"])
    return f"{balance['n_positive']} positive"


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


def summarise(cohort: CardioVAECohort, endpoint: str) -> dict:
    """A record of what the cohort was, for the run's output directory."""
    return {
        "endpoint": endpoint,
        "synthetic": cohort.synthetic,
        "n_pretrain": len(cohort.pretrain_ids),
        "n_labelled": len(cohort.labelled_ids),
        **class_balance(cohort.table, cohort.num_classes),
        "ecg_shape": [cohort.ecg_channels, cohort.ecg_length],
        "image_size": list(cohort.image_size),
    }


def modality_names() -> Sequence[str]:
    """The modalities this example fuses."""
    return (ECG, CXR)
