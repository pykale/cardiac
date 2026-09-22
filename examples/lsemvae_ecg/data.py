"""Building the LS-EMVAE cohorts: a large unlabelled one, and a small labelled one.

Two cohorts rather than one, because that asymmetry is the experiment's premise:
twelve-lead recordings with no endpoint are plentiful, and recordings paired with a
catheter measurement are not. Everything here is a decision those cohorts force; the
mechanisms all come from ``kalecardiac``.

The expected layout is in this example's README. ``DATASET.ROOT`` empty falls back to
:mod:`examples.synthetic_data`, so the workflow runs end to end with no data at all. A
synthetic run verifies the pipeline; it is not a result, and the log says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.loaddata import (
    ColumnTarget,
    ECGArraySource,
    ECGFileSource,
    ModalitySource,
    canonical_leads,
    lead_sources,
)
from kalecardiac.prepdata import (
    SignalPipeline,
    crop_or_pad,
    interpolate_missing,
    lead_selector,
    standardise,
)

logger = logging.getLogger("lsemvae_ecg")

#: The cohort table's identifier column, which is this dataset's vocabulary.
SUBJECT_COLUMN = "subject_id"


@dataclass
class LeadCohort:
    """One assembled lead-specific cohort.

    Attributes:
        pretrain_sources: One source per pretraining lead, keyed by canonical lead name.
        finetune_sources: One source per fine-tuning lead. A subset of the pretraining
            leads, which is what makes this a transfer experiment.
        target: Supervision for the labelled subjects.
        pretrain_ids: Subjects in the unlabelled pretraining cohort.
        labelled_ids: Subjects in the labelled cohort.
        table: One row per labelled subject, carrying the endpoint for stratification.
        length: Samples per lead after preprocessing.
        pretrain_leads: Lead names the pretraining model encodes, in expert order.
        finetune_leads: Lead names the downstream model reads.
        synthetic: Whether these cohorts were generated rather than read.
    """

    pretrain_sources: dict[str, ModalitySource]
    finetune_sources: dict[str, ModalitySource]
    target: ColumnTarget
    pretrain_ids: list[str]
    labelled_ids: list[str]
    table: pd.DataFrame
    length: int
    pretrain_leads: tuple[str, ...]
    finetune_leads: tuple[str, ...]
    synthetic: bool

    def frame(self) -> pd.DataFrame:
        """The table a splitter splits."""
        return self.table

    def describe(self) -> str:
        """One line summarising what was assembled."""
        positives = int(self.table["label"].sum()) if "label" in self.table else 0
        origin = "synthetic" if self.synthetic else "local"
        return (
            f"{origin}: {len(self.pretrain_ids)} unlabelled subjects x {len(self.pretrain_leads)} leads for "
            f"pretraining, {len(self.labelled_ids)} labelled ({positives} positive) x "
            f"{len(self.finetune_leads)} leads for fine-tuning | {self.length} samples per lead"
        )


def signal_pipeline(cfg) -> SignalPipeline:
    """Preprocessing for one recording: interpolate, crop or pad, standardise per lead.

    The order matters. Interpolation must come first because a NaN spreads through
    everything after it; standardisation must come last because statistics taken over
    samples that are then cropped away describe a recording the model never sees.
    """
    steps = [interpolate_missing, partial(crop_or_pad, length=cfg.ECG.LENGTH)]
    if cfg.ECG.STANDARDISE:
        steps.append(partial(standardise, per_channel=True))
    return SignalPipeline(steps)


def _source(cfg, recordings, available_leads, wanted_leads, paths=None) -> ECGArraySource:
    """One ECG source over exactly ``wanted_leads``, preprocessed and ready to split.

    Selection goes through the source's own ``select`` argument rather than through a
    transform, so that the source's ``leads`` keep describing what it returns -- which
    is what makes splitting it into per-lead modalities meaningful afterwards.

    Deriving an absent limb lead is different: it *creates* a lead rather than choosing
    one, so it runs before the source sees the recording, and the source is then told
    it holds the derived set.
    """
    wanted = canonical_leads(wanted_leads)
    pipeline = signal_pipeline(cfg)

    if cfg.ECG.DERIVE_MISSING:
        derive = lead_selector(available_leads, wanted, derive=True)
        if paths is not None:
            base_loader = _npy_loader
            return ECGFileSource(
                paths,
                leads=list(wanted),
                sampling_rate=cfg.ECG.SAMPLING_RATE,
                transform=pipeline,
                length=cfg.ECG.LENGTH,
                loader=lambda path: derive(base_loader(path)),
            )
        return ECGArraySource(
            {name: derive(value) for name, value in recordings.items()},
            leads=list(wanted),
            sampling_rate=cfg.ECG.SAMPLING_RATE,
            transform=pipeline,
            length=cfg.ECG.LENGTH,
        )

    common = {
        "leads": list(available_leads),
        "select": list(wanted),
        "sampling_rate": cfg.ECG.SAMPLING_RATE,
        "transform": pipeline,
        "length": cfg.ECG.LENGTH,
    }
    if paths is not None:
        return ECGFileSource(paths, **common)
    return ECGArraySource(recordings, **common)


def _npy_loader(path: Path) -> np.ndarray:
    """Read one recording, for the derive path to wrap."""
    return np.load(path)


def _lead_modalities(source: ECGArraySource, leads: Sequence[str]) -> dict[str, ModalitySource]:
    """Expose one modality per lead.

    The source already holds exactly these leads, so its channels line up with them and
    it splits directly. This is the whole of "each lead is a modality" at the loading
    stage; nothing downstream knows the modalities happen to be leads.
    """
    return lead_sources(source, leads=leads)


def load_local(cfg) -> LeadCohort:
    """Read both cohorts from ``DATASET.ROOT``.

    Expected layout, which the README documents in full:

    .. code-block:: text

        <ROOT>/
        |-- cohort.csv               subject_id and the endpoint column, labelled subjects
        |-- pretrain/<subject_id>.npy    (num_leads, num_samples), unlabelled
        `-- finetune/<subject_id>.npy    (num_leads, num_samples), labelled

    Raises:
        FileNotFoundError: If the table or a recording directory is absent.
        ValueError: If the table lacks the identifier or endpoint column.
    """
    root = Path(cfg.DATASET.ROOT)
    table_path = root / "cohort.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"no cohort table at {table_path}. See the example README for the expected layout; the cohorts "
            f"this example was developed on are private and are not distributed."
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

    pretrain_leads = canonical_leads(list(cfg.ECG.LEADS))
    finetune_leads = canonical_leads(list(cfg.ECG.FINETUNE_LEADS))

    pretrain_files = files("pretrain")
    finetune_files = files("finetune")
    pretrain = _source(cfg, None, pretrain_leads, pretrain_leads, paths=pretrain_files)
    finetune = _source(cfg, None, pretrain_leads, finetune_leads, paths=finetune_files)

    return _assemble(
        cfg, table, pretrain, finetune, sorted(pretrain_files), pretrain_leads, finetune_leads, synthetic=False
    )


def load_synthetic(cfg) -> LeadCohort:
    """Generate both cohorts, so the example runs with no data.

    The unlabelled cohort is drawn larger than the labelled one, mirroring the
    asymmetry the method exists to exploit.
    """
    pretrain_leads = canonical_leads(list(cfg.ECG.LEADS))
    finetune_leads = canonical_leads(list(cfg.ECG.FINETUNE_LEADS))

    unlabelled = make_synthetic_multimodal_data(
        num_subjects=cfg.SYNTHETIC.NUM_SUBJECTS * 2,
        leads=pretrain_leads,
        num_samples=cfg.SYNTHETIC.NUM_SAMPLES,
        sampling_rate=cfg.ECG.SAMPLING_RATE,
        with_images=False,
        seed=cfg.SOLVER.SEED,
    )
    labelled = make_synthetic_multimodal_data(
        num_subjects=cfg.SYNTHETIC.NUM_SUBJECTS,
        leads=pretrain_leads,
        num_samples=cfg.SYNTHETIC.NUM_SAMPLES,
        sampling_rate=cfg.ECG.SAMPLING_RATE,
        with_images=False,
        seed=cfg.SOLVER.SEED + 100,
    )

    pretrain = _source(cfg, unlabelled.ecg, pretrain_leads, pretrain_leads)
    finetune = _source(cfg, labelled.ecg, pretrain_leads, finetune_leads)

    table = pd.DataFrame(
        {
            SUBJECT_COLUMN: labelled.subject_id,
            cfg.ENDPOINT.VALUE_COLUMN: [labelled.values[name] for name in labelled.subject_id],
        }
    )
    return _assemble(
        cfg, table, pretrain, finetune, list(unlabelled.subject_id), pretrain_leads, finetune_leads, synthetic=True
    )


def _assemble(
    cfg,
    table: pd.DataFrame,
    pretrain: ECGArraySource,
    finetune: ECGArraySource,
    pretrain_ids: Sequence[str],
    pretrain_leads: tuple[str, ...],
    finetune_leads: tuple[str, ...],
    synthetic: bool,
) -> LeadCohort:
    """Split both sources into per-lead modalities and derive the endpoint."""
    available = set(finetune.recordings if hasattr(finetune, "recordings") else {})
    if hasattr(finetune, "paths"):
        available = set(finetune.paths)

    table = table[table[SUBJECT_COLUMN].isin(available)].reset_index(drop=True)
    if table.empty:
        raise ValueError("no labelled subject survived the join between the cohort table and the recordings")

    values = pd.to_numeric(table[cfg.ENDPOINT.VALUE_COLUMN], errors="coerce")
    known = values.notna()
    if not bool(known.all()):
        logger.info("dropping %d subjects with no recorded %s", int((~known).sum()), cfg.ENDPOINT.VALUE_COLUMN)
        table = table[known].reset_index(drop=True)
        values = values[known].reset_index(drop=True)

    table["value"] = values.astype(float)
    table["label"] = (values > cfg.ENDPOINT.THRESHOLD).astype(int)
    columns = {"value": "value"} if cfg.ENDPOINT.REGRESSION else {"label": "label"}

    return LeadCohort(
        pretrain_sources=_lead_modalities(pretrain, pretrain_leads),
        finetune_sources=_lead_modalities(finetune, finetune_leads),
        target=ColumnTarget(table, columns=columns, id_column=SUBJECT_COLUMN),
        pretrain_ids=list(pretrain_ids),
        labelled_ids=list(table[SUBJECT_COLUMN]),
        table=table,
        length=finetune.length,
        pretrain_leads=pretrain_leads,
        finetune_leads=finetune_leads,
        synthetic=synthetic,
    )


def load_cohort(cfg) -> LeadCohort:
    """Read the cohorts from disk, or generate them when no root is configured."""
    if cfg.DATASET.ROOT:
        return load_local(cfg)
    logger.warning(
        "DATASET.ROOT is empty, so this run uses synthetic data. It verifies that the pipeline executes; "
        "it is not a result."
    )
    return load_synthetic(cfg)


def summarise(cohort: LeadCohort, endpoint: str) -> dict:
    """A record of what the cohorts were, for the run's output directory."""
    return {
        "endpoint": endpoint,
        "synthetic": cohort.synthetic,
        "n_pretrain": len(cohort.pretrain_ids),
        "n_labelled": len(cohort.labelled_ids),
        "n_positive": int(cohort.table["label"].sum()),
        "positive_rate": float(cohort.table["label"].mean()),
        "pretrain_leads": list(cohort.pretrain_leads),
        "finetune_leads": list(cohort.finetune_leads),
        "samples_per_lead": cohort.length,
    }
