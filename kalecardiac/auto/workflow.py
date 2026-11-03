"""Pretraining, fine-tuning over folds, and interpretation, each read from one configuration.

What every cardiac workflow here does between assembling a cohort and reporting on it:
learn a representation without labels (or load one), fine-tune a copy of it onto the
endpoint for each fold, score the held-out split, and explain the last fold's model. The
model and the cohort are the experiment's; how a fold is trained and scored is read from
``cfg``, once, so two experiments cannot drift apart on it.

Pretrain, then cross-validate one comparison arm::

    pretrained = pretrain(cfg, model, MultimodalDataset(unlabelled_ids, sources), out_dir)
    folds = get_folds(cfg, table)
    summary = cross_validate(cfg, pretrained, sources, target, folds, out_dir / "arm")
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from yacs.config import CfgNode

from kalecardiac.auto.endpoint import format_headline, get_ablation_metric, get_prediction_task, score_predictions
from kalecardiac.config import parse_weights
from kalecardiac.evaluate import MetricError, SplitPredictions, predict_split, save_predictions, summarize_folds
from kalecardiac.interpret import attribution_ratios, modality_ablation, modality_attributions
from kalecardiac.loaddata import (
    BaseSplitter,
    CrossValidationSplit,
    HoldOutSplit,
    ModalitySource,
    MultimodalDataset,
    Target,
    collate_subjects,
    release_workers,
)
from kalecardiac.model.embed import MultimodalPredictor, MultimodalVAE
from kalecardiac.pipeline import CardiacTrainer, ReconstructionTask
from kalecardiac.prepdata import get_ecg_transform
from kalecardiac.utils import ensure_dir, load_checkpoint, preserve_rng_state, write_csv, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One assignment of subjects to training, validation and test, by identifier.

    Attributes:
        index: The fold's number, from 0.
        train_ids: Subjects the fold trains on.
        val_ids: Subjects carved out of the training half to select a checkpoint on.
        test_ids: Subjects the fold is scored on.
    """

    index: int
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


@dataclass
class FoldResult:
    """One fold's fine-tuned model, and how it scored on the fold's test split.

    Attributes:
        model: The fine-tuned trainer, with its best validation checkpoint restored.
        test_loader: The fold's test split, for a report that rescores it.
        predictions: What the model predicted for each test subject.
        metrics: The test split's metrics, from
            :func:`~kalecardiac.auto.score_predictions`.
    """

    model: CardiacTrainer
    test_loader: DataLoader
    predictions: SplitPredictions
    metrics: dict


def check_model_name(cfg: CfgNode, expected: str, hint: str = "") -> None:
    """Reject a ``MODEL.NAME`` other than the one a workflow builds, before anything runs.

    A configuration copied from another experiment then fails at start-up rather than
    quietly training the wrong model.

    Args:
        cfg: The run configuration.
        expected: The name of the model the caller builds.
        hint: What to run instead, put before the message's own advice.

    Raises:
        ValueError: If ``MODEL.NAME`` is not ``expected``.
    """
    if expected != cfg.MODEL.NAME:
        advice = f"{hint} " if hint else ""
        raise ValueError(
            f"MODEL.NAME is {cfg.MODEL.NAME!r}, but this workflow builds {expected!r}. {advice}To ablate the "
            f"fusion, keep MODEL.NAME and set FUSION.LATENT_METHOD."
        )


def get_source_sampling_rate(cfg: CfgNode) -> float:
    """Rate the recordings arrive at: ``ECG.SOURCE_SAMPLING_RATE``, or the model's own rate if that is 0."""
    return float(cfg.ECG.SOURCE_SAMPLING_RATE or cfg.ECG.SAMPLING_RATE)


<<<<<<< HEAD
def get_ecg_transform_from_cfg(cfg: CfgNode, length: int | None = None, num_leads: int | None = None) -> Compose:
    """:func:`~kalecardiac.prepdata.get_ecg_transform`, with its arguments read from ``cfg.ECG``.

    With ``ECG.NUM_CHANNELS`` 1 the leads are interleaved into one channel once every
    per-lead step is done, and the length is the strip's: each lead is cropped or padded
    to its share of it.

    Args:
        cfg: The run configuration.
        length: Samples per channel to produce; ``ECG.LENGTH`` by default.
        num_leads: Leads each recording reaches the transform with; ``len(ECG.LEADS)``
            by default. Read only to interleave them.

    Returns:
        Interpolate, resample from the source rate, crop or pad, then normalize each
        lead -- or, with ``ECG.NORMALIZE`` off, keep the recording's own units -- and
        interleave the leads if ``ECG.NUM_CHANNELS`` is 1.

    Raises:
        ValueError: If ``ECG.NUM_CHANNELS`` is neither 0, 1 nor the number of leads, or
            the strip's length is not a whole number of samples per lead.
    """
    length = cfg.ECG.LENGTH if length is None else length
    num_leads = len(cfg.ECG.LEADS) if num_leads is None else num_leads
    channels = int(cfg.ECG.NUM_CHANNELS)
    if channels not in (0, 1, num_leads):
        raise ValueError(
            f"ECG.NUM_CHANNELS is {channels}, but the recordings hold {num_leads} leads: set it to 1 to interleave "
            f"them into one channel, or to 0 for a channel per lead"
        )
    interleave = channels == 1 and num_leads > 1
    if interleave:
        if length % num_leads:
            raise ValueError(
                f"ECG.LENGTH {length} is not a whole number of samples for each of {num_leads} leads; set it "
                f"to the leads times the samples per lead, e.g. {num_leads * 5000} for ten seconds at 500 Hz"
            )
        length //= num_leads
    return get_ecg_transform(
        length=length,
=======
def get_ecg_transform_from_cfg(cfg: CfgNode, length: int | None = None) -> Compose:
    """:func:`~kalecardiac.prepdata.get_ecg_transform`, with its arguments read from ``cfg.ECG``.

    Args:
        cfg: The run configuration.
        length: Samples per lead to crop or pad to; ``ECG.LENGTH`` by default. A model
            that reads the leads as one channel crops each lead to its share.

    Returns:
        Interpolate, resample from the source rate, crop or pad, then normalize each
        lead -- or, with ``ECG.NORMALIZE`` off, keep the recording's own units.
    """
    return get_ecg_transform(
        length=cfg.ECG.LENGTH if length is None else length,
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        sampling_rate=cfg.ECG.SAMPLING_RATE,
        source_sampling_rate=get_source_sampling_rate(cfg),
        normalize_leads=cfg.ECG.NORMALIZE,
        # get_ecg_transform wants the choice stated; NORMALIZE off states it as a scale
        # of one, which leaves the recording in its own units.
        amplitude_scale=None if cfg.ECG.NORMALIZE else 1.0,
<<<<<<< HEAD
        interleave=interleave,
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    )


def get_dataloader(cfg: CfgNode, dataset: MultimodalDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """A loader that batches ``dataset``'s subjects, with ``DATASET.NUM_WORKERS`` workers."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_subjects,
    )


def get_folds(cfg: CfgNode, table: pd.DataFrame) -> list[Fold]:
    """The folds ``DATASET`` describes over a cohort's table.

    ``DATASET.SPLIT_MODE`` ``"cv"`` gives ``NUM_FOLDS``-fold cross-validation and
    ``"random"`` one held-out split, drawn by subject, stratified and seeded as
    :class:`~kalecardiac.loaddata.CrossValidationSplit` and
    :class:`~kalecardiac.loaddata.HoldOutSplit` draw them, with validation carved out of
    each training half.

    Args:
        cfg: The run configuration.
        table: One row per subject, with the ``DATASET.GROUP_KEY`` column and every
            ``DATASET.STRATIFY_KEYS`` column.

    Returns:
        One :class:`Fold` per split, its identifiers sorted.

    Raises:
        ValueError: If ``DATASET.SPLIT_MODE`` is neither, naming what to do instead.
    """
    common: dict[str, Any] = {
        "val_size": cfg.DATASET.VAL_RATIO,
        "group_by": cfg.DATASET.GROUP_KEY,
        "stratify_by": list(cfg.DATASET.STRATIFY_KEYS),
        "random_state": cfg.SOLVER.SEED,
    }
    splitter: BaseSplitter
    if cfg.DATASET.SPLIT_MODE == "cv":
        splitter = CrossValidationSplit(n_splits=cfg.DATASET.NUM_FOLDS, **common)
    elif cfg.DATASET.SPLIT_MODE == "random":
        splitter = HoldOutSplit(test_size=cfg.DATASET.TEST_RATIO, **common)
    else:
        raise ValueError(
            f"DATASET.SPLIT_MODE is {cfg.DATASET.SPLIT_MODE!r}, and folds are drawn for 'cv' and 'random'. A cohort "
            f"that publishes its own assignment should apply loaddata.PredefinedSplit to its table instead."
        )

    identifiers = table[cfg.DATASET.GROUP_KEY]
    return [
        Fold(
            index,
            sorted(identifiers.iloc[split["train"]]),
            sorted(identifiers.iloc[split["val"]]),
            sorted(identifiers.iloc[split["test"]]),
        )
        for index, split in enumerate(splitter.split(table))
    ]


def pretrain(
    cfg: CfgNode,
    model: MultimodalVAE,
    dataset: MultimodalDataset,
    out_dir: str | Path,
    kinds: Mapping[str, str] | None = None,
) -> MultimodalVAE:
    """Learn a representation from unlabelled subjects -- or load it, if ``MODEL.PRETRAINED`` names one.

    No endpoint is read: the objective is the model's own ELBO, weighted by ``OBJECTIVE``
    and trained as ``SOLVER`` says. The weights are written to ``<out_dir>/pretrained.pt``,
    which a later run can name as ``MODEL.PRETRAINED``.

    A checkpoint is loaded strictly, so one saved from another architecture -- another
    latent width, another kernel -- fails here rather than half-loading and fine-tuning
    encoders that are partly random.

    Args:
        cfg: The run configuration.
        model: The model to train, or to load the checkpoint into.
        dataset: The subjects to pretrain on. Their targets, if any, are not read.
        out_dir: Where the training log and the weights are written.
        kinds: Reconstruction loss per modality, as
            :class:`~kalecardiac.pipeline.ReconstructionTask` takes them; squared error
            for any modality not named.

    Returns:
        ``model``, trained or loaded.

    Raises:
        FileNotFoundError: If ``MODEL.PRETRAINED`` names a file that does not exist.
        RuntimeError: If its weights do not fit ``model``.
    """
    if cfg.MODEL.PRETRAINED:
        load_checkpoint(cfg.MODEL.PRETRAINED, model, strict=True)
        logger.info("loaded the pretrained representation from %s; pretraining skipped", cfg.MODEL.PRETRAINED)
        return model

    out_dir = ensure_dir(out_dir)
    task = ReconstructionTask(
        weights=parse_weights(list(cfg.OBJECTIVE.RECONSTRUCTION_WEIGHTS)),
        scale_factor=cfg.OBJECTIVE.SCALE_FACTOR,
        annealing_epochs=cfg.OBJECTIVE.ANNEALING_EPOCHS,
        alignment_weight=cfg.OBJECTIVE.ALIGNMENT_WEIGHT,
        unimodal_streams=cfg.OBJECTIVE.UNIMODAL_STREAMS,
        kinds=kinds,
    )
    trainer = CardiacTrainer(
        model,
        task,
        optimizer=_optimizer(cfg),
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
        grad_clip=cfg.SOLVER.GRAD_CLIP,
    )
    loader = get_dataloader(cfg, dataset, cfg.SOLVER.BATCH_SIZE, shuffle=True)
    logger.info("pretraining on %d unlabelled subjects across %d modalities", len(dataset), len(model.modalities))
    _lightning_trainer(cfg, cfg.SOLVER.MAX_EPOCHS, out_dir / "pretrain").fit(trainer, loader)
    release_workers(loader)

    torch.save(model.state_dict(), out_dir / "pretrained.pt")
    return model


def fine_tune(
    cfg: CfgNode,
    pretrained: MultimodalVAE,
    sources: Mapping[str, ModalitySource],
    target: Target,
    fold: Fold,
    out_dir: str | Path,
) -> FoldResult:
    """Fine-tune ``pretrained``'s encoders onto the endpoint for one fold, and score its test split.

    The head, the loss and any class weight come from
    :func:`~kalecardiac.auto.get_prediction_task`, over this fold's training labels
    alone. Given validation subjects, the checkpoint that scored best on them is restored
    before the test split is scored.

    ``pretrained`` is changed: its encoders become the predictor's, and with
    ``MODEL.FREEZE_ENCODERS`` off they are trained. Pass a copy to keep the original, as
    :func:`cross_validate` does.

    Args:
        cfg: The run configuration.
        pretrained: The representation to start from.
        sources: The modalities to fine-tune on, by name; a subset of the model's.
        target: Supervision for the fold's subjects.
        fold: Which subjects train, validate and test.
        out_dir: Where the fold's training log and best checkpoint are written.

    Returns:
        The fine-tuned model, its test loader, and its test split's predictions and metrics.
    """
    labels = target.values_for(list(fold.train_ids))
    task = get_prediction_task(cfg, next(iter(labels.values())).numpy())
    predictor = MultimodalPredictor(
        pretrained.latent_embedders(sources, frozen=cfg.MODEL.FREEZE_ENCODERS),
        task.build_head,
        method=cfg.FUSION.METHOD,
        fusion_dim=cfg.FUSION.FUSION_DIM or None,
        modality_dropout=cfg.FUSION.MODALITY_DROPOUT,
    )
    model = CardiacTrainer(
        predictor,
        task,
        optimizer=_optimizer(cfg),
        max_epochs=cfg.FINETUNE.MAX_EPOCHS,
        init_lr=cfg.FINETUNE.BASE_LR,
    )

    train_loader, val_loader, test_loader = (
        get_dataloader(cfg, MultimodalDataset(identifiers, sources, target=target), cfg.FINETUNE.BATCH_SIZE, shuffle)
        for identifiers, shuffle in ((fold.train_ids, True), (fold.val_ids, False), (fold.test_ids, False))
    )
    validated = len(fold.val_ids) > 0
    trainer = _lightning_trainer(
        cfg,
        cfg.FINETUNE.MAX_EPOCHS,
        Path(out_dir),
        monitor=model.monitor if validated else None,
        mode=model.monitor_mode,
    )
    trainer.fit(model, train_loader, val_loader if validated else None)

    checkpoint = trainer.checkpoint_callback
    if isinstance(checkpoint, pl.callbacks.ModelCheckpoint) and checkpoint.best_model_path:
        load_checkpoint(checkpoint.best_model_path, model, strict=False)

    predictions = predict_split(model, test_loader, split="test", fold=fold.index)
    release_workers(train_loader, val_loader)
    return FoldResult(model, test_loader, predictions, score_predictions(cfg, predictions))


def cross_validate(
    cfg: CfgNode,
    pretrained: MultimodalVAE,
    sources: Mapping[str, ModalitySource],
    target: Target,
    folds: Sequence[Fold],
    out_dir: str | Path,
    report: Callable[[Fold, FoldResult], None] | None = None,
) -> dict:
    """Fine-tune a fresh copy of ``pretrained`` on every fold, score each, and summarize the folds.

    A copy per fold, because fine-tuning changes the head and, with
    ``MODEL.FREEZE_ENCODERS`` off, the encoders too: reusing one model would carry one
    fold's test subjects into the next fold's training.

    Reports run off the training's random stream -- the last fold's
    :func:`interpret_model` when ``OUTPUT.INTERPRET`` is set, and ``report`` after every
    fold -- so turning one on or off cannot change the models trained after it.

    Writes to ``out_dir`` the test splits' ``predictions.csv`` and ``metrics.json``, a
    ``fold_<i>/`` directory of training logs per fold, and the interpretation's files.

    Args:
        cfg: The run configuration.
        pretrained: The representation every fold starts from. It is not changed.
        sources: The modalities to fine-tune on, by name.
        target: Supervision for every subject in ``folds``.
        folds: From :func:`get_folds`.
        out_dir: Where this comparison's results are written.
        report: Called with each fold and its result, before the fold's test loader is
            released, for a report that rescores the test split.

    Returns:
        The folds' metrics, summarized by :func:`~kalecardiac.evaluate.summarize_folds`.
    """
    out_dir = ensure_dir(out_dir)
    name = "+".join(sources)
    fold_metrics, fold_predictions = [], []
    for fold in folds:
        result = fine_tune(cfg, copy.deepcopy(pretrained), sources, target, fold, out_dir / f"fold_{fold.index}")
        fold_metrics.append(result.metrics)
        fold_predictions.append(result.predictions)
        logger.info("%s fold %d: %s", name, fold.index, format_headline(cfg, result.metrics))

        with preserve_rng_state():
            if cfg.OUTPUT.INTERPRET and fold is folds[-1]:
                interpret_model(cfg, result.model, result.test_loader, out_dir)
            if report is not None:
                report(fold, result)
        release_workers(result.test_loader)

    summary = summarize_folds(fold_metrics)
    save_predictions(out_dir, fold_predictions, {"folds": fold_metrics, "summary": summary})
    logger.info("%s over %d folds: %s", name, len(fold_metrics), format_headline(cfg, summary, folded=True))
    return summary


def interpret_model(cfg: CfgNode, model: CardiacTrainer, test_loader: DataLoader, out_dir: str | Path) -> dict:
    """What a fine-tuned model attended to, and what each modality was worth to it.

    Two questions rather than one. Attribution -- integrated gradients over the first
    test batch -- says where in each modality the evidence was; ablation -- each modality
    withheld and the split rescored, in :func:`~kalecardiac.auto.get_ablation_metric` --
    says whether the model could have done without it. They disagree when two modalities
    carry the same information, which is the interesting case. Ablation needs a second
    modality to fall back on, so a one-modality model has none.

    Writes ``interpretation.json`` to ``out_dir``, and ``attribution.csv``, one row per
    modality, when attribution ran.

    Args:
        cfg: The run configuration.
        model: The fine-tuned trainer.
        test_loader: The split to explain.
        out_dir: Where the files are written.

    Returns:
        ``{"attribution": ..., "ablation": ...}``, each where it could be computed.
    """
    summary: dict = {}
    batch = next(iter(test_loader))
    try:
        attributions = modality_attributions(
            model.model, batch.modalities, present=batch.present, n_steps=cfg.OUTPUT.ATTRIBUTION_STEPS
        )
        summary["attribution"] = attribution_ratios(attributions, threshold=cfg.OUTPUT.ATTRIBUTION_THRESHOLD)
    except ImportError as error:
        logger.warning("skipping attribution: %s", error)

    if len(model.model.modalities) > 1:
        _, metric, key = get_ablation_metric(cfg)
        try:
            summary["ablation"] = modality_ablation(model, test_loader, metric, key)
        except MetricError as error:
            logger.warning("skipping ablation: %s", error)

    out_dir = ensure_dir(out_dir)
    if "attribution" in summary:
        write_csv(
            out_dir / "attribution.csv",
            [{"modality": name, **entry} for name, entry in summary["attribution"].items()],
        )
    if summary:
        write_json(out_dir / "interpretation.json", summary)
    return summary


def _optimizer(cfg: CfgNode) -> dict:
    """The optimizer a :class:`~kalecardiac.pipeline.CardiacTrainer` takes, from ``SOLVER``."""
    return {"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}}


def _lightning_trainer(
    cfg: CfgNode, max_epochs: int, out_dir: Path, monitor: str | None = None, mode: str = "min"
) -> pl.Trainer:
<<<<<<< HEAD
    """A Lightning trainer configured from ``SOLVER``, checkpointing on ``monitor`` if one is given.

    Without a ``monitor`` there is no best checkpoint to keep, so none is written: Lightning
    would otherwise save the model and its optimizer state after every epoch -- gigabytes,
    for a model the size of the published CardioVAE -- which the caller then never reads.
    """
=======
    """A Lightning trainer configured from ``SOLVER``, checkpointing on ``monitor`` if one is given."""
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    callbacks: list[pl.Callback] = []
    if monitor is not None:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor=monitor, mode=mode)
        )
        if cfg.SOLVER.EARLY_STOP:
            callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=cfg.SOLVER.EARLY_STOP))

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.SOLVER.ACCELERATOR,
        devices=cfg.SOLVER.DEVICES,
        callbacks=callbacks,
<<<<<<< HEAD
        enable_checkpointing=monitor is not None,
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        logger=pl.loggers.CSVLogger(save_dir=str(out_dir), name="history"),
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
