"""Orchestration for the CardioVAE example.

Everything here is an experiment decision: where the cohort's files are, which column
defines the endpoint, how the folds are drawn, what gets written out. The model, the
objective, the trainer and the metrics all come from ``kalecardiac`` -- this module
assembles them and reports the result.

The workflow the example demonstrates:

.. code-block:: text

    chest X-ray + ECG
          |
    multimodal representation learning   (unlabelled, CardioVAE + tri-stream ELBO)
          |
    fine-tuning                          (labelled, frozen encoders + a clinical head)
          |
    cardiac haemodynamic prediction
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from examples.cardiovae_multimodal.config import CXR, ECG
from examples.cardiovae_multimodal.data import CardioVAECohort, load_cohort
from kalecardiac.config import parse_weights
from kalecardiac.evaluate import (
    MetricError,
    binary_metrics,
    predict_split,
    regression_metrics,
    roc_auc,
    save_predictions,
    summarise_folds,
)
from kalecardiac.interpret import attribution_ratios, modality_ablation, modality_attributions
from kalecardiac.loaddata import (
    CrossValidation,
    HoldOut,
    MultimodalDataset,
    collate_subjects,
    release_workers,
)
from kalecardiac.model.embed import CardioVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask, RegressionTask
from kalecardiac.utils import ensure_dir, load_checkpoint, set_seed, write_csv, write_json

logger = logging.getLogger("cardiovae_multimodal")


def build_model(cfg, cohort: CardioVAECohort) -> CardioVAE:
    """Construct the pretraining model from the configuration and the cohort's shapes."""
    return CardioVAE(
        signal_channels=cohort.ecg_channels,
        signal_length=cohort.ecg_length,
        image_channels=cfg.IMAGE.CHANNELS,
        image_size=tuple(cohort.image_size),
        latent_dim=cfg.MODEL.LATENT_DIM,
        channels=list(cfg.MODEL.CHANNELS),
        signal_encoder=cfg.ECG.ENCODER,
        # The image source scales pixels into the unit interval, so the decoder must be
        # able to produce that range and nothing outside it.
        image_output_activation="sigmoid" if cfg.IMAGE.SCALE else None,
        signal_name=ECG,
        image_name=CXR,
    )


def lightning_trainer(cfg, max_epochs: int, out_dir: Path, monitor: str | None = None, mode: str = "min"):
    """A Lightning trainer configured from the run settings."""
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
        logger=pl.loggers.CSVLogger(save_dir=str(out_dir), name="history"),
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def loader(dataset: MultimodalDataset, cfg, batch_size: int, shuffle: bool = False) -> DataLoader:
    """A DataLoader over a cohort split."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_subjects,
    )


def pretrain(cfg, cohort: CardioVAECohort, out_dir: Path) -> CardioVAE:
    """Learn a shared representation from the unlabelled pairs.

    No endpoint is used here. The point of the stage is that a large collection of
    radiographs and recordings with no catheter measurement is far easier to assemble
    than a labelled one, and the representation it yields transfers.
    """
    model = build_model(cfg, cohort)
    task = ReconstructionTask(
        weights=parse_weights(list(cfg.OBJECTIVE.RECONSTRUCTION_WEIGHTS)),
        scale_factor=cfg.OBJECTIVE.SCALE_FACTOR,
        annealing_epochs=cfg.OBJECTIVE.ANNEALING_EPOCHS,
        alignment_weight=cfg.OBJECTIVE.ALIGNMENT_WEIGHT,
        unimodal_streams=cfg.OBJECTIVE.UNIMODAL_STREAMS,
        # The radiograph is scaled into the unit interval and decoded through a
        # sigmoid, so cross-entropy is its natural reconstruction term; the recording
        # is standardised and real-valued, so squared error is.
        kinds={CXR: "bce"} if cfg.IMAGE.SCALE else {},
    )

    trainer = CardiacTrainer(
        model,
        task,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
        grad_clip=1.0,
    )

    dataset = MultimodalDataset(cohort.pretrain_ids, cohort.sources)
    train_loader = loader(dataset, cfg, cfg.SOLVER.BATCH_SIZE, shuffle=True)
    logger.info("pretraining on %d unlabelled subjects", len(dataset))
    lightning_trainer(cfg, cfg.SOLVER.MAX_EPOCHS, out_dir / "pretrain").fit(trainer, train_loader)
    release_workers(train_loader)

    torch.save(model.state_dict(), out_dir / "pretrained.pt")
    return model


def build_task(cfg, labels: Sequence[float]):
    """The downstream task: a threshold on the measurement, or the measurement itself."""
    if cfg.ENDPOINT.REGRESSION:
        return RegressionTask(hidden_dims=list(cfg.MODEL.HEAD_HIDDEN), dropout=cfg.MODEL.DROPOUT)

    pos_weight = cfg.OBJECTIVE.POS_WEIGHT
    if pos_weight == 0:
        # Derived from this fold's training labels, never from the whole cohort: a
        # class balance taken over the test split is a label leak.
        positives = float(np.sum(labels))
        pos_weight = (len(labels) - positives) / max(positives, 1.0)
    return ClassificationTask(
        pos_weight=max(pos_weight, 0.0),
        hidden_dims=list(cfg.MODEL.HEAD_HIDDEN),
        dropout=cfg.MODEL.DROPOUT,
    )


def fine_tune(
    cfg,
    cohort: CardioVAECohort,
    pretrained: CardioVAE,
    modalities: Sequence[str],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    out_dir: Path,
    fold: int,
) -> tuple[dict, CardiacTrainer, DataLoader]:
    """Fine-tune the pretrained encoders onto the endpoint, and score the held-out split."""
    sources = {name: cohort.sources[name] for name in modalities}
    labels = cohort.target.values_for(list(train_ids))
    supervision = list(labels[next(iter(labels))].numpy())
    task = build_task(cfg, supervision)

    predictor = MultimodalPredictor(
        pretrained.latent_embedders(modalities, frozen=cfg.MODEL.FREEZE_ENCODERS),
        task.build_head,
        method=cfg.FUSION.METHOD,
        fusion_dim=cfg.FUSION.FUSION_DIM or None,
        modality_dropout=cfg.FUSION.MODALITY_DROPOUT,
    )
    model = CardiacTrainer(
        predictor,
        task,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.FINETUNE.MAX_EPOCHS,
        init_lr=cfg.FINETUNE.BASE_LR,
    )

    def split_loader(identifiers, shuffle=False):
        return loader(
            MultimodalDataset(identifiers, sources, target=cohort.target),
            cfg,
            cfg.FINETUNE.BATCH_SIZE,
            shuffle=shuffle,
        )

    train_loader, val_loader, test_loader = (
        split_loader(train_ids, shuffle=True),
        split_loader(val_ids),
        split_loader(test_ids),
    )
    trainer = lightning_trainer(
        cfg, cfg.FINETUNE.MAX_EPOCHS, out_dir, monitor=model.monitor if len(val_ids) else None, mode=model.monitor_mode
    )
    trainer.fit(model, train_loader, val_loader if len(val_ids) else None)

    checkpoint = next((c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)), None)
    if checkpoint is not None and checkpoint.best_model_path:
        load_checkpoint(checkpoint.best_model_path, model, strict=False)

    predictions = predict_split(model, test_loader, split="test", fold=fold)
    metrics = score(cfg, predictions)
    release_workers(train_loader, val_loader)
    return {"predictions": predictions, "metrics": metrics}, model, test_loader


def score(cfg, predictions) -> dict:
    """Metrics for one fold, chosen by what the endpoint is."""
    if cfg.ENDPOINT.REGRESSION:
        return regression_metrics(predictions.targets["value"], predictions.scores, unit=cfg.ENDPOINT.UNIT)
    return binary_metrics(predictions.targets["label"], predictions.scores)


def interpret(cfg, model: CardiacTrainer, test_loader: DataLoader, out_dir: Path) -> dict:
    """What the model attended to, and what each modality was worth.

    Two questions rather than one: attribution says where in a recording the evidence
    was, and ablation says whether the model could have done without the modality
    entirely. They disagree when two modalities carry the same information.
    """
    summary: dict = {}
    batch = next(iter(test_loader))

    try:
        attributions = modality_attributions(model.model, batch.modalities, present=batch.present, n_steps=16)
        summary["attribution"] = attribution_ratios(attributions)
    except ImportError as error:
        logger.warning("skipping attribution: %s", error)

    if not cfg.ENDPOINT.REGRESSION and len(model.model.modalities) > 1:
        try:
            summary["ablation"] = modality_ablation(model, test_loader, roc_auc, "label")
        except MetricError as error:
            logger.warning("skipping ablation: %s", error)

    if summary:
        write_json(out_dir / "interpretation.json", summary)
    return summary


def splits(cfg, cohort: CardioVAECohort) -> list[tuple[int, list[str], list[str], list[str]]]:
    """The train/validation/test assignments this run evaluates.

    Cross-validation by default: a cardiac cohort of a few hundred subjects leaves too
    few positives in a single held-out split to separate one model from another.
    """
    frame = cohort.frame()
    if cfg.DATASET.SPLIT_MODE == "cv":
        splitter = CrossValidation(
            n_splits=cfg.DATASET.NUM_FOLDS,
            val_size=cfg.DATASET.VAL_RATIO,
            group_by=cfg.DATASET.GROUP_KEY,
            stratify_by=list(cfg.DATASET.STRATIFY_KEYS),
            random_state=cfg.SOLVER.SEED,
        )
    elif cfg.DATASET.SPLIT_MODE == "random":
        splitter = HoldOut(
            test_size=cfg.DATASET.TEST_RATIO,
            val_size=cfg.DATASET.VAL_RATIO,
            group_by=cfg.DATASET.GROUP_KEY,
            stratify_by=list(cfg.DATASET.STRATIFY_KEYS),
            random_state=cfg.SOLVER.SEED,
        )
    else:
        raise ValueError(
            f"unknown DATASET.SPLIT_MODE {cfg.DATASET.SPLIT_MODE!r}; this example supports 'cv' and 'random'. "
            f"A cohort that publishes its own assignment should read it and apply loaddata.Predefined."
        )

    identifiers = frame[cfg.DATASET.GROUP_KEY]
    return [
        (
            fold,
            sorted(identifiers.iloc[split["train"]]),
            sorted(identifiers.iloc[split["val"]]),
            sorted(identifiers.iloc[split["test"]]),
        )
        for fold, split in enumerate(splitter.split(frame))
    ]


def run(cfg, arms: Sequence[Sequence[str]] | None = None) -> dict:
    """Pretrain once, then fine-tune and score each modality arm over every fold.

    Args:
        cfg: The frozen configuration.
        arms: Modality combinations to compare. ``None`` compares each modality alone
            against the two fused, which is what makes the multimodal claim testable.

    Returns:
        Per-arm fold summaries, keyed by the arm's name.
    """
    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    cohort = load_cohort(cfg)
    logger.info("cohort: %s", cohort.describe())

    pretrained = pretrain(cfg, cohort, out_dir)
    arms = arms if arms is not None else [(ECG,), (CXR,), (ECG, CXR)]
    assignments = splits(cfg, cohort)

    results: dict = {}
    rows: list[dict] = []
    for arm in arms:
        name = "+".join(arm)
        arm_dir = ensure_dir(out_dir / name)
        fold_metrics, fold_predictions = [], []

        for fold, train_ids, val_ids, test_ids in assignments:
            # A fresh copy per fold: fine-tuning mutates the head and, when the
            # encoders are unfrozen, the encoders too, so reusing one model would carry
            # fold 0's test subjects into fold 1's training.
            model_copy = build_model(cfg, cohort)
            model_copy.load_state_dict(pretrained.state_dict())

            outcome, trained, test_loader = fine_tune(
                cfg, cohort, model_copy, arm, train_ids, val_ids, test_ids, arm_dir / f"fold_{fold}", fold
            )
            fold_metrics.append(outcome["metrics"])
            fold_predictions.append(outcome["predictions"])
            logger.info("%s fold %d: %s", name, fold, _headline(cfg, outcome["metrics"]))

            if cfg.OUTPUT.INTERPRET and fold == len(assignments) - 1:
                interpret(cfg, trained, test_loader, arm_dir)
            release_workers(test_loader)

        summary = summarise_folds(fold_metrics)
        save_predictions(arm_dir, fold_predictions, {"folds": fold_metrics, "summary": summary})
        results[name] = summary
        rows.append({"arm": name, **_flatten(summary)})
        logger.info("%s over %d folds: %s", name, len(fold_metrics), _headline(cfg, summary, folded=True))

    write_csv(out_dir / "results.csv", rows)
    write_json(out_dir / "summary.json", results)
    return results


def _headline(cfg, metrics: Mapping, folded: bool = False) -> str:
    """One line naming the metric that matters for this endpoint."""
    key = "rmse" if cfg.ENDPOINT.REGRESSION else "roc_auc"
    if not folded:
        return f"{key}={metrics.get(key, float('nan')):.3f}"
    entry = metrics.get(key, {})
    return f"{key}={entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}"


def _flatten(summary: Mapping) -> dict:
    """Fold summaries as flat columns, for a results table."""
    flat = {"n_folds": summary.get("n_folds")}
    for key, entry in summary.items():
        if isinstance(entry, Mapping) and "mean" in entry:
            flat[f"{key}_mean"] = entry["mean"]
            flat[f"{key}_std"] = entry["std"]
    return flat
