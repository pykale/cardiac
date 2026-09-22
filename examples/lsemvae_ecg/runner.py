"""Orchestration for the LS-EMVAE example.

Everything here is an experiment decision. The model, the hierarchical fusion, the
alignment objective, the trainer and the metrics all come from ``kalecardiac``.

The workflow the example demonstrates:

.. code-block:: text

    twelve-lead ECG, unlabelled
          |
    one encoder per lead -> hierarchical latent fusion   (LS-EMVAE pretraining)
          |
    fine-tuning on the six limb leads a low-cost recorder captures
          |
    cardiovascular prediction
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from examples.lsemvae_ecg.data import LeadCohort, load_cohort, summarise
from kalecardiac.evaluate import (
    MetricError,
    binary_metrics,
    predict_split,
    regression_metrics,
    roc_auc,
    save_predictions,
    summarise_folds,
)
from kalecardiac.interpret import (
    attribution_ratios,
    collect_latents,
    latent_embedding,
    modality_ablation,
    modality_attributions,
)
from kalecardiac.loaddata import CrossValidation, HoldOut, MultimodalDataset, collate_subjects, release_workers
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask, RegressionTask
from kalecardiac.utils import ensure_dir, load_checkpoint, set_seed, write_csv, write_json

logger = logging.getLogger("lsemvae_ecg")


def build_model(cfg, cohort: LeadCohort, leads: Sequence[str]) -> LSEMVAE:
    """Construct a lead-specific VAE over ``leads``."""
    return LSEMVAE(
        leads=list(leads),
        length=cohort.length,
        latent_dim=cfg.MODEL.LATENT_DIM,
        channels=list(cfg.MODEL.CHANNELS),
        encoder=cfg.ECG.ENCODER,
        fusion=cfg.FUSION.LATENT_METHOD,
        num_groups=cfg.FUSION.NUM_GROUPS,
        shared_decoder=cfg.MODEL.SHARED_DECODER,
        use_prior=cfg.FUSION.USE_PRIOR,
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


def pretrain(cfg, cohort: LeadCohort, out_dir: Path) -> LSEMVAE:
    """Learn lead-specific representations from the unlabelled twelve-lead cohort.

    No endpoint is used. The alignment term is what makes the result transferable: it
    pulls each lead's posterior towards the joint one, so a lead's representation stays
    readable when the other leads are absent -- which is exactly the situation the
    six-lead fine-tuning creates.
    """
    model = build_model(cfg, cohort, cohort.pretrain_leads)
    task = ReconstructionTask(
        scale_factor=cfg.OBJECTIVE.SCALE_FACTOR,
        annealing_epochs=cfg.OBJECTIVE.ANNEALING_EPOCHS,
        alignment_weight=cfg.OBJECTIVE.ALIGNMENT_WEIGHT,
        unimodal_streams=cfg.OBJECTIVE.UNIMODAL_STREAMS,
    )
    trainer = CardiacTrainer(
        model,
        task,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
        grad_clip=1.0,
    )

    dataset = MultimodalDataset(cohort.pretrain_ids, cohort.pretrain_sources)
    train_loader = loader(dataset, cfg, cfg.SOLVER.BATCH_SIZE, shuffle=True)
    logger.info("pretraining on %d unlabelled subjects across %d leads", len(dataset), len(cohort.pretrain_leads))
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
        # From this fold's training labels only: a class balance taken over the whole
        # cohort carries the test split's composition into the loss.
        positives = float(np.sum(labels))
        pos_weight = (len(labels) - positives) / max(positives, 1.0)
    return ClassificationTask(
        pos_weight=max(pos_weight, 0.0),
        hidden_dims=list(cfg.MODEL.HEAD_HIDDEN),
        dropout=cfg.MODEL.DROPOUT,
    )


def fine_tune(
    cfg,
    cohort: LeadCohort,
    pretrained: LSEMVAE,
    leads: Sequence[str],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    out_dir: Path,
    fold: int,
) -> tuple[dict, CardiacTrainer, DataLoader]:
    """Fine-tune the pretrained lead encoders onto the endpoint, and score the test split."""
    sources = {lead: cohort.finetune_sources[lead] for lead in leads}
    labels = cohort.target.values_for(list(train_ids))
    task = build_task(cfg, list(labels[next(iter(labels))].numpy()))

    predictor = MultimodalPredictor(
        pretrained.latent_embedders(leads, frozen=cfg.MODEL.FREEZE_ENCODERS),
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
    release_workers(train_loader, val_loader)
    return {"predictions": predictions, "metrics": score(cfg, predictions)}, model, test_loader


def score(cfg, predictions) -> dict:
    """Metrics for one fold, chosen by what the endpoint is."""
    if cfg.ENDPOINT.REGRESSION:
        return regression_metrics(predictions.targets["value"], predictions.scores, unit=cfg.ENDPOINT.UNIT)
    return binary_metrics(predictions.targets["label"], predictions.scores)


def interpret(cfg, model: CardiacTrainer, test_loader: DataLoader, out_dir: Path) -> dict:
    """Which leads the model used, by attribution and by ablation.

    Both are reported because they answer different questions. A lead can be heavily
    attributed and still be redundant -- if another lead carries the same information,
    removing it costs nothing -- and that disagreement is the interesting part.
    """
    summary: dict = {}
    batch = next(iter(test_loader))

    try:
        attributions = modality_attributions(model.model, batch.modalities, present=batch.present, n_steps=16)
        summary["attribution"] = attribution_ratios(attributions, threshold=cfg.OUTPUT.ATTRIBUTION_THRESHOLD)
        write_csv(
            out_dir / "lead_importance.csv",
            [{"lead": lead, **entry} for lead, entry in summary["attribution"].items()],
        )
    except ImportError as error:
        logger.warning("skipping attribution: %s", error)

    if not cfg.ENDPOINT.REGRESSION:
        try:
            summary["ablation"] = modality_ablation(model, test_loader, roc_auc, "label")
        except MetricError as error:
            logger.warning("skipping ablation: %s", error)

    if summary:
        write_json(out_dir / "interpretation.json", summary)
    return summary


def export_latents(cfg, pretrained: LSEMVAE, cohort: LeadCohort, out_dir: Path) -> None:
    """Project the pretrained latent space, and record how far the leads agree.

    The only check available before any label: whether the representation has structure,
    and whether the lead encoders ended up in the same coordinates.
    """
    dataset = MultimodalDataset(cohort.pretrain_ids, cohort.pretrain_sources)
    data_loader = loader(dataset, cfg, cfg.SOLVER.BATCH_SIZE)
    identifiers, joint, per_lead = collect_latents(pretrained, data_loader, per_modality=True)
    release_workers(data_loader)

    coordinates = latent_embedding(joint, method="pca", random_state=cfg.SOLVER.SEED)
    write_csv(
        out_dir / "latent_space.csv",
        [
            {"subject_id": identifier, "x": float(x), "y": float(y)}
            for identifier, (x, y) in zip(identifiers, coordinates, strict=True)
        ],
    )
    write_json(
        out_dir / "lead_agreement.json",
        {lead: float(np.mean((values - joint) ** 2)) for lead, values in per_lead.items()},
    )


def splits(cfg, cohort: LeadCohort) -> list[tuple[int, list[str], list[str], list[str]]]:
    """The train/validation/test assignments this run evaluates."""
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
            f"unknown DATASET.SPLIT_MODE {cfg.DATASET.SPLIT_MODE!r}; this example supports 'cv' and 'random'."
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


def run(cfg, lead_groups: Sequence[Sequence[str]] | None = None) -> dict:
    """Pretrain on the full lead set, then fine-tune and score each lead group.

    Args:
        cfg: The frozen configuration.
        lead_groups: Lead subsets to compare. ``None`` evaluates the configured
            fine-tuning set alone, which is the headline result; passing several makes
            the lead-configuration comparison the paper reports.

    Returns:
        Per-group fold summaries, keyed by the group's name.
    """
    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    cohort = load_cohort(cfg)
    logger.info("cohort: %s", cohort.describe())
    write_json(out_dir / "cohort.json", summarise(cohort, cfg.ENDPOINT.NAME))

    pretrained = pretrain(cfg, cohort, out_dir)
    if cfg.OUTPUT.INTERPRET:
        export_latents(cfg, pretrained, cohort, out_dir)

    groups = [tuple(cohort.finetune_leads)] if lead_groups is None else [tuple(group) for group in lead_groups]
    assignments = splits(cfg, cohort)

    results: dict = {}
    rows: list[dict] = []
    for group in groups:
        unknown = sorted(set(group) - set(cohort.finetune_sources))
        if unknown:
            raise ValueError(f"lead group names {unknown}, which the labelled cohort does not record")

        name = "+".join(group)
        group_dir = ensure_dir(out_dir / name)
        fold_metrics, fold_predictions = [], []

        for fold, train_ids, val_ids, test_ids in assignments:
            # A fresh copy per fold: fine-tuning mutates the head, so reusing one model
            # would carry fold 0's test subjects into fold 1's training.
            copy = build_model(cfg, cohort, cohort.pretrain_leads)
            copy.load_state_dict(pretrained.state_dict())

            outcome, trained, test_loader = fine_tune(
                cfg, cohort, copy, group, train_ids, val_ids, test_ids, group_dir / f"fold_{fold}", fold
            )
            fold_metrics.append(outcome["metrics"])
            fold_predictions.append(outcome["predictions"])
            logger.info("%s fold %d: %s", name, fold, _headline(cfg, outcome["metrics"]))

            if cfg.OUTPUT.INTERPRET and fold == len(assignments) - 1:
                interpret(cfg, trained, test_loader, group_dir)
            release_workers(test_loader)

        summary = summarise_folds(fold_metrics)
        save_predictions(group_dir, fold_predictions, {"folds": fold_metrics, "summary": summary})
        results[name] = summary
        rows.append({"leads": name, "n_leads": len(group), **_flatten(summary)})
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
