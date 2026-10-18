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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from examples.cardiovae_multimodal.config import CXR, ECG
from examples.cardiovae_multimodal.data import (
    CardioVAECohort,
    check_endpoint,
    load_cohort,
    modality_names,
    summarise,
)
from kalecardiac.config import parse_weights
from kalecardiac.evaluate import (
    MetricError,
    binary_metrics,
    multiclass_metrics,
    multiclass_roc_auc,
    predict_split,
    r2_score,
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
from kalecardiac.model.predict import class_weight_from_labels, positive_weight_from_labels
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask, RegressionTask
from kalecardiac.utils import ensure_dir, load_checkpoint, preserve_rng_state, set_seed, write_csv, write_json

logger = logging.getLogger("cardiovae_multimodal")


#: The model this example builds. ``MODEL.NAME`` must say so, which is what turns a
#: configuration copied from another example into a start-up error rather than a run
#: that quietly trains the wrong thing.
MODEL_NAME = "cardiovae"


def check_model_name(cfg) -> None:
    """Reject a ``MODEL.NAME`` this example cannot build.

    Raises:
        ValueError: If ``MODEL.NAME`` is anything but :data:`MODEL_NAME`, naming what to
            do instead.
    """
    if cfg.MODEL.NAME != MODEL_NAME:
        raise ValueError(
            f"MODEL.NAME is {cfg.MODEL.NAME!r}, but this example builds {MODEL_NAME!r}. For a lead-specific "
            f"model run examples.lsemvae_ecg; to ablate the fusion, keep MODEL.NAME and set FUSION.LATENT_METHOD."
        )


def resolve_arms(cfg, arms: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]]:
    """The modality arms to compare, validated before anything is loaded.

    Without ``arms``, ``DATASET.MODALITIES`` decides: each modality alone, then all of
    them fused, which is what makes the multimodal claim testable. One modality gives a
    unimodal run.

    Raises:
        ValueError: If no arm is left, or an arm names a modality this example lacks --
            caught here rather than as a ``KeyError`` deep inside fine-tuning.
    """
    if arms is None:
        names = tuple(cfg.DATASET.MODALITIES)
        arms = [(name,) for name in names] + ([names] if len(names) > 1 else [])
    resolved = [tuple(arm) for arm in arms]
    if not resolved or not all(resolved):
        raise ValueError("no modality to compare; set DATASET.MODALITIES or pass --arms")

    known = set(modality_names())
    unknown = sorted({name for arm in resolved for name in arm} - known)
    if unknown:
        raise ValueError(f"arms name {unknown}, which this example does not have; it has {sorted(known)}")
    return resolved


def build_model(cfg, cohort: CardioVAECohort) -> CardioVAE:
    """Construct the pretraining model from the configuration and the cohort's shapes."""
    return CardioVAE(
        signal_channels=cohort.ecg_channels,
        signal_length=cohort.ecg_length,
        image_channels=cfg.IMAGE.CHANNELS,
        image_size=tuple(cohort.image_size),
        latent_dim=cfg.MODEL.LATENT_DIM,
        channels=list(cfg.MODEL.CHANNELS),
        kernel_size=cfg.MODEL.KERNEL_SIZE,
        signal_encoder=cfg.ECG.ENCODER,
        # The image source scales pixels into the unit interval, so the decoder must be
        # able to produce that range and nothing outside it.
        image_output_activation="sigmoid" if cfg.IMAGE.SCALE else None,
        signal_name=ECG,
        image_name=CXR,
        fusion=cfg.FUSION.LATENT_METHOD,
        use_prior=cfg.FUSION.USE_PRIOR,
        num_groups=cfg.FUSION.NUM_GROUPS,
    )


def load_pretrained(cfg, cohort: CardioVAECohort) -> CardioVAE:
    """Load the representation from ``MODEL.PRETRAINED`` rather than learning it again.

    Strictly: a checkpoint saved from a different architecture -- another latent width,
    another kernel -- fails here rather than half-loading and fine-tuning on a model
    whose encoders are partly random.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If its weights do not match the configured architecture.
    """
    model = build_model(cfg, cohort)
    load_checkpoint(cfg.MODEL.PRETRAINED, model, strict=True)
    logger.info("loaded the pretrained representation from %s; pretraining skipped", cfg.MODEL.PRETRAINED)
    return model


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
    """The downstream task: a class of the endpoint, or the measurement itself.

    Class balance is taken from ``labels``, which must be this fold's *training* labels:
    a balance taken over the whole cohort carries the test split's composition into the
    loss. ``OBJECTIVE.POS_WEIGHT`` 0 derives it, a negative value disables it, and a
    positive one fixes a binary endpoint's positive weight.
    """
    common = {"hidden_dims": list(cfg.MODEL.HEAD_HIDDEN), "dropout": cfg.MODEL.DROPOUT}
    if cfg.ENDPOINT.REGRESSION:
        return RegressionTask(**common)

    training = torch.tensor(list(labels), dtype=torch.float32)
    derive = cfg.OBJECTIVE.POS_WEIGHT == 0
    num_classes = int(cfg.ENDPOINT.NUM_CLASSES)
    if num_classes > 2:
        weight = class_weight_from_labels(training, num_classes) if derive else None
        return ClassificationTask(num_classes=num_classes, class_weight=weight, **common)
    pos_weight = float(positive_weight_from_labels(training)) if derive else cfg.OBJECTIVE.POS_WEIGHT
    return ClassificationTask(pos_weight=max(pos_weight, 0.0), **common)


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
    if int(cfg.ENDPOINT.NUM_CLASSES) > 2:
        # The scores are logits, one column per class; the metric applies the softmax.
        return multiclass_metrics(predictions.targets["label"], predictions.scores)
    return binary_metrics(predictions.targets["label"], predictions.scores)


def headline_metric(cfg) -> str:
    """The metric a run is summarised by: the error, or the threshold-free ranking."""
    if cfg.ENDPOINT.REGRESSION:
        return "rmse"
    return "roc_auc_ovr" if int(cfg.ENDPOINT.NUM_CLASSES) > 2 else "roc_auc"


def withholding_metric(cfg) -> tuple[str, Callable, str]:
    """What a rescoring with modalities withheld is measured in: ``(name, metric, target)``.

    The threshold-free metric of each endpoint, so that a score which moves as a
    modality is withheld is not also moving with an operating point.
    """
    if cfg.ENDPOINT.REGRESSION:
        return "r2", r2_score, "value"
    if int(cfg.ENDPOINT.NUM_CLASSES) > 2:
        return "roc_auc_ovr", multiclass_roc_auc, "label"
    return "roc_auc", roc_auc, "label"


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

    if len(model.model.modalities) > 1:
        _, metric, target = withholding_metric(cfg)
        try:
            summary["ablation"] = modality_ablation(model, test_loader, metric, target)
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
        arms: Modality combinations to compare. ``None`` derives them from
            ``DATASET.MODALITIES``; see :func:`resolve_arms`.

    Returns:
        Per-arm fold summaries, keyed by the arm's name.
    """
    # Everything that can be checked without data is checked before any is loaded.
    check_model_name(cfg)
    check_endpoint(cfg)
    arms = resolve_arms(cfg, arms)

    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    cohort = load_cohort(cfg)
    logger.info("cohort: %s", cohort.describe())
    write_json(out_dir / "cohort.json", summarise(cohort, cfg.ENDPOINT.NAME))

    pretrained = load_pretrained(cfg, cohort) if cfg.MODEL.PRETRAINED else pretrain(cfg, cohort, out_dir)
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

            # Reporting is kept off the training's random stream, so turning it on
            # or off cannot change the models trained after it.
            if cfg.OUTPUT.INTERPRET and fold == len(assignments) - 1:
                with preserve_rng_state():
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
    key = headline_metric(cfg)
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
