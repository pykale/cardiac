"""Synthetic end-to-end workflows, from raw arrays to a scored prediction.

Three paths, each the shortest complete statement of something the package claims:

1. **ECG only.** A recording, an encoder, a classifier, a prediction.
2. **ECG and image.** The same, with two modalities fused -- and a subject missing one
   of them, because a cohort where every subject has everything is not the cohort these
   models are for.
3. **Lead-specific multi-lead ECG.** Pretrain a representation with no labels, then
   fine-tune it on a *subset* of the leads, which is the transfer both cardiac studies
   rest on.

Every one runs on generated data in a few seconds on a CPU, and none touches the
network or the private cohorts the studies were developed on.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.evaluate import binary_metrics, predict_split, roc_auc, save_predictions
from kalecardiac.interpret import attribution_ratios, modality_ablation, modality_attributions
from kalecardiac.loaddata import (
    ColumnTarget,
    CrossValidation,
    ECGArraySource,
    HoldOut,
    ImageArraySource,
    MultimodalDataset,
    collate_subjects,
    lead_sources,
)
from kalecardiac.model.embed import (
    LSEMVAE,
    CardioVAE,
    FeatureEmbedder,
    MultimodalPredictor,
    build_signal_encoder,
)
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask
from kalecardiac.prepdata import build_ecg_pipeline, build_image_pipeline
from kalecardiac.utils import set_seed

LENGTH = 128
LATENT = 8
IMAGE = (16, 16)


def fit(model, task, loaders, epochs: int = 2, lr: float = 1e-3) -> CardiacTrainer:
    """Train a model for a few epochs on the CPU, quietly."""
    trainer = CardiacTrainer(model, task, max_epochs=epochs, init_lr=lr)
    pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(trainer, *loaders)
    return trainer


def loader(dataset, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_subjects)


@pytest.fixture
def generated():
    return make_synthetic_multimodal_data(num_subjects=48, num_samples=LENGTH, image_size=IMAGE, seed=0)


def test_ecg_to_encoder_to_classifier_to_prediction(generated):
    """synthetic ECG -> model -> classifier -> prediction.

    No pretraining: a convolutional encoder trained from scratch with a clinical head,
    which is the supervised baseline both cardiac studies compare against. It is the
    same predictor class the pretrained workflows use, with a different embedder.
    """
    set_seed(0)
    source = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        sampling_rate=generated.sampling_rate,
        transform=build_ecg_pipeline(length=LENGTH, sampling_rate=generated.sampling_rate),
    )
    target = ColumnTarget.from_mapping({"label": generated.labels})
    frame = target.frame(generated.subject_id)

    split = next(
        HoldOut(test_size=0.3, val_size=0.2, group_by="subject_id", stratify_by=["label"], random_state=0).split(frame)
    )
    identifiers = {name: sorted(frame.iloc[split[name]]["subject_id"]) for name in ("train", "val", "test")}

    def build(names):
        return loader(MultimodalDataset(names, {"ecg": source}, target=target))

    task = ClassificationTask(hidden_dims=(16,))
    predictor = MultimodalPredictor(
        {
            "ecg": FeatureEmbedder(
                build_signal_encoder("conv", len(generated.leads), LENGTH, channels=(4, 8)), out_dim=16
            )
        },
        task.build_head,
    )
    model = fit(predictor, task, (build(identifiers["train"]), build(identifiers["val"])), epochs=4, lr=3e-3)

    predictions = predict_split(model, build(identifiers["test"]), split="test")
    assert len(predictions) == len(identifiers["test"])
    assert predictions.subject_id == identifiers["test"]
    assert np.isfinite(predictions.scores).all()

    metrics = binary_metrics(predictions.targets["label"], predictions.scores)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n"] == len(identifiers["test"])
    assert sum(metrics["confusion"].values()) == len(identifiers["test"])


def test_ecg_and_image_fuse_into_one_prediction(generated, tmp_path):
    """synthetic ECG + image -> multimodal model -> prediction.

    With one subject's radiograph withheld, because that is what the presence mask and
    the product-of-experts fusion exist for: a cohort where every subject has every
    modality is not the cohort these models are built for.
    """
    set_seed(0)
    ecg = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        sampling_rate=generated.sampling_rate,
        transform=build_ecg_pipeline(length=LENGTH, sampling_rate=generated.sampling_rate),
    )
    # One subject has no image at all.
    without_image = generated.subject_id[0]
    images = ImageArraySource(
        {name: value for name, value in generated.image.items() if name != without_image},
        channels=1,
        size=IMAGE,
        transform=build_image_pipeline(size=IMAGE),
    )
    target = ColumnTarget.from_mapping({"label": generated.labels})

    # --- pretrain a shared representation, with no labels at all
    vae = CardioVAE(
        signal_channels=len(generated.leads),
        signal_length=LENGTH,
        image_channels=1,
        image_size=IMAGE,
        latent_dim=LATENT,
        channels=(4, 8),
        signal_name="ecg",
        image_name="cxr",
    )
    sources = {"ecg": ecg, "cxr": images}
    unlabelled = loader(MultimodalDataset(generated.subject_id, sources), shuffle=True)
    fit(
        vae,
        ReconstructionTask(weights={"ecg": 10.0}, scale_factor=1e-3, kinds={"cxr": "bce"}, unimodal_streams=True),
        (unlabelled,),
        epochs=2,
    )

    # The subject with no radiograph is carried through as absent, not as zeros.
    sample = MultimodalDataset([without_image], sources, target=target)[0]
    assert bool(sample.present["ecg"]) is True
    assert bool(sample.present["cxr"]) is False

    # --- fine-tune onto the endpoint
    frame = target.frame(generated.subject_id)
    split = next(
        HoldOut(test_size=0.3, val_size=0.2, group_by="subject_id", stratify_by=["label"], random_state=0).split(frame)
    )
    identifiers = {name: sorted(frame.iloc[split[name]]["subject_id"]) for name in ("train", "val", "test")}

    def build(names):
        return loader(MultimodalDataset(names, sources, target=target))

    task = ClassificationTask(hidden_dims=(16,))
    predictor = MultimodalPredictor(vae.latent_embedders(frozen=True), task.build_head)
    model = fit(predictor, task, (build(identifiers["train"]), build(identifiers["val"])), epochs=3, lr=3e-3)

    test_loader = build(identifiers["test"])
    predictions = predict_split(model, test_loader, split="test")
    metrics = binary_metrics(predictions.targets["label"], predictions.scores)
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # --- and the result is reportable: written out, and attributable
    csv_path, json_path = save_predictions(tmp_path / "out", [predictions], {"test": metrics})
    assert csv_path.exists() and json_path.exists()

    ablation = modality_ablation(model, test_loader, roc_auc, "label")
    assert {"full", "without_ecg", "without_cxr"} <= set(ablation)


def test_lead_specific_pretraining_transfers_to_a_lead_subset(generated):
    """An LS-EMVAE-style workflow: pretrain on every lead, fine-tune on three.

    The claim under test is the one the whole design exists for -- that a representation
    learned from a full lead set can be read by a model that only ever sees part of it,
    without retraining the encoders.
    """
    set_seed(0)
    source = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        sampling_rate=generated.sampling_rate,
        transform=build_ecg_pipeline(length=LENGTH, sampling_rate=generated.sampling_rate),
    )
    every_lead = lead_sources(source)
    target = ColumnTarget.from_mapping({"label": generated.labels})

    # --- pretrain on all six leads, unlabelled
    vae = LSEMVAE(leads=generated.leads, length=LENGTH, latent_dim=LATENT, channels=(4, 8), num_groups=3)
    unlabelled = loader(MultimodalDataset(generated.subject_id, every_lead), shuffle=True)
    fit(vae, ReconstructionTask(scale_factor=1e-2, alignment_weight=0.1), (unlabelled,), epochs=3)

    # One decoder, shared: the latent must carry lead-agnostic state.
    assert len({id(vae.decoders[lead]) for lead in vae.leads}) == 1

    # --- fine-tune on three of them
    subset = ("I", "II", "aVF")
    assert set(subset) < set(generated.leads)
    sources = {lead: every_lead[lead] for lead in subset}

    frame = target.frame(generated.subject_id)
    folds = list(
        CrossValidation(n_splits=3, val_size=0.2, group_by="subject_id", stratify_by=["label"], random_state=0).split(
            frame
        )
    )
    identifiers = frame["subject_id"]

    scores, tested = [], []
    for fold, split in enumerate(folds):
        # A fresh copy per fold, so no fold's head reaches another's test subjects.
        copy = LSEMVAE(leads=generated.leads, length=LENGTH, latent_dim=LATENT, channels=(4, 8), num_groups=3)
        copy.load_state_dict(vae.state_dict())

        task = ClassificationTask(hidden_dims=(16,))
        predictor = MultimodalPredictor(copy.latent_embedders(subset, frozen=True), task.build_head)
        assert predictor.fused_dim == len(subset) * LATENT

        def build(name, split=split):
            return loader(MultimodalDataset(sorted(identifiers.iloc[split[name]]), sources, target=target))

        model = fit(predictor, task, (build("train"), build("val")), epochs=3, lr=3e-3)
        predictions = predict_split(model, build("test"), split="test", fold=fold)
        scores.append(predictions)
        tested.extend(predictions.subject_id)

    # Every subject is tested exactly once across the folds.
    assert sorted(tested) == sorted(generated.subject_id)

    pooled_labels = np.concatenate([prediction.targets["label"] for prediction in scores])
    pooled_scores = np.concatenate([prediction.scores for prediction in scores])
    assert np.isfinite(pooled_scores).all()
    assert 0.0 <= roc_auc(pooled_labels, pooled_scores) <= 1.0


def test_lead_importance_is_reported_per_lead(generated):
    """The interpretation half: which lead a lead-specific prediction rested on."""
    pytest.importorskip("captum")
    set_seed(0)

    source = ECGArraySource(
        generated.ecg,
        leads=generated.leads,
        sampling_rate=generated.sampling_rate,
        transform=build_ecg_pipeline(length=LENGTH, sampling_rate=generated.sampling_rate),
    )
    subset = ("I", "II", "aVF")
    sources = {lead: lead_sources(source)[lead] for lead in subset}
    target = ColumnTarget.from_mapping({"label": generated.labels})

    vae = LSEMVAE(leads=generated.leads, length=LENGTH, latent_dim=LATENT, channels=(4,))
    task = ClassificationTask(hidden_dims=())
    predictor = MultimodalPredictor(vae.latent_embedders(subset, frozen=False), task.build_head)

    data_loader = loader(MultimodalDataset(generated.subject_id, sources, target=target))
    model = fit(predictor, task, (data_loader,), epochs=2)

    batch = next(iter(data_loader))
    attributions = modality_attributions(model.model, batch.modalities, present=batch.present, n_steps=8)
    assert sorted(attributions) == sorted(subset)

    shares = attribution_ratios(attributions)
    assert sorted(shares) == sorted(subset)
    assert sum(entry["share"] for entry in shares.values()) == pytest.approx(1.0)


def test_a_missing_lead_does_not_stop_a_prediction(generated):
    """A subject whose recording lost a lead is still scorable.

    The reduced-lead case the cardiac models are built for, reduced to its smallest
    form: mark a lead absent and check the prediction still comes out finite and
    changed, rather than crashing or silently treating zeros as signal.
    """
    set_seed(0)
    source = ECGArraySource(generated.ecg, leads=generated.leads, sampling_rate=generated.sampling_rate)
    subset = ("I", "II", "III")
    sources = {lead: lead_sources(source)[lead] for lead in subset}
    target = ColumnTarget.from_mapping({"label": generated.labels})

    vae = LSEMVAE(
        leads=generated.leads, length=generated.ecg[generated.subject_id[0]].shape[1], latent_dim=LATENT, channels=(4,)
    )
    task = ClassificationTask(hidden_dims=())
    predictor = MultimodalPredictor(vae.latent_embedders(subset), task.build_head)
    predictor.eval()

    batch = next(iter(loader(MultimodalDataset(generated.subject_id, sources, target=target))))
    complete = predictor(batch.modalities, batch.present).prediction

    present = dict(batch.present)
    present["II"] = torch.zeros_like(present["II"])
    reduced = predictor(batch.modalities, present).prediction

    assert torch.isfinite(reduced).all()
    assert not torch.allclose(complete, reduced)
