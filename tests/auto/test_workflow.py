"""Pretraining, fine-tuning over folds and interpretation, read from one configuration.

Every run here is a tiny two-lead model on a synthetic cohort of 24 subjects, so a fold
trains in about a second; what is checked is the mechanics -- which subjects land where,
what is loaded instead of trained, what is written, and that a report cannot change the
models trained after it -- not how well anything scores.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

from kalecardiac.auto import (
    check_model_name,
    cross_validate,
    fine_tune,
    get_dataloader,
    get_ecg_transform_from_cfg,
    get_folds,
    get_source_sampling_rate,
    interpret_model,
    pretrain,
)
from kalecardiac.auto.workflow import _lightning_trainer
from kalecardiac.config import get_cfg_defaults
from kalecardiac.evaluate import MetricError
from kalecardiac.loaddata import MultimodalDataset
from kalecardiac.model.embed import LSEMVAE
from kalecardiac.pipeline import CardiacTrainer
from kalecardiac.utils import set_seed

LEADS = ("I", "II")
SUBJECTS = 24
LENGTH = 32


def workflow_cfg(tmp_path: Path, **overrides):
    """A configuration that trains for one epoch, on the CPU, in a temporary directory."""
    cfg = get_cfg_defaults()
    cfg.ENDPOINT.VALUE_COLUMN = "pressure"
    cfg.DATASET.NUM_FOLDS = 2
    cfg.DATASET.VAL_RATIO = 0.25
    cfg.MODEL.HEAD_HIDDEN = []
    cfg.SOLVER.MAX_EPOCHS = 1
    cfg.SOLVER.BATCH_SIZE = 8
    cfg.SOLVER.ACCELERATOR = "cpu"
    cfg.FINETUNE.MAX_EPOCHS = 1
    cfg.FINETUNE.BATCH_SIZE = 8
    cfg.OUTPUT.OUT_DIR = str(tmp_path)
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    return cfg


def tiny_model(latent_dim: int = 4) -> LSEMVAE:
    """A two-lead model small enough to train in a second."""
    return LSEMVAE(leads=list(LEADS), length=LENGTH, latent_dim=latent_dim, channels=(4,), fusion="poe")


@pytest.fixture
def cohort() -> tuple[MultimodalDataset, pd.DataFrame]:
    """24 subjects, two leads each, alternating labels the signals carry."""
    generator = np.random.default_rng(0)
    labels = np.tile([0, 1], SUBJECTS // 2)
    signals = generator.normal(size=(SUBJECTS, len(LEADS), 1, LENGTH)).astype(np.float32) + labels[:, None, None, None]
    identifiers = [f"s{index:02d}" for index in range(SUBJECTS)]
    dataset = MultimodalDataset.from_arrays(
        {lead: signals[:, position] for position, lead in enumerate(LEADS)},
        targets={"label": labels},
        identifiers=identifiers,
    )
    return dataset, pd.DataFrame({"subject_id": identifiers, "label": labels})


class TestCheckModelName:
    def test_the_name_a_workflow_builds_passes(self, tmp_path):
        check_model_name(workflow_cfg(tmp_path, **{"MODEL.NAME": "tiny"}), "tiny")

    def test_another_name_is_refused_with_the_hint_and_the_fusion_advice(self, tmp_path):
        cfg = workflow_cfg(tmp_path, **{"MODEL.NAME": "other"})
        expected = r"MODEL.NAME is 'other', but this workflow builds 'tiny'\. Run x\. To ablate the fusion"
        with pytest.raises(ValueError, match=expected):
            check_model_name(cfg, "tiny", hint="Run x.")

    def test_without_a_hint_the_advice_still_reads(self, tmp_path):
        with pytest.raises(ValueError, match=r"builds 'tiny'\. To ablate the fusion, keep MODEL.NAME and set"):
            check_model_name(workflow_cfg(tmp_path), "tiny")


class TestEcgTransform:
    def test_recordings_at_the_models_rate_are_not_resampled(self, tmp_path):
        cfg = workflow_cfg(tmp_path)
        assert get_source_sampling_rate(cfg) == cfg.ECG.SAMPLING_RATE
        assert "resample_signal" not in repr(get_ecg_transform_from_cfg(cfg))

    def test_a_source_rate_resamples_after_interpolating_and_before_cropping(self, tmp_path):
        cfg = workflow_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250})
        assert get_source_sampling_rate(cfg) == 250.0
        steps = repr(get_ecg_transform_from_cfg(cfg))
        assert steps.index("interpolate_signal") < steps.index("resample_signal") < steps.index("crop_or_pad_signal")

    def test_the_length_can_be_given_in_place_of_the_configured_one(self, tmp_path):
        transform = get_ecg_transform_from_cfg(workflow_cfg(tmp_path), length=10)
        assert transform(np.random.default_rng(0).normal(size=(2, 100))).shape == (2, 10)

    def test_each_lead_is_normalized(self, tmp_path):
        recording = np.vstack([np.linspace(0, 1, 64), 50 + 10 * np.linspace(0, 1, 64)])
        leads = get_ecg_transform_from_cfg(workflow_cfg(tmp_path, **{"ECG.LENGTH": 64}))(recording)
        assert np.allclose(leads.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(leads.std(axis=1), 1.0, atol=1e-4)

    def test_normalize_off_keeps_the_recordings_own_units(self, tmp_path):
        recording = np.arange(2 * 64, dtype=np.float32).reshape(2, 64)
        transform = get_ecg_transform_from_cfg(workflow_cfg(tmp_path, **{"ECG.LENGTH": 64, "ECG.NORMALIZE": False}))
        assert np.array_equal(transform(recording), recording)

<<<<<<< HEAD
    def test_one_channel_interleaves_the_leads_each_cropped_to_its_share(self, tmp_path):
        overrides = {"ECG.LEADS": list(LEADS), "ECG.LENGTH": 64, "ECG.NUM_CHANNELS": 1, "ECG.NORMALIZE": False}
        recording = np.vstack([np.full(100, 1.0), np.full(100, 2.0)])
        strip = get_ecg_transform_from_cfg(workflow_cfg(tmp_path, **overrides))(recording)
        assert np.array_equal(strip, np.tile([1.0, 2.0], 32)[None, :])

    def test_the_leads_can_be_counted_by_the_caller(self, tmp_path):
        # A cohort whose recordings hold other leads than ECG.LEADS names, as a synthetic one may.
        cfg = workflow_cfg(tmp_path, **{"ECG.LENGTH": 60, "ECG.NUM_CHANNELS": 1})
        assert get_ecg_transform_from_cfg(cfg, num_leads=3)(np.zeros((3, 50))).shape == (1, 60)

    @pytest.mark.parametrize("channels", [0, 2])
    def test_a_channel_per_lead_keeps_the_leads_apart(self, tmp_path, channels):
        cfg = workflow_cfg(tmp_path, **{"ECG.LEADS": list(LEADS), "ECG.LENGTH": 64, "ECG.NUM_CHANNELS": channels})
        assert get_ecg_transform_from_cfg(cfg)(np.zeros((2, 100))).shape == (2, 64)

    def test_a_channel_count_that_is_neither_is_refused(self, tmp_path):
        cfg = workflow_cfg(tmp_path, **{"ECG.LEADS": list(LEADS), "ECG.NUM_CHANNELS": 3})
        with pytest.raises(ValueError, match="ECG.NUM_CHANNELS is 3, but the recordings hold 2 leads"):
            get_ecg_transform_from_cfg(cfg)

    def test_a_strip_that_does_not_divide_among_the_leads_is_refused(self, tmp_path):
        cfg = workflow_cfg(tmp_path, **{"ECG.LEADS": list(LEADS), "ECG.LENGTH": 65, "ECG.NUM_CHANNELS": 1})
        with pytest.raises(ValueError, match="65 is not a whole number of samples for each of 2 leads"):
            get_ecg_transform_from_cfg(cfg)

=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a

class TestLightningTrainer:
    """The trainer every stage fits with: checkpointing on the task's monitor, stopping early if asked."""

    def test_a_monitored_run_keeps_its_best_checkpoint_and_stops_when_asked(self, tmp_path):
        cfg = workflow_cfg(tmp_path, **{"SOLVER.EARLY_STOP": 3})
        trainer = _lightning_trainer(cfg, 5, tmp_path, monitor="val_roc_auc", mode="max")
        checkpoint = trainer.checkpoint_callback
        assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
        assert (checkpoint.monitor, checkpoint.mode) == ("val_roc_auc", "max")
        stopping = [callback for callback in trainer.callbacks if isinstance(callback, pl.callbacks.EarlyStopping)]
        assert [(callback.monitor, callback.patience) for callback in stopping] == [("val_roc_auc", 3)]

    def test_early_stopping_is_off_unless_configured(self, tmp_path):
        trainer = _lightning_trainer(workflow_cfg(tmp_path), 5, tmp_path, monitor="val_loss")
        assert not any(isinstance(callback, pl.callbacks.EarlyStopping) for callback in trainer.callbacks)


class TestGetDataloader:
    def test_it_batches_subjects_with_the_configured_workers(self, tmp_path, cohort):
        dataset, _ = cohort
        loader = get_dataloader(workflow_cfg(tmp_path), dataset, batch_size=5)
        batch = next(iter(loader))
        assert batch.subject_id == ["s00", "s01", "s02", "s03", "s04"]
        assert tuple(batch.modalities["I"].shape) == (5, 1, LENGTH)
        assert loader.num_workers == 0


class TestGetFolds:
    def test_cross_validation_tests_every_subject_exactly_once(self, tmp_path, cohort):
        _, table = cohort
        folds = get_folds(workflow_cfg(tmp_path), table)
        assert [fold.index for fold in folds] == [0, 1]
        assert sorted(subject for fold in folds for subject in fold.test_ids) == sorted(table["subject_id"])

    def test_validation_is_carved_from_the_training_half_and_nothing_overlaps(self, tmp_path, cohort):
        _, table = cohort
        for fold in get_folds(workflow_cfg(tmp_path), table):
            train, val, test = set(fold.train_ids), set(fold.val_ids), set(fold.test_ids)
            assert val and not (train & val or train & test or val & test)
            assert train | val | test == set(table["subject_id"])
            assert fold.train_ids == sorted(fold.train_ids)

    def test_the_folds_are_stratified_by_label(self, tmp_path, cohort):
        _, table = cohort
        labels = dict(zip(table["subject_id"], table["label"], strict=True))
        for fold in get_folds(workflow_cfg(tmp_path), table):
            assert sum(labels[subject] for subject in fold.test_ids) == len(fold.test_ids) // 2

    def test_a_random_split_is_one_held_out_fold(self, tmp_path, cohort):
        _, table = cohort
        folds = get_folds(workflow_cfg(tmp_path, **{"DATASET.SPLIT_MODE": "random", "DATASET.TEST_RATIO": 0.25}), table)
        assert len(folds) == 1 and len(folds[0].test_ids) == SUBJECTS // 4

    def test_the_seed_fixes_the_folds(self, tmp_path, cohort):
        _, table = cohort
        assert get_folds(workflow_cfg(tmp_path), table) == get_folds(workflow_cfg(tmp_path), table)
        assert get_folds(workflow_cfg(tmp_path), table) != get_folds(
            workflow_cfg(tmp_path, **{"SOLVER.SEED": 1}), table
        )

    def test_a_published_assignment_is_pointed_at_predefined_split(self, tmp_path, cohort):
        _, table = cohort
        with pytest.raises(ValueError, match="PredefinedSplit"):
            get_folds(workflow_cfg(tmp_path, **{"DATASET.SPLIT_MODE": "predefined"}), table)


class TestPretrain:
    def test_it_trains_and_writes_the_weights_a_later_run_can_load(self, tmp_path, cohort):
        dataset, _ = cohort
        model = tiny_model()
        before = [parameter.detach().clone() for parameter in model.parameters()]

        assert pretrain(workflow_cfg(tmp_path), model, dataset, tmp_path) is model
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
        saved = torch.load(tmp_path / "pretrained.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(saved[key], value) for key, value in model.state_dict().items())

<<<<<<< HEAD
    def test_the_weights_are_the_only_checkpoint_written(self, tmp_path, cohort):
        # Lightning's default would also save the model and its optimizer after every
        # epoch: gigabytes, for a published-size CardioVAE, that nothing reads.
        pretrain(workflow_cfg(tmp_path, **{"SOLVER.MAX_EPOCHS": 2}), tiny_model(), cohort[0], tmp_path)
        assert [path.name for path in tmp_path.rglob("*.ckpt")] == []
        assert (tmp_path / "pretrained.pt").exists()

=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    def test_a_named_checkpoint_is_loaded_and_nothing_is_trained(self, tmp_path, cohort):
        dataset, _ = cohort
        saved = tiny_model()
        torch.save(saved.state_dict(), tmp_path / "given.pt")

        cfg = workflow_cfg(tmp_path, **{"MODEL.PRETRAINED": str(tmp_path / "given.pt")})
        loaded = pretrain(cfg, tiny_model(), dataset, tmp_path / "run")
        for ours, theirs in zip(loaded.parameters(), saved.parameters(), strict=True):
            assert torch.equal(ours, theirs)
        assert not (tmp_path / "run").exists()

    def test_a_checkpoint_from_another_architecture_fails_loudly(self, tmp_path, cohort):
        # Half-loading would fine-tune on encoders that are partly random.
        dataset, _ = cohort
        torch.save(tiny_model(latent_dim=8).state_dict(), tmp_path / "wider.pt")
        cfg = workflow_cfg(tmp_path, **{"MODEL.PRETRAINED": str(tmp_path / "wider.pt")})
        with pytest.raises(RuntimeError):
            pretrain(cfg, tiny_model(), dataset, tmp_path)

    def test_a_missing_checkpoint_is_reported_by_path(self, tmp_path, cohort):
        dataset, _ = cohort
        cfg = workflow_cfg(tmp_path, **{"MODEL.PRETRAINED": str(tmp_path / "absent.pt")})
        with pytest.raises(FileNotFoundError, match="absent.pt"):
            pretrain(cfg, tiny_model(), dataset, tmp_path)


class TestFineTune:
    def test_it_scores_the_folds_test_subjects(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        fold = get_folds(cfg, table)[0]
        result = fine_tune(cfg, tiny_model(), dataset.sources, dataset.target, fold, tmp_path)

        assert isinstance(result.model, CardiacTrainer)
        assert sorted(result.predictions.subject_id) == fold.test_ids
        assert result.predictions.fold == 0
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0

    def test_a_subset_of_the_pretrained_modalities_can_be_fine_tuned(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        result = fine_tune(
            cfg, tiny_model(), {"I": dataset.sources["I"]}, dataset.target, get_folds(cfg, table)[0], tmp_path
        )
        assert list(result.model.model.modalities) == ["I"]

    def test_frozen_encoders_are_not_trained(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path, **{"MODEL.FREEZE_ENCODERS": True})
        model = tiny_model()
        before = {name: value.clone() for name, value in model.encoders.state_dict().items()}
        fine_tune(cfg, model, dataset.sources, dataset.target, get_folds(cfg, table)[0], tmp_path)
        assert all(torch.equal(before[name], value) for name, value in model.encoders.state_dict().items())


def _subjects_predicted(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return sorted(row["subject_id"] for row in csv.DictReader(handle))


@pytest.mark.slow
class TestCrossValidate:
    def test_every_subject_is_predicted_once_and_the_folds_summarized(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        summary = cross_validate(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table), tmp_path)

        assert summary["n_folds"] == 2
        assert len(summary["roc_auc"]["values"]) == 2
        assert _subjects_predicted(tmp_path / "predictions.csv") == sorted(table["subject_id"])
        assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["summary"] == summary
        assert (tmp_path / "fold_0").is_dir() and (tmp_path / "fold_1").is_dir()

    def test_the_pretrained_model_is_left_as_it_was(self, tmp_path, cohort):
        # Each fold fine-tunes a copy; trained encoders would otherwise carry one fold's
        # test subjects into the next fold's training.
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path, **{"MODEL.FREEZE_ENCODERS": False})
        pretrained = tiny_model()
        before = {name: value.clone() for name, value in pretrained.state_dict().items()}
        cross_validate(cfg, pretrained, dataset.sources, dataset.target, get_folds(cfg, table), tmp_path)
        assert all(torch.equal(before[name], value) for name, value in pretrained.state_dict().items())

    def test_the_report_sees_every_fold_and_cannot_change_the_training(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        folds = get_folds(cfg, table)
        pretrained = tiny_model()
        seen = []

        def report(fold, result):
            # Draws from every generator a later fold's training reads.
            seen.append((fold.index, sorted(result.predictions.subject_id) == fold.test_ids))
            torch.rand(100), np.random.rand(100)

        set_seed(cfg.SOLVER.SEED)
        silent = cross_validate(cfg, pretrained, dataset.sources, dataset.target, folds, tmp_path / "silent")
        set_seed(cfg.SOLVER.SEED)
        reported = cross_validate(
            cfg, pretrained, dataset.sources, dataset.target, folds, tmp_path / "reported", report
        )

        assert seen == [(0, True), (1, True)]
        assert reported["roc_auc"]["values"] == silent["roc_auc"]["values"]

    def test_the_last_fold_is_interpreted_when_asked(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path, **{"OUTPUT.INTERPRET": True})
        cross_validate(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table), tmp_path)
        interpretation = json.loads((tmp_path / "interpretation.json").read_text(encoding="utf-8"))
        assert interpretation["ablation"]["drop_I"] == pytest.approx(
            interpretation["ablation"]["full"] - interpretation["ablation"]["without_I"]
        )

    def test_nothing_is_interpreted_unless_asked(self, tmp_path, cohort):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        cross_validate(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table), tmp_path)
        assert not (tmp_path / "interpretation.json").exists()


class TestInterpretModel:
    def test_the_attribution_table_has_a_row_per_modality(self, tmp_path, cohort):
        pytest.importorskip("captum")
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        result = fine_tune(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table)[0], tmp_path)
        summary = interpret_model(cfg, result.model, result.test_loader, tmp_path / "explained")

        with (tmp_path / "explained" / "attribution.csv").open(encoding="utf-8") as handle:
            assert sorted(row["modality"] for row in csv.DictReader(handle)) == list(LEADS)
        assert set(summary) == {"attribution", "ablation"}

    def test_a_one_modality_model_has_no_ablation_to_report(self, tmp_path, cohort):
        # Withholding its only modality leaves it its prior, which says nothing about the modality.
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        fold = get_folds(cfg, table)[0]
        result = fine_tune(cfg, tiny_model(), {"I": dataset.sources["I"]}, dataset.target, fold, tmp_path)
        assert "ablation" not in interpret_model(cfg, result.model, result.test_loader, tmp_path / "explained")

    def test_without_captum_the_ablation_is_still_reported(self, tmp_path, cohort, monkeypatch, caplog):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        result = fine_tune(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table)[0], tmp_path)
        monkeypatch.setattr("kalecardiac.auto.workflow.modality_attributions", _raise(ImportError("needs captum")))

        with caplog.at_level(logging.WARNING, logger="kalecardiac.auto.workflow"):
            summary = interpret_model(cfg, result.model, result.test_loader, tmp_path / "explained")
        assert set(summary) == {"ablation"}
        assert "skipping attribution: needs captum" in caplog.text
        assert not (tmp_path / "explained" / "attribution.csv").exists()

    def test_a_split_that_cannot_be_rescored_writes_nothing(self, tmp_path, cohort, monkeypatch, caplog):
        dataset, table = cohort
        cfg = workflow_cfg(tmp_path)
        result = fine_tune(cfg, tiny_model(), dataset.sources, dataset.target, get_folds(cfg, table)[0], tmp_path)
        monkeypatch.setattr("kalecardiac.auto.workflow.modality_attributions", _raise(ImportError("needs captum")))
        monkeypatch.setattr("kalecardiac.auto.workflow.modality_ablation", _raise(MetricError("one class only")))

        with caplog.at_level(logging.WARNING, logger="kalecardiac.auto.workflow"):
            assert interpret_model(cfg, result.model, result.test_loader, tmp_path / "explained") == {}
        assert "skipping ablation: one class only" in caplog.text
        assert not (tmp_path / "explained" / "interpretation.json").exists()


def _raise(error: Exception):
    """A stand-in for a function that fails with ``error``, as an optional dependency or a metric can."""

    def fail(*args, **kwargs):
        raise error

    return fail
