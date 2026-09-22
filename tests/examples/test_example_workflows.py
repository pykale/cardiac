"""The two examples, run end to end on synthetic cohorts.

These are the slowest tests in the suite and the most valuable: they exercise the whole
path from a configuration file to a written result, which is the only thing that catches
a mismatch between the library's contracts and the way an experiment composes them.

Every run here is tiny and synthetic. Nothing reaches the network, and nothing needs the
private cohorts either study was developed on.
"""

from __future__ import annotations

import json

import pytest
import torch

from examples.cardiovae_multimodal import config as cardiovae_config
from examples.cardiovae_multimodal import data as cardiovae_data
from examples.cardiovae_multimodal import runner as cardiovae_runner
from examples.lsemvae_ecg import config as lsemvae_config
from examples.lsemvae_ecg import data as lsemvae_data
from examples.lsemvae_ecg import runner as lsemvae_runner
from kalecardiac.model.embed import LSEMVAE, CardioVAE


def cardiovae_cfg(tmp_path, **overrides):
    """A tiny CardioVAE configuration that finishes in seconds."""
    cfg = cardiovae_config.get_cfg_defaults()
    cfg.DATASET.ROOT = ""
    cfg.DATASET.NUM_WORKERS = 0
    cfg.DATASET.NUM_FOLDS = 2
    cfg.DATASET.VAL_RATIO = 0.25
    cfg.ECG.LENGTH = 6 * 64
    cfg.ECG.NUM_CHANNELS = 1
    cfg.IMAGE.SIZE = [16, 16]
    cfg.MODEL.LATENT_DIM = 8
    cfg.MODEL.CHANNELS = [4, 8]
    cfg.MODEL.HEAD_HIDDEN = []
    cfg.OBJECTIVE.ANNEALING_EPOCHS = 1
    cfg.SOLVER.MAX_EPOCHS = 1
    cfg.SOLVER.BATCH_SIZE = 8
    cfg.SOLVER.ACCELERATOR = "cpu"
    cfg.FINETUNE.MAX_EPOCHS = 1
    cfg.FINETUNE.BATCH_SIZE = 8
    cfg.SYNTHETIC.NUM_SUBJECTS = 24
    cfg.SYNTHETIC.NUM_SAMPLES = 64
    cfg.SYNTHETIC.IMAGE_SIZE = [16, 16]
    cfg.OUTPUT.OUT_DIR = str(tmp_path / "cardiovae")
    cfg.OUTPUT.INTERPRET = False
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    cfg.freeze()
    return cfg


def lsemvae_cfg(tmp_path, **overrides):
    """A tiny LS-EMVAE configuration that finishes in seconds."""
    cfg = lsemvae_config.get_cfg_defaults()
    cfg.DATASET.ROOT = ""
    cfg.DATASET.NUM_WORKERS = 0
    cfg.DATASET.NUM_FOLDS = 2
    cfg.DATASET.VAL_RATIO = 0.25
    cfg.ECG.LEADS = ["I", "II", "III", "aVR"]
    cfg.ECG.FINETUNE_LEADS = ["I", "II"]
    cfg.ECG.LENGTH = 64
    cfg.MODEL.LATENT_DIM = 8
    cfg.MODEL.CHANNELS = [4, 8]
    cfg.MODEL.HEAD_HIDDEN = []
    cfg.FUSION.NUM_GROUPS = 2
    cfg.OBJECTIVE.ANNEALING_EPOCHS = 1
    cfg.SOLVER.MAX_EPOCHS = 1
    cfg.SOLVER.BATCH_SIZE = 8
    cfg.SOLVER.ACCELERATOR = "cpu"
    cfg.FINETUNE.MAX_EPOCHS = 1
    cfg.FINETUNE.BATCH_SIZE = 8
    cfg.SYNTHETIC.NUM_SUBJECTS = 24
    cfg.SYNTHETIC.NUM_SAMPLES = 64
    cfg.OUTPUT.OUT_DIR = str(tmp_path / "lsemvae")
    cfg.OUTPUT.INTERPRET = False
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    cfg.freeze()
    return cfg


class TestCardioVAEExample:
    def test_the_shipped_quick_config_is_valid(self):
        cfg = cardiovae_config.get_cfg_defaults()
        cfg.merge_from_file("examples/cardiovae_multimodal/configs/quick.yaml")
        assert cfg.DATASET.ROOT == ""

    def test_the_shipped_full_config_is_valid(self):
        cfg = cardiovae_config.get_cfg_defaults()
        cfg.merge_from_file("examples/cardiovae_multimodal/configs/full.yaml")
        # The published setup: twelve leads of 5000 samples flattened end to end.
        assert cfg.ECG.LENGTH == 60000
        assert cfg.IMAGE.SIZE == [224, 224]

    def test_the_full_config_ships_no_path(self):
        # A private cohort has no default location that could be right.
        cfg = cardiovae_config.get_cfg_defaults()
        cfg.merge_from_file("examples/cardiovae_multimodal/configs/full.yaml")
        assert cfg.DATASET.ROOT == ""

    def test_the_synthetic_cohort_assembles(self, tmp_path):
        cohort = cardiovae_data.load_cohort(cardiovae_cfg(tmp_path))
        assert cohort.synthetic is True
        assert cohort.ecg_channels == 1
        assert cohort.ecg_length == 6 * 64
        assert len(cohort.labelled_ids) > 0
        assert "synthetic" in cohort.describe()

    def test_the_model_matches_the_cohort_shapes(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        cohort = cardiovae_data.load_cohort(cfg)
        model = cardiovae_runner.build_model(cfg, cohort)
        assert isinstance(model, CardioVAE)
        assert set(model.modalities) == {cardiovae_config.ECG, cardiovae_config.CXR}

    @pytest.mark.slow
    def test_the_whole_workflow_runs_and_writes_a_result(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        results = cardiovae_runner.run(cfg, arms=[("ecg",), ("ecg", "cxr")])

        assert sorted(results) == ["ecg", "ecg+cxr"]
        for summary in results.values():
            assert summary["n_folds"] == 2
            assert 0.0 <= summary["roc_auc"]["mean"] <= 1.0

        out = tmp_path / "cardiovae"
        assert (out / "config.yaml").exists()
        assert (out / "pretrained.pt").exists()
        assert (out / "results.csv").exists()
        assert (out / "ecg+cxr" / "predictions.csv").exists()
        written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert sorted(written) == ["ecg", "ecg+cxr"]

    @pytest.mark.slow
    def test_every_subject_is_predicted_exactly_once_across_folds(self, tmp_path):
        import csv

        cfg = cardiovae_cfg(tmp_path)
        cardiovae_runner.run(cfg, arms=[("ecg",)])
        with (tmp_path / "cardiovae" / "ecg" / "predictions.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        cohort = cardiovae_data.load_cohort(cfg)
        assert sorted(row["subject_id"] for row in rows) == sorted(cohort.labelled_ids)

    @pytest.mark.slow
    def test_a_regression_endpoint_runs_through_the_same_path(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ENDPOINT.REGRESSION": True})
        results = cardiovae_runner.run(cfg, arms=[("ecg",)])
        assert "rmse" in results["ecg"]


class TestLSEMVAEExample:
    def test_the_shipped_quick_config_is_valid(self):
        cfg = lsemvae_config.get_cfg_defaults()
        cfg.merge_from_file("examples/lsemvae_ecg/configs/quick.yaml")
        assert cfg.DATASET.ROOT == ""

    def test_the_shipped_full_config_is_the_published_setup(self):
        cfg = lsemvae_config.get_cfg_defaults()
        cfg.merge_from_file("examples/lsemvae_ecg/configs/full.yaml")
        assert len(cfg.ECG.LEADS) == 12
        assert len(cfg.ECG.FINETUNE_LEADS) == 6
        assert cfg.FUSION.LATENT_METHOD == "hime"
        assert cfg.OBJECTIVE.ALIGNMENT_WEIGHT > 0

    def test_the_synthetic_cohorts_assemble(self, tmp_path):
        cohort = lsemvae_data.load_cohort(lsemvae_cfg(tmp_path))
        assert cohort.synthetic is True
        assert len(cohort.pretrain_sources) == 4
        assert len(cohort.finetune_sources) == 2
        # The unlabelled cohort is the larger one, which is the method's premise.
        assert len(cohort.pretrain_ids) > len(cohort.labelled_ids)

    def test_each_lead_is_a_single_channel_modality(self, tmp_path):
        cohort = lsemvae_data.load_cohort(lsemvae_cfg(tmp_path))
        source = cohort.finetune_sources["I"]
        assert source.get(cohort.labelled_ids[0], 0).shape == (1, cohort.length)

    def test_derive_missing_reconstructs_the_limb_leads(self, tmp_path):
        # A recorder that stored only I and II has lost nothing about the frontal
        # plane, so a six-lead model can still be fine-tuned on its recordings.
        cfg = lsemvae_cfg(
            tmp_path,
            **{
                "ECG.LEADS": ["I", "II"],
                "ECG.FINETUNE_LEADS": ["I", "II", "III", "aVF"],
                "ECG.DERIVE_MISSING": True,
                "FUSION.NUM_GROUPS": 2,
            },
        )
        cohort = lsemvae_data.load_cohort(cfg)
        assert sorted(cohort.finetune_sources) == ["I", "II", "III", "aVF"]

        # The derived lead carries Einthoven's relation, though not exactly: derivation
        # runs on the raw recording and each lead is then standardised independently,
        # which rescales II - I and III by different factors. The exact identity is
        # checked before standardisation, in tests/prepdata/test_ecg_transform.py.
        subject = cohort.labelled_ids[0]
        lead_i = cohort.finetune_sources["I"].get(subject, 0)
        lead_ii = cohort.finetune_sources["II"].get(subject, 0)
        lead_iii = cohort.finetune_sources["III"].get(subject, 0)
        assert lead_iii.shape == lead_i.shape
        correlation = torch.corrcoef(torch.stack([(lead_ii - lead_i).flatten(), lead_iii.flatten()]))[0, 1]
        assert float(correlation) > 0.9

    def test_deriving_a_precordial_lead_is_refused(self, tmp_path):
        # V1 measures a plane the limb leads do not; there is nothing to derive it from.
        cfg = lsemvae_cfg(
            tmp_path,
            **{
                "ECG.LEADS": ["I", "II"],
                "ECG.FINETUNE_LEADS": ["I", "II", "V1"],
                "ECG.DERIVE_MISSING": True,
                "FUSION.NUM_GROUPS": 2,
            },
        )
        with pytest.raises(Exception, match="plane the limb leads do not"):
            lsemvae_data.load_cohort(cfg)

    def test_the_model_encodes_one_lead_per_modality(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        cohort = lsemvae_data.load_cohort(cfg)
        model = lsemvae_runner.build_model(cfg, cohort, cohort.pretrain_leads)
        assert isinstance(model, LSEMVAE)
        assert model.leads == cohort.pretrain_leads

    @pytest.mark.slow
    def test_the_whole_workflow_runs_and_writes_a_result(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        results = lsemvae_runner.run(cfg)

        assert list(results) == ["I+II"]
        assert results["I+II"]["n_folds"] == 2

        out = tmp_path / "lsemvae"
        assert (out / "config.yaml").exists()
        assert (out / "pretrained.pt").exists()
        assert (out / "cohort.json").exists()
        assert (out / "I+II" / "predictions.csv").exists()

    @pytest.mark.slow
    def test_twelve_to_six_transfer_is_what_the_run_performs(self, tmp_path):
        # Pretraining sees four leads, fine-tuning two: the transfer the method is for.
        cfg = lsemvae_cfg(tmp_path)
        cohort = lsemvae_data.load_cohort(cfg)
        assert set(cohort.finetune_leads) < set(cohort.pretrain_leads)

        lsemvae_runner.run(cfg)
        state = torch.load(tmp_path / "lsemvae" / "pretrained.pt", map_location="cpu", weights_only=True)
        # The saved model carries every pretraining lead, so a later run may fine-tune
        # on any subset of them.
        assert all(any(f"encoders.{lead}." in key for key in state) for lead in cohort.pretrain_leads)

    @pytest.mark.slow
    def test_several_lead_groups_are_compared_in_one_run(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        results = lsemvae_runner.run(cfg, lead_groups=[("I",), ("I", "II")])
        assert sorted(results) == ["I", "I+II"]

    @pytest.mark.slow
    def test_an_unknown_lead_group_is_rejected_by_name(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        with pytest.raises(ValueError, match=r"\['V6'\]"):
            lsemvae_runner.run(cfg, lead_groups=[("V6",)])

    @pytest.mark.slow
    @pytest.mark.parametrize("fusion", ["poe", "moe", "hime"])
    def test_the_published_ablations_are_configuration_changes(self, tmp_path, fusion):
        cfg = lsemvae_cfg(tmp_path, **{"FUSION.LATENT_METHOD": fusion})
        results = lsemvae_runner.run(cfg)
        assert results["I+II"]["n_folds"] == 2

    @pytest.mark.slow
    def test_interpretation_writes_a_lead_importance_table(self, tmp_path):
        pytest.importorskip("captum")
        cfg = lsemvae_cfg(tmp_path, **{"OUTPUT.INTERPRET": True})
        lsemvae_runner.run(cfg)
        out = tmp_path / "lsemvae"
        assert (out / "latent_space.csv").exists()
        assert (out / "lead_agreement.json").exists()
        assert (out / "I+II" / "lead_importance.csv").exists()


class TestExamplesUseTheLibrary:
    """An example composes the library; it never redefines a model."""

    def test_neither_runner_defines_a_model_class(self):
        from pathlib import Path

        for path in (
            Path("examples/cardiovae_multimodal/runner.py"),
            Path("examples/lsemvae_ecg/runner.py"),
            Path("examples/cardiovae_multimodal/data.py"),
            Path("examples/lsemvae_ecg/data.py"),
        ):
            source = path.read_text(encoding="utf-8")
            assert "nn.Module" not in source, f"{path} defines a network; models belong in kalecardiac/"

    def test_both_runners_import_their_model_from_the_library(self):
        from pathlib import Path

        assert "from kalecardiac.model.embed import CardioVAE" in Path(
            "examples/cardiovae_multimodal/runner.py"
        ).read_text(encoding="utf-8")
        assert "from kalecardiac.model.embed import LSEMVAE" in Path("examples/lsemvae_ecg/runner.py").read_text(
            encoding="utf-8"
        )
