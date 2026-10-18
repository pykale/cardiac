"""Every configuration key an example accepts changes what the example does.

``tests/test_config.py`` proves each key is *read*. This proves each read *does
something*: that resampling happens, that a checkpoint is loaded instead of pretraining,
that a misplaced model name is refused. A key that is read and then ignored would pass
the first check and fail here.

Each of these keys was once silently ignored, so each test below would have failed
against the code it guards.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from examples.cardiovae_multimodal import data as cardiovae_data
from examples.cardiovae_multimodal import runner as cardiovae_runner
from examples.lsemvae_ecg import data as lsemvae_data
from examples.lsemvae_ecg import runner as lsemvae_runner
from tests.examples.test_example_workflows import cardiovae_cfg, lsemvae_cfg

SIX = 6
SAMPLES = 64


# --------------------------------------------------------------------------- #
# ECG.SOURCE_SAMPLING_RATE
# --------------------------------------------------------------------------- #


class TestSourceSamplingRate:
    def test_cardiovae_resamples_each_lead_before_flattening(self, tmp_path):
        # Each lead is a different constant, so the flattened strip must be piecewise
        # constant with integer values. Resampling *after* flattening would interpolate
        # across each join between two leads and leave fractional values there.
        cfg = cardiovae_cfg(
            tmp_path,
            **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.STANDARDISE": False, "ECG.LENGTH": SIX * SAMPLES * 2},
        )
        pipeline = cardiovae_data.ecg_pipeline(cfg, flatten=True)
        recording = np.repeat(np.arange(SIX, dtype=np.float32)[:, None], SAMPLES, axis=1)

        strip = pipeline(recording)

        assert strip.shape == (1, SIX * SAMPLES * 2)
        assert np.allclose(strip, np.round(strip)), "a resample crossed the boundary between two leads"
        blocks = strip.reshape(SIX, SAMPLES * 2)
        assert np.array_equal(blocks, np.repeat(np.arange(SIX)[:, None], SAMPLES * 2, axis=1))

    def test_cardiovae_pipeline_lists_the_steps_in_order(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SIX * SAMPLES * 2})
        steps = repr(cardiovae_data.ecg_pipeline(cfg, flatten=True))
        order = [steps.index(name) for name in ("interpolate_missing", "resample", "_flatten_leads", "crop_or_pad")]
        assert order == sorted(order), steps

    def test_no_resample_when_the_rates_agree(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        assert "resample" not in repr(cardiovae_data.ecg_pipeline(cfg, flatten=True))

    def test_cardiovae_cohort_arrives_at_the_model_rate(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SIX * SAMPLES * 2})
        cohort = cardiovae_data.load_cohort(cfg)
        # Synthetic recordings are generated at 250 Hz and doubled to 500 Hz per lead.
        assert cohort.ecg_length == SIX * SAMPLES * 2
        # Provenance records what the model sees, not what the generator produced.
        subject = cohort.labelled_ids[0]
        assert cohort.sources["ecg"].provenance(subject, 0)["sampling_rate"] == cfg.ECG.SAMPLING_RATE

    def test_lsemvae_resamples_every_lead(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SAMPLES * 2})
        cohort = lsemvae_data.load_cohort(cfg)
        assert cohort.length == SAMPLES * 2
        lead = cohort.finetune_sources["I"].get(cohort.labelled_ids[0], 0)
        assert lead.shape == (1, SAMPLES * 2)

    def test_lsemvae_pipeline_resamples_after_interpolating(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SAMPLES * 2})
        steps = repr(lsemvae_data.signal_pipeline(cfg))
        assert steps.index("interpolate_missing") < steps.index("resample") < steps.index("crop_or_pad")


# --------------------------------------------------------------------------- #
# MODEL.KERNEL_SIZE and the CardioVAE fusion keys
# --------------------------------------------------------------------------- #


class TestModelKeys:
    def test_kernel_size_reaches_the_cardiovae_encoders(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"MODEL.KERNEL_SIZE": 5})
        model = cardiovae_runner.build_model(cfg, cardiovae_data.load_cohort(cfg))
        assert model.encoders["ecg"].backbone.kernel_size == 5
        assert model.encoders["cxr"].backbone.kernel_size == 5

    def test_kernel_size_reaches_the_lsemvae_encoders(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"MODEL.KERNEL_SIZE": 5})
        cohort = lsemvae_data.load_cohort(cfg)
        model = lsemvae_runner.build_model(cfg, cohort, cohort.pretrain_leads)
        assert {model.encoders[lead].backbone.kernel_size for lead in cohort.pretrain_leads} == {5}

    @pytest.mark.parametrize("fusion", ["poe", "moe", "hime"])
    def test_cardiovae_honours_the_latent_fusion(self, tmp_path, fusion):
        # CardioVAE used to hardcode a product of experts, so this key did nothing there.
        cfg = cardiovae_cfg(tmp_path, **{"FUSION.LATENT_METHOD": fusion})
        model = cardiovae_runner.build_model(cfg, cardiovae_data.load_cohort(cfg))
        assert model.fusion_name == fusion

    def test_cardiovae_honours_use_prior(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"FUSION.USE_PRIOR": False})
        model = cardiovae_runner.build_model(cfg, cardiovae_data.load_cohort(cfg))
        assert model.use_prior is False


# --------------------------------------------------------------------------- #
# MODEL.NAME
# --------------------------------------------------------------------------- #


class TestModelName:
    def test_cardiovae_refuses_another_examples_model(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"MODEL.NAME": "lsemvae"})
        with pytest.raises(ValueError, match="examples.lsemvae_ecg"):
            cardiovae_runner.run(cfg)
        assert not (tmp_path / "cardiovae").exists(), "the name must be checked before anything is written"

    def test_lsemvae_refuses_another_examples_model(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path, **{"MODEL.NAME": "cardiovae"})
        with pytest.raises(ValueError, match="examples.cardiovae_multimodal"):
            lsemvae_runner.run(cfg)
        assert not (tmp_path / "lsemvae").exists()

    @pytest.mark.parametrize("check", [cardiovae_runner.check_model_name, lsemvae_runner.check_model_name])
    def test_the_message_points_at_the_fusion_ablation(self, tmp_path, check):
        # The likeliest reason to change MODEL.NAME is to try another fusion, which is a
        # different key; the error says so.
        cfg = cardiovae_cfg(tmp_path, **{"MODEL.NAME": "multimodal_vae"})
        with pytest.raises(ValueError, match="FUSION.LATENT_METHOD"):
            check(cfg)


# --------------------------------------------------------------------------- #
# DATASET.MODALITIES and lead groups
# --------------------------------------------------------------------------- #


class TestWhatIsCompared:
    def test_two_modalities_give_each_alone_and_both_fused(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"DATASET.MODALITIES": ["ecg", "cxr"]})
        assert cardiovae_runner.resolve_arms(cfg, None) == [("ecg",), ("cxr",), ("ecg", "cxr")]

    def test_one_modality_gives_a_unimodal_run(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"DATASET.MODALITIES": ["ecg"]})
        assert cardiovae_runner.resolve_arms(cfg, None) == [("ecg",)]

    def test_explicit_arms_override_the_configuration(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        assert cardiovae_runner.resolve_arms(cfg, [("cxr",)]) == [("cxr",)]

    def test_an_unknown_modality_is_rejected_by_name(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        with pytest.raises(ValueError, match=r"\['mri'\]"):
            cardiovae_runner.resolve_arms(cfg, [("ecg", "mri")])

    def test_an_unknown_arm_fails_before_anything_is_written(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        with pytest.raises(ValueError, match="mri"):
            cardiovae_runner.run(cfg, arms=[("mri",)])
        assert not (tmp_path / "cardiovae").exists()

    def test_no_modality_is_rejected(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"DATASET.MODALITIES": []})
        with pytest.raises(ValueError, match="no modality to compare"):
            cardiovae_runner.resolve_arms(cfg, None)

    def test_lead_names_are_canonicalised(self, tmp_path):
        # The library treats "i" and "LEAD_II" as I and II; the example must too.
        cfg = lsemvae_cfg(tmp_path)
        assert lsemvae_runner.resolve_lead_groups(cfg, [("i", "LEAD_II")]) == [("I", "II")]

    def test_the_default_group_is_the_finetune_leads(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        assert lsemvae_runner.resolve_lead_groups(cfg, None) == [("I", "II")]

    def test_an_empty_lead_group_is_rejected_with_its_own_message(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        with pytest.raises(ValueError, match="no lead group to compare"):
            lsemvae_runner.resolve_lead_groups(cfg, [()])


# --------------------------------------------------------------------------- #
# MODEL.PRETRAINED
# --------------------------------------------------------------------------- #


class TestPretrainedCheckpoint:
    def test_cardiovae_loads_the_checkpoint_it_is_given(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
        cohort = cardiovae_data.load_cohort(cfg)
        saved = cardiovae_runner.build_model(cfg, cohort)
        path = tmp_path / "pretrained.pt"
        torch.save(saved.state_dict(), path)

        loaded = cardiovae_runner.load_pretrained(cardiovae_cfg(tmp_path, **{"MODEL.PRETRAINED": str(path)}), cohort)
        for ours, theirs in zip(loaded.parameters(), saved.parameters(), strict=True):
            assert torch.equal(ours, theirs)

    def test_lsemvae_loads_the_checkpoint_it_is_given(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        cohort = lsemvae_data.load_cohort(cfg)
        saved = lsemvae_runner.build_model(cfg, cohort, cohort.pretrain_leads)
        path = tmp_path / "pretrained.pt"
        torch.save(saved.state_dict(), path)

        loaded = lsemvae_runner.load_pretrained(lsemvae_cfg(tmp_path, **{"MODEL.PRETRAINED": str(path)}), cohort)
        for ours, theirs in zip(loaded.parameters(), saved.parameters(), strict=True):
            assert torch.equal(ours, theirs)

    def test_a_checkpoint_from_another_architecture_fails_loudly(self, tmp_path):
        # Half-loading would fine-tune on encoders that are partly random.
        cfg = cardiovae_cfg(tmp_path, **{"MODEL.LATENT_DIM": 8})
        cohort = cardiovae_data.load_cohort(cfg)
        path = tmp_path / "pretrained.pt"
        torch.save(cardiovae_runner.build_model(cfg, cohort).state_dict(), path)

        wider = cardiovae_cfg(tmp_path, **{"MODEL.LATENT_DIM": 16, "MODEL.PRETRAINED": str(path)})
        with pytest.raises(RuntimeError):
            cardiovae_runner.load_pretrained(wider, cohort)

    def test_a_missing_checkpoint_is_reported_by_path(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"MODEL.PRETRAINED": str(tmp_path / "absent.pt")})
        with pytest.raises(FileNotFoundError, match="absent.pt"):
            cardiovae_runner.load_pretrained(cfg, cardiovae_data.load_cohort(cfg))

    @pytest.mark.slow
    def test_cardiovae_skips_pretraining_when_given_a_checkpoint(self, tmp_path):
        first = cardiovae_cfg(tmp_path / "first", **{"OUTPUT.OUT_DIR": str(tmp_path / "first")})
        cardiovae_runner.run(first, arms=[("ecg",)])
        checkpoint = tmp_path / "first" / "pretrained.pt"
        assert checkpoint.exists() and (tmp_path / "first" / "pretrain").exists()

        second = cardiovae_cfg(
            tmp_path / "second",
            **{"OUTPUT.OUT_DIR": str(tmp_path / "second"), "MODEL.PRETRAINED": str(checkpoint)},
        )
        results = cardiovae_runner.run(second, arms=[("ecg",)])

        # No pretraining run, and no second checkpoint: the given one was used.
        assert not (tmp_path / "second" / "pretrain").exists()
        assert not (tmp_path / "second" / "pretrained.pt").exists()
        assert results["ecg"]["n_folds"] == 2

    @pytest.mark.slow
    def test_lsemvae_skips_pretraining_when_given_a_checkpoint(self, tmp_path):
        first = lsemvae_cfg(tmp_path / "first", **{"OUTPUT.OUT_DIR": str(tmp_path / "first")})
        lsemvae_runner.run(first)
        checkpoint = tmp_path / "first" / "pretrained.pt"

        second = lsemvae_cfg(
            tmp_path / "second",
            **{"OUTPUT.OUT_DIR": str(tmp_path / "second"), "MODEL.PRETRAINED": str(checkpoint)},
        )
        lsemvae_runner.run(second)
        assert not (tmp_path / "second" / "pretrain").exists()
        assert not (tmp_path / "second" / "pretrained.pt").exists()


# --------------------------------------------------------------------------- #
# ENDPOINT.NAME
# --------------------------------------------------------------------------- #


class TestCohortRecord:
    @pytest.mark.slow
    def test_cardiovae_records_the_cohort_and_its_endpoint(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ENDPOINT.NAME": "my_endpoint"})
        cardiovae_runner.run(cfg, arms=[("ecg",)])
        record = json.loads((tmp_path / "cardiovae" / "cohort.json").read_text(encoding="utf-8"))
        assert record["endpoint"] == "my_endpoint"
        assert record["synthetic"] is True
        assert record["n_labelled"] > 0
