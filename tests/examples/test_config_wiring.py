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
from kalecardiac.auto import get_ecg_transform_from_cfg, pretrain
from kalecardiac.loaddata import MultimodalDataset
from tests.examples.test_example_workflows import cardiovae_cfg, lsemvae_cfg

SIX = 6
SAMPLES = 64


# --------------------------------------------------------------------------- #
# ECG.SOURCE_SAMPLING_RATE
# --------------------------------------------------------------------------- #


class TestSourceSamplingRate:
    def test_cardiovae_resamples_each_lead_before_flattening(self, tmp_path):
        # Each lead is a different constant, so the strip must hold only those constants.
        # Resampling *after* flattening runs the filter across neighbouring leads and
        # leaves values between them; the polyphase filter's own ripple on a constant is
        # three thousandths.
        cfg = cardiovae_cfg(
            tmp_path,
            **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.NORMALIZE": False, "ECG.LENGTH": SIX * SAMPLES * 2},
        )
<<<<<<< HEAD
        pipeline = get_ecg_transform_from_cfg(cfg, num_leads=SIX)
=======
        pipeline = cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX)
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        recording = np.repeat(np.arange(SIX, dtype=np.float32)[:, None], SAMPLES, axis=1)

        strip = pipeline(recording)

        assert strip.shape == (1, SIX * SAMPLES * 2)
        assert np.allclose(strip, np.round(strip), atol=0.05), "a resample crossed from one lead into another"

    def test_cardiovae_interleaves_the_leads_as_the_reference_does(self, tmp_path):
        # The reference flattens a (samples, leads) recording row by row, so each time
        # step's leads sit together -- the layout the published checkpoint expects.
        cfg = cardiovae_cfg(tmp_path, **{"ECG.NORMALIZE": False, "ECG.LENGTH": SIX * SAMPLES})
        recording = np.arange(SIX * SAMPLES, dtype=np.float32).reshape(SIX, SAMPLES)
<<<<<<< HEAD
        strip = get_ecg_transform_from_cfg(cfg, num_leads=SIX)(recording)
=======
        strip = cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX)(recording)
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        assert np.array_equal(strip[0], recording.T.reshape(-1))
        assert np.array_equal(strip[0, :SIX], recording[:, 0])

    def test_cardiovae_normalizes_each_lead_before_interleaving(self, tmp_path):
        # Per lead, as the reference does: a lead recorded at ten times the amplitude of
        # another comes out on the same scale.
        cfg = cardiovae_cfg(tmp_path, **{"ECG.LENGTH": 2 * SAMPLES})
        generator = np.random.default_rng(0)
        recording = np.vstack([generator.normal(size=SAMPLES), 10 * generator.normal(size=SAMPLES)])
<<<<<<< HEAD
        strip = get_ecg_transform_from_cfg(cfg, num_leads=2)(recording)
=======
        strip = cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=2)(recording)
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        leads = strip[0].reshape(SAMPLES, 2).T
        assert np.allclose(leads.std(axis=1), 1.0, atol=1e-4)

    def test_cardiovae_crops_each_lead_to_its_share(self, tmp_path):
        # A longer recording loses the end of every lead, never a whole lead.
        cfg = cardiovae_cfg(tmp_path, **{"ECG.NORMALIZE": False, "ECG.LENGTH": SIX * SAMPLES})
        recording = np.repeat(np.arange(SIX, dtype=np.float32)[:, None], 2 * SAMPLES, axis=1)
<<<<<<< HEAD
        strip = get_ecg_transform_from_cfg(cfg, num_leads=SIX)(recording)
=======
        strip = cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX)(recording)
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        assert sorted(set(strip[0].tolist())) == list(range(SIX))

    def test_a_length_that_does_not_divide_among_the_leads_is_refused(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ECG.LENGTH": SIX * SAMPLES + 1})
        with pytest.raises(ValueError, match="whole number of samples"):
<<<<<<< HEAD
            get_ecg_transform_from_cfg(cfg, num_leads=SIX)

    def test_cardiovae_pipeline_lists_the_steps_in_order(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SIX * SAMPLES * 2})
        steps = repr(get_ecg_transform_from_cfg(cfg, num_leads=SIX))
        names = ("interpolate_signal", "resample_signal", "crop_or_pad_signal", "normalize_signal", "interleave_leads")
=======
            cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX)

    def test_cardiovae_pipeline_lists_the_steps_in_order(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SIX * SAMPLES * 2})
        steps = repr(cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX))
        names = ("interpolate_signal", "resample_signal", "crop_or_pad_signal", "normalize_signal", "_interleave_leads")
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
        order = [steps.index(name) for name in names]
        assert order == sorted(order), steps

    def test_no_resample_when_the_rates_agree(self, tmp_path):
        cfg = cardiovae_cfg(tmp_path)
<<<<<<< HEAD
        assert "resample" not in repr(get_ecg_transform_from_cfg(cfg, num_leads=SIX))
=======
        assert "resample" not in repr(cardiovae_data.ecg_pipeline(cfg, flatten=True, num_leads=SIX))
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a

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
        # The example preprocesses with the library's transform, configured from its cfg.
        cfg = lsemvae_cfg(tmp_path, **{"ECG.SOURCE_SAMPLING_RATE": 250, "ECG.LENGTH": SAMPLES * 2})
        steps = repr(get_ecg_transform_from_cfg(cfg))
        assert steps.index("interpolate_signal") < steps.index("resample_signal") < steps.index("crop_or_pad_signal")


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

    @pytest.mark.parametrize(
        ("make_cfg", "runner"),
        [(cardiovae_cfg, cardiovae_runner), (lsemvae_cfg, lsemvae_runner)],
        ids=["cardiovae", "lsemvae"],
    )
    def test_the_message_points_at_the_fusion_ablation(self, tmp_path, make_cfg, runner):
        # The likeliest reason to change MODEL.NAME is to try another fusion, which is a
        # different key; the error says so.
        with pytest.raises(ValueError, match="FUSION.LATENT_METHOD"):
            runner.run(make_cfg(tmp_path, **{"MODEL.NAME": "multimodal_vae"}))


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

        given = cardiovae_cfg(tmp_path, **{"MODEL.PRETRAINED": str(path)})
        dataset = MultimodalDataset(cohort.pretrain_ids, cohort.sources)
        loaded = pretrain(given, cardiovae_runner.build_model(given, cohort), dataset, tmp_path / "run")
        for ours, theirs in zip(loaded.parameters(), saved.parameters(), strict=True):
            assert torch.equal(ours, theirs)

    def test_lsemvae_loads_the_checkpoint_it_is_given(self, tmp_path):
        cfg = lsemvae_cfg(tmp_path)
        cohort = lsemvae_data.load_cohort(cfg)
        saved = lsemvae_runner.build_model(cfg, cohort, cohort.pretrain_leads)
        path = tmp_path / "pretrained.pt"
        torch.save(saved.state_dict(), path)

        # Built over every pretraining lead, as the run builds it, so a checkpoint saved
        # from one run loads into the next.
        given = lsemvae_cfg(tmp_path, **{"MODEL.PRETRAINED": str(path)})
        dataset = MultimodalDataset(cohort.pretrain_ids, cohort.pretrain_sources)
        loaded = pretrain(
            given, lsemvae_runner.build_model(given, cohort, cohort.pretrain_leads), dataset, tmp_path / "run"
        )
        for ours, theirs in zip(loaded.parameters(), saved.parameters(), strict=True):
            assert torch.equal(ours, theirs)

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
