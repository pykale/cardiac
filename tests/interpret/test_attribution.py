"""Interpretation: attribution maps, lead ratios, wave segments, and ablation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kalecardiac.evaluate import roc_auc
from kalecardiac.interpret import (
    attribution_ratio,
    attribution_ratios,
    ecg_wave_segments,
    modality_ablation,
    modality_attributions,
    normalise_attribution,
    segment_attribution_ratios,
)
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask

LATENT = 6


@pytest.fixture
def predictor(cohort):
    vae = LSEMVAE(
        leads=cohort.leads,
        length=cohort.ecg[cohort.subject_id[0]].shape[1],
        latent_dim=LATENT,
        channels=(4,),
    )
    task = ClassificationTask(hidden_dims=())
    return MultimodalPredictor(vae.latent_embedders(frozen=False), task.build_head), task


class TestNormaliseAttribution:
    def test_rescales_to_the_unit_interval(self):
        normalised = normalise_attribution(np.array([-3.0, 0.0, 7.0]))
        assert normalised.min() == pytest.approx(0.0)
        assert normalised.max() == pytest.approx(1.0)

    def test_a_constant_map_becomes_zeros_rather_than_nan(self):
        assert np.array_equal(normalise_attribution(np.full(5, 2.0)), np.zeros(5))

    def test_the_shape_is_preserved(self):
        assert normalise_attribution(np.random.default_rng(0).normal(size=(2, 3, 4))).shape == (2, 3, 4)


class TestAttributionRatio:
    def test_a_uniform_map_clears_no_threshold(self):
        assert attribution_ratio(np.full(100, 5.0), threshold=0.5) == pytest.approx(0.0)

    def test_the_fraction_above_the_threshold_is_what_is_counted(self):
        values = np.linspace(0.0, 1.0, 101)
        assert attribution_ratio(values, threshold=0.5) == pytest.approx(51 / 101, abs=0.01)

    def test_a_lower_threshold_counts_more(self):
        values = np.random.default_rng(0).normal(size=200)
        assert attribution_ratio(values, 0.3) > attribution_ratio(values, 0.9)

    def test_a_threshold_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="must lie in"):
            attribution_ratio(np.zeros(5), threshold=1.5)

    def test_an_empty_map_is_rejected(self):
        with pytest.raises(ValueError, match="empty attribution"):
            attribution_ratio(np.array([]))


class TestAttributionRatios:
    def test_shares_sum_to_one(self):
        generator = np.random.default_rng(0)
        attributions = {lead: generator.normal(size=100) for lead in ("I", "II", "V1")}
        summary = attribution_ratios(attributions, threshold=0.5)
        assert sum(entry["share"] for entry in summary.values()) == pytest.approx(1.0)

    def test_a_lead_attributed_everywhere_takes_the_larger_share(self):
        # One lead with a broad plateau near its maximum, one with a single spike.
        broad = np.concatenate([np.ones(80), np.zeros(20)])
        spike = np.concatenate([np.zeros(99), [1.0]])
        summary = attribution_ratios({"broad": broad, "spike": spike}, threshold=0.7)
        assert summary["broad"]["share"] > summary["spike"]["share"]

    def test_modalities_of_different_sizes_are_comparable(self):
        # Normalised within each modality, so a 100-sample lead and a 10,000-pixel image
        # both report a fraction of themselves.
        short = np.concatenate([np.ones(50), np.zeros(50)])
        long = np.concatenate([np.ones(5000), np.zeros(5000)])
        summary = attribution_ratios({"short": short, "long": long}, threshold=0.7)
        assert summary["short"]["ratio"] == pytest.approx(summary["long"]["ratio"], abs=0.01)

    def test_all_zero_shares_when_nothing_clears_the_threshold(self):
        summary = attribution_ratios({"a": np.full(10, 3.0)}, threshold=0.5)
        assert summary["a"]["share"] == 0.0

    def test_no_attributions_is_rejected(self):
        with pytest.raises(ValueError, match="no attributions"):
            attribution_ratios({})


class TestSegmentAttributionRatios:
    def test_counts_only_what_falls_in_a_window(self):
        attribution = np.zeros(100)
        attribution[50] = 1.0
        summary = segment_attribution_ratios(attribution, {"R": [50], "T": [90]}, window=2, threshold=0.5)
        assert summary["R"]["n_points"] == 1
        assert summary["T"]["n_points"] == 0

    def test_shares_sum_to_one_over_the_segments(self):
        attribution = np.zeros(100)
        attribution[[10, 50]] = 1.0
        summary = segment_attribution_ratios(attribution, {"P": [10], "R": [50]}, window=1, threshold=0.5)
        assert sum(entry["share"] for entry in summary.values()) == pytest.approx(1.0)

    def test_a_wider_window_catches_more(self):
        attribution = np.zeros(100)
        attribution[45:55] = 1.0
        narrow = segment_attribution_ratios(attribution, {"R": [50]}, window=1, threshold=0.5)
        wide = segment_attribution_ratios(attribution, {"R": [50]}, window=10, threshold=0.5)
        assert wide["R"]["n_points"] > narrow["R"]["n_points"]

    def test_a_window_is_clipped_at_the_signal_edges(self):
        attribution = np.zeros(20)
        attribution[0] = 1.0
        summary = segment_attribution_ratios(attribution, {"P": [0]}, window=10, threshold=0.5)
        assert summary["P"]["n_points"] == 1

    def test_a_negative_window_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            segment_attribution_ratios(np.zeros(10), {"R": [5]}, window=-1)


class TestEcgWaveSegments:
    def test_finds_the_waves_of_a_beat_like_signal(self):
        neurokit = pytest.importorskip("neurokit2")
        signal = neurokit.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, random_state=0)
        segments = ecg_wave_segments(signal, sampling_rate=500)
        assert set(segments) == {"P", "Q", "R", "S", "T"}
        assert len(segments["R"]) > 5

    def test_indices_lie_inside_the_signal(self):
        neurokit = pytest.importorskip("neurokit2")
        signal = neurokit.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, random_state=0)
        segments = ecg_wave_segments(signal, sampling_rate=500)
        for indices in segments.values():
            assert all(0 <= index < len(signal) for index in indices)

    def test_a_multi_lead_recording_is_rejected(self):
        pytest.importorskip("neurokit2")
        with pytest.raises(ValueError, match="single lead"):
            ecg_wave_segments(np.zeros((6, 500)))

    def test_a_signal_with_no_beat_is_rejected_with_the_reason(self):
        pytest.importorskip("neurokit2")
        with pytest.raises(ValueError, match="not an ECG"):
            ecg_wave_segments(np.zeros(500), sampling_rate=500)


class TestModalityAttributions:
    def test_one_map_per_modality_in_its_own_shape(self, predictor, lead_dataset, loader_factory):
        pytest.importorskip("captum")
        model, _ = predictor
        batch = next(iter(loader_factory(lead_dataset, batch_size=4)))
        attributions = modality_attributions(model, batch.modalities, present=batch.present, n_steps=4)
        assert sorted(attributions) == sorted(batch.modalities)
        for name, values in attributions.items():
            assert values.shape == tuple(batch.modalities[name].shape)

    def test_the_maps_are_finite(self, predictor, lead_dataset, loader_factory):
        pytest.importorskip("captum")
        model, _ = predictor
        batch = next(iter(loader_factory(lead_dataset, batch_size=4)))
        attributions = modality_attributions(model, batch.modalities, n_steps=4)
        assert all(np.isfinite(values).all() for values in attributions.values())

    def test_a_present_mask_of_the_batch_size_is_expanded_correctly(self, predictor, lead_dataset, loader_factory):
        # Integrated gradients evaluates every path step in one pass, so the mask must
        # be repeated to match; a mismatch used to raise from inside the fusion.
        pytest.importorskip("captum")
        model, _ = predictor
        batch = next(iter(loader_factory(lead_dataset, batch_size=4)))
        attributions = modality_attributions(model, batch.modalities, present=batch.present, n_steps=16)
        assert attributions["I"].shape[0] == 4

    def test_the_model_is_left_in_the_mode_it_was_found_in(self, predictor, lead_dataset, loader_factory):
        pytest.importorskip("captum")
        model, _ = predictor
        model.train()
        batch = next(iter(loader_factory(lead_dataset, batch_size=4)))
        modality_attributions(model, batch.modalities, n_steps=4)
        assert model.training is True

    def test_no_modality_is_rejected(self, predictor):
        pytest.importorskip("captum")
        model, _ = predictor
        with pytest.raises(ValueError, match="no modality"):
            modality_attributions(model, {})

    def test_only_what_the_model_reads_is_attributed(self, cohort, lead_dataset, loader_factory):
        # The ordinary case: a loader built over every lead feeding a model fine-tuned
        # on three. A lead the model never reads has no attribution by definition, and
        # attributing it would fail deep inside the gradient machinery.
        pytest.importorskip("captum")
        vae = LSEMVAE(
            leads=cohort.leads, length=cohort.ecg[cohort.subject_id[0]].shape[1], latent_dim=LATENT, channels=(4,)
        )
        subset = ("I", "II", "aVF")
        task = ClassificationTask(hidden_dims=())
        model = MultimodalPredictor(vae.latent_embedders(subset, frozen=False), task.build_head)

        batch = next(iter(loader_factory(lead_dataset, batch_size=4)))
        attributions = modality_attributions(model, batch.modalities, present=batch.present, n_steps=4)
        assert sorted(attributions) == sorted(subset)

    def test_a_batch_sharing_no_modality_with_the_model_is_rejected(self, cohort, loader_factory):
        pytest.importorskip("captum")
        vae = LSEMVAE(
            leads=cohort.leads, length=cohort.ecg[cohort.subject_id[0]].shape[1], latent_dim=LATENT, channels=(4,)
        )
        task = ClassificationTask(hidden_dims=())
        model = MultimodalPredictor(vae.latent_embedders(["I"], frozen=False), task.build_head)
        with pytest.raises(ValueError, match="is read by this model"):
            modality_attributions(model, {"aVF": torch.randn(2, 1, 8)}, n_steps=4)


class TestModalityAblation:
    def test_reports_the_full_score_and_a_drop_per_modality(self, predictor, lead_dataset, loader_factory, cohort):
        model, task = predictor
        trainer = CardiacTrainer(model, task)
        loader = loader_factory(lead_dataset, batch_size=8)
        results = modality_ablation(trainer, loader, roc_auc, "label")
        assert "full" in results
        for lead in cohort.leads:
            assert f"without_{lead}" in results
            assert results[f"drop_{lead}"] == pytest.approx(results["full"] - results[f"without_{lead}"])

    def test_ablation_changes_the_score(self, predictor, lead_dataset, loader_factory):
        model, task = predictor
        trainer = CardiacTrainer(model, task)
        loader = loader_factory(lead_dataset, batch_size=8)
        results = modality_ablation(trainer, loader, roc_auc, "label", modalities=["I"])
        assert results["full"] != results["without_I"]

    def test_a_named_subset_is_honoured(self, predictor, lead_dataset, loader_factory):
        model, task = predictor
        trainer = CardiacTrainer(model, task)
        loader = loader_factory(lead_dataset, batch_size=8)
        results = modality_ablation(trainer, loader, roc_auc, "label", modalities=["I", "II"])
        assert set(results) == {"full", "without_I", "drop_I", "without_II", "drop_II"}

    def test_an_empty_loader_is_rejected(self, predictor):
        model, task = predictor
        with pytest.raises(ValueError, match="no batches"):
            modality_ablation(CardiacTrainer(model, task), [], roc_auc, "label")

    def test_a_missing_target_is_named(self, predictor, lead_dataset, loader_factory):
        model, task = predictor
        trainer = CardiacTrainer(model, task)
        loader = loader_factory(lead_dataset, batch_size=8)
        with pytest.raises(ValueError, match="no target 'value'"):
            modality_ablation(trainer, loader, roc_auc, "value")

    def test_it_ablates_what_the_model_reads_by_default(self, cohort, lead_dataset, loader_factory):
        # As with attribution: the loader carries every lead, the model reads three.
        vae = LSEMVAE(
            leads=cohort.leads, length=cohort.ecg[cohort.subject_id[0]].shape[1], latent_dim=LATENT, channels=(4,)
        )
        subset = ("I", "II", "aVF")
        task = ClassificationTask(hidden_dims=())
        trainer = CardiacTrainer(MultimodalPredictor(vae.latent_embedders(subset), task.build_head), task)

        results = modality_ablation(trainer, loader_factory(lead_dataset, batch_size=8), roc_auc, "label")
        assert set(results) == {"full", *(f"without_{lead}" for lead in subset), *(f"drop_{lead}" for lead in subset)}

    def test_the_original_batch_is_not_modified(self, predictor, lead_dataset, loader_factory):
        # Withholding is done on a copy, so a caller can reuse its loader afterwards.
        model, task = predictor
        trainer = CardiacTrainer(model, task)
        batches = list(loader_factory(lead_dataset, batch_size=8))
        before = batches[0].present["I"].clone()
        modality_ablation(trainer, batches, roc_auc, "label", modalities=["I"])
        assert torch.equal(batches[0].present["I"], before)
