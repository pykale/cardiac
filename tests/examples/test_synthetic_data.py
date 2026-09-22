"""The synthetic generators: shapes, determinism, and a learnable endpoint."""

from __future__ import annotations

import numpy as np
import pytest

from examples.synthetic_data import (
    DEFAULT_LEADS,
    make_synthetic_ecg,
    make_synthetic_image_data,
    make_synthetic_multimodal_data,
)


class TestMakeSyntheticEcg:
    def test_produces_one_recording_per_subject(self):
        subjects, recordings, severity, leads = make_synthetic_ecg(num_subjects=8, num_samples=128)
        assert len(subjects) == 8
        assert sorted(recordings) == sorted(subjects)
        assert sorted(severity) == sorted(subjects)
        assert leads == DEFAULT_LEADS

    def test_the_shape_follows_the_leads_and_samples(self):
        _, recordings, _, _ = make_synthetic_ecg(num_subjects=4, leads=["I", "II"], num_samples=256)
        for recording in recordings.values():
            assert recording.shape == (2, 256)
            assert recording.dtype == np.float32

    def test_the_same_seed_gives_the_same_cohort(self):
        first = make_synthetic_ecg(num_subjects=4, num_samples=64, seed=3)[1]
        second = make_synthetic_ecg(num_subjects=4, num_samples=64, seed=3)[1]
        assert all(np.array_equal(first[name], second[name]) for name in first)

    def test_different_seeds_give_different_cohorts(self):
        first = make_synthetic_ecg(num_subjects=4, num_samples=64, seed=3)[1]
        second = make_synthetic_ecg(num_subjects=4, num_samples=64, seed=4)[1]
        assert not np.array_equal(first["S0000"], second["S0000"])

    def test_the_leads_differ_from_each_other(self):
        # Real leads are projections onto different axes; if the generator made them
        # identical, a lead-specific model would have nothing to learn.
        _, recordings, _, _ = make_synthetic_ecg(num_subjects=2, num_samples=256, noise=0.0)
        recording = recordings["S0000"]
        assert not np.allclose(recording[0], recording[1])

    def test_severity_shapes_the_recording(self):
        # Without this the endpoint would be unlearnable and every test of a model
        # would pass on noise.
        _, recordings, severity, _ = make_synthetic_ecg(num_subjects=40, num_samples=256, noise=0.05, seed=0)
        scores = np.array([severity[name] for name in recordings])
        energies = np.array([np.abs(recordings[name][0]).mean() for name in recordings])
        assert abs(float(np.corrcoef(scores, energies)[0, 1])) > 0.3

    def test_zero_effect_removes_the_dependence(self):
        _, recordings, severity, _ = make_synthetic_ecg(num_subjects=40, num_samples=256, effect=0.0, seed=0)
        scores = np.array([severity[name] for name in recordings])
        energies = np.array([np.abs(recordings[name][0]).mean() for name in recordings])
        assert abs(float(np.corrcoef(scores, energies)[0, 1])) < 0.35

    def test_the_output_is_finite(self):
        _, recordings, _, _ = make_synthetic_ecg(num_subjects=4, num_samples=128)
        assert all(np.isfinite(recording).all() for recording in recordings.values())

    @pytest.mark.parametrize("kwargs", [{"num_subjects": 0}, {"num_samples": 0}, {"leads": []}])
    def test_degenerate_arguments_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            make_synthetic_ecg(**kwargs)


class TestMakeSyntheticImageData:
    def test_produces_one_image_per_subject(self):
        images = make_synthetic_image_data(["a", "b"], size=(16, 16))
        assert sorted(images) == ["a", "b"]
        assert images["a"].shape == (1, 16, 16)

    def test_values_lie_in_the_unit_interval(self):
        # Matching what build_image_pipeline produces and what a sigmoid decoder can
        # reconstruct; a mismatch makes the reconstruction term minimised by a constant.
        images = make_synthetic_image_data(["a"], size=(16, 16))
        assert images["a"].min() >= 0.0 and images["a"].max() <= 1.0

    def test_several_channels_are_supported(self):
        images = make_synthetic_image_data(["a"], size=(8, 8), channels=3)
        assert images["a"].shape == (3, 8, 8)

    def test_severity_changes_the_image(self):
        low = make_synthetic_image_data(["a"], severity={"a": -2.0}, size=(16, 16), noise=0.0)["a"]
        high = make_synthetic_image_data(["a"], severity={"a": 2.0}, size=(16, 16), noise=0.0)["a"]
        assert float(high.sum()) > float(low.sum())

    def test_a_degenerate_size_is_rejected(self):
        with pytest.raises(ValueError):
            make_synthetic_image_data(["a"], size=(0, 8))


class TestMakeSyntheticMultimodalData:
    def test_both_modalities_cover_the_same_subjects(self):
        cohort = make_synthetic_multimodal_data(num_subjects=12, num_samples=128)
        assert sorted(cohort.ecg) == sorted(cohort.image) == sorted(cohort.subject_id)

    def test_images_can_be_left_out(self):
        cohort = make_synthetic_multimodal_data(num_subjects=6, num_samples=64, with_images=False)
        assert cohort.image == {}

    def test_the_positive_rate_is_met_exactly(self):
        cohort = make_synthetic_multimodal_data(num_subjects=40, num_samples=64, positive_rate=0.25)
        assert sum(cohort.labels.values()) == 10

    def test_the_label_is_the_thresholded_measurement(self):
        cohort = make_synthetic_multimodal_data(num_subjects=20, num_samples=64)
        positives = [cohort.values[name] for name, label in cohort.labels.items() if label == 1]
        negatives = [cohort.values[name] for name, label in cohort.labels.items() if label == 0]
        assert min(positives) > max(negatives)

    def test_the_endpoint_is_learnable_from_the_signal(self):
        cohort = make_synthetic_multimodal_data(num_subjects=60, num_samples=256, seed=0)
        labels = np.array([cohort.labels[name] for name in cohort.subject_id])
        energies = np.array([np.abs(cohort.ecg[name][0]).mean() for name in cohort.subject_id])
        assert abs(float(np.corrcoef(labels, energies)[0, 1])) > 0.2

    def test_both_modalities_share_the_severity(self):
        # A multimodal model needs something real to fuse.
        cohort = make_synthetic_multimodal_data(num_subjects=60, num_samples=128, seed=0)
        labels = np.array([cohort.labels[name] for name in cohort.subject_id])
        brightness = np.array([cohort.image[name].sum() for name in cohort.subject_id])
        assert abs(float(np.corrcoef(labels, brightness)[0, 1])) > 0.2

    def test_the_same_seed_gives_the_same_cohort(self):
        first = make_synthetic_multimodal_data(num_subjects=8, num_samples=64, seed=5)
        second = make_synthetic_multimodal_data(num_subjects=8, num_samples=64, seed=5)
        assert first.labels == second.labels
        assert np.array_equal(first.ecg["S0000"], second.ecg["S0000"])

    def test_an_impossible_positive_rate_is_rejected(self):
        with pytest.raises(ValueError, match="positive_rate"):
            make_synthetic_multimodal_data(positive_rate=0.0)

    def test_a_custom_lead_set_is_honoured(self):
        cohort = make_synthetic_multimodal_data(num_subjects=4, leads=["I", "V1", "V6"], num_samples=64)
        assert cohort.leads == ("I", "V1", "V6")
        assert cohort.ecg["S0000"].shape[0] == 3
