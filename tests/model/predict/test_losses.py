"""Objectives: closed forms, finiteness, gradients, and the behaviour each is for."""

from __future__ import annotations

import math

import pytest
import torch

from kalecardiac.model.layers import GaussianPosterior
from kalecardiac.model.predict import (
    binary_cross_entropy,
    class_weight_from_labels,
    cross_entropy,
    elbo_loss,
    gaussian_kl_divergence,
    latent_alignment_loss,
    mean_squared_error,
    positive_weight_from_labels,
    reconstruction_loss,
)


class TestGaussianKL:
    def test_the_standard_normal_has_zero_divergence(self):
        posterior = GaussianPosterior(torch.zeros(4, 8), torch.zeros(4, 8))
        assert float(gaussian_kl_divergence(posterior)) == pytest.approx(0.0, abs=1e-6)

    def test_matches_the_closed_form(self):
        mean = torch.randn(4, 8)
        log_var = torch.randn(4, 8) * 0.3
        expected = (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=-1).mean()
        assert float(gaussian_kl_divergence(GaussianPosterior(mean, log_var))) == pytest.approx(
            float(expected), rel=1e-5
        )

    def test_it_is_non_negative(self):
        for _ in range(20):
            posterior = GaussianPosterior(torch.randn(4, 8), torch.randn(4, 8))
            assert float(gaussian_kl_divergence(posterior)) >= -1e-5

    def test_moving_away_from_the_prior_increases_it(self):
        near = GaussianPosterior(torch.full((2, 4), 0.1), torch.zeros(2, 4))
        far = GaussianPosterior(torch.full((2, 4), 3.0), torch.zeros(2, 4))
        assert float(gaussian_kl_divergence(near)) < float(gaussian_kl_divergence(far))

    def test_an_extreme_log_variance_stays_finite(self):
        # An untrained encoder emits large log-variances, and exp of one overflows to
        # inf, which reaches every parameter as a NaN gradient in a single step.
        posterior = GaussianPosterior(torch.zeros(2, 4), torch.full((2, 4), 500.0))
        assert math.isfinite(float(gaussian_kl_divergence(posterior)))

    def test_free_bits_exempt_a_near_prior_dimension(self):
        posterior = GaussianPosterior(torch.full((2, 4), 0.01), torch.zeros(2, 4))
        assert float(gaussian_kl_divergence(posterior, free_bits=0.5)) > float(gaussian_kl_divergence(posterior))

    def test_negative_free_bits_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            gaussian_kl_divergence(GaussianPosterior(torch.zeros(2, 4), torch.zeros(2, 4)), free_bits=-1.0)

    def test_gradients_flow_to_both_parameters(self):
        mean = torch.randn(4, 8, requires_grad=True)
        log_var = torch.zeros(4, 8, requires_grad=True)
        gaussian_kl_divergence(GaussianPosterior(mean, log_var)).backward()
        assert torch.isfinite(mean.grad).all() and torch.isfinite(log_var.grad).all()


class TestReconstructionLoss:
    def test_a_perfect_reconstruction_costs_nothing(self):
        target = torch.randn(4, 1, 16)
        assert float(reconstruction_loss(target, target)) == pytest.approx(0.0, abs=1e-6)

    def test_it_sums_over_elements_and_averages_over_the_batch(self):
        target = torch.zeros(4, 1, 16)
        reconstruction = torch.ones(4, 1, 16)
        # Sixteen elements each off by one, averaged over four subjects.
        assert float(reconstruction_loss(reconstruction, target)) == pytest.approx(16.0)

    def test_the_value_does_not_depend_on_the_batch_size(self):
        small = reconstruction_loss(torch.ones(2, 1, 8), torch.zeros(2, 1, 8))
        large = reconstruction_loss(torch.ones(64, 1, 8), torch.zeros(64, 1, 8))
        assert float(small) == pytest.approx(float(large))

    def test_absent_subjects_are_excluded(self):
        reconstruction = torch.ones(4, 1, 8)
        target = torch.zeros(4, 1, 8)
        target[0] = 100.0
        present = torch.tensor([False, True, True, True])
        # The absent subject's placeholder would dominate if it were counted.
        assert float(reconstruction_loss(reconstruction, target, present)) == pytest.approx(8.0)

    def test_a_wholly_absent_modality_gives_zero_rather_than_nan(self):
        value = reconstruction_loss(torch.ones(4, 1, 8), torch.zeros(4, 1, 8), torch.zeros(4, dtype=torch.bool))
        assert float(value) == pytest.approx(0.0)

    def test_a_wholly_absent_modality_keeps_the_graph(self):
        reconstruction = torch.ones(4, 1, 8, requires_grad=True)
        value = reconstruction_loss(reconstruction, torch.zeros(4, 1, 8), torch.zeros(4, dtype=torch.bool))
        value.backward()
        assert reconstruction.grad is not None

    def test_binary_cross_entropy_suits_a_unit_interval_modality(self):
        target = torch.full((2, 1, 4), 0.5)
        value = reconstruction_loss(torch.full((2, 1, 4), 0.5), target, kind="bce")
        assert float(value) == pytest.approx(4 * math.log(2), rel=1e-4)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="must match"):
            reconstruction_loss(torch.zeros(2, 4), torch.zeros(2, 5))

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown reconstruction kind"):
            reconstruction_loss(torch.zeros(2, 4), torch.zeros(2, 4), kind="huber")


class TestElbo:
    @pytest.fixture
    def parts(self):
        targets = {"a": torch.randn(4, 1, 16), "b": torch.rand(4, 1, 8)}
        reconstructions = {name: torch.zeros_like(value) for name, value in targets.items()}
        posterior = GaussianPosterior(torch.randn(4, 8) * 0.1, torch.zeros(4, 8))
        return reconstructions, targets, posterior

    def test_returns_a_scalar_and_its_terms(self, parts):
        loss, terms = elbo_loss(*parts)
        assert loss.dim() == 0
        assert {"recon", "kl", "recon_a", "recon_b"} <= set(terms)

    def test_per_modality_weights_scale_their_terms(self, parts):
        reconstructions, targets, posterior = parts
        plain, _ = elbo_loss(reconstructions, targets, posterior)
        weighted, terms = elbo_loss(reconstructions, targets, posterior, weights={"a": 10.0})
        assert float(weighted) > float(plain)
        # The reported per-modality term is unweighted, so it stays comparable.
        assert terms["recon_a"] == pytest.approx(elbo_loss(reconstructions, targets, posterior)[1]["recon_a"])

    def test_the_scale_factor_applies_only_to_reconstruction(self, parts):
        reconstructions, targets, posterior = parts
        full, terms = elbo_loss(reconstructions, targets, posterior, scale_factor=1.0)
        scaled, _ = elbo_loss(reconstructions, targets, posterior, scale_factor=0.0)
        assert float(scaled) == pytest.approx(terms["kl"], rel=1e-5)
        assert float(full) > float(scaled)

    def test_annealing_scales_only_the_kl(self, parts):
        reconstructions, targets, posterior = parts
        cold, terms = elbo_loss(reconstructions, targets, posterior, annealing_factor=0.0)
        assert float(cold) == pytest.approx(terms["recon"], rel=1e-5)

    def test_absent_subjects_do_not_contribute(self, parts):
        reconstructions, targets, posterior = parts
        present = {"a": torch.zeros(4, dtype=torch.bool), "b": torch.ones(4, dtype=torch.bool)}
        _, terms = elbo_loss(reconstructions, targets, posterior, present=present)
        assert terms["recon_a"] == pytest.approx(0.0)

    def test_a_modality_with_no_target_is_skipped(self, parts):
        reconstructions, targets, posterior = parts
        _, terms = elbo_loss(reconstructions, {"a": targets["a"]}, posterior)
        assert "recon_b" not in terms

    def test_nothing_to_reconstruct_is_rejected(self, parts):
        reconstructions, _, posterior = parts
        with pytest.raises(ValueError, match="no modality to reconstruct"):
            elbo_loss(reconstructions, {"z": torch.zeros(4, 1, 2)}, posterior)

    def test_the_objective_is_finite_and_differentiable(self):
        targets = {"a": torch.randn(4, 1, 16)}
        reconstructions = {"a": torch.randn(4, 1, 16, requires_grad=True)}
        mean = torch.randn(4, 8, requires_grad=True)
        posterior = GaussianPosterior(mean, torch.zeros(4, 8))
        loss, _ = elbo_loss(reconstructions, targets, posterior)
        loss.backward()
        assert math.isfinite(float(loss))
        assert torch.isfinite(reconstructions["a"].grad).all() and torch.isfinite(mean.grad).all()


class TestLatentAlignment:
    def test_identical_posteriors_align_perfectly(self):
        joint = GaussianPosterior(torch.randn(4, 8), torch.zeros(4, 8))
        per_modality = {name: GaussianPosterior(joint.mean.clone(), torch.zeros(4, 8)) for name in "ab"}
        assert float(latent_alignment_loss(per_modality, joint)) == pytest.approx(0.0, abs=1e-6)

    def test_divergent_posteriors_cost_more(self):
        joint = GaussianPosterior(torch.zeros(4, 8), torch.zeros(4, 8))
        near = {"a": GaussianPosterior(torch.full((4, 8), 0.1), torch.zeros(4, 8))}
        far = {"a": GaussianPosterior(torch.full((4, 8), 3.0), torch.zeros(4, 8))}
        assert float(latent_alignment_loss(near, joint)) < float(latent_alignment_loss(far, joint))

    def test_the_scale_does_not_depend_on_the_modality_count(self):
        joint = GaussianPosterior(torch.zeros(4, 8), torch.zeros(4, 8))
        posterior = GaussianPosterior(torch.ones(4, 8), torch.zeros(4, 8))
        two = latent_alignment_loss({name: posterior for name in "ab"}, joint)
        twelve = latent_alignment_loss({str(index): posterior for index in range(12)}, joint)
        assert float(two) == pytest.approx(float(twelve))

    def test_the_joint_is_detached_so_it_is_not_dragged(self):
        # The term moves the modality encoders towards the consensus, not the consensus
        # towards whichever modality is furthest away.
        joint_mean = torch.zeros(4, 8, requires_grad=True)
        modality_mean = torch.ones(4, 8, requires_grad=True)
        loss = latent_alignment_loss(
            {"a": GaussianPosterior(modality_mean, torch.zeros(4, 8))},
            GaussianPosterior(joint_mean, torch.zeros(4, 8)),
        )
        loss.backward()
        assert joint_mean.grad is None
        assert modality_mean.grad is not None

    def test_absent_subjects_are_skipped(self):
        joint = GaussianPosterior(torch.zeros(4, 8), torch.zeros(4, 8))
        means = torch.zeros(4, 8)
        means[0] = 100.0
        per_modality = {"a": GaussianPosterior(means, torch.zeros(4, 8))}
        present = {"a": torch.tensor([False, True, True, True])}
        assert float(latent_alignment_loss(per_modality, joint, present)) == pytest.approx(0.0, abs=1e-6)

    def test_no_modality_gives_zero(self):
        joint = GaussianPosterior(torch.zeros(4, 8), torch.zeros(4, 8))
        assert float(latent_alignment_loss({}, joint)) == 0.0


class TestDiscriminativeLosses:
    def test_binary_cross_entropy_on_a_confident_correct_prediction_is_small(self):
        logits = torch.tensor([10.0, -10.0])
        targets = torch.tensor([1.0, 0.0])
        assert float(binary_cross_entropy(logits, targets)) < 1e-3

    def test_binary_cross_entropy_accepts_a_trailing_axis(self):
        flat = binary_cross_entropy(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]))
        column = binary_cross_entropy(torch.tensor([[0.5], [-0.5]]), torch.tensor([1.0, 0.0]))
        assert float(flat) == pytest.approx(float(column))

    def test_a_positive_weight_raises_the_cost_of_a_missed_positive(self):
        logits = torch.tensor([-5.0, 5.0])
        targets = torch.tensor([1.0, 0.0])
        plain = binary_cross_entropy(logits, targets)
        weighted = binary_cross_entropy(logits, targets, pos_weight=torch.tensor(5.0))
        assert float(weighted) > float(plain)

    def test_cross_entropy_handles_float_targets_from_a_numeric_column(self):
        logits = torch.randn(4, 3)
        assert math.isfinite(float(cross_entropy(logits, torch.tensor([0.0, 1.0, 2.0, 1.0]))))

    def test_cross_entropy_needs_more_than_one_column(self):
        with pytest.raises(ValueError, match="num_classes >= 2"):
            cross_entropy(torch.randn(4, 1), torch.zeros(4))

    def test_mean_squared_error_is_zero_on_a_perfect_prediction(self):
        targets = torch.randn(8)
        assert float(mean_squared_error(targets, targets)) == pytest.approx(0.0, abs=1e-6)

    def test_mean_squared_error_accepts_a_trailing_axis(self):
        targets = torch.randn(8)
        assert float(mean_squared_error(targets.unsqueeze(1), targets)) == pytest.approx(0.0, abs=1e-6)


class TestClassWeights:
    def test_positive_weight_is_the_negative_to_positive_ratio(self):
        labels = torch.tensor([0.0] * 30 + [1.0] * 10)
        assert float(positive_weight_from_labels(labels)) == pytest.approx(3.0)

    def test_positive_weight_survives_a_split_with_no_positives(self):
        assert float(positive_weight_from_labels(torch.zeros(10))) == 1.0

    def test_class_weights_favour_the_rare_class(self):
        weights = class_weight_from_labels(torch.tensor([0] * 30 + [1] * 10), num_classes=2)
        assert float(weights[1]) > float(weights[0])

    def test_class_weights_are_normalised_to_mean_one(self):
        weights = class_weight_from_labels(torch.tensor([0] * 30 + [1] * 10 + [2] * 20), num_classes=3)
        assert float(weights.mean()) == pytest.approx(1.0)

    def test_an_absent_class_gets_weight_one_not_infinity(self):
        weights = class_weight_from_labels(torch.tensor([0, 0, 1]), num_classes=3)
        assert torch.isfinite(weights).all()

    def test_fewer_than_two_classes_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            class_weight_from_labels(torch.zeros(4), num_classes=1)
