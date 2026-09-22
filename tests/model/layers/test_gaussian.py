"""Gaussian expert fusion: closed forms, masking, and the grouping fix.

These are the mathematical properties the two cardiac models rest on, so they are
checked against the closed forms rather than against a recorded output.
"""

from __future__ import annotations

import math

import pytest
import torch

from kalecardiac.model.layers import (
    ABSENT_LOG_VAR,
    EXPERT_FUSIONS,
    GaussianHead,
    GaussianPosterior,
    build_expert_fusion,
    expert_groups,
    hierarchical_experts,
    mean_of_experts,
    mixture_of_experts,
    prior_expert,
    product_of_experts,
    reparameterise,
    wasserstein_barycenter,
)


class TestGaussianPosterior:
    def test_variance_and_stddev_follow_the_log_variance(self):
        posterior = GaussianPosterior(torch.zeros(2, 3), torch.full((2, 3), math.log(4.0)))
        assert torch.allclose(posterior.variance, torch.full((2, 3), 4.0))
        assert torch.allclose(posterior.stddev, torch.full((2, 3), 2.0))

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            GaussianPosterior(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_detach_and_to_preserve_values(self):
        posterior = GaussianPosterior(torch.randn(2, 3, requires_grad=True), torch.zeros(2, 3))
        detached = posterior.detach().to("cpu")
        assert not detached.mean.requires_grad
        assert torch.equal(detached.mean, posterior.mean.detach())


class TestReparameterise:
    def test_output_has_the_shape_of_the_mean(self):
        assert reparameterise(torch.zeros(4, 8), torch.zeros(4, 8)).shape == (4, 8)

    def test_zero_variance_returns_the_mean(self):
        mean = torch.randn(4, 8)
        sample = reparameterise(mean, torch.full((4, 8), -60.0))
        assert torch.allclose(sample, mean, atol=1e-6)

    def test_gradients_flow_to_both_parameters(self):
        mean = torch.zeros(4, 8, requires_grad=True)
        log_var = torch.zeros(4, 8, requires_grad=True)
        reparameterise(mean, log_var).sum().backward()
        assert mean.grad is not None and log_var.grad is not None
        # The mean's gradient is exactly one; the trick's whole point.
        assert torch.allclose(mean.grad, torch.ones_like(mean))

    def test_it_samples_in_evaluation_mode_too(self):
        # A documented departure from the reference implementations, which returned the
        # mean whenever the module was not training and so silently changed the ELBO.
        mean, log_var = torch.zeros(1000, 4), torch.zeros(1000, 4)
        assert not torch.allclose(reparameterise(mean, log_var), mean)

    def test_a_generator_makes_a_draw_reproducible(self):
        mean, log_var = torch.zeros(4, 8), torch.zeros(4, 8)
        first = reparameterise(mean, log_var, generator=torch.Generator().manual_seed(1))
        second = reparameterise(mean, log_var, generator=torch.Generator().manual_seed(1))
        assert torch.equal(first, second)


class TestProductOfExperts:
    def test_matches_the_closed_form(self, experts):
        mean, log_var = experts
        fused = product_of_experts(mean, log_var)
        precision = 1.0 / torch.exp(log_var)
        expected_mean = (mean * precision).sum(dim=0) / precision.sum(dim=0)
        assert torch.allclose(fused.mean, expected_mean, atol=1e-4)
        assert torch.allclose(fused.variance, 1.0 / precision.sum(dim=0), atol=1e-4)

    def test_the_product_is_never_less_certain_than_any_expert(self, experts):
        mean, log_var = experts
        fused = product_of_experts(mean, log_var)
        assert bool((fused.variance <= log_var.exp().min(dim=0).values + 1e-5).all())

    def test_one_expert_returns_that_expert(self):
        mean, log_var = torch.randn(1, 3, 4), torch.randn(1, 3, 4) * 0.1
        fused = product_of_experts(mean, log_var)
        assert torch.allclose(fused.mean, mean[0], atol=1e-4)

    def test_a_masked_expert_is_equivalent_to_removing_it(self, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0] = 0.0
        masked = product_of_experts(mean, log_var, mask)
        dropped = product_of_experts(mean[1:], log_var[1:])
        assert torch.allclose(masked.mean, dropped.mean, atol=1e-3)

    def test_an_absent_expert_is_given_negligible_precision(self):
        assert ABSENT_LOG_VAR >= 20.0
        mean = torch.tensor([[[5.0]], [[0.0]]])
        log_var = torch.zeros(2, 1, 1)
        mask = torch.tensor([[0.0], [1.0]])
        # The present expert's mean survives; the absent one contributes nothing.
        assert float(product_of_experts(mean, log_var, mask).mean) == pytest.approx(0.0, abs=1e-4)


class TestMixtureOfExperts:
    def test_equals_the_arithmetic_mean_of_the_parameters(self, experts):
        mean, log_var = experts
        fused = mixture_of_experts(mean, log_var)
        assert torch.allclose(fused.mean, mean.mean(dim=0), atol=1e-5)
        assert torch.allclose(fused.log_var, log_var.mean(dim=0), atol=1e-5)

    def test_mean_of_experts_is_the_unweighted_case(self, experts):
        mean, log_var = experts
        assert torch.allclose(mean_of_experts(mean, log_var).mean, mixture_of_experts(mean, log_var).mean)

    def test_weights_are_normalised_internally(self, experts):
        mean, log_var = experts
        equal = mixture_of_experts(mean, log_var, weights=torch.ones(5))
        scaled = mixture_of_experts(mean, log_var, weights=torch.full((5,), 7.0))
        assert torch.allclose(equal.mean, scaled.mean)

    def test_a_masked_expert_takes_no_share(self, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0] = 0.0
        assert torch.allclose(mixture_of_experts(mean, log_var, mask).mean, mean[1:].mean(dim=0), atol=1e-5)

    def test_a_subject_with_nothing_present_is_finite(self, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[:, 0] = 0.0
        fused = mixture_of_experts(mean, log_var, mask)
        assert torch.isfinite(fused.mean).all()

    def test_negative_weights_are_rejected(self, experts):
        mean, log_var = experts
        with pytest.raises(ValueError, match="non-negative"):
            mixture_of_experts(mean, log_var, weights=torch.tensor([-1.0, 1, 1, 1, 1]))

    def test_weights_of_the_wrong_shape_are_rejected(self, experts):
        mean, log_var = experts
        with pytest.raises(ValueError, match="weights must be"):
            mixture_of_experts(mean, log_var, weights=torch.ones(3))


class TestWassersteinBarycenter:
    def test_averages_standard_deviations_not_variances(self):
        # Two experts with variances 1 and 9: the barycenter's stddev is (1 + 3)/2 = 2,
        # so its variance is 4 -- not the variance mean of 5.
        mean = torch.zeros(2, 1, 1)
        log_var = torch.tensor([[[0.0]], [[math.log(9.0)]]])
        fused = wasserstein_barycenter(mean, log_var)
        assert float(fused.variance) == pytest.approx(4.0, rel=1e-4)

    def test_means_are_averaged(self, experts):
        mean, log_var = experts
        assert torch.allclose(wasserstein_barycenter(mean, log_var).mean, mean.mean(dim=0), atol=1e-5)

    def test_a_masked_expert_takes_no_share(self, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0] = 0.0
        fused = wasserstein_barycenter(mean, log_var, mask)
        assert torch.allclose(fused.mean, mean[1:].mean(dim=0), atol=1e-5)


class TestExpertGroups:
    def test_every_expert_lands_in_exactly_one_group(self):
        for count in range(1, 20):
            for groups in range(1, 6):
                assignment = expert_groups(count, groups)
                flattened = [index for group in assignment for index in group]
                assert sorted(flattened) == list(range(count)), (count, groups)

    def test_thirteen_experts_in_four_groups_keeps_all_thirteen(self):
        # The reference implementation took four slices of 13 // 4 == 3 and silently
        # discarded the thirteenth expert, which was the prior.
        assignment = expert_groups(13, 4)
        assert sum(len(group) for group in assignment) == 13
        assert [len(group) for group in assignment] == [4, 3, 3, 3]

    def test_groups_differ_in_size_by_at_most_one(self):
        sizes = [len(group) for group in expert_groups(14, 4)]
        assert max(sizes) - min(sizes) <= 1

    def test_more_groups_than_experts_gives_one_each(self):
        assert expert_groups(3, 10) == [[0], [1], [2]]

    @pytest.mark.parametrize("count,groups", [(0, 2), (5, 0)])
    def test_non_positive_arguments_are_rejected(self, count, groups):
        with pytest.raises(ValueError):
            expert_groups(count, groups)


class TestHierarchicalExperts:
    def test_one_group_reduces_to_a_product(self, experts):
        mean, log_var = experts
        fused = hierarchical_experts(mean, log_var, num_groups=1)
        assert torch.allclose(fused.mean, product_of_experts(mean, log_var).mean, atol=1e-5)

    def test_groups_of_one_reduce_to_a_mixture(self, experts):
        mean, log_var = experts
        fused = hierarchical_experts(mean, log_var, num_groups=mean.shape[0])
        assert torch.allclose(fused.mean, mixture_of_experts(mean, log_var).mean, atol=1e-4)

    def test_it_sits_between_a_product_and_a_mixture_in_certainty(self, experts):
        mean, log_var = experts
        product = product_of_experts(mean, log_var).variance.mean()
        mixture = mixture_of_experts(mean, log_var).variance.mean()
        hierarchical = hierarchical_experts(mean, log_var, num_groups=2).variance.mean()
        assert product <= hierarchical <= mixture

    def test_a_group_contributes_if_any_of_its_experts_does(self, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0] = 0.0
        fused = hierarchical_experts(mean, log_var, mask, num_groups=2)
        assert torch.isfinite(fused.mean).all()

    def test_a_subject_with_nothing_present_is_finite(self, experts):
        mean, log_var = experts
        fused = hierarchical_experts(mean, log_var, torch.zeros(mean.shape[:2]), num_groups=2)
        assert torch.isfinite(fused.mean).all()


class TestPriorExpert:
    def test_is_the_standard_normal(self):
        mean, log_var = prior_expert((1, 4, 8))
        assert torch.equal(mean, torch.zeros(1, 4, 8))
        assert torch.equal(log_var, torch.zeros(1, 4, 8))

    def test_mean_and_log_var_are_independent_tensors(self):
        mean, log_var = prior_expert((1, 2, 2))
        mean += 1.0
        assert torch.equal(log_var, torch.zeros(1, 2, 2))


class TestRegistry:
    @pytest.mark.parametrize("method", sorted(EXPERT_FUSIONS))
    def test_every_registered_fusion_returns_a_posterior(self, method, experts):
        mean, log_var = experts
        fuse = build_expert_fusion(method, **({"num_groups": 2} if method == "hime" else {}))
        fused = fuse(mean, log_var)
        assert fused.mean.shape == (4, 3)
        assert torch.isfinite(fused.mean).all() and torch.isfinite(fused.log_var).all()

    @pytest.mark.parametrize("method", sorted(EXPERT_FUSIONS))
    def test_every_registered_fusion_honours_a_mask(self, method, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0, 0] = 0.0
        fuse = build_expert_fusion(method, **({"num_groups": 2} if method == "hime" else {}))
        assert torch.isfinite(fuse(mean, log_var, mask).mean).all()

    @pytest.mark.parametrize("method", sorted(EXPERT_FUSIONS))
    def test_every_registered_fusion_passes_gradients(self, method, experts):
        mean, log_var = experts
        mean = mean.clone().requires_grad_(True)
        fuse = build_expert_fusion(method, **({"num_groups": 2} if method == "hime" else {}))
        fuse(mean, log_var).mean.sum().backward()
        assert mean.grad is not None and torch.isfinite(mean.grad).all()

    def test_an_unknown_method_lists_what_exists(self):
        with pytest.raises(KeyError, match="available"):
            build_expert_fusion("nonexistent")

    def test_an_option_that_does_not_apply_is_rejected_at_construction(self):
        with pytest.raises(TypeError, match="takes no option"):
            build_expert_fusion("poe", num_groups=4)


class TestShapeValidation:
    def test_a_two_dimensional_stack_is_rejected(self):
        with pytest.raises(ValueError, match="num_experts, batch, latent_dim"):
            product_of_experts(torch.zeros(4, 3), torch.zeros(4, 3))

    def test_no_experts_is_rejected(self):
        with pytest.raises(ValueError, match="at least one expert"):
            product_of_experts(torch.zeros(0, 4, 3), torch.zeros(0, 4, 3))

    def test_a_mask_of_the_wrong_shape_is_rejected(self, experts):
        mean, log_var = experts
        with pytest.raises(ValueError, match="mask must be"):
            product_of_experts(mean, log_var, torch.ones(2, 2))

    def test_mismatched_mean_and_log_var_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            product_of_experts(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))


class TestGaussianHead:
    def test_maps_features_to_a_posterior(self):
        head = GaussianHead(16, 4)
        posterior = head(torch.randn(3, 16))
        assert posterior.mean.shape == (3, 4)
        assert posterior.log_var.shape == (3, 4)

    def test_mean_and_log_var_come_from_different_parameters(self):
        head = GaussianHead(16, 4)
        posterior = head(torch.randn(3, 16))
        assert not torch.allclose(posterior.mean, posterior.log_var)

    def test_features_of_the_wrong_width_are_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            GaussianHead(16, 4)(torch.randn(3, 8))

    def test_non_positive_widths_are_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            GaussianHead(0, 4)
