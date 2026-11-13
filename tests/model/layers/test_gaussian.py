"""Gaussian expert fusion: closed forms, masking, and the grouping fix.

PyKale tests the reusable Gaussian calculations. These tests cover Cardiac's
posterior type, masked product and model fusion registry.
"""

from __future__ import annotations

import math

import pytest
import torch
from kale.embed.multimodal_fusion import hierarchical_experts, mixture_of_experts

from kalecardiac.model.layers import (
    ABSENT_LOG_VAR,
    EXPERT_FUSIONS,
    GaussianHead,
    GaussianPosterior,
    build_expert_fusion,
    prior_expert,
    product_of_experts,
    reparameterize,
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
        assert reparameterize(torch.zeros(4, 8), torch.zeros(4, 8)).shape == (4, 8)

    def test_zero_variance_returns_the_mean(self):
        mean = torch.randn(4, 8)
        sample = reparameterize(mean, torch.full((4, 8), -60.0))
        assert torch.allclose(sample, mean, atol=1e-6)

    def test_gradients_flow_to_both_parameters(self):
        mean = torch.zeros(4, 8, requires_grad=True)
        log_var = torch.zeros(4, 8, requires_grad=True)
        reparameterize(mean, log_var).sum().backward()
        assert mean.grad is not None and log_var.grad is not None
        # The mean's gradient is exactly one; the trick's whole point.
        assert torch.allclose(mean.grad, torch.ones_like(mean))

    def test_it_samples_in_evaluation_mode_too(self):
        # A documented departure from the reference implementations, which returned the
        # mean whenever the module was not training and so silently changed the ELBO.
        mean, log_var = torch.zeros(1000, 4), torch.zeros(1000, 4)
        assert not torch.allclose(reparameterize(mean, log_var), mean)

    def test_a_generator_makes_a_draw_reproducible(self):
        mean, log_var = torch.zeros(4, 8), torch.zeros(4, 8)
        first = reparameterize(mean, log_var, generator=torch.Generator().manual_seed(1))
        second = reparameterize(mean, log_var, generator=torch.Generator().manual_seed(1))
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
    def test_weighted_mixture_matches_pykale_and_returns_a_posterior(self, experts):
        mean, log_var = experts
        weights = torch.arange(1, mean.shape[0] + 1, dtype=mean.dtype)
        fused = build_expert_fusion("moe", weights=weights)(mean, log_var)
        core_mean, core_log_var = mixture_of_experts(mean, log_var, weights=weights)
        assert isinstance(fused, GaussianPosterior)
        assert torch.equal(fused.mean, core_mean)
        assert torch.equal(fused.log_var, core_log_var)

    @pytest.mark.parametrize("method", ["moe", "barycenter"])
    def test_only_the_ratio_of_the_weights_matters(self, method, experts):
        # The former normalization floored the weight total at 1e-8, so tiny weights
        # pulled the fused posterior towards zero.
        mean, log_var = experts
        unit = build_expert_fusion(method, weights=torch.ones(mean.shape[0]))(mean, log_var)
        tiny = build_expert_fusion(method, weights=torch.full((mean.shape[0],), 1e-12))(mean, log_var)
        torch.testing.assert_close(tiny.mean, unit.mean)
        torch.testing.assert_close(tiny.log_var, unit.log_var)

    @pytest.mark.parametrize(
        "method,option,value",
        [
            ("moe", "weights", torch.arange(1.0, 6.0)),
            ("barycenter", "weights", torch.arange(1.0, 6.0)),
            ("hime", "num_groups", 2),
        ],
    )
    def test_registry_options_can_be_passed_positionally(self, method, option, value, experts):
        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        positional = EXPERT_FUSIONS[method](mean, log_var, mask, value)
        keyword = EXPERT_FUSIONS[method](mean, log_var, mask=mask, **{option: value})
        assert torch.equal(positional.mean, keyword.mean)
        assert torch.equal(positional.log_var, keyword.log_var)

    def test_tiny_equal_weights_average_two_experts(self):
        mean = torch.tensor([[[0.0]], [[10.0]]])
        fused = build_expert_fusion("moe", weights=torch.tensor([1e-12, 1e-12]))(mean, torch.zeros_like(mean))
        assert float(fused.mean) == pytest.approx(5.0)

    def test_hierarchical_group_option_is_forwarded_to_pykale(self, experts):
        mean, log_var = experts
        fused = build_expert_fusion("hime", num_groups=2)(mean, log_var)
        core_mean, core_log_var = hierarchical_experts(mean, log_var, num_groups=2)
        assert torch.equal(fused.mean, core_mean)
        assert torch.equal(fused.log_var, core_log_var)

    @pytest.mark.parametrize("method", sorted(EXPERT_FUSIONS))
    def test_per_subject_masks_ignore_absent_experts_and_their_gradients(self, method):
        # Different subjects lose different experts in the same batch. Keep the
        # expert axis intact because hierarchical fusion depends on its order.
        mean = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]],
            requires_grad=True,
        )
        log_var = torch.zeros_like(mean, requires_grad=True)
        mask = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.bool)
        fuse = build_expert_fusion(method, **({"num_groups": 2} if method == "hime" else {}))

        fused = fuse(mean, log_var, mask)
        changed_absent = mean.detach() + (~mask).unsqueeze(-1) * 1000.0
        assert torch.allclose(fused.mean, fuse(changed_absent, log_var.detach(), mask).mean)
        fused.mean.sum().backward()

        assert fused.mean.shape == (3, 1)
        assert mean.grad is not None
        assert torch.equal(mean.grad[~mask], torch.zeros_like(mean.grad[~mask]))
        assert bool((mean.grad[mask] > 0).all())

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

    def test_non_tensor_parameters_are_rejected(self):
        with pytest.raises(TypeError, match="must be tensors"):
            product_of_experts([[[0.0]]], torch.zeros(1, 1, 1))

    def test_a_non_tensor_mask_is_rejected(self, experts):
        mean, log_var = experts
        with pytest.raises(TypeError, match="mask must be a tensor"):
            product_of_experts(mean, log_var, [[1.0] * mean.shape[1]] * mean.shape[0])


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


class TestKaleCardiacImports:
    """The fusion names KaleCardiac kept return PyKale's values as a posterior."""

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("mixture_of_experts", {"weights": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}),
            ("mean_of_experts", {}),
            ("wasserstein_barycenter", {"weights": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}),
            ("hierarchical_experts", {"num_groups": 2}),
        ],
    )
    def test_returns_pykale_values_as_a_posterior(self, name, kwargs, experts):
        import kale.embed.multimodal_fusion as core

        import kalecardiac.model.layers as layers

        mean, log_var = experts
        mask = torch.ones(mean.shape[:2])
        mask[0, 0] = 0.0
        fused = getattr(layers, name)(mean, log_var, mask, **kwargs)
        core_mean, core_log_var = getattr(core, name)(mean, log_var, mask, **kwargs)
        assert isinstance(fused, GaussianPosterior)
        assert torch.equal(fused.mean, core_mean)
        assert torch.equal(fused.log_var, core_log_var)

    def test_the_registry_uses_the_kept_functions(self):
        import kalecardiac.model.layers as layers

        assert EXPERT_FUSIONS["moe"] is layers.mixture_of_experts
        assert EXPERT_FUSIONS["mean"] is layers.mean_of_experts
        assert EXPERT_FUSIONS["hime"] is layers.hierarchical_experts
        assert EXPERT_FUSIONS["barycenter"] is layers.wasserstein_barycenter

    def test_get_expert_groups_matches_pykale(self):
        from kale.embed.multimodal_fusion import expert_groups

        from kalecardiac.model.layers import get_expert_groups

        assert get_expert_groups(13, 4) == expert_groups(13, 4) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


#: The fusions computed by PyKale, by function name and registry name.
DELEGATED = [
    ("mixture_of_experts", "moe"),
    ("mean_of_experts", "mean"),
    ("hierarchical_experts", "hime"),
    ("wasserstein_barycenter", "barycenter"),
]


def _entry_points():
    """Each delegated fusion through its public function and through the registry."""
    import kalecardiac.model.layers as layers

    for function, key in DELEGATED:
        yield pytest.param(getattr(layers, function), id=function)
        yield pytest.param(EXPERT_FUSIONS[key], id=f"EXPERT_FUSIONS[{key}]")


class TestDelegatedValidation:
    """The fusions computed by PyKale reject invalid inputs rather than fusing them."""

    @pytest.mark.parametrize("fusion", list(_entry_points()))
    def test_a_non_binary_mask_is_rejected(self, fusion, experts):
        mean, log_var = experts
        with pytest.raises(ValueError, match="zero or one"):
            fusion(mean, log_var, torch.full(mean.shape[:2], 0.5))

    @pytest.mark.parametrize("fusion", list(_entry_points()))
    def test_a_boolean_mask_matches_a_zero_one_mask(self, fusion, experts):
        mean, log_var = experts
        present = torch.ones(mean.shape[:2], dtype=torch.bool)
        present[0, 0] = False
        as_bool = fusion(mean, log_var, present)
        as_float = fusion(mean, log_var, present.float())
        assert torch.equal(as_bool.mean, as_float.mean)
        assert torch.equal(as_bool.log_var, as_float.log_var)

    @pytest.mark.parametrize("fusion", list(_entry_points()))
    def test_mixed_floating_dtypes_are_rejected(self, fusion, experts):
        mean, log_var = experts
        with pytest.raises(TypeError, match="same floating dtype"):
            fusion(mean, log_var.double())

    @pytest.mark.parametrize("fusion", list(_entry_points()))
    def test_a_non_tensor_mask_is_rejected(self, fusion, experts):
        mean, log_var = experts
        with pytest.raises(TypeError, match="mask must be a tensor"):
            fusion(mean, log_var, [[1.0] * mean.shape[1]] * mean.shape[0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0], ids=["nan", "inf", "negative"])
    @pytest.mark.parametrize(
        "fusion",
        [
            p
            for p in _entry_points()
            if p.id
            in {"mixture_of_experts", "EXPERT_FUSIONS[moe]", "wasserstein_barycenter", "EXPERT_FUSIONS[barycenter]"}
        ],
    )
    def test_a_non_finite_or_negative_weight_is_rejected(self, fusion, bad, experts):
        mean, log_var = experts
        weights = torch.ones(mean.shape[0])
        weights[1] = bad
        with pytest.raises(ValueError, match="finite and nonnegative"):
            fusion(mean, log_var, weights=weights)
