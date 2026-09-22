"""Prediction heads: shapes, validation, and initialisation."""

from __future__ import annotations

import pytest
import torch

from kalecardiac.model.predict import LinearHead, MLPHead


class TestLinearHead:
    def test_one_score_per_subject_by_default(self):
        assert LinearHead(8)(torch.randn(5, 8)).shape == (5, 1)

    def test_several_scores_for_a_multiclass_endpoint(self):
        assert LinearHead(8, out_features=4)(torch.randn(5, 8)).shape == (5, 4)

    def test_it_keeps_its_bias(self):
        # Cross-entropy is not shift-invariant, and the intercept is what sets the
        # predicted base rate on an imbalanced cardiac cohort.
        assert LinearHead(8).linear.bias is not None

    def test_the_bias_starts_at_zero(self):
        assert torch.equal(LinearHead(8).linear.bias, torch.zeros(1))

    def test_the_weights_are_not_all_the_same(self):
        weight = LinearHead(16, 2).linear.weight
        assert float(weight.std()) > 0.0

    def test_a_representation_of_the_wrong_width_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            LinearHead(8)(torch.randn(5, 4))

    def test_a_three_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            LinearHead(8)(torch.randn(5, 2, 8))

    def test_non_positive_widths_are_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            LinearHead(0)

    def test_it_emits_logits_not_probabilities(self):
        scores = LinearHead(8)(torch.randn(200, 8) * 10)
        assert float(scores.min()) < 0.0


class TestMLPHead:
    def test_produces_the_requested_number_of_scores(self):
        assert MLPHead(8, out_features=3, hidden_dims=(16,))(torch.randn(5, 8)).shape == (5, 3)

    def test_dropout_makes_training_stochastic_and_evaluation_stable(self):
        head = MLPHead(8, hidden_dims=(32,), dropout=0.5)
        features = torch.randn(5, 8)
        head.train()
        assert not torch.allclose(head(features), head(features))
        head.eval()
        assert torch.allclose(head(features), head(features))

    def test_it_has_more_capacity_than_a_linear_head(self):
        linear = sum(p.numel() for p in LinearHead(8).parameters())
        mlp = sum(p.numel() for p in MLPHead(8, hidden_dims=(64,)).parameters())
        assert mlp > linear

    def test_a_representation_of_the_wrong_width_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            MLPHead(8)(torch.randn(5, 4))

    def test_gradients_reach_every_layer(self):
        head = MLPHead(8, hidden_dims=(16, 16))
        head(torch.randn(5, 8)).sum().backward()
        assert all(p.grad is not None for p in head.parameters())
