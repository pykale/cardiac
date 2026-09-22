"""The downstream predictor: embedding, fusion, missing modalities, and its output."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kalecardiac.model.embed import (
    FUSION_METHODS,
    AttentionFusion,
    ConcatFusion,
    FeatureEmbedder,
    MeanFusion,
    MultimodalPredictor,
    build_fusion,
    build_signal_encoder,
    modality_dropout,
)
from kalecardiac.model.predict import LinearHead

WIDTH = 6
LENGTH = 64


class Stub(nn.Module):
    """A trivial embedder, so a fusion test is about the fusion."""

    def __init__(self, out_dim: int = WIDTH) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(LENGTH, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1)[:, :LENGTH])


@pytest.fixture
def embedders():
    return {name: Stub() for name in ("a", "b", "c")}


@pytest.fixture
def inputs():
    return {name: torch.randn(4, 1, LENGTH) for name in ("a", "b", "c")}


class TestModalityDropout:
    def test_zero_probability_is_a_no_op(self):
        mask = torch.ones(4, 3)
        assert torch.equal(modality_dropout(mask, 0.0), mask)

    def test_at_least_one_modality_always_survives(self):
        mask = torch.ones(200, 3)
        dropped = modality_dropout(mask, 0.99)
        assert bool((dropped.sum(dim=1) >= 1).all())

    def test_the_input_mask_is_not_modified(self):
        mask = torch.ones(50, 3)
        modality_dropout(mask, 0.5)
        assert torch.equal(mask, torch.ones(50, 3))

    def test_an_already_absent_modality_is_not_revived(self):
        mask = torch.zeros(50, 3)
        mask[:, 0] = 1.0
        dropped = modality_dropout(mask, 0.5)
        assert bool((dropped[:, 1:] == 0).all())


class TestFusionBlocks:
    def test_concat_widens_the_output(self):
        fusion = ConcatFusion([4, 6])
        assert fusion.output_dim == 10
        fused, weights = fusion([torch.randn(3, 4), torch.randn(3, 6)])
        assert fused.shape == (3, 10)
        assert weights is None

    def test_mean_keeps_the_width_constant(self):
        fusion = MeanFusion([5, 5, 5])
        assert fusion.output_dim == 5
        fused, weights = fusion([torch.randn(3, 5)] * 3)
        assert fused.shape == (3, 5)
        assert weights.shape == (3, 3)

    def test_mean_ignores_an_absent_modality(self):
        fusion = MeanFusion([2, 2])
        first, second = torch.ones(1, 2), torch.full((1, 2), 5.0)
        mask = torch.tensor([[1.0, 0.0]])
        fused, _ = fusion([first, second], mask)
        # The placeholder starts at zero, so the absent modality's contribution must be
        # excluded by the weighting rather than averaged in.
        assert torch.allclose(fused, first)

    def test_mean_requires_equal_widths(self):
        with pytest.raises(ValueError, match="same width"):
            MeanFusion([4, 6])

    def test_attention_weights_sum_to_one_over_present_modalities(self):
        fusion = AttentionFusion([5, 5, 5], attention_dim=4)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        _, weights = fusion([torch.randn(2, 5)] * 3, mask)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)
        assert float(weights[0, 2]) == pytest.approx(0.0, abs=1e-6)

    def test_attention_survives_a_subject_with_nothing_present(self):
        fusion = AttentionFusion([5, 5], attention_dim=4)
        fused, weights = fusion([torch.randn(2, 5)] * 2, torch.zeros(2, 2))
        assert torch.isfinite(fused).all() and torch.isfinite(weights).all()

    def test_attention_requires_equal_widths(self):
        with pytest.raises(ValueError, match="same width"):
            AttentionFusion([4, 6])

    def test_a_placeholder_replaces_an_absent_modality_rather_than_zeros(self):
        fusion = ConcatFusion([3, 3])
        with torch.no_grad():
            fusion.placeholders[1].fill_(2.0)
        fused, _ = fusion([torch.ones(1, 3), torch.ones(1, 3)], torch.tensor([[1.0, 0.0]]))
        assert torch.allclose(fused[0, 3:], torch.full((3,), 2.0))

    def test_a_mismatched_modality_count_is_rejected(self):
        with pytest.raises(ValueError, match="expected 2 modalities"):
            ConcatFusion([3, 3])([torch.randn(2, 3)])

    def test_a_mismatched_width_is_rejected(self):
        with pytest.raises(ValueError, match="should be"):
            ConcatFusion([3, 3])([torch.randn(2, 3), torch.randn(2, 4)])

    def test_a_mask_of_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="mask must be"):
            ConcatFusion([3, 3])([torch.randn(2, 3)] * 2, torch.ones(3, 2))

    def test_no_modality_is_rejected(self):
        with pytest.raises(ValueError, match="at least one modality"):
            ConcatFusion([])

    def test_an_unknown_method_lists_what_exists(self):
        with pytest.raises(KeyError, match="available"):
            build_fusion("nonexistent", [4, 4])


class TestMultimodalPredictor:
    def test_produces_a_prediction_and_a_representation(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        output = predictor(inputs)
        assert output.prediction.shape == (4, 1)
        assert output.representation.shape == (4, predictor.fused_dim)
        assert sorted(output.embeddings) == ["a", "b", "c"]

    def test_concat_without_a_fusion_dim_applies_no_projection(self, embedders, inputs):
        # The faithful reproduction of the cardiac fine-tuning heads, which concatenate
        # encoder means directly.
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        assert predictor.fused_dim == 3 * WIDTH
        assert all(isinstance(layer, nn.Identity) for layer in predictor.projections.values())

    def test_a_fusion_dim_projects_every_modality(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1), fusion_dim=5)
        assert predictor.fused_dim == 3 * 5
        assert all(isinstance(layer, nn.Linear) for layer in predictor.projections.values())

    @pytest.mark.parametrize("method", sorted(FUSION_METHODS))
    def test_every_fusion_method_works_from_configuration(self, method, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1), method=method, fusion_dim=5)
        output = predictor(inputs)
        assert output.prediction.shape == (4, 1)
        assert torch.isfinite(output.prediction).all()

    def test_attention_returns_its_weights(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1), method="attention", fusion_dim=5)
        output = predictor(inputs)
        assert output.weights.shape == (4, 3)

    def test_concat_returns_no_weights(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        assert predictor(inputs).weights is None

    def test_a_present_mask_changes_the_prediction(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        predictor.eval()
        full = predictor(inputs, {name: torch.ones(4, dtype=torch.bool) for name in inputs})
        present = {name: torch.ones(4, dtype=torch.bool) for name in inputs}
        present["b"] = torch.zeros(4, dtype=torch.bool)
        partial = predictor(inputs, present)
        assert not torch.allclose(full.prediction, partial.prediction)

    def test_modality_dropout_applies_only_in_training(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1), modality_dropout=1.0)
        predictor.eval()
        first = predictor(inputs)
        second = predictor(inputs)
        assert torch.allclose(first.prediction, second.prediction)

    def test_a_missing_input_is_reported_by_name(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        with pytest.raises(KeyError, match=r"\['c'\]"):
            predictor({"a": inputs["a"], "b": inputs["b"]})

    def test_an_embedder_without_out_dim_is_rejected(self, inputs):
        with pytest.raises(AttributeError, match="out_dim"):
            MultimodalPredictor({"a": nn.Linear(4, 4)}, lambda width: LinearHead(width, 1))

    def test_no_embedder_is_rejected(self):
        with pytest.raises(ValueError, match="at least one modality"):
            MultimodalPredictor({}, lambda width: LinearHead(width, 1))

    def test_one_modality_is_a_unimodal_baseline(self, inputs):
        predictor = MultimodalPredictor({"a": Stub()}, lambda width: LinearHead(width, 1))
        assert predictor({"a": inputs["a"]}).prediction.shape == (4, 1)

    def test_gradients_reach_every_embedder(self, embedders, inputs):
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        predictor(inputs).prediction.sum().backward()
        assert all(p.grad is not None for p in predictor.embedders.parameters())


class TestFeatureEmbedder:
    def test_wraps_a_backbone_without_pretraining(self):
        backbone = build_signal_encoder("conv", in_channels=1, length=LENGTH, channels=(4,))
        embedder = FeatureEmbedder(backbone, out_dim=8)
        assert embedder.out_dim == 8
        assert embedder(torch.randn(3, 1, LENGTH)).shape == (3, 8)

    def test_without_a_projection_it_keeps_the_backbone_width(self):
        backbone = build_signal_encoder("conv", in_channels=1, length=LENGTH, channels=(4,))
        embedder = FeatureEmbedder(backbone)
        assert embedder.out_dim == backbone.out_dim

    def test_a_backbone_without_out_dim_is_rejected(self):
        with pytest.raises(AttributeError, match="out_dim"):
            FeatureEmbedder(nn.Linear(4, 4))

    def test_it_builds_a_supervised_baseline_with_no_pretraining(self, inputs):
        # The convolutional classifier both cardiac studies compare against is this
        # embedder plus a head -- not a second model class.
        embedders = {
            name: FeatureEmbedder(build_signal_encoder("conv", 1, LENGTH, channels=(4,)), out_dim=8) for name in inputs
        }
        predictor = MultimodalPredictor(embedders, lambda width: LinearHead(width, 1))
        assert predictor(inputs).prediction.shape == (4, 1)
