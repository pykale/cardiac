"""Encoders and embedders: the contracts they check, and how they describe themselves."""

from __future__ import annotations

import pytest
from torch import nn

from kalecardiac.model.embed import FeatureEmbedder, LatentEmbedder, VariationalEncoder, build_signal_encoder
from kalecardiac.model.layers import Conv1dEncoder


def backbone() -> Conv1dEncoder:
    return Conv1dEncoder(in_channels=1, length=64, channels=(4,))


class TestBuildSignalEncoder:
    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="available: \\['conv', 'residual', 'transformer'\\]"):
            build_signal_encoder("lstm", 1, 64)


class TestContracts:
    def test_a_variational_encoder_needs_a_backbone_that_declares_its_width(self):
        with pytest.raises(AttributeError, match="must expose 'out_dim'"):
            VariationalEncoder(nn.Linear(4, 4), latent_dim=6)

    def test_a_latent_embedder_needs_a_variational_encoder(self):
        with pytest.raises(AttributeError, match="wrap a backbone in VariationalEncoder first"):
            LatentEmbedder(backbone())


class TestDescriptions:
    def test_a_variational_encoder_names_its_backbone_and_width(self):
        assert repr(VariationalEncoder(backbone(), latent_dim=6)).endswith("latent_dim=6)")

    def test_a_latent_embedder_says_whether_it_is_frozen_and_sampling(self):
        text = repr(LatentEmbedder(VariationalEncoder(backbone(), latent_dim=6), frozen=False))
        assert text.startswith("LatentEmbedder(VariationalEncoder(") and "frozen=False, sample=False" in text

    def test_a_feature_embedder_names_its_width(self):
        assert repr(FeatureEmbedder(backbone(), out_dim=5)).endswith("out_dim=5)")
