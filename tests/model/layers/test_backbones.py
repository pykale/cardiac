"""Backbone blocks: shapes, the embedder contract, and parity with PyKale.

Every backbone must satisfy one contract -- declare ``out_dim`` and map
``(batch, channels, ...)`` to ``(batch, out_dim)`` -- because that is what lets an
experiment change ``ECG.ENCODER`` without touching the model that wraps it.
"""

from __future__ import annotations

import pytest
import torch

from kalecardiac.model.layers import (
    MLP,
    Conv1dDecoder,
    Conv1dEncoder,
    Conv2dDecoder,
    Conv2dEncoder,
    ResidualBlock1d,
    ResidualConv1dEncoder,
    SinusoidalPositionalEncoding,
    TransformerSignalEncoder,
    conv_output_length,
    conv_output_size,
)

SIGNAL_BACKBONES = [
    ("conv", lambda channels: Conv1dEncoder(in_channels=channels, length=256, channels=(4, 8))),
    ("residual", lambda channels: ResidualConv1dEncoder(in_channels=channels, channels=(8, 16))),
    (
        "transformer",
        lambda channels: TransformerSignalEncoder(
            in_channels=channels, patch_size=32, d_model=16, nhead=2, num_layers=1, dim_feedforward=32
        ),
    ),
]


class TestSignalBackboneContract:
    @pytest.mark.parametrize("name,build", SIGNAL_BACKBONES, ids=[name for name, _ in SIGNAL_BACKBONES])
    def test_declares_out_dim_and_produces_it(self, name, build):
        backbone = build(1)
        features = backbone(torch.randn(3, 1, 256))
        assert features.shape == (3, backbone.out_dim)

    @pytest.mark.parametrize("name,build", SIGNAL_BACKBONES, ids=[name for name, _ in SIGNAL_BACKBONES])
    def test_handles_any_number_of_channels(self, name, build):
        for channels in (1, 6, 12):
            backbone = build(channels)
            assert backbone(torch.randn(2, channels, 256)).shape == (2, backbone.out_dim)

    @pytest.mark.parametrize("name,build", SIGNAL_BACKBONES, ids=[name for name, _ in SIGNAL_BACKBONES])
    def test_handles_any_batch_size(self, name, build):
        backbone = build(1)
        for batch in (1, 2, 7):
            assert backbone(torch.randn(batch, 1, 256)).shape[0] == batch

    @pytest.mark.parametrize("name,build", SIGNAL_BACKBONES, ids=[name for name, _ in SIGNAL_BACKBONES])
    def test_rejects_the_wrong_channel_count(self, name, build):
        with pytest.raises(ValueError, match="expected"):
            build(1)(torch.randn(2, 3, 256))

    @pytest.mark.parametrize("name,build", SIGNAL_BACKBONES, ids=[name for name, _ in SIGNAL_BACKBONES])
    def test_passes_gradients_to_its_input(self, name, build):
        signal = torch.randn(2, 1, 256, requires_grad=True)
        build(1)(signal).sum().backward()
        assert signal.grad is not None and torch.isfinite(signal.grad).all()


class TestConv1d:
    def test_output_length_matches_the_formula(self):
        assert conv_output_length(256, (4, 8, 16), kernel_size=3, stride=2, padding=1) == 32

    def test_an_over_deep_stack_is_rejected_with_advice(self):
        with pytest.raises(ValueError, match="fewer layers"):
            conv_output_length(4, (1, 1), kernel_size=5, stride=2, padding=0)

    def test_out_dim_is_channels_times_length(self):
        encoder = Conv1dEncoder(in_channels=1, length=256, channels=(4, 8, 16))
        assert encoder.out_dim == 16 * 32

    def test_rejects_the_wrong_length(self):
        with pytest.raises(ValueError, match="256"):
            Conv1dEncoder(in_channels=1, length=256, channels=(4,))(torch.randn(2, 1, 128))

    def test_batch_norm_is_optional_and_changes_the_parameters(self):
        plain = Conv1dEncoder(in_channels=1, length=64, channels=(4,), batch_norm=False)
        normed = Conv1dEncoder(in_channels=1, length=64, channels=(4,), batch_norm=True)
        assert sum(p.numel() for p in normed.parameters()) > sum(p.numel() for p in plain.parameters())
        assert normed(torch.randn(3, 1, 64)).shape == (3, normed.out_dim)

    @pytest.mark.parametrize("length", [64, 256, 1000, 1234, 5000])
    def test_the_decoder_reaches_any_length(self, length):
        decoder = Conv1dDecoder(latent_dim=8, out_channels=2, length=length, channels=(16, 8, 4))
        assert decoder(torch.randn(3, 8)).shape == (3, 2, length)

    def test_the_decoder_output_is_unbounded_by_default(self):
        # A standardised recording is centred on zero and takes both signs.
        decoder = Conv1dDecoder(latent_dim=8, out_channels=1, length=64)
        output = decoder(torch.randn(64, 8) * 5)
        assert float(output.min()) < 0.0

    def test_a_sigmoid_output_is_bounded(self):
        decoder = Conv1dDecoder(latent_dim=8, out_channels=1, length=64, output_activation="sigmoid")
        output = decoder(torch.randn(8, 8) * 5)
        assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0

    def test_an_unknown_output_activation_is_rejected(self):
        with pytest.raises(ValueError, match="unknown output_activation"):
            Conv1dDecoder(latent_dim=8, out_channels=1, length=64, output_activation="softmax")

    def test_encoder_and_decoder_round_trip_shapes(self):
        encoder = Conv1dEncoder(in_channels=6, length=256, channels=(4, 8, 16))
        decoder = Conv1dDecoder(latent_dim=encoder.out_dim, out_channels=6, length=256, channels=(16, 8, 4))
        signal = torch.randn(2, 6, 256)
        assert decoder(encoder(signal)).shape == signal.shape


class TestConv2d:
    def test_output_size_matches_the_formula(self):
        assert conv_output_size((224, 224), 3, kernel_size=3, stride=2, padding=1) == (28, 28)

    def test_non_square_images_work(self):
        encoder = Conv2dEncoder(in_channels=1, size=(40, 64), channels=(4, 8))
        assert encoder(torch.randn(2, 1, 40, 64)).shape == (2, encoder.out_dim)

    def test_rejects_the_wrong_size(self):
        with pytest.raises(ValueError, match="expected"):
            Conv2dEncoder(in_channels=1, size=(32, 32), channels=(4,))(torch.randn(2, 1, 16, 16))

    @pytest.mark.parametrize("size", [(32, 32), (40, 50), (224, 224), (50, 37)])
    def test_the_decoder_reaches_any_size(self, size):
        decoder = Conv2dDecoder(latent_dim=8, out_channels=1, size=size, channels=(16, 8, 4))
        assert decoder(torch.randn(2, 8)).shape == (2, 1, *size)

    def test_the_default_image_output_is_in_the_unit_interval(self):
        decoder = Conv2dDecoder(latent_dim=8, out_channels=1, size=(16, 16))
        output = decoder(torch.randn(8, 8) * 5)
        assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


class TestPyKaleParity:
    """The fixed-shape PyKale encoders are the special case of these.

    KaleCardiac reimplements them because PyKale's flattened widths are literals that
    are wrong for any other input shape. Where the shapes agree, the architectures must.
    """

    def test_signal_encoder_matches_kale_signal_vae_encoder(self):
        from kale.embed.signal_cnn import SignalVAEEncoder

        ours = Conv1dEncoder(in_channels=1, length=1000, channels=(16, 32, 64), kernel_size=3, stride=2, padding=1)
        theirs = SignalVAEEncoder(input_dim=1000, latent_dim=8)
        assert ours.out_dim == theirs.output_size()

    def test_image_encoder_matches_kale_image_vae_encoder(self):
        from kale.embed.image_cnn import ImageVAEEncoder

        ours = Conv2dEncoder(in_channels=1, size=(224, 224), channels=(16, 32, 64))
        theirs = ImageVAEEncoder(num_channels=1, latent_dim=8)
        assert ours.out_dim == theirs.fc_mu.in_features


class TestResidual:
    def test_a_block_preserves_length_at_stride_one(self):
        block = ResidualBlock1d(4, 8, kernel_size=7, stride=1)
        assert block(torch.randn(2, 4, 64)).shape == (2, 8, 64)

    def test_a_block_halves_length_at_stride_two(self):
        block = ResidualBlock1d(4, 8, kernel_size=7, stride=2)
        assert block(torch.randn(2, 4, 64)).shape == (2, 8, 32)

    def test_an_even_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="must be odd"):
            ResidualBlock1d(4, 8, kernel_size=8)

    def test_out_dim_is_independent_of_input_length(self):
        # Global pooling, so the same encoder takes a five- and a ten-second recording.
        encoder = ResidualConv1dEncoder(in_channels=1, channels=(8, 16))
        assert encoder(torch.randn(2, 1, 500)).shape == encoder(torch.randn(2, 1, 1000)).shape


class TestTransformer:
    def test_positional_encoding_adds_position_not_content(self):
        encoding = SinusoidalPositionalEncoding(8, max_len=16)
        tokens = torch.zeros(2, 5, 8)
        encoded = encoding(tokens)
        # Every batch row gets the same positions, and positions differ from each other.
        assert torch.equal(encoded[0], encoded[1])
        assert not torch.allclose(encoded[0, 0], encoded[0, 1])

    def test_an_odd_model_width_is_rejected(self):
        with pytest.raises(ValueError, match="positive even"):
            SinusoidalPositionalEncoding(7)

    def test_a_sequence_longer_than_the_table_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds max_len"):
            SinusoidalPositionalEncoding(8, max_len=4)(torch.zeros(1, 5, 8))

    def test_a_trailing_partial_patch_is_dropped(self):
        encoder = TransformerSignalEncoder(in_channels=1, patch_size=32, d_model=16, nhead=2, num_layers=1)
        # Evaluation mode: dropout would otherwise make two passes differ for reasons
        # that have nothing to do with the patching being tested.
        encoder.eval()
        exact = encoder(torch.zeros(1, 1, 256))
        padded = encoder(torch.zeros(1, 1, 256 + 5))
        assert torch.allclose(exact, padded, atol=1e-6)

    def test_a_signal_shorter_than_one_patch_is_rejected(self):
        encoder = TransformerSignalEncoder(in_channels=1, patch_size=64, d_model=16, nhead=2, num_layers=1)
        with pytest.raises(ValueError, match="shorter than one patch"):
            encoder(torch.randn(1, 1, 32))

    def test_last_token_pooling_is_available(self):
        encoder = TransformerSignalEncoder(
            in_channels=1, patch_size=32, d_model=16, nhead=2, num_layers=1, pooling="last"
        )
        assert encoder(torch.randn(2, 1, 256)).shape == (2, 16)

    def test_an_unknown_pooling_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pooling"):
            TransformerSignalEncoder(pooling="max")


class TestMLP:
    def test_no_hidden_layers_gives_a_single_linear_map(self):
        mlp = MLP(4, 2)
        assert mlp(torch.randn(3, 4)).shape == (3, 2)
        assert len(list(mlp.net)) == 1

    def test_hidden_layers_are_inserted_with_dropout(self):
        mlp = MLP(4, 2, hidden_dims=(8, 8), dropout=0.5)
        assert mlp(torch.randn(3, 4)).shape == (3, 2)
        assert len(list(mlp.net)) == 7

    def test_a_non_positive_width_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            MLP(4, 0)
