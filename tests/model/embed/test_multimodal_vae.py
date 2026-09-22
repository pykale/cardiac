"""The generic multimodal VAE: encoding, fusion, streams, and missing modalities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kalecardiac.model.embed import (
    LatentEmbedder,
    MultimodalVAE,
    VariationalEncoder,
    build_signal_encoder,
)
from kalecardiac.model.layers import EXPERT_FUSIONS, Conv1dDecoder

LATENT = 8
LENGTH = 64


def signal_encoder(channels: int = 1) -> VariationalEncoder:
    backbone = build_signal_encoder("conv", in_channels=channels, length=LENGTH, channels=(4, 8))
    return VariationalEncoder(backbone, latent_dim=LATENT)


def signal_decoder(channels: int = 1) -> Conv1dDecoder:
    return Conv1dDecoder(latent_dim=LATENT, out_channels=channels, length=LENGTH, channels=(8, 4))


@pytest.fixture
def three_modality_vae():
    names = ("a", "b", "c")
    return MultimodalVAE(
        encoders={name: signal_encoder() for name in names},
        decoders={name: signal_decoder() for name in names},
        latent_dim=LATENT,
        fusion="poe",
    )


@pytest.fixture
def batch():
    return {name: torch.randn(4, 1, LENGTH) for name in ("a", "b", "c")}


class TestConstruction:
    def test_no_encoder_is_rejected(self):
        with pytest.raises(ValueError, match="at least one modality encoder"):
            MultimodalVAE(encoders={})

    def test_a_decoder_without_an_encoder_is_rejected(self):
        with pytest.raises(ValueError, match="no encoder"):
            MultimodalVAE(encoders={"a": signal_encoder()}, decoders={"b": signal_decoder()}, latent_dim=LATENT)

    def test_an_encoder_with_a_different_latent_width_is_rejected(self):
        other = VariationalEncoder(build_signal_encoder("conv", 1, LENGTH, channels=(4,)), latent_dim=LATENT + 1)
        with pytest.raises(ValueError, match="same latent space"):
            MultimodalVAE(encoders={"a": signal_encoder(), "b": other}, latent_dim=LATENT)

    def test_an_encoder_without_a_latent_dim_is_rejected(self):
        with pytest.raises(ValueError, match="VariationalEncoder"):
            MultimodalVAE(encoders={"a": nn.Linear(4, 4)}, latent_dim=LATENT)

    def test_an_unknown_fusion_is_rejected(self):
        with pytest.raises(KeyError, match="unknown latent fusion"):
            MultimodalVAE(encoders={"a": signal_encoder()}, latent_dim=LATENT, fusion="nonexistent")

    def test_a_modality_without_a_decoder_is_encoded_but_not_reconstructed(self):
        model = MultimodalVAE(
            encoders={"a": signal_encoder(), "b": signal_encoder()},
            decoders={"a": signal_decoder()},
            latent_dim=LATENT,
        )
        assert model.reconstructed == ("a",)


class TestEncoding:
    def test_produces_a_joint_and_a_per_modality_posterior(self, three_modality_vae, batch):
        joint, posteriors = three_modality_vae.encode(batch)
        assert joint.mean.shape == (4, LATENT)
        assert sorted(posteriors) == ["a", "b", "c"]
        assert all(p.mean.shape == (4, LATENT) for p in posteriors.values())

    def test_a_subset_of_modalities_is_enough(self, three_modality_vae, batch):
        joint, posteriors = three_modality_vae.encode({"a": batch["a"]})
        assert joint.mean.shape == (4, LATENT)
        assert list(posteriors) == ["a"]

    def test_no_modality_is_rejected(self, three_modality_vae):
        with pytest.raises(ValueError, match="no modality supplied"):
            three_modality_vae.encode({"unknown": torch.randn(4, 1, LENGTH)})

    def test_only_restricts_to_one_modality(self, three_modality_vae, batch):
        _, posteriors = three_modality_vae.encode(batch, only="b")
        assert list(posteriors) == ["b"]

    def test_only_must_name_a_supplied_modality(self, three_modality_vae, batch):
        with pytest.raises(ValueError, match="was not supplied"):
            three_modality_vae.encode({"a": batch["a"]}, only="b")

    def test_the_prior_expert_sharpens_a_single_modality(self, batch):
        with_prior = MultimodalVAE(encoders={"a": signal_encoder()}, latent_dim=LATENT, fusion="poe", use_prior=True)
        without = MultimodalVAE(encoders={"a": signal_encoder()}, latent_dim=LATENT, fusion="poe", use_prior=False)
        without.load_state_dict(with_prior.state_dict())
        joint_with, _ = with_prior.encode({"a": batch["a"]})
        joint_without, _ = without.encode({"a": batch["a"]})
        # Multiplying by N(0, 1) can only add precision.
        assert bool((joint_with.variance <= joint_without.variance + 1e-6).all())

    def test_a_mixture_ignores_the_prior_setting(self, batch):
        # A prior expert in a mixture only drags the joint towards zero.
        model = MultimodalVAE(encoders={"a": signal_encoder()}, latent_dim=LATENT, fusion="moe", use_prior=True)
        assert model._prior_applies is False


class TestForward:
    def test_reconstructs_every_decoded_modality(self, three_modality_vae, batch):
        output = three_modality_vae(batch)
        assert sorted(output.joint.reconstructions) == ["a", "b", "c"]
        for name, reconstruction in output.joint.reconstructions.items():
            assert reconstruction.shape == batch[name].shape

    def test_shorthand_properties_reach_the_joint_stream(self, three_modality_vae, batch):
        output = three_modality_vae(batch)
        assert output.posterior is output.joint.posterior
        assert output.latent is output.joint.latent

    def test_unimodal_streams_are_absent_unless_requested(self, three_modality_vae, batch):
        assert three_modality_vae(batch).unimodal == {}

    def test_unimodal_streams_give_one_pass_per_modality(self, three_modality_vae, batch):
        output = three_modality_vae(batch, unimodal_streams=True)
        assert sorted(output.unimodal) == ["a", "b", "c"]
        for name, stream in output.unimodal.items():
            assert stream.modalities == (name,)
            # Each stream still reconstructs every modality: that is what forces an
            # encoder to explain its siblings from its own evidence.
            assert sorted(stream.reconstructions) == ["a", "b", "c"]

    def test_the_latent_is_a_sample_not_the_mean(self, three_modality_vae, batch):
        output = three_modality_vae(batch)
        assert not torch.allclose(output.latent, output.posterior.mean)

    def test_present_is_carried_through_to_the_output(self, three_modality_vae, batch):
        present = {name: torch.ones(4, dtype=torch.bool) for name in batch}
        present["b"] = torch.zeros(4, dtype=torch.bool)
        output = three_modality_vae(batch, present)
        assert sorted(output.present) == ["a", "b", "c"]
        assert not bool(output.present["b"].any())

    def test_gradients_reach_every_encoder_and_decoder(self, three_modality_vae, batch):
        output = three_modality_vae(batch)
        loss = sum(value.sum() for value in output.joint.reconstructions.values())
        loss.backward()
        assert all(p.grad is not None for p in three_modality_vae.encoders.parameters())
        assert all(p.grad is not None for p in three_modality_vae.decoders.parameters())


class TestMissingModalities:
    def test_an_absent_modality_changes_the_joint(self, three_modality_vae, batch):
        present = {name: torch.ones(4, dtype=torch.bool) for name in batch}
        full, _ = three_modality_vae.encode(batch, present)
        present["b"] = torch.zeros(4, dtype=torch.bool)
        partial, _ = three_modality_vae.encode(batch, present)
        assert not torch.allclose(full.mean, partial.mean)

    def test_masking_a_modality_matches_withholding_it(self, three_modality_vae, batch):
        present = {
            "a": torch.ones(4, dtype=torch.bool),
            "b": torch.zeros(4, dtype=torch.bool),
            "c": torch.ones(4, dtype=torch.bool),
        }
        masked, _ = three_modality_vae.encode(batch, present)
        withheld, _ = three_modality_vae.encode({"a": batch["a"], "c": batch["c"]})
        assert torch.allclose(masked.mean, withheld.mean, atol=1e-3)

    def test_a_subject_with_nothing_present_still_yields_a_posterior(self, three_modality_vae, batch):
        present = {name: torch.zeros(4, dtype=torch.bool) for name in batch}
        joint, _ = three_modality_vae.encode(batch, present)
        assert torch.isfinite(joint.mean).all()


class TestSharedDecoder:
    def test_one_module_under_several_keys_shares_its_parameters(self):
        decoder = signal_decoder()
        shared = MultimodalVAE(
            encoders={name: signal_encoder() for name in "abc"},
            decoders=dict.fromkeys("abc", decoder),
            latent_dim=LATENT,
        )
        separate = MultimodalVAE(
            encoders={name: signal_encoder() for name in "abc"},
            decoders={name: signal_decoder() for name in "abc"},
            latent_dim=LATENT,
        )
        shared_count = sum(p.numel() for p in shared.decoders.parameters())
        separate_count = sum(p.numel() for p in separate.decoders.parameters())
        assert shared_count * 3 == separate_count

    def test_every_modality_reaches_the_shared_decoder(self):
        decoder = signal_decoder()
        model = MultimodalVAE(
            encoders={name: signal_encoder() for name in "abc"},
            decoders=dict.fromkeys("abc", decoder),
            latent_dim=LATENT,
        )
        batch = {name: torch.randn(2, 1, LENGTH) for name in "abc"}
        output = model(batch)
        # One decoder, three reconstructions -- each from the same latent, so they are
        # identical; the leads differ only through what they contributed to the latent.
        assert torch.allclose(output.joint.reconstructions["a"], output.joint.reconstructions["b"])

    def test_the_repr_says_the_decoder_is_shared(self):
        decoder = signal_decoder()
        model = MultimodalVAE(
            encoders={name: signal_encoder() for name in "abc"},
            decoders=dict.fromkeys("abc", decoder),
            latent_dim=LATENT,
        )
        assert "shared decoder" in repr(model)


class TestFusionIsConfiguration:
    @pytest.mark.parametrize("fusion", sorted(EXPERT_FUSIONS))
    def test_every_fusion_produces_a_working_model(self, fusion, batch):
        kwargs = {"num_groups": 2} if fusion == "hime" else {}
        model = MultimodalVAE(
            encoders={name: signal_encoder() for name in "abc"},
            decoders={name: signal_decoder() for name in "abc"},
            latent_dim=LATENT,
            fusion=fusion,
            **kwargs,
        )
        output = model(batch)
        assert output.posterior.mean.shape == (4, LATENT)
        assert torch.isfinite(output.posterior.mean).all()

    def test_the_fusion_is_the_only_difference_between_two_models(self, batch):
        # The claim the architecture rests on: an ablation is a configuration change.
        first = MultimodalVAE(
            encoders={name: signal_encoder() for name in "ab"},
            decoders={name: signal_decoder() for name in "ab"},
            latent_dim=LATENT,
            fusion="poe",
        )
        second = MultimodalVAE(
            encoders={name: signal_encoder() for name in "ab"},
            decoders={name: signal_decoder() for name in "ab"},
            latent_dim=LATENT,
            fusion="moe",
        )
        second.load_state_dict(first.state_dict())
        pair = {name: batch[name] for name in "ab"}
        assert not torch.allclose(first.encode(pair)[0].mean, second.encode(pair)[0].mean)


class TestLatentEmbedders:
    def test_wraps_every_encoder_by_default(self, three_modality_vae):
        embedders = three_modality_vae.latent_embedders()
        assert sorted(embedders) == ["a", "b", "c"]
        assert all(isinstance(embedder, LatentEmbedder) for embedder in embedders.values())
        assert all(embedder.out_dim == LATENT for embedder in embedders.values())

    def test_a_subset_is_how_transfer_to_fewer_modalities_works(self, three_modality_vae):
        embedders = three_modality_vae.latent_embedders(["a", "c"])
        assert sorted(embedders) == ["a", "c"]

    def test_an_unknown_modality_is_rejected(self, three_modality_vae):
        with pytest.raises(KeyError, match="no modality"):
            three_modality_vae.latent_embedders(["z"])

    def test_frozen_embedders_stop_gradients_reaching_the_encoder(self, three_modality_vae, batch):
        embedder = three_modality_vae.latent_embedders(["a"], frozen=True)["a"]
        assert not any(p.requires_grad for p in embedder.encoder.parameters())
        # Nothing on the path requires a gradient, so the output carries no graph at
        # all -- the strongest form of "this encoder will not move".
        assert not embedder(batch["a"]).requires_grad

    def test_a_frozen_embedder_stays_in_evaluation_mode(self, three_modality_vae):
        # A frozen module with batch normalisation would otherwise keep updating its
        # running statistics from whatever batches pass through it.
        embedder = three_modality_vae.latent_embedders(["a"], frozen=True)["a"]
        embedder.train(True)
        assert embedder.training is True
        assert embedder.encoder.training is False

    def test_unfrozen_embedders_pass_gradients_through(self, three_modality_vae, batch):
        embedder = three_modality_vae.latent_embedders(["a"], frozen=False)["a"]
        embedder(batch["a"]).sum().backward()
        assert any(p.grad is not None for p in embedder.encoder.parameters())
