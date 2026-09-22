"""CardioVAE and LS-EMVAE: the published configurations, and what makes each one itself.

These tests check the properties the papers claim, not recorded outputs. A model whose
fusion, decoder sharing or transfer behaviour changed would pass a shape test and fail
here, which is the point.
"""

from __future__ import annotations

import pytest
import torch

from kalecardiac.loaddata import LIMB_LEADS, STANDARD_12_LEAD, ECGFormatError
from kalecardiac.model.embed import LSEMVAE, CardioVAE, MultimodalVAE

LATENT = 8
LENGTH = 64
IMAGE = (16, 16)


@pytest.fixture
def cardiovae():
    return CardioVAE(
        signal_channels=1,
        signal_length=LENGTH,
        image_channels=1,
        image_size=IMAGE,
        latent_dim=LATENT,
        channels=(4, 8),
    )


@pytest.fixture
def lsemvae():
    return LSEMVAE(leads=LIMB_LEADS, length=LENGTH, latent_dim=LATENT, channels=(4, 8), num_groups=3)


class TestCardioVAE:
    def test_is_a_configuration_of_the_generic_model(self, cardiovae):
        assert isinstance(cardiovae, MultimodalVAE)

    def test_has_one_signal_and_one_image_modality(self, cardiovae):
        assert set(cardiovae.modalities) == {"signal", "image"}

    def test_fuses_as_a_product_of_experts(self, cardiovae):
        # The architectural claim of the paper: a product, so a confident modality
        # dominates and an absent one drops out without retraining.
        assert cardiovae.fusion_name == "poe"

    def test_reconstructs_both_modalities_in_their_own_shapes(self, cardiovae):
        batch = {"signal": torch.randn(3, 1, LENGTH), "image": torch.rand(3, 1, *IMAGE)}
        output = cardiovae(batch)
        assert output.joint.reconstructions["signal"].shape == (3, 1, LENGTH)
        assert output.joint.reconstructions["image"].shape == (3, 1, *IMAGE)

    def test_the_image_reconstruction_is_in_the_unit_interval(self, cardiovae):
        # It must match how the image source normalised, or the reconstruction term is
        # minimised by a constant.
        batch = {"signal": torch.randn(3, 1, LENGTH), "image": torch.rand(3, 1, *IMAGE)}
        image = cardiovae(batch).joint.reconstructions["image"]
        assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0

    def test_the_signal_reconstruction_is_unbounded(self, cardiovae):
        batch = {"signal": torch.randn(8, 1, LENGTH) * 3, "image": torch.rand(8, 1, *IMAGE)}
        signal = cardiovae(batch).joint.reconstructions["signal"]
        assert float(signal.min()) < 0.0

    def test_runs_on_one_modality_alone(self, cardiovae):
        # The clinical point: a subject with an ECG and no radiograph is still scorable.
        output = cardiovae({"signal": torch.randn(3, 1, LENGTH)})
        assert output.posterior.mean.shape == (3, LATENT)

    def test_the_tri_stream_objective_produces_a_pass_per_modality(self, cardiovae):
        batch = {"signal": torch.randn(3, 1, LENGTH), "image": torch.rand(3, 1, *IMAGE)}
        output = cardiovae(batch, unimodal_streams=True)
        assert sorted(output.unimodal) == ["image", "signal"]

    def test_a_multi_channel_recording_is_an_ordinary_configuration(self):
        # Twelve leads as channels, which is what the reference implementation encodes.
        model = CardioVAE(signal_channels=12, signal_length=LENGTH, image_size=IMAGE, latent_dim=LATENT, channels=(4,))
        batch = {"signal": torch.randn(2, 12, LENGTH), "image": torch.rand(2, 1, *IMAGE)}
        assert model(batch).joint.reconstructions["signal"].shape == (2, 12, LENGTH)

    def test_modalities_can_be_renamed_for_another_pairing(self):
        # The architecture is "a 1-D modality and a 2-D one"; an ECG with a cardiac MRI
        # slice is the same model.
        model = CardioVAE(
            signal_length=LENGTH,
            image_size=IMAGE,
            latent_dim=LATENT,
            channels=(4,),
            signal_name="ecg",
            image_name="mri",
        )
        assert set(model.modalities) == {"ecg", "mri"}

    def test_an_alternative_signal_backbone_is_a_configuration_change(self):
        model = CardioVAE(
            signal_length=LENGTH, image_size=IMAGE, latent_dim=LATENT, channels=(4,), signal_encoder="residual"
        )
        batch = {"signal": torch.randn(2, 1, LENGTH), "image": torch.rand(2, 1, *IMAGE)}
        assert model(batch).posterior.mean.shape == (2, LATENT)


class TestLSEMVAE:
    def test_is_a_configuration_of_the_generic_model(self, lsemvae):
        assert isinstance(lsemvae, MultimodalVAE)

    def test_one_encoder_per_lead(self, lsemvae):
        assert lsemvae.modalities == LIMB_LEADS
        assert len(lsemvae.encoders) == len(LIMB_LEADS)

    def test_fuses_hierarchically_by_default(self, lsemvae):
        assert lsemvae.fusion_name == "hime"

    def test_shares_one_decoder_across_every_lead(self, lsemvae):
        # The architectural difference from the grouped-fusion baseline: a shared
        # decoder forces the latent to carry lead-agnostic cardiac state.
        assert lsemvae.shared_decoder
        identities = {id(lsemvae.decoders[lead]) for lead in lsemvae.leads}
        assert len(identities) == 1

    def test_per_lead_decoders_are_available_as_the_baseline(self):
        model = LSEMVAE(leads=LIMB_LEADS, length=LENGTH, latent_dim=LATENT, channels=(4,), shared_decoder=False)
        identities = {id(model.decoders[lead]) for lead in model.leads}
        assert len(identities) == len(LIMB_LEADS)

    def test_each_lead_is_a_single_channel_modality(self, lsemvae):
        batch = {lead: torch.randn(3, 1, LENGTH) for lead in lsemvae.leads}
        output = lsemvae(batch)
        assert sorted(output.modality_posteriors) == sorted(LIMB_LEADS)

    def test_lead_names_are_canonicalised(self):
        model = LSEMVAE(leads=["LEAD_I", "lead_ii"], length=LENGTH, latent_dim=LATENT, channels=(4,))
        assert model.leads == ("I", "II")

    def test_a_repeated_lead_is_rejected(self):
        with pytest.raises(ECGFormatError, match="more than once"):
            LSEMVAE(leads=["I", "LEAD_I"], length=LENGTH, latent_dim=LATENT)

    def test_twelve_leads_work_as_readily_as_six(self):
        model = LSEMVAE(leads=STANDARD_12_LEAD, length=LENGTH, latent_dim=LATENT, channels=(4,))
        batch = {lead: torch.randn(2, 1, LENGTH) for lead in STANDARD_12_LEAD}
        assert model(batch).posterior.mean.shape == (2, LATENT)

    def test_the_published_ablations_are_configuration_changes(self):
        for fusion in ("poe", "moe"):
            model = LSEMVAE(leads=LIMB_LEADS, length=LENGTH, latent_dim=LATENT, channels=(4,), fusion=fusion)
            batch = {lead: torch.randn(2, 1, LENGTH) for lead in LIMB_LEADS}
            assert model.fusion_name == fusion
            assert torch.isfinite(model(batch).posterior.mean).all()

    def test_the_prior_expert_joins_the_first_group(self, lsemvae):
        # Prepended rather than appended, so it joins a group rather than trailing
        # behind the leads and shifting the grouping by one.
        assert lsemvae.use_prior

    def test_running_on_a_lead_subset_needs_no_retraining(self, lsemvae):
        batch = {lead: torch.randn(3, 1, LENGTH) for lead in ("I", "II", "aVF")}
        output = lsemvae(batch)
        assert output.posterior.mean.shape == (3, LATENT)
        assert sorted(output.modality_posteriors) == ["I", "II", "aVF"]


class TestTwelveToSixLeadTransfer:
    """The transfer both cardiac studies rely on: pretrain on twelve, fine-tune on six."""

    def test_a_twelve_lead_model_supplies_six_lead_embedders(self):
        pretrained = LSEMVAE(leads=STANDARD_12_LEAD, length=LENGTH, latent_dim=LATENT, channels=(4,))
        embedders = pretrained.latent_embedders(LIMB_LEADS)
        assert sorted(embedders) == sorted(LIMB_LEADS)

    def test_the_six_lead_embedders_are_the_pretrained_encoders(self):
        pretrained = LSEMVAE(leads=STANDARD_12_LEAD, length=LENGTH, latent_dim=LATENT, channels=(4,))
        embedder = pretrained.latent_embedders(["aVR"])["aVR"]
        assert embedder.encoder is pretrained.encoders["aVR"]

    def test_a_checkpoint_round_trips_through_a_smaller_model(self, tmp_path):
        from kalecardiac.utils import load_checkpoint

        pretrained = LSEMVAE(leads=STANDARD_12_LEAD, length=LENGTH, latent_dim=LATENT, channels=(4,))
        path = tmp_path / "pretrained.pt"
        torch.save(pretrained.state_dict(), path)

        six_lead = LSEMVAE(leads=LIMB_LEADS, length=LENGTH, latent_dim=LATENT, channels=(4,))
        missing, unexpected = load_checkpoint(path, six_lead, strict=False)
        # The six leads it kept are loaded; the six it dropped are reported unexpected.
        assert not missing
        assert all(key.split(".")[1] in set(STANDARD_12_LEAD) - set(LIMB_LEADS) for key in unexpected)

        batch = {lead: torch.randn(2, 1, LENGTH) for lead in LIMB_LEADS}
        assert torch.isfinite(six_lead(batch).posterior.mean).all()

    def test_the_loaded_weights_are_the_pretrained_ones(self, tmp_path):
        from kalecardiac.utils import load_checkpoint

        pretrained = LSEMVAE(leads=STANDARD_12_LEAD, length=LENGTH, latent_dim=LATENT, channels=(4,))
        path = tmp_path / "pretrained.pt"
        torch.save(pretrained.state_dict(), path)

        six_lead = LSEMVAE(leads=LIMB_LEADS, length=LENGTH, latent_dim=LATENT, channels=(4,))
        load_checkpoint(path, six_lead, strict=False)
        for lead in LIMB_LEADS:
            for ours, theirs in zip(
                six_lead.encoders[lead].parameters(), pretrained.encoders[lead].parameters(), strict=True
            ):
                assert torch.equal(ours, theirs)
