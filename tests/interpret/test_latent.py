"""Latent-space inspection: collecting representations and projecting them."""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from kalecardiac.interpret import collect_latents, latent_embedding
from kalecardiac.model.embed import LSEMVAE

LATENT = 6


@pytest.fixture
def vae(cohort):
    return LSEMVAE(
        leads=cohort.leads,
        length=cohort.ecg[cohort.subject_id[0]].shape[1],
        latent_dim=LATENT,
        channels=(4,),
    )


class TestCollectLatents:
    def test_one_row_per_subject_in_loader_order(self, vae, lead_dataset, loader_factory, cohort):
        identifiers, joint, per_modality = collect_latents(vae, loader_factory(lead_dataset, batch_size=8))
        assert identifiers == cohort.subject_id
        assert joint.shape == (len(cohort.subject_id), LATENT)
        assert per_modality == {}

    def test_per_modality_means_are_returned_when_asked_for(self, vae, lead_dataset, loader_factory, cohort):
        _, _, per_modality = collect_latents(vae, loader_factory(lead_dataset, batch_size=8), per_modality=True)
        assert sorted(per_modality) == sorted(cohort.leads)
        for values in per_modality.values():
            assert values.shape == (len(cohort.subject_id), LATENT)

    def test_it_returns_the_mean_not_a_sample(self, vae, lead_dataset, loader_factory):
        # A description of a cohort must not change between two passes over it.
        loader = loader_factory(lead_dataset, batch_size=8)
        first = collect_latents(vae, loader)[1]
        second = collect_latents(vae, loader)[1]
        assert np.allclose(first, second)

    def test_the_model_is_left_in_the_mode_it_was_found_in(self, vae, lead_dataset, loader_factory):
        vae.train()
        collect_latents(vae, loader_factory(lead_dataset, batch_size=8))
        assert vae.training is True

    def test_a_model_without_encode_is_rejected_with_advice(self, lead_dataset, loader_factory):
        with pytest.raises(AttributeError, match="MultimodalVAE"):
            collect_latents(nn.Linear(4, 4), loader_factory(lead_dataset, batch_size=8))

    def test_an_empty_loader_is_rejected(self, vae):
        with pytest.raises(ValueError, match="no batches"):
            collect_latents(vae, [])


class TestLatentEmbedding:
    @pytest.fixture
    def features(self):
        return np.random.default_rng(0).normal(size=(40, LATENT))

    def test_pca_projects_to_the_requested_dimensions(self, features):
        assert latent_embedding(features, method="pca").shape == (40, 2)
        assert latent_embedding(features, method="pca", n_components=3).shape == (40, 3)

    def test_pca_is_deterministic(self, features):
        first = latent_embedding(features, method="pca")
        second = latent_embedding(features, method="pca")
        assert np.allclose(first, second)

    def test_fitting_on_a_subset_still_transforms_everything(self, features):
        coordinates = latent_embedding(features, method="pca", fit_on=features[:20])
        assert coordinates.shape == (40, 2)

    def test_umap_is_available_when_installed(self, features):
        pytest.importorskip("umap")
        assert latent_embedding(features, method="umap").shape == (40, 2)

    def test_umap_bounds_its_neighbourhood_by_the_cohort(self, features):
        # UMAP's default of fifteen neighbours fails outright on a split smaller than
        # that, which a cardiac fold routinely is.
        pytest.importorskip("umap")
        assert latent_embedding(features[:8], method="umap").shape == (8, 2)

    def test_a_one_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match=r"\(N, D\)"):
            latent_embedding(np.zeros(10))

    def test_an_unknown_method_is_rejected(self, features):
        with pytest.raises(ValueError, match="unknown method"):
            latent_embedding(features, method="tsne")


def _fit(model, loader, task, epochs: int = 4, lr: float = 1e-3) -> None:
    """Run a short CPU training loop, with everything noisy or persistent turned off."""
    import pytorch_lightning as pl

    from kalecardiac.pipeline import CardiacTrainer

    pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(CardiacTrainer(model, task, init_lr=lr), loader)


def test_pretraining_reshapes_the_latent_space(cohort, lead_dataset, loader_factory):
    """Training moves the representation, rather than leaving it at its initialisation."""
    from kalecardiac.pipeline import ReconstructionTask

    length = cohort.ecg[cohort.subject_id[0]].shape[1]
    vae = LSEMVAE(leads=cohort.leads, length=length, latent_dim=LATENT, channels=(4, 8))
    loader = loader_factory(lead_dataset, batch_size=8)

    before = collect_latents(vae, loader)[1]
    _fit(vae, loader, ReconstructionTask(scale_factor=1e-2))
    after = collect_latents(vae, loader)[1]

    assert np.isfinite(after).all()
    assert not np.allclose(before, after)


def test_the_alignment_term_pulls_lead_latents_towards_the_joint(cohort, lead_dataset, loader_factory):
    """What LS-EMVAE's regulariser is for, measured directly.

    Nothing in an ELBO requires the per-lead posteriors to agree with the joint one, so
    a lead's encoder can settle in its own corner of the latent space and be useless on
    its own. This checks that adding the term does what it was added for -- and that
    training without it does not do so by accident.
    """
    from kalecardiac.pipeline import ReconstructionTask

    length = cohort.ecg[cohort.subject_id[0]].shape[1]
    loader = loader_factory(lead_dataset, batch_size=8)

    def disagreement(model) -> float:
        _, joint, per_modality = collect_latents(model, loader, per_modality=True)
        return float(np.mean([np.mean((values - joint) ** 2) for values in per_modality.values()]))

    aligned = LSEMVAE(leads=cohort.leads, length=length, latent_dim=LATENT, channels=(4, 8))
    unaligned = LSEMVAE(leads=cohort.leads, length=length, latent_dim=LATENT, channels=(4, 8))
    unaligned.load_state_dict(aligned.state_dict())
    start = disagreement(aligned)

    _fit(aligned, loader, ReconstructionTask(scale_factor=1e-2, alignment_weight=10.0), epochs=6, lr=1e-3)
    _fit(unaligned, loader, ReconstructionTask(scale_factor=1e-2, alignment_weight=0.0), epochs=6, lr=1e-3)

    assert disagreement(aligned) < start
    assert disagreement(aligned) < disagreement(unaligned)
