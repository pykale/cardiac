"""The trainer: one class, both halves of a cardiac workflow, tiny real runs."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

from kalecardiac.loaddata import MultimodalDataset, lead_sources
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask, RegressionTask

LATENT = 6


def lightning_trainer(max_epochs: int = 1) -> pl.Trainer:
    """A Lightning trainer with everything noisy or persistent turned off."""
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


@pytest.fixture
def vae(cohort):
    return LSEMVAE(
        leads=cohort.leads,
        length=cohort.ecg[cohort.subject_id[0]].shape[1],
        latent_dim=LATENT,
        channels=(4, 8),
        num_groups=3,
    )


@pytest.fixture
def lead_loader(lead_dataset, loader_factory):
    return loader_factory(lead_dataset, batch_size=8)


class TestConstruction:
    def test_takes_any_model_and_any_task(self, vae, lead_loader):
        assert isinstance(CardiacTrainer(vae, ReconstructionTask()), CardiacTrainer)

    def test_a_negative_gradient_clip_is_rejected(self, vae):
        with pytest.raises(ValueError, match="non-negative"):
            CardiacTrainer(vae, ReconstructionTask(), grad_clip=-1.0)

    def test_the_monitored_metric_follows_the_task(self, vae, cohort, label_target):
        generative = CardiacTrainer(vae, ReconstructionTask())
        assert (generative.monitor, generative.monitor_mode) == ("valid_loss", "min")

        task = ClassificationTask(hidden_dims=())
        predictor = MultimodalPredictor(vae.latent_embedders(), task.build_head)
        discriminative = CardiacTrainer(predictor, task)
        assert (discriminative.monitor, discriminative.monitor_mode) == ("valid_auc", "max")


class TestPretraining:
    def test_a_generative_run_lowers_its_objective(self, vae, lead_loader):
        # No annealing: with it, the objective at epoch 0 has no KL term and the one at
        # the end has a full one, so the two are not comparable quantities.
        task = ReconstructionTask(scale_factor=1e-2, annealing_epochs=0)
        model = CardiacTrainer(vae, task, init_lr=1e-3, max_epochs=4)
        batch = next(iter(lead_loader))
        before, _ = model.compute_loss(batch, "train")
        lightning_trainer(max_epochs=4).fit(model, lead_loader)
        after, _ = model.compute_loss(batch, "train")
        assert float(after) < float(before)

    def test_annealing_raises_the_objective_as_the_kl_weight_rises(self, vae, lead_loader):
        # The counterpart: under annealing the reported objective may rise while the
        # model improves, because the term being added was not there before.
        task = ReconstructionTask(scale_factor=1e-2, annealing_epochs=10)
        model = CardiacTrainer(vae, task)
        batch = next(iter(lead_loader))
        _, early = model.task.loss(model.forward(batch), batch, epoch=0)
        _, late = model.task.loss(model.forward(batch), batch, epoch=10)
        assert early["annealing"] == 0.0 and late["annealing"] == 1.0

    def test_it_needs_no_labels(self, cohort, ecg_source, loader_factory):
        # The whole point of pretraining: a cohort with no endpoint can still train.
        unlabelled = MultimodalDataset(cohort.subject_id, lead_sources(ecg_source))
        model = CardiacTrainer(
            LSEMVAE(leads=cohort.leads, length=ecg_source.length, latent_dim=LATENT, channels=(4,)),
            ReconstructionTask(scale_factor=1e-3),
        )
        lightning_trainer().fit(model, loader_factory(unlabelled, batch_size=8))

    def test_the_tri_stream_objective_runs(self, vae, lead_loader):
        model = CardiacTrainer(vae, ReconstructionTask(scale_factor=1e-3, unimodal_streams=True))
        lightning_trainer().fit(model, lead_loader)

    def test_the_alignment_term_runs(self, vae, lead_loader):
        model = CardiacTrainer(vae, ReconstructionTask(scale_factor=1e-3, alignment_weight=0.1))
        lightning_trainer().fit(model, lead_loader)

    def test_gradient_clipping_is_applied(self, vae, lead_loader):
        model = CardiacTrainer(vae, ReconstructionTask(scale_factor=1.0), grad_clip=0.5)
        lightning_trainer().fit(model, lead_loader)
        assert all(torch.isfinite(p).all() for p in vae.parameters())


class TestFineTuning:
    @pytest.fixture
    def predictor_trainer(self, vae, cohort):
        task = ClassificationTask(hidden_dims=(16,))
        predictor = MultimodalPredictor(vae.latent_embedders(["I", "II", "aVF"]), task.build_head)
        return CardiacTrainer(predictor, task, init_lr=1e-3, max_epochs=2)

    @pytest.fixture
    def subset_loader(self, cohort, ecg_source, label_target, loader_factory):
        sources = {lead: source for lead, source in lead_sources(ecg_source).items() if lead in ("I", "II", "aVF")}
        return loader_factory(MultimodalDataset(cohort.subject_id, sources, target=label_target), batch_size=8)

    def test_a_classification_run_completes(self, predictor_trainer, subset_loader):
        lightning_trainer(max_epochs=2).fit(predictor_trainer, subset_loader, subset_loader)

    def test_it_logs_the_task_metric(self, predictor_trainer, subset_loader):
        trainer = lightning_trainer(max_epochs=2)
        trainer.fit(predictor_trainer, subset_loader, subset_loader)
        assert "valid_auc" in trainer.callback_metrics

    def test_frozen_encoders_do_not_move(self, vae, cohort, subset_loader):
        task = ClassificationTask(hidden_dims=())
        embedders = vae.latent_embedders(["I", "II", "aVF"], frozen=True)
        before = {name: [p.clone() for p in e.encoder.parameters()] for name, e in embedders.items()}
        model = CardiacTrainer(MultimodalPredictor(embedders, task.build_head), task, init_lr=1e-2)
        lightning_trainer(max_epochs=2).fit(model, subset_loader)
        for name, embedder in embedders.items():
            for after, original in zip(embedder.encoder.parameters(), before[name], strict=True):
                assert torch.equal(after, original)

    def test_unfrozen_encoders_do_move(self, vae, cohort, subset_loader):
        task = ClassificationTask(hidden_dims=())
        embedders = vae.latent_embedders(["I"], frozen=False)
        before = [p.clone() for p in embedders["I"].encoder.parameters()]
        sources = {"I": None}
        model = CardiacTrainer(MultimodalPredictor(embedders, task.build_head), task, init_lr=1e-2)
        loader = subset_loader
        # Drop the other leads from the batch the predictor sees, which the model does
        # by naming only "I" among its modalities.
        lightning_trainer(max_epochs=2).fit(model, _only(loader, sources))
        assert any(
            not torch.equal(after, original)
            for after, original in zip(embedders["I"].encoder.parameters(), before, strict=True)
        )

    def test_a_regression_run_completes(self, vae, cohort, ecg_source, loader_factory):
        from kalecardiac.loaddata import ColumnTarget

        task = RegressionTask(hidden_dims=())
        predictor = MultimodalPredictor(vae.latent_embedders(["I", "II"]), task.build_head)
        sources = {lead: source for lead, source in lead_sources(ecg_source).items() if lead in ("I", "II")}
        dataset = MultimodalDataset(
            cohort.subject_id, sources, target=ColumnTarget.from_mapping({"value": cohort.values})
        )
        loader = loader_factory(dataset, batch_size=8)
        model = CardiacTrainer(predictor, task, init_lr=1e-3)
        trainer = lightning_trainer()
        trainer.fit(model, loader, loader)
        assert "valid_r2" in trainer.callback_metrics


class TestPrediction:
    def test_predict_tracks_no_gradients_and_restores_the_mode(self, vae, lead_loader):
        model = CardiacTrainer(vae, ReconstructionTask())
        model.train()
        batch = next(iter(lead_loader))
        output = model.predict(batch)
        assert not output.posterior.mean.requires_grad
        assert model.training is True

    def test_an_absent_modality_reaches_the_model(self, vae, lead_loader):
        task = ClassificationTask(hidden_dims=())
        predictor = MultimodalPredictor(vae.latent_embedders(), task.build_head)
        model = CardiacTrainer(predictor, task)
        model.eval()

        batch = next(iter(lead_loader))
        full = model.predict(batch).prediction
        batch.present["I"] = torch.zeros_like(batch.present["I"])
        partial = model.predict(batch).prediction
        assert not torch.allclose(full, partial)


def _only(loader, names):
    """Yield batches restricted to ``names``, so a one-modality model can consume them."""
    from kalecardiac.loaddata import SubjectBatch

    class Restricted:
        def __iter__(self):
            for batch in loader:
                yield SubjectBatch(
                    subject_id=batch.subject_id,
                    modalities={name: batch.modalities[name] for name in names},
                    present={name: batch.present[name] for name in names},
                    target=batch.target,
                    metadata=batch.metadata,
                )

        def __len__(self):
            return len(loader)

    return Restricted()
