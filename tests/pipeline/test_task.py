"""Tasks: what each decides, and how a generative objective differs from a predictive one."""

from __future__ import annotations

import pytest
import torch

from kalecardiac.evaluate import MetricError
from kalecardiac.loaddata import LABEL_KEY, VALUE_KEY, SubjectBatch
from kalecardiac.model.embed import PredictionOutput
from kalecardiac.model.embed.multimodal_vae import VAEOutput, VAEStream
from kalecardiac.model.layers import GaussianPosterior
from kalecardiac.model.predict import LinearHead, MLPHead
from kalecardiac.pipeline import ClassificationTask, ReconstructionTask, RegressionTask


def batch_with(target: dict, modalities: dict | None = None) -> SubjectBatch:
    modalities = modalities or {"a": torch.randn(4, 1, 8)}
    return SubjectBatch(
        subject_id=[f"s{index}" for index in range(4)],
        modalities=modalities,
        present={name: torch.ones(4, dtype=torch.bool) for name in modalities},
        target=target,
    )


class TestClassificationTask:
    def test_binary_endpoints_get_one_logit(self):
        assert ClassificationTask().out_features == 1

    def test_multiclass_endpoints_get_one_logit_per_class(self):
        assert ClassificationTask(num_classes=4).out_features == 4

    def test_the_head_is_linear_without_hidden_widths(self):
        assert isinstance(ClassificationTask(hidden_dims=()).build_head(8), LinearHead)

    def test_the_head_has_a_hidden_layer_by_default(self):
        assert isinstance(ClassificationTask().build_head(8), MLPHead)

    def test_the_loss_is_finite_and_reports_itself(self):
        task = ClassificationTask()
        output = PredictionOutput(prediction=torch.randn(4, 1), representation=torch.randn(4, 8))
        loss, parts = task.loss(output, batch_with({LABEL_KEY: torch.tensor([0.0, 1.0, 0.0, 1.0])}))
        assert torch.isfinite(loss)
        assert parts["loss"] == pytest.approx(float(loss))

    def test_the_positive_weight_follows_the_task_to_its_device(self):
        task = ClassificationTask(pos_weight=3.0)
        assert task.pos_weight is not None
        assert "pos_weight" in dict(task.named_buffers())

    def test_the_positive_weight_stays_out_of_the_checkpoint(self):
        # It belongs to one fold's class balance, not to the model.
        assert "pos_weight" not in ClassificationTask(pos_weight=3.0).state_dict()

    def test_scores_are_one_number_per_subject(self):
        task = ClassificationTask()
        output = PredictionOutput(prediction=torch.randn(4, 1), representation=torch.randn(4, 8))
        assert task.scores(output).shape == (4,)

    def test_multiclass_scores_are_the_positive_class_probability(self):
        task = ClassificationTask(num_classes=3)
        output = PredictionOutput(prediction=torch.randn(4, 3), representation=torch.randn(4, 8))
        scores = task.scores(output)
        assert scores.shape == (4,)
        assert bool(((scores >= 0) & (scores <= 1)).all())

    def test_the_metric_is_an_auc(self):
        task = ClassificationTask()
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
        assert task.metric(scores, {LABEL_KEY: torch.tensor([0.0, 1.0, 0.0, 1.0])}) == pytest.approx(1.0)
        assert task.metric_name == "auc" and task.metric_mode == "max"

    def test_a_one_class_split_raises_a_metric_error_not_a_crash(self):
        task = ClassificationTask()
        with pytest.raises(MetricError):
            task.metric(torch.randn(4), {LABEL_KEY: torch.zeros(4)})

    def test_fewer_than_two_classes_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            ClassificationTask(num_classes=1)

    def test_a_class_weight_of_the_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="must have 3 entries"):
            ClassificationTask(num_classes=3, class_weight=[1.0, 1.0])


class TestRegressionTask:
    def test_the_loss_is_a_mean_squared_error(self):
        task = RegressionTask()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        output = PredictionOutput(prediction=targets.unsqueeze(1), representation=torch.randn(4, 8))
        loss, _ = task.loss(output, batch_with({VALUE_KEY: targets}))
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_the_metric_is_r_squared_and_runs_upwards(self):
        task = RegressionTask()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert task.metric(targets, {VALUE_KEY: targets}) == pytest.approx(1.0)
        assert task.metric_name == "r2" and task.metric_mode == "max"

    def test_it_supervises_a_continuous_key(self):
        assert RegressionTask().target_keys == (VALUE_KEY,)


class TestReconstructionTask:
    @pytest.fixture
    def output(self):
        modalities = {"a": torch.randn(4, 1, 8), "b": torch.randn(4, 1, 8)}
        posterior = GaussianPosterior(torch.randn(4, 6) * 0.1, torch.zeros(4, 6))
        stream = VAEStream(
            posterior=posterior,
            latent=posterior.mean,
            reconstructions={name: torch.zeros_like(value) for name, value in modalities.items()},
            modalities=("a", "b"),
        )
        return modalities, VAEOutput(
            joint=stream,
            modality_posteriors={name: posterior for name in modalities},
            present={name: torch.ones(4, dtype=torch.bool) for name in modalities},
        )

    def test_it_takes_no_target(self):
        assert ReconstructionTask().target_keys == ()

    def test_it_has_no_prediction_head(self):
        with pytest.raises(NotImplementedError, match="no prediction head"):
            ReconstructionTask().build_head(8)

    def test_it_accumulates_no_per_subject_score(self, output):
        # Its quality is the objective itself; there is nothing per-subject to rank.
        assert ReconstructionTask().scores(output[1]) is None

    def test_the_batch_modalities_are_the_supervision(self, output):
        modalities, vae_output = output
        loss, parts = ReconstructionTask().loss(vae_output, batch_with({}, modalities))
        assert torch.isfinite(loss)
        assert "recon_a" in parts and "kl" in parts

    def test_annealing_rises_linearly_then_holds(self):
        task = ReconstructionTask(annealing_epochs=10)
        assert task.annealing_factor(0) == 0.0
        assert task.annealing_factor(5) == pytest.approx(0.5)
        assert task.annealing_factor(20) == 1.0

    def test_no_annealing_means_full_weight_immediately(self):
        assert ReconstructionTask(annealing_epochs=0).annealing_factor(0) == 1.0

    def test_annealing_changes_the_loss_across_epochs(self, output):
        modalities, vae_output = output
        task = ReconstructionTask(annealing_epochs=10)
        batch = batch_with({}, modalities)
        early, _ = task.loss(vae_output, batch, epoch=0)
        late, _ = task.loss(vae_output, batch, epoch=10)
        assert float(late) > float(early)

    def test_unimodal_streams_are_requested_through_forward_kwargs(self):
        assert ReconstructionTask(unimodal_streams=True).forward_kwargs == {"unimodal_streams": True}
        assert ReconstructionTask().forward_kwargs == {"unimodal_streams": False}

    def test_a_unimodal_stream_adds_to_the_objective(self, output):
        modalities, vae_output = output
        joint_only = ReconstructionTask()
        tri_stream = ReconstructionTask()
        batch = batch_with({}, modalities)
        base, _ = joint_only.loss(vae_output, batch)

        # Attach a stream and check the objective grows by that stream's contribution.
        vae_output.unimodal = {"a": vae_output.joint}
        with_stream, parts = tri_stream.loss(vae_output, batch)
        assert float(with_stream) > float(base)
        assert "elbo_a_only" in parts

    def test_the_alignment_term_is_reported_when_enabled(self, output):
        modalities, vae_output = output
        _, parts = ReconstructionTask(alignment_weight=0.5).loss(vae_output, batch_with({}, modalities))
        assert "alignment" in parts

    def test_the_alignment_term_is_absent_when_disabled(self, output):
        modalities, vae_output = output
        _, parts = ReconstructionTask(alignment_weight=0.0).loss(vae_output, batch_with({}, modalities))
        assert "alignment" not in parts

    def test_negative_schedule_settings_are_rejected(self):
        with pytest.raises(ValueError, match="annealing_epochs"):
            ReconstructionTask(annealing_epochs=-1)
        with pytest.raises(ValueError, match="alignment_weight"):
            ReconstructionTask(alignment_weight=-1.0)

    def test_the_monitored_metric_is_the_loss(self):
        assert ReconstructionTask().metric_name == "loss"
        assert ReconstructionTask().metric_mode == "min"

    def test_asking_for_an_epoch_metric_raises_a_metric_error(self, output):
        with pytest.raises(MetricError, match="no epoch metric"):
            ReconstructionTask().metric(torch.zeros(4), {})
