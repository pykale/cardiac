"""Collecting predictions: traceability to subjects, and what gets written out."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from kalecardiac.evaluate import SplitPredictions, binary_metrics, predict_split, save_predictions
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask

LATENT = 6


@pytest.fixture
def trained(cohort, lead_dataset, loader_factory):
    vae = LSEMVAE(
        leads=cohort.leads,
        length=cohort.ecg[cohort.subject_id[0]].shape[1],
        latent_dim=LATENT,
        channels=(4,),
    )
    task = ClassificationTask(hidden_dims=())
    predictor = MultimodalPredictor(vae.latent_embedders(), task.build_head)
    return CardiacTrainer(predictor, task), loader_factory(lead_dataset, batch_size=8)


class TestPredictSplit:
    def test_covers_every_subject_exactly_once(self, trained, cohort):
        model, loader = trained
        predictions = predict_split(model, loader, split="test")
        assert len(predictions) == len(cohort.subject_id)
        assert predictions.subject_id == cohort.subject_id

    def test_a_binary_score_is_one_column(self, trained):
        model, loader = trained
        assert predict_split(model, loader).scores.ndim == 1

    def test_the_target_comes_back_alongside(self, trained, cohort):
        model, loader = trained
        predictions = predict_split(model, loader)
        assert set(predictions.targets) == {"label"}
        expected = np.array([cohort.labels[name] for name in predictions.subject_id], dtype=float)
        assert np.array_equal(predictions.targets["label"], expected)

    def test_a_multiclass_score_keeps_its_columns(self, cohort, lead_dataset, loader_factory):
        vae = LSEMVAE(
            leads=cohort.leads, length=cohort.ecg[cohort.subject_id[0]].shape[1], latent_dim=LATENT, channels=(4,)
        )
        task = ClassificationTask(num_classes=3, hidden_dims=())
        model = CardiacTrainer(MultimodalPredictor(vae.latent_embedders(), task.build_head), task)
        predictions = predict_split(model, loader_factory(lead_dataset, batch_size=8))
        assert predictions.scores.shape == (len(cohort.subject_id), 3)

    def test_the_model_is_left_in_the_mode_it_was_found_in(self, trained):
        model, loader = trained
        model.train()
        predict_split(model, loader)
        assert model.training is True

    def test_the_fold_is_recorded(self, trained):
        model, loader = trained
        assert predict_split(model, loader, fold=3).fold == 3

    def test_an_empty_loader_is_rejected(self, trained):
        model, _ = trained
        with pytest.raises(ValueError, match="no batches"):
            predict_split(model, [], split="test")


class TestSplitPredictionRows:
    def test_one_row_per_subject_with_its_identifier(self):
        predictions = SplitPredictions(
            split="test",
            subject_id=["a", "b"],
            scores=np.array([0.2, 0.8]),
            targets={"label": np.array([0.0, 1.0])},
        )
        rows = predictions.rows()
        assert [row["subject_id"] for row in rows] == ["a", "b"]
        assert rows[1]["score"] == pytest.approx(0.8)
        assert rows[1]["label"] == pytest.approx(1.0)

    def test_a_fold_appears_only_when_there_is_one(self):
        without = SplitPredictions("test", ["a"], np.array([0.1])).rows()[0]
        with_fold = SplitPredictions("test", ["a"], np.array([0.1]), fold=2).rows()[0]
        assert "fold" not in without
        assert with_fold["fold"] == 2

    def test_multiclass_scores_become_one_column_each(self):
        rows = SplitPredictions("test", ["a"], np.array([[0.1, 0.7, 0.2]])).rows()
        assert {"score_0", "score_1", "score_2"} <= set(rows[0])


class TestSavePredictions:
    def test_writes_a_csv_and_a_metrics_file(self, trained, tmp_path):
        model, loader = trained
        predictions = predict_split(model, loader, split="test")
        metrics = {"test": binary_metrics(predictions.targets["label"], predictions.scores)}
        csv_path, json_path = save_predictions(tmp_path / "out", [predictions], metrics)

        assert csv_path.exists() and json_path.exists()
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(predictions)
        assert json.loads(json_path.read_text(encoding="utf-8"))["test"]["n"] == len(predictions)

    def test_metrics_are_optional(self, trained, tmp_path):
        model, loader = trained
        _, json_path = save_predictions(tmp_path / "out", [predict_split(model, loader)])
        assert json_path is None

    def test_several_folds_land_in_one_file(self, trained, tmp_path):
        model, loader = trained
        folds = [predict_split(model, loader, split="test", fold=index) for index in range(3)]
        csv_path, _ = save_predictions(tmp_path / "out", folds)
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3 * len(folds[0])
        assert {row["fold"] for row in rows} == {"0", "1", "2"}
