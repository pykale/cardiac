"""Running a model over a split, and keeping what it predicted.

Metrics summarise; predictions are the evidence. Keeping them per subject is what lets
a result be re-scored at another threshold, pooled across folds, or handed to a
clinician who wants to see which subjects the model got wrong -- none of which a saved
AUC supports.

Predictions are keyed by subject identifier throughout, so a row can be traced back to
the recording it came from without depending on loader order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kalecardiac.loaddata.multimodal_access import SubjectBatch
from kalecardiac.utils.io import write_csv, write_json


@dataclass
class SplitPredictions:
    """What a model predicted for one split.

    Attributes:
        split (str): Which split, e.g. ``"test"``.
        subject_id (list[str]): Identifiers, in prediction order.
        scores (np.ndarray): ``(N,)`` or ``(N, C)`` predictions, as the task emits
            them.
        targets (dict[str, np.ndarray]): Supervision for the same subjects, keyed as
            the batch carried it. Empty for an unlabelled split.
        fold (int | None): Which cross-validation fold produced this, if any.
    """

    split: str
    subject_id: list[str]
    scores: np.ndarray
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    fold: int | None = None

    def __len__(self) -> int:
        return len(self.subject_id)

    def rows(self) -> list[dict]:
        """One record per subject, for a CSV or a table."""
        records = []
        for index, identifier in enumerate(self.subject_id):
            record: dict = {"subject_id": identifier, "split": self.split}
            if self.fold is not None:
                record["fold"] = self.fold
            score = self.scores[index]
            if np.ndim(score) == 0:
                record["score"] = float(score)
            else:
                record.update({f"score_{column}": float(value) for column, value in enumerate(np.ravel(score))})
            record.update({key: float(values[index]) for key, values in self.targets.items()})
            records.append(record)
        return records


def predict_split(
    model: nn.Module,
    loader: Iterable[SubjectBatch],
    split: str = "test",
    fold: int | None = None,
    device: torch.device | str | None = None,
) -> SplitPredictions:
    """Run a model over a loader and collect what it predicted.

    Works with a :class:`~kalecardiac.pipeline.CardiacTrainer`, which exposes
    ``predict``, or with any module returning an output that carries ``prediction``.
    The model is left in the mode it was found in.

    Args:
        model: The trained model.
        loader: Yields :class:`~kalecardiac.loaddata.SubjectBatch` batches.
        split: Name recorded on the result.
        fold: Cross-validation fold, if any.
        device: Where to run. ``None`` uses whatever the model is already on.

    Returns:
        The predictions, in loader order.

    Raises:
        ValueError: If the loader yields nothing.
    """
    was_training = model.training
    model.eval()
    if device is not None:
        model.to(device)

    identifiers: list[str] = []
    scores: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {}

    try:
        with torch.no_grad():
            for batch in loader:
                if device is not None:
                    batch = batch.to(device)
                output = model.predict(batch) if hasattr(model, "predict") else model(batch.modalities, batch.present)
                prediction = output.prediction.detach().cpu().float()
                scores.append(prediction.reshape(len(batch), -1))
                identifiers.extend(batch.subject_id)
                for key, value in batch.target.items():
                    targets.setdefault(key, []).append(value.detach().cpu().float().reshape(-1))
    finally:
        model.train(was_training)

    if not identifiers:
        raise ValueError(f"the {split!r} loader yielded no batches; a split with no subjects cannot be scored")

    stacked = torch.cat(scores).numpy()
    return SplitPredictions(
        split=split,
        subject_id=identifiers,
        # Squeezed only when there is one column, so a binary score is (N,) and a
        # multiclass one stays (N, C).
        scores=stacked[:, 0] if stacked.shape[1] == 1 else stacked,
        targets={key: torch.cat(values).numpy() for key, values in targets.items()},
        fold=fold,
    )


def save_predictions(
    out_dir: str | Path,
    predictions: Iterable[SplitPredictions],
    metrics: Mapping[str, Mapping] | None = None,
) -> tuple[Path, Path | None]:
    """Write predictions, and any metrics, to a directory.

    Args:
        out_dir: Where to write.
        predictions: One result per split, or per fold and split.
        metrics: Metrics keyed however the caller reports them.

    Returns:
        ``(predictions_path, metrics_path)``, the second ``None`` if no metrics were
        given.
    """
    out_dir = Path(out_dir)
    rows = [row for prediction in predictions for row in prediction.rows()]
    predictions_path = write_csv(out_dir / "predictions.csv", rows)
    metrics_path = write_json(out_dir / "metrics.json", dict(metrics)) if metrics is not None else None
    return predictions_path, metrics_path
