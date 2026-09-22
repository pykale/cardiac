# Quickstart

From a fresh checkout to a trained cardiac model, with no data.

## Install

KaleCardiac supports Python 3.10, 3.11 and 3.12. Install
[PyTorch](https://pytorch.org/get-started/locally/) matching your hardware first, then:

```bash
git clone https://github.com/pykale/cardiac.git
cd cardiac
pip install -e .
```

Heavy libraries are kept out of the core install:

```bash
pip install -e ".[ecg]"        # wfdb, neurokit2 — reading vendor formats, wave delineation
pip install -e ".[imaging]"    # pillow, pydicom — reading images from disk
pip install -e ".[interpret]"  # captum, umap-learn — attribution and latent projection
pip install -e ".[dev]"        # pytest, ruff, mypy, pre-commit
```

The signal and image transforms are pure NumPy and Torch, so the core install is enough
to preprocess arrays, train every model and score it. The extras are for reading formats
and explaining results.

Check it:

```bash
python -c "import kalecardiac, torch, kale; print(kalecardiac.__version__)"
```

## Run an example, with no data

Both supported studies run on generated cohorts when no data path is given:

```bash
python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/quick.yaml
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml
```

Each takes about a minute on a CPU and writes predictions, metrics and a configuration
record to `outputs/`. The first log line says the run is synthetic — **a number from it
verifies the pipeline, it is not a result**.

## The same thing from Python

The shortest complete workflow: load, preprocess, pretrain without labels, fine-tune on
an endpoint, score.

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from examples.synthetic_data import make_synthetic_multimodal_data
from kalecardiac.evaluate import binary_metrics, predict_split
from kalecardiac.loaddata import (
    ColumnTarget,
    ECGArraySource,
    HoldOut,
    MultimodalDataset,
    collate_subjects,
    lead_sources,
)
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask
from kalecardiac.prepdata import build_ecg_pipeline
from kalecardiac.utils import set_seed

set_seed(2026)
cohort = make_synthetic_multimodal_data(num_subjects=64, num_samples=512)

# --- load: one modality per lead -------------------------------------------------
recordings = ECGArraySource(
    cohort.ecg,
    leads=cohort.leads,
    sampling_rate=cohort.sampling_rate,
    transform=build_ecg_pipeline(length=512, sampling_rate=cohort.sampling_rate),
)
sources = lead_sources(recordings)  # {"I": ..., "II": ..., ...}
target = ColumnTarget.from_mapping({"label": cohort.labels})


def loader(identifiers, with_target=True, shuffle=False):
    dataset = MultimodalDataset(identifiers, sources, target=target if with_target else None)
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, collate_fn=collate_subjects)


# --- pretrain: no labels are used here -------------------------------------------
vae = LSEMVAE(leads=cohort.leads, length=512, latent_dim=64, fusion="hime", num_groups=3)
pretrainer = CardiacTrainer(vae, ReconstructionTask(scale_factor=1e-3, alignment_weight=0.1))
pl.Trainer(max_epochs=5, accelerator="cpu", logger=False, enable_checkpointing=False).fit(
    pretrainer, loader(cohort.subject_id, with_target=False, shuffle=True)
)

# --- fine-tune: frozen encoders, a subset of the leads ----------------------------
frame = target.frame(cohort.subject_id)
split = next(
    HoldOut(test_size=0.25, val_size=0.2, group_by="subject_id", stratify_by=["label"], random_state=0).split(frame)
)
ids = {name: sorted(frame.iloc[split[name]]["subject_id"]) for name in ("train", "val", "test")}

task = ClassificationTask(hidden_dims=(64,))
predictor = MultimodalPredictor(vae.latent_embedders(["I", "II", "aVF"], frozen=True), task.build_head)
model = CardiacTrainer(predictor, task, init_lr=1e-3)
pl.Trainer(max_epochs=10, accelerator="cpu", logger=False, enable_checkpointing=False).fit(
    model, loader(ids["train"], shuffle=True), loader(ids["val"])
)

# --- score ------------------------------------------------------------------------
predictions = predict_split(model, loader(ids["test"]), split="test")
print(binary_metrics(predictions.targets["label"], predictions.scores))
```

The one line worth pausing on is `vae.latent_embedders(["I", "II", "aVF"], frozen=True)`.
The model was pretrained on six leads and is being fine-tuned on three — that is the
twelve-to-six transfer both cardiac studies rest on, and it is a list argument.

## On your own cohort

Nothing above is specific to synthetic data. Replace the source, keep everything else:

```python
from kalecardiac.loaddata import ECGFileSource

recordings = ECGFileSource(
    {"S001": "/data/ecg/S001.npy", "S002": "/data/ecg/S002.npy"},
    leads=["I", "II", "III", "aVR", "aVL", "aVF"],
    sampling_rate=500,
    transform=build_ecg_pipeline(length=5000, sampling_rate=500),
)
```

`ECGFileSource` reads in the DataLoader worker and reuses the most recent read, so
exploding a recording into per-lead modalities costs one read per subject rather than
one per lead. For a format it does not know — WFDB, a DICOM waveform, a vendor export —
pass a `loader`:

```python
import wfdb

ECGFileSource(paths, leads=..., loader=lambda path: wfdb.rdsamp(str(path.with_suffix("")))[0].T)
```

Then point an example at it:

```bash
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/full.yaml \
    DATASET.ROOT /path/to/cohort
```

Each example's README documents the cohort layout it expects in full.

## Changing the experiment without changing the code

Every model choice is configuration:

```bash
# Which fusion mechanism learns the representation
python -m examples.lsemvae_ecg.main FUSION.LATENT_METHOD poe
python -m examples.lsemvae_ecg.main FUSION.LATENT_METHOD moe

# Which backbone encodes each lead
python -m examples.lsemvae_ecg.main ECG.ENCODER transformer

# Whether the alignment term is used at all
python -m examples.lsemvae_ecg.main OBJECTIVE.ALIGNMENT_WEIGHT 0.0

# Which lead configurations to compare
python -m examples.lsemvae_ecg.main --leads I,I+II,I+II+III

# Predict the measurement rather than a threshold on it
python -m examples.lsemvae_ecg.main ENDPOINT.REGRESSION True
```

## Interpreting a trained model

```python
from kalecardiac.evaluate import roc_auc
from kalecardiac.interpret import attribution_ratios, modality_ablation, modality_attributions

batch = next(iter(loader(ids["test"])))
attributions = modality_attributions(model.model, batch.modalities, present=batch.present)

print(attribution_ratios(attributions))  # which lead was attended to
print(modality_ablation(model, loader(ids["test"]), roc_auc, "label"))  # which lead was needed
```

The two answer different questions and can disagree — a lead can be heavily attributed
and still redundant. See [multimodal_fusion.md](multimodal_fusion.md) and
[architecture.md](architecture.md#evaluation-and-interpretation).

## Development

```bash
pip install -e ".[dev,interpret]"
pre-commit run --all-files && pytest --cov=kalecardiac
```

The suite needs no data and no network. To skip the whole-workflow example runs, which
are the slow part:

```bash
pytest -m "not slow"
```

## Next

- [architecture.md](architecture.md) — why the package is shaped this way
- [models.md](models.md) — what each model contributes, and where it departs from its source
- [multimodal_fusion.md](multimodal_fusion.md) — the fusion mechanisms, and choosing one
- [API_overview.md](../API_overview.md) — the public API, stage by stage
