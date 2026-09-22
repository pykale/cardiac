# Examples

Each example is named `<method>_<modality>`, and each is a complete study: a cohort, an
endpoint, a pretraining stage, a fine-tuning stage and a report.

| Example | Modalities | Method | Endpoint |
| --- | --- | --- | --- |
| [`cardiovae_multimodal/`](cardiovae_multimodal/) | Chest X-ray + ECG | Product-of-experts VAE, tri-stream ELBO | Cardiac haemodynamic instability |
| [`lsemvae_ecg/`](lsemvae_ecg/) | Multi-lead ECG, one modality per lead | Hierarchical-expert VAE, latent alignment | Pulmonary hypertension |
| [`synthetic_data.py`](synthetic_data.py) | Synthetic | — | — (no data needed) |

## What belongs here, and what does not

The boundary is the whole point of the layout:

> **Mechanisms belong in the library; dataset and experiment decisions belong here.**

Anything that names a cohort, a column, a clinical threshold, a file layout or an output
directory is an experiment, so it lives in this directory. Everything an experiment
*composes* — the models, the fusion mechanisms, the objectives, the trainer, the
metrics, the interpretation — comes from `kalecardiac` and is exercised by the tests
under `tests/`.

That is enforced rather than merely stated:
[`tests/test_package_structure.py`](../tests/test_package_structure.py) parses every
library module and fails if one mentions a cohort name outside a docstring, hardcodes an
absolute path, or imports from `examples/`.

In practice it means an example never defines a model:

```python
# What an example does
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask

model = LSEMVAE(leads=cfg.ECG.LEADS, length=cfg.ECG.LENGTH, fusion=cfg.FUSION.LATENT_METHOD)
```

## Structure

Every example has the same four parts, so moving between them costs nothing:

```text
<example>/
├── README.md      what the study is, what data it needs, how to read its output
├── config.py      the package defaults extended with this experiment's settings
├── data.py        where the cohort's files are, and what its columns mean
├── runner.py      the orchestration: pretrain, fold, fine-tune, score, report
├── main.py        the command line
└── configs/
    ├── quick.yaml a synthetic run that finishes in about a minute on a CPU
    └── full.yaml  the published setup
```

Examples run as modules from the repository root, so their imports resolve as ordinary
packages and no `sys.path` handling is needed:

```bash
python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/quick.yaml
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml
```

## Running without data

Both examples fall back to [`synthetic_data.py`](synthetic_data.py) when
`DATASET.ROOT` is empty, which is what the `quick.yaml` configurations do. The whole
workflow then runs with no data, no credentials and no network.

The synthetic cohorts have a real dependence between signal and endpoint, so a broken
pipeline scores about 0.5 and a working one scores clearly above it — a smoke test that
passes on noise tests nothing. They are **not** physiologically accurate ECGs, and a
number from a synthetic run is a verification, not a result. Every run says so in its
first log line.

## Data and privacy

Neither study's cohort is in this repository, and neither can be: they are clinical
registries and a biobank holding identifiable patient data, governed by the approvals
under which they were collected. Nothing patient-derived is committed here — no
recordings, no images, no identifiers, no labels, no derived per-subject files.

What *is* here is the workflow, and enough documentation of the expected schema that a
researcher with approved access to a comparable cohort can point `DATASET.ROOT` at
their own copy and reproduce it. Each example's README gives the layout in full.

## Splitting

Both examples cross-validate by default, because a cardiac cohort of a few hundred
subjects leaves too few positives in a single held-out split to separate one model from
another.

| `DATASET.SPLIT_MODE` | Test set | Use when |
| --- | --- | --- |
| `cv` (default) | Each of `NUM_FOLDS` folds in turn | The cohort is small, which it usually is |
| `random` | A fresh stratified draw | A single held-out split is enough |
| `predefined` | An assignment the study fixed | The cohort publishes one — read it and apply `loaddata.Predefined` |

Splitting is always subject-level: a subject contributing two recordings has both on the
same side, and validation is always carved out of the training half, never out of the
test set. Anything derived from labels — a class weight, most obviously — is computed
from the training split of each fold and nowhere else.
