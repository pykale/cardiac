# CardioVAE: chest X-ray and ECG for cardiac haemodynamics

Multimodal representation learning from the two cheapest cardiac investigations, then
fine-tuning onto an endpoint that otherwise requires a catheter.

```text
chest X-ray + ECG
      ↓
multimodal representation learning     (unlabelled, product-of-experts VAE, tri-stream ELBO)
      ↓
fine-tuning                            (labelled, frozen encoders + a clinical head)
      ↓
cardiac haemodynamic prediction
```

This reproduces the workflow of Suvon et al., *Multimodal Variational Autoencoder for
Low-cost Cardiac Hemodynamics Instability Detection*, MICCAI 2024
([arXiv:2403.13658](https://arxiv.org/abs/2403.13658),
[reference implementation](https://github.com/Shef-AIRE/AI4Cardiothoracic-CardioVAE)).
Every model, objective, metric and training loop comes from `kalecardiac`; this
directory holds only what is specific to the experiment — the cohort layout, the
endpoint definition, the fold structure, and the report.

## Run it

```bash
# No data needed: synthetic cohort, a minute on a CPU
python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/quick.yaml

# On your own cohort
python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/full.yaml \
    DATASET.ROOT /path/to/cohort

# Compare what each modality is worth
python -m examples.cardiovae_multimodal.main --arms ecg,cxr,ecg+cxr

# Predict the measurement itself rather than a threshold on it
python -m examples.cardiovae_multimodal.main ENDPOINT.REGRESSION True
```

With `DATASET.ROOT` empty the run generates its cohort. That verifies the pipeline
executes end to end; **it is not a result**, and the run says so in its first log line.

## Data

The cohort the paper used is a clinical registry holding identifiable patient data and
is not distributed — not here and not anywhere. The example reads any cohort with the
layout below, so a researcher with approved access to one can reproduce the workflow by
supplying their own path.

```text
<DATASET.ROOT>/
├── cohort.csv              one row per labelled subject
├── ecg/
│   └── <subject_id>.npy    (num_leads, num_samples) float array
└── cxr/
    └── <subject_id>.npy    (height, width) or (channels, height, width) float array
```

`cohort.csv` needs two columns, and may carry any others:

| Column | Meaning |
| --- | --- |
| `subject_id` | Matches the recording and image filenames |
| `ENDPOINT.VALUE_COLUMN` | The measurement the endpoint thresholds, e.g. a mean pulmonary arterial pressure in mmHg |

Subjects with a recording and an image but no row in the table still take part in
**pretraining** — that asymmetry is the point of the method. Subjects with no recorded
measurement are dropped from fine-tuning rather than assumed negative, and the run
logs how many.

Pretraining in the paper used large public collections of radiographs and ECGs
available from PhysioNet under their own credentialing; point `DATASET.ROOT` at a
directory prepared from those, or pretrain separately and load the checkpoint with
`MODEL.PRETRAINED`.

## What the configuration controls

| Setting | Effect |
| --- | --- |
| `ECG.NUM_CHANNELS` | `1` flattens every lead into one long signal, as the reference implementation does; `12` encodes the leads as channels |
| `ECG.LENGTH` | Samples the recording is cropped or padded to. With flattening this is *leads × samples per lead* |
| `OBJECTIVE.RECONSTRUCTION_WEIGHTS` | Per-modality weights. The ECG is weighted ten times the radiograph because a sum of squares over 60,000 samples and one over 50,176 pixels are not comparable quantities |
| `OBJECTIVE.SCALE_FACTOR` | Brings the summed reconstruction into range of the KL term |
| `OBJECTIVE.ANNEALING_EPOCHS` | Epochs over which the KL weight rises from 0 to 1, so the model learns to reconstruct before it is asked to regularise |
| `OBJECTIVE.UNIMODAL_STREAMS` | The tri-stream objective: an ELBO through the pair and one through each modality alone, so an encoder stays usable when the other is absent |
| `MODEL.FREEZE_ENCODERS` | Freeze the pretrained encoders while fine-tuning, as the paper does on a small labelled cohort |
| `ENDPOINT.THRESHOLD` | The clinical cut-off applied to the measurement |
| `ENDPOINT.REGRESSION` | Predict the measurement rather than the threshold |

## Outputs

```text
<OUTPUT.OUT_DIR>/
├── config.yaml             the exact configuration this run used
├── pretrained.pt           the representation-learning weights
├── results.csv             one row per modality arm
├── summary.json            fold means and spreads
└── <arm>/
    ├── predictions.csv     one row per subject per fold, traceable by subject_id
    ├── metrics.json        per-fold metrics and their summary
    └── interpretation.json attribution shares and modality ablation
```

## Reading the numbers

Cross-validation over a few hundred subjects is what makes a cardiac result reportable
at all at this cohort size, but the folds share training subjects and are not
independent: the standard deviation across them describes how the estimate varied, not
a confidence interval for the population. For the latter, `kalecardiac.evaluate`
provides `bootstrap_ci`, which resamples subjects instead.

The modality arms are scored on the same subjects by construction
(`DATASET.REQUIRE_MODALITIES`), so the comparison between them is meaningful. Comparing
against the published figures is not: those were obtained on a different cohort.

## Attribution

The methodology is the authors'; the implementation here is a refactor, and any
difference is documented in [`docs/models.md`](../../docs/models.md). Please cite the
original paper if you build on this.
