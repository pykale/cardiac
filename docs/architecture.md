# KaleCardiac architecture

Why the package is shaped the way it is. For running things, see the
[quickstart](quickstart.md); for the models themselves, [models.md](models.md); for the
fusion mechanisms, [multimodal_fusion.md](multimodal_fusion.md).

## The pipeline

KaleCardiac follows [PyKale](https://github.com/pykale/pykale)'s verb-oriented pipeline.
Each stage names what it *does*, and a class belongs to exactly one:

```text
loaddata  →  prepdata  →  model  →  pipeline  →  evaluate  →  interpret
  access      transform    learn     train        score       explain
```

```text
kalecardiac/
├── auto/        high-level construction (AutoCardiac* classes, not yet built)
├── loaddata/    access APIs per modality, plus subject-level splitting
├── prepdata/    transforms over signals and images
├── model/       layers/ blocks, embed/ encoders and models, predict/ heads and objectives
├── pipeline/    the trainer and its tasks
├── evaluate/    metrics and prediction scoring, by task
├── interpret/   attribution, lead importance, latent inspection
├── utils/       seeding, artefact writing, checkpoint loading
└── config.py    configuration schema and defaults
```

The dependency order is the pipeline order, and it is enforced:
[`tests/test_package_structure.py`](../tests/test_package_structure.py) parses every
module and fails if `loaddata` imports `model`, or any stage imports `pipeline`.

## The boundary that matters most

> **Mechanisms belong in the library. Dataset and experiment decisions belong in
> `examples/`.**

Anything naming a cohort, a column, a clinical threshold, a file layout or an output
directory is an experiment. The same test suite fails if a library module mentions a
cohort name outside a docstring, hardcodes an absolute path, or imports from
`examples/`. Docstrings are exempt because that is where a refactored model credits the
study it came from, and attribution is required rather than forbidden.

This is not tidiness. Both studies KaleCardiac was built from are research codebases
whose model definitions, file paths, lead orderings and hyperparameters are interleaved
in single notebooks, which is why their ablations are separate near-duplicate files
rather than configuration changes. Separating the two is what turns "the model without
the mixture step" from a 280-line copy into `FUSION.LATENT_METHOD=poe`.

## Data: one contract, any number of modalities

A cohort is a set of subjects and, per subject, some number of *sources*:

```python
MultimodalDataset(
    identifiers,
    sources={"ecg": ECGArraySource(...), "cxr": ImageArraySource(...)},
    target=ColumnTarget(...),
)
```

Nothing about that is specific to two modalities. Twelve single-lead sources, one
recording and an image, or an image and a clinical vector are the same call with a
different dictionary, because sources are **named rather than positional** and each is
asked for one subject at a time. Adding a modality is a dictionary entry; adding a
*kind* of modality is one `ModalitySource` subclass.

### ECG

An ECG is a set of **named leads** sampled together, and both of those are declared
rather than assumed. A cohort may hold twelve leads, the six limb leads a reduced-lead
recorder captures, or a single rhythm strip; a source that assumed one of those would
quietly transpose, reorder or truncate the others.

Lead names are matched case-insensitively with a `LEAD_` prefix stripped, so `aVR`,
`AVR` and `LEAD_aVR` are one lead. That is not fussiness: in one reference
implementation the lead list in the preprocessing script and the list the encoder
indexes disagree — aVL and aVF are transposed between them — so a per-lead model learns
one lead's representation from another's signal, and nothing crashes.

`lead_sources()` turns one recording source into a named source per lead. That single
function is the whole of "each lead is a modality" at the loading stage; after it, a
twelve-lead cohort is an ordinary thirteen-source `MultimodalDataset` and nothing
downstream knows the modalities happen to be leads.

### Images

An image modality is a `(channels, height, width)` tensor, which is equally a chest
radiograph, a short-axis cardiac MRI slice, a CT section or an echocardiography frame.
What differs between them is how the file is read and what preprocessing it needs, and
both are arguments.

Volumetric modalities are deliberately out of scope for now. A cine stack is a different
contract — a spacing, an orientation, a slice axis — and adding a fourth axis to
half-support it would make the classes wrong for both.

### Missing modalities are first class

A source returns `None` for a subject it has nothing for. The dataset substitutes a
zero placeholder of the right shape and records the absence in `present`; the
placeholder is never evidence, because fusion reads `present` and drops the modality.

That is what lets one cohort hold subjects with twelve leads and subjects with six, and
it is not an edge case in cardiac work — it is the deployment scenario. A model
pretrained on a full diagnostic recording has to run on whatever the device in front of
the patient captured.

## Preprocessing: nothing is fitted

Every transform in `prepdata` is a function of **one** recording or **one** image, so a
source may apply it without knowing which split the subject landed in. There is no
fitted state to leak.

A quantity estimated *across* subjects — a cohort-wide mean, a quantile clip — would be
fold-local state and would belong with the fold. Splitting lives in `loaddata` rather
than `prepdata` for the mirror-image reason: a splitter decides which subjects load into
which loader, and the thing that creates the folds cannot be state belonging to one.
scikit-learn draws the same line between `model_selection` and `preprocessing`.

## Models: three stages, one class each

```text
model/
├── layers/    blocks that transform tensors and know no modality
├── embed/     encoders adapted to a contract, and the models that fuse them
└── predict/   heads, and every objective
```

A **layer** transforms tensors. An **embedder** adapts a layer to a contract —
`out_dim` for a predictor, `latent_dim` and a posterior for a VAE. A **head** turns a
representation into a score. `VariationalEncoder(backbone, latent_dim)` is the adapter
that turns *any* backbone into a variational one, which is why changing
`ECG.ENCODER` from `conv` to `residual` to `transformer` touches nothing else.

### One multimodal VAE

Every generative cardiac model here is `MultimodalVAE` with different arguments:

| `fusion` | decoders | the published model this gives |
| --- | --- | --- |
| `poe` | per modality | MVAE (Wu and Goodman, 2018); **CardioVAE** |
| `mean` | per modality | the parameter-averaging MMVAE+ variant |
| `barycenter` | per modality | a Wasserstein-barycenter multimodal VAE |
| `hime` | per modality | a grouped mixture-of-products VAE |
| `hime` | **shared** | **LS-EMVAE** |

Two arguments, five published models. `CardioVAE` and `LSEMVAE` exist as named classes
because those are the names the literature uses and a reader will look for them — but
they are configurations, not architectures, and an ablation is a configuration sweep
rather than a second file to keep in step.

## Representation learning is not a stage

Pretraining a representation and fine-tuning it onto an endpoint are the two halves of
every cardiac workflow here, and neither is a pipeline stage. A multimodal VAE is a
model in `model.embed`; the ELBO that trains it is an objective in `model.predict`; and
the difference between the two halves is **which task** is handed to the trainer:

```python
CardiacTrainer(vae, task=ReconstructionTask(...))  # pretrain, no labels
CardiacTrainer(predictor, task=ClassificationTask(...))  # fine-tune on an endpoint
CardiacTrainer(predictor, task=RegressionTask())  # or on a measured value
```

`LatentEmbedder` is the bridge: it wraps a pretrained variational encoder and presents
its posterior *mean* as an embedding. `vae.latent_embedders(leads, frozen=True)` is
therefore the whole of "freeze the encoders and fine-tune", and naming a **subset** of
the leads is the whole of twelve-to-six transfer.

## One trainer

`CardiacTrainer` takes a model and a task, and neither constrains the other. A task
decides exactly four things — the head, the loss, what the model is asked to produce,
and the epoch metric — which is why adding an endpoint costs a task rather than a
trainer.

There is deliberately no `CardioVAETrainer`, no `LSEMVAETrainer`, no `ECGTrainer`.
There is nothing for one to add: the model is an argument and the objective is an
argument, so a per-model trainer would be a constructor with a name.

**Why this takes a model where KaleCancer's trainer builds one.** KaleCancer's
`CohortTrainer` constructs a fusion model from a set of embedders, because every
experiment there is discriminative. A cardiac workflow is two-stage, and the generative
model that pretrains the representation and the discriminative one that reads it are
different models trained in different runs over cohorts of different sizes. A trainer
that could construct only one of them could not pretrain.

## Evaluation and interpretation

`evaluate` names no endpoint. Pulmonary hypertension detection and elevated wedge
pressure are both a binary label; a mean pulmonary arterial pressure is a number. The
metric set is chosen for a clinical reading rather than a leaderboard — sensitivity and
specificity alongside ROC-AUC, because on a cohort with 20% positives calling everybody
negative scores 80% accuracy.

`interpret` answers three different questions and keeps them apart:

| Question | Mechanism |
| --- | --- |
| Where in a recording was the evidence? | `modality_attributions` (integrated gradients) |
| Which lead, or which modality? | `attribution_ratios` (the attribution ratio and its share) |
| Which part of the cardiac cycle? | `segment_attribution_ratios` with `ecg_wave_segments` |
| Could the model have done without it? | `modality_ablation` |

Attribution and ablation can disagree, and that is informative rather than a bug: a lead
can be heavily attributed and still redundant, because another lead carries the same
information. Nothing here plots — arrays and dictionaries come out, and a publication
figure names a cohort and a claim, so it belongs in `examples/`.

## Reuse of PyKale

KaleCardiac is built on PyKale rather than beside it:

| Used from PyKale | For |
| --- | --- |
| `kale.embed.base_cnn.BaseCNN` | Convolution-block construction |
| `kale.embed.multimodal_fusion.ProductOfExperts` | The product-of-experts closed form |
| `kale.prepdata.signal_transform.interpolate_signal` | Filling dropped samples |
| `kale.pipeline.base_nn_trainer.BaseNNTrainer` | The trainer's Lightning scaffolding |
| `kale.utils.seed.set_seed`, `kale.utils.initialize_nn` | Seeding and head initialisation |

PyKale already carries fixed-shape forms of two encoders here — `SignalVAEEncoder` and
`ImageVAEEncoder`, contributed from the CardioVAE work — and a bimodal
`SignalImageVAE`. KaleCardiac reimplements the encoders rather than wrapping them
because their flattened widths are literals (`64 * 28 * 28`) that are wrong for any
other input shape; `tests/model/layers/test_backbones.py` asserts that the two agree
where the shapes do. `kale.predict.decode` is not importable without
`torch_geometric`, so the decoders and the linear head are implemented here too, as
KaleCancer also found.

## What is deliberately not here

- **No `auto` implementation.** Turning a configuration into a workflow is what an
  example's runner does; which part of that has no dataset in it is not yet clear, and
  an empty API pretending otherwise would be worse than none.
- **No trainer per model, no dataset class per modality combination, no registry with
  one implementation.** Each would be an abstraction with no second user.
- **No vendored third-party model code.** The residual 1-D encoder is a clean
  implementation rather than a copy of the widely-vendored `ResNet1D`, and the ECG
  foundation models the studies benchmark against stay in their own repositories.

## Related documents

- [Quickstart](quickstart.md) — running the pipeline
- [Models](models.md) — CardioVAE, LS-EMVAE, the baselines, and documented departures
- [Multimodal fusion](multimodal_fusion.md) — the fusion mechanisms and when each applies
- [API overview](../API_overview.md) — the public API, stage by stage
- [AGENTS.md](../AGENTS.md) — conventions and constraints for contributors
