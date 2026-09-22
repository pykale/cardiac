# API overview

The public API of `kalecardiac`, stage by stage, following the pipeline:

```text
load  →  preprocess  →  embed  →  fuse  →  predict  →  evaluate  →  interpret
```

For *why* it is shaped this way, see [docs/architecture.md](docs/architecture.md). For a
running example, [docs/quickstart.md](docs/quickstart.md).

---

## The whole workflow in one page

```python
from kalecardiac.loaddata import ColumnTarget, ECGArraySource, CrossValidation, MultimodalDataset, lead_sources
from kalecardiac.prepdata import build_ecg_pipeline
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask
from kalecardiac.evaluate import binary_metrics, predict_split
from kalecardiac.interpret import attribution_ratios, modality_attributions

# load ------------------------------------------------------------------ #
recordings = ECGArraySource(
    arrays, leads=LEADS, sampling_rate=500, transform=build_ecg_pipeline(length=5000, sampling_rate=500)
)
sources = lead_sources(recordings)  # one modality per lead
target = ColumnTarget(table, columns={"label": "has_ph"}, id_column="subject_id")
dataset = MultimodalDataset(subject_ids, sources, target=target)

# embed and fuse: pretrain a representation, with no labels -------------- #
vae = LSEMVAE(leads=LEADS, length=5000, latent_dim=256, fusion="hime")
CardiacTrainer(vae, ReconstructionTask(alignment_weight=0.1))

# predict: fine-tune onto an endpoint, on a subset of the leads ---------- #
task = ClassificationTask(pos_weight=3.0)
predictor = MultimodalPredictor(vae.latent_embedders(SIX_LEADS, frozen=True), task.build_head)
model = CardiacTrainer(predictor, task)

# evaluate and interpret ------------------------------------------------- #
predictions = predict_split(model, test_loader, split="test")
metrics = binary_metrics(predictions.targets["label"], predictions.scores)
shares = attribution_ratios(modality_attributions(model.model, batch.modalities))
```

---

## `loaddata` — access, by modality

### Records

| Name | What it is |
| --- | --- |
| `SubjectSample` | One subject: `subject_id`, `modalities`, `present`, `target`, `metadata` |
| `SubjectBatch` | The collated form, with a leading batch dimension and a `.to(device)` |
| `collate_subjects` | The `collate_fn` a DataLoader needs |
| `release_workers(*loaders)` | Shut down persistent worker pools deliberately, rather than racing interpreter shutdown |

### Sources

A `ModalitySource` answers two questions per subject: what its value is, and what shape
to substitute when there is not one. Implement it to add a *kind* of modality; adding a
modality is a dictionary entry.

| Name | Reads |
| --- | --- |
| `ECGArraySource` | Recordings in memory, by named lead |
| `ECGFileSource` | One recording file per subject, lazily, with a one-entry cache |
| `LeadSource` | A single lead of an ECG source, as its own modality |
| `lead_sources(source, leads=None, prefix="")` | **Splits a recording into one source per lead** — the whole of "each lead is a modality" |
| `ImageArraySource` / `ImageFileSource` | Any 2-D cardiac modality |
| `ArraySource` | Anything already one array per subject — a clinical vector, a frozen embedding |

### Lead vocabulary

`STANDARD_12_LEAD`, `LIMB_LEADS`, `PRECORDIAL_LEADS`, and `canonical_lead` /
`canonical_leads` / `lead_indices`, which match names case-insensitively with a `LEAD_`
prefix stripped. `as_lead_array` and `as_image_array` coerce shapes, transposing only
where it is unambiguous.

### Supervision

| Name | What it does |
| --- | --- |
| `ColumnTarget(frame, columns, id_column)` | Reads named columns of a table; `{"label": "has_ph"}` or `{"value": "mpap_mmhg"}` |
| `ColumnTarget.from_mapping({"label": {...}})` | The door in for a cohort that never had a table |
| `Target`, `check_target` | The contract, and the runtime check a `Protocol` alone cannot give |
| `LABEL_KEY`, `VALUE_KEY` | The batch target keys, so nothing guesses at a name |

### Splitting

Splitter objects with scikit-learn's shape, producing a **named** mapping rather than a
pair. Rows sharing `group_by` always land together — in a cardiac cohort that is not a
formality, since one subject often contributes several recordings.

| Name | Test set |
| --- | --- |
| `HoldOut(test_size, val_size, group_by, stratify_by)` | One stratified, grouped draw |
| `CrossValidation(n_splits, ...)` | Each fold in turn. The default for a cohort of a few hundred |
| `Predefined(assignment, ...)` | An assignment the study fixed |
| `train_test_split(frame, ...)` | The function form, for a single division |
| `subject_frame(ids, **columns)` | Builds the table a splitter splits, for a cohort with none |

`SPLIT_NAMES` is `("train", "val", "test")`. Validation is always carved out of the
training half.

---

## `prepdata` — transforms

**Nothing here is fitted.** Every transform is a function of one recording or one image,
so a source may apply it without knowing which split the subject landed in.

### Signals — `(channels, samples)` in, the same out

| Name | Does |
| --- | --- |
| `interpolate_missing` | Fills NaNs along time. Runs first: a NaN spreads through everything after it |
| `crop_or_pad(signal, length, centre=False)` | Forces one length |
| `resample(signal, source_rate, target_rate)` | Linear interpolation on a regular grid |
| `standardise(signal, per_channel=True)` | Zero mean, unit variance. Discards absolute millivolts |
| `scale_amplitude(signal, scale)` | Divides by a constant, **keeping** absolute amplitude |
| `clip_amplitude(signal, limit)` | Bounds a saturated electrode's excursion |
| `SignalPipeline([...])` | Composes them; what a source's `transform` takes |
| `build_ecg_pipeline(length, sampling_rate, ...)` | Assembles the usual four, in the order that is not interchangeable |

### ECG — the part that needs to know a lead from a channel

| Name | Does |
| --- | --- |
| `select_leads(signal, available, wanted)` | Selection *and* ordering in one operation |
| `derive_limb_leads(signal, available, wanted)` | Reconstructs III, aVR, aVL, aVF from I and II — exactly, by Einthoven and Goldberger |
| `lead_selector(available, wanted, derive=False)` | The one-argument callable form, for a pipeline |
| `DERIVABLE_LIMB_LEADS` | Which leads follow from I and II, and how |

### Images — `(channels, height, width)` in, the same out

`resize_image`, `scale_image`, `standardise_image`, `ImagePipeline`,
`build_image_pipeline(size, scale=True)`.

---

## `model.layers` — blocks that know no modality

### Gaussian experts — the mathematical core

All take `mean` and `log_var` of `(M, B, D)` and an optional `(M, B)` mask, and return a
`GaussianPosterior`.

| Name | Combines by |
| --- | --- |
| `product_of_experts` | Precision-weighted product. Never less certain than any expert |
| `mixture_of_experts` | Weighted parameter average. No expert dominates |
| `mean_of_experts` | The equal-weight case |
| `hierarchical_experts(..., num_groups)` | Product within groups, mixture across them (HiME) |
| `wasserstein_barycenter` | Mean of means and of **standard deviations** |
| `EXPERT_FUSIONS`, `build_expert_fusion(name, **kw)` | The registry, and construction by name |
| `expert_groups(num_experts, num_groups)` | Contiguous grouping that uses **every** expert |
| `prior_expert(shape)` | The standard-normal expert |
| `reparameterise(mean, log_var)` | The differentiable sample. Always samples |
| `GaussianPosterior` | `.mean`, `.log_var`, `.variance`, `.stddev`, `.sample()` |
| `GaussianHead(in_features, latent_dim)` | Features to a posterior |

### Backbones and decoders

`Conv1dEncoder` / `Conv1dDecoder`, `Conv2dEncoder` / `Conv2dDecoder`,
`ResidualConv1dEncoder`, `TransformerSignalEncoder`, `MLP`, plus `conv_output_length`
and `conv_output_size`.

---

## `model.embed` — encoders, models, and fusion

### Adapters

| Name | Turns a backbone into |
| --- | --- |
| `VariationalEncoder(backbone, latent_dim)` | A modality of a multimodal VAE |
| `FeatureEmbedder(backbone, out_dim=None)` | A modality of a predictor, trained from scratch |
| `LatentEmbedder(encoder, frozen=True)` | **The transfer bridge**: a pretrained encoder's posterior mean, as an embedding |
| `build_signal_encoder(name, ...)`, `build_image_encoder(...)` | Construction by name; `SIGNAL_ENCODERS` lists them |
| `freeze(module, frozen=True)` | Stops or resumes gradient accumulation |

### Generative models

```python
MultimodalVAE(encoders, decoders, latent_dim, fusion="poe", use_prior=True, **fusion_kwargs)
```

- `.encode(modalities, present, only=None)` → `(joint, per_modality)` posteriors
- `.decode(latent)` → one reconstruction per decoded modality
- `.forward(modalities, present, unimodal_streams=False)` → `VAEOutput`
- `.latent_embedders(modalities=None, frozen=True)` → the embedders a predictor fuses

`VAEOutput` carries `joint` (a `VAEStream`), `unimodal`, `modality_posteriors` and
`present`, with `.posterior` and `.latent` as shorthands for the joint.

Named configurations: `CardioVAE(...)` and `LSEMVAE(leads=..., ...)`.

### Discriminative model

```python
MultimodalPredictor(embedders, head_factory, method="concat", fusion_dim=None, modality_dropout=0.0)
```

Returns a `PredictionOutput` with `prediction`, `representation`, `embeddings` and, for
attention fusion, `weights`. Fusion blocks: `ConcatFusion`, `MeanFusion`,
`AttentionFusion`, with `FUSION_METHODS` and `build_fusion`. `modality_dropout` teaches
a model to survive a missing lead.

---

## `model.predict` — heads and objectives

| Head | For |
| --- | --- |
| `LinearHead(in_features, out_features=1)` | A linear map; right when the encoder is large and the cohort small |
| `MLPHead(in_features, out_features, hidden_dims, dropout)` | One hidden layer and dropout, as both cardiac studies use |

| Objective | For |
| --- | --- |
| `elbo_loss(reconstructions, targets, posterior, weights, present, kinds, annealing_factor, scale_factor, free_bits)` | The multimodal ELBO |
| `gaussian_kl_divergence(posterior, free_bits=0.0)` | The KL term, clamped so an untrained encoder cannot produce a NaN |
| `reconstruction_loss(recon, target, present, kind)` | Per modality, excluding absent subjects |
| `latent_alignment_loss(modality_posteriors, joint, present)` | Pulls each modality towards the consensus. The joint is detached |
| `binary_cross_entropy`, `cross_entropy`, `mean_squared_error` | The discriminative objectives |
| `class_weight_from_labels`, `positive_weight_from_labels` | Class balance, from the **training** split only |

---

## `pipeline` — one trainer, several tasks

```python
CardiacTrainer(model, task, optimizer=None, max_epochs=50, init_lr=1e-3, grad_clip=0.0)
```

`.forward(batch)`, `.predict(batch)` (no gradients, mode restored), `.compute_loss(batch,
split_name)`, and `.monitor` / `.monitor_mode` for configuring a checkpoint or
early-stopping callback without knowing which way the metric runs.

| Task | Supervises on | Metric |
| --- | --- | --- |
| `ReconstructionTask(weights, scale_factor, annealing_epochs, alignment_weight, unimodal_streams, kinds, free_bits)` | The batch's own modalities — **no labels** | The objective |
| `ClassificationTask(num_classes=2, pos_weight, class_weight, hidden_dims, dropout)` | `label` | ROC-AUC |
| `RegressionTask(hidden_dims, dropout)` | `value` | R² |

A task decides exactly four things: the head, the loss, what the model is asked to
produce (`forward_kwargs`), and the epoch metric. A new endpoint is a new task, never a
new trainer.

---

## `evaluate` — scoring, by task

| Name | Returns |
| --- | --- |
| `roc_auc(labels, scores)` | The area |
| `sensitivity_specificity(labels, scores, threshold)` | The clinical pair |
| `binary_metrics(labels, scores, threshold)` | Counts, ranking metrics, threshold metrics and the confusion matrix, JSON-ready |
| `multiclass_metrics(labels, probabilities, class_names)` | The same for several classes |
| `mean_roc_curve(labels_per_run, scores_per_run)` | Curves interpolated onto a shared grid, with a variability band |
| `r2_score`, `regression_metrics(targets, predictions, unit)` | RMSE, MAE, signed bias and R², in the endpoint's units |
| `predict_split(model, loader, split, fold)` | A `SplitPredictions`, traceable by `subject_id` |
| `save_predictions(out_dir, predictions, metrics)` | A CSV and a JSON |
| `summarise_folds(fold_metrics)` | Mean and spread **over folds** |
| `pool_folds(fold_predictions)` | Every fold's labels and scores, for a **pooled** metric |
| `bootstrap_ci(labels, scores, metric, ...)` | A percentile interval, resampling *subjects* |

`MetricError` is raised where a metric cannot be computed — a one-class split, say —
which a trainer treats as "skip this epoch's metric" rather than a failure.

---

## `interpret` — what the model used

| Name | Answers |
| --- | --- |
| `modality_attributions(model, modalities, target, present, n_steps, baselines)` | Where in each modality the evidence was. Attributes only what the model reads |
| `attribution_ratio(attribution, threshold)` | How much of one modality was strongly attributed |
| `attribution_ratios(attributions, threshold)` | Per modality, with each one's share — the lead-importance table |
| `segment_attribution_ratios(attribution, segments, window, threshold)` | Which part of the cardiac cycle |
| `ecg_wave_segments(signal, sampling_rate)` | Where P, Q, R, S and T are (needs `[ecg]`) |
| `modality_ablation(model, loader, metric, target_key, modalities=None)` | What the metric loses when a modality is withheld |
| `collect_latents(model, loader, per_modality=False)` | The learned representation over a cohort |
| `latent_embedding(features, method="pca"|"umap", ...)` | A low-dimensional projection of it |
| `normalise_attribution(attribution)` | Min-max rescaling, with its caveat documented |

Everything returns arrays and dictionaries. Nothing plots.

---

## `utils` and `config`

`set_seed`, `seed_worker`, `ensure_dir`, `write_json`, `write_csv`, and
`load_checkpoint(path, model, strict)` — which unwraps a `torch.compile` or Lightning
prefix, so loading a pretrained cardiac model is one line rather than a key-mangling
dictionary comprehension.

`get_cfg_defaults()` returns the YACS schema; `parse_weights(["ecg:10", "cxr:1"])` reads
per-modality weights, which are pairs because the modality names belong to the dataset
rather than to the schema.

---

## What the boundary looks like in practice

| Belongs in `kalecardiac/` | Belongs in `examples/` |
| --- | --- |
| `ECGArraySource`, `lead_sources` | Where the recordings are on disk |
| `build_ecg_pipeline` | Which rate *this* cohort was recorded at |
| `hierarchical_experts`, `LSEMVAE` | Which lead configuration a study fine-tunes on |
| `elbo_loss`, `latent_alignment_loss` | The weights a paper reported |
| `binary_metrics`, `bootstrap_ci` | What the endpoint is, and where its threshold falls |
| `modality_ablation` | The figure it goes into |

Enforced by [`tests/test_package_structure.py`](tests/test_package_structure.py), which
fails if a library module names a cohort outside a docstring, hardcodes an absolute
path, or imports from `examples/`.
