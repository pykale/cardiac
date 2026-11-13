# API overview

The public API of `kalecardiac`, stage by stage, following the pipeline:

```text
load  →  preprocess  →  embed  →  fuse  →  predict  →  evaluate  →  interpret
```

For *why* it is shaped this way, see [docs/architecture.md](docs/architecture.md). For a running example, [docs/quickstart.md](docs/quickstart.md).

**Names follow PyKale.** Modules are named for the modality they serve (`signal_access`, `image_access`, `signal_transform`, `image_transform`), functions that act start with a verb (`load_`, `get_`, `check_`, `normalize_`, `interpolate_`, `resample_`), and spelling is American, as in PyKale, scikit-learn and PyTorch. Where PyKale already has a function, the name is PyKale's: `interpolate_signal` and `normalize_signal` are the functions `kale.prepdata.signal_transform` has, for the `(channels, samples)` layout used here. Metrics and losses are nouns, as in scikit-learn and `torch.nn.functional`.

**Nothing here duplicates a library this one depends on.** Transforms are chained with `torchvision.transforms.Compose`; classification and regression losses are PyTorch's; class weights come from scikit-learn's `compute_class_weight`; resampling is SciPy's polyphase `resample_poly`; R², RMSE and MAE are scikit-learn's. What this package defines is what those libraries do not have.

---

## The whole workflow in one page

```python
from kalecardiac.loaddata import ColumnTarget, ECGArraySource, CrossValidationSplit, MultimodalDataset, get_lead_sources
from kalecardiac.prepdata import get_ecg_transform
from kalecardiac.model.embed import LSEMVAE, MultimodalPredictor
from kalecardiac.pipeline import CardiacTrainer, ClassificationTask, ReconstructionTask
from kalecardiac.evaluate import binary_metrics, predict_split
from kalecardiac.interpret import attribution_ratios, modality_attributions

# load ------------------------------------------------------------------ #
recordings = ECGArraySource(
    arrays, leads=LEADS, sampling_rate=500, transform=get_ecg_transform(length=5000, sampling_rate=500)
)
sources = get_lead_sources(recordings)  # one modality per lead
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

A `ModalitySource` answers two questions per subject: what its value is, and what shape to substitute when there is not one. Implement it to add a *kind* of modality; adding a modality is a dictionary entry.

| Name | Reads |
| --- | --- |
| `ECGArraySource` | Recordings in memory, by named lead |
| `ECGFileSource` | One recording file per subject, lazily, with a one-entry cache |
| `ECGFileSource.from_dicom(root, select=None, transform=None)` | A folder of DICOM ECG exports, one folder per subject: the leads and their rate are read from the headers, and every subject's leads are found by name |
| `LeadSource` | A single lead of an ECG source, as its own modality |
| `get_lead_sources(source, leads=None, prefix="")` | **Splits a recording into one source per lead** — the whole of "each lead is a modality" |
| `ImageArraySource` / `ImageFileSource` | Any 2-D cardiac modality |
| `ImageFileSource.from_dicom(root, transform=None)` | A folder of DICOM image exports, one folder per subject, each image as a viewer displays it |
| `ArraySource` | Anything already one array per subject — a clinical vector, a frozen embedding |
| `MultimodalDataset.from_arrays({"signal": ecg, "image": cxr}, targets={"label": y})` | A cohort straight from aligned tensors, as PyKale's `SignalImageDataset` takes them |

### Lead vocabulary

`STANDARD_12_LEAD`, `LIMB_LEADS`, `PRECORDIAL_LEADS`, and `normalize_lead_name` / `normalize_lead_names` / `get_lead_indices`, which match names case-insensitively with a `LEAD_` or `Lead` prefix stripped -- so DICOM's `Lead I (Einthoven)` is lead `I`. `check_ecg_array` and `check_image_array` coerce shapes, transposing only where it is unambiguous.

### DICOM

Read with PyKale's `kale.loaddata.image_access.read_dicom_phases`, one subject per folder, the folder's name -- not the header's `PatientID` -- identifying the subject, since that is what pairs one modality's export with another's. Needs the `imaging` extra.

| Name | What it does |
| --- | --- |
| `get_dicom_paths(root)` | Each subject's folder (or `.dcm` file) under `root`, keyed by name |
| `read_dicom(path)` | The one DICOM a subject's folder holds, as a `pydicom` dataset; several are an error, not a guess |
| `read_dicom_ecg(path, group=None)` | `(recording, leads, rate)`: the recorded waveform -- not a derived median beat -- in millivolts, its leads named canonically |
| `read_dicom_image(path)` | The pixels after the modality LUT, the VOI LUT or window, and a `MONOCHROME1` inversion |

### Supervision

| Name | What it does |
| --- | --- |
| `ColumnTarget(frame, columns, id_column)` | Reads named columns of a table; `{"label": "has_ph"}` or `{"value": "mpap_mmhg"}` |
| `ColumnTarget.from_mapping({"label": {...}})` | The door in for a cohort that never had a table |
| `Target`, `check_target` | The contract, and the runtime check a `Protocol` alone cannot give |
| `LABEL_KEY`, `VALUE_KEY` | The batch target keys, so nothing guesses at a name |

### Splitting

Splitter objects with scikit-learn's shape, producing a **named** mapping rather than a pair. Rows sharing `group_by` always land together — in a cardiac cohort that is not a formality, since one subject often contributes several recordings.

| Name | Test set |
| --- | --- |
| `HoldOutSplit(test_size, val_size, group_by, stratify_by)` | One stratified, grouped draw |
| `CrossValidationSplit(n_splits, ...)` | Each fold in turn. The default for a cohort of a few hundred |
| `PredefinedSplit(assignment, ...)` | An assignment the study fixed |

`SPLIT_NAMES` is `("train", "val", "test")`. Validation is always carved out of the training half; `val_size=0` gives a plain two-way split. Each splitter draws its folds with scikit-learn's `StratifiedGroupKFold`, `StratifiedKFold` or `StratifiedShuffleSplit` and adds what those lack: a third, validation set, and stratification by several columns at once (`get_stratification_labels`).

---

## `prepdata` — transforms

**Nothing here is fitted.** Every transform is a function of one recording or one image, so a source may apply it without knowing which split the subject landed in.

### Signals — `(channels, samples)` in, the same out

| Name | Does |
| --- | --- |
| `interpolate_signal` | Fills NaNs along time. Runs first: a NaN spreads through everything after it |
| `crop_or_pad_signal(signal, length, center=False)` | Forces one length |
| `resample_signal(signal, source_rate, target_rate)` | SciPy's polyphase resampler: filters before it decimates, so nothing aliases |
| `normalize_signal(signal, per_channel=True)` | Zero mean, unit variance — PyKale's function, channels first. Discards absolute millivolts |
| `scale_signal(signal, scale)` | Divides by a constant, **keeping** absolute amplitude |
| `clip_signal(signal, limit)` | Bounds a saturated electrode's excursion |
| `check_signal(signal)` | Coerces to a float32 `(channels, samples)` array, as scikit-learn's `check_array` does |
| `get_ecg_transform(length, sampling_rate, ..., interleave=False)` | The usual four, in the order that is not interchangeable, as a `Compose`; `interleave` adds `interleave_leads` last |

Steps are chained with `torchvision.transforms.Compose`; a source's `transform` takes one.

### ECG — the part that needs to know a lead from a channel

| Name | Does |
| --- | --- |
| `select_leads(signal, available, wanted)` | Selection *and* ordering in one operation |
| `derive_limb_leads(signal, available, wanted)` | Reconstructs III, aVR, aVL, aVF from I and II — exactly, by Einthoven and Goldberger |
| `get_lead_selector(available, wanted, derive=False)` | The one-argument callable form, for a `Compose` |
| `interleave_leads(signal)` | `(leads, samples)` to `(1, samples * leads)`, sample by sample, as CardioVAE reads an ECG |
| `DERIVABLE_LIMB_LEADS` | Which leads follow from I and II, and how |

### Images — `(channels, height, width)` in, the same out

`resize_image`, `scale_image_intensity`, `normalize_image`, `check_image`, and `get_image_transform(size, scale=True)`, which returns a `Compose`.

---

## `model.layers` — blocks that know no modality

### Gaussian experts — the mathematical core

All take `mean` and `log_var` of `(M, B, D)` and an optional `(M, B)` mask, and return a `GaussianPosterior`.

| Name | Combines by |
| --- | --- |
| `product_of_experts` | Precision-weighted product. Never less certain than any expert |
| `mixture_of_experts` | Weighted parameter average. No expert dominates |
| `mean_of_experts` | The equal-weight case |
| `hierarchical_experts(..., num_groups)` | Product within groups, average across them (HiME) |
| `wasserstein_barycenter` | Mean of means and of **standard deviations** |
| `EXPERT_FUSIONS`, `build_expert_fusion(name, **kw)` | The registry, and construction by name |
| `get_expert_groups(num_experts, num_groups)` | Contiguous grouping that uses **every** expert |
| `prior_expert(shape)` | The standard-normal expert |
| `reparameterize(mean, log_var)` | The differentiable sample. Always samples |
| `GaussianPosterior` | `.mean`, `.log_var`, `.variance`, `.stddev`, `.sample()` |
| `GaussianHead(in_features, latent_dim)` | Features to a posterior |

### Backbones and decoders

`Conv1dEncoder` / `Conv1dDecoder`, `Conv2dEncoder` / `Conv2dDecoder`, `ResidualConv1dEncoder`, `TransformerSignalEncoder`, `MLP`, plus `conv_output_length` and `conv_output_size`.

---

## `model.embed` — encoders, models, and fusion

### Adapters

| Name | Turns a backbone into |
| --- | --- |
| `VariationalEncoder(backbone, latent_dim)` | A modality of a multimodal VAE |
| `FeatureEmbedder(backbone, out_dim=None)` | A modality of a predictor, trained from scratch |
| `LatentEmbedder(encoder, frozen=True)` | **The transfer bridge**: a pretrained encoder's posterior mean, as an embedding |
| `build_signal_encoder(name, ...)`, `build_image_encoder(...)` | Construction by name; `SIGNAL_ENCODERS` lists them |

Freezing is PyTorch's own `module.requires_grad_(False)`, which `LatentEmbedder` calls.

### Generative models

```python
MultimodalVAE(encoders, decoders, latent_dim, fusion="poe", use_prior=True, **fusion_kwargs)
```

- `.encode(modalities, present, only=None)` → `(joint, per_modality)` posteriors
- `.decode(latent)` → one reconstruction per decoded modality
- `.forward(modalities, present, unimodal_streams=False)` → `VAEOutput`
- `.latent_embedders(modalities=None, frozen=True)` → the embedders a predictor fuses

`VAEOutput` carries `joint` (a `VAEStream`), `unimodal`, `modality_posteriors` and `present`, with `.posterior` and `.latent` as shorthands for the joint.

Named configurations: `CardioVAE(..., fusion="poe")` and `LSEMVAE(leads=..., fusion="hime")`. Both take `fusion`, `use_prior` and `kernel_size`, so an ablation is an argument rather than a model.

`CardioVAE()` with its defaults is PyKale's `SignalImageVAE()` parameter for parameter, and `CardioVAE.load_pykale_state_dict(state_dict)` loads the weights PyKale -- or the reference implementation -- saved; `remap_pykale_state_dict` is the renaming it applies.

### Discriminative model

```python
MultimodalPredictor(embedders, head_factory, method="concat", fusion_dim=None, modality_dropout=0.0)
```

Returns a `PredictionOutput` with `prediction`, `representation`, `embeddings` and, for attention fusion, `weights`. Fusion blocks: `ConcatFusion`, `MeanFusion`, `AttentionFusion`, with `FUSION_METHODS` and `build_fusion`. `modality_dropout` teaches a model to survive a missing lead.

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

The discriminative objectives are PyTorch's: a task calls `binary_cross_entropy_with_logits`, `cross_entropy` and `mse_loss` directly. A class weight comes from scikit-learn's `compute_class_weight("balanced", ...)`, over the **training** split only.

---

## `pipeline` — one trainer, several tasks

```python
CardiacTrainer(model, task, optimizer=None, max_epochs=50, init_lr=1e-3, grad_clip=0.0)
```

`.forward(batch)`, `.predict(batch)` (no gradients, mode restored), `.compute_loss(batch, split_name)`, and `.monitor` / `.monitor_mode` for configuring a checkpoint or early-stopping callback without knowing which way the metric runs.

| Task | Supervises on | Metric |
| --- | --- | --- |
| `ReconstructionTask(weights, scale_factor, annealing_epochs, alignment_weight, unimodal_streams, kinds, free_bits)` | The batch's own modalities — **no labels** | The objective |
| `ClassificationTask(num_classes=2, pos_weight, class_weight, hidden_dims, dropout)` | `label` | ROC-AUC; macro one-versus-rest above two classes |
| `RegressionTask(hidden_dims, dropout)` | `value` | R² |

A task decides exactly four things: the head, the loss, what the model is asked to produce (`forward_kwargs`), and the epoch metric. A new endpoint is a new task, never a new trainer.

---

## `evaluate` — scoring, by task

| Name | Returns |
| --- | --- |
| `roc_auc(labels, scores)` | The area |
| `sensitivity_specificity(labels, scores, threshold)` | The clinical pair |
| `binary_metrics(labels, scores, threshold)` | Counts, ranking metrics, threshold metrics and the confusion matrix, JSON-ready |
| `multiclass_roc_auc(labels, scores)` | Macro one-versus-rest area: every class weighs the same, however rare |
| `multiclass_metrics(labels, scores, class_names)` | Accuracy, balanced accuracy, macro F1, the macro area and the confusion matrix |
| `mean_roc_curve(labels_per_run, scores_per_run)` | Curves interpolated onto a shared grid, with a variability band |
| `regression_metrics(targets, predictions, unit)` | RMSE, MAE and R², plus the signed bias, in the endpoint's units |
| `predict_split(model, loader, split, fold)` | A `SplitPredictions`, traceable by `subject_id` |
| `save_predictions(out_dir, predictions, metrics)` | A CSV and a JSON |
| `summarize_folds(fold_metrics)` | Mean and spread **over folds** |
| `flatten_fold_summary(summary)` | One results-table row from a fold summary |
| `pool_folds(fold_predictions)` | Every fold's labels and scores, for a **pooled** metric |
| `bootstrap_ci(labels, scores, metric, ...)` | A percentile interval, resampling *subjects* |

Binary labels must be exactly 0 or 1, and multiclass labels must be valid class indices. A two-column multiclass score uses binary ROC-AUC. Bootstrap resamples with NaN or infinite metrics are skipped; a non-finite metric on the full split raises `MetricError`.

`MetricError` is raised where a metric cannot be computed — a one-class split, say — which a trainer treats as "skip this epoch's metric" rather than a failure.

Scores may be logits or probabilities throughout. Binary scores in `[0, 1]` and multiclass rows in `[0, 1]` that sum to one are read as probabilities, and anything else as logits. Multiclass logits pass through a softmax because one-versus-rest ranking can change after that conversion.

`pool_folds` pairs each fold's targets with its scores and rejects a fold with unequal lengths. Pool predictions from different fitted models only when their scores share a comparable scale, such as calibrated probabilities; otherwise summarize the metric calculated separately for each fold.

---

## `interpret` — what the model used

| Name | Answers |
| --- | --- |
| `modality_attributions(model, modalities, target, present, n_steps, baselines)` | Where in each modality the evidence was. Attributes only what the model reads |
| `attribution_ratio(attribution, threshold)` | How much of one modality was strongly attributed |
| `attribution_ratios(attributions, threshold)` | Per modality, with each one's share — the lead-importance table |
| `segment_attribution_ratios(attribution, segments, window, threshold)` | Which part of the cardiac cycle |
| `delineate_ecg_waves(signal, sampling_rate)` | Where P, Q, R, S and T are (needs `[ecg]`) |
| `modality_ablation(model, loader, metric, target_key, modalities=None)` | What the metric loses when a modality is withheld. The metric sees `(N,)` scores, or `(N, K)` logits for a multiclass model |
| `progressive_ablation(model, loader, metric, target_key, order)` | How the metric degrades as modalities are withheld cumulatively — the reduced-lead deployment curve |
| `collect_latents(model, loader, per_modality=False)` | The learned representation over a cohort |
| `project_latents(features, method="pca"|"umap", ...)` | A low-dimensional projection of it |
| `normalize_attribution(attribution)` | Min-max rescaling, with its caveat documented |

Everything returns arrays and dictionaries. Nothing plots.

---

## `auto` — the same workflow, read from a configuration

Every stage above takes explicit arguments. `kalecardiac.auto` reads them from a YACS configuration instead, and holds what every experiment does with the same settings the same way, so an example keeps only where its cohort is, what its columns mean and which model it builds:

```python
from kalecardiac.auto import check_endpoint, cross_validate, derive_endpoint, get_endpoint_target, get_folds, pretrain

check_endpoint(cfg)  # before any data is read
table = derive_endpoint(cfg, table)  # adds label, and value, from ENDPOINT
target = get_endpoint_target(cfg, table, "subject_id")
pretrained = pretrain(cfg, LSEMVAE(leads=LEADS, length=5000), MultimodalDataset(unlabelled_ids, sources), out_dir)
summary = cross_validate(cfg, pretrained, six_lead_sources, target, get_folds(cfg, table), out_dir / "six_leads")
```

| Name | Reads | Does |
| --- | --- | --- |
| `check_endpoint(cfg)` | `ENDPOINT` | Refuses an endpoint that cannot be built, before any data is read |
| `derive_endpoint(cfg, table)` | `ENDPOINT` | Adds `label` — a threshold on `VALUE_COLUMN`, or a class read from `LABEL_COLUMN` — and `value`; drops subjects with no endpoint |
| `get_endpoint_target(cfg, table, id_column)` | `ENDPOINT` | The `ColumnTarget` the task is supervised by |
| `get_prediction_task(cfg, labels)` | `ENDPOINT`, `MODEL`, `OBJECTIVE` | The task, its class weights taken from the training fold's labels by `compute_class_weight` |
| `score_predictions(cfg, predictions)` | `ENDPOINT` | Binary, multiclass or regression metrics |
| `get_headline_metric(cfg)`, `get_ablation_metric(cfg)` | `ENDPOINT` | The metric a run is summarized by, and the threshold-free one a withheld modality is measured in |
| `get_ecg_transform_from_cfg(cfg, length=None, num_leads=None)` | `ECG` | `prepdata.get_ecg_transform` with its arguments read from `ECG`; `NUM_CHANNELS` 1 interleaves the leads |
| `get_folds(cfg, table)` | `DATASET`, `SOLVER` | `Fold`s, from `CrossValidationSplit` or `HoldOutSplit` |
| `pretrain(cfg, model, dataset, out_dir, kinds=None)` | `OBJECTIVE`, `SOLVER`, `MODEL` | Learns a representation without labels, or loads `MODEL.PRETRAINED` strictly |
| `fine_tune(cfg, pretrained, sources, target, fold, out_dir)` | `FINETUNE`, `FUSION`, `MODEL` | One fold: a predictor over the pretrained encoders, its best checkpoint restored, its test split scored |
| `cross_validate(cfg, pretrained, sources, target, folds, out_dir, report=None)` | the above, `OUTPUT` | A fresh copy per fold; predictions and metrics written; the last fold interpreted; `report` called after each fold, off the training's random stream |
| `interpret_model(cfg, model, loader, out_dir)` | `OUTPUT` | Attribution and modality ablation, written as `interpretation.json` and `attribution.csv` |

---

## `utils` and `config`

`set_seed`, `seed_worker`, `ensure_dir`, `write_json`, `write_csv`, and `load_checkpoint(path, model, strict)` — which unwraps a `torch.compile` or Lightning prefix, so loading a pretrained cardiac model is one line rather than a key-mangling dictionary comprehension.

`preserve_rng_state()` is a context manager for work that only reports — an ablation, an attribution, a latent projection. Every DataLoader pass draws from PyTorch's generator, so without it, turning a report on or off would change every model trained after it.

`get_cfg_defaults()` returns the YACS schema; `parse_weights(["ecg:10", "cxr:1"])` reads per-modality weights, which are pairs because the modality names belong to the dataset rather than to the schema.

`kalecardiac.utils.renames` lists the names the API had before it took PyKale's. Each subpackage consults it, so `from kalecardiac.prepdata import standardise` fails saying it is now `normalize_signal`, and a YAML file setting `ECG.STANDARDISE` fails saying `ECG.NORMALIZE`, rather than either failing as a name that never existed.

---

## What the boundary looks like in practice

| Belongs in `kalecardiac/` | Belongs in `examples/` |
| --- | --- |
| `ECGArraySource`, `get_lead_sources` | Where the recordings are on disk |
| `get_ecg_transform` | Which rate *this* cohort was recorded at |
| `hierarchical_experts`, `LSEMVAE` | Which lead configuration a study fine-tunes on |
| `elbo_loss`, `latent_alignment_loss` | The weights a paper reported |
| `binary_metrics`, `bootstrap_ci` | What the endpoint is, and where its threshold falls |
| `modality_ablation` | The figure it goes into |

Enforced by [`tests/test_package_structure.py`](tests/test_package_structure.py), which fails if a library module names a cohort outside a docstring, hardcodes an absolute path, or imports from `examples/`.
