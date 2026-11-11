# Changelog

Notable changes to KaleCardiac. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) from 1.0 onwards.

## [Unreleased]

### Added

- **`kalecardiac.auto`: the workflow both examples share, read from a configuration.**
  `auto.endpoint` turns `ENDPOINT` into labels, a task and scores (`check_endpoint`,
  `derive_endpoint`, `get_endpoint_target`, `get_prediction_task`, `score_predictions`,
  `get_headline_metric`, `get_ablation_metric`); `auto.workflow` pretrains, draws the
  folds, fine-tunes, scores and interprets (`pretrain`, `get_folds`, `fine_tune`,
  `cross_validate`, `interpret_model`, `get_ecg_transform_from_cfg`). The two runners
  had each carried a copy of all of it; a test now fails if a runner defines `pretrain`,
  `fine_tune` or `cross_validate` again.
- `ENDPOINT` and `FINETUNE` are sections of the package schema, with defaults that name
  nothing, rather than nodes each example created. `SOLVER.GRAD_CLIP`,
  `OUTPUT.ATTRIBUTION_THRESHOLD` and `OUTPUT.ATTRIBUTION_STEPS` replace values both
  examples hardcoded, at the same values.
- `evaluate.flatten_fold_summary`, a fold summary as one row of a results table.
- **A tutorial notebook**, `tutorials/mdas_tutorial/tutorial-CardioVAE.ipynb`, with its
  pretraining and fine-tuning configurations beside it in `configs/`
  (`config_cardiovae_pretrain.yaml` and `config_cardiovae_finetune.yaml`), each merged
  onto the package's defaults: paired chest X-rays and twelve-lead ECGs, read from raw
  DICOM, taken through pretraining, fine-tuning, evaluation and interpretation in a few
  calls each. In Colab it installs KaleCardiac from GitHub with pip and fetches its own
  folder alone, then downloads the dataset from Dropbox into the runtime, with no Google
  Drive; one commented-out cell points it at a user's own data instead, whose ECGs are
  resampled from the rate their DICOM headers state. A test runs the notebook on a
  synthetic cohort written as DICOM in the same layout, without the download. It
  replaces `tutorials/cardiac_abnormality_assessment.ipynb`, which loaded prepared
  tensors.
- **A second tutorial notebook**, `tutorials/mdas_tutorial/tutorial-LSEMVAE.ipynb`, in the
  same layout and on the same dataset, reading its ECGs alone with each lead a modality:
  LS-EMVAE is pretrained on all twelve leads from raw DICOM, with its HiME fusion, shared
  decoder and latent alignment, its leads' agreement with the joint posterior checked,
  fine-tuned as published -- each lead's frozen posterior mean, concatenated -- on the six
  limb leads or, with `USE_12_LEADS`, all twelve, scored again with leads withheld down to
  lead I, and interpreted per lead. Its configurations are
  `configs/config_lsemvae_pretrain.yaml` and `config_lsemvae_finetune.yaml`, writing to
  their own output folders. Its Colab preparation is the CardioVAE tutorial's, cell for
  cell, which a test holds it to; another runs it on a synthetic DICOM cohort.
- **DICOM, read through PyKale's reader.** `ECGFileSource.from_dicom` and
  `ImageFileSource.from_dicom` take a folder holding one folder per subject, the
  folder's name identifying the subject. A DICOM ECG's leads and rate are read from its
  header, and every other subject's leads are found by name, so an export storing aVF
  before aVL cannot transpose them; a DICOM image is read as a viewer displays it, its
  rescale, window and `MONOCHROME1` inversion applied. Underneath: `get_dicom_paths`,
  `read_dicom` (PyKale's `read_dicom_phases`), `read_dicom_ecg` and `read_dicom_image`.
  `normalize_lead_name` reads DICOM's `Lead I (Einthoven)` as `I`. The `imaging` extra
  now installs scikit-image, which PyKale's reader imports, and pydicom 3.
- `prepdata.interleave_leads`, and `get_ecg_transform(interleave=True)`: the twelve
  leads interleaved sample by sample into one channel, as CardioVAE reads an ECG.
  `ECG.NUM_CHANNELS` is a key of the package schema, read by
  `get_ecg_transform_from_cfg`, so the CardioVAE example and the tutorial interleave
  the same way; the example's own copy of the step is gone.
- **The published CardioVAE weights load.** `CardioVAE.load_pykale_state_dict` reads a
  checkpoint saved from PyKale's `SignalImageVAE` or the reference implementation,
  through `remap_pykale_state_dict` and PyKale's own `remap_state_dict_keys`. Loaded
  into both, the same weights give identical encoder and decoder outputs.
- `MultimodalDataset.from_arrays`, a cohort straight from aligned tensors, as PyKale's
  `SignalImageDataset` takes them.
- **Multiclass endpoints, end to end.** `multiclass_roc_auc` (macro one-versus-rest, so
  a rare class weighs as much as a common one). `ClassificationTask` monitors it when
  `num_classes > 2`, and the trainer accumulates a row of class probabilities per
  subject. Both examples take `ENDPOINT.NUM_CLASSES` and `ENDPOINT.LABEL_COLUMN`, weight
  the classes from each training fold, and report class counts.
- **Robustness to missing leads.** `progressive_ablation` withholds modalities
  cumulatively through the presence mask. The LS-EMVAE example writes
  `robustness.csv` per lead group and fold (`ROBUSTNESS.ENABLED`, `ROBUSTNESS.ORDER`).
- `preserve_rng_state`, which runs report-only work without moving any random number
  generator that later training reads.
- `CardioVAE` takes `fusion`, `use_prior`, `num_groups` and `kernel_size`;
  `LSEMVAE` takes `kernel_size`; `Conv1dEncoder` and `Conv2dEncoder` take `padding`.
- Example configuration keys that were declared but never read now act:
  `ECG.SOURCE_SAMPLING_RATE` resamples each lead, `MODEL.KERNEL_SIZE` sets the
  convolutions, `MODEL.PRETRAINED` loads a checkpoint strictly and skips pretraining,
  `MODEL.NAME` is checked at start-up, and CardioVAE honours `FUSION.LATENT_METHOD`,
  `FUSION.USE_PRIOR` and `FUSION.NUM_GROUPS`.
- Modality ablation runs for a regression endpoint too, measured in R².
- An API reference built with Sphinx, one page per pipeline stage, and a Read the Docs
  configuration. The library's docstring examples now run as doctests.
- `py.typed`: the package's type hints are part of its interface.
- `CONTRIBUTING.md`, issue forms that refuse patient data, and a pull-request template.
- CI jobs that run both examples from the command line, build and install the
  distributions in a clean environment, test the source distribution on its own, and
  build the documentation with warnings as errors.
- Tests that a working pipeline scores clearly above chance and that an unlearnable
  endpoint scores about 0.5 — a score well above chance there would be a leak.
- Tests for every line of the library; CI fails below 98% coverage.

### Changed

- **Shared evaluation functions now come directly from PyKale.** Import classification
  metrics and `MetricError` from `kale.evaluate.classification_metrics`, regression
  metrics from `kale.evaluate.regression_metrics`, and `summarize_folds` from
  `kale.evaluate.cross_validation`. The duplicate KaleCardiac classification and
  regression modules, their one-line wrappers and the `summarize_folds` wrapper are
  removed. Existing imports from `kalecardiac.evaluate` must use those PyKale paths.
  `kalecardiac.evaluate` still provides prediction records, `flatten_fold_summary`,
  and adapters for `bootstrap_ci` and `pool_folds`. The former keeps its labels-first
  argument order, seed of 2026 and skipping of `ValueError` resamples; the latter
  accepts `SplitPredictions`. Results on valid inputs are unchanged to the bit. The
  PyKale functions change behavior for inputs on which these metrics were undefined:
  - a two-column multiclass score is scored as the binary area instead of failing in
    scikit-learn;
  - a binary label other than exactly 0 or 1, or a multiclass label that is not exactly
    a class index (`0.999999`, `0.5`, NaN), raises `MetricError` rather than being cast;
  - `bootstrap_ci` skips a resample whose metric is NaN or infinite, and raises
    `MetricError` when the metric is not finite on every subject, rather than returning
    a NaN interval; a one-class split now fails on the split itself, before any resample;
  - `pool_folds` raises when a fold's targets and scores differ in length;
  - some error messages are PyKale's wording ("sample" for "subject", "labels must be
    class indices 0 to K-1").

- **The examples run through `kalecardiac.auto`.** A runner keeps what it compares, the
  model it builds and its own reports; LS-EMVAE's robustness curve runs as the `report`
  `cross_validate` calls after each fold. Both preprocess with
  `prepdata.get_ecg_transform` rather than their own copies of its steps, and both
  command lines read their group flag, configuration and results table through
  `examples/cli.py`.
- Each fold fine-tunes a copy of the pretrained model, where each runner rebuilt the
  model and loaded the weights into it. Building drew random numbers that copying does
  not, so fold scores differ from earlier runs.
- LS-EMVAE's `lead_importance.csv` is `attribution.csv`, which CardioVAE now writes too.
  A model reading one modality reports no ablation -- withholding its only modality
  leaves it nothing to read -- as CardioVAE already did and LS-EMVAE's single-lead
  groups did not.
- `MultimodalVAE.latent_embedders` is annotated as taking any iterable of names, and
  `save_predictions` any mapping of metrics, as both always accepted.
- **Names follow PyKale.** Modules are named for their modality, functions that act
  start with a verb, and identifiers use American spelling, as in PyKale, scikit-learn
  and PyTorch. Among the renames:

  | Before | Now |
  | --- | --- |
  | `loaddata.ecg_access` | `loaddata.signal_access` |
  | `interpolate_missing`, `standardise`, `resample` | `interpolate_signal`, `normalize_signal`, `resample_signal` |
  | `crop_or_pad`, `scale_amplitude`, `clip_amplitude` | `crop_or_pad_signal`, `scale_signal`, `clip_signal` |
  | `as_signal`, `as_image`, `as_lead_array`, `as_image_array` | `check_signal`, `check_image`, `check_ecg_array`, `check_image_array` |
  | `build_ecg_pipeline`, `build_image_pipeline`, `lead_selector` | `get_ecg_transform`, `get_image_transform`, `get_lead_selector` |
  | `scale_image`, `standardise_image` | `scale_image_intensity`, `normalize_image` |
  | `canonical_lead(s)`, `lead_indices`, `lead_sources` | `normalize_lead_name(s)`, `get_lead_indices`, `get_lead_sources` |
  | `HoldOut`, `CrossValidation`, `Predefined`, `CohortSplitter` | `HoldOutSplit`, `CrossValidationSplit`, `PredefinedSplit`, `BaseSplitter` |
  | `composite_labels`, `expert_groups`, `reparameterise` | `get_stratification_labels`, `get_expert_groups`, `reparameterize` |
  | `summarise_folds`, `normalise_attribution` | `summarize_folds`, `normalize_attribution` |
  | `ecg_wave_segments`, `latent_embedding` | `delineate_ecg_waves`, `project_latents` |
  | `ECG.STANDARDISE` | `ECG.NORMALIZE` |
  | the arguments `standardise_leads`, `standardise`, `centre`, `centre_crop` | `normalize_leads`, `normalize`, `center`, `center_crop` |

  Code or a YAML file still using an old name fails with the new one in the message:
  importing an old name from the subpackage that exported it raises an `ImportError`
  naming its replacement, or the dependency's function that replaced it, and
  `ECG.STANDARDISE` is a renamed key to YACS.

- **What a dependency already provides is no longer duplicated.** Transforms are chained
  with `torchvision.transforms.Compose`, replacing `SignalPipeline`, `ImagePipeline` and
  `step_name`. Tasks call PyTorch's losses directly, replacing `binary_cross_entropy`,
  `cross_entropy` and `mean_squared_error`; class weights come from scikit-learn's
  `compute_class_weight`, replacing `class_weight_from_labels` and
  `positive_weight_from_labels`; `r2_score` and the regression metrics are
  scikit-learn's; freezing is `module.requires_grad_`, replacing `freeze`; and
  `train_test_split` and `subject_frame` are gone in favour of `HoldOutSplit` and pandas.
  The ablations rescore through `predict_split` rather than a loop of their own.
- `resample_signal` is SciPy's polyphase `resample_poly`, which filters before it
  decimates, rather than linear interpolation, which aliased on downsampling.
- `CardioVAE()` defaults to the published 60,000-sample signal, and its `kernel_size`
  sets the decoders as well as the encoders, as in the published model.
- `torchvision` and `scipy` are declared dependencies; both were already installed.
- Multiclass metrics accept logits and apply the softmax themselves, as the binary ones
  accept logits; a one-versus-rest ranking, unlike a binary one, depends on it.
- Metric and attribution functions are annotated as taking any array-like, which they
  always accepted.
- The command lines take `argv`, configure logging in `main()` rather than on import,
  and reject a `--leads` or `--arms` value with an empty name.
- The source distribution carries the tests, the examples, the tutorials and the
  documentation; the wheel still carries the library alone. A tutorial's downloaded
  dataset archive, like the data it unpacks, is kept out of both the distribution and git.
- The licence is declared as an SPDX expression, which setuptools requires from 2027.

### Fixed

- Pretraining wrote a full Lightning checkpoint — the model and its optimizer state,
  2.4 GB for the tutorial's CardioVAE — after every epoch, which nothing read: the
  weights are saved as `pretrained.pt`. A run with nothing to monitor now keeps no
  checkpoint, and the tutorial's ten pretraining epochs take 34 s on one GPU rather
  than 342 s.
- **Cross-validation folds were not stratified.** With `group_by` set,
  `CrossValidationSplit` used scikit-learn's `StratifiedGroupKFold(shuffle=True)`, which
  shuffles each group's class counts without shuffling which group they belong to: the
  folds it balanced were not the folds it returned. On a table of one row per subject,
  as both examples split, a two-fold split could hold one positive in one test fold and
  eleven in the other. The groups are now numbered in a seeded random order and split
  unshuffled, and every fold holds the cohort's share of each class; every example's
  folds change.
- `Conv1dDecoder` and `Conv2dDecoder` with `kernel_size=3` built, then failed on the
  first forward pass with a negative output padding. Every kernel width now multiplies
  the length by the stride exactly.
- `CardioVAE`'s decoders used kernel 4 where the published model uses kernel 3, so the
  published weights could not load and the documented parameter-for-parameter match
  with PyKale's `SignalImageVAE` did not hold for the decoders.
- The CardioVAE example flattened the twelve leads end to end and normalized the
  flattened strip as one signal. The reference implementation normalizes each lead and
  interleaves them sample by sample, which is what the published checkpoint expects; the
  example now does the same, and cropping shortens every lead rather than dropping some.
- The CardioVAE example declared one lead unless `ECG.LEADS` was set, so a real
  twelve-lead cohort failed on its first recording; it now declares the twelve leads,
  in the order MIMIC-IV-ECG stores them -- aVF before aVL.
- A multiclass model was monitored on class 1 against the rest, so checkpoint
  selection favoured telling that one class apart.
- `modality_ablation` scored a multiclass model on the first column of its output only.
- Turning a report on or off (`OUTPUT.INTERPRET`, `ROBUSTNESS.ENABLED`) changed the
  models trained after it: every DataLoader pass draws from PyTorch's generator.
- An unknown or empty `--leads` group failed only after pretraining had run.
- `ECG.SOURCE_SAMPLING_RATE`, `MODEL.KERNEL_SIZE` and `MODEL.PRETRAINED` were accepted
  and silently ignored; `ECG.MODALITY` is removed, having never been read.

## [0.1.0a0]

Initial version: the verb-oriented pipeline (`loaddata`, `prepdata`, `model`,
`pipeline`, `evaluate`, `interpret`), one multimodal VAE covering CardioVAE, LS-EMVAE
and their published baselines, one trainer for pretraining and fine-tuning, and the two
examples on synthetic cohorts.
