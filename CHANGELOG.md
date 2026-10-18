# Changelog

Notable changes to KaleCardiac. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) from 1.0 onwards.

## [Unreleased]

### Added

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
  `ECG.SOURCE_SAMPLING_RATE` resamples each lead (before flattening, in CardioVAE),
  `MODEL.KERNEL_SIZE` sets the encoders, `MODEL.PRETRAINED` loads a checkpoint strictly
  and skips pretraining, `MODEL.NAME` is checked at start-up, and CardioVAE honours
  `FUSION.LATENT_METHOD`, `FUSION.USE_PRIOR` and `FUSION.NUM_GROUPS`.
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
- Tests for every line of the library, including the trainer's test loop and its
  recovery from a one-class validation split, which had never run; CI fails below 98%.

### Changed

- Multiclass metrics accept logits and apply the softmax themselves, as the binary ones
  accept logits; a one-versus-rest ranking, unlike a binary one, depends on it.
- Metric and attribution functions are annotated as taking any array-like, which they
  always accepted.
- The command lines take `argv`, configure logging in `main()` rather than on import,
  and reject a `--leads` or `--arms` value with an empty name.
- The source distribution carries the tests, the examples they compose and the
  documentation; the wheel still carries the library alone.
- The licence is declared as an SPDX expression, which setuptools requires from 2027.
- The examples derive a binary class weight with `positive_weight_from_labels` instead
  of an inline formula. The two differ only for a training fold with no positive
  subject, where the weight is now 1.

### Fixed

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
