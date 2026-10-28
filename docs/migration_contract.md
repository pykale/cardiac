# Fusion and evaluation migration contract

This records KaleCardiac's public behavior before shared implementations move to PyKale. The names below remain importable from KaleCardiac during migration. For supported binary masks, matching floating dtypes and finite nonnegative weights, a PyKale replacement should pass the existing tests and the per-subject mask test without changing scientific results. Invalid inputs may now raise stricter errors, as described below.

## Core-backed adoption in progress

The Gaussian averaging, barycenter, grouping and hierarchical functions now delegate their calculations to the proposed PyKale APIs while keeping KaleCardiac's import paths and `GaussianPosterior` return type. `product_of_experts` already delegates its calculation to PyKale's `ProductOfExperts`. KaleCardiac retains `EXPERT_FUSIONS`, `build_expert_fusion`, prior insertion in its models, and its existing validation messages for invalid weight shapes and negative weights.

The inventory below records the original migration candidates. It does not mean that `GaussianPosterior` or KaleCardiac's `product_of_experts` wrapper will move to core; PyKale returns tensor pairs, while Cardiac keeps those application-facing APIs.

This adapter work is being tested against the PyKale development branch behind [PR #609](https://github.com/pykale/pykale/pull/609) and [PR #613](https://github.com/pykale/pykale/pull/613). KaleCardiac's minimum `pykale` version remains unchanged until those APIs are merged and released; before merging the adoption PR, install the released version in a clean environment, update the dependency floor, and rerun the full test suite and synthetic examples. PyKale rejects non-binary masks, mismatched floating dtypes and non-finite weights, so Cardiac adoption tests make those input changes explicit.

## Public API inventory

### Gaussian fusion

`kalecardiac.model.layers.gaussian` defines these names. They are also exported from `kalecardiac.model.layers`.

**Original migration candidates (for reference)**

- `GaussianPosterior`: frozen mean and log-variance pair with `.variance`, `.stddev`, `.sample()`, `.detach()` and `.to()`.
- `product_of_experts`: masked Gaussian precision product.
- `mixture_of_experts` and `mean_of_experts`: weighted parameter average and its equal-weight form.
- `wasserstein_barycenter`: weighted mean of means and standard deviations.
- `hierarchical_experts` and `expert_groups`: ordered, contiguous groups; product within groups and average across present groups.

**Reconcile with existing PyKale APIs**

- `prior_expert` and `reparameterise`: standard-normal expert and differentiable sampling. Check existing core equivalents before moving them.

**Keep in KaleCardiac during migration**

- `EXPERT_FUSIONS` and `build_expert_fusion`: keep the `poe`, `moe`, `hime`, `mean` and `barycenter` names and construction-time option checking.
- `GaussianHead`: keep with the Cardiac encoders until encoder ownership is decided.
- `ABSENT_LOG_VAR`: preserve its current value of `20.0` for compatibility.

When a core candidate moves, keep its Cardiac import path through a compatibility adapter so that the `GaussianPosterior` return type remains available.

### Evaluation

`kalecardiac.evaluate` exports these names.

**Candidates for PyKale core**

- `MetricError`: public `ValueError` subtype for undefined metrics.
- `multiclass_roc_auc` and `multiclass_metrics`: macro one-versus-rest area and multiclass summary.
- `mean_roc_curve`, `summarise_folds`, `pool_folds` and `bootstrap_ci`: fold aggregation and subject bootstrap.

**Reconcile with existing PyKale APIs**

- `roc_auc`, `sensitivity_specificity` and `binary_metrics`: binary ranking and threshold metrics. Higher scores mean more positive.
- `r2_score` and `regression_metrics`: higher R² is better; errors use the target's units.

**Keep in KaleCardiac during migration**

- `SplitPredictions`, `predict_split` and `save_predictions`: they consume `SubjectBatch` and preserve Cardiac's subject IDs and CSV/JSON schema.

## Gaussian behavior

### Inputs and masks

- `mean` and `log_var` have shape `(M, B, D)`: expert, subject, latent dimension.
- The optional presence mask has shape `(M, B)`. One or `True` means present.
- The output has shape `(B, D)` and keeps the input device and dtype.
- Invalid shapes and an empty expert axis raise `ValueError`.
- Different subjects in one batch may have different missing experts. An absent expert's mean has zero gradient for that subject. Product fusion gives the absent expert negligible finite precision; averaging methods give it zero share.

`hime` keeps the original expert order when forming contiguous groups. `expert_groups(13, 4)` returns groups of sizes `[4, 3, 3, 3]`.

### Prior and method behavior

The low-level fusion functions do **not** insert a prior. `MultimodalVAE(use_prior=True)` prepends a standard-normal expert for `poe` and `hime` only. The prior remains present when all observed modalities are missing.

`moe`, `mean` and `barycenter` ignore `use_prior`. With no present expert, they fall back to an equal average of the supplied experts. A low-level masked PoE without a prior gives a finite, highly uncertain result; this differs from the model-level prior behavior.

`moe` averages Gaussian **parameters** and returns one Gaussian. `mean` is its equal-weight case. Weights for `moe` and `barycenter` have shape `(M,)` or `(M, B)` and are normalized after masking.

### Scientific parity

The reference-model departures in [models.md](models.md) must be preserved:

- Keep the prior in the 13-expert grouping.
- Calculate KL against the posterior actually sampled.
- Allow explicit sampling during evaluation.

## Evaluation behavior

### Alignment and score interpretation

Metric arrays are positional: each subject's labels and scores must occupy the same row. The array-based metrics do not accept or align subject IDs. `predict_split` preserves loader order in `SplitPredictions.subject_id`. `rows()` writes one row per subject with `split`, optional `fold`, a scalar `score` or `score_0`… columns, and target values. An indexed core API must validate alignment before discarding IDs.

- Binary ROC-AUC ranks raw scores. Threshold metrics accept probabilities or logits; logits pass through a sigmoid, and the threshold is on probability. A one-class binary split raises `MetricError`.
- Multiclass scores have shape `(N, K)`. Logits pass through a softmax. `multiclass_roc_auc` requires every class. `multiclass_metrics` omits `roc_auc_ovr` when a class is absent.

### Result schemas

- `binary_metrics`: `n`, `n_positive`, `positive_rate`, `threshold`, `roc_auc`, `average_precision`, `accuracy`, `balanced_accuracy`, `sensitivity`, `specificity`, `f1`, `mcc`, and `confusion` with `tn`, `fp`, `fn`, `tp`.
- `multiclass_metrics`: `n`, `accuracy`, `balanced_accuracy`, `f1_macro`, `confusion`, `classes`, and `roc_auc_ovr` when defined.
- `regression_metrics`: `n`, `rmse`, `mae`, signed `bias`, `unit`, and `r2`. A constant target gives `r2: null`.
- `mean_roc_curve`: `fpr`, `tpr_mean`, `tpr_std`, `tpr_runs`, `auc_mean`, `auc_std`, and `auc_runs` on one interpolated grid.

### Folds and bootstrap

`summarise_folds` reports `n_folds` and, for each numeric metric, `mean`, population `std`, and `values`. It weights folds equally. `pool_folds` concatenates predictions and weights subjects equally.

`bootstrap_ci` draws subjects with replacement using the given seed. It skips resamples for which the metric raises `MetricError` or `ValueError` and returns `point`, `low`, `high`, `confidence`, and `n_used`. It raises `MetricError` if no draw is usable. `n_used` can be below the requested `n_resamples` and must stay visible in any adapter. The interval reflects resampling of this split, not model-refit uncertainty.

## Verification baseline

The baseline suites are `tests/model/layers/test_gaussian.py` and `tests/evaluate/`. Model-level prior behavior and scientific departures are covered in `tests/model/embed/` and [models.md](models.md). Any intentional behavior change during migration needs a separate, reviewed specification and a test showing the new behavior.
