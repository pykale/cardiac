# Fusion and evaluation migration contract

This records KaleCardiac's public behavior before the shared implementations move to PyKale. The names below remain importable from KaleCardiac during migration. A PyKale replacement should pass the existing tests plus the per-subject mask test without changing scientific results. This is a record of current behavior, not a new API.

## Public API inventory

`kalecardiac.model.layers.gaussian` defines the names below; they are also exported by `kalecardiac.model.layers`. “Core candidate” means reusable numerical behavior. The Cardiac import path needs a compatibility re-export when the implementation moves.

| Name | Disposition | Contract to preserve |
| --- | --- | --- |
| `GaussianPosterior` | Core candidate | Frozen mean/log-variance pair with `.variance`, `.stddev`, `.sample()`, `.detach()` and `.to()` |
| `product_of_experts` | Core candidate | Masked Gaussian precision product, returning a `GaussianPosterior` |
| `mixture_of_experts`, `mean_of_experts` | Core candidates | Weighted parameter average and its equal-weight alias |
| `wasserstein_barycenter` | Core candidate | Weighted mean of means and standard deviations |
| `hierarchical_experts`, `expert_groups` | Core candidates | Contiguous, ordered groups; product within each and average across present groups |
| `prior_expert`, `reparameterise` | Reconcile with core | Standard-normal pair and differentiable sampling; check existing PyKale equivalents before moving |
| `EXPERT_FUSIONS`, `build_expert_fusion` | Cardiac compatibility surface | The `poe`, `moe`, `hime`, `mean`, `barycenter` names and construction-time option checking |
| `GaussianHead` | Retain until encoder ownership is decided | Maps a feature vector to mean and log variance; used by Cardiac encoders |
| `ABSENT_LOG_VAR` | Compatibility constant | Finite absent-expert log variance, currently `20.0` |

`kalecardiac.evaluate` exports all names below. Core candidates are subject-independent calculations. The prediction adapter remains local while it consumes `SubjectBatch` and Cardiac output schemas.

| Name | Disposition | Contract to preserve |
| --- | --- | --- |
| `MetricError` | Core candidate | Public `ValueError` subtype for undefined metrics |
| `roc_auc`, `sensitivity_specificity`, `binary_metrics` | Reconcile with core | Binary ranking and threshold metrics; higher scores mean more positive |
| `multiclass_roc_auc`, `multiclass_metrics` | Core candidates | Macro one-versus-rest area, classification summary |
| `r2_score`, `regression_metrics` | Reconcile with core | Higher R² is better; regression errors use target units |
| `mean_roc_curve`, `summarise_folds`, `pool_folds`, `bootstrap_ci` | Core candidates | Fold aggregation and subject bootstrap with the result schemas below |
| `SplitPredictions`, `predict_split`, `save_predictions` | Retain in Cardiac | `SubjectBatch` adapter, subject IDs and existing CSV/JSON schema |

## Gaussian behavior

- The low-level fusion functions take `mean` and `log_var` of shape `(M, B, D)`, with expert, subject and latent axes in that order. The optional presence mask has shape  `(M, B)`; one or `True` means present. Outputs have shape `(B, D)` and retain the
  input tensor's device and dtype. Invalid shapes and an empty expert axis raise   `ValueError`.
- A mask may differ for every subject in one batch. An absent expert's mean has zero gradient for that subject. Product fusion gives it negligible finite precision; averaging methods give it zero share. `hime` keeps the original expert order when  forming contiguous groups. `expert_groups(13, 4)` returns sizes `[4, 3, 3, 3]`.
- The low-level functions do **not** insert a prior. `MultimodalVAE(use_prior=True)` prepends a standard-normal expert for `poe` and `hime` only. It is always present,including when all observed modalities are missing. `moe`, `mean` and  `barycenter`  ignore `use_prior`; with no present expert they fall back to an equal average of the  supplied experts. A low-level masked PoE with no prior gives a finite, highly  uncertain result; it should not be mistaken for the model-level prior behavior.
- `moe` averages Gaussian **parameters** and returns one Gaussian, rather than a  mixture distribution. `mean` is its equal-weight case. Weights for `moe` and  `barycenter` have shape `(M,)` or `(M, B)` and are normalized after masking.
- The reference-model departures documented in [models.md](models.md) are parity  requirements: keep the prior in the 13-expert grouping, calculate KL against the  posterior actually sampled, and allow explicit sampling even during evaluation.

## Evaluation behavior

- Metric arrays are positional: labels and scores for a subject must occupy the same  row. The array-based metrics do not accept or align subject IDs. `predict_split`
  preserves loader order in `SplitPredictions.subject_id`; `rows()` writes one row per
  subject with `split`, optional `fold`, a scalar `score` or `score_0`… columns, and
  target values. An indexed core API must validate alignment before discarding IDs.
- Binary ROC-AUC ranks raw scores. Threshold metrics accept probabilities or logits;
  logits are passed through a sigmoid and the threshold is on probability. A one-class
  binary split raises `MetricError`. Multiclass inputs are `(N, K)` probabilities or
  logits; logits receive a softmax. `multiclass_roc_auc` requires every class, while
  `multiclass_metrics` omits `roc_auc_ovr` if a class is absent.
- `binary_metrics` returns `n`, `n_positive`, `positive_rate`, `threshold`,
  `roc_auc`, `average_precision`, `accuracy`, `balanced_accuracy`, `sensitivity`,
  `specificity`, `f1`, `mcc` and a `confusion` mapping with `tn`, `fp`, `fn`, `tp`.
  `multiclass_metrics` returns `n`, `accuracy`, `balanced_accuracy`, `f1_macro`,
  `confusion`, `classes` and, when defined, `roc_auc_ovr`.
  `regression_metrics` returns `n`, `rmse`, `mae`, signed `bias`, `unit` and `r2`;
  a constant target gives `r2: null`.
- `mean_roc_curve` gives `fpr`, `tpr_mean`, `tpr_std`, `tpr_runs`, `auc_mean`,
  `auc_std` and `auc_runs` on one interpolated grid. `summarise_folds` reports
  `n_folds` and, per numeric metric, `mean`, population `std` and `values`; it weights
  folds equally. `pool_folds` concatenates predictions and weights subjects equally.
- `bootstrap_ci` draws subjects with replacement using the given seed. It skips
  resamples for which the metric raises `MetricError` or `ValueError`, returns `point`,
  `low`, `high`, `confidence`, `n_used`, and raises `MetricError` if no draw is usable.
  `n_used` can be below the requested `n_resamples`; it must stay visible in any
  adapter. The interval reflects resampling of this split, not model-refit uncertainty.

The baseline suites are `tests/model/layers/test_gaussian.py` and `tests/evaluate/`.
The model-level prior and scientific departures are covered in `tests/model/embed/` and
[models.md](models.md). Any intentional change during migration needs a separate,
reviewed specification and a test showing the new behavior.
