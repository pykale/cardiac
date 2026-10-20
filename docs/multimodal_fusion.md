# Multimodal fusion

Cardiac modalities are **incommensurable**. A ten-second twelve-lead recording is 60,000 numbers; a chest radiograph is 50,176 pixels; a clinical record is fifty covariates. There is no useful way to concatenate the raw inputs, so every model here encodes each modality to a fixed-width representation first and fuses in that space.

KaleCardiac fuses at two levels:

- **Latent fusion** combines one Gaussian distribution per modality. It is used by `MultimodalVAE`, configured with `FUSION.LATENT_METHOD`, and implemented in `kalecardiac.model.layers.gaussian`.
- **Embedding fusion** combines one vector per modality. It is used by `MultimodalPredictor`, configured with `FUSION.METHOD`, and implemented in `kalecardiac.model.embed.multimodal_fusion`.

They are independent: either changes without the other.

---

## Latent fusion: combining distributions

Each modality's encoder proposes a diagonal Gaussian over a shared latent space. The question every multimodal VAE answers differently is: **given one Gaussian per modality, what is the joint posterior?**

All five answers take the same shapes — `mean` and `log_var` of `(M, B, D)` for `M` experts, and an optional `(M, B)` mask — so swapping between them is a string.

### Product of experts (`poe`)

Multiply the Gaussians. The product of diagonal Gaussians is itself Gaussian, with precision the sum of the experts' precisions and mean their precision-weighted average.

```text
joint precision = sum(expert precisions)
joint mean      = sum(expert mean × expert precision) / joint precision
```

**What follows:** the joint is *never less certain than any expert*. Adding a modality can only narrow the posterior, and a confident modality dominates an uncertain one.

**When it fits:** the modalities are views of one underlying state — which for the leads of a single heartbeat, or an ECG and a radiograph of the same chest, is literally true.

**Missing modalities:** an absent expert is given a log-variance of 20 (a variance of about 5×10⁸), so its precision is negligible and it drops out of the product without any special case. That is what lets a model trained on twelve leads run on six. `tests/model/layers/test_gaussian.py` asserts that masking an expert gives the same result as removing it.

**The prior expert:** `MultimodalVAE` includes a standard-normal expert by default for `poe` and `hime`; the low-level fusion functions do not insert one. It keeps a single-modality posterior proper, and gives a subject with *no* present modality a defined result instead of a zero-divided-by-zero one — the convention since Wu and Goodman (2018).

### Mixture of experts (`moe`) and mean (`mean`)

Average the *parameters*:

```text
joint mean         = sum(weight × expert mean)
joint log-variance = sum(weight × expert log-variance)
```

Averaging the parameters, not mixing the distributions: a true mixture of Gaussians is not Gaussian, and the moment-matched average is what keeps the KL term in closed form.

**What follows:** no expert can dominate. A confident modality and an uninformative one contribute equally, which keeps a weak modality from being ignored — and keeps a noisy one from being discounted.

**When it fits:** the modalities are of comparable reliability, or a product's tendency to be dominated by one over-confident encoder is the failure being avoided.

`mean_of_experts` is the equal-weight case, named separately because it is the plainest possible fusion and therefore the baseline an ablation compares against.

### Hierarchical modality experts (`hime`)

A product **within** contiguous groups of experts, then a mixture **across** the groups. LS-EMVAE's mechanism.

For twelve leads with a prior and four groups, the groups are:

| Group | Experts | Within the group |
| --- | --- | --- |
| 1 | Prior, I, II, III | Product |
| 2 | aVR, aVL, aVF | Product |
| 3 | V1, V2, V3 | Product |
| 4 | V4, V5, V6 | Product |

The four resulting group posteriors are averaged.

**What follows:** the joint gains the sharpness of a product without letting one over-confident group determine the latent on its own. It sits between the two in certainty, which `tests/model/layers/test_gaussian.py` asserts directly:

```python
product_variance <= hierarchical_variance <= mixture_variance
```

Two degenerate cases are worth knowing, and both are tested: one group reduces to a product, and groups of one reduce to a mixture.

**Grouping is contiguous over the expert axis**, so the order experts are stacked in decides which reinforce which. For an ECG in conventional lead order that puts the limb leads in the early groups and the precordial leads in the later ones, which is anatomically sensible. A different order is a different grouping, and therefore a modelling choice worth stating. A group whose experts are all absent takes no share of the mixture.

> **A correction to the reference implementation.** Its grouping computed `group_size = M // num_groups` and took `num_groups` slices of that width, silently discarding the last `M % num_groups` experts. With twelve leads plus the prior and four groups that dropped the prior entirely. `expert_groups` distributes the remainder instead, so the group sizes are `[4, 3, 3, 3]` and every expert is used. See [models.md](models.md#documented-departures).

### Wasserstein barycenter (`barycenter`)

Average the means and the **standard deviations** — not the variances, and not the precisions:

```text
joint mean               = sum(weight × expert mean)
joint standard deviation = sum(weight × expert standard deviation)
```

For diagonal Gaussians this is the 2-Wasserstein barycenter. It sits between the other two: unlike a product it cannot become more certain than its experts, and unlike a parameter-averaged mixture it averages in the scale a distance is measured in. Two experts with variances 1 and 9 give a barycenter variance of 4, not the variance mean of 5 — which the tests check directly.

### Choosing one

- Use `poe` when modalities are views of one state and missing modalities must degrade gracefully.
- Use `moe` when one encoder's over-confidence swamps the others.
- Use `hime` when related modalities should reinforce one another within groups.
- Use `barycenter` when the distribution's geometry matters more than a parameter average.

---

## Embedding fusion: combining vectors

Once a representation exists, a downstream model fuses *vectors*:

```text
embedders  →  optional projection  →  fusion  →  head
```

- **`concat`** joins vectors into a width of `Σ dᵢ`. It makes no assumption that modalities are interchangeable; the head learns their relative weight.
- **`mean`** produces width `d` by averaging interchangeable modality vectors. Its constant width lets one head serve different modality counts.
- **`attention`** also produces width `d` from interchangeable modality vectors. It reports a weight for each modality and subject.

`concat` is what both cardiac studies fine-tune with — twelve leads of 256 dimensions concatenated into one classifier. `mean` and `attention` require equal widths, so set `FUSION.FUSION_DIM` to project first.

**`FUSION.FUSION_DIM = 0` applies no projection**, which is what concatenating pretrained encoder means directly amounts to and what the reference implementations do. Setting an integer adds a learned projection per modality, which is what lets embedders of differing widths interoperate.

**Absent modalities** are replaced by a *learned placeholder* rather than by zeros, because an all-zero vector is indistinguishable from a genuine one. For `mean` and `attention` the modality is excluded from the weighting entirely.

**`attention` returns its weights** on the output, so "which lead did this subject's prediction rest on" is answerable per subject. That is the per-subject counterpart of `modality_ablation`, which answers the same question per cohort.

---

## Modality dropout

`FUSION.MODALITY_DROPOUT` randomly marks present modalities absent during training. At least one modality always survives, so no subject is left with nothing to predict from.

This is how a cardiac model is taught to survive its own deployment: a reduced-lead recorder, a subject with no recent radiograph, an electrode that came off. It applies in training only — evaluation is deterministic.

---

## Where fusion does *not* happen

**Early fusion, on raw inputs, is not available and is not an oversight.** A 60,000-sample recording and a 224×224 image have no shared axis to concatenate along. Encoding each to a fixed-width representation first is what makes them combinable at all.

**Fusion does not decide what is missing.** The presence mask is carried from `loaddata` through the batch and into the fusion unchanged. A fusion reads it and never writes it, which keeps "what this subject actually has" a property of the cohort rather than something a model inferred from zeros.

## See also

- [models.md](models.md) — which fusion each published model uses
- [architecture.md](architecture.md) — where the fusion stage sits in the pipeline
- `tests/model/layers/test_gaussian.py` — every property above, asserted
- [migration_contract.md](migration_contract.md) — public fusion and evaluation behavior to preserve during the PyKale migration
