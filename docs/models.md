# Models

What KaleCardiac implements, what each model contributes, and where an implementation
departs from the work it was refactored from.

Every generative model here is a configuration of one class. Read
[`MultimodalVAE`](#multimodalvae-the-generic-model) first; the named models are short.

## Common interfaces

Three contracts, and everything else composes them.

### Backbone

Transforms a modality into a flat feature vector. Declares `out_dim`.

```python
backbone(x)  # (batch, channels, ...) -> (batch, out_dim)
```

| Backbone | Input | `out_dim` depends on length? | Use when |
| --- | --- | --- | --- |
| `Conv1dEncoder` | `(B, C, L)` signal | Yes — it flattens | The recordings are a fixed length, which after preprocessing they are |
| `ResidualConv1dEncoder` | `(B, C, L)` signal | No — it pools | A deeper receptive field is wanted, or the length varies |
| `TransformerSignalEncoder` | `(B, C, L)` signal | No — it pools over patches | Long-range structure across beats matters |
| `Conv2dEncoder` | `(B, C, H, W)` image | Yes — it flattens | Any 2-D cardiac modality |

Selected by name: `build_signal_encoder("conv" | "residual" | "transformer", ...)`, or
`ECG.ENCODER` in a configuration.

### Variational encoder

Any backbone plus a Gaussian head. Declares `latent_dim`.

```python
encoder = VariationalEncoder(backbone, latent_dim=256)
posterior = encoder(x)  # GaussianPosterior(mean, log_var), each (B, latent_dim)
```

### Embedder

One modality in, one vector per subject out. Declares `out_dim`.

```python
LatentEmbedder(pretrained_encoder, frozen=True)  # posterior mean; the transfer bridge
FeatureEmbedder(backbone, out_dim=128)  # trained from scratch
```

---

## `MultimodalVAE`: the generic model

One encoder per modality proposes a Gaussian over a shared latent space; a latent
fusion combines them into a joint posterior; a sample from it is decoded back into every
modality.

```python
MultimodalVAE(
    encoders={"ecg": VariationalEncoder(...), "cxr": VariationalEncoder(...)},
    decoders={"ecg": Conv1dDecoder(...), "cxr": Conv2dDecoder(...)},
    latent_dim=256,
    fusion="poe",
)
```

**Inputs** — `forward(modalities, present=None, unimodal_streams=False)` where
`modalities` maps a name to a tensor and `present` maps a name to a `(B,)` boolean.

**Output** — a `VAEOutput` carrying:

| Field | What it is |
| --- | --- |
| `joint` | The pass that saw every present modality: its posterior, its latent sample, and one reconstruction per decoded modality |
| `unimodal` | One pass per modality on its own. Empty unless requested |
| `modality_posteriors` | Each encoder's own proposal, before fusion |
| `present` | The presence mask, as the batch carried it |

**Two arguments, five published models:**

| `fusion` | decoders | Model |
| --- | --- | --- |
| `poe` | per modality | MVAE (Wu and Goodman, 2018); **CardioVAE** |
| `mean` | per modality | the parameter-averaging MMVAE+ variant |
| `barycenter` | per modality | a Wasserstein-barycenter multimodal VAE |
| `hime` | per modality | a grouped mixture-of-products VAE |
| `hime` | shared | **LS-EMVAE** |

A shared decoder is expressed by passing the *same module object* under several keys:
`nn.ModuleDict` registers the parameters once, so the optimiser sees one decoder and
every modality's gradient reaches it.

**Encode-only modalities.** A modality with an encoder but no decoder takes part in the
posterior and is not reconstructed — which is what a clinical vector alongside a
recording wants.

**Streams.** Trained on the joint posterior alone, an encoder is free to depend on its
siblings and becomes useless when they are absent. `unimodal_streams=True` adds a pass
through each modality on its own, so each encoder must also explain its own modality
unaided.

---

## `CardioVAE`

> Suvon et al., *Multimodal Variational Autoencoder for Low-cost Cardiac Hemodynamics
> Instability Detection*, MICCAI 2024.
> [arXiv:2403.13658](https://arxiv.org/abs/2403.13658) ·
> [reference implementation](https://github.com/Shef-AIRE/AI4Cardiothoracic-CardioVAE)

**The contribution is clinical rather than architectural.** The two cheapest and most
widely available cardiac investigations — a chest radiograph and a resting ECG — carry
enough shared signal that a representation learned from large *unlabelled* collections
of them transfers to a small labelled cohort with an invasively measured endpoint. The
architecture carrying that idea is a product-of-experts multimodal VAE trained with a
tri-stream objective.

```python
CardioVAE(
    signal_channels=1,
    signal_length=60000,  # twelve leads flattened end to end
    image_channels=1,
    image_size=(224, 224),
    latent_dim=256,
)
```

| Component | Where it is |
| --- | --- |
| Signal encoder/decoder | `Conv1dEncoder` + `VariationalEncoder`, `Conv1dDecoder` |
| Image encoder/decoder | `Conv2dEncoder` + `VariationalEncoder`, `Conv2dDecoder` |
| Fusion | `product_of_experts`, with the standard-normal prior expert |
| Objective | `ReconstructionTask(unimodal_streams=True, weights=..., scale_factor=...)` |
| Transfer | `LatentEmbedder` + `MultimodalPredictor` + `ClassificationTask` |

**Why the reconstruction weights.** A sum of squares over 60,000 ECG samples and one
over 50,176 pixels are not comparable quantities. The paper weights the ECG ten times
the radiograph; `OBJECTIVE.RECONSTRUCTION_WEIGHTS` and `OBJECTIVE.SCALE_FACTOR` are the
two controls, and KL annealing is the third.

**Relationship to PyKale.** PyKale carries an earlier fixed-shape form of this model as
`kale.embed.multimodal_encoder.SignalImageVAE`, with
`kale.pipeline.multimodal_trainer.SignalImageTriStreamVAETrainer` for its objective.
`CardioVAE` built with the default arguments is the same architecture parameter for
parameter, and differs in three ways that follow from being a library component: the
signal length, image size and channel widths are arguments; the modalities are named,
so a third is a dictionary entry; and missing modalities are carried per subject by a
mask rather than by passing `None` for a whole batch.

### Documented departure

**Sampling no longer depends on the module's training flag.** The reference
implementation's `reparametrize` returns a sample while training and the mean while
evaluating. Here sampling is always sampling, and a caller who wants the mean asks for
`posterior.mean`.

*Why:* whether to sample is a property of what the caller is doing — an ELBO needs a
sample, a downstream embedding wants a deterministic vector — not of a module flag.
Tying it to `training` silently changes the objective the moment a validation pass runs,
so a validation ELBO and a training ELBO are not the same quantity. `LatentEmbedder`
takes the mean explicitly, which is the behaviour the original got by accident.

---

## `LSEMVAE`

> Suvon et al., *Multimodal Latent Fusion of ECG Leads for Early Assessment of
> Pulmonary Hypertension*. [arXiv:2503.13470](https://arxiv.org/abs/2503.13470) ·
> [reference implementation](https://github.com/Shef-AIRE/LS-EMVAE)

**The contribution is a reframing.** A twelve-lead ECG is usually treated as one
twelve-channel signal, which asserts that the leads are commensurable channels of a
single view. They are not: each lead projects the heart's electrical activity onto a
different axis, so they are better modelled as *different views of the same underlying
state* — which is precisely what a multimodal VAE is for.

```python
LSEMVAE(
    leads=STANDARD_12_LEAD,
    length=5000,
    latent_dim=256,
    fusion="hime",
    num_groups=4,
    shared_decoder=True,
)
```

Three pieces carry it:

| Piece | Where it is | What it buys |
| --- | --- | --- |
| **Lead-specific encoders** | `lead_sources()` + one `VariationalEncoder` per lead | An encoder that keeps working when the other leads are absent |
| **Hierarchical expert fusion (HiME)** | `hierarchical_experts` | Leads seeing related territory reinforce one another, without one group determining the latent alone |
| **Shared decoder** | the same module under every key | Forces the latent to carry lead-agnostic cardiac state, rather than letting each lead hide information in a private decoder |
| **Latent alignment** | `latent_alignment_loss`, via `OBJECTIVE.ALIGNMENT_WEIGHT` | Every lead's representation readable in the same coordinates, which is what makes fine-tuning on a lead *subset* work |

The published ablations are configuration changes:

```bash
FUSION.LATENT_METHOD poe          # no mixture step
FUSION.LATENT_METHOD moe          # no product step
OBJECTIVE.ALIGNMENT_WEIGHT 0.0    # no alignment
MODEL.SHARED_DECODER False        # the grouped-fusion baseline
```

**Lead order is a modelling choice.** Grouping is contiguous over the expert axis, so
the order leads are stacked in decides which reinforce which. The conventional order
puts the limb leads in the early groups and the precordial leads in the later ones,
which is anatomically sensible; a different order is a different grouping.

### Documented departures

Two, both deliberate, both verified by tests.

**1. The prior expert is no longer silently discarded.**

The reference grouping computed `group_size = num_experts // num_groups` and then took
`num_groups` slices of that width:

```python
group_size = num_experts // num_groups  # 13 // 4 == 3
for i in range(num_groups):
    subset = mus[i * group_size : (i + 1) * group_size]  # covers experts 0..11
```

With twelve leads plus the prior and four groups, that is four groups of three covering
experts 0 to 11 — **the thirteenth expert, the prior, was never used**. So the
`use_prior` setting had no effect in the twelve-lead configuration, and a subject with
no present lead had an undefined posterior.

`expert_groups` distributes the remainder instead, so with thirteen experts and four
groups the sizes are `[4, 3, 3, 3]` and every expert is used. `use_prior=False` fuses
the leads alone, which is what the original effectively did.
*Tested in* `tests/model/layers/test_gaussian.py::TestExpertGroups`.

**2. The KL term is taken against the posterior that was actually sampled from.**

The reference training loop drew `z` from the HiME posterior but evaluated the KL
divergence against the *arithmetic mean* of the per-lead posteriors:

```python
qz_x_list, px_z_list, latent_z_list = model(lead_inputs)  # z from HiME
mu_joint = torch.mean(mus, dim=0)  # KL against the mean
loss = elbo_loss(px_z_list, lead_inputs, mu_joint, logvar_joint, ...)
```

So the objective regularised a distribution the model never sampled, and the sampled one
was left unconstrained.

Here one posterior serves both. A reader wanting the original's KL term exactly can set
`fusion="mean"`, which makes the fused posterior the arithmetic mean so that sampling
and regularisation agree by construction.

---

## Reusable baselines

These are library components rather than named models, because each is a configuration
of something more general:

| Baseline in the source studies | Here |
| --- | --- |
| MVAE (product of experts over leads) | `MultimodalVAE(fusion="poe")` |
| MoPoE (grouped product, then mixture) | `MultimodalVAE(fusion="hime", shared_decoder=False)` |
| MMVAE+ (parameter averaging) | `MultimodalVAE(fusion="mean")` |
| WB-VAE (Wasserstein barycenter) | `MultimodalVAE(fusion="barycenter")` |
| Dual VAE / beta-VAE | `MultimodalVAE` with one modality; `free_bits` in `ReconstructionTask` for the capacity term |
| Supervised CNN classifier | `FeatureEmbedder(Conv1dEncoder(...))` + `MultimodalPredictor` + `ClassificationTask` |
| Vanilla transformer classifier | `FeatureEmbedder(TransformerSignalEncoder(...))` + the same |

## Not migrated, and why

| Not here | Reason |
| --- | --- |
| `ResNet1D` / `xresnet1d` as vendored | Third-party research code with its own provenance and unclear licence, available from its author. `ResidualConv1dEncoder` is a clean implementation offering the same architecture family |
| ECG-FM, ST-MEM, TimesFM | External foundation models with their own repositories, weights and licences. Wrapping them is an experiment's job |
| CMVAE, MCMAE | A contrastive objective and a masked-autoencoder objective — different training paradigms, each appearing once. Adding either as a one-off would be an abstraction with a single user |
| Publication figure scripts | A figure names a cohort, a colour scheme and a claim. `interpret` returns the numbers; the figure belongs in `examples/` |
| Notebook baselines re-declaring the same encoder | Already covered: every one of them is `Conv1dEncoder` with different constants |

## Citation

If you build on either model, cite the original work:

```bibtex
@inproceedings{suvon2024cardiovae,
  title     = {Multimodal Variational Autoencoder for Low-cost Cardiac Hemodynamics Instability Detection},
  author    = {Suvon, Mohammod N. I. and Tripathi, Prasun C. and Fan, Wenrui and Zhou, Shuo and
               Liu, Xianyuan and Alabed, Samer and Osmani, Venet and Swift, Andrew J. and
               Chen, Chen and Lu, Haiping},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2024},
}

@article{suvon2025lsemvae,
  title   = {Multimodal Latent Fusion of ECG Leads for Early Assessment of Pulmonary Hypertension},
  author  = {Suvon, Mohammod N. I. and others},
  journal = {arXiv preprint arXiv:2503.13470},
  year    = {2025},
}
```
