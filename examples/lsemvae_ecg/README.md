# LS-EMVAE: lead-specific representation learning from multi-lead ECG

Treating each ECG lead as its own modality, pretraining on plentiful unlabelled
twelve-lead recordings, and fine-tuning on the six leads a low-cost recorder captures.

```text
multi-lead ECG, unlabelled
      ↓
lead-specific representations          (one variational encoder per lead)
      ↓
hierarchical multimodal latent fusion  (product within lead groups, mixture across them)
      ↓
fine-tuning                            (a lead subset, frozen encoders, a clinical head)
      ↓
cardiovascular prediction
```

This reproduces the workflow of Suvon et al., *Multimodal Latent Fusion of ECG Leads
for Early Assessment of Pulmonary Hypertension*
([arXiv:2503.13470](https://arxiv.org/abs/2503.13470),
[reference implementation](https://github.com/Shef-AIRE/LS-EMVAE)). Every model,
objective, metric and training loop comes from `kalecardiac`; this directory holds only
what is specific to the experiment.

## Why lead-specific

A twelve-lead ECG is usually modelled as one twelve-channel signal, which asserts that
the leads are interchangeable channels of a single view. They are not: each lead
projects the heart's electrical activity onto a different axis, so they are better
treated as *different views of the same underlying state* — which is what a multimodal
VAE is built for. Two things follow that the channel view cannot give:

- **an encoder per lead that keeps working when the others are absent**, which is what
  makes a model pretrained on twelve leads usable on six;
- **a principled fusion**, where leads that see related territory reinforce one another
  through a product, and no single group of leads determines the latent on its own.

## Run it

```bash
# No data needed: synthetic cohorts, about a minute on a CPU
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml

# On your own cohorts
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/full.yaml \
    DATASET.ROOT /path/to/cohort

# How much each lead configuration is worth
python -m examples.lsemvae_ecg.main --leads I,I+II,I+II+III,I+II+III+aVR+aVL+aVF

# The published ablations, as configuration rather than other models
python -m examples.lsemvae_ecg.main FUSION.LATENT_METHOD poe        # no mixture step
python -m examples.lsemvae_ecg.main FUSION.LATENT_METHOD moe        # no product step
python -m examples.lsemvae_ecg.main OBJECTIVE.ALIGNMENT_WEIGHT 0.0  # no latent alignment
python -m examples.lsemvae_ecg.main MODEL.SHARED_DECODER False      # a decoder per lead
```

With `DATASET.ROOT` empty the run generates both cohorts. That verifies the pipeline
executes end to end; **it is not a result**, and the run says so in its first log line.

## Data

Both cohorts in the paper are private clinical data — a pulmonary hypertension registry
and a biobank — and neither is distributed. The example reads any cohorts with the
layout below.

```text
<DATASET.ROOT>/
├── cohort.csv                   one row per labelled subject
├── pretrain/
│   └── <subject_id>.npy         (num_leads, num_samples), unlabelled, ECG.LEADS
└── finetune/
    └── <subject_id>.npy         (num_leads, num_samples), labelled, ECG.LEADS
```

`cohort.csv` needs two columns, and may carry any others:

| Column | Meaning |
| --- | --- |
| `subject_id` | Matches the recording filenames under `finetune/` |
| `ENDPOINT.VALUE_COLUMN` | The measurement the endpoint thresholds, e.g. a mean pulmonary arterial pressure in mmHg |

The two directories are deliberately separate: pretraining consumes the large
unlabelled set, fine-tuning the small labelled one, and keeping them apart is what
makes it visible that no label reached the representation.

**Lead ordering.** Recordings are stored in `ECG.LEADS` order, and the names are
matched case-insensitively with any `LEAD_` prefix stripped, so `aVR`, `AVR` and
`LEAD_aVR` are the same lead. This is not pedantry: in the reference implementation the
lead list in the preprocessing script and the list the encoder indexes disagree — aVL
and aVF are transposed between them — so a per-lead model learns aVL's representation
from aVF's signal, and nothing crashes. Stating the order once and matching by name is
what prevents that.

## What the configuration controls

| Setting | Effect |
| --- | --- |
| `ECG.LEADS` | Leads the pretraining cohort records, in the order they are stacked into experts |
| `ECG.FINETUNE_LEADS` | Leads the labelled cohort records. Fewer than `ECG.LEADS` is the transfer this example is about |
| `ECG.DERIVE_MISSING` | Reconstruct an absent limb lead from I and II. Exact by Einthoven's and Goldberger's relations, so it costs nothing where those two were recorded |
| `FUSION.LATENT_METHOD` | `hime` is the published model; `poe` and `moe` are its two ablations |
| `FUSION.NUM_GROUPS` | Lead groups a hierarchical fusion forms before mixing across them |
| `MODEL.SHARED_DECODER` | One decoder for every lead, which forces the latent to carry lead-agnostic cardiac state |
| `OBJECTIVE.ALIGNMENT_WEIGHT` | Pulls each lead's posterior towards the joint one, which is what makes fine-tuning on a lead subset work |
| `ENDPOINT.THRESHOLD` | The clinical cut-off applied to the measurement |

Lead order matters to `hime`, because grouping is contiguous over the expert axis: the
conventional order puts the limb leads in the early groups and the precordial leads in
the later ones. A different order is a different grouping, and therefore a modelling
choice worth stating.

## Outputs

```text
<OUTPUT.OUT_DIR>/
├── config.yaml              the exact configuration this run used
├── cohort.json              what was assembled, and whether it was synthetic
├── pretrained.pt            the representation-learning weights
├── latent_space.csv         a two-dimensional projection of the pretrained latent
├── lead_agreement.json      how far each lead's posterior sits from the joint one
├── results.csv              one row per lead configuration
└── <leads>/
    ├── predictions.csv      one row per subject per fold
    ├── metrics.json         per-fold metrics and their summary
    ├── lead_importance.csv  attribution ratio and share per lead
    └── interpretation.json  the same, with the ablation
```

`lead_agreement.json` is the check available before any label exists: if the alignment
term did its job, every lead's posterior mean sits close to the joint one, and a lead
that does not is one whose encoder has gone its own way.

## Reading the lead importance

`lead_importance.csv` reports each lead's *integrated-gradient attribution ratio* — the
fraction of its samples whose normalised attribution clears a threshold — and that
fraction's share of the total. The ablation beside it answers a different question: what
the metric loses when the lead is withheld entirely.

The two can disagree, and the disagreement is informative rather than a bug. A lead can
be heavily attributed and still be redundant, because another lead carries the same
information; that is expected of leads which are, after all, projections of one signal.

## Attribution

The methodology is the authors'; the implementation here is a refactor. Two departures
from the reference implementation are deliberate and documented in
[`docs/models.md`](../../docs/models.md): the prior expert is no longer silently
discarded by the grouping, and the KL term is taken against the posterior that was
actually sampled from. Please cite the original paper if you build on this.
