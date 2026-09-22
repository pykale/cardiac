# KaleCardiac

> *Multimodal machine learning for cardiovascular research: ECG, chest X-ray, cardiac imaging, and clinical data, with representation learning, multimodal fusion, and clinical prediction.*

-----------------------------------------

<!-- Keep badges to just ONE line, i.e. only the most important badges! -->
[![Built on PyKale](https://img.shields.io/badge/built%20on-PyKale-5699C6)](https://github.com/pykale/pykale)
[![CI](https://github.com/pykale/cardiac/actions/workflows/ci.yml/badge.svg)](https://github.com/pykale/cardiac/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pykale/cardiac/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

[Getting Started](https://github.com/pykale/cardiac#how-to-use) |
[Documentation](https://github.com/pykale/cardiac/tree/main/docs) |
[Examples](https://github.com/pykale/cardiac/tree/main/examples) |
[Contributing](https://github.com/pykale/cardiac#step-2-building-and-contributing) |
[Architecture](https://github.com/pykale/cardiac/blob/main/docs/architecture.md)

KaleCardiac is a cardiovascular library built on [PyKale](https://github.com/pykale/pykale), a library in the [PyTorch ecosystem](https://pytorch.org/ecosystem/), aiming to make cardiac machine learning more accessible to interdisciplinary research by bridging gaps between clinical data, software, and end users. Both machine learning experts and clinical researchers can do better research with our accessible, scalable, and sustainable design, guided by green machine learning principles. KaleCardiac inherits PyKale's unified *pipeline-based* API and extends it where cardiology needs more: [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) across electrocardiography, radiography and cardiac imaging, and *representation learning* that turns plentiful unlabelled recordings into a model a small labelled cohort can be fine-tuned on.

The problem cardiac machine learning keeps running into is not model capacity. It is that the **labels are expensive and the recordings are not**. A haemodynamic measurement requires a catheter; an ECG requires two minutes and a machine that costs very little. Learning a representation from unlabelled recordings and transferring it to the small labelled cohort is therefore the shape of the problem, and it is what this library is organised around. Two further things follow from clinical reality rather than from preference: modalities go missing, and a twelve-lead diagnostic recording is not what a low-cost device captures — so **absent modalities are first class**, and a model pretrained on twelve leads runs on six without retraining.

KaleCardiac enforces the same *standardization* and *minimalism* as PyKale, via green machine learning concepts of *reducing* repetitions and redundancy, *reusing* existing resources, and *recycling* learning models across areas. Modules are built on PyKale rather than beside it. The clearest expression of that minimalism is here: **five published multimodal cardiac models are one class with two arguments**, so an ablation is a configuration change rather than a near-duplicate file.

#### Pipeline-based API

- `loaddata` loads data from disk or memory as input, per modality, with subject-level leakage-safe splitting
- `prepdata` preprocesses data to fit machine learning modules below (resampling, cropping, normalisation, lead selection)
- `model.layers` holds blocks that transform tensors: convolutional, residual and transformer encoders, decoders, and the Gaussian expert mechanisms
- `model.embed` embeds data in a new space to learn a new representation (`MultimodalVAE` and its published configurations `CardioVAE` and `LSEMVAE`, plus the fusion that combines modalities)
- `model.predict` turns a representation into a score: `LinearHead` and `MLPHead`, with every objective that trains them — including the multimodal ELBO
- `evaluate` evaluates the performance using some metrics, chosen for a clinical reading: ROC-AUC alongside sensitivity and specificity, and regression metrics in the endpoint's own units
- `interpret` interprets the features and outputs via post-prediction analysis: gradient attribution, lead importance, modality ablation, latent inspection
- `auto` selects and constructs a workflow from a configuration alone (`AutoCardiac*` classes, planned)
- `pipeline` specifies a machine learning workflow by combining several other modules: one *trainer* takes a *model* and a *task*, and **pretraining is a task**, not a second trainer

#### Supported modalities

| Modality | Representation | Status |
| --- | --- | --- |
| **ECG** | Named leads, any count, any length, any rate — as one recording or one modality per lead | Implemented |
| **Chest X-ray** | `(channels, height, width)` | Implemented |
| **Cardiac MRI / CT slice, echocardiography frame** | The same image contract; nothing in it is radiograph-specific | Implemented |
| **Clinical vectors** | One array per subject | Implemented |
| **Volumetric cine stacks** | A spacing, an orientation, a slice axis | Planned |

#### Supported model families

| Family | Classes |
| --- | --- |
| Signal encoders | `Conv1dEncoder`, `ResidualConv1dEncoder`, `TransformerSignalEncoder` |
| Image encoders | `Conv2dEncoder` |
| Variational adapters | `VariationalEncoder` turns *any* backbone into one |
| Multimodal VAEs | `MultimodalVAE` with `poe`, `moe`, `hime`, `mean` or `barycenter` fusion |
| Published cardiac models | `CardioVAE`, `LSEMVAE` |
| Discriminative | `MultimodalPredictor` with `concat`, `mean` or `attention` fusion |

#### Example usage

- `examples` demonstrate real applications on specific datasets with a standardized structure.

## How to Use

### Step 0: Installation

KaleCardiac supports Python 3.10, 3.11, or 3.12. Before installing `kalecardiac`, we suggest you to first [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware, and then [install PyKale](https://pykale.readthedocs.io/en/latest/installation.html) following its official instructions.

Installation of `kalecardiac` from source:

```bash
git clone https://github.com/pykale/cardiac.git
cd cardiac
pip install -e .
```

Heavy libraries are kept out of the core install and grouped into extras, so that an ECG workflow does not pull in an imaging stack:

| Extra | Packages | When to use |
| --- | --- | --- |
| `ecg` | wfdb, neurokit2 | Reading vendor ECG formats, and delineating P/QRS/T waves |
| `imaging` | pillow, pydicom | Reading images and DICOM from disk |
| `interpret` | captum, umap-learn | Attribution and latent-space projection |
| `dev` | pytest, ruff, mypy, pre-commit | Development and CI |

The signal and image transforms are pure NumPy and Torch, so the core install is enough to preprocess arrays, train every model and score it. For more details, see [the quickstart guide](https://github.com/pykale/cardiac/blob/main/docs/quickstart.md).

### Step 1: Tutorials and Examples

Start with the [quickstart](https://github.com/pykale/cardiac/blob/main/docs/quickstart.md), which walks through a complete workflow — load, pretrain without labels, fine-tune, score, interpret — in about forty lines. **No data or credentials are needed:** both examples generate their cohorts when no data path is given, so the whole thing runs offline in about a minute on a CPU.

```bash
python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/quick.yaml
python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml
```

Browse through the [**examples**](https://github.com/pykale/cardiac/tree/main/examples) to see the two cardiac use cases KaleCardiac was built from:

| Example | Modalities | Workflow |
| --- | --- | --- |
| [`cardiovae_multimodal`](https://github.com/pykale/cardiac/tree/main/examples/cardiovae_multimodal) | Chest X-ray + ECG | Pretrain a product-of-experts VAE, fine-tune on a haemodynamic endpoint |
| [`lsemvae_ecg`](https://github.com/pykale/cardiac/tree/main/examples/lsemvae_ecg) | Multi-lead ECG, one modality per lead | Pretrain on twelve leads, fine-tune on the six a low-cost recorder captures |

Which model, which fusion, which backbone and which leads are all configuration, not code:

```bash
python -m examples.lsemvae_ecg.main FUSION.LATENT_METHOD poe      # an ablation
python -m examples.lsemvae_ecg.main ECG.ENCODER transformer       # another backbone
python -m examples.lsemvae_ecg.main --leads I,I+II,I+II+III       # lead configurations compared
python -m examples.lsemvae_ecg.main ENDPOINT.REGRESSION True      # predict the measurement, not the threshold
```

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an [issue](https://github.com/pykale/cardiac/issues) if you find something wrong.

### Step 2: Building and Contributing

Build new modules and/or projects with KaleCardiac referring to the [architecture guide](https://github.com/pykale/cardiac/blob/main/docs/architecture.md), e.g., on how to modify an existing pipeline or build a new one. New code belongs in the pipeline stage that names what it does, and anything specific to one dataset belongs in `examples` rather than in the library.

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) KaleCardiac to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions) about it.
- Use KaleCardiac in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/cardiac/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/cardiac/graphs/contributors).

Run the same checks as CI before opening a pull request:

```bash
pip install -e ".[dev,interpret]"
pre-commit run --all-files && pytest --cov=kalecardiac
```

The test suite needs no data and no network; `pytest -m "not slow"` skips the whole-workflow example runs. Conventions and architectural constraints are documented in [AGENTS.md](https://github.com/pykale/cardiac/blob/main/AGENTS.md), which applies to human and automated contributors alike, and several of them are enforced by [`tests/test_package_structure.py`](https://github.com/pykale/cardiac/blob/main/tests/test_package_structure.py) rather than left to review. See PyKale's [contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) for more details. The participation in this open source project is subject to PyKale's [Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md).

## Data and privacy

**No patient data is in this repository, and none can be.** Both studies KaleCardiac was built from were developed on private clinical cohorts — a pulmonary hypertension registry and a biobank — governed by the approvals under which they were collected. Nothing patient-derived is committed here: no recordings, no images, no identifiers, no labels, no derived per-subject files, no credentials.

What *is* here is the workflow, and enough documentation of the expected schema that a researcher with approved access to a comparable cohort can supply their own path and reproduce it:

```yaml
DATASET:
  ROOT: /path/to/your/local/cohort
```

Each example's README documents the layout it expects in full. Everything else runs on synthetic cohorts, which is what the test suite and the `quick.yaml` configurations use. A synthetic run verifies that the pipeline executes; **it is not a result**, and every run says so in its first log line.

## Who We Are

### The Team

KaleCardiac is developed within the [PyKale](https://github.com/pykale/pykale) project at the University of Sheffield, with contributions from many other [contributors](https://github.com/pykale/cardiac/graphs/contributors).

### Citation

KaleCardiac does not have a publication of its own yet. Please consider citing the PyKale [CIKM2022 paper](https://doi.org/10.1145/3511808.3557676) below if you find _KaleCardiac_ useful to your research.

```lang-latex
    @inproceedings{pykale-cikm2022,
      title     = {{PyKale}: Knowledge-Aware Machine Learning from Multiple Sources in {Python}},
      author    = {Haiping Lu and Xianyuan Liu and Shuo Zhou and Robert Turner and Peizhen Bai and Raivo Koot and Mustafa Chasmai and Lawrence Schobs and Hao Xu},
      booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM)},
      doi       = {10.1145/3511808.3557676},
      year      = {2022}
    }
```

If you use the `CardioVAE` or `LSEMVAE` models, please also cite the work they come from:

```lang-latex
    @inproceedings{suvon2024cardiovae,
      title     = {Multimodal Variational Autoencoder for Low-cost Cardiac Hemodynamics Instability Detection},
      author    = {Mohammod N. I. Suvon and Prasun C. Tripathi and Wenrui Fan and Shuo Zhou and Xianyuan Liu and
                   Samer Alabed and Venet Osmani and Andrew J. Swift and Chen Chen and Haiping Lu},
      booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
      doi       = {10.48550/arXiv.2403.13658},
      year      = {2024}
    }

    @article{suvon2025lsemvae,
      title   = {Multimodal Latent Fusion of {ECG} Leads for Early Assessment of Pulmonary Hypertension},
      author  = {Mohammod N. I. Suvon and others},
      journal = {arXiv preprint arXiv:2503.13470},
      year    = {2025}
    }
```

### Acknowledgements

KaleCardiac is built on [PyKale](https://github.com/pykale/pykale) and inherits its [acknowledgements](https://github.com/pykale/pykale#acknowledgements).

The two model families it ships were refactored from research code released by the [Shef-AIRE](https://github.com/Shef-AIRE) group at the University of Sheffield: [AI4Cardiothoracic-CardioVAE](https://github.com/Shef-AIRE/AI4Cardiothoracic-CardioVAE) and [LS-EMVAE](https://github.com/Shef-AIRE/LS-EMVAE). The methodology is theirs; the implementation here is a clean-room refactor into reusable APIs, and the few places where it deliberately departs from the original are documented in [docs/models.md](https://github.com/pykale/cardiac/blob/main/docs/models.md) rather than changed silently. Both build in turn on the multimodal VAE of Wu and Goodman (2018).
