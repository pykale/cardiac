# AGENTS.md

Instructions for coding agents working in this repository. Human contributors should
read [README.md](README.md) and [docs/architecture.md](docs/architecture.md).

## Setup

Python 3.10–3.12 is required. Install PyTorch first if a specific CUDA build is needed,
then:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,interpret]"     # Windows: .venv/Scripts/pip
```

Verify the install:

```bash
.venv/bin/python -c "import kalecardiac, torch, kale; print(kalecardiac.__version__)"
```

## Commands

| Task | Command |
| --- | --- |
| Run tests | `.venv/bin/pytest` |
| Run tests, skipping the whole-workflow runs | `.venv/bin/pytest -m "not slow"` |
| Run tests with coverage | `.venv/bin/pytest --cov=kalecardiac --cov-report=term-missing` |
| Format | `.venv/bin/ruff format .` |
| Lint | `.venv/bin/ruff check .` |
| Run an example end to end | `.venv/bin/python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml` |

Both `pre-commit run --all-files` (ruff, ruff-format, mypy) and `pytest` must pass before
a change is complete. These are the same checks CI runs.

## Data

**There is no data in this repository, and none may be added.** Both studies this
package was built from were developed on private clinical cohorts, and nothing
patient-derived belongs here: no recordings, no images, no identifiers, no labels, no
derived per-subject files, no credentials, no pretrained artefacts trained on them.

Tests and examples use `examples/synthetic_data.py`, which generates cohorts with a real
dependence between signal and endpoint — so a broken pipeline scores about 0.5 and a
working one scores clearly above it. They are **not** physiologically accurate ECGs.

Nothing in the test suite may reach the network. A run on synthetic data is a
verification, never a result; say so when reporting one, as the examples do in their
first log line.

Cohort layouts are documented in each example's README so that a researcher with
approved access can supply `DATASET.ROOT`. Document a schema; never commit one.

## Layout

```
kalecardiac/
├── auto/        high-level construction (AutoCardiac* classes, planned)
├── loaddata/    access APIs per modality, plus subject-level splitting
├── prepdata/    transforms over signals and images
├── model/       layers/ blocks, embed/ encoders and models, predict/ heads and objectives
├── pipeline/    the trainer and its tasks
├── evaluate/    metrics and prediction scoring, by task
├── interpret/   attribution, lead importance, latent inspection
├── utils/       seeding, artefact writing, checkpoint loading
└── config.py    configuration schema and defaults
```

### There is one trainer

`CardiacTrainer` takes a *model* and a *task*, and neither constrains the other. Do not
add `CardioVAETrainer`, `LSEMVAETrainer`, `ECGTrainer` or `CXRTrainer` — there is
nothing for one to add, because the model is an argument and the objective is an
argument.

A task decides exactly four things: the head, the loss, what the model is asked to
produce (`forward_kwargs`), and the epoch metric. **A new endpoint is a new
`PredictionTask`, never a new trainer.**

**Pretraining is a task.** `ReconstructionTask` supervises on the batch's own modalities
and takes no labels, which is why representation learning needs no second trainer and is
not a pipeline stage.

### There is one multimodal VAE

`MultimodalVAE` with a `fusion` argument and a decoder mapping covers five published
models. `CardioVAE` and `LSEMVAE` are thin configurations of it, exported under those
names because that is what the literature calls them.

**Do not add a model class for an ablation.** "Without the mixture step" is
`fusion="poe"`. "Without the shared decoder" is `shared_decoder=False`. If a change can
be a constructor argument, it is one.

### `loaddata/` is one access API per kind of data

```
loaddata/
├── ecg_access.py         ECGArraySource, ECGFileSource, LeadSource, lead_sources, the lead vocabulary
├── image_access.py       ImageArraySource, ImageFileSource
├── multimodal_access.py  MultimodalDataset, ModalitySource, ColumnTarget, the record types
└── splitting.py          HoldOut, CrossValidation, Predefined -- scikit-learn's shape
```

Adding a modality is a dictionary entry. Adding a *kind* of modality is one
`ModalitySource` subclass. Never a new dataset class per combination.

### `model/` is three stages, and a class belongs to exactly one

```
model/
├── layers/    blocks that transform tensors and know no modality
├── embed/     encoders adapted to a contract, and the models that fuse them
└── predict/   heads, and every objective including the ELBO
```

A **layer** transforms tensors. An **embedder** adapts one to a contract (`out_dim`, or
`latent_dim` and a posterior). A **head** turns a representation into a score. Put a new
block in `layers/` and adapt it in `embed/`, never both at once.

### Splitting stays in `loaddata/`, not `prepdata/`

A splitter decides which subjects load into which loader; `prepdata/` is transforms. The
thing that creates the folds cannot be state belonging to one. scikit-learn draws the
same line between `model_selection` and `preprocessing`.

## Constraints

These are enforced by [`tests/test_package_structure.py`](tests/test_package_structure.py),
which parses every library module. They are not style preferences.

- **Nothing in `kalecardiac/` may name a dataset, a cohort, a registry or an
  experiment.** Not in code, not in a comment. Docstrings are exempt, because that is
  where a refactored model credits the study it came from.
- **No absolute paths.** A path is an experiment's, never a library's.
- **The library never imports `examples/`.** The dependency runs one way.
- **Stage imports follow the pipeline order.** `loaddata` must not import `model`; no
  stage may import `pipeline`.
- **Every module, public class and public function has a docstring.**

Further constraints that tests check case by case:

- **Splitting is subject-level.** A subject's recordings must never span two splits —
  including the train/validation carve, not only train/test.
- **Validation is always carved out of the training half**, never out of the test set.
- **Anything derived from labels is computed on the training split of each fold.** A
  class weight most obviously; recomputing it per fold is not optional.
- **Missing modalities are carried by the presence mask**, never inferred from zeros. A
  zero-filled placeholder keeps a batch rectangular and is never evidence.
- **Lead names are matched canonically.** `aVR`, `AVR` and `LEAD_aVR` are one lead. The
  reference implementation this package refactors has aVL and aVF transposed between its
  preprocessing script and its encoder index map, which silently trains a per-lead model
  on the wrong lead.
- **`persistent_workers` loaders are released with `release_workers`** when the split
  that built them is finished.

## Preserving scientific behaviour

This package is a refactor of published work, not a rewrite of it.

Where a model here corresponds to a published one, its mathematical behaviour must
match. Where it deliberately does not — because the original contains a bug, an
ambiguity or a numerical hazard — **document the departure in
[docs/models.md](docs/models.md), and test it.** Three are documented already: the
discarded prior expert, the KL term taken against a posterior that was never sampled,
and sampling that depended on a module flag.

Do not change an algorithm silently, and do not "fix" something that merely looks odd
without establishing that it is wrong.

## Examples

Named `<method>_<modality>`. Each has the same five parts — `README.md`, `config.py`,
`data.py`, `runner.py`, `main.py` — plus `configs/quick.yaml` (synthetic, about a
minute) and `configs/full.yaml` (the published setup).

Examples run as modules from the repository root — `python -m examples.<name>.main` —
so their imports resolve as ordinary packages. Do not add `sys.path` manipulation.

**An example composes the library; it never defines a model.** A test asserts that
neither runner contains `nn.Module`:

```python
# Right
from kalecardiac.model.embed import LSEMVAE
model = LSEMVAE(leads=cfg.ECG.LEADS, fusion=cfg.FUSION.LATENT_METHOD)

# Wrong
class LSEMVAE(nn.Module):    # 500 lines redefining what the library already has
```

An example owns: configuration, cohort construction, endpoint definition, the fold
structure, running the pipeline, and reporting.

## Conventions

- Follow [PyKale](https://github.com/pykale/pykale) conventions: verb-oriented stages,
  Google-style docstrings, type hints, YACS configuration.
- **Reuse PyKale APIs where one exists rather than reimplementing.** Where something is
  reimplemented — `Conv1dEncoder` over `SignalVAEEncoder`, `LinearHead` over
  `kale.predict.decode` — the reason is in the module docstring, and a test asserts the
  two agree where their shapes do.
- Line length 120. Formatting is enforced by ruff-format; do not hand-format.
- Comments explain why, not what. Do not restate the code.
- New configuration belongs in `kalecardiac/config.py`, never hardcoded in a module —
  unless it names a dataset, in which case it belongs in an example's `config.py`.
- An error message should say what to do next. `"crop or pad the cohort first; see
  kalecardiac.prepdata.crop_or_pad"` beats `"shape mismatch"`.

## Adding a component

1. Implement it in the matching `kalecardiac/` subpackage.
2. Export it from that subpackage's `__init__.py`.
3. Add tests in the mirrored `tests/` path, using synthetic data.
4. Add any new setting to `kalecardiac/config.py` with a comment saying why it exists.
5. Run the format, lint and test commands above.

Before adding an abstraction, check it has a second user. This package has no protocol
with one implementation, no registry with one entry, and no factory that exists for
theoretical extensibility.
