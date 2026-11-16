# Contributing to KaleCardiac

Thank you for helping. KaleCardiac follows the conventions of [PyKale](https://github.com/pykale/pykale), and contributing works much as it does there. This page covers the workflow; the engineering rules — the layout, what belongs in the library and what in an example, the constraints the tests enforce — are in [AGENTS.md](AGENTS.md), which applies to people as much as to coding agents. Read it before a first pull request.

## Clinical data: never

**Nothing patient-derived may enter this repository** — not recordings, images, identifiers, labels, per-subject derived files, file paths from a clinical system, screenshots of records, credentials, or weights trained on private data. This holds for issues and pull-request discussions as well as for commits.

Every test and example runs on [`examples/synthetic_data.py`](examples/synthetic_data.py), which generates cohorts whose endpoint really depends on the signal. If a bug appears only on your cohort, reproduce it on a synthetic one before reporting it: change the generator's arguments — the number of subjects, the leads, the class balance, the sampling rate — until it shows. A report that cannot be reproduced without private data cannot be acted on here.

## Setting up

Python 3.10–3.12. PyKale 0.2 requires `torch<=2.6.0`, so if you need a particular CUDA build of PyTorch, install a 2.6-or-earlier build first.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,interpret,ecg]"     # Windows: .venv\Scripts\pip
.venv/bin/pre-commit install
```

## Making a change

1. Open an issue first for anything beyond a small fix, so the design can be agreed before the code is written.
2. Branch from `main`. Keep a pull request to one purpose.
3. Write the tests with the code, in the mirrored path under `tests/`, on synthetic data. A test must be deterministic, run on a CPU, and never reach the network.
4. Run the checks CI runs. Both must pass:

   ```bash
   pre-commit run --all-files     # ruff, ruff-format, mypy
   pytest                         # includes the library's doctests
   ```

   `pytest -m "not slow"` skips the whole-workflow runs while you iterate.
5. Update what the change touches: the docstrings, the guides in `docs/`, [API_overview.md](API_overview.md), and an example's README for a configuration key.
6. Title the pull request as its changelog entry, e.g. "Add a PPG encoder", and give it one label: `new feature`, `bug`, `refactor`, `enhancement`, `dependencies`, `tests` or `documentation`. The changelog is generated from these when the next version is released, so don't edit a changelog by hand.

### Scientific behaviour

KaleCardiac refactors published models; it does not reinterpret them. A change to how a model computes something must preserve the published behaviour, or — where the original is wrong, ambiguous or numerically unsafe — document the departure in [docs/models.md](docs/models.md) and test it. Do not "fix" something that only looks odd without establishing that it is wrong.

### Documentation

The API reference is generated from the docstrings, one page per pipeline stage. A new module needs an `automodule` line in the matching `docs/kalecardiac.<stage>.rst`; a test fails until it has one. To build the site locally, as Read the Docs does:

```bash
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Releasing

Releases follow PyKale's process, with the steps maintainers did by hand for PyKale 0.2.2 automated:

1. Bump the version in `kalecardiac/_version.txt`, e.g. to `0.2.0` or `0.2.0rc1`, and open a pull request from a branch of this repository titled `Release 0.2.0`.
2. `changelog.yml` collects the pull requests merged since the last tag, grouped by label, and commits the section to `.github/CHANGELOG.md` on the branch. Within each group the entries are sorted by pull request number, titles lose any `feat:`-style prefix and start with a capital letter, and earlier release pull requests are left out. It also posts the section as a comment and runs CI on the new commit.
3. Review the section and edit it on the branch if needed. A rerun never overwrites an existing section.
4. Approve the pull request. `release-on-approval.yml` waits for the checks, merges, creates the `v0.2.0` GitHub release with the changelog section as its notes, builds the package with `build-dist.yml` and publishes it: alpha, beta and release-candidate versions go to Test PyPI, the rest to PyPI.

Only approvals from owners, members and collaborators, of the pull request's latest commit, start a release. If a check fails, fix it, push, and approve again.

Publishing uses PyPI trusted publishing, so no PyPI token is stored. pypi.org and test.pypi.org each trust two workflows of `pykale/cardiac`: `release-on-approval.yml` and `release.yml`, which publishes a release created by hand. The publishers use the environments `pypi` and `testpypi`. The repository needs the `new feature`, `bug`, `refactor`, `enhancement`, `dependencies`, `tests` and `documentation` labels that `.github/changelog-config.json` groups by. New issues and pull requests are added to the [KaleCardiac project](https://github.com/orgs/pykale/projects/16), which needs the `ADD_TO_PROJECT_PAT` secret.

Changes up to 0.1.0a0, and the unreleased work recorded since, are in the earlier, hand-written [CHANGELOG.md](CHANGELOG.md).

## Reporting a bug or proposing a feature

Use the issue templates. For a bug, include the smallest synthetic reproduction you can and the output of:

```bash
python -c "import kalecardiac, torch, kale; print(kalecardiac.__version__, torch.__version__, kale.__version__)"
```

For a feature, say which pipeline stage it belongs to and who else would use it: the package adds an abstraction only when it has a second user.

## Licence

By contributing, you agree that your contribution is licensed under the [MIT licence](LICENSE) that covers the project.
