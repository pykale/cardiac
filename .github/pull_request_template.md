## What this changes

<!-- One or two sentences on what and why, and the issue it closes, e.g. "Closes #12". -->

## Checklist

- [ ] `pre-commit run --all-files` and `pytest` pass locally
- [ ] New behaviour is tested on synthetic data, deterministically, on a CPU, without the network
- [ ] Nothing patient-derived is added -- no data, identifiers, absolute paths, credentials or private checkpoints, in code, tests, docs or outputs
- [ ] Library code names no dataset, cohort, registry or experiment outside a docstring
- [ ] Documentation is updated where the change reaches it: docstrings, `docs/`, `API_overview.md`, an example's README, `CHANGELOG.md`
- [ ] Any departure from a published model's behaviour is documented in `docs/models.md` and tested
