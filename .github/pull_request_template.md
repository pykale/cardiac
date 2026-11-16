Fixes #{issue_number}.

### Description
A few sentences describing the changes proposed in this pull request.

### Status
**Ready/Work in progress/Hold**

### Automated review (delete after viewing)
Maintainers may run automated reviewers (Copilot review and/or @codex review) as a first pass. Please focus responses on Must-fix and Important items first. If something is unclear, ask for clarification.

### Review labels and assignment (delete after selection)
If you've been given access to this repository with role Triage or above, on the right:

- Select **one most appropriate** label. The release changelog groups pull requests by it.
- **When** your pull request is **ready** for review, select a reviewer. Use the suggested one if unsure.

If you cannot see an option to select a reviewer/label, that means one maintainer will get notified upon your pull request and then select for you.

### Types of changes
<!--- Put an `x` in all the boxes that apply, and remove the not applicable items -->
- [x] Non-breaking change (fix or new feature that would not break existing functionality).
- [ ] Breaking change (fix or new feature that would cause existing functionality to change).
- [ ] New tests added to cover the changes.
- [ ] In-line docstrings updated.
- [ ] [Documentation at `docs`](https://github.com/pykale/cardiac/tree/main/docs) manually updated for new API.

### Checklist
- [ ] `pre-commit run --all-files` and `pytest` pass locally
- [ ] New behaviour is tested on synthetic data, deterministically, on a CPU, without the network
- [ ] Nothing patient-derived is added -- no data, identifiers, absolute paths, credentials or private checkpoints, in code, tests, docs or outputs
- [ ] Library code names no dataset, cohort, registry or experiment outside a docstring
- [ ] Documentation is updated where the change reaches it: docstrings, `docs/`, `API_overview.md`, an example's README
- [ ] Any departure from a published model's behaviour is documented in `docs/models.md` and tested
