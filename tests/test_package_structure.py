"""Architectural constraints, enforced rather than documented.

Two rules carry most of this package's design, and both are the kind that erodes
quietly under maintenance:

1. **Nothing in ``kalecardiac/`` names a dataset, a cohort or an experiment.** The
   library supplies mechanisms; the decisions belong in ``examples/``. The moment a
   registry name or a column name from one study appears in a module, every other study
   has to work around it.
2. **The pipeline stages depend downstream, never upstream.** ``loaddata`` must not
   import ``model``, and no stage may import ``pipeline``, or the verb-oriented order
   stops being real.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import kalecardiac

PACKAGE = Path(kalecardiac.__file__).parent

#: Every module of the library, so a new file is covered without being listed.
MODULES = sorted(path for path in PACKAGE.rglob("*.py"))

#: Names from the private cohorts and the two papers' experiments. A library module
#: mentioning one of these has taken a dataset decision that belongs to an example.
#:
#: Word-bounded, and matched case-insensitively against code only -- docstrings are
#: where a model's provenance is credited, and attribution is required rather than
#: forbidden.
FORBIDDEN = (
    "aspire",
    "ukbiobank",
    "uk_biobank",
    "mimic",
    "physionet",
    "pawp",
    "mpap",
    "hancock",
)

#: Which stages a stage may import from. The pipeline order is load -> prep -> model ->
#: pipeline, with evaluate and interpret downstream of all of it.
ALLOWED_IMPORTS = {
    "utils": set(),
    "loaddata": {"utils"},
    "prepdata": {"utils", "loaddata"},
    "model": {"utils", "loaddata", "prepdata"},
    "evaluate": {"utils", "loaddata", "model"},
    "pipeline": {"utils", "loaddata", "prepdata", "model", "evaluate"},
    "interpret": {"utils", "loaddata", "prepdata", "model", "evaluate"},
    "auto": {"utils", "loaddata", "prepdata", "model", "evaluate", "pipeline", "interpret"},
    "config": set(),
}


def stage_of(path: Path) -> str:
    """Which pipeline stage a module belongs to."""
    relative = path.relative_to(PACKAGE)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def code_without_docstrings(path: Path) -> str:
    """The module's source with every docstring removed.

    Docstrings are where a refactored model credits the study it came from, so they are
    exempt from the dataset-name rule that the code is held to.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    # Comments are code, not documentation: a comment naming a cohort is as much of a
    # coupling as an identifier naming one.
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for number, line in enumerate(lines, start=1) if number not in spans)


class TestNoDatasetNames:
    @pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(PACKAGE)))
    def test_no_module_names_a_cohort_or_an_experiment(self, path):
        code = code_without_docstrings(path).lower()
        found = [name for name in FORBIDDEN if re.search(rf"\b{name}\b", code)]
        assert not found, (
            f"{path.relative_to(PACKAGE)} mentions {found} outside a docstring. Cohort and experiment names "
            f"belong in examples/; the library supplies the mechanisms those decisions are expressed with."
        )

    def test_no_module_hardcodes_an_absolute_path(self):
        offenders = []
        for path in MODULES:
            code = code_without_docstrings(path)
            if re.search(r"""["'](?:/(?:home|users|mnt|data)/|[A-Za-z]:[\\/])""", code):
                offenders.append(str(path.relative_to(PACKAGE)))
        assert not offenders, f"absolute paths in {offenders}; a path is an experiment's, never a library's"


class TestStageDependencies:
    @pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(PACKAGE)))
    def test_a_stage_imports_only_from_upstream_stages(self, path):
        stage = stage_of(path)
        allowed = ALLOWED_IMPORTS.get(stage)
        if allowed is None:
            pytest.skip(f"{stage} is not a pipeline stage")

        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kalecardiac."):
                imported.add(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("kalecardiac."):
                        imported.add(alias.name.split(".")[1])

        violations = sorted(imported - allowed - {stage})
        assert not violations, (
            f"{path.relative_to(PACKAGE)} imports from {violations}, which are not upstream of {stage!r}. "
            f"The stages run load -> prep -> model -> pipeline, and the imports must follow."
        )


class TestPublicApi:
    @pytest.mark.parametrize(
        "module",
        ["loaddata", "prepdata", "model", "pipeline", "evaluate", "interpret", "utils"],
    )
    def test_every_exported_name_exists(self, module):
        imported = __import__(f"kalecardiac.{module}", fromlist=["__all__"])
        missing = [name for name in getattr(imported, "__all__", []) if not hasattr(imported, name)]
        assert not missing, f"kalecardiac.{module}.__all__ promises {missing}, which it does not define"

    def test_the_package_exposes_its_stages_lazily(self):
        assert set(kalecardiac.__all__) == {
            "__version__",
            "auto",
            "loaddata",
            "prepdata",
            "model",
            "pipeline",
            "evaluate",
            "interpret",
            "utils",
        }

    def test_an_unknown_attribute_raises_rather_than_importing(self):
        # The lazy-import hook must not try to import an arbitrary name as a submodule.
        with pytest.raises(AttributeError, match="no attribute"):
            _ = kalecardiac.nonexistent

    def test_the_version_is_readable(self):
        assert re.match(r"^\d+\.\d+\.\d+", kalecardiac.__version__)

    def test_every_stage_can_be_imported(self):
        for name in ("auto", "loaddata", "prepdata", "model", "pipeline", "evaluate", "interpret", "utils"):
            assert getattr(kalecardiac, name) is not None


class TestDocstringsAndTyping:
    @pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(PACKAGE)))
    def test_every_module_has_a_docstring(self, path):
        assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), (
            f"{path.relative_to(PACKAGE)} has no module docstring"
        )

    @pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(PACKAGE)))
    def test_every_public_class_and_function_has_a_docstring(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        undocumented = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef)
            and not node.name.startswith("_")
            and ast.get_docstring(node) is None
        ]
        assert not undocumented, f"{path.relative_to(PACKAGE)} has undocumented public names {undocumented}"


class TestTestsMirrorThePackage:
    def test_every_stage_has_a_test_directory(self):
        tests = Path(__file__).parent
        for stage in ("loaddata", "prepdata", "model", "pipeline", "evaluate", "interpret", "utils"):
            assert (tests / stage).is_dir(), f"no tests/{stage}/ mirroring kalecardiac/{stage}/"

    def test_the_library_never_imports_the_examples(self):
        # The dependency runs one way: an example imports the library, never the
        # reverse. Otherwise a dataset decision reaches the library through the back
        # door.
        for path in MODULES:
            source = path.read_text(encoding="utf-8")
            assert "import examples" not in source and "from examples" not in source, (
                f"{path.relative_to(PACKAGE)} imports from examples/"
            )
