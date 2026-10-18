"""The documentation covers the whole library, and every page of it is reachable.

Sphinx documents what an ``automodule`` directive names and nothing else, so a module
added without a line in ``docs/`` would ship undocumented and the build would not say
so. These checks need no Sphinx: they read the reference pages as text.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def library_modules() -> set[str]:
    """Every module and package under ``kalecardiac/``, dotted."""
    modules = set()
    for path in (ROOT / "kalecardiac").rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        modules.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return modules


def directives(name: str) -> set[str]:
    """Every module named by a Sphinx directive of this kind across the reference pages."""
    pattern = re.compile(rf"^\.\. {name}:: (\S+)", re.MULTILINE)
    return {match for page in DOCS.glob("*.rst") for match in pattern.findall(page.read_text(encoding="utf-8"))}


def table_of_contents() -> set[str]:
    """Pages the index's toctrees list."""
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    return set(re.findall(r"^ {3}([\w.]+)\s*$", index, re.MULTILINE))


def test_every_library_module_is_in_the_api_reference():
    missing = sorted(library_modules() - directives("automodule"))
    assert not missing, f"undocumented modules {missing}; add each to the matching docs/kalecardiac.<stage>.rst"


def test_no_reference_page_names_a_module_that_does_not_exist():
    stale = sorted(directives("automodule") - library_modules())
    assert not stale, f"docs/ documents {stale}, which no longer exist"


def test_every_reference_page_and_guide_is_reachable_from_the_index():
    pages = {page.stem for page in DOCS.glob("kalecardiac*.rst")} | {guide.stem for guide in DOCS.glob("*.md")}
    unreachable = sorted(pages - table_of_contents())
    assert not unreachable, f"{unreachable} are not in any toctree in docs/index.rst"
