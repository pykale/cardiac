"""Sphinx configuration for the KaleCardiac documentation.

The guides are the Markdown files beside this one, read by MyST; the API reference is
one page per pipeline stage, ``kalecardiac.<stage>.rst``, as PyKale lays out its own.
Build with warnings as errors, which is what Read the Docs does::

    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Document the source tree this file sits in, whatever copy of the package is installed.
sys.path.insert(0, str(ROOT))

project = "KaleCardiac"
author = "PyKale"
copyright = "PyKale contributors"
release = (ROOT / "kalecardiac" / "_version.txt").read_text(encoding="utf-8").strip()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Google-style docstrings throughout, as PyKale writes them. An ``Attributes:`` section
# becomes instance-variable fields rather than attribute entries, which a dataclass's
# fields would otherwise describe a second time.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# The guides link to one another's headings, as GitHub renders them.
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"KaleCardiac {release}"
