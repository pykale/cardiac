"""The tutorial notebook runs top to bottom, and is stored without outputs.

A notebook is documentation that executes, and documentation drifts: an API change the
notebook missed would surface only when a reader ran it. This runs every code cell in
order, in one namespace, as a kernel would -- skipping only the cells tagged for Colab.
Without the workshop data the notebook builds synthetic stand-ins, so nothing is
downloaded and nothing private is needed.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "tutorials" / "cardiac_abnormality_assessment.ipynb"


def code_cells() -> list[dict]:
    """The notebook's code cells, in order."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def colab_only(cell: dict) -> bool:
    return "skip-execution" in cell["metadata"].get("tags", [])


def test_only_the_colab_setup_is_skipped():
    skipped = ["".join(cell["source"]) for cell in code_cells() if colab_only(cell)]
    assert len(skipped) == 2
    assert any("pip install" in source for source in skipped)
    assert any("drive.mount" in source for source in skipped)


def test_the_notebook_is_stored_without_outputs():
    # Outputs of a run on the workshop data would carry patient-derived images and
    # numbers into the repository.
    for cell in code_cells():
        assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.slow
def test_the_notebook_runs_on_its_synthetic_stand_ins(tmp_path, monkeypatch):
    pytest.importorskip("captum")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__main__"}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
        for index, cell in enumerate(code_cells()):
            if not colab_only(cell):
                exec(compile("".join(cell["source"]), f"tutorial cell {index}", "exec"), namespace)
    plt.close("all")

    assert namespace["SYNTHETIC"] is True
    assert "it is not a result" in printed.getvalue()
    assert set(namespace["summary"].index) == {"accuracy", "roc_auc", "mcc", "sensitivity", "specificity"}
    assert (tmp_path / "cardiovae_pretrained.pt").exists()
    # The interpretation produced an importance for every sample of the plotted lead.
    assert namespace["ecg_importance"].shape == namespace["ecg"].shape
