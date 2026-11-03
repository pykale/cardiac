"""The tutorial notebook runs top to bottom on raw DICOM files, and is stored without outputs.

A notebook is documentation that executes, and documentation drifts: an API change the
notebook missed would surface only when a reader ran it. This runs every code cell in
order, in one namespace, as a kernel would -- skipping only the cells tagged for Colab,
which install the package and download the dataset. The dataset is generated instead: a
synthetic cohort written as DICOM in the tutorial's layout, so nothing is downloaded and
nothing private is needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from examples.cardiovae_multimodal import config as cardiovae_config

TUTORIAL = Path(__file__).resolve().parents[1] / "tutorials" / "cardiac-abnormality-assessment"
NOTEBOOK = TUTORIAL / "tutorial-CardioVAE.ipynb"
CONFIGURATIONS = ("tutorial_pretrain.yaml", "tutorial_finetune.yaml")


def cells(kind: str = "code") -> list[dict]:
    """The notebook's cells of one kind, in order."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == kind]


def source(cell: dict) -> str:
    return "".join(cell["source"])


def colab_only(cell: dict) -> bool:
    return "skip-execution" in cell["metadata"].get("tags", [])


def test_the_configurations_are_the_cardiovae_examples():
    # They live with the model they configure, and come with KaleCardiac rather than with
    # the downloaded dataset; that each merges into the example is test_config's check.
    code = "\n".join(source(cell) for cell in cells())
    for name in CONFIGURATIONS:
        assert (cardiovae_config.CONFIG_DIR / name).is_file()
        assert name in code


def test_only_the_install_and_the_download_are_skipped():
    # The download reaches the network, which nothing in the test suite may.
    skipped = [source(cell) for cell in cells() if colab_only(cell)]
    assert len(skipped) == 2
    assert any("pip install" in text for text in skipped)
    assert any("dropbox" in text.lower() for text in skipped)


def test_the_download_asks_dropbox_for_the_file_rather_than_its_web_page():
    download = next(source(cell) for cell in cells() if "DATA_URL" in source(cell))
    url = re.search(r'^DATA_URL = "([^"]+)"$', download, re.MULTILINE)
    assert url is not None, "the download cell should set DATA_URL, uncommented"
    assert url.group(1).startswith("https://www.dropbox.com/") and url.group(1).endswith("dl=1")


def test_your_own_data_is_one_cell_to_uncomment():
    own = [source(cell) for cell in cells() if "/path/to/your/" in source(cell)]
    assert len(own) == 1
    assert "DATASET.ROOT" in own[0]
    assert all(line.startswith("#") for line in own[0].splitlines() if line.strip())


def test_the_data_is_read_from_raw_dicom_rather_than_a_drive_or_prepared_tensors():
    code = "\n".join(source(cell) for cell in cells())
    assert "ECGFileSource.from_dicom" in code and "ImageFileSource.from_dicom" in code
    for absent in ("drive.mount", "google.colab", "torch.load", "_tensor_"):
        assert absent not in code


def test_each_stage_of_the_workflow_has_its_sections_in_order():
    headings = [line for cell in cells("markdown") for line in source(cell).splitlines() if line.startswith("#")]
    expected = [
        "## Step 0: Environment Preparation",
        "### Dataset download",
        "### Configuration",
        "### Your own data",
        "## Step 1: Multimodal Pretraining",
        "### Data loading",
        "### Data preparation",
        "### Multimodal pretraining",
        "## Step 2: Multimodal Fine-tuning",
        "### Data loading",
        "### Data preparation",
        "### Multimodal fine-tuning",
        "## Step 3: Evaluation",
        "## Step 4: Interpretation",
    ]
    remaining = iter(headings)
    missing = [heading for heading in expected if heading not in remaining]
    assert not missing, f"{missing} not found in order among {headings}"


def test_the_notebook_is_stored_without_outputs():
    # Outputs of a run on the tutorial dataset would carry patient-derived images and
    # numbers into the repository.
    for cell in cells():
        assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.slow
def test_the_notebook_runs_on_a_synthetic_dicom_cohort(tmp_path, monkeypatch, dicom_writers):
    pytest.importorskip("captum")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ten seconds at 50 Hz rather than 500, and small radiographs, keep CardioVAE small
    # enough to train on a CPU in seconds.
    for name, labelled in (("kalecardiac_pretraining", False), ("kalecardiac_finetuning", True)):
        dicom_writers.write_synthetic_dicom_cohort(
            tmp_path / "data" / name, num_samples=500, sampling_rate=50, image_size=(64, 64), with_labels=labelled
        )
    monkeypatch.chdir(tmp_path)

    # The tutorial's configurations set neither size, so the defaults' shrunken ones survive
    # the merge.
    published = cardiovae_config.get_cfg_defaults

    def small_defaults():
        cfg = published()
        cfg.ECG.SAMPLING_RATE = 50
        cfg.ECG.LENGTH = 12 * 500
        cfg.IMAGE.SIZE = [32, 32]
        cfg.SOLVER.ACCELERATOR = "cpu"
        return cfg

    monkeypatch.setattr(cardiovae_config, "get_cfg_defaults", small_defaults)

    namespace: dict = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for index, cell in enumerate(cells()):
                if not colab_only(cell):
                    exec(compile(source(cell), f"tutorial cell {index}", "exec"), namespace)
    finally:
        plt.close("all")

    pretrain_set, fold, result = namespace["pretrain_set"], namespace["fold"], namespace["result"]
    assert Path.cwd() == tmp_path, "the setup moved out of a working directory that was not the tutorial's"
    assert len(pretrain_set) == 20
    assert {name: tuple(value.shape) for name, value in pretrain_set[0].modalities.items()} == {
        "ecg": (1, 6000),
        "cxr": (1, 32, 32),
    }
    assert (tmp_path / "outputs" / "cardiovae_tutorial" / "pretraining" / "pretrained.pt").exists()
    # 20% of 20 subjects held out for testing, and 20% of the other 16 for validation.
    assert (len(fold.train_ids), len(fold.val_ids), len(fold.test_ids)) == (13, 3, 4)
    assert {"accuracy", "roc_auc", "mcc", "sensitivity", "specificity"} <= set(result.metrics)
    assert set(namespace["summary"]["attribution"]) == {"ecg", "cxr"}
    # The interpretation produced an importance for every sample of the plotted lead.
    assert namespace["ecg_importance"].shape == namespace["ecg_signal"].shape == (500,)
    assert namespace["cxr_importance"].shape == (32, 32)
