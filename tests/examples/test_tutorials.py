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
import yaml

from kalecardiac.config import get_cfg_defaults

TUTORIAL = Path(__file__).resolve().parents[2] / "tutorials" / "mdas_tutorial"
NOTEBOOK = TUTORIAL / "tutorial-CardioVAE.ipynb"
CONFIGS = TUTORIAL / "configs"


def cells(kind: str = "code") -> list[dict]:
    """The notebook's cells of one kind, in order."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == kind]


def source(cell: dict) -> str:
    return "".join(cell["source"])


def code() -> str:
    return "\n".join(source(cell) for cell in cells())


def colab_only(cell: dict) -> bool:
    return "skip-execution" in cell["metadata"].get("tags", [])


def test_the_configurations_beside_the_notebook_merge_onto_kalecardiacs_defaults():
    # Colab installs the package and fetches this folder alone, so the configurations can
    # lean on nothing but the package's own schema; fine-tuning starts from pretraining's.
    cfg_pt = get_cfg_defaults()
    cfg_pt.merge_from_file(str(CONFIGS / "config_pretrain.yaml"))
    cfg_ft = cfg_pt.clone()
    cfg_ft.merge_from_file(str(CONFIGS / "config_finetune.yaml"))
    assert (cfg_pt.ECG.LENGTH, cfg_pt.ECG.NUM_CHANNELS, list(cfg_pt.IMAGE.SIZE)) == (60000, 1, [224, 224])
    assert (cfg_ft.ENDPOINT.LABEL_COLUMN, cfg_ft.ECG.LENGTH) == ("label", cfg_pt.ECG.LENGTH)
    assert "configs/config_pretrain.yaml" in code() and "configs/config_finetune.yaml" in code()


def test_colab_installs_the_package_and_fetches_this_folder_alone():
    install = next(source(cell) for cell in cells() if "pip install" in source(cell))
    assert "git+https://github.com/pykale/cardiac" in install
    assert "sparse-checkout set tutorials/mdas_tutorial" in install
    # A failed install must show its error, rather than surface later as a missing module.
    assert "%%capture" not in install


def test_the_notebook_imports_only_what_installing_kalecardiac_provides():
    # Neither ships with Colab, nor is installed by the package: lightning_fabric is
    # Lightning's internals, and examples/ is not part of the package.
    assert not re.search(r"^\s*(import|from) (lightning_fabric|examples)\b", code(), re.MULTILINE)


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
    assert "ECGFileSource.from_dicom" in code() and "ImageFileSource.from_dicom" in code()
    for absent in ("drive.mount", "google.colab", "torch.load", "_tensor_"):
        assert absent not in code()


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
    # The tutorial's folder as Colab has it after the install cell, its configurations shrunk.
    (tmp_path / "configs").mkdir()
    for name in ("config_pretrain.yaml", "config_finetune.yaml"):
        settings = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
        if name == "config_pretrain.yaml":
            settings["ECG"].update(SAMPLING_RATE=50, LENGTH=12 * 500)
            settings["IMAGE"]["SIZE"] = [32, 32]
            settings["SOLVER"]["ACCELERATOR"] = "cpu"
        (tmp_path / "configs" / name).write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    namespace: dict = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for index, cell in enumerate(cells()):
                if not colab_only(cell):
                    exec(compile(source(cell), f"tutorial cell {index}", "exec"), namespace)
    finally:
        plt.close("all")

    pretrain_set, fold, result = namespace["pretrain_set"], namespace["fold"], namespace["result"]
    assert len(pretrain_set) == 20
    assert {name: tuple(value.shape) for name, value in pretrain_set[0].modalities.items()} == {
        "ecg": (1, 6000),
        "cxr": (1, 32, 32),
    }
    assert (tmp_path / "outputs" / "pretraining" / "pretrained.pt").exists()
    # 20% of 20 subjects held out for testing, and 20% of the other 16 for validation.
    assert (len(fold.train_ids), len(fold.val_ids), len(fold.test_ids)) == (13, 3, 4)
    assert {"accuracy", "roc_auc", "mcc", "sensitivity", "specificity"} <= set(result.metrics)
    assert set(namespace["summary"]["attribution"]) == {"ecg", "cxr"}
    # The interpretation produced an importance for every sample of the plotted lead.
    assert namespace["ecg_importance"].shape == namespace["ecg_signal"].shape == (500,)
    assert namespace["cxr_importance"].shape == (32, 32)
