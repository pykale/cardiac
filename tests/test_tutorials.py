"""The tutorial notebooks run top to bottom on raw DICOM files, and are stored without outputs.

A notebook is documentation that executes, and documentation drifts: an API change a
notebook missed would surface only when a reader ran it. This runs every code cell in
order, in one namespace, as a kernel would -- skipping only the cells tagged for Colab,
which install the package and download the dataset. The dataset is generated instead: a
synthetic cohort written as DICOM in the tutorials' layout, so nothing is downloaded and
nothing private is needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from kalecardiac.config import get_cfg_defaults
from kalecardiac.loaddata import LIMB_LEADS, STANDARD_12_LEAD

TUTORIAL = Path(__file__).resolve().parents[1] / "tutorials" / "mdas_tutorial"
CONFIGS = TUTORIAL / "configs"
CARDIOVAE = TUTORIAL / "tutorial-CardioVAE.ipynb"
LSEMVAE = TUTORIAL / "tutorial-LSEMVAE.ipynb"
# Each notebook's pretraining and fine-tuning configurations, beside it in configs/.
CONFIGURATIONS = {
    CARDIOVAE: ("config_cardiovae_pretrain.yaml", "config_cardiovae_finetune.yaml"),
    LSEMVAE: ("config_lsemvae_pretrain.yaml", "config_lsemvae_finetune.yaml"),
}
each_notebook = pytest.mark.parametrize("notebook", list(CONFIGURATIONS), ids=lambda path: path.stem)


def cells(notebook: Path, kind: str = "code") -> list[dict]:
    """A notebook's cells of one kind, in order."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    return [cell for cell in content["cells"] if cell["cell_type"] == kind]


def source(cell: dict) -> str:
    return "".join(cell["source"])


def code(notebook: Path) -> str:
    return "\n".join(source(cell) for cell in cells(notebook))


def colab_only(cell: dict) -> bool:
    return "skip-execution" in cell["metadata"].get("tags", [])


def configurations(notebook: Path):
    """A notebook's two configurations, merged as it merges them: fine-tuning onto pretraining's."""
    pretrain, finetune = CONFIGURATIONS[notebook]
    cfg_pt = get_cfg_defaults()
    cfg_pt.merge_from_file(str(CONFIGS / pretrain))
    cfg_ft = cfg_pt.clone()
    cfg_ft.merge_from_file(str(CONFIGS / finetune))
    return cfg_pt, cfg_ft


@each_notebook
def test_the_configurations_beside_the_notebook_merge_onto_kalecardiacs_defaults(notebook):
    # Colab installs the package and fetches this folder alone, so the configurations can
    # lean on nothing but the package's own schema; fine-tuning starts from pretraining's.
    cfg_pt, cfg_ft = configurations(notebook)
    assert (cfg_ft.ENDPOINT.LABEL_COLUMN, cfg_ft.ECG.LENGTH) == ("label", cfg_pt.ECG.LENGTH)
    for name in CONFIGURATIONS[notebook]:
        assert f"configs/{name}" in code(notebook)


def test_cardiovae_reads_an_interleaved_strip_and_a_radiograph():
    cfg_pt, _ = configurations(CARDIOVAE)
    assert (cfg_pt.ECG.LENGTH, cfg_pt.ECG.NUM_CHANNELS, list(cfg_pt.IMAGE.SIZE)) == (60000, 1, [224, 224])


def test_lsemvae_is_fine_tuned_on_the_limb_leads_of_the_twelve_it_was_pretrained_on():
    cfg_pt, cfg_ft = configurations(LSEMVAE)
    # A channel per lead, so each recording splits into its leads.
    assert (cfg_pt.ECG.LENGTH, cfg_pt.ECG.NUM_CHANNELS) == (5000, 0)
    assert (tuple(cfg_pt.ECG.LEADS), tuple(cfg_ft.ECG.LEADS)) == (STANDARD_12_LEAD, LIMB_LEADS)


def test_the_tutorials_share_a_folder_but_not_their_outputs():
    # Colab fetches one folder for both, so neither may write over what the other wrote.
    folders = [cfg.OUTPUT.OUT_DIR for notebook in CONFIGURATIONS for cfg in configurations(notebook)]
    assert len(set(folders)) == len(folders)


@each_notebook
def test_colab_installs_the_package_and_fetches_this_folder_alone(notebook):
    install = next(source(cell) for cell in cells(notebook) if "pip install" in source(cell))
    assert "git+https://github.com/pykale/cardiac" in install
    assert "sparse-checkout set tutorials/mdas_tutorial" in install
    # A failed install must show its error, rather than surface later as a missing module.
    assert "%%capture" not in install


def test_the_tutorials_prepare_colab_alike():
    # Installing, setting up and downloading are one procedure for both, so a fix to one is a fix to both.
    def preparation(notebook: Path) -> list[str]:
        return [
            source(cell) for cell in cells(notebook) if colab_only(cell) or "import pytorch_lightning" in source(cell)
        ]

    assert preparation(LSEMVAE) == preparation(CARDIOVAE)


@each_notebook
def test_the_notebook_imports_only_what_installing_kalecardiac_provides(notebook):
    # Neither ships with Colab, nor is installed by the package: lightning_fabric is
    # Lightning's internals, and examples/ is not part of the package.
    assert not re.search(r"^\s*(import|from) (lightning_fabric|examples)\b", code(notebook), re.MULTILINE)


@each_notebook
def test_only_the_install_and_the_download_are_skipped(notebook):
    # The download reaches the network, which nothing in the test suite may.
    skipped = [source(cell) for cell in cells(notebook) if colab_only(cell)]
    assert len(skipped) == 2
    assert any("pip install" in text for text in skipped)
    assert any("dropbox" in text.lower() for text in skipped)


@each_notebook
def test_the_download_asks_dropbox_for_the_file_rather_than_its_web_page(notebook):
    download = next(source(cell) for cell in cells(notebook) if "DATA_URL" in source(cell))
    url = re.search(r'^DATA_URL = "([^"]+)"$', download, re.MULTILINE)
    assert url is not None, "the download cell should set DATA_URL, uncommented"
    assert url.group(1).startswith("https://www.dropbox.com/") and url.group(1).endswith("dl=1")


@each_notebook
def test_your_own_data_is_one_cell_to_uncomment(notebook):
    own = [source(cell) for cell in cells(notebook) if "/path/to/your/" in source(cell)]
    assert len(own) == 1
    assert "DATASET.ROOT" in own[0]
    assert all(line.startswith("#") for line in own[0].splitlines() if line.strip())


@pytest.mark.parametrize(
    ("notebook", "readers"),
    [
        (CARDIOVAE, ("ECGFileSource.from_dicom", "ImageFileSource.from_dicom")),
        (LSEMVAE, ("ECGFileSource.from_dicom",)),
    ],
    ids=[CARDIOVAE.stem, LSEMVAE.stem],
)
def test_the_data_is_read_from_raw_dicom_rather_than_a_drive_or_prepared_tensors(notebook, readers):
    for reader in readers:
        assert reader in code(notebook)
    for absent in ("drive.mount", "google.colab", "torch.load", "_tensor_"):
        assert absent not in code(notebook)


@each_notebook
def test_each_stage_of_the_workflow_has_its_sections_in_order(notebook):
    headings = [
        line for cell in cells(notebook, "markdown") for line in source(cell).splitlines() if line.startswith("#")
    ]
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


@each_notebook
def test_the_notebook_is_stored_without_outputs(notebook):
    # Outputs of a run on the tutorial dataset would carry patient-derived images and
    # numbers into the repository.
    for cell in cells(notebook):
        assert cell["outputs"] == [] and cell["execution_count"] is None


def run_notebook(
    notebook: Path,
    tmp_path: Path,
    monkeypatch,
    dicom_writers,
    shrink: Callable[[str, dict], None],
    edit: Callable[[str], str] | None = None,
):
    """Run a notebook's cells, but Colab's, on a synthetic DICOM cohort.

    ``shrink`` changes each configuration file's settings; ``edit``, if given, each cell's
    source, as a reader changing a setting in the notebook would.
    """
    pytest.importorskip("captum")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ten seconds at 50 Hz rather than 500, and small radiographs, keep the models small
    # enough to train on a CPU in seconds.
    for name, labelled in (("kalecardiac_pretraining", False), ("kalecardiac_finetuning", True)):
        dicom_writers.write_synthetic_dicom_cohort(
            tmp_path / "data" / name, num_samples=500, sampling_rate=50, image_size=(64, 64), with_labels=labelled
        )
    # The tutorial's folder as Colab has it after the install cell, its configurations shrunk.
    (tmp_path / "configs").mkdir()
    for name in CONFIGURATIONS[notebook]:
        settings = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
        shrink(name, settings)
        (tmp_path / "configs" / name).write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    namespace: dict = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for index, cell in enumerate(cells(notebook)):
                if not colab_only(cell):
                    text = source(cell) if edit is None else edit(source(cell))
                    exec(compile(text, f"{notebook.stem} cell {index}", "exec"), namespace)
    finally:
        plt.close("all")
    return namespace


@pytest.mark.slow
def test_the_cardiovae_notebook_runs_on_a_synthetic_dicom_cohort(tmp_path, monkeypatch, dicom_writers):
    def shrink(name: str, settings: dict) -> None:
        if name == "config_cardiovae_pretrain.yaml":
            settings["ECG"].update(SAMPLING_RATE=50, LENGTH=12 * 500)
            settings["IMAGE"]["SIZE"] = [32, 32]
            settings["SOLVER"]["ACCELERATOR"] = "cpu"

    namespace = run_notebook(CARDIOVAE, tmp_path, monkeypatch, dicom_writers, shrink)

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


@pytest.mark.slow
@pytest.mark.parametrize("twelve_leads", [False, True], ids=["6L-ECG", "12L-ECG"])
def test_the_lsemvae_notebook_runs_on_a_synthetic_dicom_cohort(tmp_path, monkeypatch, dicom_writers, twelve_leads):
    def shrink(name: str, settings: dict) -> None:
        if name == "config_lsemvae_pretrain.yaml":
            settings["ECG"].update(SAMPLING_RATE=50, LENGTH=500)
            settings["SOLVER"]["ACCELERATOR"] = "cpu"

    # The notebook's switch between fine-tuning on the six limb leads and on all twelve.
    switch = "USE_12_LEADS = False"
    assert code(LSEMVAE).count(switch) == 1

    def choose_leads(text: str) -> str:
        return text.replace(switch, f"USE_12_LEADS = {twelve_leads}")

    namespace = run_notebook(LSEMVAE, tmp_path, monkeypatch, dicom_writers, shrink, edit=choose_leads)
    leads = STANDARD_12_LEAD if twelve_leads else LIMB_LEADS

    pretrain_set, fold, result = namespace["pretrain_set"], namespace["fold"], namespace["result"]
    assert len(pretrain_set) == 20
    # Pretrained on all twelve leads, each a modality of its own, fused by HiME.
    assert {name: tuple(value.shape) for name, value in pretrain_set[0].modalities.items()} == dict.fromkeys(
        STANDARD_12_LEAD, (1, 500)
    )
    assert (list(namespace["model"].modalities), namespace["model"].fusion_name) == (list(STANDARD_12_LEAD), "hime")
    assert (tmp_path / "outputs" / "lsemvae_pretraining" / "pretrained.pt").exists()
    assert set(namespace["distance"].index) == set(STANDARD_12_LEAD)
    # Fine-tuned on the chosen leads, scored again with each lead fewer down to lead I, and explained per lead.
    assert list(namespace["finetune_leads"]) == list(leads)
    assert (len(fold.train_ids), len(fold.val_ids), len(fold.test_ids)) == (13, 3, 4)
    assert {"accuracy", "roc_auc", "mcc", "sensitivity", "specificity"} <= set(result.metrics)
    assert [step["present"] for step in namespace["steps"]][-1] == ["I"]
    assert [len(step["present"]) for step in namespace["steps"]] == list(range(len(leads), 0, -1))
    assert set(namespace["summary"]["attribution"]) == set(leads)
    assert namespace["ecg_importance"].shape == namespace["ecg_signals"].shape == (len(leads), 500)
