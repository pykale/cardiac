"""The two command lines: what they parse, what they refuse, and that they run.

Every example is documented as ``python -m examples.<name>.main``, so that is what is
run here -- in-process for the parsing, and as a real subprocess once per example, which
is the only way to exercise the module entry point, its logging and its exit code. The
runs use the same tiny synthetic configurations as the workflow tests, so nothing here
needs data or reaches the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.cardiovae_multimodal import main as cardiovae_main
from examples.lsemvae_ecg import main as lsemvae_main
from tests.examples.test_example_workflows import cardiovae_cfg, lsemvae_cfg

REPO = Path(__file__).resolve().parents[2]


def tiny_config(make_cfg, tmp_path: Path) -> Path:
    """Write a tiny synthetic configuration to disk, as a user's ``--cfg`` file would be."""
    path = tmp_path / "tiny.yaml"
    path.write_text(make_cfg(tmp_path).dump(), encoding="utf-8")
    return path


class TestParsing:
    def test_arms_are_split_into_modality_tuples(self):
        assert cardiovae_main.arg_parse(["--arms", "ecg,cxr,ecg+cxr"]).arms == [("ecg",), ("cxr",), ("ecg", "cxr")]

    def test_lead_groups_are_split_and_stripped(self):
        assert lsemvae_main.arg_parse(["--leads", " I , I + II "]).leads == [("I",), ("I", "II")]

    def test_no_flag_leaves_the_choice_to_the_configuration(self):
        assert cardiovae_main.arg_parse([]).arms is None
        assert lsemvae_main.arg_parse([]).leads is None

    @pytest.mark.parametrize("text", ["I,,II", "I,", "I++II", "+"])
    def test_an_empty_lead_name_is_reported_against_the_flag(self, text, capsys):
        with pytest.raises(SystemExit) as exit_info:
            lsemvae_main.arg_parse(["--leads", text])
        assert exit_info.value.code == 2
        assert "argument --leads" in capsys.readouterr().err

    def test_an_empty_arm_is_reported_against_the_flag(self, capsys):
        with pytest.raises(SystemExit):
            cardiovae_main.arg_parse(["--arms", "ecg,"])
        assert "argument --arms" in capsys.readouterr().err

    def test_overrides_follow_the_flags(self):
        args = lsemvae_main.arg_parse(["--cfg", "run.yaml", "--leads", "I", "SOLVER.MAX_EPOCHS", "2"])
        assert args.cfg == "run.yaml"
        assert args.opts == ["SOLVER.MAX_EPOCHS", "2"]

    def test_importing_a_command_line_configures_no_logging(self):
        # basicConfig used to run at import; importing a module must not reconfigure
        # the caller's logging. A fresh interpreter is the only clean place to look.
        code = (
            "import logging; import examples.lsemvae_ecg.main, examples.cardiovae_multimodal.main; "
            "print(len(logging.getLogger().handlers))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "0"


@pytest.mark.slow
class TestRunning:
    def test_cardiovae_runs_from_a_config_file_with_overrides(self, tmp_path, capsys):
        out = tmp_path / "cli"
        results = cardiovae_main.main(
            ["--cfg", str(tiny_config(cardiovae_cfg, tmp_path)), "--arms", "ecg", "OUTPUT.OUT_DIR", str(out)]
        )
        assert list(results) == ["ecg"]
        printed = capsys.readouterr().out
        assert "ecg" in printed and "roc_auc" in printed and f"written to {out}" in printed
        # The override reached the run: the configuration it wrote names the new directory.
        assert str(out) in (out / "config.yaml").read_text(encoding="utf-8")

    def test_lsemvae_runs_from_a_config_file_with_overrides(self, tmp_path, capsys):
        out = tmp_path / "cli"
        results = lsemvae_main.main(
            [
                "--cfg",
                str(tiny_config(lsemvae_cfg, tmp_path)),
                "--leads",
                "I,I+II",
                "OUTPUT.OUT_DIR",
                str(out),
                "ENDPOINT.REGRESSION",
                "True",
            ]
        )
        assert sorted(results) == ["I", "I+II"]
        assert "rmse" in capsys.readouterr().out
        assert json.loads((out / "cohort.json").read_text(encoding="utf-8"))["synthetic"] is True

    @pytest.mark.parametrize("module", ["examples.cardiovae_multimodal.main", "examples.lsemvae_ecg.main"])
    def test_the_documented_module_command_runs_and_says_it_is_synthetic(self, module, tmp_path):
        make_cfg = cardiovae_cfg if "cardiovae" in module else lsemvae_cfg
        out = tmp_path / "subprocess"
        result = subprocess.run(
            [sys.executable, "-m", module, "--cfg", str(tiny_config(make_cfg, tmp_path)), "OUTPUT.OUT_DIR", str(out)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "written to" in result.stdout
        assert (out / "summary.json").exists()
        # A synthetic run is a verification, never a result, and says so before anything else it reports.
        records = [line for line in result.stderr.splitlines() if " INFO " in line or " WARNING " in line]
        assert "it is not a result" in records[0], records[:3]
