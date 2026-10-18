"""Configuration: defaults that commit to nothing, and the weight parser."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from yacs.config import CfgNode

from kalecardiac.config import get_cfg_defaults, parse_weights
from kalecardiac.model.embed import FUSION_METHODS, SIGNAL_ENCODERS
from kalecardiac.model.layers import EXPERT_FUSIONS


class TestDefaults:
    def test_returns_an_independent_copy(self):
        first, second = get_cfg_defaults(), get_cfg_defaults()
        first.SOLVER.SEED = 999
        assert second.SOLVER.SEED != 999

    def test_no_default_points_at_anybody_s_disk(self):
        # A clinical cohort is private: there is no path that could be right, so an
        # experiment must be told where its copy lives.
        assert get_cfg_defaults().DATASET.ROOT == ""

    def test_no_default_asserts_a_lead_configuration(self):
        # The library must not decide that an ECG has twelve leads, or six.
        assert get_cfg_defaults().ECG.LEADS == []

    def test_the_named_defaults_are_ones_the_library_implements(self):
        cfg = get_cfg_defaults()
        assert cfg.ECG.ENCODER in SIGNAL_ENCODERS
        assert cfg.FUSION.LATENT_METHOD in EXPERT_FUSIONS
        assert cfg.FUSION.METHOD in FUSION_METHODS

    def test_split_mode_is_one_the_loaddata_stage_supports(self):
        assert get_cfg_defaults().DATASET.SPLIT_MODE in {"predefined", "cv", "random"}

    def test_the_schema_covers_every_stage(self):
        cfg = get_cfg_defaults()
        for section in ("DATASET", "ECG", "IMAGE", "MODEL", "FUSION", "OBJECTIVE", "SOLVER", "OUTPUT"):
            assert isinstance(getattr(cfg, section), CfgNode)

    def test_an_experiment_can_extend_it(self):
        cfg = get_cfg_defaults()
        cfg.ENDPOINT = CfgNode()
        cfg.ENDPOINT.NAME = "example"
        assert cfg.ENDPOINT.NAME == "example"

    def test_it_merges_from_a_list_as_the_command_line_supplies(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_list(["SOLVER.MAX_EPOCHS", "3", "FUSION.METHOD", "attention"])
        assert cfg.SOLVER.MAX_EPOCHS == 3
        assert cfg.FUSION.METHOD == "attention"

    def test_it_round_trips_through_yaml(self, tmp_path):
        cfg = get_cfg_defaults()
        cfg.SOLVER.SEED = 7
        path = tmp_path / "cfg.yaml"
        path.write_text(cfg.dump(), encoding="utf-8")

        reloaded = get_cfg_defaults()
        reloaded.merge_from_file(str(path))
        assert reloaded.SOLVER.SEED == 7

    def test_an_unknown_key_is_rejected(self):
        # A typo in a command-line override is caught rather than silently ignored,
        # which is what keeps a run's configuration honest about what it applied.
        cfg = get_cfg_defaults()
        with pytest.raises(AssertionError, match="Non-existent key"):
            cfg.merge_from_list(["SOLVER.NONEXISTENT", "1"])


class TestParseWeights:
    def test_reads_name_and_weight_pairs(self):
        assert parse_weights(["ecg:10", "cxr:1"]) == {"ecg": 10.0, "cxr": 1.0}

    def test_an_empty_list_gives_no_weights(self):
        assert parse_weights([]) == {}

    def test_a_float_weight_is_kept(self):
        assert parse_weights(["ecg:0.5"]) == {"ecg": 0.5}

    def test_a_lead_name_with_no_separator_is_rejected(self):
        with pytest.raises(ValueError, match="expected 'name:weight'"):
            parse_weights(["ecg"])

    def test_a_non_numeric_weight_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="weight for 'ecg'"):
            parse_weights(["ecg:heavy"])

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="expected 'name:weight'"):
            parse_weights([":10"])


# --------------------------------------------------------------------------- #
# Dead keys
#
# A configuration key nothing reads is worse than a missing one: a user sets it, the run
# completes, and the setting silently had no effect. Six such keys once shipped here --
# one of them documented in an example README as the way to load a checkpoint. These
# tests make that impossible to reintroduce.
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_CONFIG = ROOT / "kalecardiac" / "config.py"
EXAMPLES = ("cardiovae_multimodal", "lsemvae_ecg")

#: ``cfg.SECTION.KEY`` used as a value. An assignment (``cfg.SECTION.KEY = ...``) is not
#: a read, so setting a key in an example's config does not count as using it.
_READ = r"\.{section}\.{key}\b(?!\s*=(?!=))"


def _declared(source: str, prefix: str) -> list[tuple[str, str]]:
    """``(SECTION, KEY)`` pairs assigned as ``<prefix>.SECTION.KEY = ...``."""
    # MULTILINE so that a ``^`` in ``prefix`` anchors at every line, not only the first.
    return sorted(set(re.findall(rf"{prefix}\.([A-Z_]+)\.([A-Z_]+)\s*=(?!=)", source, re.MULTILINE)))


def _is_read(section: str, key: str, text: str) -> bool:
    return re.search(_READ.format(section=section, key=key), text) is not None


class TestNoDeadKeys:
    def test_every_package_key_is_read_somewhere(self):
        keys = _declared(PACKAGE_CONFIG.read_text(encoding="utf-8"), r"^_C")
        assert keys, "found no keys in kalecardiac/config.py -- has the declaration style changed?"

        readers = [
            path
            for root in (ROOT / "kalecardiac", ROOT / "examples")
            for path in root.rglob("*.py")
            if path.resolve() != PACKAGE_CONFIG
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in readers)
        dead = [f"{section}.{key}" for section, key in keys if not _is_read(section, key, text)]
        assert not dead, (
            f"{dead} are declared in kalecardiac/config.py but nothing reads them, so setting them silently "
            f"does nothing. Wire each into the code it configures, or remove it from the schema."
        )

    @pytest.mark.parametrize("example", EXAMPLES)
    def test_every_key_an_example_sets_is_read_by_that_example(self, example):
        folder = ROOT / "examples" / example
        keys = _declared((folder / "config.py").read_text(encoding="utf-8"), r"\bcfg")
        text = "\n".join(path.read_text(encoding="utf-8") for path in folder.glob("*.py") if path.name != "config.py")
        dead = [f"{section}.{key}" for section, key in keys if not _is_read(section, key, text)]
        assert not dead, (
            f"examples/{example}/config.py sets {dead}, but nothing in that example reads them. A user editing "
            f"them would see no effect. Read each where it applies, or stop setting it."
        )

    @pytest.mark.parametrize("yaml_path", sorted((ROOT / "examples").rglob("*.yaml")), ids=lambda path: path.name)
    def test_every_shipped_yaml_merges_into_its_example(self, yaml_path):
        # A key removed from the schema but left in a shipped YAML would break that
        # configuration only when someone ran it, which is too late.
        example = yaml_path.parent.parent.name
        module = __import__(f"examples.{example}.config", fromlist=["get_cfg_defaults"])
        module.get_cfg_defaults().merge_from_file(str(yaml_path))

    def test_the_guard_catches_a_dead_key(self):
        # The guard itself must not be vacuous.
        source = "_C.SECTION = CfgNode()\n_C.SECTION.LIVE = 1\n_C.SECTION.DEAD = 2\n"
        keys = _declared(source, r"^_C")
        assert keys == [("SECTION", "DEAD"), ("SECTION", "LIVE")]
        uses = "x = cfg.SECTION.LIVE\ncfg.SECTION.DEAD = 3\n"
        assert _is_read("SECTION", "LIVE", uses)
        assert not _is_read("SECTION", "DEAD", uses), "an assignment must not count as a read"
