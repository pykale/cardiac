"""Configuration: defaults that commit to nothing, and the weight parser."""

from __future__ import annotations

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
