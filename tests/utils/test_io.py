"""Artefact writing, and reading a checkpoint back through its wrappers."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from kalecardiac.utils import ensure_dir, load_checkpoint, set_seed, write_csv, write_json
from kalecardiac.utils.io import strip_state_dict


class TestEnsureDir:
    def test_creates_a_directory_and_its_parents(self, tmp_path):
        created = ensure_dir(tmp_path / "a" / "b" / "c")
        assert created.is_dir()

    def test_an_existing_directory_is_left_alone(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "keep.txt").write_text("x", encoding="utf-8")
        ensure_dir(tmp_path / "a")
        assert (tmp_path / "a" / "keep.txt").exists()


class TestWriteJson:
    def test_writes_readable_json(self, tmp_path):
        path = write_json(tmp_path / "out" / "metrics.json", {"auc": 0.8})
        assert json.loads(path.read_text(encoding="utf-8")) == {"auc": 0.8}

    def test_numpy_scalars_and_arrays_are_converted(self, tmp_path):
        payload = {"auc": np.float64(0.75), "curve": np.arange(3), "n": np.int64(4)}
        path = write_json(tmp_path / "metrics.json", payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == {"auc": 0.75, "curve": [0, 1, 2], "n": 4}

    def test_an_unserialisable_object_is_rejected_by_type(self, tmp_path):
        with pytest.raises(TypeError, match="not JSON serialisable"):
            write_json(tmp_path / "x.json", {"model": nn.Linear(2, 2)})

    def test_the_file_ends_with_a_newline(self, tmp_path):
        path = write_json(tmp_path / "x.json", {"a": 1})
        assert path.read_text(encoding="utf-8").endswith("\n")


class TestWriteCsv:
    def test_the_header_is_the_union_of_keys_in_first_seen_order(self, tmp_path):
        path = write_csv(tmp_path / "x.csv", [{"b": 1, "a": 2}, {"c": 3}])
        with path.open(encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == ["b", "a", "c"]

    def test_a_row_missing_a_column_is_written_blank(self, tmp_path):
        path = write_csv(tmp_path / "x.csv", [{"a": 1, "b": 2}, {"a": 3}])
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[1]["b"] == ""

    def test_declared_fieldnames_make_an_unexpected_key_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            write_csv(tmp_path / "x.csv", [{"a": 1, "surprise": 2}], fieldnames=["a"])

    def test_no_rows_writes_an_empty_file(self, tmp_path):
        path = write_csv(tmp_path / "x.csv", [])
        assert path.read_text(encoding="utf-8") == ""


class TestStripStateDict:
    def test_a_compiled_prefix_is_removed(self):
        stripped = strip_state_dict({"_orig_mod.layer.weight": torch.zeros(1)})
        assert list(stripped) == ["layer.weight"]

    def test_a_lightning_model_prefix_is_removed(self):
        stripped = strip_state_dict({"model.layer.weight": torch.zeros(1), "model.layer.bias": torch.zeros(1)})
        assert sorted(stripped) == ["layer.bias", "layer.weight"]

    def test_a_partial_model_prefix_is_left_alone(self):
        # Only a state dict where *every* key is prefixed is a wrapped one; otherwise
        # "model" is a genuine submodule name and stripping it would break the load.
        state = {"model.a": torch.zeros(1), "head.b": torch.zeros(1)}
        assert sorted(strip_state_dict(state)) == ["head.b", "model.a"]

    def test_an_ordinary_state_dict_is_unchanged(self):
        state = {"layer.weight": torch.zeros(1)}
        assert strip_state_dict(state) == state


class TestLoadCheckpoint:
    @pytest.fixture
    def model(self):
        return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

    def test_a_bare_state_dict_round_trips(self, model, tmp_path):
        path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), path)
        fresh = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        missing, unexpected = load_checkpoint(path, fresh)
        assert not missing and not unexpected
        for ours, theirs in zip(fresh.parameters(), model.parameters(), strict=True):
            assert torch.equal(ours, theirs)

    def test_a_lightning_checkpoint_is_unwrapped(self, model, tmp_path):
        path = tmp_path / "epoch.ckpt"
        torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}, "epoch": 3}, path)
        fresh = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        missing, unexpected = load_checkpoint(path, fresh)
        assert not missing and not unexpected

    def test_a_compiled_checkpoint_is_unwrapped(self, model, tmp_path):
        path = tmp_path / "compiled.pt"
        torch.save({f"_orig_mod.{k}": v for k, v in model.state_dict().items()}, path)
        fresh = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        assert load_checkpoint(path, fresh) == ([], [])

    def test_a_partial_load_reports_what_did_not_match(self, model, tmp_path):
        path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), path)
        smaller = nn.Sequential(nn.Linear(4, 3))
        missing, unexpected = load_checkpoint(path, smaller, strict=False)
        assert not missing
        assert unexpected == ["2.weight", "2.bias"]

    def test_a_strict_load_of_a_mismatched_model_raises(self, model, tmp_path):
        path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), path)
        with pytest.raises(RuntimeError):
            load_checkpoint(path, nn.Sequential(nn.Linear(4, 3)), strict=True)

    def test_a_missing_file_is_reported_by_path(self, model, tmp_path):
        with pytest.raises(FileNotFoundError, match="no checkpoint at"):
            load_checkpoint(tmp_path / "absent.pt", model)


class TestSetSeed:
    def test_the_same_seed_gives_the_same_draw(self):
        set_seed(11)
        first = torch.randn(5)
        set_seed(11)
        assert torch.equal(first, torch.randn(5))

    def test_it_seeds_numpy_as_well_as_torch(self):
        set_seed(11)
        first = np.random.rand(5)
        set_seed(11)
        assert np.allclose(first, np.random.rand(5))
