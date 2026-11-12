"""Old names fail with what replaced them, in code and in configuration files."""

from __future__ import annotations

import importlib

import pytest

from kalecardiac.config import get_cfg_defaults
from kalecardiac.utils.renames import RENAMED

OLD_NAMES = [(package, old, new) for package, names in RENAMED.items() for old, new in names.items()]


@pytest.mark.parametrize(("package", "old", "new"), OLD_NAMES, ids=[f"{p}.{o}" for p, o, _ in OLD_NAMES])
def test_importing_an_old_name_says_what_replaced_it(package, old, new):
    with pytest.raises(ImportError) as raised:
        exec(f"from {package} import {old}", {})
    message = str(raised.value)
    assert f"{package}.{old}" in message
    if new.startswith("removed:"):
        assert new.removeprefix("removed: ") in message
    else:
        assert f"is now {new}" in message


@pytest.mark.parametrize(
    ("package", "new"),
    [(package, new) for package, _, new in OLD_NAMES if not new.startswith("removed:")],
    ids=[f"{p}.{n}" for p, _, n in OLD_NAMES if not n.startswith("removed:")],
)
def test_every_named_replacement_exists(package, new):
    # A hint pointing at a name that does not exist would send a reader nowhere.
    assert hasattr(importlib.import_module(package), new)


def test_an_unknown_name_is_an_ordinary_attribute_error():
    # So hasattr, and every tool that probes a module, behaves as it would without the hints.
    import kalecardiac.prepdata as prepdata

    assert not hasattr(prepdata, "no_such_function")
    with pytest.raises(AttributeError, match="has no attribute 'no_such_function'"):
        prepdata.no_such_function  # noqa: B018


def test_no_old_name_is_still_exported():
    for package, names in RENAMED.items():
        assert not set(names) & set(importlib.import_module(package).__all__), package


class TestRenamedConfigKey:
    def test_an_old_yaml_file_fails_with_the_new_key(self, tmp_path):
        old = tmp_path / "old.yaml"
        old.write_text("ECG:\n  STANDARDISE: false\n", encoding="utf-8")
        with pytest.raises(KeyError, match="ECG.STANDARDISE was renamed to ECG.NORMALIZE"):
            get_cfg_defaults().merge_from_file(str(old))

    def test_an_old_command_line_override_fails_the_same_way(self):
        with pytest.raises(KeyError, match="ECG.STANDARDISE was renamed to ECG.NORMALIZE"):
            get_cfg_defaults().merge_from_list(["ECG.STANDARDISE", "False"])

    def test_the_new_key_works(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_list(["ECG.NORMALIZE", "False"])
        assert cfg.ECG.NORMALIZE is False
