"""What both examples' command lines share: the groups they compare, the configuration, the table.

Each example's ``main.py`` keeps its own flag -- ``--arms`` for modality combinations,
``--leads`` for lead groups -- and reads it with :func:`parse_groups`; both then load the
configuration and print the result the same way.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping

from yacs.config import CfgNode

from kalecardiac.auto import get_headline_metric


def parse_groups(text: str, example: str = "a,a+b") -> list[tuple[str, ...]]:
    """Read a comparison flag: groups separated by ``,``, the members of a group joined by ``+``.

    Args:
        text: The flag's value.
        example: A valid value, quoted in the error for an invalid one.

    Raises:
        argparse.ArgumentTypeError: If a name is empty, as a stray separator makes one,
            so the mistake is reported against the flag rather than deep inside a run.
    """
    groups = [tuple(member.strip() for member in group.split("+")) for group in text.split(",")]
    if not all(member for group in groups for member in group):
        raise argparse.ArgumentTypeError(
            f"{text!r} has an empty name; separate groups with ',' and join a group's members with '+', "
            f"e.g. {example!r}"
        )
    return groups


def load_cfg(defaults: Callable[[], CfgNode], args: argparse.Namespace) -> CfgNode:
    """An example's defaults with ``--cfg``'s file merged in, then the command line's overrides, frozen."""
    cfg = defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def print_results(cfg: CfgNode, results: Mapping[str, Mapping], label: str, width: int = 12) -> None:
    """Print each compared group's headline metric over the folds, and where the run was written."""
    key = get_headline_metric(cfg)
    print()
    print(f"{label:<{width}} {'metric':<12} mean +/- sd over folds")
    for name, summary in results.items():
        entry = summary.get(key, {})
        print(
            f"{name:<{width}} {key:<12} {entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}"
        )
    print(f"\nwritten to {cfg.OUTPUT.OUT_DIR}")
