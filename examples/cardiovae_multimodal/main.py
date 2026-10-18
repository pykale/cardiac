"""CardioVAE: multimodal representation learning from chest X-rays and ECGs.

Reproduces the workflow of Suvon et al., *Multimodal Variational Autoencoder for
Low-cost Cardiac Hemodynamics Instability Detection*, MICCAI 2024, using the reusable
components in ``kalecardiac``: pretrain a product-of-experts multimodal VAE on
unlabelled pairs, then fine-tune its frozen encoders onto an invasively measured
haemodynamic endpoint.

Run it::

    python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/quick.yaml
    python -m examples.cardiovae_multimodal.main --cfg examples/cardiovae_multimodal/configs/full.yaml DATASET.ROOT /path/to/cohort
    python -m examples.cardiovae_multimodal.main --arms ecg,cxr,ecg+cxr
    python -m examples.cardiovae_multimodal.main ENDPOINT.REGRESSION True

With no ``DATASET.ROOT`` the run uses synthetic data and needs nothing else. That
verifies the pipeline executes; it is not a result.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from examples.cardiovae_multimodal.config import get_cfg_defaults
from examples.cardiovae_multimodal.runner import headline_metric, run


def parse_arms(text: str) -> list[tuple[str, ...]]:
    """Read ``--arms``: groups separated by ``,``, the members of a group joined by ``+``.

    Raises:
        argparse.ArgumentTypeError: If a name is empty, as a stray separator makes one,
            so the mistake is reported against the flag rather than deep inside a run.
    """
    groups = [tuple(member.strip() for member in group.split("+")) for group in text.split(",")]
    if not all(member for group in groups for member in group):
        raise argparse.ArgumentTypeError(
            f"{text!r} has an empty name; separate groups with ',' and join a group's members with '+', "
            f"e.g. 'ecg,cxr,ecg+cxr'"
        )
    return groups


def arg_parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description="CardioVAE multimodal cardiac prediction")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--arms",
        default=None,
        type=parse_arms,
        help="comma-separated modality arms to compare, e.g. 'ecg,cxr,ecg+cxr'",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. SOLVER.MAX_EPOCHS 5")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    """Run the example from the command line, print a table, and return the results."""
    # Configured here rather than at import, so importing this module has no side effect.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = arg_parse(argv)
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    results = run(cfg, arms=args.arms)

    print()
    print(f"{'arm':<12} {'metric':<12} mean +/- sd over folds")
    key = headline_metric(cfg)
    for name, summary in results.items():
        entry = summary.get(key, {})
        print(f"{name:<12} {key:<12} {entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}")
    print(f"\nwritten to {cfg.OUTPUT.OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
