"""LS-EMVAE: lead-specific representation learning from multi-lead ECG.

Reproduces the workflow of Suvon et al., *Multimodal Latent Fusion of ECG Leads for
Early Assessment of Pulmonary Hypertension* (arXiv:2503.13470), using the reusable
components in ``kalecardiac``: treat each lead as its own modality, pretrain a
hierarchical-expert multimodal VAE on unlabelled twelve-lead recordings, then fine-tune
the frozen lead encoders on a small labelled cohort that records only six.

Run it::

    python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/quick.yaml
    python -m examples.lsemvae_ecg.main --cfg examples/lsemvae_ecg/configs/full.yaml DATASET.ROOT /path/to/cohort
    python -m examples.lsemvae_ecg.main --leads I,II,III+I+II FUSION.LATENT_METHOD poe
    python -m examples.lsemvae_ecg.main ENDPOINT.REGRESSION True

With no ``DATASET.ROOT`` the run uses synthetic data and needs nothing else. That
verifies the pipeline executes; it is not a result.

The two published ablations are configuration changes rather than other models::

    FUSION.LATENT_METHOD poe          # no mixture step
    FUSION.LATENT_METHOD moe          # no product step
    OBJECTIVE.ALIGNMENT_WEIGHT 0.0    # no latent alignment
    MODEL.SHARED_DECODER False        # a decoder per lead
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from examples.lsemvae_ecg.config import get_cfg_defaults
from examples.lsemvae_ecg.runner import headline_metric, run


def parse_lead_groups(text: str) -> list[tuple[str, ...]]:
    """Read ``--leads``: groups separated by ``,``, the members of a group joined by ``+``.

    Raises:
        argparse.ArgumentTypeError: If a name is empty, as a stray separator makes one,
            so the mistake is reported against the flag rather than deep inside a run.
    """
    groups = [tuple(member.strip() for member in group.split("+")) for group in text.split(",")]
    if not all(member for group in groups for member in group):
        raise argparse.ArgumentTypeError(
            f"{text!r} has an empty name; separate groups with ',' and join a group's members with '+', "
            f"e.g. 'I,I+II,I+II+III'"
        )
    return groups


def arg_parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description="LS-EMVAE lead-specific ECG prediction")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--leads",
        default=None,
        type=parse_lead_groups,
        help="comma-separated lead groups to compare, each joined by '+', e.g. 'I,I+II,I+II+III'",
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

    results = run(cfg, lead_groups=args.leads)

    print()
    print(f"{'leads':<28} {'metric':<12} mean +/- sd over folds")
    key = headline_metric(cfg)
    for name, summary in results.items():
        entry = summary.get(key, {})
        print(f"{name:<28} {key:<12} {entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}")
    print(f"\nwritten to {cfg.OUTPUT.OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
