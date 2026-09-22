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

from examples.cardiovae_multimodal.config import get_cfg_defaults
from examples.cardiovae_multimodal.runner import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioVAE multimodal cardiac prediction")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated modality arms to compare, e.g. 'ecg,cxr,ecg+cxr'",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. SOLVER.MAX_EPOCHS 5")
    return parser.parse_args()


def main() -> None:
    args = arg_parse()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    arms = [tuple(arm.split("+")) for arm in args.arms.split(",")] if args.arms else None
    results = run(cfg, arms=arms)

    print()
    print(f"{'arm':<12} {'metric':<10} mean +/- sd over folds")
    key = "rmse" if cfg.ENDPOINT.REGRESSION else "roc_auc"
    for name, summary in results.items():
        entry = summary.get(key, {})
        print(f"{name:<12} {key:<10} {entry.get('mean', float('nan')):.3f} +/- {entry.get('std', float('nan')):.3f}")
    print(f"\nwritten to {cfg.OUTPUT.OUT_DIR}")


if __name__ == "__main__":
    main()
