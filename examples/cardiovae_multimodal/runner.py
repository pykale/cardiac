"""Orchestration for the CardioVAE example.

Everything here is an experiment decision: which modality arms are compared, which
model is built from the cohort's shapes, what gets written out. The model, the
objective and the metrics come from ``kalecardiac``, and so does the workflow that
pretrains, fine-tunes over the folds and scores -- :mod:`kalecardiac.auto`.

The workflow the example demonstrates:

.. code-block:: text

    chest X-ray + ECG
          |
    multimodal representation learning   (unlabelled, CardioVAE + tri-stream ELBO)
          |
    fine-tuning                          (labelled, frozen encoders + a clinical head)
          |
    cardiac haemodynamic prediction
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from examples.cardiovae_multimodal.config import CXR, ECG
from examples.cardiovae_multimodal.data import CardioVAECohort, load_cohort, modality_names, summarize_cohort
from kalecardiac.auto import check_endpoint, check_model_name, cross_validate, get_folds, pretrain
from kalecardiac.evaluate import flatten_fold_summary
from kalecardiac.loaddata import MultimodalDataset
from kalecardiac.model.embed import CardioVAE
from kalecardiac.utils import ensure_dir, set_seed, write_csv, write_json

logger = logging.getLogger("cardiovae_multimodal")


#: The model this example builds. ``MODEL.NAME`` must say so, which is what turns a
#: configuration copied from another example into a start-up error rather than a run
#: that quietly trains the wrong thing.
MODEL_NAME = "cardiovae"


def resolve_arms(cfg, arms: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]]:
    """The modality arms to compare, validated before anything is loaded.

    Without ``arms``, ``DATASET.MODALITIES`` decides: each modality alone, then all of
    them fused, which is what makes the multimodal claim testable. One modality gives a
    unimodal run.

    Raises:
        ValueError: If no arm is left, or an arm names a modality this example lacks --
            caught here rather than as a ``KeyError`` deep inside fine-tuning.
    """
    if arms is None:
        names = tuple(cfg.DATASET.MODALITIES)
        arms = [(name,) for name in names] + ([names] if len(names) > 1 else [])
    resolved = [tuple(arm) for arm in arms]
    if not resolved or not all(resolved):
        raise ValueError("no modality to compare; set DATASET.MODALITIES or pass --arms")

    known = set(modality_names())
    unknown = sorted({name for arm in resolved for name in arm} - known)
    if unknown:
        raise ValueError(f"arms name {unknown}, which this example does not have; it has {sorted(known)}")
    return resolved


def build_model(cfg, cohort: CardioVAECohort) -> CardioVAE:
    """Construct the pretraining model from the configuration and the cohort's shapes."""
    return CardioVAE(
        signal_channels=cohort.ecg_channels,
        signal_length=cohort.ecg_length,
        image_channels=cfg.IMAGE.CHANNELS,
        image_size=tuple(cohort.image_size),
        latent_dim=cfg.MODEL.LATENT_DIM,
        channels=list(cfg.MODEL.CHANNELS),
        kernel_size=cfg.MODEL.KERNEL_SIZE,
        signal_encoder=cfg.ECG.ENCODER,
        # The image source scales pixels into the unit interval, so the decoder must be
        # able to produce that range and nothing outside it.
        image_output_activation="sigmoid" if cfg.IMAGE.SCALE else None,
        signal_name=ECG,
        image_name=CXR,
        fusion=cfg.FUSION.LATENT_METHOD,
        use_prior=cfg.FUSION.USE_PRIOR,
        num_groups=cfg.FUSION.NUM_GROUPS,
    )


def run(cfg, arms: Sequence[Sequence[str]] | None = None) -> dict:
    """Pretrain once, then fine-tune and score each modality arm over every fold.

    Args:
        cfg: The frozen configuration.
        arms: Modality combinations to compare. ``None`` derives them from
            ``DATASET.MODALITIES``; see :func:`resolve_arms`.

    Returns:
        Per-arm fold summaries, keyed by the arm's name.
    """
    # Everything that can be checked without data is checked before any is loaded.
    check_model_name(cfg, MODEL_NAME, hint="For a lead-specific model run examples.lsemvae_ecg.")
    check_endpoint(cfg)
    arms = resolve_arms(cfg, arms)

    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    cohort = load_cohort(cfg)
    logger.info("cohort: %s", cohort.describe())
    write_json(out_dir / "cohort.json", summarize_cohort(cohort, cfg.ENDPOINT.NAME))

    pretrained = pretrain(
        cfg,
        build_model(cfg, cohort),
        MultimodalDataset(cohort.pretrain_ids, cohort.sources),
        out_dir,
        # The radiograph is scaled into the unit interval and decoded through a sigmoid,
        # so cross-entropy is its natural reconstruction term; the recording is
        # standardised and real-valued, so squared error is.
        kinds={CXR: "bce"} if cfg.IMAGE.SCALE else {},
    )
    folds = get_folds(cfg, cohort.table)

    results: dict = {}
    rows: list[dict] = []
    for arm in arms:
        name = "+".join(arm)
        sources = {modality: cohort.sources[modality] for modality in arm}
        results[name] = cross_validate(cfg, pretrained, sources, cohort.target, folds, out_dir / name)
        rows.append({"arm": name, **flatten_fold_summary(results[name])})

    write_csv(out_dir / "results.csv", rows)
    write_json(out_dir / "summary.json", results)
    return results
