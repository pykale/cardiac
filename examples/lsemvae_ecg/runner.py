"""Orchestration for the LS-EMVAE example.

Everything here is an experiment decision: which lead groups are compared, the order
their leads are withheld in, what is written out. The model, the hierarchical fusion,
the alignment objective and the metrics come from ``kalecardiac``, and so does the
workflow that pretrains, fine-tunes over the folds and scores --
:mod:`kalecardiac.auto`.

The workflow the example demonstrates:

.. code-block:: text

    twelve-lead ECG, unlabelled
          |
    one encoder per lead -> hierarchical latent fusion   (LS-EMVAE pretraining)
          |
    fine-tuning on the six limb leads a low-cost recorder captures
          |
    cardiovascular prediction
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from examples.lsemvae_ecg.data import LeadCohort, load_cohort, summarize_cohort
from kalecardiac.auto import (
    Fold,
    FoldResult,
    check_endpoint,
    check_model_name,
    cross_validate,
    get_ablation_metric,
    get_dataloader,
    get_folds,
    pretrain,
)
from kalecardiac.evaluate import MetricError, flatten_fold_summary
from kalecardiac.interpret import collect_latents, progressive_ablation, project_latents
from kalecardiac.loaddata import MultimodalDataset, normalize_lead_name, normalize_lead_names, release_workers
from kalecardiac.model.embed import LSEMVAE
from kalecardiac.pipeline import CardiacTrainer
from kalecardiac.utils import ensure_dir, preserve_rng_state, set_seed, write_csv, write_json

logger = logging.getLogger("lsemvae_ecg")


#: The model this example builds. ``MODEL.NAME`` must say so, which is what turns a
#: configuration copied from another example into a start-up error rather than a run
#: that quietly trains the wrong thing.
MODEL_NAME = "lsemvae"


def resolve_lead_groups(cfg, lead_groups: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]]:
    """The lead groups to compare, canonicalised and validated before anything is loaded.

    Checked here rather than in the fold loop, because there it was only reached after
    pretraining had already run: a misspelt lead cost a whole pretraining run before it
    failed. Names are canonicalised as everywhere else, so ``avf`` is ``aVF``.

    Raises:
        ValueError: If no group is left, or a group names a lead the labelled cohort
            does not record.
    """
    available = normalize_lead_names(list(cfg.ECG.FINETUNE_LEADS))
    if lead_groups is None:
        groups = [available]
    else:
        # Emptiness is checked before canonicalising, whose own error is about declaring
        # a source's leads and would not say what is actually wrong with a --leads group.
        requested = [tuple(group) for group in lead_groups]
        if not requested or not all(requested):
            raise ValueError("no lead group to compare; pass --leads with at least one lead per group")
        groups = [normalize_lead_names(group) for group in requested]

    unknown = sorted({lead for group in groups for lead in group} - set(available))
    if unknown:
        raise ValueError(
            f"lead group names {unknown}, which the labelled cohort does not record; ECG.FINETUNE_LEADS is "
            f"{list(available)}"
        )
    return [tuple(group) for group in groups]


def resolve_robustness_order(cfg) -> tuple[str, ...]:
    """``ROBUSTNESS.ORDER``, canonicalised and validated before anything is loaded.

    Raises:
        ValueError: If it names a lead twice, or one the labelled cohort does not record.
    """
    order = tuple(normalize_lead_name(lead) for lead in cfg.ROBUSTNESS.ORDER)
    repeated = sorted({lead for lead in order if order.count(lead) > 1})
    if repeated:
        raise ValueError(f"ROBUSTNESS.ORDER names {repeated} more than once; each lead is withheld once")
    available = normalize_lead_names(list(cfg.ECG.FINETUNE_LEADS))
    unknown = sorted(set(order) - set(available))
    if unknown:
        raise ValueError(
            f"ROBUSTNESS.ORDER names {unknown}, which the labelled cohort does not record; ECG.FINETUNE_LEADS is "
            f"{list(available)}"
        )
    return order


def withholding_order(priority: Sequence[str], group: Sequence[str]) -> list[str]:
    """The order a group's leads are withheld in, leaving its last lead present.

    ``priority`` first, as far as the group has those leads, then the rest from the end
    of the group backwards -- so a group in the conventional order is reduced towards
    lead I, the lead a single-lead wearable records.
    """
    ordered = [lead for lead in priority if lead in group]
    ordered += [lead for lead in reversed(group) if lead not in ordered]
    return ordered[: len(group) - 1]


def build_model(cfg, cohort: LeadCohort, leads: Sequence[str]) -> LSEMVAE:
    """Construct a lead-specific VAE over ``leads``."""
    return LSEMVAE(
        leads=list(leads),
        length=cohort.length,
        latent_dim=cfg.MODEL.LATENT_DIM,
        channels=list(cfg.MODEL.CHANNELS),
        kernel_size=cfg.MODEL.KERNEL_SIZE,
        encoder=cfg.ECG.ENCODER,
        fusion=cfg.FUSION.LATENT_METHOD,
        num_groups=cfg.FUSION.NUM_GROUPS,
        shared_decoder=cfg.MODEL.SHARED_DECODER,
        use_prior=cfg.FUSION.USE_PRIOR,
    )


def robustness(
    cfg, model: CardiacTrainer, test_loader: DataLoader, group: Sequence[str], fold: int, priority: Sequence[str]
) -> list[dict]:
    """How the fine-tuned model degrades as its leads are withheld at inference.

    The deployment question the lead-specific design exists to answer: a model
    fine-tuned on a lead group, shown one lead fewer at each step until one is left.
    Each lead has its own encoder and an absent lead is an absent expert, so the model
    has a principled answer at every step; a channel-stacked model would have none.

    Returns:
        One row per step, for ``robustness.csv``.
    """
    name, metric, target = get_ablation_metric(cfg)
    steps = progressive_ablation(model, test_loader, metric, target, withholding_order(priority, group))
    return [
        {
            "fold": fold,
            "step": step["step"],
            "n_leads": len(step["present"]),
            "leads": "+".join(step["present"]),
            "withheld": "+".join(step["withheld"]),
            name: step["score"],
        }
        for step in steps
    ]


def record_robustness(
    cfg, group: Sequence[str], priority: Sequence[str], curve: list[dict], fold: Fold, result: FoldResult
) -> None:
    """Add one fold's robustness curve to ``curve``: the report each fold of a group runs."""
    try:
        curve.extend(robustness(cfg, result.model, result.test_loader, group, fold.index, priority))
    except MetricError as error:
        logger.warning("skipping robustness for %s fold %d: %s", "+".join(group), fold.index, error)


def export_latents(cfg, pretrained: LSEMVAE, cohort: LeadCohort, out_dir: Path) -> None:
    """Project the pretrained latent space, and record how far the leads agree.

    The only check available before any label: whether the representation has structure,
    and whether the lead encoders ended up in the same coordinates.
    """
    data_loader = get_dataloader(
        cfg, MultimodalDataset(cohort.pretrain_ids, cohort.pretrain_sources), cfg.SOLVER.BATCH_SIZE
    )
    identifiers, joint, per_lead = collect_latents(pretrained, data_loader, per_modality=True)
    release_workers(data_loader)

    coordinates = project_latents(joint, method="pca", random_state=cfg.SOLVER.SEED)
    write_csv(
        out_dir / "latent_space.csv",
        [
            {"subject_id": identifier, "x": float(x), "y": float(y)}
            for identifier, (x, y) in zip(identifiers, coordinates, strict=True)
        ],
    )
    write_json(
        out_dir / "lead_agreement.json",
        {lead: float(np.mean((values - joint) ** 2)) for lead, values in per_lead.items()},
    )


def run(cfg, lead_groups: Sequence[Sequence[str]] | None = None) -> dict:
    """Pretrain on the full lead set, then fine-tune and score each lead group.

    Args:
        cfg: The frozen configuration.
        lead_groups: Lead subsets to compare. ``None`` evaluates the configured
            fine-tuning set alone, which is the headline result; passing several makes
            the lead-configuration comparison the paper reports.

    Returns:
        Per-group fold summaries, keyed by the group's name.
    """
    # Everything that can be checked without data is checked before any is loaded.
    check_model_name(cfg, MODEL_NAME, hint="For a signal-and-image model run examples.cardiovae_multimodal.")
    check_endpoint(cfg)
    groups = resolve_lead_groups(cfg, lead_groups)
    priority = resolve_robustness_order(cfg)

    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    cohort = load_cohort(cfg)
    logger.info("cohort: %s", cohort.describe())
    write_json(out_dir / "cohort.json", summarize_cohort(cohort, cfg.ENDPOINT.NAME))

    # Built over every pretraining lead, which is what the representation -- and any
    # MODEL.PRETRAINED checkpoint -- holds; each group then fine-tunes on a subset of
    # them, so a twelve-lead representation serves a six-lead run unchanged.
    pretrained = pretrain(
        cfg,
        build_model(cfg, cohort, cohort.pretrain_leads),
        MultimodalDataset(cohort.pretrain_ids, cohort.pretrain_sources),
        out_dir,
    )
    # Reporting is kept off the training's random stream, so turning a report on or
    # off cannot change the models trained after it.
    if cfg.OUTPUT.INTERPRET:
        with preserve_rng_state():
            export_latents(cfg, pretrained, cohort, out_dir)
    folds = get_folds(cfg, cohort.table)

    results: dict = {}
    rows: list[dict] = []
    for group in groups:
        name = "+".join(group)
        curve: list[dict] = []
        report = partial(record_robustness, cfg, group, priority, curve)
        results[name] = cross_validate(
            cfg,
            pretrained,
            {lead: cohort.finetune_sources[lead] for lead in group},
            cohort.target,
            folds,
            out_dir / name,
            report=report if cfg.ROBUSTNESS.ENABLED and len(group) > 1 else None,
        )
        if curve:
            write_csv(out_dir / name / "robustness.csv", curve)
            logger.info("%s with leads withheld: %s", name, _curve_line(curve, get_ablation_metric(cfg)[0]))
        rows.append({"leads": name, "n_leads": len(group), **flatten_fold_summary(results[name])})

    write_csv(out_dir / "results.csv", rows)
    write_json(out_dir / "summary.json", results)
    return results


def _curve_line(curve: Sequence[Mapping], key: str) -> str:
    """The robustness curve as one line: the fold mean at each number of leads."""
    by_count: dict[int, list[float]] = {}
    for row in curve:
        by_count.setdefault(int(row["n_leads"]), []).append(float(row[key]))
    return " | ".join(
        f"{count} lead{'s' if count > 1 else ''} {key}={np.mean(values):.3f}"
        for count, values in sorted(by_count.items(), reverse=True)
    )
