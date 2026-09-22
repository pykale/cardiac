"""Configuration for the LS-EMVAE lead-specific ECG example.

Extends the package defaults with what *this experiment* adds: which leads the
pretraining cohort records, which subset the labelled cohort records, the endpoint
column, and the objective weights the paper used.

The defaults reproduce the setup of Suvon et al. (arXiv:2503.13470): pretrain on
twelve leads, fine-tune on the six limb leads a reduced-lead recorder captures. Both
cohorts there are private -- a clinical pulmonary hypertension registry and a biobank
-- so ``DATASET.ROOT`` is empty and a run either points at a local copy or falls back
to the synthetic generator.
"""

from __future__ import annotations

from yacs.config import CfgNode

from kalecardiac.config import get_cfg_defaults as _package_defaults
from kalecardiac.loaddata import LIMB_LEADS, STANDARD_12_LEAD

#: Leads the pretraining cohort records, and the subset the labelled cohort has. The
#: gap between them is the transfer the paper is about: a twelve-lead diagnostic
#: recording is made in a hospital, a six-lead one by a device that costs far less.
PRETRAIN_LEADS = list(STANDARD_12_LEAD)
FINETUNE_LEADS = list(LIMB_LEADS)

#: Endpoints the example can predict, each a threshold on a pressure measured at
#: right-heart catheterisation.
ENDPOINTS = ("ph", "pawp")


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults extended for this experiment."""
    cfg = _package_defaults()

    # ---------------------------------------------------------------- data
    cfg.DATASET.MODALITIES = FINETUNE_LEADS
    cfg.DATASET.SPLIT_MODE = "cv"
    cfg.DATASET.NUM_FOLDS = 5
    cfg.DATASET.STRATIFY_KEYS = ["label"]

    # ---------------------------------------------------------------- ECG
    # Ten seconds at 500 Hz, one encoder per lead.
    cfg.ECG.LEADS = PRETRAIN_LEADS
    cfg.ECG.LENGTH = 5000
    cfg.ECG.SAMPLING_RATE = 500
    cfg.ECG.MODALITY = "lead"
    cfg.ECG.STANDARDISE = True
    cfg.ECG.ENCODER = "conv"

    # Leads the downstream cohort is fine-tuned on. Naming fewer leads here than
    # ECG.LEADS is the twelve-to-six transfer; naming the same is the matched setting.
    cfg.ECG.FINETUNE_LEADS = FINETUNE_LEADS
    # Reconstruct an absent limb lead from I and II rather than dropping it. Exact by
    # Einthoven's and Goldberger's relations, so it costs nothing where those two were
    # recorded.
    cfg.ECG.DERIVE_MISSING = False

    # ---------------------------------------------------------------- model
    cfg.MODEL.NAME = "lsemvae"
    cfg.MODEL.LATENT_DIM = 256
    cfg.MODEL.CHANNELS = [16, 32, 64]
    cfg.MODEL.HEAD_HIDDEN = [128]
    cfg.MODEL.DROPOUT = 0.5
    cfg.MODEL.FREEZE_ENCODERS = True
    # One decoder shared by every lead, which is what forces the latent to carry
    # lead-agnostic cardiac state. False gives each lead its own, the baseline the
    # paper compares against.
    cfg.MODEL.SHARED_DECODER = True

    # ---------------------------------------------------------------- fusion
    # Hierarchical modality experts: a product within groups of leads, a mixture across
    # the groups. "poe" and "moe" are the two published ablations.
    cfg.FUSION.LATENT_METHOD = "hime"
    cfg.FUSION.NUM_GROUPS = 4
    cfg.FUSION.USE_PRIOR = True
    cfg.FUSION.METHOD = "concat"
    cfg.FUSION.FUSION_DIM = 0

    # ---------------------------------------------------------------- objective
    cfg.OBJECTIVE.SCALE_FACTOR = 1e-3
    cfg.OBJECTIVE.ANNEALING_EPOCHS = 50
    # The latent alignment term: pull each lead's posterior mean towards the joint one,
    # so a lead's representation stays readable when the others are absent. This is
    # what makes fine-tuning on a lead subset work.
    cfg.OBJECTIVE.ALIGNMENT_WEIGHT = 0.1
    cfg.OBJECTIVE.UNIMODAL_STREAMS = False

    # ---------------------------------------------------------------- endpoint
    cfg.ENDPOINT = CfgNode()
    cfg.ENDPOINT.NAME = ENDPOINTS[0]
    cfg.ENDPOINT.VALUE_COLUMN = "mpap_mmhg"
    cfg.ENDPOINT.THRESHOLD = 20.0
    cfg.ENDPOINT.UNIT = "mmHg"
    cfg.ENDPOINT.REGRESSION = False

    # ---------------------------------------------------------------- run
    cfg.SOLVER.SEED = 123
    cfg.SOLVER.BASE_LR = 5e-4
    cfg.SOLVER.WEIGHT_DECAY = 1e-5
    cfg.SOLVER.MAX_EPOCHS = 100
    cfg.SOLVER.BATCH_SIZE = 128
    cfg.SOLVER.OPTIMIZER = "AdamW"

    cfg.FINETUNE = CfgNode()
    cfg.FINETUNE.MAX_EPOCHS = 50
    cfg.FINETUNE.BASE_LR = 1e-4
    cfg.FINETUNE.BATCH_SIZE = 32

    # ---------------------------------------------------------------- synthetic
    cfg.SYNTHETIC = CfgNode()
    cfg.SYNTHETIC.NUM_SUBJECTS = 64
    cfg.SYNTHETIC.NUM_SAMPLES = 512

    cfg.OUTPUT.OUT_DIR = "outputs/lsemvae_ecg"
    cfg.OUTPUT.INTERPRET = True
    # Attribution above this normalised threshold counts towards a lead's share, which
    # is how the per-lead importance table is produced.
    cfg.OUTPUT.ATTRIBUTION_THRESHOLD = 0.7
    return cfg
