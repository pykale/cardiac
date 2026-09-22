"""Configuration schema for KaleCardiac pipelines.

Defaults are dataset-agnostic: nothing here names a cohort, an endpoint, a lead
configuration a particular registry happens to record, or a path on anyone's disk.
Those are supplied by an experiment configuration under ``examples/``, by
command-line overrides, or by the Python API.

The schema is deliberately small. A setting earns its place here only if more than
one experiment would set it; anything one dataset needs belongs to that dataset's own
configuration, which extends this one.
"""

from __future__ import annotations

from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_C.DATASET = CfgNode()
# Directory the experiment reads its cohort from. Empty by design: a clinical cohort
# is private, so there is no default that could be right, and an experiment must be
# told where its copy lives.
_C.DATASET.ROOT = ""
# Modalities taking part in this run, named as the dataset names them. One name gives
# a unimodal baseline; several are fused.
_C.DATASET.MODALITIES = ["ecg"]
# Subjects lacking any of these modalities are dropped, so every arm of a comparison
# is scored on the same subjects.
_C.DATASET.REQUIRE_MODALITIES = []
_C.DATASET.NUM_WORKERS = 0
# How the test set is chosen:
#
#   "predefined"  an assignment the dataset publishes or the study fixed, applied with
#                 loaddata.Predefined. Preferred wherever one exists, because
#                 re-drawing the test set makes a result incomparable with other work.
#   "cv"          NUM_FOLDS-fold cross-validation over the whole cohort. What both
#                 cardiac use cases report, their cohorts being small.
#   "random"      a fresh stratified draw using the ratios below.
#
# Validation is always carved out of the training half, never out of the test set.
_C.DATASET.SPLIT_MODE = "cv"
_C.DATASET.VAL_RATIO = 0.15
_C.DATASET.TEST_RATIO = 0.2
_C.DATASET.NUM_FOLDS = 5
# Cohort column whose rows must never span two splits. A subject with several
# recordings is the case this exists for.
_C.DATASET.GROUP_KEY = "subject_id"
# Cohort columns whose distribution is preserved across splits.
_C.DATASET.STRATIFY_KEYS = ["label"]

# ---------------------------------------------------------------------------
# ECG
#
# The shape of the recordings a model is built for. No default asserts twelve leads
# or six: LEADS is the vocabulary the experiment supplies, and its length is how many
# lead modalities a lead-specific model creates.
# ---------------------------------------------------------------------------
_C.ECG = CfgNode()
# Lead names, in the order the cohort stores them. Empty means the experiment has not
# declared them, which every ECG pipeline here requires it to do.
_C.ECG.LEADS = []
# Samples per lead after preprocessing, and the rate they are sampled at.
_C.ECG.LENGTH = 5000
_C.ECG.SAMPLING_RATE = 500
# Rate the recordings arrive at, when it differs from SAMPLING_RATE and they must be
# resampled. 0 means they already arrive at SAMPLING_RATE.
_C.ECG.SOURCE_SAMPLING_RATE = 0
# Per-lead zero mean and unit variance, as both cardiac use cases apply.
_C.ECG.STANDARDISE = True
# Backbone each lead or each recording is encoded with; see
# kalecardiac.model.embed.SIGNAL_ENCODERS.
_C.ECG.ENCODER = "conv"
# Whether each lead is its own modality with its own encoder ("lead"), or the leads
# are the channels of one recording encoded together ("recording").
_C.ECG.MODALITY = "recording"

# ---------------------------------------------------------------------------
# Image
#
# Applies to any cardiovascular image modality -- chest X-ray, a cardiac MRI slice, an
# echocardiography frame -- because none of the mechanisms below distinguishes them.
# ---------------------------------------------------------------------------
_C.IMAGE = CfgNode()
_C.IMAGE.SIZE = [224, 224]
_C.IMAGE.CHANNELS = 1
# Rescale pixel values to the unit interval before the encoder sees them.
_C.IMAGE.SCALE = True

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_C.MODEL = CfgNode()
# Which model to build. An experiment's runner maps this to a class; the names the two
# supplied examples use are "cardiovae", "lsemvae" and "multimodal_vae".
_C.MODEL.NAME = "multimodal_vae"
_C.MODEL.LATENT_DIM = 256
# Channel widths of a convolutional encoder, and of its decoder reversed.
_C.MODEL.CHANNELS = [16, 32, 64]
_C.MODEL.KERNEL_SIZE = 3
_C.MODEL.DROPOUT = 0.5
# Width of the hidden layer in a prediction head. Empty gives a linear head.
_C.MODEL.HEAD_HIDDEN = [128]
# Checkpoint to initialise from when fine-tuning. Empty trains from scratch.
_C.MODEL.PRETRAINED = ""
# Freeze the pretrained encoders during fine-tuning, as both cardiac use cases do.
_C.MODEL.FREEZE_ENCODERS = True

# ---------------------------------------------------------------------------
# Fusion
#
# Two independent choices, so either can change from configuration alone:
# LATENT_METHOD combines per-modality Gaussians inside a multimodal VAE, and METHOD
# combines per-modality embeddings in a downstream predictor.
# ---------------------------------------------------------------------------
_C.FUSION = CfgNode()
# One of kalecardiac.model.layers.EXPERT_FUSIONS: "poe", "moe", "hime", "mean",
# "barycenter".
_C.FUSION.LATENT_METHOD = "poe"
# Groups the experts are divided into before a hierarchical fusion; used by "hime".
_C.FUSION.NUM_GROUPS = 4
# Include the standard-normal prior as an extra expert; used by "poe" and "hime".
_C.FUSION.USE_PRIOR = True
# One of kalecardiac.model.embed.FUSION_METHODS: "concat", "mean", "attention".
_C.FUSION.METHOD = "concat"
# Width every modality is projected to before fusion. 0 keeps each embedder's own
# width, which is what concatenating encoder means directly amounts to.
_C.FUSION.FUSION_DIM = 0
# Chance of marking each present modality absent during training, which is how a model
# is taught to survive a missing lead or a missing scan.
_C.FUSION.MODALITY_DROPOUT = 0.0

# ---------------------------------------------------------------------------
# Objective
#
# Weights of the generative objective; see kalecardiac.pipeline.ReconstructionTask.
# ---------------------------------------------------------------------------
_C.OBJECTIVE = CfgNode()
# Per-modality reconstruction weights, as "name:weight" pairs read by parse_weights.
# A modality not named here has weight 1.0. Pairs rather than a nested node because
# the modality names are the dataset's, not this schema's.
_C.OBJECTIVE.RECONSTRUCTION_WEIGHTS = []
# Scales the summed reconstruction term, keeping it commensurable with the KL term
# when a modality has tens of thousands of samples.
_C.OBJECTIVE.SCALE_FACTOR = 1.0
# Epochs over which the KL weight rises from zero to one. 0 disables annealing.
_C.OBJECTIVE.ANNEALING_EPOCHS = 0
# Weight on the latent alignment term, which pulls each modality's posterior mean
# towards the joint one. 0 disables it.
_C.OBJECTIVE.ALIGNMENT_WEIGHT = 0.0
# Also take an ELBO through each modality on its own, so an encoder stays usable when
# the others are absent. This is the tri-stream objective CardioVAE pretrains with.
_C.OBJECTIVE.UNIMODAL_STREAMS = False
# Weight on the positive class for a binary endpoint. 0 derives it from the training
# split's balance; a negative value disables it.
_C.OBJECTIVE.POS_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2026
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BATCH_SIZE = 32
# Epochs without improvement in the monitored metric before stopping. 0 disables it.
_C.SOLVER.EARLY_STOP = 0
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.ACCELERATOR = "auto"
_C.SOLVER.DEVICES = "auto"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "outputs"
# Export per-modality attribution and ablation summaries after evaluation.
_C.OUTPUT.INTERPRET = False


def get_cfg_defaults() -> CfgNode:
    """Return a copy of the default configuration."""
    return _C.clone()


def parse_weights(pairs: list[str]) -> dict[str, float]:
    """Read ``["ecg:10", "cxr:1"]`` as ``{"ecg": 10.0, "cxr": 1.0}``.

    Per-modality weights are configured as pairs because the modality names belong to
    the dataset rather than to this schema, and a ``CfgNode`` cannot gain keys it was
    not declared with.

    Args:
        pairs: ``name:weight`` strings, as ``OBJECTIVE.RECONSTRUCTION_WEIGHTS`` holds.

    Returns:
        Weight per modality name.

    Raises:
        ValueError: If an entry is not ``name:weight``, or the weight is not a number.
    """
    weights: dict[str, float] = {}
    for pair in pairs:
        name, separator, value = str(pair).partition(":")
        if not separator or not name:
            raise ValueError(f"expected 'name:weight', got {pair!r}")
        try:
            weights[name] = float(value)
        except ValueError as error:
            raise ValueError(f"weight for {name!r} is not a number: {value!r}") from error
    return weights
