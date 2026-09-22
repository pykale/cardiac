"""Configuration for the CardioVAE chest-X-ray-and-ECG example.

Extends the package defaults with what *this experiment* adds: which endpoint to
predict, how the cohort's files are laid out, and the objective weights the paper used.
None of that belongs in :mod:`kalecardiac.config`, which must stay dataset-agnostic.

The defaults below reproduce the setup of Suvon et al., MICCAI 2024, on a cohort with
the schema described in this example's README. The cohort itself is a private clinical
registry and is not distributed; ``DATASET.ROOT`` is empty so that a run either points
at a real copy or falls back to the synthetic generator.
"""

from __future__ import annotations

from yacs.config import CfgNode

from kalecardiac.config import get_cfg_defaults as _package_defaults

#: Modality names this experiment uses, matching the directory names in its schema.
ECG = "ecg"
CXR = "cxr"

#: Endpoints the example can predict. Both are thresholds applied to a pressure
#: measured at right-heart catheterisation, which is the invasive test the whole point
#: of the model is to avoid.
ENDPOINTS = ("chdi", "mpap")


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults extended for this experiment."""
    cfg = _package_defaults()

    # ---------------------------------------------------------------- data
    cfg.DATASET.MODALITIES = [ECG, CXR]
    # Both arms are scored on the subjects that have a radiograph, so the unimodal and
    # multimodal numbers are commensurable. Without this the ECG arm would be scored on
    # a larger cohort and the comparison would mean nothing.
    cfg.DATASET.REQUIRE_MODALITIES = [CXR]
    cfg.DATASET.SPLIT_MODE = "cv"
    cfg.DATASET.NUM_FOLDS = 10
    cfg.DATASET.STRATIFY_KEYS = ["label"]

    # ---------------------------------------------------------------- modalities
    # A ten-second twelve-lead recording at 500 Hz, flattened into one channel. The
    # reference implementation encodes the leads concatenated end to end rather than as
    # channels; ECG.MODALITY selects which, and "recording" with NUM_CHANNELS 1 is the
    # flattened form.
    cfg.ECG.LENGTH = 60000
    cfg.ECG.SAMPLING_RATE = 500
    cfg.ECG.MODALITY = "recording"
    cfg.ECG.STANDARDISE = True
    # Channels the recording is presented with. 1 flattens every lead into one long
    # signal, as the paper does; 12 encodes the leads as channels, which is the more
    # natural reading and costs nothing to try.
    cfg.ECG.NUM_CHANNELS = 1

    cfg.IMAGE.SIZE = [224, 224]
    cfg.IMAGE.CHANNELS = 1
    cfg.IMAGE.SCALE = True

    # ---------------------------------------------------------------- model
    cfg.MODEL.NAME = "cardiovae"
    cfg.MODEL.LATENT_DIM = 256
    cfg.MODEL.CHANNELS = [16, 32, 64]
    cfg.MODEL.HEAD_HIDDEN = [128]
    cfg.MODEL.DROPOUT = 0.5
    cfg.MODEL.FREEZE_ENCODERS = True

    # ---------------------------------------------------------------- objective
    # The ECG is weighted ten times the radiograph: a sum of squares over 60,000
    # samples and one over 50,176 pixels are not comparable quantities, and without the
    # weight the image term dominates the ELBO.
    cfg.OBJECTIVE.RECONSTRUCTION_WEIGHTS = [f"{ECG}:10", f"{CXR}:1"]
    cfg.OBJECTIVE.SCALE_FACTOR = 1e-4
    cfg.OBJECTIVE.ANNEALING_EPOCHS = 50
    # The tri-stream objective: an ELBO through the pair, and one through each modality
    # alone, so an encoder stays usable when the other modality is missing.
    cfg.OBJECTIVE.UNIMODAL_STREAMS = True
    cfg.OBJECTIVE.ALIGNMENT_WEIGHT = 0.0

    # ---------------------------------------------------------------- endpoint
    cfg.ENDPOINT = CfgNode()
    cfg.ENDPOINT.NAME = ENDPOINTS[0]
    # The column the cohort's table records the measurement in, and the threshold above
    # which the endpoint is positive. Both are this dataset's, which is why they are
    # here and not in the package schema.
    cfg.ENDPOINT.VALUE_COLUMN = "mpap_mmhg"
    cfg.ENDPOINT.THRESHOLD = 20.0
    cfg.ENDPOINT.UNIT = "mmHg"
    # Predict the measurement itself rather than the threshold applied to it. Keeps the
    # information the threshold throws away, and is a RegressionTask instead.
    cfg.ENDPOINT.REGRESSION = False

    # ---------------------------------------------------------------- run
    cfg.SOLVER.SEED = 2026
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.MAX_EPOCHS = 100
    cfg.SOLVER.BATCH_SIZE = 128
    cfg.SOLVER.OPTIMIZER = "Adam"

    # Epochs of supervised fine-tuning, which is a shorter run than pretraining on a
    # far smaller cohort.
    cfg.FINETUNE = CfgNode()
    cfg.FINETUNE.MAX_EPOCHS = 50
    cfg.FINETUNE.BASE_LR = 1e-3
    cfg.FINETUNE.BATCH_SIZE = 32

    # ---------------------------------------------------------------- synthetic
    # Used when DATASET.ROOT is empty, so the example runs end to end with no data.
    cfg.SYNTHETIC = CfgNode()
    cfg.SYNTHETIC.NUM_SUBJECTS = 64
    cfg.SYNTHETIC.NUM_SAMPLES = 2048
    cfg.SYNTHETIC.IMAGE_SIZE = [32, 32]

    cfg.OUTPUT.OUT_DIR = "outputs/cardiovae_multimodal"
    cfg.OUTPUT.INTERPRET = True
    return cfg
