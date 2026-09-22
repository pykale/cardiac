"""Which parts of the input a cardiac prediction rested on.

Three mechanisms, answering three different questions, and the distinction between them
matters because they are routinely conflated:

**Where in a recording.** :func:`modality_attributions` runs integrated gradients over
a model whose input is a dictionary of modalities, giving an attribution per sample of
every lead and per pixel of every image. That is a per-subject, per-timepoint answer.

**Which lead, or which modality.** :func:`attribution_ratios` reduces those
attributions to one number per modality: the fraction of its samples whose normalised
attribution clears a threshold, and that fraction's share of the total. The LS-EMVAE
study calls this the integrated-gradient attribution ratio, and it is what turns a
per-timepoint map into a lead ranking.

**Which part of the cardiac cycle.** :func:`segment_attribution_ratios` does the same
within named windows of the signal, so "the model attends to the T wave" becomes a
number. The windows are supplied by the caller; :func:`ecg_wave_segments` finds them
with NeuroKit2 for anyone who has it.

:func:`modality_ablation` is the counterpart that needs no gradients: withhold a
modality at inference and measure what the metric loses. It answers a cohort-level
question where attribution answers a subject-level one, and the two disagreeing is
informative rather than a bug -- a lead can be heavily attributed and still be
redundant, if another lead carries the same information.

Nothing here plots. Attributions and ratios are returned as arrays and dictionaries, so
a publication figure -- which is an experiment's concern -- is built in ``examples/``
from numbers the library produced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from kalecardiac.evaluate.classification_metrics import MetricError
from kalecardiac.loaddata.multimodal_access import SubjectBatch

#: Smallest attribution range treated as real. Below it every sample was attributed
#: equally, and normalising would turn floating-point noise into a full-scale map.
MIN_RANGE = 1e-10


def _require_captum():
    """Import Captum, explaining the install if it is absent."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'gradient attribution needs captum. Install it with: pip install "kalecardiac[interpret]"'
        ) from error
    return IntegratedGradients


def normalise_attribution(attribution: np.ndarray) -> np.ndarray:
    """Rescale an attribution map to ``[0, 1]``.

    Min-max rather than by absolute value, matching the cardiac reference
    implementations. The consequence is worth stating: a *negative* attribution -- a
    region pushing the prediction away from the predicted class -- maps near 0 and is
    then indistinguishable from a region that contributed nothing. Read a ratio
    computed from this as "share of strongly positive evidence", not "share of
    influence".

    Args:
        attribution: Any shape.

    Returns:
        The same shape, rescaled. All zeros where the input was constant.
    """
    attribution = np.asarray(attribution, dtype=np.float64)
    if attribution.size == 0:
        return attribution
    lowest, highest = attribution.min(), attribution.max()
    span = highest - lowest
    if span < MIN_RANGE:
        return np.zeros_like(attribution)
    return (attribution - lowest) / span


def modality_attributions(
    model: nn.Module,
    modalities: Mapping[str, Tensor],
    target: int | Tensor | None = None,
    present: Mapping[str, Tensor] | None = None,
    n_steps: int = 50,
    baselines: Mapping[str, Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """Integrated-gradient attributions for every modality of one batch.

    Integrated gradients accumulates the model's gradient along a straight path from a
    baseline to the input, which makes the attributions sum to the difference in output
    between the two -- the property that lets them be compared between modalities at
    all.

    The **baseline** is the choice that decides what the attributions mean. An all-zero
    baseline over a standardised recording is "the mean signal", so an attribution reads
    as evidence relative to an average heart; over an unstandardised one it is
    "no signal at all", which is not a recording any subject could produce. The default
    is zeros, and a caller whose preprocessing makes that wrong should pass their own.

    **Only what the model reads is attributed.** A batch commonly carries more
    modalities than the model consumes -- a loader built over twelve leads feeding a
    model fine-tuned on six, say. A modality the model never reads has no attribution by
    definition, and asking for one produces an error from deep inside the gradient
    machinery rather than a useful answer, so a model that declares its ``modalities``
    is taken at its word and the rest are dropped.

    Args:
        model: Takes ``(modalities, present)`` and returns an output carrying
            ``prediction``. If it declares ``modalities``, only those are attributed.
        modalities: One tensor per modality, keyed by name.
        target: Output column to attribute. ``None`` attributes the predicted class for
            a multi-column output, and the single score for a one-column one.
        present: ``(B,)`` boolean per modality, held fixed through the attribution.
        n_steps: Steps along the integration path. More is more accurate and slower.
        baselines: One baseline per modality. ``None`` uses zeros.

    Returns:
        One ``(B, ...)`` attribution array per attributed modality, in that modality's
        own shape.

    Raises:
        ImportError: If Captum is not installed.
        ValueError: If ``modalities`` is empty, or holds none the model reads.
    """
    integrated_gradients = _require_captum()
    if not modalities:
        raise ValueError("no modality to attribute")

    declared = getattr(model, "modalities", None)
    names = [name for name in modalities if name in declared] if declared is not None else list(modalities)
    if not names:
        raise ValueError(
            f"none of the supplied modalities {sorted(modalities)} is read by this model, which reads "
            f"{sorted(declared or ())}"
        )

    inputs = tuple(modalities[name].detach().clone().requires_grad_(True) for name in names)
    batch_size = inputs[0].shape[0]

    def wrapped(*values: Tensor) -> Tensor:
        # Integrated gradients evaluates every step of its path in one pass, so the
        # tensors arriving here are the batch repeated n_steps times. The presence mask
        # is not an attributed input and so is not expanded for us; repeating it the
        # same way is what keeps it aligned with the rows it describes.
        repeats = values[0].shape[0] // batch_size
        expanded = None if present is None else {name: value.repeat(repeats) for name, value in present.items()}
        output = model(dict(zip(names, values, strict=True)), expanded)
        prediction = output.prediction
        return prediction if prediction.dim() > 1 else prediction.unsqueeze(1)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = wrapped(*inputs)
        if target is None:
            target = (
                torch.zeros(prediction.shape[0], dtype=torch.long)
                if prediction.shape[1] == 1
                else prediction.argmax(dim=1)
            )

        baseline_tuple = (
            None if baselines is None else tuple(baselines[name].to(inputs[index]) for index, name in enumerate(names))
        )
        attributions = integrated_gradients(wrapped).attribute(
            inputs=inputs, baselines=baseline_tuple, target=target, n_steps=n_steps
        )
    finally:
        model.train(was_training)

    return {name: attributions[index].detach().cpu().numpy() for index, name in enumerate(names)}


def attribution_ratio(attribution: np.ndarray, threshold: float = 0.7) -> float:
    """Fraction of an attribution map above a normalised threshold.

    The per-modality summary the LS-EMVAE study reports. It measures *how much* of a
    lead was strongly attributed, not how strongly -- a lead with one enormous spike
    scores lower than one with a broad, moderate region, which is usually the reading a
    clinician wants from an ECG.

    Args:
        attribution: Any shape; flattened before counting.
        threshold: Normalised attribution a sample must reach to count.

    Returns:
        The fraction, in ``[0, 1]``.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]``, or the map is empty.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    normalised = normalise_attribution(attribution).ravel()
    if normalised.size == 0:
        raise ValueError("cannot take a ratio of an empty attribution map")
    return float((normalised >= threshold).mean())


def attribution_ratios(attributions: Mapping[str, np.ndarray], threshold: float = 0.7) -> dict[str, dict[str, float]]:
    """Attribution ratio per modality, and each modality's share of the total.

    The lead-importance table: for a twelve-lead model this is twelve rows saying how
    much of each lead the model attended to, and what proportion of its total attention
    that was.

    Attribution is normalised **within** each modality, so the ratios are comparable
    across modalities of different lengths -- a lead of 5000 samples and an image of
    50,176 pixels both report a fraction of themselves. They are not comparable in
    absolute magnitude: a modality whose attributions are uniformly small still has a
    maximum, and normalising rescales it to 1.

    Args:
        attributions: One attribution array per modality, as
            :func:`modality_attributions` returns.
        threshold: Normalised attribution a sample must reach to count.

    Returns:
        ``{modality: {"ratio": ..., "share": ...}}``, the shares summing to 1 (or all
        zero when no modality cleared the threshold).

    Raises:
        ValueError: If ``attributions`` is empty.
    """
    if not attributions:
        raise ValueError("no attributions to summarise")

    ratios = {name: attribution_ratio(values, threshold) for name, values in attributions.items()}
    total = sum(ratios.values())
    return {name: {"ratio": ratio, "share": (ratio / total if total > 0 else 0.0)} for name, ratio in ratios.items()}


def segment_attribution_ratios(
    attribution: np.ndarray,
    segments: Mapping[str, Sequence[int]],
    window: int = 20,
    threshold: float = 0.7,
) -> dict[str, dict[str, float]]:
    """Attribution ratio within named windows of a signal.

    Turns "where in the beat" into numbers: given the sample indices of each wave of the
    cardiac cycle, this reports how much of the strongly attributed signal falls inside
    each one. Windows may overlap, and a sample inside two of them is counted in both,
    so the shares are of the *sum* of the per-segment counts rather than of the whole
    signal.

    Args:
        attribution: ``(num_samples,)`` attribution for one lead. A multi-dimensional
            map is flattened, which is right for a single lead and wrong for several --
            attribute leads separately.
        segments: Sample indices per named segment, e.g. ``{"R": [312, 812, ...]}``.
        window: Half-width in samples around each index. At 500 Hz a window of 20 is
            40 ms either side, about the width of a QRS complex.
        threshold: Normalised attribution a sample must reach to count.

    Returns:
        ``{segment: {"ratio": ..., "share": ..., "n_points": ...}}``.

    Raises:
        ValueError: If ``window`` is negative, or the attribution is empty.
    """
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")

    normalised = normalise_attribution(attribution).ravel()
    if normalised.size == 0:
        raise ValueError("cannot take a ratio of an empty attribution map")
    important = set(np.flatnonzero(normalised >= threshold).tolist())

    counts: dict[str, int] = {}
    for name, indices in segments.items():
        covered: set[int] = set()
        for index in indices:
            start = max(int(index) - window, 0)
            end = min(int(index) + window + 1, normalised.size)
            covered.update(range(start, end))
        counts[name] = len(covered & important)

    total = sum(counts.values())
    return {
        name: {
            "ratio": count / normalised.size,
            "share": (count / total if total > 0 else 0.0),
            "n_points": count,
        }
        for name, count in counts.items()
    }


def ecg_wave_segments(signal: np.ndarray, sampling_rate: float = 500.0) -> dict[str, list[int]]:
    """Locate the P, Q, R, S and T waves of a single-lead ECG.

    A thin wrapper over NeuroKit2's R-peak detection and wavelet delineation, so that
    :func:`segment_attribution_ratios` can be given cardiac landmarks rather than
    arbitrary windows. It is here rather than in ``prepdata`` because nothing in the
    modelling path needs it: it exists to make an attribution readable.

    Args:
        signal: ``(num_samples,)`` single-lead recording.
        sampling_rate: Samples per second.

    Returns:
        Sample indices per wave. A wave the delineator could not find is an empty list
        rather than a missing key, so a caller can always index it.

    Raises:
        ImportError: If NeuroKit2 is not installed.
        ValueError: If the signal is not one-dimensional, or no R peak was found --
            which usually means the recording is too short, too noisy, or not an ECG.
    """
    try:
        import neurokit2 as nk
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'ECG wave delineation needs neurokit2. Install it with: pip install "kalecardiac[ecg]"'
        ) from error

    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"expected a single lead of shape (num_samples,), got {signal.shape}")

    # NeuroKit signals a recording it cannot read in several ways -- an empty peak
    # list, an IndexError from inside its own gradient search -- and every one of them
    # means the same thing to a caller, so they are reported as one.
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        _, peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_peaks = np.asarray(peaks.get("ECG_R_Peaks", []), dtype=float)
        r_peaks = r_peaks[np.isfinite(r_peaks)].astype(int)
    except (IndexError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"no R peak was detected ({error}); the recording may be too short, too noisy, or not an ECG"
        ) from error
    if r_peaks.size == 0:
        raise ValueError("no R peak was detected; the recording may be too short, too noisy, or not an ECG")

    try:
        _, waves = nk.ecg_delineate(cleaned, rpeaks=r_peaks.tolist(), sampling_rate=sampling_rate, method="dwt")
    except (IndexError, ValueError) as error:
        raise ValueError(f"the R peaks were found but the waves could not be delineated ({error})") from error

    def clean(values) -> list[int]:
        array = np.asarray(values if values is not None else [], dtype=float)
        return array[np.isfinite(array)].astype(int).tolist()

    return {
        "P": clean(waves.get("ECG_P_Peaks")),
        "Q": clean(waves.get("ECG_Q_Peaks")),
        "R": r_peaks.tolist(),
        "S": clean(waves.get("ECG_S_Peaks")),
        "T": clean(waves.get("ECG_T_Peaks")),
    }


def modality_ablation(
    model: nn.Module,
    loader: Iterable[SubjectBatch],
    metric: Callable[[np.ndarray, np.ndarray], float],
    target_key: str,
    modalities: Sequence[str] | None = None,
) -> dict[str, float]:
    """How much each modality is worth, by withholding it at inference.

    Marks one modality absent for every subject and rescores the split. What drops most
    is what the model could least do without -- which is a different question from what
    it attributed most to, because a heavily attributed lead whose information another
    lead also carries loses nothing when removed.

    Withholding is done through the presence mask rather than by zeroing the input, so a
    model that handles missing modalities properly sees exactly what it would see in
    deployment: an absent modality, not a flat recording.

    Args:
        model: Takes a batch and returns an output carrying ``prediction``; a
            :class:`~kalecardiac.pipeline.CardiacTrainer` or any module with
            ``predict``.
        loader: Yields :class:`~kalecardiac.loaddata.SubjectBatch` batches.
        metric: Takes ``(targets, scores)`` and returns a float, e.g.
            :func:`~kalecardiac.evaluate.roc_auc`.
        target_key: Which batch target the metric scores against.
        modalities: Which to ablate. ``None`` ablates every modality the *model* reads
            where it declares them, and otherwise every modality the batch carries.
            Withholding one the model never reads changes nothing, so it is not an
            error -- just a row of zeros nobody asked for.

    Returns:
        ``{"full": ...}`` with the intact score, then ``{"without_<name>": ...}`` per
        modality, and ``{"drop_<name>": ...}`` for the difference. A modality whose
        ablated split cannot be scored is omitted rather than reported as zero.

    Raises:
        ValueError: If the loader yields nothing, or carries no such target.
    """
    batches = list(loader)
    if not batches:
        raise ValueError("the loader yielded no batches")
    if target_key not in batches[0].target:
        raise ValueError(f"batches carry no target {target_key!r}; available: {sorted(batches[0].target)}")

    if modalities is not None:
        names = list(modalities)
    else:
        declared = getattr(getattr(model, "model", model), "modalities", None)
        names = list(declared) if declared is not None else list(batches[0].modalities)
    targets = np.concatenate([batch.target[target_key].detach().cpu().numpy().reshape(-1) for batch in batches])

    def score(withhold: str | None) -> np.ndarray:
        scores = []
        for batch in batches:
            if withhold is None:
                used = batch
            else:
                present = dict(batch.present)
                present[withhold] = torch.zeros_like(present[withhold])
                used = SubjectBatch(
                    subject_id=batch.subject_id,
                    modalities=batch.modalities,
                    present=present,
                    target=batch.target,
                    metadata=batch.metadata,
                )
            output = model.predict(used) if hasattr(model, "predict") else model(used.modalities, used.present)
            scores.append(output.prediction.detach().cpu().reshape(len(used), -1)[:, 0].numpy())
        return np.concatenate(scores)

    results: dict[str, float] = {"full": float(metric(targets, score(None)))}
    for name in names:
        try:
            value = float(metric(targets, score(name)))
        except MetricError:
            continue
        results[f"without_{name}"] = value
        results[f"drop_{name}"] = results["full"] - value
    return results
