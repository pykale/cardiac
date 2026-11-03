"""ECG preprocessing that needs to know a lead from a channel.

:mod:`kalecardiac.prepdata.signal_transform` handles anything true of a sampled
waveform. What is left here is the part that depends on *which* channel is which:
selecting a lead subset, reordering leads into a canonical sequence, and deriving the
augmented limb leads that a reduced-lead recorder leaves out.

The lead vocabulary itself lives in :mod:`kalecardiac.loaddata.signal_access`, beside the
sources that read recordings, because what a lead is called is a property of the data
rather than of a transform.

**Why lead order is worth a module of its own.** The two cardiac codebases this package
refactors both fix a lead order as a literal list, and in one of them the list in the
preprocessing script and the list the encoder indexes disagree -- aVL and aVF are
transposed between them. Nothing crashes, and a per-lead model silently learns aVL's
representation from aVF's signal. :func:`get_ecg_transform` exists so that the order is
stated once, canonically, and checked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
from torchvision.transforms import Compose

from kalecardiac.loaddata.signal_access import (
    ECGFormatError,
    get_lead_indices,
    normalize_lead_name,
    normalize_lead_names,
)
from kalecardiac.prepdata.signal_transform import (
<<<<<<< HEAD
    check_signal,
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    crop_or_pad_signal,
    interpolate_signal,
    normalize_signal,
    resample_signal,
    scale_signal,
)

#: The augmented limb leads, and how each is derived from leads I and II by Einthoven's
#: and Goldberger's relations. A reduced-lead recorder often stores only I and II, from
#: which the remaining four limb leads follow exactly rather than approximately.
DERIVABLE_LIMB_LEADS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "III": lambda lead_i, lead_ii: lead_ii - lead_i,
    "aVR": lambda lead_i, lead_ii: -(lead_i + lead_ii) / 2.0,
    "aVL": lambda lead_i, lead_ii: lead_i - lead_ii / 2.0,
    "aVF": lambda lead_i, lead_ii: lead_ii - lead_i / 2.0,
}


def select_leads(signal: ArrayLike, available: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    """Take ``wanted`` leads out of a recording, in the order given.

    Selection and ordering in one operation, because they are the same operation: a
    model built for ``["I", "II", "V1"]`` needs those three rows in that order, and
    doing it in two steps is how they come apart.

    Args:
        signal: ``(len(available), samples)`` recording.
        available: Lead names in the order the recording stores them.
        wanted: Lead names to take, in the order to produce.

    Returns:
        A ``(len(wanted), samples)`` array.

    Raises:
        ECGFormatError: If a wanted lead is absent, or the recording's row count does
            not match ``available``.
    """
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(available):
        raise ECGFormatError(
            f"recording of shape {array.shape} does not match the {len(available)} declared leads {list(available)}"
        )
    return np.ascontiguousarray(array[get_lead_indices(available, wanted)])


def derive_limb_leads(signal: ArrayLike, available: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    """Return ``wanted`` leads, deriving any missing limb lead from I and II.

    III, aVR, aVL and aVF are exact linear combinations of I and II, so a recorder that
    stored only those two has lost nothing about the frontal plane. Reconstructing them
    is what lets a model built for six limb leads run on a two-lead recording -- and,
    equally, what lets a six-lead cohort be assembled from a device that stores the
    leads in a reduced form.

    The precordial leads are *not* derivable: V1 to V6 measure the horizontal plane and
    carry information the limb leads do not, so asking for one that is absent raises.

    Args:
        signal: ``(len(available), samples)`` recording.
        available: Lead names in the order the recording stores them.
        wanted: Lead names to produce, in order.

    Returns:
        A ``(len(wanted), samples)`` array.

    Raises:
        ECGFormatError: If a wanted lead is neither present nor derivable, or if
            derivation is needed and I and II are not both present.
    """
    array = np.asarray(signal, dtype=np.float32)
    present = normalize_lead_names(available)
    chosen = normalize_lead_names(wanted)
    if array.ndim != 2 or array.shape[0] != len(present):
        raise ECGFormatError(f"recording of shape {array.shape} does not match the {len(present)} declared leads")

    missing = [lead for lead in chosen if lead not in present]
    unknown = [lead for lead in missing if lead not in DERIVABLE_LIMB_LEADS]
    if unknown:
        raise ECGFormatError(
            f"leads {unknown} are absent and cannot be derived: only {sorted(DERIVABLE_LIMB_LEADS)} follow from "
            f"I and II, while the precordial leads measure a plane the limb leads do not"
        )
    if missing and not {"I", "II"} <= set(present):
        raise ECGFormatError(f"deriving {missing} needs both I and II, but the recording holds {list(present)}")

    rows = {lead: array[index] for index, lead in enumerate(present)}
    if missing:
        lead_i, lead_ii = rows["I"], rows["II"]
        for lead in missing:
            rows[lead] = DERIVABLE_LIMB_LEADS[lead](lead_i, lead_ii)
    return np.ascontiguousarray(np.vstack([rows[lead] for lead in chosen]), dtype=np.float32)


<<<<<<< HEAD
def interleave_leads(signal: ArrayLike) -> np.ndarray:
    """Interleave a recording's leads, sample by sample, into one channel.

    ``(num_leads, num_samples)`` becomes ``(1, num_samples * num_leads)``, time-major: the
    first ``num_leads`` values are every lead's first sample. This is how CardioVAE reads
    a multi-lead ECG -- its reference implementation flattens a ``(samples, leads)``
    recording row by row -- and the layout its published weights expect. Every per-lead
    step has to come first: resampling or normalizing the strip would mix each lead into
    its neighbours.

        >>> interleave_leads(np.array([[1, 2, 3], [10, 20, 30]]))
        array([[ 1., 10.,  2., 20.,  3., 30.]], dtype=float32)

    Args:
        signal: ``(num_leads, num_samples)`` recording.

    Returns:
        A ``(1, num_leads * num_samples)`` array.
    """
    return np.ascontiguousarray(check_signal(signal).T).reshape(1, -1)


=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
def get_ecg_transform(
    length: int,
    sampling_rate: float = 500.0,
    source_sampling_rate: float | None = None,
    normalize_leads: bool = True,
    amplitude_scale: float | None = None,
    center_crop: bool = False,
<<<<<<< HEAD
    interleave: bool = False,
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
) -> Compose:
    """Assemble the preprocessing both cardiac use cases apply, in the order they apply it.

    Interpolate dropped samples, resample if the cohort was recorded at another rate,
    crop or pad to a fixed length, then normalise amplitude. The order is not
    interchangeable: interpolation must precede everything, because a NaN spreads
    through a resample; and normalisation must follow cropping, because statistics taken
    over samples that are then discarded describe a recording the model never sees.

    Args:
        length: Samples per lead the model expects.
        sampling_rate: Rate to produce, in hertz.
        source_sampling_rate: Rate the recordings arrive at. ``None`` or equal to
            ``sampling_rate`` skips resampling.
        normalize_leads: Zero mean and unit variance per lead. What both cardiac use
            cases apply; see :func:`~kalecardiac.prepdata.normalize_signal` for what it costs.
        amplitude_scale: Divide by this constant instead of normalizing, keeping
            absolute amplitude meaningful. Mutually exclusive with
            ``normalize_leads``.
        center_crop: Take the middle of a long recording rather than its start.
<<<<<<< HEAD
        interleave: Finally, interleave the leads into one channel of ``length`` times
            the leads' samples, with :func:`interleave_leads`, as CardioVAE reads an ECG.
            Last, so that every step before it works on one lead at a time.
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a

    Returns:
        A :class:`torchvision.transforms.Compose` of the steps, to hand to an ECG
        source's ``transform``.

    Raises:
        ValueError: If both or neither normalisation is requested.
    """
    if normalize_leads and amplitude_scale is not None:
        raise ValueError(
            "choose one of normalize_leads and amplitude_scale: normalizing discards the absolute "
            "amplitude that scaling exists to preserve"
        )

    steps: list[Callable[[np.ndarray], np.ndarray]] = [interpolate_signal]
    if source_sampling_rate is not None and source_sampling_rate != sampling_rate:
        steps.append(partial(resample_signal, source_rate=source_sampling_rate, target_rate=sampling_rate))
    steps.append(partial(crop_or_pad_signal, length=length, center=center_crop))

    if normalize_leads:
        steps.append(normalize_signal)
    elif amplitude_scale is not None:
        steps.append(partial(scale_signal, scale=amplitude_scale))
    else:
        raise ValueError(
            "no normalisation requested; pass normalize_leads=True, or amplitude_scale to keep absolute "
            "amplitude, and state the choice rather than leaving raw units to the optimiser"
        )
<<<<<<< HEAD
    if interleave:
        steps.append(interleave_leads)
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    return Compose(steps)


def get_lead_selector(available: Sequence[str], wanted: Sequence[str], derive: bool = False) -> Callable:
    """A one-argument callable that selects leads, for use as a transform step.

    Args:
        available: Lead names in the order the recording stores them.
        wanted: Lead names to produce, in order.
        derive: Reconstruct a missing limb lead from I and II rather than raising.

    Returns:
        A callable taking a recording and returning the selected leads.
    """
    chosen = normalize_lead_names(wanted)
    function = derive_limb_leads if derive else select_leads

    def select(signal: np.ndarray) -> np.ndarray:
        return function(signal, available, chosen)

    # Named for what it does, so a Compose of steps reads as one in a log.
    select.__name__ = select.__qualname__ = f"select_leads({', '.join(chosen)})"
    return select


__all__ = [
    "DERIVABLE_LIMB_LEADS",
    "get_ecg_transform",
    "normalize_lead_name",
    "normalize_lead_names",
    "derive_limb_leads",
    "get_lead_selector",
<<<<<<< HEAD
    "interleave_leads",
=======
>>>>>>> ad59998c08bbc42197e20661ea2a6d3cb2d2657a
    "select_leads",
]
