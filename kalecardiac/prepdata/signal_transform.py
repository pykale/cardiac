"""Transforms over multichannel signals.

Everything here operates on a ``(channels, samples)`` NumPy array and returns one, so
the same functions serve a twelve-lead ECG, a single rhythm strip, a pressure trace or
any other sampled cardiac waveform. None of them knows what the channels are called:
naming is :mod:`kalecardiac.prepdata.ecg_transform`'s job, and a transform that had to
be told it was working on lead aVR would not be a transform.

**None of these is fitted.** Each is a function of one recording, so applying it inside
a source is safe whichever split the subject landed in. That is a deliberate boundary:
anything estimated *across* subjects -- a cohort-wide mean, a quantile clip -- is
fold-local state, must be fitted on the training split alone, and would belong to a
fitted preprocessor rather than to this module.

The two amplitude normalizations differ in what they take as given, and the difference
matters clinically. :func:`normalize_signal` removes each channel's own mean and scale,
which is what both cardiac use cases apply and what makes recordings from different
machines comparable -- at the cost of discarding absolute millivolts, so a model cannot
learn from QRS amplitude. :func:`scale_signal` divides by a fixed constant instead,
keeping relative amplitude across channels and subjects intact.

**Names follow PyKale.** :func:`interpolate_signal` and :func:`normalize_signal` are the
functions :mod:`kale.prepdata.signal_transform` provides under the same names, for the
``(channels, samples)`` layout this package uses throughout; PyKale's take ``(samples,
channels)``. Steps are chained with :class:`torchvision.transforms.Compose`, as PyKale
chains its image transforms, rather than by a pipeline class of this package's own::

    >>> from functools import partial
    >>> from torchvision.transforms import Compose
    >>> transform = Compose([interpolate_signal, partial(crop_or_pad_signal, length=5000), normalize_signal])
    >>> transform(np.random.default_rng(0).normal(size=(12, 4000))).shape
    (12, 5000)

A source applies it to every recording it returns, e.g.
``ECGArraySource(recordings, leads=..., transform=transform)``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from kale.prepdata.signal_transform import interpolate_signal as _kale_interpolate
from numpy.typing import ArrayLike
from scipy.signal import resample_poly

#: Smallest standard deviation treated as real. Below it a channel is flat -- a
#: disconnected electrode, a saturated amplifier -- and dividing by its true scale
#: would turn rounding noise into a full-amplitude signal.
MIN_SCALE = 1e-8


def check_signal(value: ArrayLike) -> np.ndarray:
    """Coerce a recording to a 2-D ``(channels, samples)`` float32 array.

    A 1-D input is treated as one channel. Unlike
    :func:`kalecardiac.loaddata.check_ecg_array` this never transposes, because a
    transform has not been told how many channels to expect and so cannot tell a
    ``(samples, channels)`` recording from a ``(channels, samples)`` one.

    Args:
        value: The recording.

    Returns:
        A ``(channels, samples)`` array.

    Raises:
        ValueError: If the input is not 1-D or 2-D, or is empty.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D (channels, samples) signal, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("cannot transform an empty signal")
    return array


def interpolate_signal(signal: ArrayLike) -> np.ndarray:
    """Fill NaNs by linear interpolation along time, per channel.

    Dropped samples are ordinary in exported recordings, and every later transform
    propagates a NaN through the whole channel, so this runs first.

    Delegates to :func:`kale.prepdata.signal_transform.interpolate_signal`, which
    interpolates along the first axis of a ``(samples, channels)`` array; the transpose
    is here so that this package's ``(channels, samples)`` convention holds throughout.

    Args:
        signal: ``(channels, samples)`` recording.

    Returns:
        The recording with NaNs filled. A channel that is entirely NaN is returned as
        zeros, since there is nothing to interpolate between.
    """
    array = check_signal(signal)
    filled = np.asarray(_kale_interpolate(array.T), dtype=np.float32).T
    return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)


def crop_or_pad_signal(signal: ArrayLike, length: int, pad_value: float = 0.0, center: bool = False) -> np.ndarray:
    """Force a recording to exactly ``length`` samples.

    Longer recordings are cropped and shorter ones padded, which is how a cohort of
    recordings that a device wrote at slightly different durations becomes a batch.

    Args:
        signal: ``(channels, samples)`` recording.
        length: Samples per channel to produce.
        pad_value: Value padded with. Zero is right after :func:`normalize_signal`, where it
            is the channel mean; for raw millivolts it is not, and edge-padding or
            dropping the recording may be the better choice.
        center: Take the middle of a long recording, and pad a short one on both sides.
            The default takes the start, matching both cardiac use cases.

    Returns:
        A ``(channels, length)`` array.

    Raises:
        ValueError: If ``length`` is not positive.
    """
    array = check_signal(signal)
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")

    channels, samples = array.shape
    if samples == length:
        return array
    if samples > length:
        start = (samples - length) // 2 if center else 0
        return np.ascontiguousarray(array[:, start : start + length])

    padded = np.full((channels, length), float(pad_value), dtype=np.float32)
    start = (length - samples) // 2 if center else 0
    padded[:, start : start + samples] = array
    return padded


def resample_signal(signal: ArrayLike, source_rate: float, target_rate: float) -> np.ndarray:
    """Resample a recording from ``source_rate`` to ``target_rate``.

    Polyphase filtering by :func:`scipy.signal.resample_poly`, which low-pass filters
    before it decimates, so downsampling does not fold content above the new Nyquist
    frequency back into the band. The rate ratio is reduced to integer factors -- 500 to
    250 Hz is down by 2, 360 to 500 Hz up by 25 and down by 18 -- and the recording is
    extended linearly at both ends first, which keeps the filter's start-up transient
    off its first and last samples.

    Each channel is filtered on its own. Resample a multi-lead ECG *before* flattening
    its leads into one strip, or the filter runs across the join between two leads.

    Args:
        signal: ``(channels, samples)`` recording.
        source_rate: Rate the recording was sampled at, in hertz.
        target_rate: Rate to produce, in hertz.

    Returns:
        A ``(channels, ceil(samples * target_rate / source_rate))`` array.

    Raises:
        ValueError: If either rate is not positive.
    """
    array = check_signal(signal)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"sampling rates must be positive, got {source_rate} and {target_rate}")
    if source_rate == target_rate:
        return array
    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    resampled = resample_poly(array, ratio.numerator, ratio.denominator, axis=1, padtype="line")
    return np.ascontiguousarray(resampled, dtype=np.float32)


def normalize_signal(signal: ArrayLike, per_channel: bool = True) -> np.ndarray:
    """Remove the mean and divide by the standard deviation.

    PyKale's :func:`kale.prepdata.signal_transform.normalize_signal` under the same name,
    for ``(channels, samples)`` rather than ``(samples, channels)`` -- a test asserts the
    two agree. It is reimplemented rather than called because of a flat channel: PyKale
    divides by its true scale whenever that is non-zero, so a lead whose deviation is
    rounding noise comes out at full amplitude, where this centres it and leaves it flat.

    Per channel by default, which is what both cardiac use cases apply and what makes
    recordings from different machines and gains comparable. It also discards absolute
    amplitude, so a model trained on normalized recordings cannot learn from QRS
    voltage; where that matters, use :func:`scale_signal`.

    A channel whose scale is below :data:`MIN_SCALE` is centred but not scaled, so a
    flat lead stays flat instead of becoming amplified rounding noise.

    Args:
        signal: ``(channels, samples)`` recording.
        per_channel: Normalize each channel separately. ``False`` uses one mean and
            scale for the whole recording, preserving the relative amplitude between
            channels.

    Returns:
        The normalized recording.
    """
    array = check_signal(signal)
    axis: tuple[int, ...] = (1,) if per_channel else (0, 1)
    mean = array.mean(axis=axis, keepdims=True)
    scale = array.std(axis=axis, keepdims=True)
    return ((array - mean) / np.where(scale < MIN_SCALE, 1.0, scale)).astype(np.float32)


def scale_signal(signal: ArrayLike, scale: float = 1.0) -> np.ndarray:
    """Divide a recording by a fixed constant.

    The alternative to :func:`normalize_signal` for a cohort where absolute amplitude is
    evidence: dividing every recording by the same number brings values into a range an
    optimiser is comfortable with while leaving the ratios between channels, and
    between subjects, exactly as recorded.

    Args:
        signal: ``(channels, samples)`` recording.
        scale: Divisor, in the recording's own units -- millivolts, say.

    Returns:
        The scaled recording.

    Raises:
        ValueError: If ``scale`` is zero.
    """
    array = check_signal(signal)
    if scale == 0:
        raise ValueError("scale must be non-zero")
    return (array / float(scale)).astype(np.float32)


def clip_signal(signal: ArrayLike, limit: float) -> np.ndarray:
    """Clip a recording to ``[-limit, limit]``.

    A saturated electrode or a defibrillator artefact produces excursions orders of
    magnitude past physiology, which dominate a sum-of-squares reconstruction loss.
    Clipping bounds their influence without discarding the recording.

    Args:
        signal: ``(channels, samples)`` recording.
        limit: Largest magnitude kept, in the recording's units.

    Returns:
        The clipped recording.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    array = check_signal(signal)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return np.clip(array, -float(limit), float(limit)).astype(np.float32)
