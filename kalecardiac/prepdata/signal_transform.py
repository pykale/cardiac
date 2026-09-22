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

The two standardisation functions differ in what they take as given, and the difference
matters clinically. :func:`standardise` removes each channel's own mean and scale, which
is what both cardiac use cases apply and what makes recordings from different machines
comparable -- at the cost of discarding absolute millivolts, so a model cannot learn
from QRS amplitude. :func:`scale_amplitude` divides by a fixed constant instead, keeping
relative amplitude across channels and subjects intact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from kale.prepdata.signal_transform import interpolate_signal as _kale_interpolate
from numpy.typing import ArrayLike

#: Smallest standard deviation treated as real. Below it a channel is flat -- a
#: disconnected electrode, a saturated amplifier -- and dividing by its true scale
#: would turn rounding noise into a full-amplitude signal.
MIN_SCALE = 1e-8


def step_name(step: Callable) -> str:
    """Name a pipeline step for a repr, seeing through ``functools.partial``.

    A pipeline built from configuration is mostly partials, and a repr reading
    ``[partial, partial, partial]`` tells a reader nothing about what was applied.

    Args:
        step: Any callable.

    Returns:
        The underlying function's name, with the bound keywords a partial carries.
    """
    if isinstance(step, partial):
        bound = ", ".join(f"{key}={value!r}" for key, value in step.keywords.items())
        return f"{step_name(step.func)}({bound})"
    return getattr(step, "__name__", type(step).__name__)


def as_signal(value: ArrayLike) -> np.ndarray:
    """Coerce a recording to a 2-D ``(channels, samples)`` float32 array.

    A 1-D input is treated as one channel. Unlike
    :func:`kalecardiac.loaddata.as_lead_array` this never transposes, because a
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


def interpolate_missing(signal: ArrayLike) -> np.ndarray:
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
    array = as_signal(signal)
    filled = np.asarray(_kale_interpolate(array.T), dtype=np.float32).T
    return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)


def crop_or_pad(signal: ArrayLike, length: int, pad_value: float = 0.0, centre: bool = False) -> np.ndarray:
    """Force a recording to exactly ``length`` samples.

    Longer recordings are cropped and shorter ones padded, which is how a cohort of
    recordings that a device wrote at slightly different durations becomes a batch.

    Args:
        signal: ``(channels, samples)`` recording.
        length: Samples per channel to produce.
        pad_value: Value padded with. Zero is right after :func:`standardise`, where it
            is the channel mean; for raw millivolts it is not, and edge-padding or
            dropping the recording may be the better choice.
        centre: Take the middle of a long recording, and pad a short one on both sides.
            The default takes the start, matching both cardiac use cases.

    Returns:
        A ``(channels, length)`` array.

    Raises:
        ValueError: If ``length`` is not positive.
    """
    array = as_signal(signal)
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")

    channels, samples = array.shape
    if samples == length:
        return array
    if samples > length:
        start = (samples - length) // 2 if centre else 0
        return np.ascontiguousarray(array[:, start : start + length])

    padded = np.full((channels, length), float(pad_value), dtype=np.float32)
    start = (length - samples) // 2 if centre else 0
    padded[:, start : start + samples] = array
    return padded


def resample(signal: ArrayLike, source_rate: float, target_rate: float) -> np.ndarray:
    """Resample a recording from ``source_rate`` to ``target_rate`` by interpolation.

    Linear interpolation on a regular grid, which is adequate for the modest rate
    changes a cardiac cohort needs (500 Hz to 250 Hz, 1000 Hz to 500 Hz) and has no
    filter to misconfigure. It is *not* an anti-aliased decimation: downsampling by a
    large factor folds high-frequency content back into the band, so for a large
    reduction filter first with a library built for it.

    Args:
        signal: ``(channels, samples)`` recording.
        source_rate: Rate the recording was sampled at, in hertz.
        target_rate: Rate to produce, in hertz.

    Returns:
        A ``(channels, round(samples * target_rate / source_rate))`` array.

    Raises:
        ValueError: If either rate is not positive.
    """
    array = as_signal(signal)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"sampling rates must be positive, got {source_rate} and {target_rate}")
    if source_rate == target_rate:
        return array

    samples = array.shape[1]
    target_samples = max(int(round(samples * target_rate / source_rate)), 1)
    # Endpoint-inclusive grids, so the first and last samples of the output sit at the
    # first and last of the input rather than drifting by a fraction of a period.
    source_grid = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    target_grid = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    resampled = np.vstack([np.interp(target_grid, source_grid, channel) for channel in array])
    return np.ascontiguousarray(resampled, dtype=np.float32)


def standardise(signal: ArrayLike, per_channel: bool = True) -> np.ndarray:
    """Remove the mean and divide by the standard deviation.

    Per channel by default, which is what both cardiac use cases apply and what makes
    recordings from different machines and gains comparable. It also discards absolute
    amplitude, so a model trained on standardised recordings cannot learn from QRS
    voltage; where that matters, use :func:`scale_amplitude`.

    A channel whose scale is below :data:`MIN_SCALE` is centred but not scaled, so a
    flat lead stays flat instead of becoming amplified rounding noise.

    Args:
        signal: ``(channels, samples)`` recording.
        per_channel: Standardise each channel separately. ``False`` uses one mean and
            scale for the whole recording, preserving the relative amplitude between
            channels.

    Returns:
        The standardised recording.
    """
    array = as_signal(signal)
    axis: tuple[int, ...] = (1,) if per_channel else (0, 1)
    mean = array.mean(axis=axis, keepdims=True)
    scale = array.std(axis=axis, keepdims=True)
    return ((array - mean) / np.where(scale < MIN_SCALE, 1.0, scale)).astype(np.float32)


def scale_amplitude(signal: ArrayLike, scale: float = 1.0) -> np.ndarray:
    """Divide a recording by a fixed constant.

    The alternative to :func:`standardise` for a cohort where absolute amplitude is
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
    array = as_signal(signal)
    if scale == 0:
        raise ValueError("scale must be non-zero")
    return (array / float(scale)).astype(np.float32)


def clip_amplitude(signal: ArrayLike, limit: float) -> np.ndarray:
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
    array = as_signal(signal)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return np.clip(array, -float(limit), float(limit)).astype(np.float32)


class SignalPipeline:
    """A sequence of signal transforms applied in order.

    What a source's ``transform`` argument takes, so that the whole preprocessing of a
    modality is one value that can be built from configuration, logged and compared::

        >>> pipeline = SignalPipeline([interpolate_missing, partial(crop_or_pad, length=5000), standardise])
        >>> ECGArraySource(recordings, leads=LEADS, transform=pipeline)

    Composition rather than a class hierarchy: every step is an ordinary function of
    one array, so a caller can add their own without implementing anything.

    Args:
        steps: Callables applied left to right, each taking and returning a
            ``(channels, samples)`` array.

    Raises:
        TypeError: If a step is not callable.
    """

    def __init__(self, steps: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        for step in steps:
            if not callable(step):
                raise TypeError(f"every step must be callable, got {type(step).__name__}")
        self.steps = list(steps)

    def __call__(self, signal: ArrayLike) -> np.ndarray:
        array = as_signal(signal)
        for step in self.steps:
            array = as_signal(step(array))
        return array

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"SignalPipeline([{', '.join(step_name(step) for step in self.steps)}])"
