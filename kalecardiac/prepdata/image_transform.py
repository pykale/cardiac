"""Transforms over cardiovascular images.

Everything here operates on a ``(channels, height, width)`` NumPy array and returns one,
so a radiograph, a cardiac MRI slice and an echocardiography frame go through the same
functions. As in :mod:`kalecardiac.prepdata.signal_transform`, nothing is fitted across
subjects: each transform is a function of one image, so applying it inside a source
cannot leak a held-out subject's statistics into the training split.

Resizing is bilinear via :mod:`torch`, which is already a dependency, rather than
through Pillow or SciPy. That keeps the core install free of an imaging stack and makes
the behaviour identical to what a model's own interpolation would do.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike

from kalecardiac.prepdata.signal_transform import MIN_SCALE, step_name


def as_image(value: ArrayLike) -> np.ndarray:
    """Coerce an image to a 3-D ``(channels, height, width)`` float32 array.

    A 2-D input gains a leading channel axis. Unlike
    :func:`kalecardiac.loaddata.as_image_array` this never moves an axis, because a
    transform has not been told how many channels to expect.

    Args:
        value: The image.

    Returns:
        A ``(channels, height, width)`` array.

    Raises:
        ValueError: If the input is not 2-D or 3-D, or is empty.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D (channels, height, width) image, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("cannot transform an empty image")
    return array


def resize_image(image: ArrayLike, size: Sequence[int], antialias: bool = True) -> np.ndarray:
    """Resize an image to ``(height, width)`` by bilinear interpolation.

    Args:
        image: ``(channels, height, width)`` image.
        size: Target ``(height, width)``.
        antialias: Filter before downsampling. Worth keeping on: a cardiac radiograph
            reduced from 2000 to 224 pixels without it aliases fine structure such as
            lines and wires into the coarse texture a model then learns from.

    Returns:
        A ``(channels, height, width)`` array of the requested size.

    Raises:
        ValueError: If ``size`` is not two positive integers.
    """
    array = as_image(image)
    if len(size) != 2 or any(int(extent) < 1 for extent in size):
        raise ValueError(f"size must be two positive integers (height, width), got {tuple(size)}")

    target = (int(size[0]), int(size[1]))
    if array.shape[1:] == target:
        return array
    tensor = torch.from_numpy(array).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=target, mode="bilinear", align_corners=False, antialias=antialias
    )
    return np.ascontiguousarray(resized.squeeze(0).numpy(), dtype=np.float32)


def scale_image(image: ArrayLike, maximum: float | None = None) -> np.ndarray:
    """Rescale pixel values to the unit interval.

    Args:
        image: ``(channels, height, width)`` image.
        maximum: Value mapped to 1, e.g. 255 for an 8-bit export. ``None`` uses the
            image's own maximum, which is right for a modality with no fixed range --
            and wrong whenever there is one, because it makes a uniformly dark image as
            bright as a well-exposed one.

    Returns:
        The rescaled image, clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``maximum`` is not positive.
    """
    array = as_image(image)
    if maximum is None:
        lowest, highest = float(array.min()), float(array.max())
        span = highest - lowest
        if span < MIN_SCALE:
            return np.zeros_like(array)
        return ((array - lowest) / span).astype(np.float32)
    if maximum <= 0:
        raise ValueError(f"maximum must be positive, got {maximum}")
    return np.clip(array / float(maximum), 0.0, 1.0).astype(np.float32)


def standardise_image(image: ArrayLike, mean: Sequence[float] | None = None, std: Sequence[float] | None = None):
    """Subtract a mean and divide by a standard deviation, per channel.

    Args:
        image: ``(channels, height, width)`` image.
        mean: One value per channel. ``None`` uses the image's own channel means, which
            is the right default for a modality whose absolute intensity is not
            calibrated between machines.
        std: One value per channel. ``None`` uses the image's own channel deviations.

    Returns:
        The standardised image.

    Raises:
        ValueError: If a supplied statistic does not have one value per channel.
    """
    array = as_image(image)
    channels = array.shape[0]

    if mean is None:
        centre = array.mean(axis=(1, 2), keepdims=True)
    elif len(mean) != channels:
        raise ValueError(f"mean must have one value per channel ({channels}), got {len(mean)}")
    else:
        centre = np.asarray(mean, dtype=np.float32).reshape(channels, 1, 1)

    if std is None:
        scale = array.std(axis=(1, 2), keepdims=True)
    elif len(std) != channels:
        raise ValueError(f"std must have one value per channel ({channels}), got {len(std)}")
    else:
        scale = np.asarray(std, dtype=np.float32).reshape(channels, 1, 1)

    return ((array - centre) / np.where(scale < MIN_SCALE, 1.0, scale)).astype(np.float32)


class ImagePipeline:
    """A sequence of image transforms applied in order.

    The image counterpart of
    :class:`~kalecardiac.prepdata.signal_transform.SignalPipeline`, and what an image
    source's ``transform`` argument takes.

    Args:
        steps: Callables applied left to right, each taking and returning a
            ``(channels, height, width)`` array.

    Raises:
        TypeError: If a step is not callable.
    """

    def __init__(self, steps: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        for step in steps:
            if not callable(step):
                raise TypeError(f"every step must be callable, got {type(step).__name__}")
        self.steps = list(steps)

    def __call__(self, image: ArrayLike) -> np.ndarray:
        array = as_image(image)
        for step in self.steps:
            array = as_image(step(array))
        return array

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"ImagePipeline([{', '.join(step_name(step) for step in self.steps)}])"


def build_image_pipeline(
    size: Sequence[int],
    scale: bool = True,
    maximum: float | None = None,
    standardise: bool = False,
) -> ImagePipeline:
    """Assemble the usual image preprocessing: resize, then normalise.

    Args:
        size: Target ``(height, width)``.
        scale: Rescale pixel values to the unit interval, which is what the cardiac use
            cases' sigmoid image decoder requires of its target.
        maximum: Value mapped to 1 when scaling; see :func:`scale_image`.
        standardise: Zero mean and unit variance per channel instead. Mutually
            exclusive with ``scale``, and incompatible with a sigmoid decoder.

    Returns:
        A pipeline to hand to an image source's ``transform``.

    Raises:
        ValueError: If both normalisations are requested.
    """
    if scale and standardise:
        raise ValueError("choose one of scale and standardise: a standardised image is not in [0, 1]")

    steps: list[Callable[[np.ndarray], np.ndarray]] = [lambda image: resize_image(image, size)]
    steps[0].__name__ = f"resize_image({tuple(int(extent) for extent in size)})"  # type: ignore[attr-defined]
    if scale:
        steps.append(lambda image: scale_image(image, maximum))
        steps[-1].__name__ = "scale_image"  # type: ignore[attr-defined]
    elif standardise:
        steps.append(standardise_image)
    return ImagePipeline(steps)
