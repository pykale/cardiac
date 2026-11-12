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
from functools import partial

import numpy as np
import torch
from numpy.typing import ArrayLike
from torchvision.transforms import Compose

from kalecardiac.prepdata.signal_transform import MIN_SCALE


def check_image(value: ArrayLike) -> np.ndarray:
    """Coerce an image to a 3-D ``(channels, height, width)`` float32 array.

    A 2-D input gains a leading channel axis. Unlike
    :func:`kalecardiac.loaddata.check_image_array` this never moves an axis, because a
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
    array = check_image(image)
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


def scale_image_intensity(image: ArrayLike, maximum: float | None = None) -> np.ndarray:
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
    array = check_image(image)
    if maximum is None:
        lowest, highest = float(array.min()), float(array.max())
        span = highest - lowest
        if span < MIN_SCALE:
            return np.zeros_like(array)
        return ((array - lowest) / span).astype(np.float32)
    if maximum <= 0:
        raise ValueError(f"maximum must be positive, got {maximum}")
    return np.clip(array / float(maximum), 0.0, 1.0).astype(np.float32)


def normalize_image(
    image: ArrayLike, mean: Sequence[float] | None = None, std: Sequence[float] | None = None
) -> np.ndarray:
    """Subtract a mean and divide by a standard deviation, per channel.

    With ``mean`` and ``std`` given this is :func:`torchvision.transforms.functional.normalize`
    on an array; without them it uses the image's own statistics, which torchvision's
    has no form for and which is what a radiograph from an uncalibrated machine needs.

    Args:
        image: ``(channels, height, width)`` image.
        mean: One value per channel. ``None`` uses the image's own channel means, which
            is the right default for a modality whose absolute intensity is not
            calibrated between machines.
        std: One value per channel. ``None`` uses the image's own channel deviations.

    Returns:
        The normalized image.

    Raises:
        ValueError: If a supplied statistic does not have one value per channel.
    """
    array = check_image(image)
    channels = array.shape[0]

    if mean is None:
        center = array.mean(axis=(1, 2), keepdims=True)
    elif len(mean) != channels:
        raise ValueError(f"mean must have one value per channel ({channels}), got {len(mean)}")
    else:
        center = np.asarray(mean, dtype=np.float32).reshape(channels, 1, 1)

    if std is None:
        scale = array.std(axis=(1, 2), keepdims=True)
    elif len(std) != channels:
        raise ValueError(f"std must have one value per channel ({channels}), got {len(std)}")
    else:
        scale = np.asarray(std, dtype=np.float32).reshape(channels, 1, 1)

    return ((array - center) / np.where(scale < MIN_SCALE, 1.0, scale)).astype(np.float32)


def get_image_transform(
    size: Sequence[int],
    scale: bool = True,
    maximum: float | None = None,
    normalize: bool = False,
) -> Compose:
    """Assemble the usual image preprocessing: resize, then rescale or normalize.

    Args:
        size: Target ``(height, width)``.
        scale: Rescale pixel values to the unit interval, which is what the cardiac use
            cases' sigmoid image decoder requires of its target.
        maximum: Value mapped to 1 when scaling; see :func:`scale_image_intensity`.
        normalize: Zero mean and unit variance per channel instead. Mutually exclusive
            with ``scale``, and incompatible with a sigmoid decoder.

    Returns:
        A :class:`torchvision.transforms.Compose` of the steps, to hand to an image
        source's ``transform``.

    Raises:
        ValueError: If both are requested.
    """
    if scale and normalize:
        raise ValueError("choose one of scale and normalize: a normalized image is not in [0, 1]")

    steps: list[Callable[[np.ndarray], np.ndarray]] = [
        partial(resize_image, size=tuple(int(extent) for extent in size))
    ]
    if scale:
        steps.append(partial(scale_image_intensity, maximum=maximum))
    elif normalize:
        steps.append(normalize_image)
    return Compose(steps)
