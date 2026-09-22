"""Access to cardiovascular images, whatever produced them.

Nothing here is specific to a chest X-ray. An image modality is a ``(channels, height,
width)`` tensor per subject, which is equally a radiograph, a short-axis cardiac MRI
slice, a CT section or an echocardiography frame; what differs between them is how the
file is read and what preprocessing it needs, and both of those are arguments rather
than assumptions.

Two sources, mirroring :mod:`kalecardiac.loaddata.ecg_access`:

* :class:`ImageArraySource` for arrays already in memory;
* :class:`ImageFileSource` for one file per subject, read in the DataLoader worker.

Volumetric modalities are deliberately out of scope for now. A cardiac MRI *slice* is
an image and belongs here; a whole cine stack is a different contract -- a spacing, an
orientation, a slice axis -- and adding a fourth axis to these classes to half-support
it would make them wrong for both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from kalecardiac.loaddata.multimodal_access import ModalitySource


class ImageFormatError(ValueError):
    """Raised when an image does not have the shape it was declared with."""


def as_image_array(value: ArrayLike, channels: int) -> np.ndarray:
    """Coerce an image to ``(channels, height, width)`` float32.

    A two-dimensional input gains a leading axis, and a ``(height, width, channels)``
    input is moved -- but only when the trailing axis is the declared channel count and
    the leading one is not, because a ``3 x H x 3`` image is ambiguous and guessing
    would silently transpose real pixels.

    Args:
        value: The image, in any array-like form.
        channels: How many channels it should hold.

    Returns:
        A ``(channels, height, width)`` array.

    Raises:
        ImageFormatError: If the shape cannot be reconciled with ``channels``.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        if channels != 1:
            raise ImageFormatError(f"a 2-D image has one channel, but {channels} were declared")
        return array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ImageFormatError(f"expected a 2-D or 3-D image, got shape {array.shape}")

    if array.shape[0] == channels:
        return array
    if array.shape[-1] == channels:
        return np.ascontiguousarray(np.moveaxis(array, -1, 0))
    raise ImageFormatError(f"image of shape {array.shape} has no axis of length {channels}")


class ImageArraySource(ModalitySource):
    """Images held in memory, one per subject.

    Args:
        images: One ``(channels, height, width)`` array per subject, keyed by
            identifier. ``(height, width)`` and ``(height, width, channels)`` are
            accepted and reshaped. Subjects absent from the mapping are missing.
        channels: Channels each image holds.
        size: ``(height, width)`` after the transform, used for the placeholder and
            checked on every read. Inferred from the first image when omitted.
        transform: Applied to the ``(channels, height, width)`` array before it becomes
            a tensor. This is where a :mod:`kalecardiac.prepdata` pipeline plugs in.

    Raises:
        ImageFormatError: If ``size`` is omitted and ``images`` is empty.
    """

    def __init__(
        self,
        images: Mapping[str, ArrayLike],
        channels: int = 1,
        size: Sequence[int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.transform = transform
        # Held by reference rather than copied, as ECGArraySource holds its recordings.
        self.images = images

        if size is None:
            if not self.images:
                raise ImageFormatError("size must be given when images is empty; there is no shape to infer")
            first = next(iter(self.images))
            size = tuple(int(extent) for extent in self._read(first).shape[1:])
        self.size = tuple(int(extent) for extent in size)

    def _read(self, identifier: str) -> Tensor:
        array = as_image_array(self.images[identifier], self.channels)
        if self.transform is not None:
            array = np.asarray(self.transform(array), dtype=np.float32)
            if array.ndim != 3 or array.shape[0] != self.channels:
                raise ImageFormatError(
                    f"the transform returned shape {array.shape} for subject {identifier!r}; it must keep "
                    f"({self.channels}, height, width)"
                )
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def get(self, identifier: str, index: int) -> Tensor | None:
        if identifier not in self.images:
            return None
        image = self._read(identifier)
        if tuple(image.shape[1:]) != self.size:
            raise ImageFormatError(
                f"subject {identifier!r} has a {tuple(image.shape[1:])} image but the source declares "
                f"{self.size}. Resize the cohort to one size first; see kalecardiac.prepdata.resize_image."
            )
        return image

    def placeholder(self) -> Tensor:
        return torch.zeros(self.channels, *self.size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.images)} subjects | {self.channels}x{self.size[0]}x{self.size[1]})"


def _default_loader(path: Path) -> np.ndarray:
    """Read an image saved as ``.npy`` or as an ordinary picture file.

    Pillow covers PNG, JPEG and TIFF, which is what a preprocessed imaging cohort is
    usually exported as. DICOM is not handled here: a DICOM carries windowing and
    photometric interpretation that must be applied before the pixels mean anything,
    and doing that silently would be worse than asking for a ``loader``.
    """
    if path.suffix.lower() == ".npy":
        return np.load(path)
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f'reading {path.suffix} images needs pillow. Install it with: pip install "kalecardiac[imaging]"'
        ) from error
    with Image.open(path) as handle:
        return np.asarray(handle, dtype=np.float32)


class ImageFileSource(ImageArraySource):
    """Images read from one file per subject.

    The lazy counterpart of :class:`ImageArraySource`, for a cohort too large to hold
    in memory.

    Args:
        paths: One file per subject, keyed by identifier.
        channels: Channels each image holds after ``loader``.
        size: ``(height, width)`` after the transform. Inferred from the first file
            when omitted, which reads one file at construction.
        transform: Applied to the array; see :class:`ImageArraySource`.
        loader: Reads one file into an array. Defaults to ``.npy`` and anything Pillow
            opens. Pass one wrapping ``pydicom`` for DICOM, having decided how the
            pixels are windowed.
    """

    def __init__(
        self,
        paths: Mapping[str, str | Path],
        channels: int = 1,
        size: Sequence[int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        loader: Callable[[Path], ArrayLike] | None = None,
    ) -> None:
        self.paths = {str(key): Path(value) for key, value in paths.items()}
        self.loader = loader or _default_loader
        super().__init__(images=_LazyImages(self), channels=channels, size=size, transform=transform)

    def read_file(self, identifier: str) -> np.ndarray:
        """Read one subject's file.

        Args:
            identifier: Which subject.

        Returns:
            The raw array as stored, before reshaping.
        """
        return np.asarray(self.loader(self.paths[identifier]), dtype=np.float32)

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        path = self.paths.get(identifier)
        return {} if path is None else {"image_path": str(path)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.paths)} subjects | {self.channels}x{self.size[0]}x{self.size[1]})"


class _LazyImages(Mapping):
    """A read-through view of a file source's paths; see ``ecg_access._LazyRecordings``."""

    def __init__(self, source: ImageFileSource) -> None:
        self._source = source

    def __getitem__(self, identifier: str) -> np.ndarray:
        return self._source.read_file(identifier)

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._source.paths

    def __iter__(self):
        return iter(self._source.paths)

    def __len__(self) -> int:
        return len(self._source.paths)
