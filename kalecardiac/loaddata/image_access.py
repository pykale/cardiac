"""Access to cardiovascular images, whatever produced them.

Nothing here is specific to a chest X-ray. An image modality is a ``(channels, height,
width)`` tensor per subject, which is equally a radiograph, a short-axis cardiac MRI
slice, a CT section or an echocardiography frame; what differs between them is how the
file is read and what preprocessing it needs, and both of those are arguments rather
than assumptions.

Two sources, mirroring :mod:`kalecardiac.loaddata.signal_access`:

* :class:`ImageArraySource` for arrays already in memory;
* :class:`ImageFileSource` for one file per subject, read in the DataLoader worker --
  and, with :meth:`ImageFileSource.from_dicom`, for a folder of DICOM exports, one
  subject per folder.

**DICOM is read with PyKale's reader**, :func:`kale.loaddata.image_access.read_dicom_phases`,
which this module's :func:`read_dicom` calls for a subject's folder; the ECG sources read
their DICOM waveforms through it too. What is added here is what a model needs on top: the
subject's identifier taken from its folder (:func:`get_dicom_paths`), and the pixels as a
viewer shows them (:func:`read_dicom_image`).

Volumetric modalities are deliberately out of scope for now. A cardiac MRI *slice* is
an image and belongs here; a whole cine stack is a different contract -- a spacing, an
orientation, a slice axis -- and adding a fourth axis to these classes to half-support
it would make them wrong for both.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from kalecardiac.loaddata.multimodal_access import ModalitySource

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


class ImageFormatError(ValueError):
    """Raised when an image does not have the shape it was declared with."""


def _require_dicom():
    """Import pydicom and PyKale's DICOM reader, explaining the install if either is absent."""
    try:
        import pydicom
        from kale.loaddata.image_access import read_dicom_phases
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading DICOM needs pydicom and scikit-image, which PyKale's DICOM reader imports. Install them "
            'with: pip install "kalecardiac[imaging]"'
        ) from error
    return pydicom, read_dicom_phases


def get_dicom_paths(root: str | Path) -> dict[str, Path]:
    """Where each subject's DICOM is, under a folder holding one entry per subject.

    The layout a hospital export usually takes: a folder per subject, named by the
    subject's identifier, with the DICOM file anywhere beneath it --
    ``root/<subject>/1/DICOM/00000000.dcm``, say. A ``.dcm`` file directly under ``root``
    is a subject too, named by its stem.

    The folder's name, not the header's ``PatientID``, identifies the subject, because it
    is what pairs one modality's export with another's: the systems that export an ECG
    and a radiograph seldom write the same patient identifier into their headers.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as root:
        ...     for subject in ("S2", "S1"):
        ...         (Path(root) / subject / "1").mkdir(parents=True)
        ...     sorted(get_dicom_paths(root))
        ['S1', 'S2']

    Args:
        root: The folder holding one entry per subject.

    Returns:
        Each subject's folder or file, keyed by identifier, in sorted order. Hidden
        entries, and files that are not ``.dcm``, are skipped.

    Raises:
        FileNotFoundError: If ``root`` is not a folder.
        ValueError: If a folder and a file name the same subject.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no folder {root}; it should hold one folder (or .dcm file) per subject")
    paths: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subject = entry.name
        elif entry.suffix.lower() == ".dcm":
            subject = entry.stem
        else:
            continue
        if subject in paths:
            raise ValueError(f"{root} holds both {paths[subject].name} and {entry.name} for subject {subject!r}")
        paths[subject] = entry
    return paths


def read_dicom(path: str | Path) -> Dataset:
    """The one DICOM a subject's folder holds, read with PyKale's DICOM reader.

    A folder is searched for ``.dcm`` files at any depth by
    :func:`kale.loaddata.image_access.read_dicom_phases`, and must hold exactly one:
    with several, which of them is the subject's -- a frontal radiograph or a lateral
    one, a resting ECG or an exercise one -- would be decided by file order. A path to a
    ``.dcm`` file is read directly, with ``pydicom.dcmread``.

    Args:
        path: A subject's folder, or a ``.dcm`` file.

    Returns:
        The ``pydicom`` dataset.

    Raises:
        ImportError: If pydicom or scikit-image is not installed.
        FileNotFoundError: If ``path`` does not exist, or a folder holds no ``.dcm`` file.
        ValueError: If a folder holds more than one.
    """
    pydicom, read_dicom_phases = _require_dicom()
    path = Path(path)
    if path.is_file():
        return pydicom.dcmread(path)
    if not path.is_dir():
        raise FileNotFoundError(f"no DICOM file or folder at {path}")
    with warnings.catch_warnings():
        # PyKale's reader records each file's path on its dataset as FilePath, which pydicom
        # warns is not a DICOM keyword. It says nothing about the file, so it is not shown.
        warnings.filterwarnings("ignore", message="Camel case attribute 'FilePath'", category=UserWarning)
        datasets = read_dicom_phases(str(path), sort_instance=False)
    if not datasets:
        raise FileNotFoundError(f"no .dcm file under {path}")
    if len(datasets) > 1:
        names = sorted(Path(dataset.FilePath).name for dataset in datasets)
        raise ValueError(
            f"{path} holds {len(datasets)} DICOM files {names[:3]}; keep one per subject's folder, or read the "
            f"one you mean by its path"
        )
    return datasets[0]


def read_dicom_image(path: str | Path) -> np.ndarray:
    """One image from DICOM, its pixels as a viewer displays them.

    A DICOM image stores what the detector measured, and separately how to display it.
    Three steps turn the first into the second, applied here in the order the standard
    gives them, through pydicom:

    1. the modality LUT -- usually a rescale slope and intercept -- which maps stored
       values to the modality's units;
    2. the VOI LUT or window, which selects the range of those values the image is read
       in;
    3. for ``MONOCHROME1``, an inversion, so that a higher value is always brighter.

    What a model then reads is the radiograph a clinician reads, whichever machine and
    export produced it. A colour image is returned as stored.

    Args:
        path: A subject's folder holding one DICOM image, or a ``.dcm`` file; see
            :func:`read_dicom`.

    Returns:
        A ``(height, width)`` float32 array, or ``(height, width, 3)`` for colour.

    Raises:
        ImageFormatError: If the DICOM holds no pixels -- an ECG waveform, say -- or holds
            several frames, which make a volume rather than an image.
    """
    dataset = read_dicom(path)
    if "PixelData" not in dataset:
        kind = getattr(dataset.get("SOPClassUID"), "name", "a DICOM without pixel data")
        raise ImageFormatError(f"{path} holds no image: it is {kind}")
    frames = int(dataset.get("NumberOfFrames") or 1)
    if frames > 1:
        raise ImageFormatError(f"{path} holds {frames} frames, which make a volume; this reads one image per subject")

    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    pixels = dataset.pixel_array
    photometric = dataset.get("PhotometricInterpretation", "")
    if photometric in ("MONOCHROME1", "MONOCHROME2"):
        pixels = apply_voi_lut(apply_modality_lut(pixels, dataset), dataset)
        if photometric == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
    return np.asarray(pixels, dtype=np.float32)


def check_image_array(value: ArrayLike, channels: int) -> np.ndarray:
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
        array = check_image_array(self.images[identifier], self.channels)
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
    usually exported as. DICOM is not guessed from a suffix: reading one means applying
    its windowing and photometric interpretation, which :func:`read_dicom_image` does and
    says it does, so a DICOM file asks for it by name.
    """
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        raise ImageFormatError(
            f"{path.name} is DICOM; pass loader=read_dicom_image, or build the source with ImageFileSource.from_dicom"
        )
    if suffix == ".npy":
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
            opens. Pass :func:`read_dicom_image` for DICOM files -- or build the source
            from a folder of them with :meth:`from_dicom`.
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

    @classmethod
    def from_dicom(
        cls,
        root: str | Path,
        channels: int = 1,
        size: Sequence[int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> ImageFileSource:
        """One image per subject, from a folder of DICOM exports, read as each subject is needed.

        ``root`` holds a folder (or a ``.dcm`` file) per subject, named by its identifier --
        see :func:`get_dicom_paths` -- and each subject's image is read by
        :func:`read_dicom_image`, through PyKale's DICOM reader, in the DataLoader worker.

        Args:
            root: The folder of subjects.
            channels: Channels each image holds; 1 for a radiograph.
            size: ``(height, width)`` after the transform. Inferred from the first subject
                when omitted, which reads that subject's file.
            transform: Applied to each image; see :class:`ImageArraySource`.

        Returns:
            The source, keyed by subject.

        Raises:
            ImageFormatError: If ``root`` holds no subject.
        """
        paths = get_dicom_paths(root)
        if not paths:
            raise ImageFormatError(f"no subject folder or .dcm file under {root}")
        return cls(paths, channels=channels, size=size, transform=transform, loader=read_dicom_image)

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
    """A read-through view of a file source's paths; see ``signal_access._LazyRecordings``."""

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
