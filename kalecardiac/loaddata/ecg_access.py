"""Access to electrocardiogram recordings, by lead.

An ECG is a set of named leads sampled together, so the two things this module insists
on are the *names* and the *rate*. Both are declared by the caller rather than assumed:
a cohort may hold twelve leads, the six limb leads a reduced-lead device records, or a
single rhythm strip, and a source that assumed one of those would quietly transpose,
reorder or truncate the others.

Two sources cover where recordings come from:

* :class:`ECGArraySource` for arrays already in memory, which is what a preprocessed
  cohort and every synthetic fixture look like;
* :class:`ECGFileSource` for one file per subject, read in the DataLoader worker.

and :class:`LeadSource` turns either of them into *one lead*, which is the mechanism
behind treating each lead as its own modality::

    >>> sources = lead_sources(ecg, leads=["I", "II", "V1"])
    >>> sorted(sources)
    ['I', 'II', 'V1']

That is all "lead-specific multimodal learning" needs from the loading stage: after it,
a twelve-lead cohort is an ordinary thirteen-source :class:`~kalecardiac.loaddata.MultimodalDataset`
and nothing downstream knows the modalities happen to be leads.

**Lead names are matched case-insensitively and without a ``LEAD_`` prefix**, because
the same lead is written ``aVR``, ``AVR`` and ``LEAD_aVR`` by three different export
tools, and a cohort silently missing a lead because of its spelling is the failure this
prevents.
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

#: The six leads derived from the limb electrodes, in their conventional order. A
#: reduced-lead recorder captures these, which is why transferring a model trained on
#: twelve leads to six is a question worth asking of a cardiac library.
LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")

#: The six chest leads, in their conventional order.
PRECORDIAL_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")

#: The standard diagnostic twelve-lead order.
STANDARD_12_LEAD: tuple[str, ...] = LIMB_LEADS + PRECORDIAL_LEADS


class ECGFormatError(ValueError):
    """Raised when a recording does not have the shape or leads it was declared with."""


def canonical_lead(name: str) -> str:
    """Return the canonical spelling of a lead name.

    Strips a ``LEAD_`` prefix and matches case-insensitively against
    :data:`STANDARD_12_LEAD`, so ``"LEAD_avr"``, ``"AVR"`` and ``"aVR"`` all give
    ``"aVR"``. A name that is not a standard lead is returned with only the prefix
    stripped, which keeps the function usable for a cohort with its own vocabulary.

    Args:
        name: Lead name as the cohort spells it.

    Returns:
        The canonical spelling.
    """
    stripped = str(name).strip()
    if stripped.upper().startswith("LEAD_"):
        stripped = stripped[len("LEAD_") :]
    for standard in STANDARD_12_LEAD:
        if stripped.upper() == standard.upper():
            return standard
    return stripped


def canonical_leads(names: Sequence[str]) -> tuple[str, ...]:
    """Canonicalise a sequence of lead names, rejecting duplicates.

    Args:
        names: Lead names as the cohort spells them.

    Returns:
        The canonical spellings, in the order given.

    Raises:
        ECGFormatError: If ``names`` is empty, or two names canonicalise to the same
            lead -- which means the cohort has stored one lead twice under different
            spellings, and choosing between them is not this function's decision.
    """
    if not names:
        raise ECGFormatError("an ECG source must declare its leads; there is no default lead set")
    canonical = tuple(canonical_lead(name) for name in names)
    duplicates = sorted({lead for lead in canonical if canonical.count(lead) > 1})
    if duplicates:
        raise ECGFormatError(f"lead names {duplicates} appear more than once after canonicalisation: {list(names)}")
    return canonical


def lead_indices(available: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """Positions of ``wanted`` within ``available``, matched canonically.

    The one place lead selection and lead *ordering* are decided, so that asking for
    ``["II", "I"]`` returns them in that order rather than in the recording's.

    Args:
        available: Leads the recording holds, in storage order.
        wanted: Leads to take, in the order they should come out.

    Returns:
        Index into ``available`` for each entry of ``wanted``.

    Raises:
        ECGFormatError: If a wanted lead is not available.
    """
    positions = {lead: index for index, lead in enumerate(canonical_leads(available))}
    chosen = canonical_leads(wanted)
    missing = [lead for lead in chosen if lead not in positions]
    if missing:
        raise ECGFormatError(f"leads {missing} are not in this recording, which holds {sorted(positions)}")
    return [positions[lead] for lead in chosen]


def as_lead_array(value: ArrayLike, num_leads: int) -> np.ndarray:
    """Coerce a recording to ``(num_leads, num_samples)`` float32.

    A one-dimensional input is accepted for a single-lead recording, and a
    ``(num_samples, num_leads)`` input is transposed -- but only when the two axes
    differ in length, because a square recording is genuinely ambiguous and guessing
    at it would transpose real data without anybody noticing.

    Args:
        value: The recording, in any array-like form.
        num_leads: How many leads it should hold.

    Returns:
        A ``(num_leads, num_samples)`` array.

    Raises:
        ECGFormatError: If the shape cannot be reconciled with ``num_leads``.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        if num_leads != 1:
            raise ECGFormatError(f"a 1-D recording holds one lead, but {num_leads} were declared")
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ECGFormatError(f"expected a 2-D (leads, samples) recording, got shape {array.shape}")

    if array.shape[0] == num_leads:
        return array
    if array.shape[1] == num_leads:
        return np.ascontiguousarray(array.T)
    raise ECGFormatError(
        f"recording of shape {array.shape} has no axis of length {num_leads}; check the declared leads"
    )


class ECGArraySource(ModalitySource):
    """ECG recordings held in memory, one per subject.

    The general case once a cohort has been preprocessed into arrays, and what every
    test fixture here uses. Nothing is assumed about the number of leads or the
    length: both come from what the caller declares, and every recording is checked
    against them as it is read.

    Args:
        recordings: One ``(num_leads, num_samples)`` array per subject, keyed by
            identifier. A ``(num_samples, num_leads)`` array is transposed, and a 1-D
            array is accepted when a single lead is declared. Subjects absent from the
            mapping are treated as missing.
        leads: Lead names in the order the arrays store them.
        sampling_rate: Samples per second, carried as provenance so an attribution or
            a delineation can be placed in time.
        select: Leads to take, in the order they should be returned. ``None`` keeps
            every lead in storage order.
        transform: Applied to the selected ``(num_leads, num_samples)`` array before it
            becomes a tensor. This is where a :mod:`kalecardiac.prepdata` pipeline
            plugs in, and where any fold-local fitted state would have to arrive, so it
            is an argument rather than cohort state.

            A transform **may change the number of channels**: deriving the limb leads
            from two recorded ones, or flattening every lead into a single long signal,
            are both ordinary preprocessing. The source measures what comes out and
            holds every later recording to it, so an inconsistency is caught at the
            subject that caused it. :attr:`leads` then names what was *selected* and
            :attr:`channels` what is *returned*; where a transform has made them
            differ, splitting the source by lead is no longer meaningful and is refused.
        length: Samples per channel the transform is expected to produce, used for the
            placeholder and checked on every read. Inferred from the first recording
            when omitted.
        channels: Channels the transform is expected to produce. Inferred alongside
            ``length``, and equal to the number of selected leads when there is no
            transform.

    Raises:
        ECGFormatError: If ``leads`` is empty or repeats, if ``select`` names a lead
            that is not there, or if ``length`` is omitted and ``recordings`` is empty.
    """

    def __init__(
        self,
        recordings: Mapping[str, ArrayLike],
        leads: Sequence[str],
        sampling_rate: float = 500.0,
        select: Sequence[str] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        length: int | None = None,
        channels: int | None = None,
    ) -> None:
        self.source_leads = canonical_leads(leads)
        self.leads = self.source_leads if select is None else canonical_leads(select)
        self._indices = lead_indices(self.source_leads, self.leads)
        self.sampling_rate = float(sampling_rate)
        self.transform = transform
        # Held by reference rather than copied: a file-backed source passes a
        # read-through mapping, and copying it would read the whole cohort here.
        self.recordings = recordings

        if length is None or channels is None:
            if not self.recordings:
                if length is None:
                    raise ECGFormatError("length must be given when recordings is empty; there is no shape to infer")
                channels = len(self.leads) if channels is None else channels
            else:
                probe = self._read(next(iter(self.recordings)))
                channels = int(probe.shape[0]) if channels is None else channels
                length = int(probe.shape[1]) if length is None else length
        self.length = int(length)
        self.channels = int(channels)

    @property
    def num_leads(self) -> int:
        """How many leads this source selects, before any transform."""
        return len(self.leads)

    @property
    def splits_by_lead(self) -> bool:
        """Whether this source's channels still correspond one-to-one with its leads.

        False once a transform has reshaped them -- by flattening the leads into one
        signal, say -- at which point asking for "lead aVR" of the output is asking for
        something that is no longer there.
        """
        return self.channels == self.num_leads

    def _read(self, identifier: str) -> Tensor:
        array = as_lead_array(self.recordings[identifier], len(self.source_leads))
        array = array[self._indices]
        if self.transform is not None:
            array = np.asarray(self.transform(array), dtype=np.float32)
            if array.ndim != 2:
                raise ECGFormatError(
                    f"the transform returned shape {array.shape} for subject {identifier!r}; it must return a "
                    f"2-D (channels, num_samples) array"
                )
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def get(self, identifier: str, index: int) -> Tensor | None:
        if identifier not in self.recordings:
            return None
        recording = self._read(identifier)
        if tuple(recording.shape) != (self.channels, self.length):
            raise ECGFormatError(
                f"subject {identifier!r} yields a {tuple(recording.shape)} recording but the source declares "
                f"({self.channels}, {self.length}). Every subject must reach a batch at one shape; crop or pad "
                f"the cohort first, with kalecardiac.prepdata.crop_or_pad."
            )
        return recording

    def placeholder(self) -> Tensor:
        return torch.zeros(self.channels, self.length)

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        return {"sampling_rate": self.sampling_rate, "leads": self.leads}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self.recordings)} subjects | leads={list(self.leads)} | "
            f"shape={self.channels}x{self.length} | {self.sampling_rate:g} Hz)"
        )


def _default_loader(path: Path) -> np.ndarray:
    """Read a recording saved as ``.npy`` or ``.pt``.

    Two formats because those are what a preprocessing script leaves behind. Anything
    else -- WFDB, DICOM waveforms, a vendor export -- is a *dataset's* format, so it
    arrives as a ``loader`` argument rather than as a branch here.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=True).numpy()
    raise ECGFormatError(
        f"no default loader for {path.suffix!r}; pass loader= to read this format, e.g. one wrapping wfdb.rdsamp"
    )


class ECGFileSource(ECGArraySource):
    """ECG recordings read from one file per subject.

    The lazy counterpart of :class:`ECGArraySource`, for a cohort too large to hold in
    memory. Files are read in the DataLoader worker, and the most recent read is kept,
    so exploding one recording into per-lead modalities costs one read per subject
    rather than one per lead.

    Args:
        paths: One file per subject, keyed by identifier.
        leads: Lead names in the order the files store them.
        sampling_rate: Samples per second.
        select: Leads to take, in the order they should be returned.
        transform: Applied to the selected array; see :class:`ECGArraySource`.
        length: Samples per lead after the transform. Inferred from the first file
            when omitted, which reads one file at construction.
        loader: Reads one file into an array. Defaults to ``.npy`` and ``.pt``.

    Raises:
        ECGFormatError: As :class:`ECGArraySource`, or if a file cannot be read.
    """

    def __init__(
        self,
        paths: Mapping[str, str | Path],
        leads: Sequence[str],
        sampling_rate: float = 500.0,
        select: Sequence[str] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        length: int | None = None,
        channels: int | None = None,
        loader: Callable[[Path], ArrayLike] | None = None,
    ) -> None:
        self.paths = {str(key): Path(value) for key, value in paths.items()}
        self.loader = loader or _default_loader
        self._cached: tuple[str, np.ndarray] | None = None
        super().__init__(
            recordings=_LazyRecordings(self),
            leads=leads,
            sampling_rate=sampling_rate,
            select=select,
            transform=transform,
            length=length,
            channels=channels,
        )

    def read_file(self, identifier: str) -> np.ndarray:
        """Read one subject's file, reusing the most recent read.

        Args:
            identifier: Which subject.

        Returns:
            The raw array as stored, before lead selection.
        """
        if self._cached is not None and self._cached[0] == identifier:
            return self._cached[1]
        array = np.asarray(self.loader(self.paths[identifier]), dtype=np.float32)
        self._cached = (identifier, array)
        return array

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self.paths)} subjects | leads={list(self.leads)} | "
            f"shape={self.channels}x{self.length} | {self.sampling_rate:g} Hz)"
        )


class _LazyRecordings(Mapping):
    """A read-through view of a file source's paths, so it can reuse the array source.

    :class:`ECGArraySource` asks its ``recordings`` only for membership, iteration and
    one subject's array at a time, which a file source can answer without holding any
    of them. Keeping it a mapping is what lets the lead selection, the transform and
    the length check live in one place instead of two.
    """

    def __init__(self, source: ECGFileSource) -> None:
        self._source = source

    def __getitem__(self, identifier: str) -> np.ndarray:
        return self._source.read_file(identifier)

    def __contains__(self, identifier: object) -> bool:
        # Delegated rather than inherited: Mapping.__contains__ probes __getitem__,
        # which here would read a file merely to answer whether one exists.
        return identifier in self._source.paths

    def __iter__(self):
        return iter(self._source.paths)

    def __len__(self) -> int:
        return len(self._source.paths)


class LeadSource(ModalitySource):
    """A single lead of an ECG source, presented as its own modality.

    This is the whole mechanism behind lead-specific multimodal learning: the model
    sees a named modality of shape ``(1, num_samples)`` and never learns that its
    siblings are the other leads of the same recording.

    Args:
        source: The recording source to take a lead from.
        lead: Which lead, matched canonically against the source's leads.

    Raises:
        ECGFormatError: If ``source`` does not hold that lead.
    """

    def __init__(self, source: ECGArraySource, lead: str) -> None:
        if not source.splits_by_lead:
            raise ECGFormatError(
                f"this source returns {source.channels} channels for {source.num_leads} selected leads, so a "
                f"transform has reshaped them and its channels are no longer leads. Split the recording into "
                f"leads before flattening or deriving, not after."
            )
        self.source = source
        self.lead = canonical_lead(lead)
        self._index = lead_indices(source.leads, [self.lead])[0]

    def get(self, identifier: str, index: int) -> Tensor | None:
        recording = self.source.get(identifier, index)
        return None if recording is None else recording[self._index : self._index + 1]

    def placeholder(self) -> Tensor:
        return torch.zeros(1, self.source.length)

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        return {"sampling_rate": self.source.sampling_rate}

    def __repr__(self) -> str:
        return f"LeadSource({self.lead} of {self.source!r})"


def lead_sources(
    source: ECGArraySource,
    leads: Sequence[str] | None = None,
    prefix: str = "",
) -> dict[str, LeadSource]:
    """Split one ECG source into a named source per lead.

    The loading half of "each lead is a modality". Pass the result straight to
    :class:`~kalecardiac.loaddata.MultimodalDataset` and every stage after it treats
    the leads as it treats any other set of modalities -- which is what lets a model
    pretrained on twelve of them be fine-tuned on six by naming six here.

    Args:
        source: The recording source.
        leads: Which leads to expose, in order. ``None`` exposes every lead the source
            returns.
        prefix: Prepended to each modality name, for a cohort with more than one
            recording per subject (``"rest_"``, ``"stress_"``).

    Returns:
        One :class:`LeadSource` per lead, keyed by ``prefix`` plus the canonical lead
        name.

    Raises:
        ECGFormatError: If a named lead is not in ``source``.
    """
    chosen = source.leads if leads is None else canonical_leads(leads)
    return {f"{prefix}{lead}": LeadSource(source, lead) for lead in chosen}
