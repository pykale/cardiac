"""Access to electrocardiogram recordings, by lead.

An ECG is a set of named leads sampled together, so the two things this module insists
on are the *names* and the *rate*. Both are declared by the caller rather than assumed:
a cohort may hold twelve leads, the six limb leads a reduced-lead device records, or a
single rhythm strip, and a source that assumed one of those would quietly transpose,
reorder or truncate the others.

Two sources cover where recordings come from:

* :class:`ECGArraySource` for arrays already in memory, which is what a preprocessed
  cohort and every synthetic fixture look like;
* :class:`ECGFileSource` for one file per subject, read in the DataLoader worker -- and,
  with :meth:`ECGFileSource.from_dicom`, for a folder of DICOM ECG exports, whose
  headers name every lead, so the names are read rather than declared.

and :class:`LeadSource` turns either of them into *one lead*, which is the mechanism
behind treating each lead as its own modality::

    >>> ecg = ECGArraySource({"s1": np.zeros((3, 500))}, leads=["I", "II", "V1"])
    >>> sources = get_lead_sources(ecg)
    >>> sorted(sources)
    ['I', 'II', 'V1']
    >>> sources["V1"].get("s1", 0).shape
    torch.Size([1, 500])

That is all "lead-specific multimodal learning" needs from the loading stage: after it,
a twelve-lead cohort is an ordinary twelve-source :class:`~kalecardiac.loaddata.MultimodalDataset`
and nothing downstream knows the modalities happen to be leads.

**Lead names are matched case-insensitively and without a ``LEAD_`` prefix**, because
the same lead is written ``aVR``, ``AVR`` and ``LEAD_aVR`` by three different export
tools -- and ``Lead aVR`` by a DICOM header -- and a cohort silently missing a lead
because of its spelling is the failure this prevents.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from kalecardiac.loaddata.image_access import get_dicom_paths, read_dicom
from kalecardiac.loaddata.multimodal_access import ModalitySource

#: The six leads derived from the limb electrodes, in their conventional order. A
#: reduced-lead recorder captures these, which is why transferring a model trained on
#: twelve leads to six is a question worth asking of a cardiac library.
LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")

#: The six chest leads, in their conventional order.
PRECORDIAL_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")

#: The standard diagnostic twelve-lead order.
STANDARD_12_LEAD: tuple[str, ...] = LIMB_LEADS + PRECORDIAL_LEADS


#: Millivolts in one unit of each voltage a DICOM waveform may state its channels in,
#: by UCUM code.
_MILLIVOLTS_PER_UNIT: dict[str, float] = {"uV": 1e-3, "mV": 1.0, "V": 1e3}

#: The prefix export tools put before a lead's name: ``LEAD_aVR``, ``Lead aVR``.
_LEAD_PREFIX = re.compile(r"^lead[\s_]+(?=\S)", re.IGNORECASE)

#: A parenthesised qualifier after a lead's name, as in DICOM's ``Lead I (Einthoven)``.
_QUALIFIER = re.compile(r"\s*\(.*\)$")


class ECGFormatError(ValueError):
    """Raised when a recording does not have the shape or leads it was declared with."""


def normalize_lead_name(name: str) -> str:
    """Return the canonical spelling of a lead name.

    Strips a ``LEAD_`` or ``Lead`` prefix, and matches case-insensitively against
    :data:`STANDARD_12_LEAD` -- with or without a parenthesised qualifier -- so
    ``"LEAD_avr"``, ``"AVR"`` and ``"aVR"`` all give ``"aVR"``, and DICOM's
    ``"Lead I (Einthoven)"`` gives ``"I"``. A name that is not a standard lead is returned
    with only the prefix stripped, which keeps the function usable for a cohort with its
    own vocabulary.

        >>> normalize_lead_name("LEAD_avr"), normalize_lead_name("Lead I (Einthoven)")
        ('aVR', 'I')

    Args:
        name: Lead name as the cohort spells it.

    Returns:
        The canonical spelling.
    """
    stripped = _LEAD_PREFIX.sub("", str(name).strip())
    for candidate in (stripped, _QUALIFIER.sub("", stripped)):
        for standard in STANDARD_12_LEAD:
            if candidate.upper() == standard.upper():
                return standard
    return stripped


def normalize_lead_names(names: Sequence[str]) -> tuple[str, ...]:
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
    canonical = tuple(normalize_lead_name(name) for name in names)
    duplicates = sorted({lead for lead in canonical if canonical.count(lead) > 1})
    if duplicates:
        raise ECGFormatError(f"lead names {duplicates} appear more than once after canonicalisation: {list(names)}")
    return canonical


def get_lead_indices(available: Sequence[str], wanted: Sequence[str]) -> list[int]:
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
    positions = {lead: index for index, lead in enumerate(normalize_lead_names(available))}
    chosen = normalize_lead_names(wanted)
    missing = [lead for lead in chosen if lead not in positions]
    if missing:
        raise ECGFormatError(f"leads {missing} are not in this recording, which holds {sorted(positions)}")
    return [positions[lead] for lead in chosen]


def check_ecg_array(value: ArrayLike, num_leads: int) -> np.ndarray:
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
        self.source_leads = normalize_lead_names(leads)
        self.leads = self.source_leads if select is None else normalize_lead_names(select)
        self._indices = get_lead_indices(self.source_leads, self.leads)
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
        array = check_ecg_array(self.recordings[identifier], len(self.source_leads))
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
                f"the cohort first, with kalecardiac.prepdata.crop_or_pad_signal."
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


def read_dicom_ecg(path: str | Path, group: int | None = None) -> tuple[np.ndarray, tuple[str, ...], float]:
    """One ECG from DICOM: the recording, the leads it holds, and the rate they were sampled at.

    DICOM stores an ECG as one or more groups of channels, each channel naming its lead
    in a code -- ``"Lead I (Einthoven)"``, ``"Lead aVR"``. The names come back canonical
    (see :func:`normalize_lead_name`), so a lead is found by its name rather than its
    position: two exports storing aVL and aVF in opposite orders give the same leads.

    The file is read by :func:`~kalecardiac.loaddata.read_dicom`, through PyKale's DICOM
    reader, and the samples by ``pydicom``, which applies each channel's sensitivity and
    baseline.

    Args:
        path: A subject's folder holding one DICOM ECG, or a ``.dcm`` file.
        group: Which group of channels to read. ``None`` reads the first one recorded
            rather than derived -- the rhythm strip, not the median beat an ECG cart
            computes from it.

    Returns:
        ``(recording, leads, sampling_rate)``: a ``(num_leads, num_samples)`` float32
        array, in millivolts wherever a channel states its sensitivity in a voltage; the
        leads' canonical names, in stored order; and samples per second.

    Raises:
        ECGFormatError: If the DICOM holds no waveform, ``group`` does not exist, a
            channel names no lead or two name the same one, or a channel's unit is not a
            voltage.
    """
    dataset = read_dicom(path)
    groups = dataset.get("WaveformSequence")
    if not groups:
        kind = getattr(dataset.get("SOPClassUID"), "name", "a DICOM without a waveform")
        raise ECGFormatError(f"{path} holds no waveform: it is {kind}")
    if group is None:
        group = next((index for index, item in enumerate(groups) if item.get("WaveformOriginality") == "ORIGINAL"), 0)
    if not 0 <= group < len(groups):
        raise ECGFormatError(f"{path} holds {len(groups)} waveform group(s), so there is no group {group}")

    channels = groups[group].ChannelDefinitionSequence
    try:
        leads = normalize_lead_names([_dicom_lead_name(channel) for channel in channels])
    except ECGFormatError as error:
        raise ECGFormatError(f"{path}: {error}") from error
    millivolts = np.array([_millivolts_per_unit(channel, path) for channel in channels])
    recording = dataset.waveform_array(group) * millivolts
    return np.ascontiguousarray(recording.T, dtype=np.float32), leads, float(groups[group].SamplingFrequency)


def _dicom_lead_name(channel) -> str:
    """The lead a DICOM waveform channel records, as its source code or its label names it."""
    source = channel.get("ChannelSourceSequence")
    name = (source[0].get("CodeMeaning") if source else None) or channel.get("ChannelLabel")
    if not name:
        raise ECGFormatError("a channel names no lead: it has no source code meaning and no label")
    return str(name)


def _millivolts_per_unit(channel, path: str | Path) -> float:
    """Millivolts in one of a DICOM channel's sample units; 1 where it states no sensitivity, leaving it as stored."""
    units = channel.get("ChannelSensitivityUnitsSequence")
    if "ChannelSensitivity" not in channel or not units:
        return 1.0
    code = str(units[0].CodeValue)
    if code not in _MILLIVOLTS_PER_UNIT:
        raise ECGFormatError(
            f"{path} states a channel in {code!r}, which is not a voltage this reader converts; it reads "
            f"{sorted(_MILLIVOLTS_PER_UNIT)}"
        )
    return _MILLIVOLTS_PER_UNIT[code]


def _read_dicom_leads(path: Path, leads: Sequence[str], sampling_rate: float, group: int | None) -> np.ndarray:
    """One subject's DICOM ECG with its leads in the cohort's order, held to the cohort's rate."""
    recording, stored, rate = read_dicom_ecg(path, group=group)
    if rate != sampling_rate:
        raise ECGFormatError(
            f"{path} is sampled at {rate:g} Hz and the cohort's first subject at {sampling_rate:g} Hz; bring them to "
            f"one rate before loading -- see kalecardiac.prepdata.resample_signal"
        )
    try:
        return recording[get_lead_indices(stored, leads)]
    except ECGFormatError as error:
        raise ECGFormatError(f"{path}: {error}") from error


def _default_loader(path: Path) -> np.ndarray:
    """Read a recording saved as ``.npy`` or ``.pt``.

    Two formats because those are what a preprocessing script leaves behind. A DICOM ECG
    names its own leads, so it is read by :meth:`ECGFileSource.from_dicom`, which takes
    the names from it. Anything else -- WFDB, a vendor export -- is a *dataset's* format,
    so it arrives as a ``loader`` argument rather than as a branch here.
    """
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        raise ECGFormatError(
            f"{path.name} is DICOM, which names its own leads; build the source with ECGFileSource.from_dicom"
        )
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
        loader: Reads one file into an array. Defaults to ``.npy`` and ``.pt``. For a
            folder of DICOM ECGs, use :meth:`from_dicom`, which reads the leads' names
            from the files.

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

    @classmethod
    def from_dicom(
        cls,
        root: str | Path,
        select: Sequence[str] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        length: int | None = None,
        channels: int | None = None,
        sampling_rate: float | None = None,
        group: int | None = None,
    ) -> ECGFileSource:
        """One ECG per subject, from a folder of DICOM exports, read as each subject is needed.

        ``root`` holds a folder (or a ``.dcm`` file) per subject, named by its identifier
        -- see :func:`~kalecardiac.loaddata.get_dicom_paths` -- and each subject's ECG is
        read by :func:`read_dicom_ecg`, through PyKale's DICOM reader, in the DataLoader
        worker.

        The leads and their rate are read from the first subject's header rather than
        declared, and every other subject is held to them: its leads are found by name and
        returned in the first subject's order, so an export that stores aVF before aVL
        cannot transpose them, and a subject recorded at another rate or missing a lead
        is an error that names its file.

        Args:
            root: The folder of subjects.
            select: Leads to take, in the order the model reads them. ``None`` keeps every
                lead, in the order the first subject stores them.
            transform: Applied to the selected leads; see :class:`ECGArraySource`.
            length: Samples per channel after the transform. Inferred from the first
                subject when omitted.
            channels: Channels after the transform. Inferred alongside ``length``.
            sampling_rate: Rate of what the source returns, recorded as provenance.
                ``None`` takes the header's, which is right unless ``transform`` resamples.
            group: Which group of channels to read; see :func:`read_dicom_ecg`.

        Returns:
            The source, keyed by subject.

        Raises:
            ECGFormatError: If ``root`` holds no subject, or the first subject's ECG
                cannot be read.
        """
        paths = get_dicom_paths(root)
        if not paths:
            raise ECGFormatError(f"no subject folder or .dcm file under {root}")
        _, leads, rate = read_dicom_ecg(next(iter(paths.values())), group=group)
        return cls(
            paths,
            leads=leads,
            sampling_rate=rate if sampling_rate is None else sampling_rate,
            select=select,
            transform=transform,
            length=length,
            channels=channels,
            # A partial of a module-level function, rather than a closure, so the source
            # pickles into DataLoader workers on a platform that spawns them.
            loader=partial(_read_dicom_leads, leads=leads, sampling_rate=rate, group=group),
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
        self.lead = normalize_lead_name(lead)
        self._index = get_lead_indices(source.leads, [self.lead])[0]

    def get(self, identifier: str, index: int) -> Tensor | None:
        recording = self.source.get(identifier, index)
        return None if recording is None else recording[self._index : self._index + 1]

    def placeholder(self) -> Tensor:
        return torch.zeros(1, self.source.length)

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        return {"sampling_rate": self.source.sampling_rate}

    def __repr__(self) -> str:
        return f"LeadSource({self.lead} of {self.source!r})"


def get_lead_sources(
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
    chosen = source.leads if leads is None else normalize_lead_names(leads)
    return {f"{prefix}{lead}": LeadSource(source, lead) for lead in chosen}
