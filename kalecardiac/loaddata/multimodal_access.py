"""Generic access to any combination of per-subject modalities.

A cohort is a set of subjects and, for each, some number of *sources* that can be
asked for that subject's data. What those sources are is the caller's business::

    >>> dataset = MultimodalDataset(
    ...     identifiers,
    ...     sources={
    ...         "ecg": ECGArraySource(recordings, leads=LEADS_12),
    ...         "cxr": ImageArraySource(images),
    ...         "clinical": ArraySource(vectors),
    ...     },
    ...     target=target,
    ... )

Nothing about that is specific to ECG plus imaging, or to two modalities. Twelve
single-lead sources, one recording source, an image and a clinical vector, or any mix
of them are the same call with a different dictionary, because the sources are named
rather than positional and each is asked for one subject at a time. Adding a modality
is adding a dictionary entry; adding a *kind* of modality is implementing
:class:`ModalitySource`.

That is what makes "every lead is a modality" a configuration rather than a model
rewrite: :func:`kalecardiac.loaddata.lead_sources` turns one ECG source into twelve
named ones, and everything downstream sees an ordinary multimodal cohort.

**Missing data is expected, not exceptional.** A source returns ``None`` for a subject
it has nothing for, and the dataset substitutes a zero placeholder of the right shape
while recording the absence in ``present``. The placeholder is never evidence: fusion
reads ``present`` and drops the modality for that subject. That is what lets one
cohort hold subjects with twelve leads and subjects with six, or subjects with a chest
X-ray and subjects without.

The record types every access API in this package produces live here too --
:class:`SubjectSample`, :class:`SubjectBatch` and the collation between them --
because assembling a subject is what they exist for.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

#: Batch target key for a binary or multiclass endpoint.
LABEL_KEY = "label"

#: Batch target key for a continuous endpoint, such as a measured pressure.
VALUE_KEY = "value"


@dataclass(slots=True)
class SubjectSample:
    """One subject, every modality, ready for a model.

    Tensors only -- files are read inside the source, in the DataLoader worker.

    Attributes:
        subject_id (str): Identifier, so a prediction traces back to a subject.
        modalities (dict[str, Tensor]): Features by modality, e.g. ``{"ecg":
            (n_leads, n_samples), "cxr": (c, h, w)}``.
        present (dict[str, Tensor]): 0-d bool per modality. An absent modality is
            still present in ``modalities``, zero-filled, to keep batches uniform;
            this is what tells a fusion layer to ignore those zeros.
        target (dict[str, Tensor]): Supervision values. Empty for an unlabelled
            cohort, which is what a pretraining run uses.
        metadata (dict[str, Any]): Provenance that is *not* model input -- a sampling
            rate, a lead order, a source filename. Carried so an attribution can be
            placed back on the recording it came from, and ignored by every model.
    """

    subject_id: str
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    target: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubjectBatch:
    """A collated batch. The same fields as :class:`SubjectSample`, batched.

    Attributes:
        subject_id (list[str]): Identifiers, in batch order.
        modalities (dict[str, Tensor]): Leading dimension ``B``.
        present (dict[str, Tensor]): ``(B,)`` bool per modality.
        target (dict[str, Tensor]): ``(B,)`` per key.
        metadata (dict[str, list]): Per-subject provenance in batch order, one list
            per key. Never model input; see :class:`SubjectSample`.
    """

    subject_id: list[str]
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    target: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, list] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.subject_id)

    def to(self, device: torch.device | str) -> SubjectBatch:
        """Return a copy with every tensor moved to ``device``; metadata is left alone."""
        return SubjectBatch(
            subject_id=self.subject_id,
            modalities={name: value.to(device) for name, value in self.modalities.items()},
            present={name: value.to(device) for name, value in self.present.items()},
            target={key: value.to(device) for key, value in self.target.items()},
            metadata=self.metadata,
        )


def _require_same_keys(samples: list[SubjectSample], attribute: str) -> list[str]:
    """Return the shared keys of ``attribute`` across samples, or raise.

    Disagreement means a cohort built them inconsistently; collating anyway would
    silently drop a modality for the whole batch.
    """
    first = list(getattr(samples[0], attribute))
    expected = set(first)
    for sample in samples[1:]:
        found = set(getattr(sample, attribute))
        if found != expected:
            raise ValueError(
                f"samples disagree about {attribute}: {sample.subject_id!r} has {sorted(found)}, "
                f"{samples[0].subject_id!r} has {sorted(expected)}. Every sample from one cohort must "
                f"carry the same keys; an absent modality is zero-filled with present=False, not omitted."
            )
    return first


def collate_subjects(samples: list[SubjectSample]) -> SubjectBatch:
    """Collate samples into a :class:`SubjectBatch`.

    Pass as ``collate_fn`` to a ``DataLoader``.

    Every modality here is fixed-shape: a cohort's ECG recordings are cropped or
    padded to one length before they reach a model, and its images are resized, so
    there is nothing ragged to pad over. A source that cannot promise that must say
    so by raising, which is what the shape check below does -- a recording that
    reaches a batch at the wrong length is a preprocessing bug, and padding it here
    would hide which stage let it through.

    Args:
        samples: One per subject, as :class:`MultimodalDataset` yields them.

    Returns:
        The collated batch.

    Raises:
        ValueError: If ``samples`` is empty, if they disagree about which modalities or
            target keys they carry, or if a modality's shape varies between subjects.
    """
    if not samples:
        raise ValueError("cannot collate an empty list of samples")

    modality_names = _require_same_keys(samples, "modalities")
    target_names = _require_same_keys(samples, "target")

    modalities: dict[str, Tensor] = {}
    for name in modality_names:
        values = [sample.modalities[name] for sample in samples]
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) != 1:
            raise ValueError(
                f"modality {name!r} has samples of differing shapes {sorted(shapes)}. Recordings must be "
                f"cropped or padded to one length, and images resized, before they reach a batch; see "
                f"kalecardiac.prepdata."
            )
        modalities[name] = torch.stack(values)

    return SubjectBatch(
        subject_id=[sample.subject_id for sample in samples],
        modalities=modalities,
        present={name: torch.stack([s.present[name] for s in samples]) for name in modality_names},
        target={key: torch.stack([s.target[key] for s in samples]) for key in target_names},
        metadata={key: [s.metadata[key] for s in samples] for key in samples[0].metadata},
    )


def release_workers(*loaders: DataLoader) -> None:
    """Shut down each loader's worker processes now, rather than at interpreter exit.

    ``persistent_workers=True`` keeps a pool alive for its loader's lifetime, which is
    what makes epoch-heavy training fast. The cost is that the pool outlives whatever
    created it, so a cross-validated run that builds three loaders per fold leaves a
    pool per loader for the garbage collector.

    Collected together at interpreter shutdown, each pool's queue-feeder thread races
    the connection handles it is still writing to; on Windows that surfaces as
    ``ValueError: semaphore or lock released too many times``. Dropping the iterator
    here runs the same finaliser, one pool at a time and while the interpreter is
    still healthy.

    Loaders without worker processes are unaffected, so this is safe to call on any.
    """
    for loader in loaders:
        # The iterator owns the pool; PyTorch exposes no public shutdown, but releasing
        # the only reference to it runs its finaliser deterministically.
        loader._iterator = None
    gc.collect()


# --------------------------------------------------------------------------- #
# Modality sources
# --------------------------------------------------------------------------- #


class ModalitySource(ABC):
    """One modality's data, answered per subject.

    Implement this to add a kind of modality. The dataset asks only two things of a
    source -- what a subject's value is, and what shape to substitute when there is
    not one -- so a source can read a file, index a dictionary, or compute something,
    without the dataset knowing which.
    """

    @abstractmethod
    def get(self, identifier: str, index: int) -> Tensor | None:
        """This subject's value, or ``None`` if the source has nothing for them.

        Args:
            identifier: Which subject.
            index: Position in the dataset, for a source whose reads are stochastic
                and want a reproducible per-sample seed.
        """

    @abstractmethod
    def placeholder(self) -> Tensor:
        """A zero-filled stand-in of the right shape, for a subject with no value.

        Needed so a batch stays rectangular. It is never read as evidence: the dataset
        marks the modality absent in ``present``, and fusion drops it.
        """

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        """Where this subject's value came from, for a batch's ``metadata``.

        Never model input. Empty unless a source has something to trace, which is why
        this is not abstract.
        """
        return {}


class ArraySource(ModalitySource):
    """Precomputed per-subject arrays, held in memory.

    The general case for anything already reduced to one array per subject: an
    encoded clinical row, a frozen embedding, a derived summary. ECG recordings and
    images have their own sources, which add the validation those need;
    this one asserts only that every value has the same shape.

    Args:
        values: One array per subject, keyed by identifier. Subjects absent from the
            mapping are treated as missing. Tensors, NumPy arrays and sequences are
            all accepted and converted to float32.
        shape: Shape of the placeholder. Inferred from the first value when omitted,
            which requires at least one.

    Raises:
        ValueError: If ``shape`` is omitted and ``values`` is empty, leaving no shape
            to place a missing subject against, or if the values disagree in shape.
    """

    def __init__(self, values: Mapping[str, ArrayLike], shape: Sequence[int] | None = None) -> None:
        self.values = {key: torch.as_tensor(np.asarray(value), dtype=torch.float32) for key, value in values.items()}
        shapes = {tuple(value.shape) for value in self.values.values()}
        if len(shapes) > 1:
            raise ValueError(f"every subject's value must have the same shape, got {sorted(shapes)}")
        if shape is None:
            if not shapes:
                raise ValueError("shape must be given when values is empty; there is no shape to infer")
            shape = next(iter(shapes))
        elif shapes and tuple(shape) not in shapes:
            raise ValueError(f"shape {tuple(shape)} does not match the values' shape {next(iter(shapes))}")
        self.shape = tuple(int(size) for size in shape)

    def get(self, identifier: str, index: int) -> Tensor | None:
        return self.values.get(identifier)

    def placeholder(self) -> Tensor:
        return torch.zeros(self.shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.values)} subjects | shape={self.shape})"


class MultimodalDataset(Dataset[SubjectSample]):
    """A cohort of subjects, each assembled from any number of modality sources.

    Yields the :class:`SubjectSample` every trainer here reads, so a one-source cohort
    and a thirteen-source cohort are the same object with different arguments.

    Args:
        identifiers: Subjects in this split, in the order they will be indexed.
        sources: One :class:`ModalitySource` per modality, keyed by the name the
            modality carries into the batch and into the model's encoders.
        target: Supervision, keyed by identifier. ``None`` for an unlabelled cohort,
            which is what representation-learning pretraining uses.
        metadata_from: Which sources contribute provenance. Defaults to every source
            that offers any. Naming a subset avoids re-reading a large recording purely
            to record where it came from.

    Raises:
        ValueError: If ``sources`` is empty, ``identifiers`` repeats a subject, or
            ``metadata_from`` names a source that was not given.
    """

    def __init__(
        self,
        identifiers: Sequence[str],
        sources: Mapping[str, ModalitySource],
        target: Target | None = None,
        metadata_from: Sequence[str] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("a cohort needs at least one modality source")
        identifiers = [str(identifier) for identifier in identifiers]
        duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"identifiers repeat {duplicates[:5]}; a subject appearing twice would be scored twice "
                f"and could land in two splits"
            )
        if metadata_from is not None:
            unknown = set(metadata_from) - set(sources)
            if unknown:
                raise ValueError(f"metadata_from names {sorted(unknown)}, which are not sources: {sorted(sources)}")

        self.identifiers = identifiers
        self.sources = dict(sources)
        self.target = target
        self.metadata_from = tuple(sources if metadata_from is None else metadata_from)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> SubjectSample:
        identifier = self.identifiers[index]

        modalities: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for name, source in self.sources.items():
            value = source.get(identifier, index)
            # A placeholder keeps the batch rectangular; ``present`` is what a model
            # reads, so the zeros are never treated as evidence.
            modalities[name] = value if value is not None else source.placeholder()
            present[name] = torch.tensor(value is not None)

        metadata: dict[str, Any] = {}
        for name in self.metadata_from:
            metadata.update(self.sources[name].provenance(identifier, index))

        return SubjectSample(
            subject_id=identifier,
            modalities=modalities,
            present=present,
            target={} if self.target is None else self.target.for_(identifier),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        modalities = ", ".join(f"{name}={type(source).__name__}" for name, source in self.sources.items())
        return f"MultimodalDataset({len(self)} subjects | {modalities})"


# --------------------------------------------------------------------------- #
# Supervision
# --------------------------------------------------------------------------- #


class ColumnTarget:
    """Supervision read from named columns of a table, keyed by identifier.

    Satisfies the :class:`Target` contract for any endpoint whose values are already
    columns: ``{"label": "has_ph"}`` for a binary one, ``{"value": "mpap_mmhg"}`` for
    a continuous one. Which column means what is the caller's declaration, so the same
    class serves an endpoint this package has never heard of.

    Renaming is explicit because the batch key and the column name are different
    vocabularies: a cohort table calls it ``mpap_mmhg``, a batch target calls it
    ``value``, and writing the map out is what keeps that from being a silent guess.

    Args:
        frame: Table carrying ``id_column`` and every named column.
        columns: Batch target key mapped to the column supplying it.
        id_column: Column holding the identifier.

    Raises:
        KeyError: If a named column is absent, or an identifier is duplicated.
        ValueError: If a named column holds a value that is missing or not numeric --
            an unknown outcome is not a negative one, and coercing it to zero would
            silently invent a label.
    """

    def __init__(self, frame: pd.DataFrame, columns: Mapping[str, str], id_column: str = "subject_id") -> None:
        missing = [column for column in (id_column, *columns.values()) if column not in frame.columns]
        if missing:
            raise KeyError(f"table has no column(s) {missing}; available: {list(frame.columns)}")

        identifiers = frame[id_column].astype(str)
        if identifiers.duplicated().any():
            repeated = sorted(identifiers[identifiers.duplicated()].unique())
            raise KeyError(f"{id_column} must be unique, but {repeated[:5]} repeat")

        self.columns = dict(columns)
        self._values: dict[str, Tensor] = {}
        for key, column in self.columns.items():
            numeric = pd.to_numeric(frame[column], errors="coerce")
            unknown = numeric.isna()
            if bool(unknown.any()):
                examples = list(identifiers[unknown][:5])
                raise ValueError(
                    f"column {column!r} has {int(unknown.sum())} missing or non-numeric value(s), e.g. for "
                    f"{examples}. An unknown outcome is not a negative one: decide explicitly before building "
                    f"the cohort -- drop these subjects, or encode what a blank means in your data dictionary."
                )
            self._values[key] = torch.as_tensor(numeric.to_numpy(dtype=float), dtype=torch.float32)
        self._row_of = {identifier: row for row, identifier in enumerate(identifiers)}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Mapping[str, Any]], id_column: str = "subject_id") -> ColumnTarget:
        """Build a target from ``{batch_key: {identifier: value}}``.

        The door in for a cohort that never had a table: labels read from a directory
        layout, or produced by a synthetic generator.

        Args:
            values: One mapping of identifier to value per batch target key.
            id_column: Name the identifier column is given in the table built here.

        Returns:
            A target over every identifier the first key covers.

        Raises:
            ValueError: If ``values`` is empty, or the keys do not cover the same
                subjects -- a subject with a label but no measured value is a cohort
                definition question, not something to fill in silently.
        """
        if not values:
            raise ValueError("from_mapping needs at least one target key")
        subjects = [list(mapping) for mapping in values.values()]
        if any(set(other) != set(subjects[0]) for other in subjects[1:]):
            raise ValueError("every target key must cover the same subjects")

        frame = pd.DataFrame({id_column: [str(identifier) for identifier in subjects[0]]})
        for key, mapping in values.items():
            frame[key] = [mapping[identifier] for identifier in subjects[0]]
        return cls(frame, columns={key: key for key in values}, id_column=id_column)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns this target reads, so a cohort can check them before binding."""
        return tuple(self.columns.values())

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """This subject's values for :attr:`SubjectSample.target`."""
        row = self._row_of[identifier]
        return {key: values[row] for key, values in self._values.items()}

    def values_for(self, identifiers: Sequence[str]) -> dict[str, Tensor]:
        """The same values for many subjects: :meth:`for_` with a batch dimension."""
        rows = torch.tensor([self._row_of[identifier] for identifier in identifiers], dtype=torch.long)
        return {key: values[rows] for key, values in self._values.items()}

    def stratify_labels(self, identifiers: Sequence[str]) -> np.ndarray:
        """Labels to stratify a split on: the ``label`` key, or the first key given."""
        key = LABEL_KEY if LABEL_KEY in self._values else next(iter(self._values))
        return self.values_for(identifiers)[key].numpy()

    def frame(self, identifiers: Sequence[str] | None = None, id_column: str = "subject_id") -> pd.DataFrame:
        """A table of identifiers and target values, for a splitter to stratify over.

        Args:
            identifiers: Subjects to include, in order. Defaults to every subject.
            id_column: Name the identifier column is given.

        Returns:
            One row per identifier, with a column per batch target key.
        """
        identifiers = list(self._row_of) if identifiers is None else [str(name) for name in identifiers]
        values = self.values_for(identifiers)
        return pd.DataFrame({id_column: identifiers, **{key: value.numpy() for key, value in values.items()}})

    def __repr__(self) -> str:
        return f"ColumnTarget({len(self._row_of)} subjects | keys={sorted(self.columns)})"


class Target(Protocol):
    """What a cohort requires of a supervision target.

    Keyed by identifier rather than row position, so a cohort can be subset and
    recombined without realignment.

    Anything derived from a *training fold* is not a target. A class weight is the
    case to watch: it is a property of one fold's label balance, so it belongs to the
    task that consumes it, not here.
    """

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns this target reads."""
        ...

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """This subject's values for ``SubjectSample.target``.

        Named keys, never a positional pack: a ``tensor([label, value])`` built
        backwards runs perfectly and trains against the wrong endpoint.
        """
        ...

    def values_for(self, identifiers: Sequence[str]) -> dict[str, Tensor]:
        """The same values for many subjects: ``for_`` with a leading batch dimension.

        Every consumer that is not a ``DataLoader`` wants this form -- a whole split's
        labels for a metric, a class balance for a loss weight. Must return the same
        keys and dtypes as :meth:`for_`, and agree with it value for value.
        """
        ...


def check_target(target: object) -> None:
    """Verify an object satisfies :class:`Target`, or raise naming what is missing.

    A ``Protocol`` alone verifies nothing at runtime, so this is the check a cohort
    applies before trusting an object it was handed.

    Raises:
        TypeError: If any part of the contract is absent.
    """
    missing = [name for name in ("required_columns", "for_", "values_for") if not hasattr(target, name)]
    if missing:
        raise TypeError(
            f"{type(target).__name__} is not a valid Target: missing {missing}. A target must declare "
            f"required_columns and implement for_(identifier) and values_for(identifiers). See "
            f"kalecardiac.loaddata.multimodal_access.Target."
        )
