"""Reading and writing experiment artefacts.

Results go out in machine-readable formats, and a pretrained model comes back in
through :func:`load_checkpoint`, which is the one place that knows what a checkpoint
file may be wrapped in.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

#: Prefix ``torch.compile`` adds to every parameter name of a compiled module. A
#: checkpoint saved from a compiled model carries it and will not load into an
#: uncompiled one, which is a trap worth handling once rather than at each call site.
_COMPILE_PREFIX = "_orig_mod."


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value: Any) -> Any:
    """Convert a NumPy scalar or array to its Python equivalent."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON.

    NumPy scalars and arrays are converted, because metrics reach this function from
    NumPy and scikit-learn far more often than from plain Python.

    Args:
        path: File to write.
        payload: Anything JSON-serialisable once NumPy values are converted.

    Returns:
        The path written.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_plain)
        handle.write("\n")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """Write ``rows`` as CSV.

    Without ``fieldnames`` the header is the union of the keys present, in the order
    first seen, and a row missing one of them is written blank. Passing ``fieldnames``
    is a declaration that every row matches it, so an unexpected key is an error.

    Args:
        path: File to write.
        rows: One mapping per row.
        fieldnames: Header, when the caller wants it fixed.

    Returns:
        The path written.
    """
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", encoding="utf-8", newline="") as handle:
        header = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def strip_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a state dict with wrapper prefixes removed.

    Handles the two wrappers a research checkpoint is most often saved through:
    ``torch.compile``, which prefixes every key with ``_orig_mod.``, and a Lightning
    module holding the network as ``self.model``.

    Args:
        state: Parameter tensors keyed by name, as saved.

    Returns:
        The same tensors, keyed by the names the bare module uses.
    """
    stripped = {key.replace(_COMPILE_PREFIX, ""): value for key, value in state.items()}
    if stripped and all(key.startswith("model.") for key in stripped):
        stripped = {key[len("model.") :]: value for key, value in stripped.items()}
    return stripped


def load_checkpoint(path: str | Path, model: nn.Module, strict: bool = True) -> tuple[list[str], list[str]]:
    """Load saved weights into ``model``, unwrapping how they were saved.

    A checkpoint may be a bare state dict or a Lightning checkpoint with the weights
    under ``"state_dict"``, and its keys may carry a ``torch.compile`` or Lightning
    prefix. Resolving that here is what lets an example load a pretrained cardiac
    model in one line instead of rewriting the key-mangling each time.

    ``weights_only=True``: a checkpoint is data, and unpickling arbitrary objects out
    of one is how a downloaded file becomes code execution.

    Args:
        path: Checkpoint file.
        model: Module to load into, modified in place.
        strict: Require the keys to match exactly. Pass ``False`` when loading a
            pretrained encoder bank into a model that has since gained a head, which
            is the ordinary fine-tuning case.

    Returns:
        ``(missing, unexpected)`` key names, as ``load_state_dict`` reports them.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If ``strict`` and the keys do not match.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    result = model.load_state_dict(strip_state_dict(state), strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
