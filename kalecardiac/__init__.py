"""KaleCardiac: cardiovascular machine learning for the PyKale ecosystem.

Organised as a verb-oriented pipeline (load -> prep -> model -> evaluate -> interpret),
with ``auto`` for high-level construction. Representation learning is not a stage of
its own: a multimodal VAE is a model in ``model.embed``, the objective that trains it
is in ``model.predict``, and pretraining and fine-tuning are two tasks handed to the
same trainer in ``pipeline``.
"""

from importlib import import_module
from pathlib import Path

__version__ = Path(__file__).with_name("_version.txt").read_text(encoding="utf-8").strip()

_SUBMODULES = frozenset(
    {
        "auto",
        "loaddata",
        "prepdata",
        "model",
        "pipeline",
        "evaluate",
        "interpret",
        "utils",
    }
)

__all__ = sorted(_SUBMODULES | {"__version__"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
