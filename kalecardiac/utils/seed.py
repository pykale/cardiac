"""Reproducibility helpers."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from kale.utils.seed import set_seed as _kale_set_seed


def set_seed(seed: int = 2026) -> None:
    """Seed Python, NumPy and PyTorch, and make CUDA kernels deterministic.

    Delegates to :func:`kale.utils.seed.set_seed`. Exact reproducibility still depends
    on matching software and hardware, and a variational model draws noise every
    forward pass, so two runs agree only if they consume the generator in the same
    order.

    Args:
        seed: Value every generator is seeded with.
    """
    _kale_set_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker.

    Pass as ``worker_init_fn`` so that augmentation and any stochastic read are
    reproducible with ``num_workers > 0``, where each worker otherwise inherits an
    unseeded NumPy and Python RNG.

    Args:
        worker_id: Worker index, supplied by the DataLoader and unused; the seed comes
            from ``torch.initial_seed``, which PyTorch already varies per worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Run a block without moving any random number generator that later code reads.

    For work that only *reports* -- an ablation, an attribution, a latent projection.
    Such work draws random numbers it never uses: every DataLoader iteration takes a
    seed from PyTorch's generator, even with ``shuffle=False`` and no workers. Left
    alone, turning a report on or off changes every model trained after it, so two runs
    that differ only in what they write out would not have trained the same models.

    Python's, NumPy's and PyTorch's generators are restored on exit, including CUDA's on
    every visible device, and including when the block raises.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
