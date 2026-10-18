"""Seeding: reproducible runs, seeded workers, and reports that leave the training stream alone."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from kalecardiac.utils import preserve_rng_state, seed_worker, set_seed


def draws() -> tuple[float, float, float]:
    """One draw from each generator a training run reads."""
    return random.random(), float(np.random.rand()), float(torch.rand(()))


class TestSetSeed:
    def test_the_same_seed_gives_the_same_draws(self):
        set_seed(7)
        first = draws()
        set_seed(7)
        assert draws() == first

    def test_a_different_seed_gives_different_draws(self):
        set_seed(7)
        first = draws()
        set_seed(8)
        assert draws() != first


class TestSeedWorker:
    def test_seeds_numpy_and_python_from_torch(self):
        torch.manual_seed(11)
        seed_worker(0)
        expected = np.random.RandomState(torch.initial_seed() % 2**32).rand()
        assert float(np.random.rand()) == pytest.approx(expected)

        torch.manual_seed(11)
        seed_worker(0)
        first = random.random()
        torch.manual_seed(11)
        seed_worker(3)
        # The worker index is not part of the seed: torch already varies it per worker.
        assert random.random() == first


class TestPreserveRngState:
    def test_draws_inside_do_not_move_any_generator_outside(self):
        set_seed(5)
        expected = draws()

        set_seed(5)
        with preserve_rng_state():
            draws()
            draws()
        assert draws() == expected

    def test_a_dataloader_pass_is_kept_off_the_stream(self):
        # The reason this exists: iterating a DataLoader takes a seed from torch's
        # generator even with shuffle off and no workers.
        loader = DataLoader(list(range(4)), batch_size=2, shuffle=False, num_workers=0)

        torch.manual_seed(3)
        list(loader)
        moved = float(torch.rand(()))
        torch.manual_seed(3)
        untouched = float(torch.rand(()))
        assert moved != untouched, "a DataLoader pass no longer draws; this guard can be simplified"

        torch.manual_seed(3)
        with preserve_rng_state():
            list(loader)
        assert float(torch.rand(())) == untouched

    def test_the_state_is_restored_when_the_block_raises(self):
        set_seed(5)
        expected = draws()

        set_seed(5)
        with pytest.raises(RuntimeError, match="report failed"), preserve_rng_state():
            draws()
            raise RuntimeError("report failed")
        assert draws() == expected
