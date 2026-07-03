"""Determinism of the seeding helpers."""

from __future__ import annotations

import numpy as np
import torch

from scalpel.seed import set_seed, torch_generator


def test_torch_reproducible() -> None:
    set_seed(123)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_numpy_reproducible() -> None:
    set_seed(7)
    a = np.random.rand(10)
    set_seed(7)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = torch.randn(10)
    set_seed(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_generator_is_independent_of_global_rng() -> None:
    # A dedicated generator produces the same tensor regardless of how much the
    # global RNG has been consumed beforehand.
    g1 = torch_generator(42)
    a = torch.randn(5, generator=g1)

    torch.randn(1000)  # perturb global RNG

    g2 = torch_generator(42)
    b = torch.randn(5, generator=g2)
    assert torch.equal(a, b)
