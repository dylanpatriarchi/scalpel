"""Shared fixtures. Everything here is CPU-only and download-free."""

from __future__ import annotations

import pytest
import torch

from scalpel.sae import SAEWrapper


@pytest.fixture
def identity_sae() -> SAEWrapper:
    """An exact identity SAE: reconstruct(x) == relu(x)."""
    return SAEWrapper.mock(d_model=8, hook_name="blocks.0.hook_resid_post")


@pytest.fixture
def random_sae() -> SAEWrapper:
    """A small overcomplete SAE with fixed, seeded random weights."""
    gen = torch.Generator().manual_seed(0)
    d_model, d_sae = 4, 16
    return SAEWrapper(
        W_enc=torch.randn(d_model, d_sae, generator=gen),
        W_dec=torch.randn(d_sae, d_model, generator=gen),
        b_enc=torch.zeros(d_sae),
        b_dec=torch.zeros(d_model),
        hook_name="blocks.1.hook_resid_post",
        layer=1,
    )


@pytest.fixture
def positive_acts() -> torch.Tensor:
    """Nonnegative activations that an identity SAE reconstructs exactly."""
    gen = torch.Generator().manual_seed(1)
    return torch.rand(32, 8, generator=gen) + 0.1
