"""Steering-vector construction and the injection hook."""

from __future__ import annotations

import pytest
import torch

from scalpel.sae import SAEWrapper
from scalpel.steering import apply_steering, build_sae_vector, make_steering_hook


def test_build_sae_vector_is_decoder_column(random_sae: SAEWrapper) -> None:
    for i in (0, 3, 15):
        assert torch.equal(build_sae_vector(random_sae, i), random_sae.W_dec[i])


def test_build_sae_vector_normalized_is_unit_norm(random_sae: SAEWrapper) -> None:
    v = build_sae_vector(random_sae, 4, normalize=True)
    assert torch.linalg.vector_norm(v).item() == pytest.approx(1.0, abs=1e-6)
    # Same direction as the raw column.
    raw = random_sae.W_dec[4]
    cos = torch.dot(v, raw) / (torch.linalg.vector_norm(v) * torch.linalg.vector_norm(raw))
    assert cos.item() == pytest.approx(1.0, abs=1e-6)


def test_apply_steering_adds_scaled_vector() -> None:
    resid = torch.zeros(2, 3, 4)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_steering(resid, vector, 2.0)
    expected = torch.zeros(2, 3, 4) + 2.0 * vector
    assert torch.equal(out, expected)


def test_apply_steering_coef_zero_is_identity() -> None:
    resid = torch.randn(5, 4)
    out = apply_steering(resid, torch.randn(4), 0.0)
    assert torch.equal(out, resid)


def test_apply_steering_negative_coef_suppresses() -> None:
    resid = torch.zeros(1, 4)
    vector = torch.ones(4)
    out = apply_steering(resid, vector, -3.0)
    assert torch.equal(out, torch.full((1, 4), -3.0))


def test_apply_steering_matches_dtype() -> None:
    resid = torch.zeros(2, 4, dtype=torch.float16)
    out = apply_steering(resid, torch.ones(4, dtype=torch.float32), 1.0)
    assert out.dtype == torch.float16


def test_make_steering_hook_applies() -> None:
    vector = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hook = make_steering_hook(vector, 5.0)
    resid = torch.zeros(1, 2, 4)
    out = hook(resid, hook="ignored")
    assert torch.equal(out, resid + 5.0 * vector)


def test_steering_hook_signature_accepts_transformerlens_call() -> None:
    # TransformerLens calls hook(activation, hook=hook_point).
    hook = make_steering_hook(torch.ones(4), 1.0)
    resid = torch.zeros(3, 4)
    out = hook(resid, hook=object())
    assert out.shape == resid.shape
