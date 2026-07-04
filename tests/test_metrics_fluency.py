"""Fluency metric math: perplexity and KL divergence."""

from __future__ import annotations

import math

import pytest
import torch

from scalpel.metrics.fluency import kl_divergence, perplexity_from_nll, token_perplexity


def test_perplexity_from_nll() -> None:
    assert perplexity_from_nll(0.0) == pytest.approx(1.0)
    assert perplexity_from_nll(math.log(2)) == pytest.approx(2.0)


def test_token_perplexity_uniform_is_vocab_size() -> None:
    # Uniform logits over a vocab of 4 => perplexity 4 regardless of targets.
    logits = torch.zeros(3, 4)
    targets = torch.tensor([0, 1, 3])
    assert token_perplexity(logits, targets) == pytest.approx(4.0)


def test_token_perplexity_confident_correct_is_near_one() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    assert token_perplexity(logits, targets) == pytest.approx(1.0, abs=1e-3)


def test_kl_self_is_zero() -> None:
    logits = torch.randn(5)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_kl_manual() -> None:
    p = torch.log(torch.tensor([0.5, 0.5]))
    q = torch.log(torch.tensor([0.25, 0.75]))
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-6)


def test_kl_nonnegative() -> None:
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        p = torch.randn(8, generator=gen)
        q = torch.randn(8, generator=gen)
        assert kl_divergence(p, q) >= -1e-6


def test_kl_averages_over_leading_dims() -> None:
    p = torch.randn(4, 6)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)
