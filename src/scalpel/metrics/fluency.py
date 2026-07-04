"""Fluency metrics: perplexity and KL divergence.

These prove that steering does not lobotomise the model. Perplexity of a steered
generation *under the unsteered model* rises when the output becomes incoherent;
KL divergence between the steered and unsteered next-token distributions measures
how far the intervention pushes the model off its base behaviour.

All functions are pure tensor/number ops and are unit-tested against hand
computations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def perplexity_from_nll(nll: float) -> float:
    """Perplexity from a mean negative log-likelihood (natural log)."""
    return math.exp(nll)


def token_perplexity(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Perplexity of ``target_ids`` under per-position ``logits``.

    ``logits`` is ``[seq, vocab]`` and ``target_ids`` is ``[seq]``; position ``t``
    of ``logits`` is taken to predict ``target_ids[t]`` (the caller aligns any
    shift). Uniform logits over a vocab of size ``V`` give perplexity ``V``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll = -chosen.mean()
    return math.exp(nll.item())


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """``KL(P || Q)`` in nats, where P, Q are softmax of the given logits.

    Accepts ``[vocab]`` or ``[..., vocab]`` (the divergence is averaged over any
    leading dimensions). ``KL(P || P) == 0`` and the result is always ``>= 0``.
    """
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())
