"""Steering: add a feature direction to the residual stream during generation.

The steering vector for feature ``i`` is the SAE decoder column ``W_dec[i]`` — a
direction in residual-stream space. During the forward pass we add
``coef * vector`` to the residual at the SAE's hook point. Sweeping ``coef``
(including negative values, which *suppress* the concept) is what produces the
dose-response curve in later milestones.

The functions here are pure and backend-agnostic so they can be unit-tested
without a model; backends call :func:`make_steering_hook` to build the hook they
register.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .sae import SAEWrapper
from .seed import torch_generator

# A TransformerLens-style forward hook: (activation, hook) -> activation.
SteeringHook = Callable[..., torch.Tensor]


def build_sae_vector(
    sae: SAEWrapper, feature_index: int, *, normalize: bool = False
) -> torch.Tensor:
    """Return the steering vector for a feature (its decoder column).

    With ``normalize=True`` the vector is scaled to unit L2 norm, so ``coef``
    becomes the absolute magnitude added; by default the raw decoder direction
    is returned (its norm carries the SAE's own scale).
    """
    vector = sae.feature_direction(feature_index)
    if normalize:
        norm = torch.linalg.vector_norm(vector)
        if norm > 0:
            vector = vector / norm
    return vector


def match_norm(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale ``vector`` to have the same L2 norm as ``reference``.

    Used to put every candidate steering direction on the *same* scale, so a
    shared coefficient sweep is an apples-to-apples comparison and the only
    difference between SAE / random / mean-difference is the *direction*.
    """
    v_norm = torch.linalg.vector_norm(vector)
    if v_norm == 0:
        return vector
    return vector / v_norm * torch.linalg.vector_norm(reference)


def build_random_vector(reference: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """A seeded Gaussian direction, norm-matched to ``reference``.

    This is the **required** control: if a random direction of equal norm steers
    the concept just as well per unit of fluency cost, the SAE feature is not
    special. Determinism comes from an explicit generator.
    """
    generator = torch_generator(seed)
    vector = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return match_norm(vector, reference)


def meandiff_vector(pos_mean: torch.Tensor, neg_mean: torch.Tensor) -> torch.Tensor:
    """Mean-difference direction ``mean(pos) - mean(neg)`` in residual space."""
    return pos_mean - neg_mean


def apply_steering(resid: torch.Tensor, vector: torch.Tensor, coef: float) -> torch.Tensor:
    """Return ``resid + coef * vector`` with device/dtype aligned to ``resid``.

    ``vector`` is ``[d_model]`` and broadcasts over the ``[batch, seq, d_model]``
    (or ``[seq, d_model]``) residual.
    """
    aligned = vector.to(device=resid.device, dtype=resid.dtype)
    return resid + coef * aligned


def make_steering_hook(vector: torch.Tensor, coef: float) -> SteeringHook:
    """Build a forward hook that adds ``coef * vector`` to the residual."""

    def hook(resid: torch.Tensor, hook: object = None) -> torch.Tensor:
        return apply_steering(resid, vector, coef)

    return hook
