"""The model-backend abstraction.

A backend is the thin seam between Scalpel and whatever library actually runs
the transformer with hooks. Milestone 1 only needs ``capture_resid`` (to read
the residual stream for a reconstruction check) plus a couple of properties;
generation and steering hooks arrive in later milestones and are declared here
so implementations share one contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelBackend(Protocol):
    """Everything Scalpel needs from a hooked model."""

    @property
    def d_model(self) -> int:
        """Residual-stream width."""
        ...

    @property
    def device(self) -> str:
        """Concrete device the model runs on."""
        ...

    def capture_resid(self, text: str | list[str], hook_name: str) -> torch.Tensor:
        """Return residual activations at ``hook_name``.

        The result is flattened to ``[n_tokens, d_model]`` on CPU so downstream
        SAE math is device-agnostic.
        """
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> str:
        """Generate text, optionally adding ``coef * vector`` at ``hook_name``."""
        ...

    def token_nll(self, text: str) -> float:
        """Mean negative log-likelihood of ``text`` under the unsteered model.

        Perplexity is ``exp`` of this; used to score the fluency of a steered
        generation against the base model.
        """
        ...

    def next_token_logits(
        self,
        prompt: str,
        *,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> torch.Tensor:
        """Last-position next-token logits ``[vocab]``, optionally steered.

        Comparing steered vs. unsteered logits over a fixed prompt set gives the
        KL-divergence fluency signal.
        """
        ...
