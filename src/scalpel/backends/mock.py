"""A synthetic, download-free backend.

The mock backend produces deterministic nonnegative "activations" so the full
smoke path (capture -> SAE reconstruct -> stats) runs offline, in CI, and in
unit tests with zero model downloads. Paired with :meth:`SAEWrapper.mock` the
identity SAE reconstructs these activations exactly.
"""

from __future__ import annotations

import torch

from ..config import ScalpelConfig
from ..seed import torch_generator


class MockBackend:
    """A tiny fake model with a fixed, small residual width."""

    def __init__(self, cfg: ScalpelConfig, d_model: int = 16) -> None:
        self.cfg = cfg
        self._d_model = d_model
        self._device = "cpu"
        self._seed = cfg.seed

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def device(self) -> str:
        return self._device

    def _n_tokens(self, text: str | list[str]) -> int:
        texts = [text] if isinstance(text, str) else text
        return max(8, sum(len(t.split()) for t in texts))

    def capture_resid(self, text: str | list[str], hook_name: str) -> torch.Tensor:
        # Nonnegative so the identity mock SAE reconstructs exactly; seeded so
        # the same text yields the same activations across runs.
        gen = torch_generator(self._seed, device="cpu")
        n = self._n_tokens(text)
        return torch.randn(n, self._d_model, generator=gen).abs()

    VOCAB = 32

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> str:
        tag = "" if coef == 0.0 or vector is None else f" [steered coef={coef:g}]"
        return f"{prompt} ...[mock generation]{tag}"

    def _prompt_seed(self, prompt: str) -> int:
        # Deterministic per-prompt offset independent of PYTHONHASHSEED.
        return self._seed + sum(prompt.encode("utf-8")) % 1000

    def token_nll(self, text: str) -> float:
        # Deterministic, finite pseudo-loss so perplexity is well-defined.
        return 2.0 + (len(text) % 5) * 0.1

    def next_token_logits(
        self,
        prompt: str,
        *,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> torch.Tensor:
        gen = torch_generator(self._prompt_seed(prompt), device="cpu")
        logits = torch.randn(self.VOCAB, generator=gen)
        if vector is not None and coef != 0.0:
            # A coef-dependent shift so KL(steered || base) grows with |coef|.
            logits = logits + coef * 0.1 * torch.arange(self.VOCAB, dtype=torch.float32)
        return logits
