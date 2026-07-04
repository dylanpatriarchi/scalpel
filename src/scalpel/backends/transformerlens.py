"""TransformerLens backend (default).

Runs gpt2-small and gemma-2-2b via ``HookedTransformer`` with clean HookPoints
on the residual stream. Requires the optional ``models`` extra; it is imported
lazily so the package (and the mock path) work without the heavy stack. Not
exercised by the offline unit tests — covered by the gated real-model smoke job.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from ..config import ScalpelConfig, resolve_device

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TransformerLensBackend:
    """Hooked-model backend built on TransformerLens."""

    def __init__(self, cfg: ScalpelConfig) -> None:
        from transformer_lens import HookedTransformer

        self.cfg = cfg
        self._device = resolve_device(cfg.model.device)
        # MPS on some PyTorch builds warns it "may be silently incorrect". The
        # user opts into MPS via the config; acknowledge it so output stays
        # clean, and flag the correctness caveat in the README.
        if self._device == "mps":
            os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        dtype = _DTYPES.get(cfg.model.dtype, torch.float32)
        self.model: Any = HookedTransformer.from_pretrained(
            cfg.model.name, device=self._device, dtype=dtype
        )

    @property
    def d_model(self) -> int:
        return int(self.model.cfg.d_model)

    @property
    def device(self) -> str:
        return self._device

    def capture_resid(self, text: str | list[str], hook_name: str) -> torch.Tensor:
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens, names_filter=hook_name)
        acts = cache[hook_name]
        return acts.reshape(-1, acts.shape[-1]).float().cpu()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> str:
        # Greedy decoding keeps before/after comparisons deterministic so the
        # only variable is the steering vector.
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "verbose": False,
            "do_sample": False,
        }
        if vector is None or coef == 0.0 or hook_name is None:
            return str(self.model.generate(prompt, **kwargs))

        from ..steering import make_steering_hook

        hook = make_steering_hook(vector.to(self._device), coef)
        with self.model.hooks(fwd_hooks=[(hook_name, hook)]):
            return str(self.model.generate(prompt, **kwargs))

    def token_nll(self, text: str) -> float:
        tokens = self.model.to_tokens(text)
        loss = self.model(tokens, return_type="loss")
        return float(loss.item())

    def next_token_logits(
        self,
        prompt: str,
        *,
        hook_name: str | None = None,
        vector: torch.Tensor | None = None,
        coef: float = 0.0,
    ) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt)
        if vector is None or coef == 0.0 or hook_name is None:
            logits = self.model(tokens, return_type="logits")
        else:
            from ..steering import make_steering_hook

            hook = make_steering_hook(vector.to(self._device), coef)
            with self.model.hooks(fwd_hooks=[(hook_name, hook)]):
                logits = self.model(tokens, return_type="logits")
        return logits[0, -1].float().detach().cpu()
