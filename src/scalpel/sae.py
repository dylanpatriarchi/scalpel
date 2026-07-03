"""Sparse-autoencoder wrapper.

The SAE is a linear encoder/decoder pair with a sparsity nonlinearity:

    features    = relu((x - b_dec) @ W_enc + b_enc)      # [..., d_sae]
    x_hat       = features @ W_dec + b_dec               # [..., d_model]

This matches the SAELens convention (a decoder "pre-bias" ``b_dec`` subtracted
before encoding). The columns of the decoder are the *feature directions*: row
``i`` of ``W_dec`` is the unit of residual-stream space that latent ``i`` writes
to, and it is exactly the vector we later add to steer the model.

The math here is deliberately backend-agnostic so it can be unit-tested with a
hand-built tiny SAE and no model download.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ReconStats:
    """Reconstruction quality of an SAE on a batch of activations."""

    mse: float
    variance_explained: float
    mean_l0: float
    n_tokens: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mse": self.mse,
            "variance_explained": self.variance_explained,
            "mean_l0": self.mean_l0,
            "n_tokens": self.n_tokens,
        }


class SAEWrapper:
    """A minimal, typed view over an SAE's weights and core operations."""

    def __init__(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        *,
        hook_name: str,
        layer: int,
        raw: Any = None,
    ) -> None:
        if W_enc.shape[0] != W_dec.shape[1]:
            raise ValueError(
                f"W_enc d_model ({W_enc.shape[0]}) != W_dec d_model ({W_dec.shape[1]})"
            )
        if W_enc.shape[1] != W_dec.shape[0]:
            raise ValueError(
                f"W_enc d_sae ({W_enc.shape[1]}) != W_dec d_sae ({W_dec.shape[0]})"
            )
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.hook_name = hook_name
        self.layer = layer
        self.raw = raw  # underlying sae_lens.SAE, if any

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    def _as_float(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.W_enc.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activations ``[..., d_model]`` to sparse features ``[..., d_sae]``."""
        x = self._as_float(x)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.relu(pre)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Map features ``[..., d_sae]`` back to activations ``[..., d_model]``."""
        return features @ self.W_dec + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode/decode round-trip."""
        return self.decode(self.encode(x))

    def feature_direction(self, index: int) -> torch.Tensor:
        """Return the decoder column (residual-stream direction) for a latent.

        This is the steering vector for feature ``index``.
        """
        if not 0 <= index < self.d_sae:
            raise IndexError(f"feature index {index} out of range [0, {self.d_sae})")
        return self.W_dec[index]

    def reconstruction_error(self, x: torch.Tensor) -> ReconStats:
        """Compute MSE, fraction of variance explained, and mean L0 sparsity.

        Args:
            x: Activations of shape ``[n_tokens, d_model]`` (or broadcastable).
        """
        x = self._as_float(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, self.d_model)

        features = self.encode(x)
        x_hat = self.decode(features)

        resid = x - x_hat
        mse = torch.mean(resid**2)

        total = torch.mean((x - x.mean(dim=0, keepdim=True)) ** 2)
        variance_explained = 1.0 - (mse / (total + 1e-12))

        mean_l0 = torch.mean((features > 0).float().sum(dim=-1))

        return ReconStats(
            mse=float(mse.item()),
            variance_explained=float(variance_explained.item()),
            mean_l0=float(mean_l0.item()),
            n_tokens=int(x.shape[0]),
        )

    # -- constructors ----------------------------------------------------

    @classmethod
    def mock(cls, d_model: int = 16, *, hook_name: str, layer: int = 0) -> SAEWrapper:
        """An identity SAE for the download-free smoke path.

        With identity weights and zero biases, ``reconstruct(x) == relu(x)``,
        so nonnegative activations reconstruct exactly. This lets the smoke
        command exercise the full path and print sensible stats offline.
        """
        eye = torch.eye(d_model, dtype=torch.float32)
        zeros_sae = torch.zeros(d_model, dtype=torch.float32)
        zeros_model = torch.zeros(d_model, dtype=torch.float32)
        return cls(
            W_enc=eye,
            W_dec=eye,
            b_enc=zeros_sae,
            b_dec=zeros_model,
            hook_name=hook_name,
            layer=layer,
        )

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str, device: str = "cpu"
    ) -> SAEWrapper:
        """Load a released SAE through SAELens.

        Requires the optional ``models`` extra. Not exercised by the offline
        unit tests; covered by the gated real-model smoke job.
        """
        from sae_lens import SAE  # type: ignore[import-not-found]

        loaded = SAE.from_pretrained(release, sae_id, device=device)
        # SAELens has returned either an SAE or a (sae, cfg, sparsity) tuple
        # across versions; normalise both.
        sae = loaded[0] if isinstance(loaded, tuple) else loaded

        hook_name = getattr(sae.cfg, "hook_name", f"blocks.{getattr(sae.cfg, 'hook_layer', 0)}")
        layer = int(getattr(sae.cfg, "hook_layer", 0))
        return cls(
            W_enc=sae.W_enc.detach(),
            W_dec=sae.W_dec.detach(),
            b_enc=sae.b_enc.detach(),
            b_dec=sae.b_dec.detach(),
            hook_name=hook_name,
            layer=layer,
            raw=sae,
        )
