"""Backend factory."""

from __future__ import annotations

from ..config import Backend, ScalpelConfig
from .base import ModelBackend
from .mock import MockBackend

__all__ = ["ModelBackend", "MockBackend", "build_backend"]


def build_backend(cfg: ScalpelConfig) -> ModelBackend:
    """Construct the backend named in ``cfg.model.backend``.

    Heavy backends are imported lazily so the mock path never pulls in the
    optional model stack.
    """
    backend = cfg.model.backend
    if backend == Backend.mock:
        return MockBackend(cfg)
    if backend == Backend.transformerlens:
        from .transformerlens import TransformerLensBackend

        return TransformerLensBackend(cfg)
    if backend == Backend.nnsight:
        raise NotImplementedError("The nnsight backend is wired up in a later milestone")
    raise ValueError(f"Unknown backend: {backend}")
