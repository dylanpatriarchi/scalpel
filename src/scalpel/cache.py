"""On-disk cache for captured activations.

Capturing residual-stream activations is the expensive part of a run, so we key
them by a stable hash of ``(model, hook, prompt, ...)`` and reuse them across
invocations. The cache is content-addressed: the same inputs always map to the
same file, and changing any component yields a new key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

DEFAULT_CACHE_DIR = ".scalpel_cache"


def make_key(*parts: object) -> str:
    """Return a stable hex digest for the given parts.

    The parts are joined with a separator that cannot appear in the string
    form, so ``make_key("a", "bc")`` and ``make_key("ab", "c")`` differ.
    """
    joined = "\x1f".join(repr(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class ActivationCache:
    """A tiny content-addressed tensor cache backed by ``torch.save``."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pt"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> torch.Tensor | None:
        """Return the cached tensor for ``key`` or ``None`` if absent."""
        path = self._path(key)
        if not path.exists():
            return None
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"Cached object for {key} is not a tensor")
        return obj

    def put(self, key: str, tensor: torch.Tensor) -> Path:
        """Persist ``tensor`` under ``key`` and return the file path."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        torch.save(tensor.detach().cpu(), path)
        return path
