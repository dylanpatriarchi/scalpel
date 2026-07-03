"""Content-addressed activation cache."""

from __future__ import annotations

from pathlib import Path

import torch

from scalpel.cache import ActivationCache, make_key


def test_make_key_deterministic() -> None:
    assert make_key("gpt2", "blocks.7", "hello") == make_key("gpt2", "blocks.7", "hello")


def test_make_key_is_order_sensitive() -> None:
    assert make_key("a", "bc") != make_key("ab", "c")
    assert make_key("x", "y") != make_key("y", "x")


def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = ActivationCache(tmp_path)
    key = make_key("model", "hook", "prompt")
    tensor = torch.randn(4, 3)

    assert cache.get(key) is None
    assert not cache.has(key)

    cache.put(key, tensor)

    assert cache.has(key)
    loaded = cache.get(key)
    assert loaded is not None
    assert torch.equal(loaded, tensor)


def test_get_missing_returns_none(tmp_path: Path) -> None:
    cache = ActivationCache(tmp_path)
    assert cache.get(make_key("nope")) is None


def test_put_moves_to_cpu(tmp_path: Path) -> None:
    cache = ActivationCache(tmp_path)
    key = make_key("k")
    cache.put(key, torch.randn(2, 2))
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.device.type == "cpu"
