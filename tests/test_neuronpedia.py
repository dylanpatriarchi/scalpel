"""Neuronpedia label fetch (HTTP injected — no network)."""

from __future__ import annotations

from scalpel.neuronpedia import fetch_label

GOOD = '{"explanations": [{"description": "mentions of dogs and pets"}]}'
EMPTY = '{"explanations": []}'


def test_fetch_label_parses_description() -> None:
    def http_get(url: str, timeout: float) -> str:
        assert "gpt2-small/7-res-jb/1234" in url
        return GOOD

    assert (
        fetch_label("gpt2-small/7-res-jb", 1234, http_get=http_get) == "mentions of dogs and pets"
    )


def test_fetch_label_empty_explanations() -> None:
    assert fetch_label("m/s", 1, http_get=lambda url, timeout: EMPTY) is None


def test_fetch_label_missing_key() -> None:
    assert fetch_label("m/s", 1, http_get=lambda url, timeout: "{}") is None


def test_fetch_label_malformed_json_returns_none() -> None:
    assert fetch_label("m/s", 1, http_get=lambda url, timeout: "not json") is None


def test_fetch_label_network_error_returns_none() -> None:
    def boom(url: str, timeout: float) -> str:
        raise OSError("connection refused")

    assert fetch_label("m/s", 1, http_get=boom) is None
