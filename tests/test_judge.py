"""Ollama judge client (HTTP injected — no network)."""

from __future__ import annotations

import pytest

from scalpel.judge import OllamaJudge, parse_score


def test_parse_score_plain_integer() -> None:
    assert parse_score("7") == pytest.approx(0.7)


def test_parse_score_extracts_first_number() -> None:
    assert parse_score("Score: 9 out of 10") == pytest.approx(0.9)


def test_parse_score_clamps_above_scale() -> None:
    assert parse_score("12") == 1.0


def test_parse_score_no_number_is_none() -> None:
    assert parse_score("no idea") is None


def test_judge_score_success() -> None:
    def http_post(url: str, body: bytes, timeout: float) -> str:
        assert url.endswith("/api/generate")
        assert b"qwen2.5:7b" in body
        return '{"response": "8"}'

    judge = OllamaJudge(http_post=http_post)
    assert judge.score("a text about dogs", "dog") == pytest.approx(0.8)


def test_judge_score_malformed_json_is_none() -> None:
    judge = OllamaJudge(http_post=lambda url, body, timeout: "not json")
    assert judge.score("x", "dog") is None


def test_judge_score_network_error_is_none() -> None:
    def boom(url: str, body: bytes, timeout: float) -> str:
        raise OSError("refused")

    judge = OllamaJudge(http_post=boom)
    assert judge.score("x", "dog") is None


def test_judge_available() -> None:
    up = OllamaJudge(http_post=lambda url, body, timeout: '{"response": "5"}')
    down = OllamaJudge(http_post=lambda url, body, timeout: "garbage")
    assert up.available() is True
    assert down.available() is False


def test_host_trailing_slash_stripped() -> None:
    judge = OllamaJudge(host="http://localhost:11434/")
    assert judge.host == "http://localhost:11434"
