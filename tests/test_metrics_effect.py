"""Effect metric: keyword scorer and judge-backed scorer."""

from __future__ import annotations

import pytest

from scalpel.metrics.effect import JudgeScorer, KeywordScorer, keyword_effect_score


def test_keyword_counts_occurrences() -> None:
    # "dog" appears in "dog" and "dogs" -> 2 occurrences -> 2/3.
    assert keyword_effect_score("my dog loves other dogs", ["dog"]) == pytest.approx(2 / 3)


def test_keyword_saturates_at_one() -> None:
    text = "dog dog dog dog dog"
    assert keyword_effect_score(text, ["dog"]) == 1.0


def test_keyword_zero_when_absent() -> None:
    assert keyword_effect_score("the sky is blue", ["dog"]) == 0.0


def test_keyword_case_insensitive_and_multiterm() -> None:
    assert keyword_effect_score("A DOG and a PUPPY", ["dog", "puppy"]) == pytest.approx(2 / 3)


def test_keyword_empty_terms_raises() -> None:
    with pytest.raises(ValueError, match="term"):
        keyword_effect_score("x", [])


def test_keyword_scorer_matches_function() -> None:
    scorer = KeywordScorer(["dog"], saturation=2)
    assert scorer.score("dog dog") == 1.0
    assert scorer.score("one dog") == pytest.approx(0.5)


def test_keyword_scorer_requires_terms() -> None:
    with pytest.raises(ValueError, match="term"):
        KeywordScorer([])


class _FakeJudge:
    def __init__(self, value: float | None) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def score(self, text: str, concept: str) -> float | None:
        self.calls.append((text, concept))
        return self.value


def test_judge_scorer_passes_through() -> None:
    judge = _FakeJudge(0.8)
    scorer = JudgeScorer(judge, "dog")
    assert scorer.score("a dog") == 0.8
    assert judge.calls == [("a dog", "dog")]


def test_judge_scorer_none_becomes_zero() -> None:
    scorer = JudgeScorer(_FakeJudge(None), "dog")
    assert scorer.score("whatever") == 0.0
