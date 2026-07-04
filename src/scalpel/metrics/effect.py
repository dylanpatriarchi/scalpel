"""Effect metric: how strongly a generation expresses the target concept.

Two interchangeable scorers, both returning a value in ``[0, 1]``:

* :class:`KeywordScorer` — deterministic, dependency-free term counting. This is
  the reproducible default and the one CI uses.
* :class:`JudgeScorer` — delegates to an LLM judge (Ollama) for a graded score
  when a local judge model is available.

The spec allows "a lightweight classifier or an LLM-judge"; we provide both and
keep the interface identical so a sweep is agnostic to which one it uses.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


def keyword_effect_score(text: str, terms: list[str], *, saturation: int = 3) -> float:
    """Graded concept-presence score from term occurrences, clamped to ``[0, 1]``.

    Counts (case-insensitive, substring) occurrences of any term and maps them
    through ``min(1, count / saturation)`` so the score is monotone in how much
    the concept appears and saturates at ``saturation`` mentions.
    """
    if not terms:
        raise ValueError("At least one term is required")
    lowered = text.lower()
    count = sum(lowered.count(term.lower()) for term in terms if term)
    return min(1.0, count / saturation)


@runtime_checkable
class EffectScorer(Protocol):
    """Scores concept presence of a single generation in ``[0, 1]``."""

    def score(self, text: str) -> float: ...


class KeywordScorer:
    """Deterministic effect scorer based on concept term frequency."""

    def __init__(self, terms: list[str], *, saturation: int = 3) -> None:
        if not terms:
            raise ValueError("KeywordScorer needs at least one term")
        self.terms = terms
        self.saturation = saturation

    def score(self, text: str) -> float:
        return keyword_effect_score(text, self.terms, saturation=self.saturation)


class JudgeScorer:
    """Effect scorer backed by an LLM judge (e.g. Ollama).

    Falls back to ``0.0`` when the judge is unreachable so a sweep never crashes
    on a transient judge failure; callers that want a hard failure should probe
    the judge first.
    """

    def __init__(self, judge: SupportsJudge, concept: str) -> None:
        self.judge = judge
        self.concept = concept

    def score(self, text: str) -> float:
        result = self.judge.score(text, self.concept)
        return 0.0 if result is None else result


@runtime_checkable
class SupportsJudge(Protocol):
    def score(self, text: str, concept: str) -> float | None: ...
