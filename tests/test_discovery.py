"""Feature-discovery math and the end-to-end driver (with a fake backend)."""

from __future__ import annotations

import pytest
import torch

from scalpel.discovery import (
    contrastive_scores,
    discover_features,
    label_snippets,
    max_activating_examples,
    rank_features,
)
from scalpel.sae import SAEWrapper

# -- pure functions ------------------------------------------------------


def test_label_snippets_case_insensitive_substring() -> None:
    snippets = ["A DOG barks", "the sky is blue", "hotdog stand", "a cat sleeps"]
    pos, neg = label_snippets(snippets, ["dog"])
    # "DOG" (case) and "hotdog" (substring) both count as positive.
    assert pos == [0, 2]
    assert neg == [1, 3]


def test_label_snippets_multiple_terms() -> None:
    snippets = ["a dog", "a cat", "a bird"]
    pos, neg = label_snippets(snippets, ["dog", "cat"])
    assert pos == [0, 1]
    assert neg == [2]


def test_label_snippets_empty_terms_raises() -> None:
    with pytest.raises(ValueError, match="term"):
        label_snippets(["x"], [""])


def test_contrastive_scores_manual() -> None:
    matrix = torch.tensor(
        [
            [4.0, 0.0],  # pos
            [2.0, 0.0],  # pos
            [0.0, 3.0],  # neg
            [0.0, 1.0],  # neg
        ]
    )
    scores, pos_mean, neg_mean = contrastive_scores(matrix, [0, 1], [2, 3])
    assert torch.allclose(pos_mean, torch.tensor([3.0, 0.0]))
    assert torch.allclose(neg_mean, torch.tensor([0.0, 2.0]))
    assert torch.allclose(scores, torch.tensor([3.0, -2.0]))


def test_contrastive_scores_no_negatives_uses_pos_mean() -> None:
    matrix = torch.tensor([[4.0, 2.0], [2.0, 0.0]])
    scores, pos_mean, neg_mean = contrastive_scores(matrix, [0, 1], [])
    assert torch.allclose(neg_mean, torch.zeros(2))
    assert torch.allclose(scores, pos_mean)


def test_contrastive_scores_no_positives_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        contrastive_scores(torch.zeros(2, 2), [], [0, 1])


def test_rank_features_descending() -> None:
    scores = torch.tensor([0.1, 5.0, -3.0, 2.0])
    assert rank_features(scores, 2) == [1, 3]


def test_rank_features_clamps_k() -> None:
    scores = torch.tensor([1.0, 2.0])
    assert len(rank_features(scores, 10)) == 2


def test_max_activating_examples_order() -> None:
    matrix = torch.tensor([[0.0], [9.0], [3.0]])
    snippets = ["a", "b", "c"]
    assert max_activating_examples(matrix, snippets, 0, 2) == ["b", "c"]


# -- end-to-end driver ---------------------------------------------------


class FakeConceptBackend:
    """A backend where feature 2 fires strongly iff the snippet says 'dog'."""

    d_model = 4

    @property
    def device(self) -> str:
        return "cpu"

    def capture_resid(self, text: str, hook_name: str) -> torch.Tensor:
        vec = torch.full((1, self.d_model), 0.1)
        if "dog" in text.lower():
            vec[0, 2] = 5.0
        return vec

    def generate(self, prompt: str, **kwargs: object) -> str:  # pragma: no cover
        return prompt


def test_discover_features_selects_the_concept_feature() -> None:
    sae = SAEWrapper.mock(d_model=4, hook_name="blocks.0.hook_resid_post")
    snippets = ["a dog runs", "the sky is blue", "my dog barks", "green grass grows"]

    result = discover_features(
        FakeConceptBackend(), sae, snippets, concept="dog", top_k=4, examples_k=2
    )

    assert result.n_pos == 2
    assert result.n_neg == 2
    # Feature 2 is the one that separates dog-snippets from the rest.
    assert result.hits[0].index == 2
    assert result.hits[0].score == pytest.approx(4.9)  # 5.0 pos_mean - 0.1 neg_mean
    assert "dog" in result.hits[0].examples[0].lower()


def test_discover_features_no_match_raises() -> None:
    sae = SAEWrapper.mock(d_model=4, hook_name="blocks.0.hook_resid_post")
    with pytest.raises(ValueError, match="No snippet"):
        discover_features(FakeConceptBackend(), sae, ["only cats here"], concept="dog")
