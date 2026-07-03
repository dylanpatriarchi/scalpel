"""Feature discovery: which SAE latents fire on a concept.

Given a concept (e.g. "dog") and a corpus, we want the SAE features most
*specifically* associated with it. The method is contrastive:

1. Split the corpus into concept-positive snippets (they mention the concept)
   and the rest (negatives), by keyword match.
2. Encode every snippet's residual activations to SAE features and reduce each
   snippet to a per-feature activation (max over its tokens).
3. Score each feature by ``mean(activation | positive) - mean(activation | negative)``.
   A high score means the feature fires on the concept and stays quiet elsewhere.
4. Rank, and for each top feature collect its max-activating snippets so a human
   can eyeball what it represents.

The scoring/labeling/example functions are pure tensor ops and are unit-tested
without a model; :func:`discover_features` wires them to a backend + SAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .backends.base import ModelBackend
from .cache import ActivationCache, make_key
from .sae import SAEWrapper


@dataclass
class FeatureHit:
    """One discovered feature and the evidence for it."""

    index: int
    score: float
    pos_mean: float
    neg_mean: float
    label: str | None = None
    examples: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    concept: str
    terms: list[str]
    hits: list[FeatureHit]
    n_pos: int
    n_neg: int
    hook_name: str
    neuronpedia_id: str | None = None


def label_snippets(snippets: list[str], terms: list[str]) -> tuple[list[int], list[int]]:
    """Partition snippet indices into (positive, negative) by keyword match.

    A snippet is positive if it contains any of ``terms`` (case-insensitive,
    whole-substring). Everything else is negative.
    """
    terms_lower = [t.lower() for t in terms if t]
    if not terms_lower:
        raise ValueError("At least one non-empty concept term is required")
    pos: list[int] = []
    neg: list[int] = []
    for i, snippet in enumerate(snippets):
        text = snippet.lower()
        (pos if any(t in text for t in terms_lower) else neg).append(i)
    return pos, neg


def contrastive_scores(
    feature_matrix: torch.Tensor, pos_idx: list[int], neg_idx: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(scores, pos_mean, neg_mean)`` over features.

    ``feature_matrix`` is ``[n_snippets, d_sae]``. With no negatives the
    negative mean is zero (score falls back to the positive mean).
    """
    if not pos_idx:
        raise ValueError("Need at least one positive snippet to score features")
    d_sae = feature_matrix.shape[1]
    pos_mean = feature_matrix[pos_idx].mean(dim=0)
    neg_mean = (
        feature_matrix[neg_idx].mean(dim=0)
        if neg_idx
        else torch.zeros(d_sae, dtype=feature_matrix.dtype)
    )
    return pos_mean - neg_mean, pos_mean, neg_mean


def rank_features(scores: torch.Tensor, top_k: int) -> list[int]:
    """Indices of the ``top_k`` highest-scoring features, descending."""
    k = min(top_k, scores.numel())
    return torch.topk(scores, k=k).indices.tolist()


def max_activating_examples(
    feature_matrix: torch.Tensor, snippets: list[str], feature_index: int, k: int
) -> list[str]:
    """The ``k`` snippets where ``feature_index`` activates most strongly."""
    column = feature_matrix[:, feature_index]
    k = min(k, column.numel())
    order = torch.topk(column, k=k).indices.tolist()
    return [snippets[i] for i in order]


def snippet_feature_matrix(
    backend: ModelBackend,
    sae: SAEWrapper,
    snippets: list[str],
    hook_name: str,
    *,
    model_name: str = "",
    cache: ActivationCache | None = None,
) -> torch.Tensor:
    """Reduce each snippet to a ``[d_sae]`` feature vector (max over tokens).

    Optionally caches per-snippet residual activations keyed by
    ``(model, hook, snippet)`` so repeated discovery runs are cheap.
    """
    rows: list[torch.Tensor] = []
    for snippet in snippets:
        acts: torch.Tensor | None = None
        key = make_key(model_name, hook_name, snippet) if cache is not None else ""
        if cache is not None:
            acts = cache.get(key)
        if acts is None:
            acts = backend.capture_resid(snippet, hook_name)
            if cache is not None:
                cache.put(key, acts)
        features = sae.encode(acts)  # [tokens, d_sae]
        rows.append(features.max(dim=0).values.detach().cpu())
    return torch.stack(rows)  # [n_snippets, d_sae]


def discover_features(
    backend: ModelBackend,
    sae: SAEWrapper,
    snippets: list[str],
    concept: str,
    *,
    terms: list[str] | None = None,
    top_k: int = 10,
    examples_k: int = 5,
    cache: ActivationCache | None = None,
    model_name: str = "",
) -> DiscoveryResult:
    """Run contrastive feature discovery for ``concept`` over ``snippets``."""
    terms = terms or [concept]
    pos_idx, neg_idx = label_snippets(snippets, terms)
    if not pos_idx:
        raise ValueError(
            f"No snippet in the corpus matches concept terms {terms!r}; "
            "pass --corpus or --terms with matching text."
        )

    matrix = snippet_feature_matrix(
        backend, sae, snippets, sae.hook_name, model_name=model_name, cache=cache
    )
    scores, pos_mean, neg_mean = contrastive_scores(matrix, pos_idx, neg_idx)
    ranked = rank_features(scores, top_k)

    hits = [
        FeatureHit(
            index=idx,
            score=float(scores[idx].item()),
            pos_mean=float(pos_mean[idx].item()),
            neg_mean=float(neg_mean[idx].item()),
            examples=max_activating_examples(matrix, snippets, idx, examples_k),
        )
        for idx in ranked
    ]
    return DiscoveryResult(
        concept=concept,
        terms=terms,
        hits=hits,
        n_pos=len(pos_idx),
        n_neg=len(neg_idx),
        hook_name=sae.hook_name,
        neuronpedia_id=sae.neuronpedia_id,
    )
