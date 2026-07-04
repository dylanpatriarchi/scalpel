"""Coefficient sweep: the dose-response experiment.

For each steering coefficient we generate a completion per prompt and measure:

* **effect**   — concept presence of the generation (effect scorer),
* **perplexity** — of that generation under the *unsteered* model (fluency),
* **kl**       — KL(steered ‖ unsteered) of the next-token distribution (fluency).

Aggregating the mean over a fixed prompt set at each coefficient yields the
dose-response and fluency curves. The steering vector is supplied by the caller
(the SAE decoder column, or a baseline direction in milestone 5), so the exact
same sweep machinery scores the controls.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .backends.base import ModelBackend
from .cache import ActivationCache, capture_cached
from .metrics.effect import EffectScorer
from .metrics.fluency import kl_divergence, perplexity_from_nll
from .seed import set_seed
from .steering import meandiff_vector


@dataclass
class SweepRow:
    coef: float
    mean_effect: float
    mean_perplexity: float
    mean_kl: float
    n_prompts: int
    # Off-target concept effects for the specificity check (name -> mean effect).
    probe_effects: dict[str, float] = field(default_factory=dict)


@dataclass
class SweepResult:
    label: str
    hook_name: str
    rows: list[SweepRow]

    def to_dicts(self) -> list[dict[str, float | str]]:
        out: list[dict[str, float | str]] = []
        for row in self.rows:
            record: dict[str, float | str] = {
                "label": self.label,
                "coef": row.coef,
                "mean_effect": row.mean_effect,
                "mean_perplexity": row.mean_perplexity,
                "mean_kl": row.mean_kl,
                "n_prompts": row.n_prompts,
            }
            for name, value in row.probe_effects.items():
                record[f"effect_{name}"] = value
            out.append(record)
        return out

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_dicts()
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"label": self.label, "hook_name": self.hook_name, "rows": self.to_dicts()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_sweep(
    backend: ModelBackend,
    vector: torch.Tensor,
    hook_name: str,
    prompts: list[str],
    coefs: list[float],
    scorer: EffectScorer,
    *,
    label: str = "sae",
    max_new_tokens: int = 40,
    seed: int = 0,
    probe_scorers: dict[str, EffectScorer] | None = None,
) -> SweepResult:
    """Run the coefficient sweep for one steering ``vector``.

    ``probe_scorers`` (name -> scorer) score off-target concepts on the same
    generations for the specificity check: the target effect should rise while
    the probes stay flat.
    """
    if not prompts:
        raise ValueError("Need at least one prompt for a sweep")
    probe_scorers = probe_scorers or {}

    # Unsteered next-token distributions are the KL reference.
    base_logits = [backend.next_token_logits(prompt, hook_name=hook_name) for prompt in prompts]

    rows: list[SweepRow] = []
    for coef in coefs:
        effects: list[float] = []
        perplexities: list[float] = []
        kls: list[float] = []
        probe_values: dict[str, list[float]] = {name: [] for name in probe_scorers}
        for prompt, base in zip(prompts, base_logits, strict=True):
            set_seed(seed)
            generation = backend.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                hook_name=hook_name,
                vector=vector,
                coef=coef,
            )
            effects.append(scorer.score(generation))
            perplexities.append(perplexity_from_nll(backend.token_nll(generation)))
            steered_logits = backend.next_token_logits(
                prompt, hook_name=hook_name, vector=vector, coef=coef
            )
            kls.append(kl_divergence(steered_logits, base))
            for name, probe in probe_scorers.items():
                probe_values[name].append(probe.score(generation))
        rows.append(
            SweepRow(
                coef=float(coef),
                mean_effect=_mean(effects),
                mean_perplexity=_mean(perplexities),
                mean_kl=_mean(kls),
                n_prompts=len(prompts),
                probe_effects={name: _mean(vals) for name, vals in probe_values.items()},
            )
        )
    return SweepResult(label=label, hook_name=hook_name, rows=rows)


def mean_resid(
    backend: ModelBackend,
    texts: list[str],
    hook_name: str,
    *,
    cache: ActivationCache | None = None,
    model_name: str = "",
) -> torch.Tensor:
    """Mean residual activation ``[d_model]`` over all tokens of ``texts``."""
    if not texts:
        raise ValueError("Need at least one text to average residuals")
    total: torch.Tensor | None = None
    count = 0
    for text in texts:
        acts = capture_cached(backend, text, hook_name, cache=cache, model_name=model_name)
        total = acts.sum(dim=0) if total is None else total + acts.sum(dim=0)
        count += acts.shape[0]
    assert total is not None
    return total / count


def build_meandiff_direction(
    backend: ModelBackend,
    hook_name: str,
    pos_texts: list[str],
    neg_texts: list[str],
    *,
    cache: ActivationCache | None = None,
    model_name: str = "",
) -> torch.Tensor:
    """Mean-difference steering direction from concept-positive/negative texts."""
    pos = mean_resid(backend, pos_texts, hook_name, cache=cache, model_name=model_name)
    neg = mean_resid(backend, neg_texts, hook_name, cache=cache, model_name=model_name)
    return meandiff_vector(pos, neg)
