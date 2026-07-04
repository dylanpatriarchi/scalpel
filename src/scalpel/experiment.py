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
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .backends.base import ModelBackend
from .metrics.effect import EffectScorer
from .metrics.fluency import kl_divergence, perplexity_from_nll
from .seed import set_seed


@dataclass
class SweepRow:
    coef: float
    mean_effect: float
    mean_perplexity: float
    mean_kl: float
    n_prompts: int


@dataclass
class SweepResult:
    label: str
    hook_name: str
    rows: list[SweepRow]

    def to_dicts(self) -> list[dict[str, float | str]]:
        return [{"label": self.label, **asdict(row)} for row in self.rows]

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
) -> SweepResult:
    """Run the coefficient sweep for one steering ``vector``."""
    if not prompts:
        raise ValueError("Need at least one prompt for a sweep")

    # Unsteered next-token distributions are the KL reference.
    base_logits = [backend.next_token_logits(prompt, hook_name=hook_name) for prompt in prompts]

    rows: list[SweepRow] = []
    for coef in coefs:
        effects: list[float] = []
        perplexities: list[float] = []
        kls: list[float] = []
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
        rows.append(
            SweepRow(
                coef=float(coef),
                mean_effect=_mean(effects),
                mean_perplexity=_mean(perplexities),
                mean_kl=_mean(kls),
                n_prompts=len(prompts),
            )
        )
    return SweepResult(label=label, hook_name=hook_name, rows=rows)
