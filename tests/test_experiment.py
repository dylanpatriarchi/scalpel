"""Coefficient sweep orchestration and results serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scalpel.experiment import (
    SweepResult,
    SweepRow,
    build_meandiff_direction,
    mean_resid,
    run_sweep,
)
from scalpel.metrics.effect import KeywordScorer


class FakeSweepBackend:
    """Effect and divergence both grow with |coef|."""

    d_model = 4

    @property
    def device(self) -> str:
        return "cpu"

    def capture_resid(self, text: str, hook_name: str) -> torch.Tensor:  # pragma: no cover
        return torch.zeros(1, self.d_model)

    def generate(self, prompt: str, *, max_new_tokens: int, hook_name, vector, coef) -> str:
        # More steering -> more "dog" mentions in the completion.
        return prompt + " dog" * int(abs(coef))

    def token_nll(self, text: str) -> float:
        return 1.0

    def next_token_logits(self, prompt: str, *, hook_name=None, vector=None, coef=0.0):
        base = torch.zeros(4)
        if vector is not None and coef != 0.0:
            base = base + coef * torch.tensor([0.0, 1.0, 2.0, 3.0])
        return base


def test_run_sweep_is_monotone_in_coef() -> None:
    backend = FakeSweepBackend()
    result = run_sweep(
        backend,
        vector=torch.ones(4),
        hook_name="blocks.0.hook_resid_post",
        prompts=["a", "b"],
        coefs=[0.0, 1.0, 3.0],
        scorer=KeywordScorer(["dog"]),
    )
    effects = [r.mean_effect for r in result.rows]
    kls = [r.mean_kl for r in result.rows]

    assert effects[0] == 0.0
    assert effects[0] < effects[1] < effects[2]
    assert kls[0] == pytest.approx(0.0, abs=1e-6)
    assert kls[1] < kls[2]
    assert all(r.n_prompts == 2 for r in result.rows)


def test_run_sweep_records_probe_effects() -> None:
    backend = FakeSweepBackend()
    result = run_sweep(
        backend,
        vector=torch.ones(4),
        hook_name="h",
        prompts=["a"],
        coefs=[0.0, 2.0],
        scorer=KeywordScorer(["dog"]),
        probe_scorers={"cat": KeywordScorer(["cat"])},
    )
    # Generations only contain "dog", never "cat" -> probe stays flat at 0.
    assert all("cat" in row.probe_effects for row in result.rows)
    assert all(row.probe_effects["cat"] == 0.0 for row in result.rows)
    dicts = result.to_dicts()
    assert "effect_cat" in dicts[0]


class _ConstResidBackend:
    d_model = 3

    @property
    def device(self) -> str:
        return "cpu"

    def __init__(self, value: float) -> None:
        self.value = value

    def capture_resid(self, text: str, hook_name: str) -> torch.Tensor:
        # Two tokens, all equal to `value`.
        return torch.full((2, self.d_model), self.value)

    def generate(self, *a, **k):  # pragma: no cover
        return ""

    def token_nll(self, text: str) -> float:  # pragma: no cover
        return 1.0

    def next_token_logits(self, *a, **k):  # pragma: no cover
        return torch.zeros(3)


def test_mean_resid() -> None:
    backend = _ConstResidBackend(2.0)
    out = mean_resid(backend, ["x", "y"], "h")
    assert torch.allclose(out, torch.full((3,), 2.0))


def test_mean_resid_empty_raises() -> None:
    with pytest.raises(ValueError, match="text"):
        mean_resid(_ConstResidBackend(1.0), [], "h")


def test_build_meandiff_direction() -> None:
    class _PosNeg:
        d_model = 3

        @property
        def device(self) -> str:
            return "cpu"

        def capture_resid(self, text: str, hook_name: str) -> torch.Tensor:
            value = 5.0 if text.startswith("pos") else 1.0
            return torch.full((2, self.d_model), value)

        def generate(self, *a, **k):  # pragma: no cover
            return ""

        def token_nll(self, text: str) -> float:  # pragma: no cover
            return 1.0

        def next_token_logits(self, *a, **k):  # pragma: no cover
            return torch.zeros(3)

    direction = build_meandiff_direction(_PosNeg(), "h", ["pos1", "pos2"], ["neg1"])
    assert torch.allclose(direction, torch.full((3,), 4.0))  # 5 - 1


def test_run_sweep_no_prompts_raises() -> None:
    with pytest.raises(ValueError, match="prompt"):
        run_sweep(
            FakeSweepBackend(),
            torch.ones(4),
            "h",
            [],
            [1.0],
            KeywordScorer(["dog"]),
        )


def _result() -> SweepResult:
    return SweepResult(
        label="sae",
        hook_name="blocks.7.hook_resid_pre",
        rows=[SweepRow(0.0, 0.0, 10.0, 0.0, 2), SweepRow(2.0, 0.5, 12.0, 0.3, 2)],
    )


def test_write_csv(tmp_path: Path) -> None:
    path = _result().write_csv(tmp_path / "r.csv")
    lines = path.read_text().strip().splitlines()
    assert lines[0].split(",") == [
        "label",
        "coef",
        "mean_effect",
        "mean_perplexity",
        "mean_kl",
        "n_prompts",
    ]
    assert len(lines) == 3


def test_write_json(tmp_path: Path) -> None:
    path = _result().write_json(tmp_path / "r.json")
    data = json.loads(path.read_text())
    assert data["hook_name"] == "blocks.7.hook_resid_pre"
    assert len(data["rows"]) == 2
    assert data["rows"][1]["mean_effect"] == 0.5
