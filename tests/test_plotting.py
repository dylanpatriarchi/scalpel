"""Plot generation (skipped if matplotlib is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scalpel.experiment import SweepResult, SweepRow

pytest.importorskip("matplotlib")

from scalpel.plotting import (  # noqa: E402
    plot_baseline_comparison,
    plot_dose_response,
    plot_fluency,
    plot_specificity,
)


def _result(label: str = "sae", probes: bool = False) -> SweepResult:
    def row(coef: float, eff: float, ppl: float, kl: float) -> SweepRow:
        pe = {"cat": 0.0, "ocean": 0.0} if probes else {}
        return SweepRow(coef, eff, ppl, kl, 3, probe_effects=pe)

    return SweepResult(
        label=label,
        hook_name="blocks.7.hook_resid_pre",
        rows=[row(-4.0, 0.0, 30.0, 0.5), row(0.0, 0.1, 10.0, 0.0), row(4.0, 0.9, 25.0, 0.4)],
    )


def test_plot_dose_response(tmp_path: Path) -> None:
    path = plot_dose_response(_result(), tmp_path / "dr.png", concept="dog")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_fluency(tmp_path: Path) -> None:
    path = plot_fluency(_result(), tmp_path / "fl.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_baseline_comparison(tmp_path: Path) -> None:
    results = {
        "sae": _result("sae"),
        "random": _result("random"),
        "meandiff": _result("meandiff"),
    }
    path = plot_baseline_comparison(results, tmp_path / "b.png", concept="dog")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_specificity(tmp_path: Path) -> None:
    path = plot_specificity(_result(probes=True), tmp_path / "s.png", target="dog")
    assert path.exists()
    assert path.stat().st_size > 0
