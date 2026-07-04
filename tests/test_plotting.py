"""Plot generation (skipped if matplotlib is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scalpel.experiment import SweepResult, SweepRow

pytest.importorskip("matplotlib")

from scalpel.plotting import plot_dose_response, plot_fluency  # noqa: E402


def _result() -> SweepResult:
    return SweepResult(
        label="sae",
        hook_name="blocks.7.hook_resid_pre",
        rows=[
            SweepRow(-4.0, 0.0, 30.0, 0.5, 3),
            SweepRow(0.0, 0.1, 10.0, 0.0, 3),
            SweepRow(4.0, 0.9, 25.0, 0.4, 3),
        ],
    )


def test_plot_dose_response(tmp_path: Path) -> None:
    path = plot_dose_response(_result(), tmp_path / "dr.png", concept="dog")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_fluency(tmp_path: Path) -> None:
    path = plot_fluency(_result(), tmp_path / "fl.png")
    assert path.exists()
    assert path.stat().st_size > 0
