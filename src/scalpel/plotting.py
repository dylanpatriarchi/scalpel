"""Plots for the sweep results (matplotlib, imported lazily).

Kept separate from the numerics so the core package and unit tests do not depend
on matplotlib. Install the ``viz`` extra to use these.
"""

from __future__ import annotations

from pathlib import Path

from .experiment import SweepResult


def _import_pyplot():  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    return plt


def plot_dose_response(result: SweepResult, path: str | Path, *, concept: str = "") -> Path:
    """Effect vs. coefficient — the headline dose-response curve."""
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coefs = [row.coef for row in result.rows]
    effects = [row.mean_effect for row in result.rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coefs, effects, marker="o", color="#2563eb")
    ax.axvline(0.0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("mean concept effect  [0, 1]")
    title = "Dose-response" + (f": {concept}" if concept else "")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fluency(result: SweepResult, path: str | Path) -> Path:
    """Perplexity and KL vs. coefficient on twin axes."""
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coefs = [row.coef for row in result.rows]
    perplexity = [row.mean_perplexity for row in result.rows]
    kl = [row.mean_kl for row in result.rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(coefs, perplexity, marker="o", color="#dc2626", label="perplexity")
    ax1.set_xlabel("steering coefficient")
    ax1.set_ylabel("perplexity (unsteered model)", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")

    ax2 = ax1.twinx()
    ax2.plot(coefs, kl, marker="s", color="#7c3aed", label="KL")
    ax2.set_ylabel("KL(steered ‖ unsteered)  [nats]", color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")

    ax1.set_title("Fluency cost vs. steering")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
