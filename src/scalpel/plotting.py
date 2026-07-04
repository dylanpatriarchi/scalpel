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


def plot_baseline_comparison(
    results: dict[str, SweepResult], path: str | Path, *, concept: str = ""
) -> Path:
    """Effect vs. KL for each steering direction — the credibility control.

    A more *targeted* direction sits up and to the left: more concept effect for
    the same divergence from the base model. The SAE feature should dominate the
    random and mean-difference baselines.
    """
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "sae": ("#2563eb", "o", "SAE feature"),
        "random": ("#9ca3af", "x", "random (norm-matched)"),
        "meandiff": ("#f59e0b", "s", "mean-difference"),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    # Scatter, not a line: points come from different coefficients, so connecting
    # them in KL order would fabricate a misleading trajectory.
    for label, result in results.items():
        color, marker, name = styles.get(label, ("#111827", ".", label))
        kls = [row.mean_kl for row in result.rows]
        effects = [row.mean_effect for row in result.rows]
        ax.scatter(kls, effects, marker=marker, color=color, label=name, s=55, alpha=0.85)
    ax.set_xlabel("KL(steered ‖ unsteered)  [nats]  — fluency cost")
    ax.set_ylabel("mean concept effect  [0, 1]")
    ax.set_title("Targeted-ness per unit fluency cost" + (f": {concept}" if concept else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_specificity(result: SweepResult, path: str | Path, *, target: str = "target") -> Path:
    """Target concept vs. off-target probes across the coefficient sweep."""
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coefs = [row.coef for row in result.rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        coefs,
        [row.mean_effect for row in result.rows],
        marker="o",
        linewidth=2.2,
        color="#2563eb",
        label=f"{target} (target)",
    )
    probe_names = list(result.rows[0].probe_effects) if result.rows else []
    for name in probe_names:
        ax.plot(
            coefs,
            [row.probe_effects.get(name, 0.0) for row in result.rows],
            marker=".",
            linewidth=1.0,
            alpha=0.8,
            label=name,
        )
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("mean concept effect  [0, 1]")
    ax.set_title("Specificity: target rises, off-targets stay flat")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
