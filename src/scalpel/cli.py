"""Command-line interface.

Milestone 1 implements ``scalpel smoke`` (load a model + SAE and report
reconstruction quality). ``discover``, ``steer`` and ``eval`` are declared so
the surface is stable but land in later milestones.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .backends import ModelBackend, build_backend
from .cache import ActivationCache
from .config import Backend, ScalpelConfig, load_config, mock_config
from .corpus import load_corpus
from .discovery import discover_features
from .experiment import SweepResult, run_sweep
from .metrics.effect import EffectScorer, JudgeScorer, KeywordScorer
from .neuronpedia import fetch_label
from .sae import SAEWrapper
from .seed import set_seed

SAMPLE_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "Interpretability lets us read the mind of a neural network.",
    "Paris is the capital of France, and it sits on the Seine.",
]


def _load_cfg(config: str | None, backend: str | None, seed: int | None) -> ScalpelConfig:
    cfg = load_config(config) if config else mock_config()
    if backend is not None:
        cfg = cfg.model_copy(
            update={"model": cfg.model.model_copy(update={"backend": Backend(backend)})}
        )
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    return cfg


def build_sae(cfg: ScalpelConfig, backend: ModelBackend) -> SAEWrapper:
    """Build the SAE for a run: an identity mock SAE, or a real released one."""
    if cfg.model.backend == Backend.mock:
        return SAEWrapper.mock(
            d_model=backend.d_model,
            hook_name=cfg.sae.resolved_hook_name,
            layer=cfg.sae.layer,
        )
    return SAEWrapper.from_pretrained(cfg.sae.release, cfg.sae.sae_id, device=backend.device)


def cmd_smoke(args: argparse.Namespace) -> int:
    """Load model + SAE and report reconstruction error on sample text."""
    cfg = _load_cfg(args.config, args.backend, args.seed)
    set_seed(cfg.seed)

    backend = build_backend(cfg)
    sae = build_sae(cfg, backend)
    hook_name = sae.hook_name if cfg.model.backend != Backend.mock else cfg.sae.resolved_hook_name

    text = args.text if args.text else SAMPLE_TEXT
    acts = backend.capture_resid(text, hook_name)
    stats = sae.reconstruction_error(acts)

    print("scalpel smoke")
    print(f"  backend           : {cfg.model.backend.value}")
    print(f"  model             : {cfg.model.name}")
    print(f"  device            : {backend.device}")
    print(f"  hook              : {hook_name}")
    print(f"  d_model / d_sae   : {sae.d_model} / {sae.d_sae}")
    print(f"  tokens            : {stats.n_tokens}")
    print(f"  reconstruction MSE: {stats.mse:.6f}")
    print(f"  variance explained: {stats.variance_explained:.4f}")
    print(f"  mean L0 (sparsity): {stats.mean_l0:.2f}")
    return 0


def _not_yet(name: str, milestone: int) -> int:
    print(f"`scalpel {name}` is implemented in milestone {milestone}.", file=sys.stderr)
    return 2


def cmd_discover(args: argparse.Namespace) -> int:
    """Find the SAE features most associated with a concept."""
    cfg = _load_cfg(args.config, args.backend, args.seed)
    set_seed(cfg.seed)

    backend = build_backend(cfg)
    sae = build_sae(cfg, backend)
    snippets = load_corpus(args.corpus)
    terms = args.terms if args.terms else [args.concept]
    cache = ActivationCache(args.cache_dir) if args.cache_dir else None

    try:
        result = discover_features(
            backend,
            sae,
            snippets,
            args.concept,
            terms=terms,
            top_k=args.top_k,
            examples_k=args.examples,
            cache=cache,
            model_name=cfg.model.name,
        )
    except ValueError as exc:
        print(f"discover: {exc}", file=sys.stderr)
        return 2

    if args.labels and result.neuronpedia_id:
        for hit in result.hits:
            hit.label = fetch_label(result.neuronpedia_id, hit.index)

    print(f"scalpel discover  concept={result.concept!r}  terms={result.terms}")
    print(f"  hook              : {result.hook_name}")
    print(f"  neuronpedia       : {result.neuronpedia_id or '(none)'}")
    print(f"  positive/negative : {result.n_pos}/{result.n_neg} snippets")
    print(f"  top {len(result.hits)} features (contrastive score = pos_mean - neg_mean):")
    for rank, hit in enumerate(result.hits, start=1):
        label = f"  — {hit.label}" if hit.label else ""
        print(
            f"    #{rank:<2} feature {hit.index:<6} "
            f"score={hit.score:+.3f}  pos={hit.pos_mean:.3f} neg={hit.neg_mean:.3f}{label}"
        )
        if hit.examples:
            print(f'         e.g. "{hit.examples[0]}"')
    return 0


def cmd_steer(args: argparse.Namespace) -> int:
    """Generate with and without a feature-steering vector (qualitative before/after)."""
    import torch

    cfg = _load_cfg(args.config, args.backend, args.seed)
    set_seed(cfg.seed)

    backend = build_backend(cfg)
    sae = build_sae(cfg, backend)
    try:
        vector = sae.feature_direction(args.feature)
    except IndexError as exc:
        print(f"steer: {exc}", file=sys.stderr)
        return 2
    hook_name = sae.hook_name
    norm = float(torch.linalg.vector_norm(vector).item())

    set_seed(cfg.seed)
    base = backend.generate(args.prompt, max_new_tokens=args.max_new_tokens, hook_name=hook_name)
    set_seed(cfg.seed)
    steered = backend.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        hook_name=hook_name,
        vector=vector,
        coef=args.coef,
    )

    print(f"scalpel steer  feature={args.feature}  coef={args.coef:g}")
    print(f"  hook              : {hook_name}")
    print(f"  |direction| / step: {norm:.3f} / {abs(args.coef) * norm:.3f}")
    print("\n--- unsteered ---")
    print(base)
    print("\n--- steered ---")
    print(steered)
    return 0


def _build_scorer(args: argparse.Namespace, cfg: ScalpelConfig, terms: list[str]) -> EffectScorer:
    """Pick the effect scorer: Ollama judge if requested and reachable, else keyword."""
    if args.judge:
        from .judge import OllamaJudge

        judge = OllamaJudge(cfg.eval.judge_model, cfg.eval.judge_host)
        if judge.available():
            print(f"  scorer            : Ollama judge ({cfg.eval.judge_model})")
            return JudgeScorer(judge, args.concept)
        print(
            f"  scorer            : judge {cfg.eval.judge_model} unreachable, "
            "falling back to keyword",
            file=sys.stderr,
        )
    print(f"  scorer            : keyword {terms}")
    return KeywordScorer(terms)


def _print_table(result: SweepResult) -> None:
    print(f"  {'coef':>8} {'effect':>8} {'perplexity':>12} {'KL':>8}")
    for row in result.rows:
        print(
            f"  {row.coef:>8.2f} {row.mean_effect:>8.3f} "
            f"{row.mean_perplexity:>12.3f} {row.mean_kl:>8.4f}"
        )


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the coefficient sweep and produce the dose-response + fluency results."""
    cfg = _load_cfg(args.config, args.backend, args.seed)
    set_seed(cfg.seed)

    backend = build_backend(cfg)
    sae = build_sae(cfg, backend)
    try:
        vector = sae.feature_direction(args.feature)
    except IndexError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2

    terms = args.terms if args.terms else [args.concept]
    coefs = args.coefs if args.coefs else cfg.steer.coefs
    prompts = args.prompts if args.prompts else cfg.eval.prompts

    print(f"scalpel eval  feature={args.feature}  concept={args.concept!r}")
    print(f"  hook              : {sae.hook_name}")
    print(f"  coefs             : {coefs}")
    print(f"  prompts           : {len(prompts)}")
    scorer = _build_scorer(args, cfg, terms)

    result = run_sweep(
        backend,
        vector,
        sae.hook_name,
        prompts,
        coefs,
        scorer,
        label="sae",
        max_new_tokens=args.max_new_tokens,
        seed=cfg.seed,
    )
    _print_table(result)

    out = Path(args.out)
    csv_path = result.write_csv(out / "sweep_sae.csv")
    json_path = result.write_json(out / "sweep_sae.json")
    print(f"  wrote             : {csv_path}  {json_path}")

    if not args.no_plots:
        try:
            from .plotting import plot_dose_response, plot_fluency

            dr = plot_dose_response(result, out / "dose_response.png", concept=args.concept)
            fl = plot_fluency(result, out / "fluency.png")
            print(f"  plots             : {dr}  {fl}")
        except ImportError:
            print("  plots             : skipped (install the 'viz' extra)", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scalpel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Load a model + SAE and check reconstruction.")
    smoke.add_argument("--config", help="Path to a YAML config (defaults to the mock backend).")
    smoke.add_argument(
        "--backend",
        choices=[b.value for b in Backend],
        help="Override the config backend (e.g. 'mock' for a download-free run).",
    )
    smoke.add_argument("--seed", type=int, help="Override the config seed.")
    smoke.add_argument("--text", nargs="+", help="Sample text to reconstruct.")
    smoke.set_defaults(func=cmd_smoke)

    discover = sub.add_parser("discover", help="Find SAE features for a concept.")
    discover.add_argument("--concept", required=True, help="Concept to search for, e.g. 'dog'.")
    discover.add_argument(
        "--terms",
        nargs="+",
        help="Keyword(s) marking a snippet as concept-positive (default: the concept).",
    )
    discover.add_argument("--config", help="YAML config (defaults to the mock backend).")
    discover.add_argument(
        "--backend", choices=[b.value for b in Backend], help="Override the config backend."
    )
    discover.add_argument("--seed", type=int, help="Override the config seed.")
    discover.add_argument("--corpus", help="Path to a corpus file (defaults to the bundled one).")
    discover.add_argument("--top-k", type=int, default=10, help="How many features to report.")
    discover.add_argument(
        "--examples", type=int, default=5, help="Max-activating examples per feature."
    )
    discover.add_argument(
        "--labels", action="store_true", help="Fetch Neuronpedia labels (network)."
    )
    discover.add_argument("--cache-dir", help="Cache captured activations under this directory.")
    discover.set_defaults(func=cmd_discover)

    steer = sub.add_parser("steer", help="Steer generation with a feature direction.")
    steer.add_argument("--feature", type=int, required=True, help="SAE feature index to steer on.")
    steer.add_argument(
        "--coef", type=float, required=True, help="Steering coefficient (may be <0)."
    )
    steer.add_argument("--prompt", required=True, help="Prompt to complete.")
    steer.add_argument("--config", help="YAML config (defaults to the mock backend).")
    steer.add_argument(
        "--backend", choices=[b.value for b in Backend], help="Override the config backend."
    )
    steer.add_argument("--seed", type=int, help="Override the config seed.")
    steer.add_argument("--max-new-tokens", type=int, default=40, help="Tokens to generate.")
    steer.set_defaults(func=cmd_steer)

    ev = sub.add_parser("eval", help="Run the coefficient sweep and metrics.")
    ev.add_argument("--feature", type=int, required=True, help="SAE feature index to sweep.")
    ev.add_argument("--concept", required=True, help="Concept the effect metric scores for.")
    ev.add_argument("--terms", nargs="+", help="Keyword terms for the scorer (default: concept).")
    ev.add_argument(
        "--coefs",
        nargs="+",
        type=float,
        help="Coefficients to sweep (default: config steer.coefs).",
    )
    ev.add_argument(
        "--prompts", nargs="+", help="Prompts to evaluate (default: config eval.prompts)."
    )
    ev.add_argument("--config", help="YAML config (defaults to the mock backend).")
    ev.add_argument(
        "--backend", choices=[b.value for b in Backend], help="Override the config backend."
    )
    ev.add_argument("--seed", type=int, help="Override the config seed.")
    ev.add_argument("--max-new-tokens", type=int, default=40, help="Tokens to generate per prompt.")
    ev.add_argument("--judge", action="store_true", help="Score with the Ollama LLM-judge.")
    ev.add_argument("--out", default="outputs", help="Directory for results and plots.")
    ev.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    ev.set_defaults(func=cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
