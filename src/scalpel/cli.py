"""Command-line interface.

Milestone 1 implements ``scalpel smoke`` (load a model + SAE and report
reconstruction quality). ``discover``, ``steer`` and ``eval`` are declared so
the surface is stable but land in later milestones.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .backends import ModelBackend, build_backend
from .cache import ActivationCache
from .config import Backend, ScalpelConfig, load_config, mock_config
from .corpus import load_corpus
from .discovery import discover_features
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
    return _not_yet("steer", 3)


def cmd_eval(args: argparse.Namespace) -> int:
    return _not_yet("eval", 4)


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
    steer.add_argument("--feature", type=int, required=True)
    steer.add_argument("--coef", type=float, required=True)
    steer.add_argument("--prompt", required=True)
    steer.add_argument("--config")
    steer.set_defaults(func=cmd_steer)

    ev = sub.add_parser("eval", help="Run the coefficient sweep and metrics.")
    ev.add_argument("--config")
    ev.set_defaults(func=cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
