# Scalpel

**Causally steer an LLM's behavior by intervening on a single Sparse Autoencoder
(SAE) feature — and prove it with numbers, not screenshots.**

Scalpel finds an interpretable feature inside an open language model using a
released SAE suite, then adds that feature's decoder direction to the residual
stream during generation. The point of the project is not that steering *looks*
like it works — it is the **measured, controlled evidence** that a single feature
direction causally controls a specific, human-interpretable behavior:

- a **dose-response curve** (effect vs. steering coefficient, including negative
  coefficients for suppression),
- a **fluency check** (perplexity / KL divergence vs. the unsteered model, to
  prove we are not lobotomizing it),
- a **specificity check** (the target concept rises while unrelated behaviors stay
  flat), and
- **baseline controls** — the SAE feature is compared against a random direction
  of equal norm and a mean-difference steering vector. *Without the
  random-direction control the result is not credible, so it is never skipped.*

> Status: **Milestone 1 complete** — scaffold, config, SAE loading, and an
> offline reconstruction sanity check. Later milestones add feature discovery,
> the steering hook, the metrics, the baseline controls, and the demo notebook.
> See the [roadmap](#roadmap).

---

## Why these choices

| Decision | Choice | Rationale |
| --- | --- | --- |
| Core language | **Python** | The SAE ecosystem (SAELens, TransformerLens, nnsight, the released weights) is Python-only. |
| Hooking library | **TransformerLens** (`HookedSAETransformer`) | Clean residual-stream HookPoints; supports gpt2-small and gemma-2-2b; runs on CPU/MPS/CUDA. |
| SAE loading | **SAELens** | Loads Gemma Scope and GPT-2 SAEs; decoder columns are the feature directions. |
| Default showcase | **Gemma Scope · `gemma-2-2b`** | Smoothest integration (Neuronpedia feature labels, JumpReLU 16k SAEs); runs on one consumer GPU or Apple MPS. |
| CPU / CI path | **gpt2-small + `gpt2-small-res-jb`** | Runs anywhere with no GPU, so reviewers and CI can run something end to end. |
| Optional heavy backend | **Qwen-Scope · `Qwen3-8B`** via nnsight | Newer, larger; configurable for users with GPU headroom. |
| LLM-judge | **Ollama** (`qwen2.5:7b`) | Scoring concept presence is text-in/text-out — a perfect local, free judge. A deterministic keyword scorer is the CI fallback. |

Nothing about the model, SAE, layer, or feature is hardcoded — everything is
driven by a YAML config (see [`configs/`](configs/)).

### A note on Ollama

Ollama **cannot be the steered model**: steering injects a vector into the
residual stream *during the forward pass*, which requires PyTorch forward hooks.
Ollama serves GGUF text-in/text-out over REST with no access to intermediate
activations. Scalpel therefore uses Ollama only as the **LLM-judge** for the
effect metric, and steers a Hugging Face / TransformerLens model instead.

---

## Install

Scalpel targets **Python 3.11–3.12** (PyTorch / TransformerLens wheels lag newer
interpreters). Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

# Core + dev tools (enough for the mock path, the unit tests, and CI):
uv pip install -e ".[dev]"

# Add the heavy model stack when you want to run a real model:
uv pip install -e ".[models,judge,viz]"
```

Copy the environment template and adjust as needed:

```bash
cp .env.example .env
```

---

## Quick start

Run the **download-free smoke check** (mock backend — no model, no network). It
exercises the full path: capture activations → SAE encode/decode → report
reconstruction quality.

```bash
scalpel smoke --backend mock
```

Run the **real CPU smoke check** on gpt2-small + a released SAE (downloads the
model and SAE on first run, then caches them; needs the `models` extra):

```bash
scalpel smoke --config configs/gpt2-small.yaml
```

Real output on gpt2-small (`gpt2-small-res-jb`, layer 7) — the SAE recovers
~99.9% of the residual-stream variance with ~47 active latents per token, which
is exactly the healthy range for these SAEs:

```
scalpel smoke
  backend           : transformerlens
  model             : gpt2
  device            : cpu
  hook              : blocks.7.hook_resid_pre
  d_model / d_sae   : 768 / 24576
  tokens            : 48
  reconstruction MSE: 0.654657
  variance explained: 0.9991
  mean L0 (sparsity): 47.00
```

(The MSE is large only because the residual stream itself has a large norm —
variance-explained is the meaningful reconstruction metric.)

---

## CLI

| Command | Status | Purpose |
| --- | --- | --- |
| `scalpel smoke` | ✅ M1 | Load a model + SAE and report reconstruction error. |
| `scalpel discover --concept X` | 🔜 M2 | Find the SAE feature(s) most associated with a concept. |
| `scalpel steer --feature N --coef C --prompt "..."` | 🔜 M3 | Steer generation with a feature direction. |
| `scalpel eval` | 🔜 M4 | Coefficient sweep → dose-response, fluency, specificity, baselines. |

---

## How steering works (the mechanics)

The decoder columns of the SAE are directions in residual-stream space. Row `i`
of `W_dec` is the unit that latent `i` writes to — and it is exactly the vector
Scalpel adds to steer the model:

```
steering_vector = SAE.W_dec[feature_index]          # shape [d_model]
residual[layer] += coef * steering_vector           # added during generation
```

Sweeping `coef` (including negative values for suppression) produces the
dose-response curve. The **baseline controls** replace `steering_vector` with:

- a **random Gaussian direction** L2-normalised to the same norm, and
- a **mean-difference** vector: `mean(resid | concept-positive) − mean(resid | concept-negative)`.

The SAE feature should deliver more targeted effect **per unit of fluency cost**
than either baseline.

---

## Development

```bash
pytest             # unit suite (CPU-only, no downloads)
ruff check .       # lint
black --check .    # format
mypy               # types
```

The unit tests are fully offline: the mock backend and hand-built tiny SAEs
cover the SAE math, steering-vector construction, metric math, config, and CLI
wiring without touching a real model. CI runs this suite on
`{ubuntu, macos} × {py3.11, py3.12}`; a separate, gated job runs the real
gpt2-small reconstruction smoke.

---

## Roadmap

1. **✅ Scaffold + reconstruction sanity** — package, config, SAE loading, `scalpel smoke`.
2. **Feature discovery** — max-activating examples over a corpus; select a target feature; Neuronpedia labels.
3. **Steering hook** — inject the feature direction during generation; qualitative before/after.
4. **Measurement** — effect + fluency metrics; coefficient sweep → dose-response plot.
5. **Controls** — random-direction and mean-difference baselines; specificity panel.
6. **Package** — CLI polish, reproducible demo notebook, README with plots + results table.

---

## Reproducibility

Fixed seeds across Python / NumPy / PyTorch; content-addressed activation
caching; greedy or seeded sampling. Some GPU/MPS kernels are not bit-for-bit
deterministic even with a fixed seed — this residual nondeterminism is expected
and documented rather than hidden.

**Apple Silicon / MPS caveat.** On some PyTorch builds TransformerLens warns that
the MPS backend "may be silently incorrect". The portable `gpt2-small` config is
therefore pinned to CPU. The `gemma-2-2b` config uses `device: auto` (→ MPS on an
M-series Mac) for speed; Scalpel acknowledges the opt-in, but for numbers you
intend to report, cross-check a subset on CPU or CUDA.

## License

[MIT](LICENSE) © 2026 Dylan Patriarchi
