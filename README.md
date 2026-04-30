<div align="center">

# DantinoX

*"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura..."*

A decoder-only Transformer built from scratch in **JAX** and **Flax NNX** — complete with a training pipeline, autoregressive generation, hyperparameter sweeps, and a benchmarking suite.

<br>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-000000?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-8A2BE2?style=flat-square)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange?style=flat-square)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen?style=flat-square)](https://github.com/winstonsmith1897/DantinoX/actions)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=github)](https://winstonsmith1897.github.io/DantinoX/)

**[Documentation](https://winstonsmith1897.github.io/DantinoX/) · [Coverage Report](https://winstonsmith1897.github.io/DantinoX/coverage/) · [API Reference](https://winstonsmith1897.github.io/DantinoX/api/)**

</div>

---

## Overview

**DantinoX** is a research-grade library for building, training, and benchmarking decoder-only Transformers in pure JAX. It is designed as a transparent, modular codebase for studying how architectural choices — attention mechanism, positional encoding, MoE routing — affect convergence, memory footprint, and inference throughput.

The library ships as an installable Python package (`pip install dantinox`) with a unified CLI, a programmatic Python API, a typed configuration dataclass, and a full test suite.

### Implemented Architectures

| Component | Variants |
| :--- | :--- |
| **Attention** | Multi-Head (MHA) · Grouped-Query (GQA) · Multi-Head Latent (MLA) |
| **Feed-Forward** | Dense MLP (SwiGLU / GELU) · Sparse Mixture-of-Experts (Top-K) |
| **Positional Encoding** | Rotary (RoPE) · Absolute Sinusoidal · Learned |
| **Attention Masking** | Causal · Sliding Window |
| **Memory Optimizations** | Gradient Checkpointing (`nnx.remat`) · Weight Tying · Static KV-Cache |
| **Training** | Gradient Accumulation · AdamW / Adafactor / Lion · Cosine LR Schedule |
| **Tokenizers** | Character-level · Byte-Pair Encoding (BPE) |

---

## Installation

### From PyPI

```bash
pip install dantinox                        # core only
pip install "dantinox[data]"               # + HuggingFace datasets
pip install "dantinox[benchmark]"          # + pandas / matplotlib / scipy
pip install "dantinox[all]"               # everything including dev tools
```

### From Source

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

conda create -n dantinox python=3.12 -y
conda activate dantinox

make install      # installs JAX + all extras in editable mode
```

> **GPU support:** replace the JAX CPU wheels with `pip install -U "jax[cuda12]"` after running `make install`.

---

## Quick Start

### CLI

DantinoX registers a single `dantinox` entry-point with five subcommands:

```bash
# Train from a YAML config
dantinox train --config configs/default_config.yaml --data_path data/corpus.txt

# Override any config field from the command line
dantinox train --config configs/default_config.yaml --data_path data/corpus.txt \
    --lr 3e-4 --use_moe true --num_blocks 8

# Generate text from a saved checkpoint
dantinox generate --run_dir runs/run_20260101_120000 \
    --prompt "Nel mezzo del cammin " --max_new_tokens 200 --temperature 1.2

# Run a W&B Bayesian hyperparameter sweep
dantinox sweep --sweep_config configs/sweep.yaml --data_path data/corpus.txt

# Benchmark all run directories and save metrics to CSV
dantinox benchmark --runs_dir runs --out_csv benchmark_results.csv

# Generate plots from benchmark results
dantinox plot --in_csv benchmark_results.csv --out_dir plots
```

### Python API

```python
from dantinox import Trainer, Generator, BenchmarkRunner
from core.config import Config

# --- Training ---
config = Config(
    dim=512, n_heads=16, head_size=32, kv_heads=4,
    num_blocks=12, max_context=512,
    use_moe=True, n_experts=4, top_k_mlp=2,
    lr=3e-4, batch_size=64, grad_accum=4, epochs=100,
)
trainer = Trainer(config)
run_dir = trainer.fit("data/corpus.txt")

# --- Generation ---
gen = Generator(run_dir)
text = gen.generate(
    "Nel mezzo del cammin ",
    max_new_tokens=200,
    temperature=1.2,
    top_p=0.9,
    use_cache=True,
)
print(text)

# --- Benchmarking ---
runner = BenchmarkRunner("runs")
df = runner.run(out_csv="benchmark_results.csv")
print(df[["run", "type", "params_m", "prefill_ms"]].to_string())
```

---

## Configuration

All architecture and training settings live in a single typed dataclass. YAML files are fully supported and can be partially overridden from the CLI.

```yaml
# configs/default_config.yaml

model:
  dim: 512                      # Hidden dimension (must equal n_heads × head_size)
  n_heads: 16                   # Query heads
  kv_heads: 4                   # Key/value heads — set < n_heads to enable GQA
  head_size: 32                 # Per-head dimension
  num_blocks: 12                # Transformer depth
  max_context: 512              # Maximum sequence length
  weight_tying: true            # Tie embedding ↔ LM-head weights
  activation: gelu              # Activation function (gelu | silu)
  use_swiglu: true              # Replace MLP activation with SwiGLU gate
  gradient_checkpointing: true  # Recompute activations to reduce VRAM
  dropout_rate: 0.15

moe:
  use_moe: false                # Toggle Sparse MoE (true) vs Dense MLP (false)
  n_experts: 4                  # Total number of experts
  top_k_mlp: 2                  # Active experts per token
  expansion: 4                  # Expert hidden-dimension multiplier
  alpha_balance: 0.1            # Load-balancing loss weight

attention:
  use_rotary_pos: true          # Rotary Positional Embedding (RoPE)
  sliding_window: false         # Restrict attention to a local window
  context_window: 4             # Window size (if sliding_window: true)
  no_sink: true                 # Sigmoid attention gate for training stability

  # Multi-Head Latent Attention (MLA)
  mla: false
  down_dim_q: 256               # Query compression dimension
  down_dim_kv: 256              # Key/Value compression dimension
  rope_dim: 32                  # RoPE dimensions for decoupled key encoding

tokenizer:
  tokenizer_type: char          # char | bpe
  tokenizer_path: null

data:
  dataset_source: local         # local | huggingface
  dataset_name: ""

training:
  lr: 0.005
  batch_size: 128
  grad_accum: 16
  optimizer: adamw              # adamw | adafactor | lion
  epochs: 1000
  warmup_steps: 420
  seed: 42
```

### Config Validation

The `Config` dataclass enforces constraints at instantiation:

```python
Config(dim=512, n_heads=16, head_size=32)   # OK — 16 × 32 = 512
Config(dim=512, n_heads=16, head_size=31)   # ConfigError: dim must equal n_heads × head_size
Config(dim=512, n_heads=16, kv_heads=3)     # ConfigError: n_heads must be divisible by kv_heads
```

---

## CLI Reference

### `dantinox train`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--config` | `configs/default_config.yaml` | YAML config file |
| `--data_path` | — | Path to plain-text corpus |
| `--run_dir` | auto-generated | Output directory for weights and logs |
| `--wandb_project` | — | W&B project name for live logging |
| `--<field>` | config value | Override any `Config` field directly |

### `dantinox generate`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--run_dir` | **required** | Run directory with `config.yaml` + `model_weights.msgpack` |
| `--prompt` | `"Nel mezzo del cammin "` | Input text prefix |
| `--max_new_tokens` | `150` | Number of tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top_p` | `null` | Nucleus sampling threshold |
| `--top_k` | `null` | Top-K sampling limit |
| `--greedy` | `false` | Deterministic greedy decoding |
| `--no_cache` | `false` | Disable KV-cache (slower, for debugging) |
| `--seed` | `42` | RNG seed |

### `dantinox benchmark`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--runs_dir` | `runs` | Directory containing run sub-directories |
| `--runs` | all | Specific run names to benchmark |
| `--out_csv` | — | Save results to this CSV path |

### `dantinox sweep`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--sweep_config` | `configs/sweep.yaml` | W&B sweep YAML |
| `--config` | `configs/default_config.yaml` | Base model config (overridden by sweep) |
| `--data_path` | **required** | Training corpus |
| `--wandb_project` | `DantinoX` | W&B project |
| `--count` | unlimited | Maximum sweep runs |

---

## Project Structure

```
DantinoX/
├── core/                        # Neural network primitives
│   ├── config.py                # Config dataclass — single source of truth
│   ├── model.py                 # Transformer: embedding → blocks → LM head
│   ├── attention.py             # MHA / GQA / MLA + RoPE + KV-cache
│   ├── block.py                 # Transformer block (Attention + FFN + LayerNorm)
│   ├── mlp.py                   # Dense MLP (SwiGLU / GELU)
│   ├── moe.py                   # Sparse Mixture-of-Experts with load-balancing loss
│   └── generation.py            # Autoregressive decode loop (fori_loop + vmap)
│
├── dantinox/                    # Installable library package
│   ├── cli.py                   # `dantinox` entry-point (train/generate/sweep/benchmark/plot)
│   ├── trainer.py               # Trainer — JIT training loop, logging, checkpointing
│   ├── generator.py             # Generator — checkpoint loading + text generation
│   ├── bench.py                 # BenchmarkRunner — latency / throughput / FLOPs
│   ├── plotting.py              # Plotter — automated figure generation
│   └── exceptions.py            # Exception hierarchy (DantinoXError → sub-classes)
│
├── utils/
│   ├── tokenizer.py             # CharTokenizer · BPETokenizer · Tokenizer Protocol
│   └── helpers.py               # Loss · batch sampling · LR schedule
│
├── configs/                     # YAML configuration files
│   ├── default_config.yaml
│   └── sweep.yaml
│
├── tests/                       # Pytest test suite (22 tests)
│   ├── conftest.py              # Session-scoped Config fixtures
│   ├── test_model.py            # Forward pass · GQA · MoE · weight tying · JIT
│   └── test_mla.py              # MLA training · inference cache · RoPE constraints
│
├── pyproject.toml               # Package metadata, deps, ruff, mypy, pytest config
├── Makefile                     # Development targets
└── mkdocs.yml                   # Documentation site configuration
```

### Exception Hierarchy

```
DantinoXError
├── ConfigError        — invalid or inconsistent Config fields
├── CheckpointError    — missing run directory, config, or weights
├── BenchmarkError     — failure loading or running a benchmark
└── PlotError          — missing CSV or plot module import failure
```

---

## Development

All common workflows are exposed through `make`:

```bash
make install      # Install package + all dev/doc dependencies (editable)
make test         # Run test suite with coverage report
make lint         # Ruff static analysis
make typecheck    # Mypy type checking
make check        # lint + typecheck + test  (run before every push)
make build        # Build sdist + wheel into dist/
make publish      # Upload dist/ to PyPI via twine
make clean        # Remove build artefacts and __pycache__
```

### Running Tests

```bash
make test

# Or directly:
JAX_PLATFORM_NAME=cpu python -m pytest tests/ -v
```

The suite runs on CPU (no GPU required) and covers:

- Forward-pass output shapes for MHA, GQA, and MLA
- KV-cache correctness and accumulation
- MoE load-balancing loss propagation
- Weight tying between embedding and LM head
- JIT compilation stability
- `Config` validation (dim constraints, GQA divisibility, MLA rope_dim)
- `Config` round-trip serialization (`to_dict` / `from_dict`)

Coverage output is written to `docs/coverage/` and published automatically with the documentation site.

### Code Quality

The project enforces a strict quality baseline:

| Tool | Configuration | What it checks |
| :--- | :--- | :--- |
| **ruff** | `pyproject.toml` | Style (E/W), imports (I), pyupgrade (UP), bugbear (B), simplify (SIM) |
| **mypy** | `pyproject.toml` | Full type annotation coverage across `dantinox/`, `core/`, `utils/` |
| **pytest** | `pyproject.toml` | 22 unit tests, CPU-only, session-scoped fixtures |

---

## Training Artifacts

Each training run writes an isolated artifact directory:

```
runs/run_20260101_120000/
├── config.yaml              # Exact config used for the run
├── model_summary.json       # Parameter counts per component
├── training_log.csv         # step, train_loss, val_loss, bal_loss, ms_per_step
└── model_weights.msgpack    # Serialized model state (Flax msgpack format)
```

The training loop logs to console via `tqdm` with live loss postfix, and optionally streams metrics to **Weights & Biases** when `--wandb_project` is specified.

---

## Benchmarking

`BenchmarkRunner` measures latency and throughput across a matrix of sequence lengths and batch sizes using XLA cost analysis for FLOPs:

```python
from dantinox import BenchmarkRunner
from dantinox.plotting import Plotter

df = BenchmarkRunner("runs").run(out_csv="benchmark_results.csv")
Plotter("benchmark_results.csv", out_dir="plots").run()
```

**Reported metrics per run:**

| Metric | Description |
| :--- | :--- |
| `params_m` | Total trainable parameters (millions) |
| `theoretical_cache_mb` | KV-cache memory at `max_context` (MB) |
| `prefill_ms` | Prefill latency for a 256-token prompt |
| `tps_{64,128,256,512}` | Decode throughput (tok/s) at each sequence length |
| `tps_bs{1,4,16,64,...}` | Decode throughput at each batch size |
| `decode_gflops` | FLOPs per decode step (XLA cost analysis) |
| `prefill_arith_int` | Arithmetic intensity of the prefill kernel |
| `val_loss` | Final validation loss from `training_log.csv` |

---

## Empirical Results

Ablation studies were conducted via W&B Bayesian sweeps over 100+ configurations. Key findings:

- **MLA vs GQA vs MHA:** MLA achieves lower KV-cache memory with comparable perplexity when `down_dim_kv ≤ dim / 4`.
- **SwiGLU:** Consistently outperforms GELU by ~0.05 val-loss across all model sizes.
- **Sliding Window:** Improves training speed on long contexts with negligible perplexity loss when `context_window ≥ 64`.
- **Attention Gating (`no_sink`):** Stabilizes training when combined with RoPE at high learning rates.
- **MoE (Top-2/4):** Matches dense perplexity at 60% of the active-parameter count.

Full charts and analysis: [Ablation Studies](https://winstonsmith1897.github.io/DantinoX/ablation_studies/)

---

## Documentation

The full documentation is built with MkDocs Material and deployed to GitHub Pages:

```bash
# Rebuild and deploy
mkdocs gh-deploy --force
```

Sections:

- [Architecture](https://winstonsmith1897.github.io/DantinoX/architecture/) — attention variants, MoE, positional encodings
- [Training & Sweeps](https://winstonsmith1897.github.io/DantinoX/training/) — training loop internals, W&B integration
- [Inference & Generation](https://winstonsmith1897.github.io/DantinoX/generation/) — KV-cache, decoding strategies
- [Benchmarks](https://winstonsmith1897.github.io/DantinoX/benchmarks/) — throughput and FLOPs analysis
- [API Reference](https://winstonsmith1897.github.io/DantinoX/api/) — auto-generated from docstrings
- [Coverage Report](https://winstonsmith1897.github.io/DantinoX/coverage/) — line-level test coverage

---

## License

MIT — see [LICENSE](LICENSE).
