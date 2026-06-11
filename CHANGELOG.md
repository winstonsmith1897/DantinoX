# Changelog

All notable changes to DantinoX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.4.0] — 2026-06-11

### Changed — package layout (action required by 0.5.0)

- **`core` and `utils` moved into the package**: `core/` → `dantinox/core/`, `utils/` → `dantinox/utils/`. Installing dantinox no longer pollutes site-packages with generic top-level `core` / `utils` packages. Thin top-level shims keep old imports (`from core.config import …`) working with a `DeprecationWarning`; **the shims will be removed in 0.5.0**.
- Dependency bounds pinned (`jax>=0.4.30,<0.10`, `flax>=0.10,<0.13`) — JAX/Flax make frequent breaking changes; the library is now tested against an explicit range.
- New optional extra `elf` (`pip install dantinox[elf]`) installs `transformers` for the frozen T5 encoder used by ELF.

### Added — paradigm Trainer reaches feature parity

- **Validation split** (`TrainingConfig.val_frac`, default 0.1): the best checkpoint is now selected by *validation* loss, not training loss.
- **Gradient accumulation** (`grad_accum`) via `optax.MultiSteps`; the LR schedule advances per optimizer update, not per micro-step.
- **Early stopping** (`patience` epochs without validation improvement).
- **bf16 parameter casting** (`use_bf16`).
- **Full train-state checkpointing**: `train_state.msgpack` stores model + optimizer state every epoch; `Trainer.fit(..., resume=True)` continues a crashed run exactly (epoch, schedule position, best-loss bookkeeping).
- **ELF training through the paradigm Trainer**: new `Paradigm.on_train_start` / `Paradigm.prepare_batch` hooks drive the frozen T5 encoder outside JIT (norm-stats initialisation + per-batch contextual embeddings).
- **Memory-mapped token cache**: corpora are tokenised once into `<source>.<tok>.tokens.npy` and re-read via `np.load(mmap_mode="r")` — large corpora are no longer held in RAM as Python lists.
- Run metadata: the paradigm Trainer now writes `config.yaml` and `tokenizer.json`, so `Generator` / `dx.quick_generate` work on its runs.
- Legacy bridge: `Trainer(Config(...))` (monolithic config) still works — it converts to the paradigm API with a `DeprecationWarning` — and passing the wrong config type raises a `TypeError` that shows the correct usage.

### Fixed

- Paradigm Trainer derived `seq_len`/`vocab_size` from `TrainingConfig` defaults (512/200) instead of the model config — batches now match `max_context` and the tokenizer is checked against the model vocabulary.
- `ARParadigm.generate`, `DiscreteParadigm.generate`, and `ContinuousParadigm.generate` called `core.generation` with non-existent signatures.
- `ContinuousParadigm.loss_fn` skipped T5 embedding normalisation (`model.encode`).
- `dantinox.training` stack used the pre-0.11 Flax NNX `Optimizer` API (`nnx.Optimizer(model, tx)` / `update(grads)`) and crashed on current Flax.
- Multi-device path crashed (`replicate()` on a Module; non-divisible batch sizes now fall back to fewer devices with a warning).
- Checkpoint serialisation of NNX state on Flax ≥ 0.12 (typed RNG keys are excluded; states round-trip via `to_pure_dict`).
- `dx.profile()` no longer smuggles FLOPs via `result.__dict__` — `ProfilingResult` has a real `flops` field.
- `dx.quick_generate()` honours its `paradigm` and `tokenizer` arguments.
- Removed uses of the long-deprecated `jax.random.KeyArray` type alias.

### Deprecated

- Top-level `core` / `utils` packages (use `dantinox.core` / `dantinox.utils`).
- `dantinox.trainer.Trainer` (monolithic-Config engine) — use `dantinox.Trainer` with a Paradigm. The CLI still drives the legacy engine for `sweep`/`find-lr`.

### Repo

- `site/` (generated docs) and root scratchpad scripts untracked; scratch scripts moved to `scripts/scratch/`, demos to `examples/`.
- `examples/DantinoX_Colab.ipynb` rewritten for the paradigm API, including a new ELF section.

---

## [0.1.0] — 2026-04-30

### Added

**Library package (`dantinox/`)**
- `Trainer` class — programmatic training with optional W&B logging, `tqdm` progress bar, and `run_dir` control.
- `Generator` class — loads a checkpoint and generates text with full sampling control (greedy, Top-K, Top-P, temperature).
- `BenchmarkRunner` class — benchmarks decode throughput, prefill latency, and FLOPs across a directory of run checkpoints.
- `Plotter` class — wraps all standalone plot scripts and generates PNG figures from a benchmark CSV.
- `dantinox.exceptions` — typed exception hierarchy: `DantinoXError`, `ConfigError`, `CheckpointError`, `BenchmarkError`, `PlotError`.
- `dantinox/py.typed` — PEP 561 marker for type-checker support.

**CLI (`dantinox` entry point)**
- `dantinox train` — train a model from a YAML config and corpus, with per-field CLI overrides.
- `dantinox generate` — generate text from a saved checkpoint.
- `dantinox sweep` — launch a W&B Bayesian hyperparameter sweep.
- `dantinox benchmark` — benchmark all (or selected) run directories.
- `dantinox plot` — generate all benchmark plots from a results CSV, with group filtering.

**Configuration (`core/config.py`)**
- `Config.from_dict()` — construct from a plain dict, ignoring unknown keys.
- `Config.to_dict()` — serialise to a plain dict.
- `Config.__repr__()` — human-readable summary including attention type and MoE flag.
- Replaced silent `assert` statements with `ValueError` messages.

**Packaging**
- `pyproject.toml` with `[tool.pytest]`, `[tool.ruff]`, and `[tool.mypy]` sections.
- Optional dependency groups: `data`, `benchmark`, `dev`, `all`.
- `tqdm` added as a core dependency.
- `scipy` added to the `benchmark` extra (used by insight plots).

**Testing**
- `tests/conftest.py` — shared session-scoped fixtures using the real `core.config.Config` with validated tiny configurations: MHA, GQA, MLA, MoE.
- `tests/test_model.py` — rewritten to use real `Config`; covers forward shape, NaN checks, KV cache, MoE loss, JIT compilation, weight tying, GQA, MLA, and Config validation.
- `tests/test_mla.py` — rewritten to use real `Config`; covers training and inference modes, cache shape, cache accumulation, JIT, and `rope_dim` constraint enforcement.

**CI**
- `.github/workflows/tests.yml` — runs `ruff`, `mypy`, and `pytest` on Python 3.10 and 3.12 on every push/PR. Installs via `pip install -e ".[dev]"`.
- `.github/workflows/release.yml` — builds and publishes to PyPI on any `v*` git tag using OIDC trusted publishing.

**Documentation**
- `docs/index.md` — tabbed quickstart: Library API, CLI, legacy scripts. Updated grid cards and project structure tree.
- `docs/api.md` — new "High-level API" section with auto-generated docs for `Trainer`, `Generator`, `BenchmarkRunner`, and `Plotter`.
- `.gitignore` — restructured with labelled sections; added `site/`, `plots/`, `*.safetensors`, IDE dirs, and labelled scratchpad exclusions.

### Changed
- `train.py`, `generate.py`, `benchmark.py` are now thin wrappers that delegate to `dantinox.cli`.
- All `dantinox/` modules now use `logging` instead of `print` statements.

---

## [0.0.1] — 2026-01-01 (pre-library)

Initial research codebase. Standalone scripts for training (`train.py`), generation (`generate.py`), benchmarking (`benchmark.py`), and hyperparameter sweeps (`train_sweep.py`). Attention variants: MHA, GQA, MLA. 90+ W&B runs.
