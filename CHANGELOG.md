# Changelog

All notable changes to DantinoX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
