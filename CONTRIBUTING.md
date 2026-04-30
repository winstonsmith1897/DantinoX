# Contributing to DantinoX

Thank you for considering a contribution. The sections below cover everything you need to go from a fresh clone to an open pull request.

---

## Development setup

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

# Create a dedicated environment (Python 3.10 or 3.12 recommended)
conda create -n dantinox-dev python=3.12 -y
conda activate dantinox-dev

# Install JAX with CPU support for local dev (GPU not required for tests)
pip install "jax[cpu]" jaxlib

# Install the package in editable mode with all dev tools
pip install -e ".[all]"
```

---

## Running the test suite

```bash
# Run the full test suite
pytest

# Run a single file
pytest tests/test_model.py -v

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=dantinox --cov=core --cov-report=term-missing
```

Tests force JAX onto CPU via `JAX_PLATFORM_NAME=cpu` in `conftest.py`, so no GPU is required.

---

## Linting and type-checking

```bash
# Lint (errors only, no auto-fix)
ruff check dantinox/ core/ utils/

# Lint and auto-fix safe issues
ruff check --fix dantinox/ core/ utils/

# Type-check
mypy dantinox/ core/
```

CI runs both on every push. Ruff errors are blocking; mypy is non-blocking until full annotation coverage is reached.

---

## Project structure

```
dantinox/       Public library API — Trainer, Generator, BenchmarkRunner, Plotter, CLI
core/           Internal implementation — Config, Transformer, Attention, generation engine
utils/          Tokenizers and training helpers
tests/          pytest suite (conftest.py has shared fixtures)
configs/        YAML configs for training and sweeps
docs/           MkDocs Material source
plot_*.py       Standalone benchmark visualisation scripts (wrapped by Plotter)
```

---

## Making a change

1. **Fork** the repo and create a branch: `git checkout -b feat/my-feature`
2. Write code and **tests** for it. All public API changes must be covered.
3. Make sure `ruff check` and `pytest` both pass.
4. Update `CHANGELOG.md` under `[Unreleased]` with a one-line entry.
5. Open a pull request against `main`.

### Commit style

Use short imperative sentences:

```
Add RoPE scaling factor to Config
Fix MLA cache shape when batch_size > 1
Refactor Trainer to accept a pre-built tokenizer
```

---

## Versioning and releases

DantinoX follows [Semantic Versioning](https://semver.org):

| Change | Version bump |
|--------|-------------|
| Breaking API change | Major (`1.0.0 → 2.0.0`) |
| Backwards-compatible feature | Minor (`0.1.0 → 0.2.0`) |
| Bug fix or internal refactor | Patch (`0.1.0 → 0.1.1`) |

Releases are cut by pushing a `v*` tag — the `release.yml` CI workflow then builds and publishes to PyPI automatically.

```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Reporting bugs

Open a GitHub issue with:

- Python and JAX versions (`python --version`, `python -c "import jax; print(jax.__version__)"`)
- Minimal reproducible example
- Full traceback

---

## Questions

Open a GitHub Discussion or reach out via the issue tracker.
