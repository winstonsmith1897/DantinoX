# Contributing

Contributions — bug fixes, new paradigms, new benchmark tasks, documentation improvements — are welcome. This page covers the full workflow.

---

## Setup

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX
pip install -U "jax[cpu]" jaxlib
pip install -e ".[all]"
pre-commit install        # install hooks (interrogate + ruff + mypy)
```

---

## Workflow

1. **Fork & branch** — `git checkout -b feat/my-feature` from `main`.
2. **Write code** — follow the [Developer Guide](guides/index.md) for the relevant extension point.
3. **Write docstrings** — every new public class and function must have a Google-style docstring.
4. **Write tests** — add tests in `tests/`. Use `pytest -k my_feature` for quick iteration.
5. **Run CI locally** — `make check` runs lint + typecheck + tests + docstring coverage.
6. **Open a PR** — target `main`. The CI pipeline must be fully green.

---

## Code quality checks

All checks run automatically on each PR via GitHub Actions. Run them locally with:

```bash
make check          # lint + typecheck + test + doccheck (full suite)
make lint           # ruff only
make typecheck      # mypy only
make test           # pytest only
make doccheck       # interrogate only
```

### Checklist before opening a PR

- [ ] `make lint` passes (ruff — style, imports, bugbear)
- [ ] `make typecheck` passes (mypy — typed function signatures)
- [ ] `make test` passes — no new failures, new code has tests
- [ ] `make doccheck` passes — 100 % docstring coverage on new public symbols
- [ ] New features have documentation (at least one relevant `.md` page updated)

---

## Docstring standard

Use **Google style**:

```python
def corrupt(
    tokens: jnp.ndarray,
    t: jnp.ndarray,
    rng: jax.random.KeyArray,
    schedule: NoiseSchedule,
    mask_id: int,
) -> jnp.ndarray:
    """Apply random token masking at corruption level t.

    For each sample i, masks tokens independently with probability
    ``schedule(t[i])``, replacing them with ``mask_id``.

    Args:
        tokens: Integer token IDs of shape ``[B, T]``.
        t: Per-sample corruption levels ``[B]`` in ``[0, 1]``.
        rng: JAX random key.
        schedule: Callable mapping t ∈ [0,1] → masking probability ∈ [0,1].
        mask_id: Vocabulary index for the ``[MASK]`` token.

    Returns:
        Corrupted token IDs of shape ``[B, T]``.
    """
```

**Rules:**
- One-line summary — imperative mood, ends without a period.
- `Args:` section — every parameter, type omitted (already in the signature).
- `Returns:` section — shape and dtype for arrays; plain description for others.
- `Raises:` section — list `ValueError`, `TypeError`, etc. only when they are explicitly raised.
- No `Example:` block unless the usage is non-obvious.

---

## Testing conventions

```python
# tests/test_my_feature.py
import pytest
import jax.numpy as jnp
from flax import nnx


def test_output_shape():
    """Verify the output tensor has the expected shape."""
    ...

def test_input_validation():
    """Verify that invalid inputs raise early with clear messages."""
    with pytest.raises(ValueError, match="causal=False"):
        ...

@pytest.mark.slow
def test_full_training_loop():
    """End-to-end smoke test — skipped in fast CI mode."""
    ...
```

**Conventions:**
- Function names start with `test_`.
- Each test has a one-line docstring.
- Mark slow tests with `@pytest.mark.slow` — they are excluded from fast CI (`pytest -m "not slow"`).
- Test files mirror module paths: `dantinox/paradigms/ar.py` → `tests/test_ar_paradigm.py`.

---

## Documentation

New features need documentation. The bar is:

- **New public API** → update the relevant `docs/api/*.md` page (mkdocstrings pulls from docstrings automatically, but the page must reference the symbol via `:::`)
- **New paradigm / task / chart** → update the corresponding guide page
- **New config field** → update `docs/architecture/core.md` config reference table

Build and preview the docs locally:

```bash
make docs-serve     # starts MkDocs dev server at http://127.0.0.1:8000
make docs-build     # full static build into site/
```

---

## Release process (maintainers)

```bash
make bump-patch     # 0.3.15 → 0.3.16
git add pyproject.toml && git commit -m "chore: bump to 0.3.16"
git tag v0.3.16 && git push origin main --tags
make build && make publish
```
