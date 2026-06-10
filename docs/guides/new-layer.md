# Adding a Core Layer

This guide walks through adding a new building block to `core/` — for example, a new attention variant, MLP type, or normalization layer.

---

## Step 1: Implement the NNX module in `core/`

All core modules are Flax NNX modules. Add a new file or extend an existing one:

```python
# core/attention.py  (or a new core/my_attention.py)
from flax import nnx
import jax.numpy as jnp

class LinearAttention(nnx.Module):
    """Linear (O(T)) attention via kernel feature maps.

    Args:
        dim: Hidden dimension.
        n_heads: Number of attention heads.
        rngs: Flax NNX random state.
    """

    def __init__(self, dim: int, n_heads: int, rngs: nnx.Rngs) -> None:
        self.q = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.n_heads = n_heads

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply linear attention.

        Args:
            x: Input tensor of shape ``[B, T, D]``.

        Returns:
            Output tensor of shape ``[B, T, D]``.
        """
        q = nnx.elu(self.q(x)) + 1.0   # positive feature map
        k = nnx.elu(self.k(x)) + 1.0
        v = self.v(x)
        # O(T) linear attention
        kv = jnp.einsum("btd, bte -> bde", k, v)
        return jnp.einsum("btd, bde -> bte", q, kv)
```

**Checklist:**
- [ ] Google-style docstring on the class and every public method
- [ ] Type annotations on `__init__` and `__call__`
- [ ] No mutable Python state (side-effect-free, XLA-compatible)

---

## Step 2: Add a config field

Config fields are the single source of truth for architecture choices:

```python
# core/config.py
@dataclass
class ModelConfig:
    ...
    use_linear_attn: bool = False   # Enable O(T) linear attention
```

---

## Step 3: Wire it into `Block` or `Transformer`

```python
# core/block.py
from core.attention import Attention, LinearAttention

class Block(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        if config.use_linear_attn:
            self.attn = LinearAttention(config.dim, config.n_heads, rngs=rngs)
        else:
            self.attn = Attention(config, rngs=rngs)
        ...
```

---

## Step 4: Write a unit test

```python
# tests/test_linear_attention.py
import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from core.attention import LinearAttention

def test_linear_attention_output_shape():
    rngs  = nnx.Rngs(0)
    layer = LinearAttention(dim=64, n_heads=4, rngs=rngs)
    x     = jnp.ones((2, 16, 64))
    y     = layer(x)
    assert y.shape == (2, 16, 64)

def test_linear_attention_deterministic():
    rngs  = nnx.Rngs(0)
    layer = LinearAttention(dim=64, n_heads=4, rngs=rngs)
    x     = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 64))
    assert jnp.allclose(layer(x), layer(x))
```

Run: `pytest tests/test_linear_attention.py -v`

---

## Step 5: Update the default config YAML (optional)

```yaml
# configs/default_config.yaml
model:
  use_linear_attn: false   # New field — default off for backward compat
```

---

## Checklist summary

- [ ] NNX module with Google docstrings + type annotations
- [ ] Config field with a sensible default
- [ ] Wired into `Block` or `Transformer` via the config field
- [ ] At least one unit test covering shape + determinism
- [ ] YAML config updated with the new field
- [ ] `interrogate` passes: `make doccheck`
