"""
Shared pytest fixtures for DantinoX tests.

All model fixtures use the real ``core.config.Config`` so that
``__post_init__`` validation is exercised and no silent divergence
can occur between test and production configs.
"""

import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Force CPU so tests run anywhere without a GPU.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from core.config import Config

# ── Tiny model configs ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_config() -> Config:
    """Minimal MHA config that satisfies all Config constraints."""
    return Config(
        dim=128,
        n_heads=4,
        head_size=32,       # 4 * 32 == 128 ✓
        num_blocks=2,
        vocab_size=256,
        max_context=64,
        kv_heads=2,         # 4 % 2 == 0 ✓
        gradient_checkpointing=False,
        dropout_rate=0.0,
        epochs=1,
        batch_size=4,
        grad_accum=1,
        eval_iters=1,
    )


@pytest.fixture(scope="session")
def tiny_gqa_config() -> Config:
    """Minimal GQA config (kv_heads < n_heads)."""
    return Config(
        dim=128,
        n_heads=4,
        head_size=32,
        num_blocks=2,
        vocab_size=256,
        max_context=64,
        kv_heads=1,
        gradient_checkpointing=False,
        dropout_rate=0.0,
        epochs=1,
        batch_size=4,
        grad_accum=1,
    )


@pytest.fixture(scope="session")
def tiny_mla_config() -> Config:
    """Minimal MLA config with decoupled RoPE."""
    return Config(
        dim=256,
        n_heads=4,
        head_size=64,       # 4 * 64 == 256 ✓
        num_blocks=2,
        vocab_size=256,
        max_context=128,
        kv_heads=2,
        mla=True,
        inference=False,
        down_dim_q=32,
        down_dim_kv=32,
        rope_dim=32,        # 32 <= 64 ✓
        gradient_checkpointing=False,
        dropout_rate=0.0,
        epochs=1,
        batch_size=4,
        grad_accum=1,
    )


@pytest.fixture(scope="session")
def tiny_moe_config() -> Config:
    """Minimal MoE config."""
    return Config(
        dim=128,
        n_heads=4,
        head_size=32,
        num_blocks=2,
        vocab_size=256,
        max_context=64,
        kv_heads=4,
        use_moe=True,
        n_experts=4,
        top_k_mlp=2,
        gradient_checkpointing=False,
        dropout_rate=0.0,
        epochs=1,
        batch_size=4,
        grad_accum=1,
    )


# ── Common inputs ─────────────────────────────────────────────────────────────

@pytest.fixture
def batch_input(tiny_config: Config) -> jnp.ndarray:
    """A (batch=2, seq=10) token tensor."""
    key = jax.random.PRNGKey(0)
    return jax.random.randint(key, (2, 10), 0, tiny_config.vocab_size)


@pytest.fixture
def rngs() -> nnx.Rngs:
    return nnx.Rngs(0)
