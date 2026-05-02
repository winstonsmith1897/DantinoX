"""Tests for MLA Attention using the real Config."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from core.attention import Attention
from core.config import Config

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mla_train_config() -> Config:
    return Config(
        dim=256, n_heads=4, head_size=64,  # 4*64=256
        kv_heads=2, num_blocks=2, vocab_size=256, max_context=128,
        mla=True, inference=False,
        down_dim_q=32, down_dim_kv=32, rope_dim=32,
        gradient_checkpointing=False, dropout_rate=0.0,
    )


@pytest.fixture(scope="module")
def mla_inference_config() -> Config:
    return Config(
        dim=256, n_heads=4, head_size=64,
        kv_heads=2, num_blocks=2, vocab_size=256, max_context=128,
        mla=True, inference=True,
        down_dim_q=32, down_dim_kv=32, rope_dim=32,
        gradient_checkpointing=False, dropout_rate=0.0,
    )


@pytest.fixture
def hidden(mla_train_config) -> jnp.ndarray:
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (2, 10, mla_train_config.dim))


# ── Training mode ─────────────────────────────────────────────────────────────

def test_training_output_shape(mla_train_config, hidden):
    model = Attention(mla_train_config, nnx.Rngs(0))
    out, cache = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0)
    assert out.shape == hidden.shape, f"Expected {hidden.shape}, got {out.shape}"


def test_training_no_nan(mla_train_config, hidden):
    model = Attention(mla_train_config, nnx.Rngs(0))
    out, _ = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0)
    assert not jnp.isnan(out).any()


# ── Inference / KV-cache mode ─────────────────────────────────────────────────

def test_inference_cache_shape(mla_inference_config):
    config = mla_inference_config
    model = Attention(config, nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(2), (2, 1, config.dim))
    out_1, cache_1 = model(x, use_cache=True, kv_cache=(None, None), cache_index=0)

    assert out_1.shape == (2, 1, config.dim)
    # MLA compressed KV cache: (batch, max_context, down_dim_kv)
    assert cache_1[0].shape == (2, config.max_context, config.down_dim_kv)


def test_inference_cache_accumulates(mla_inference_config):
    config = mla_inference_config
    model = Attention(config, nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 1, config.dim))
    out_1, cache_1 = model(x, use_cache=True, kv_cache=(None, None), cache_index=0)
    out_2, cache_2 = model(x, use_cache=True, kv_cache=cache_1, cache_index=1)

    assert out_2.shape == (2, 1, config.dim)
    assert isinstance(cache_2, tuple)
    assert not jnp.isnan(out_2).any()


# ── JIT compilation ───────────────────────────────────────────────────────────

def test_jit_compilation(mla_train_config, hidden):
    model = Attention(mla_train_config, nnx.Rngs(0))

    @nnx.jit
    def forward(m, x):
        return m(x, use_cache=False, kv_cache=(None, None), cache_index=0)

    out, _ = forward(model, hidden)
    assert out is not None
    assert not jnp.isnan(out).any()


# ── rope_dim constraint ───────────────────────────────────────────────────────

def test_rope_dim_too_large_rejected():
    with pytest.raises(ValueError, match="rope_dim"):
        Config(
            dim=256, n_heads=4, head_size=64, kv_heads=2,
            mla=True, use_rotary_pos=True,
            rope_dim=128,  # 128 > 64 (head_size) — must raise
        )


def test_rope_dim_equal_head_size_accepted():
    cfg = Config(
        dim=256, n_heads=4, head_size=64, kv_heads=2, num_blocks=2,
        vocab_size=64, max_context=64,
        mla=True, use_rotary_pos=True,
        rope_dim=64,  # equal to head_size — allowed
        gradient_checkpointing=False, dropout_rate=0.0,
    )
    assert cfg.rope_dim == cfg.head_size
