"""Tests for core.model.Transformer using the real Config."""

import jax.numpy as jnp
import pytest
from flax import nnx

from core.config import Config
from core.model import Transformer

# ── Forward pass ─────────────────────────────────────────────────────────────

def test_training_forward_shape(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    logits, kv_caches, bal_loss = model(
        batch_input, use_cache=False, kv_caches=None, cache_index=0
    )
    assert logits.shape == (*batch_input.shape, tiny_config.vocab_size)
    assert len(kv_caches) == tiny_config.num_blocks
    assert not jnp.isnan(logits).any()


def test_no_nan_in_output(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    logits, _, _ = model(batch_input, use_cache=False, kv_caches=None, cache_index=0)
    assert not jnp.isnan(logits).any()
    assert not jnp.isinf(logits).any()


# ── KV cache ─────────────────────────────────────────────────────────────────

def test_inference_kv_cache(tiny_config, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    x = jnp.array([[42, 7]], dtype=jnp.int32)

    logits_1, cache_1, _ = model(x, use_cache=True, kv_caches=None, cache_index=0)
    assert logits_1.shape == (1, 2, tiny_config.vocab_size)

    x_next = jnp.array([[15]], dtype=jnp.int32)
    logits_2, cache_2, _ = model(x_next, use_cache=True, kv_caches=cache_1, cache_index=2)
    assert logits_2.shape == (1, 1, tiny_config.vocab_size)


# ── MoE ──────────────────────────────────────────────────────────────────────

def test_moe_balancing_loss(tiny_moe_config, batch_input, rngs):
    model = Transformer(tiny_moe_config, rngs=rngs)
    _, _, bal_loss = model(batch_input, use_cache=False, kv_caches=None, cache_index=0)
    assert bal_loss is not None
    assert float(bal_loss) >= 0.0


# ── JIT compilation ───────────────────────────────────────────────────────────

def test_jit_compilation(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)

    @nnx.jit
    def forward(m, x):
        return m(x, use_cache=False, kv_caches=None, cache_index=0)

    logits, _, _ = forward(model, batch_input)
    assert logits is not None
    assert not jnp.isnan(logits).any()


# ── Weight tying ──────────────────────────────────────────────────────────────

def test_weight_tying(rngs):
    config = Config(
        dim=128, n_heads=4, head_size=32, num_blocks=2,
        vocab_size=256, max_context=64, kv_heads=2,
        weight_tying=True, gradient_checkpointing=False, dropout_rate=0.0,
    )
    model = Transformer(config, rngs=rngs)
    assert jnp.array_equal(model.lm_head.kernel, model.wte.embedding.T)


def test_no_weight_tying(rngs):
    config = Config(
        dim=128, n_heads=4, head_size=32, num_blocks=2,
        vocab_size=256, max_context=64, kv_heads=2,
        weight_tying=False, gradient_checkpointing=False, dropout_rate=0.0,
    )
    model = Transformer(config, rngs=rngs)
    assert model.lm_head.kernel.shape == (128, 256)


# ── GQA ──────────────────────────────────────────────────────────────────────

def test_gqa_forward(tiny_gqa_config, batch_input, rngs):
    model = Transformer(tiny_gqa_config, rngs=rngs)
    logits, _, _ = model(batch_input, use_cache=False, kv_caches=None, cache_index=0)
    assert logits.shape == (*batch_input.shape, tiny_gqa_config.vocab_size)
    assert not jnp.isnan(logits).any()


# ── MLA ──────────────────────────────────────────────────────────────────────

def test_mla_training_forward(tiny_mla_config, rngs):
    model = Transformer(tiny_mla_config, rngs=rngs)
    x = jnp.ones((2, 8), dtype=jnp.int32)
    logits, _, _ = model(x, use_cache=False, kv_caches=None, cache_index=0)
    assert logits.shape == (2, 8, tiny_mla_config.vocab_size)
    assert not jnp.isnan(logits).any()


# ── Config validation ─────────────────────────────────────────────────────────

def test_config_dim_mismatch():
    with pytest.raises(ValueError, match="dim.*must equal"):
        Config(dim=100, n_heads=4, head_size=32)  # 100 != 4*32


def test_config_kv_heads_not_divisible():
    with pytest.raises(ValueError, match="divisible"):
        Config(dim=128, n_heads=4, head_size=32, kv_heads=3)  # 4 % 3 != 0


def test_config_mla_rope_too_large():
    with pytest.raises(ValueError, match="rope_dim"):
        Config(
            dim=128, n_heads=4, head_size=32, kv_heads=2,
            mla=True, rope_dim=64,  # 64 > 32 (head_size)
        )


# ── Config serialisation ──────────────────────────────────────────────────────

def test_config_to_dict_roundtrip(tiny_config):
    d = tiny_config.to_dict()
    restored = Config.from_dict(d)
    assert restored.dim == tiny_config.dim
    assert restored.n_heads == tiny_config.n_heads
    assert restored.mla == tiny_config.mla


def test_config_from_dict_ignores_unknown_keys(tiny_config):
    d = tiny_config.to_dict()
    d["totally_unknown_key"] = 999
    restored = Config.from_dict(d)  # should not raise
    assert not hasattr(restored, "totally_unknown_key")


def test_config_repr(tiny_config):
    r = repr(tiny_config)
    assert "MHA" in r or "GQA" in r or "MLA" in r
    assert "dim=" in r
