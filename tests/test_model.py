"""Tests for core.model.Transformer using the real Config."""

import jax.numpy as jnp
import pytest
from flax import nnx

from dantinox.core.config import Config
from dantinox.core.model import Transformer

# ── Forward pass ─────────────────────────────────────────────────────────────

def test_training_forward_shape(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    out = model(batch_input)
    assert out.logits.shape == (*batch_input.shape, tiny_config.vocab_size)
    assert len(out.kv_caches) == tiny_config.num_blocks
    assert not jnp.isnan(out.logits).any()


def test_no_nan_in_output(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    logits, _, _ = model(batch_input)
    assert not jnp.isnan(logits).any()
    assert not jnp.isinf(logits).any()


# ── KV cache ─────────────────────────────────────────────────────────────────

def test_inference_kv_cache(tiny_config, rngs):
    model = Transformer(tiny_config, rngs=rngs)
    x = jnp.array([[42, 7]], dtype=jnp.int32)

    logits_1, cache_1, _ = model(x, caches=tuple((None,None) for _ in range(model.num_blocks)), cache_index=0)
    assert logits_1.shape == (1, 2, tiny_config.vocab_size)

    x_next = jnp.array([[15]], dtype=jnp.int32)
    logits_2, cache_2, _ = model(x_next, caches=cache_1, cache_index=2)
    assert logits_2.shape == (1, 1, tiny_config.vocab_size)


# ── MoE ──────────────────────────────────────────────────────────────────────

def test_moe_balancing_loss(tiny_moe_config, batch_input, rngs):
    model = Transformer(tiny_moe_config, rngs=rngs)
    _, _, bal_loss = model(batch_input)
    assert bal_loss is not None
    assert float(bal_loss) >= 0.0


# ── JIT compilation ───────────────────────────────────────────────────────────

def test_jit_compilation(tiny_config, batch_input, rngs):
    model = Transformer(tiny_config, rngs=rngs)

    @nnx.jit
    def forward(m, x):
        return m(x)

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
    # With weight tying, lm_head is None and wte.embedding is used directly.
    assert model.lm_head is None
    assert model.weight_tying is True
    # Forward pass must still produce correct output shape.
    x = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    logits, _, _ = model(x)
    assert logits.shape == (1, 3, 256)


def test_no_weight_tying(rngs):
    config = Config(
        dim=128, n_heads=4, head_size=32, num_blocks=2,
        vocab_size=256, max_context=64, kv_heads=2,
        weight_tying=False, gradient_checkpointing=False, dropout_rate=0.0,
    )
    model = Transformer(config, rngs=rngs)
    assert model.lm_head is not None
    assert model.lm_head.kernel[...].shape == (128, 256)


# ── GQA ──────────────────────────────────────────────────────────────────────

def test_gqa_forward(tiny_gqa_config, batch_input, rngs):
    model = Transformer(tiny_gqa_config, rngs=rngs)
    logits, _, _ = model(batch_input)
    assert logits.shape == (*batch_input.shape, tiny_gqa_config.vocab_size)
    assert not jnp.isnan(logits).any()


# ── MLA ──────────────────────────────────────────────────────────────────────

def test_mla_training_forward(tiny_mla_config, rngs):
    model = Transformer(tiny_mla_config, rngs=rngs)
    x = jnp.ones((2, 8), dtype=jnp.int32)
    logits, _, _ = model(x)
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
