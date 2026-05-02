"""Tests for the library improvements: RMSNorm, ModelOutput, schedulers, RoPE scaling, Flash Attention."""

import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from core.block import RMSNorm, _build_norm
from core.config import Config
from core.model import Transformer
from core.output import ModelOutput
from dantinox.trainer import _build_schedule

# ── RMSNorm ───────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        rngs = nnx.Rngs(0)
        norm = RMSNorm(64, rngs=rngs)
        x = jnp.ones((2, 8, 64))
        y = norm(x)
        assert y.shape == x.shape

    def test_no_nan(self):
        rngs = nnx.Rngs(0)
        norm = RMSNorm(64, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))
        y = norm(x)
        assert not jnp.isnan(y).any()

    def test_normalises_rms(self):
        rngs = nnx.Rngs(0)
        norm = RMSNorm(64, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 64)) * 10
        y = norm(x)
        # After RMSNorm the RMS should be ~1 (scale=ones at init)
        rms = jnp.sqrt(jnp.mean(y ** 2))
        assert abs(float(rms) - 1.0) < 0.05

    def test_build_norm_rmsnorm(self):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=2,
            vocab_size=64, max_context=32, kv_heads=2,
            norm_type="rmsnorm", gradient_checkpointing=False, dropout_rate=0.0,
        )
        rngs = nnx.Rngs(0)
        norm = _build_norm(config, 128, rngs)
        assert isinstance(norm, RMSNorm)

    def test_build_norm_layernorm(self):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=2,
            vocab_size=64, max_context=32, kv_heads=2,
            norm_type="layernorm", gradient_checkpointing=False, dropout_rate=0.0,
        )
        rngs = nnx.Rngs(0)
        norm = _build_norm(config, 128, rngs)
        assert isinstance(norm, nnx.LayerNorm)


# ── ModelOutput ───────────────────────────────────────────────────────────────

class TestModelOutput:
    def _make_output(self):
        logits = jnp.ones((2, 8, 64))
        return ModelOutput(logits=logits, kv_caches=(), aux_loss=0.0)

    def test_attribute_access(self):
        out = self._make_output()
        assert out.logits.shape == (2, 8, 64)
        assert out.aux_loss == 0.0

    def test_positional_unpacking(self):
        out = self._make_output()
        logits, kv_caches, aux_loss = out
        assert logits.shape == (2, 8, 64)

    def test_model_returns_model_output(self):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=2,
            vocab_size=64, max_context=32, kv_heads=2,
            gradient_checkpointing=False, dropout_rate=0.0,
        )
        model = Transformer(config, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert isinstance(out, ModelOutput)
        assert out.logits.shape == (1, 8, 64)


# ── RoPE scaling ──────────────────────────────────────────────────────────────

class TestRoPEScaling:
    def _make_model(self, rope_scale_factor: float):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=1,
            vocab_size=64, max_context=32, kv_heads=2,
            use_rotary_pos=True, rope_scale_factor=rope_scale_factor,
            gradient_checkpointing=False, dropout_rate=0.0,
        )
        return Transformer(config, rngs=nnx.Rngs(0))

    def test_default_no_scaling(self):
        model = self._make_model(1.0)
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert not jnp.isnan(out.logits).any()

    def test_scaled_rope(self):
        model = self._make_model(2.0)
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert not jnp.isnan(out.logits).any()

    def test_scaled_rope_angles_differ(self):
        from core.attention import Attention
        config_base = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=1,
            vocab_size=64, max_context=32, kv_heads=2,
            rope_scale_factor=1.0, gradient_checkpointing=False, dropout_rate=0.0,
        )
        config_scaled = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=1,
            vocab_size=64, max_context=32, kv_heads=2,
            rope_scale_factor=4.0, gradient_checkpointing=False, dropout_rate=0.0,
        )
        rngs = nnx.Rngs(0)
        attn_base   = Attention(config_base,   rngs=rngs)
        attn_scaled = Attention(config_scaled, rngs=rngs)
        assert not jnp.allclose(attn_base.angle, attn_scaled.angle)


# ── Flash Attention toggle ────────────────────────────────────────────────────

class TestFlashAttention:
    def _config(self, use_flash: bool) -> Config:
        return Config(
            dim=128, n_heads=4, head_size=32, num_blocks=2,
            vocab_size=64, max_context=32, kv_heads=4,
            use_flash_attention=use_flash,
            gradient_checkpointing=False, dropout_rate=0.0,
        )

    def test_flash_off_by_default(self):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=1,
            vocab_size=64, max_context=32, kv_heads=2,
            gradient_checkpointing=False, dropout_rate=0.0,
        )
        assert config.use_flash_attention is False

    def test_flash_disabled_runs(self):
        model = Transformer(self._config(False), rngs=nnx.Rngs(0))
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert not jnp.isnan(out.logits).any()

    def test_flash_enabled_runs(self):
        model = Transformer(self._config(True), rngs=nnx.Rngs(0))
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert not jnp.isnan(out.logits).any()

    def test_flash_and_non_flash_same_shape(self):
        x = jnp.ones((1, 8), dtype=jnp.int32)
        out_std   = Transformer(self._config(False), rngs=nnx.Rngs(0))(x, use_cache=False, kv_caches=None, cache_index=0)
        out_flash = Transformer(self._config(True),  rngs=nnx.Rngs(0))(x, use_cache=False, kv_caches=None, cache_index=0)
        assert out_std.logits.shape == out_flash.logits.shape


# ── Scheduler factory ─────────────────────────────────────────────────────────

class TestSchedulerFactory:
    def _config(self, schedule: str) -> Config:
        return Config(
            dim=128, n_heads=4, head_size=32, num_blocks=1,
            vocab_size=64, max_context=32, kv_heads=2,
            lr=0.001, warmup_steps=10, lr_schedule=schedule,
            gradient_checkpointing=False, dropout_rate=0.0,
        )

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "constant", "wsd"])
    def test_schedule_returns_callable(self, schedule):
        config = self._config(schedule)
        sched = _build_schedule(config, total_steps=100)
        lr_at_0  = float(sched(0))
        lr_at_50 = float(sched(50))
        assert lr_at_0 >= 0.0
        assert lr_at_50 > 0.0

    def test_cosine_warmup(self):
        config = self._config("cosine")
        sched = _build_schedule(config, total_steps=100)
        assert float(sched(0)) < float(sched(10))   # warming up

    def test_constant_stays_flat(self):
        config = self._config("constant")
        sched = _build_schedule(config, total_steps=100)
        lr_20 = float(sched(20))
        lr_80 = float(sched(80))
        assert abs(lr_20 - lr_80) < 1e-6

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            Config(
                dim=128, n_heads=4, head_size=32, num_blocks=1,
                vocab_size=64, max_context=32, kv_heads=2,
                lr_schedule="oops",
            )


# ── RMSNorm model end-to-end ──────────────────────────────────────────────────

class TestRMSNormModel:
    def test_rmsnorm_model_no_nan(self):
        config = Config(
            dim=128, n_heads=4, head_size=32, num_blocks=2,
            vocab_size=64, max_context=32, kv_heads=2,
            norm_type="rmsnorm", gradient_checkpointing=False, dropout_rate=0.0,
        )
        model = Transformer(config, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert not jnp.isnan(out.logits).any()
        assert not jnp.isinf(out.logits).any()
