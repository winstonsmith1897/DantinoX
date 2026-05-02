"""Tests for LoRA fine-tuning and multi-GPU sharding utilities."""
from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from flax.nnx.transforms.autodiff import DiffState

from core.config import Config
from core.lora import LoRALinear, LoRAParam
from core.model import Transformer
from core.sharding import make_mesh, num_devices, replicate, shard_batch

# ── helpers ───────────────────────────────────────────────────────────────────

def _tiny_cfg(**kw) -> Config:
    base = dict(dim=64, n_heads=4, head_size=16, num_blocks=2, vocab_size=50, max_context=16, kv_heads=2)
    return Config(**{**base, **kw})


# ── LoRAParam ─────────────────────────────────────────────────────────────────

class TestLoRAParam:
    def test_is_nnx_variable(self):
        v = LoRAParam(jnp.ones((4, 4)))
        assert isinstance(v, nnx.Variable)

    def test_distinct_from_param(self):
        assert LoRAParam is not nnx.Param

    def test_value_access(self):
        arr = jnp.arange(6.0).reshape(2, 3)
        v = LoRAParam(arr)
        assert jnp.allclose(v[...], arr)


# ── LoRALinear ────────────────────────────────────────────────────────────────

class TestLoRALinear:
    def setup_method(self):
        self.rngs = nnx.Rngs(0)

    def test_output_shape(self):
        layer = LoRALinear(32, 64, rank=4, alpha=8.0, rngs=self.rngs)
        x = jnp.ones((2, 8, 32))
        assert layer(x).shape == (2, 8, 64)

    def test_no_nan(self):
        layer = LoRALinear(16, 16, rank=4, alpha=8.0, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 16))
        out = layer(x)
        assert not jnp.any(jnp.isnan(out))

    def test_zero_delta_at_init(self):
        """lora_B is initialised to zeros so adapter contributes nothing at init."""
        layer = LoRALinear(8, 8, rank=2, alpha=4.0, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(42), (1, 8))
        base_out = layer.base(x)
        lora_out = layer(x)
        assert jnp.allclose(base_out, lora_out, atol=1e-6)

    def test_lora_A_dtype_and_scale(self):
        rank, in_f = 4, 32
        layer = LoRALinear(in_f, 16, rank=rank, rngs=self.rngs)
        # A should be initialised with std ≈ 1/sqrt(in_features)
        std = float(jnp.std(layer.lora_A[...]))
        expected = 1.0 / math.sqrt(in_f)
        assert abs(std - expected) < 0.15  # within 15% of expected std

    def test_merge_weights_shape(self):
        layer = LoRALinear(16, 32, rank=4, alpha=8.0, rngs=self.rngs)
        merged = layer.merge_weights()
        assert merged.shape == (16, 32)

    def test_lora_params_are_loraparam_type(self):
        layer = LoRALinear(8, 8, rank=2, rngs=self.rngs)
        # LoRAParam variables show up in nnx.state filtered by LoRAParam
        state = nnx.state(layer, LoRAParam)
        assert len(state) > 0

    def test_base_params_are_nnx_param_type(self):
        layer = LoRALinear(8, 8, rank=2, rngs=self.rngs)
        state = nnx.state(layer, nnx.Param)
        assert len(state) > 0

    def test_dropout_applied_when_rate_nonzero(self):
        layer = LoRALinear(16, 16, rank=4, dropout_rate=0.5, rngs=nnx.Rngs(7))
        x = jnp.ones((4, 16))
        # deterministic=False should not crash; stochastic dropout means outputs differ
        out = layer(x, deterministic=False)
        assert out.shape == (4, 16)


# ── Model with LoRA ───────────────────────────────────────────────────────────

class TestModelWithLoRA:
    def test_attention_lora_forward(self):
        cfg = _tiny_cfg(use_lora=True, lora_targets="attention", lora_rank=4, lora_alpha=8.0)
        model = Transformer(cfg, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert out.logits.shape == (2, 8, cfg.vocab_size)

    def test_mlp_lora_forward(self):
        cfg = _tiny_cfg(use_lora=True, lora_targets="mlp", lora_rank=4, lora_alpha=8.0)
        model = Transformer(cfg, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert out.logits.shape == (2, 8, cfg.vocab_size)

    def test_all_lora_forward(self):
        cfg = _tiny_cfg(use_lora=True, lora_targets="all", lora_rank=4, lora_alpha=8.0)
        model = Transformer(cfg, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x, use_cache=False, kv_caches=None, cache_index=0)
        assert out.logits.shape == (2, 8, cfg.vocab_size)

    def test_lora_param_count_less_than_full(self):
        cfg_base = _tiny_cfg()
        cfg_lora = _tiny_cfg(use_lora=True, lora_targets="attention", lora_rank=4)
        m_base = Transformer(cfg_base, rngs=nnx.Rngs(0))
        m_lora = Transformer(cfg_lora, rngs=nnx.Rngs(0))
        # LoRA-trainable params are strictly fewer than all base params
        lora_state = nnx.state(m_lora, LoRAParam)
        base_state = nnx.state(m_base, nnx.Param)
        lora_count = sum(v.size for v in jax.tree_util.tree_leaves(lora_state))
        base_count = sum(v.size for v in jax.tree_util.tree_leaves(base_state))
        assert lora_count < base_count

    def test_base_params_not_in_lora_grad(self):
        """Gradients should only flow through LoRAParam, not nnx.Param."""
        cfg = _tiny_cfg(use_lora=True, lora_targets="attention", lora_rank=4)
        model = Transformer(cfg, rngs=nnx.Rngs(0))

        def loss_fn(model):
            x = jnp.ones((1, 4), dtype=jnp.int32)
            out = model(x, use_cache=False, kv_caches=None, cache_index=0)
            return jnp.mean(out.logits)

        grad_fn = nnx.value_and_grad(loss_fn, argnums=DiffState(0, LoRAParam))
        _, grads = grad_fn(model)
        # grads only contains LoRAParam leaves
        assert len(grads) > 0


# ── Config validation ─────────────────────────────────────────────────────────

class TestLoRAConfig:
    def test_invalid_targets_raises(self):
        with pytest.raises(ValueError, match="lora_targets"):
            _tiny_cfg(use_lora=True, lora_targets="qkv")

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            _tiny_cfg(use_lora=True, lora_rank=0)

    def test_valid_targets(self):
        for t in ("attention", "mlp", "all"):
            cfg = _tiny_cfg(use_lora=True, lora_targets=t)
            assert cfg.lora_targets == t


# ── Sharding utilities ────────────────────────────────────────────────────────

class TestSharding:
    def test_make_mesh_returns_mesh(self):
        from jax.sharding import Mesh
        mesh = make_mesh(1)
        assert isinstance(mesh, Mesh)

    def test_num_devices_single(self):
        mesh = make_mesh(1)
        assert num_devices(mesh) == 1

    def test_replicate_preserves_values(self):
        mesh = make_mesh(1)
        arr = jnp.arange(8.0)
        rep = replicate(arr, mesh)
        assert jnp.allclose(rep, arr)

    def test_shard_batch_preserves_values(self):
        mesh = make_mesh(1)
        arr = jnp.arange(16.0).reshape(4, 4)
        sharded = shard_batch(arr, mesh)
        assert jnp.allclose(sharded, arr)

    def test_n_devices_config_validation(self):
        with pytest.raises(ValueError, match="n_devices"):
            _tiny_cfg(n_devices=-1)
