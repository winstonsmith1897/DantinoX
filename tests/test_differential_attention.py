"""Tests for differential attention in MHA and GQA variants."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from dantinox.core.attention import build_attention
from dantinox.core.config import Config

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(differential: bool = True, kv_heads: int = 4, **overrides) -> Config:
    base = dict(
        dim=128, n_heads=4, head_size=32, kv_heads=kv_heads,
        num_blocks=2, vocab_size=256, max_context=64,
        gradient_checkpointing=False, dropout_rate=0.0,
        no_sink=False, use_rotary_pos=True,
        differential=differential,
    )
    base.update(overrides)
    return Config(**base)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mha_diff_cfg() -> Config:
    return _cfg(kv_heads=4)


@pytest.fixture(scope="module")
def gqa_diff_cfg() -> Config:
    return _cfg(kv_heads=2)


@pytest.fixture(scope="module")
def mha_std_cfg() -> Config:
    return _cfg(differential=False, kv_heads=4)


@pytest.fixture
def hidden() -> jnp.ndarray:
    return jax.random.normal(jax.random.PRNGKey(7), (2, 10, 128))


# ── Initialization ────────────────────────────────────────────────────────────

def test_mha_diff_init(mha_diff_cfg):
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    assert hasattr(model, "q2k2"), "q2k2 projection missing"
    assert hasattr(model, "lambda_q1"), "lambda_q1 missing"
    assert hasattr(model, "lambda_k1"), "lambda_k1 missing"
    assert hasattr(model, "lambda_q2"), "lambda_q2 missing"
    assert hasattr(model, "lambda_k2"), "lambda_k2 missing"
    assert hasattr(model, "diff_norm"), "diff_norm missing"


def test_gqa_diff_init(gqa_diff_cfg):
    model = build_attention(gqa_diff_cfg, nnx.Rngs(0))
    assert hasattr(model, "q2k2")
    assert hasattr(model, "lambda_q1")
    assert hasattr(model, "diff_norm")


def test_std_has_no_diff_params(mha_std_cfg):
    model = build_attention(mha_std_cfg, nnx.Rngs(0))
    assert not hasattr(model, "q2k2")
    assert not hasattr(model, "lambda_q1")


def test_lambda_param_shapes(mha_diff_cfg):
    """Each λ parameter must be [n_heads, head_size]."""
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    expected = (mha_diff_cfg.n_heads, mha_diff_cfg.head_size)
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
        p = getattr(model, name)
        assert p.get_value().shape == expected, f"{name} shape {p.get_value().shape} != {expected}"


def test_lambda_init_at_zero_init(mha_diff_cfg):
    """At zero init, λ = exp(0)−exp(0)+λ_init = λ_init for every head."""
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    lam = model._compute_lambda()
    expected = mha_diff_cfg.lambda_init
    assert jnp.allclose(lam, expected, atol=1e-6), f"λ at init should be {expected}, got {lam}"


# ── Output shape and NaN ──────────────────────────────────────────────────────

def test_mha_output_shape(mha_diff_cfg, hidden):
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    out, _ = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)
    assert out.shape == hidden.shape


def test_gqa_output_shape(gqa_diff_cfg, hidden):
    model = build_attention(gqa_diff_cfg, nnx.Rngs(0))
    out, _ = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)
    assert out.shape == hidden.shape


def test_no_nan_mha(mha_diff_cfg, hidden):
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    out, _ = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)
    assert not jnp.isnan(out).any(), "NaN in MHA differential output"


def test_no_nan_gqa(gqa_diff_cfg, hidden):
    model = build_attention(gqa_diff_cfg, nnx.Rngs(0))
    out, _ = model(hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)
    assert not jnp.isnan(out).any(), "NaN in GQA differential output"


# ── Differential effect ───────────────────────────────────────────────────────

def test_differential_changes_output(mha_diff_cfg, mha_std_cfg, hidden):
    """Activating differential attention must change the output."""
    out_diff, _ = build_attention(mha_diff_cfg, nnx.Rngs(42))(
        hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True
    )
    out_std, _ = build_attention(mha_std_cfg, nnx.Rngs(42))(
        hidden, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True
    )
    assert not jnp.allclose(out_diff, out_std, atol=1e-4), (
        "Differential and standard outputs are identical — second Q/K stream has no effect"
    )


# ── Flash vs general path consistency ────────────────────────────────────────

def test_flash_general_path_consistency(mha_diff_cfg):
    """Flash and general paths must agree up to float32 rounding."""
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 8, 128))

    model_gen   = build_attention(mha_diff_cfg, nnx.Rngs(0))
    model_flash = build_attention(mha_diff_cfg, nnx.Rngs(0))
    model_flash.use_flash = True

    out_gen,   _ = model_gen(x,   use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)
    out_flash, _ = model_flash(x, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)

    max_diff = float(jnp.abs(out_gen - out_flash).max())
    assert jnp.allclose(out_gen, out_flash, atol=1e-4), f"Flash/general mismatch: max diff = {max_diff:.2e}"


# ── KV cache ─────────────────────────────────────────────────────────────────

def test_kv_cache_initial_shapes(mha_diff_cfg):
    """First cache step must populate k, v, and k2 caches."""
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 1, 128))

    out, cache = model(x, use_cache=True, kv_cache=(None, None), cache_index=0, deterministic=True)

    assert out.shape == (2, 1, 128)
    assert cache[0] is not None, "k cache not populated"
    assert cache[1] is not None, "v cache not populated"
    assert cache[2] is not None, "k2 cache not populated (differential)"

    B, H, G, T, D = cache[0].shape
    assert T == mha_diff_cfg.max_context
    assert D == mha_diff_cfg.head_size


def test_kv_cache_accumulation(mha_diff_cfg):
    """Cache must accumulate: different token input at step 2 gives different output from step 1."""
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))
    x1 = jax.random.normal(jax.random.PRNGKey(5), (2, 1, 128))
    x2 = jax.random.normal(jax.random.PRNGKey(6), (2, 1, 128))

    out1, cache1 = model(x1, use_cache=True, kv_cache=(None, None), cache_index=0, deterministic=True)
    out2, _      = model(x2, use_cache=True, kv_cache=cache1,        cache_index=1, deterministic=True)

    assert out2.shape == (2, 1, 128)
    assert not jnp.isnan(out2).any()
    assert not jnp.allclose(out1, out2), "Outputs for different tokens should differ"


# ── JIT ───────────────────────────────────────────────────────────────────────

def test_jit_compilation(mha_diff_cfg, hidden):
    model = build_attention(mha_diff_cfg, nnx.Rngs(0))

    @nnx.jit
    def forward(m, x):
        return m(x, use_cache=False, kv_cache=(None, None), cache_index=0, deterministic=True)

    out, _ = forward(model, hidden)
    assert out.shape == hidden.shape
    assert not jnp.isnan(out).any()
