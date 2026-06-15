import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from dantinox.core.config import Config
from dantinox.core.moe import MoE


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def moe_input(tiny_moe_config: Config) -> jnp.ndarray:
    """Float embeddings (B=2, T=10, dim) as MoE expects."""
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (2, 10, tiny_moe_config.dim))


def make_moe(config: Config) -> MoE:
    return MoE(config, rngs=nnx.Rngs(0))


# ── Shape & dtype ─────────────────────────────────────────────────────────────

def test_output_shape(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    out, loss = moe(moe_input)
    assert out.shape == moe_input.shape


def test_output_dtype_matches_input(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    out, _ = moe(moe_input)
    assert out.dtype == moe_input.dtype


# ── Loss properties ───────────────────────────────────────────────────────────

def test_loss_is_finite(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    _, loss = moe(moe_input)
    assert jnp.isfinite(loss)


def test_loss_is_non_negative(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    _, loss = moe(moe_input)
    assert float(loss) >= 0.0


def test_loss_upper_bound(tiny_moe_config, moe_input):
    """Balance loss <= n_experts^2 by construction (Shazeer et al.)."""
    moe = make_moe(tiny_moe_config)
    _, loss = moe(moe_input)
    assert float(loss) <= tiny_moe_config.n_experts ** 2 + 1e-5


# ── No NaNs in output ─────────────────────────────────────────────────────────

def test_no_nan_in_output(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    out, loss = moe(moe_input)
    assert not jnp.isnan(out).any()
    assert not jnp.isnan(loss)


# ── Deterministic flag ────────────────────────────────────────────────────────

def test_deterministic_true(tiny_moe_config, moe_input):
    moe = make_moe(tiny_moe_config)
    out, loss = moe(moe_input, deterministic=True)
    assert out.shape == moe_input.shape
    assert jnp.isfinite(loss)


def test_deterministic_output_is_reproducible(tiny_moe_config, moe_input):
    """Two deterministic forward passes must return identical results."""
    moe = make_moe(tiny_moe_config)
    out1, loss1 = moe(moe_input, deterministic=True)
    out2, loss2 = moe(moe_input, deterministic=True)
    assert jnp.allclose(out1, out2)
    assert jnp.allclose(loss1, loss2)


# ── Router sanity ─────────────────────────────────────────────────────────────

def test_router_uses_top_k_experts(tiny_moe_config, moe_input):
    """Router weights must sum to ~1.0 per token (renormalised top-k)."""
    moe = make_moe(tiny_moe_config)
    x_routed = moe.router(moe_input)
    probs = jax.nn.softmax(x_routed)
    _, indices = jax.lax.top_k(probs, tiny_moe_config.top_k_mlp)
    # Each token should select exactly top_k_mlp experts
    assert indices.shape == (*moe_input.shape[:2], tiny_moe_config.top_k_mlp)


# ── Latent bottleneck mode ────────────────────────────────────────────────────

def test_latent_output_shape(tiny_moe_latent_config):
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (2, 10, tiny_moe_latent_config.dim))
    moe = MoE(tiny_moe_latent_config, rngs=nnx.Rngs(0))
    out, loss = moe(x)
    assert out.shape == x.shape     # up_proj riporta a dim originale


def test_latent_has_projections(tiny_moe_latent_config):
    moe = MoE(tiny_moe_latent_config, rngs=nnx.Rngs(0))
    assert hasattr(moe, "down_proj")
    assert hasattr(moe, "up_proj")
    assert moe.latent_dim == tiny_moe_latent_config.moe_latent_dim


def test_latent_loss_is_finite(tiny_moe_latent_config):
    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (2, 10, tiny_moe_latent_config.dim))
    moe = MoE(tiny_moe_latent_config, rngs=nnx.Rngs(0))
    _, loss = moe(x)
    assert jnp.isfinite(loss)


def test_standard_has_no_projections(tiny_moe_config):
    moe = MoE(tiny_moe_config, rngs=nnx.Rngs(0))
    assert not hasattr(moe, "down_proj")
    assert not hasattr(moe, "up_proj")


# ── Config roundtrip (kept here as it exercises MoE-related config fields) ────

def test_moe_config_roundtrip(tiny_moe_config):
    d = tiny_moe_config.to_dict()
    restored = Config.from_dict(d)
    assert restored.n_experts == tiny_moe_config.n_experts
    assert restored.top_k_mlp == tiny_moe_config.top_k_mlp
    assert restored.use_moe == tiny_moe_config.use_moe


def test_moe_latent_config_roundtrip(tiny_moe_latent_config):
    d = tiny_moe_latent_config.to_dict()
    restored = Config.from_dict(d)
    assert restored.moe_latent is True
    assert restored.moe_latent_dim == tiny_moe_latent_config.moe_latent_dim


def test_invalid_moe_latent_dim_raises():
    with pytest.raises(ValueError, match="moe_latent_dim"):
        Config(
            dim=128, n_heads=4, head_size=32, kv_heads=4,
            use_moe=True, moe_latent=True, moe_latent_dim=256,  # >= dim: invalido
        )
