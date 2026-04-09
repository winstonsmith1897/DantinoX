from core.attention import Attention
import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@dataclass
class Config:
    max_context: int = 128
    head_size: int = 64
    n_heads: int = 4
    dim: int = 256
    kv_heads: int = 2
    no_sink: bool = False
    sliding_window: bool = False
    use_rotary_pos: bool = True
    dropout_rate: float = 0.0
    down_dim_q: int = 32
    down_dim_kv: int = 32
    mla: bool = True
    inference: bool = False
    rope_dim: int = 32

@pytest.fixture
def setup_attention():
    config = Config()
    rngs = nnx.Rngs(0)
    x = jax.random.normal(rngs.dropout(), (2, 10, config.dim))
    return config, rngs, x


def test_attention_training_shape(setup_attention):
    config, rngs, x = setup_attention
    config.inference = False
    
    model = Attention(config, rngs)
    
    out, cache = model(x, use_cache=False, kv_cache=(None, None), cache_index=0)
    
    assert out.shape == x.shape, f"Shape errata! Atteso {x.shape}, ottenuto {out.shape}"
    assert not jnp.isnan(out).any(), "L'output contiene dei NaN!"

def test_attention_inference_mla_caching(setup_attention):
    config, rngs, _ = setup_attention
    config.inference = True
    config.mla = True
    
    model = Attention(config, rngs)
    
    x_single = jax.random.normal(rngs.dropout(), (2, 1, config.dim))
    
    out_1, cache_1 = model(x_single, use_cache=True, kv_cache=(None, None), cache_index=0)
    
    assert out_1.shape == (2, 1, config.dim)
    assert cache_1[0].shape == (2, config.max_context, config.down_dim_kv)
    
    out_2, cache_2 = model(x_single, use_cache=True, kv_cache=cache_1, cache_index=1)
    
    assert out_2.shape == (2, 1, config.dim)
    assert isinstance(cache_2, tuple)

def test_attention_jit_compilation(setup_attention):
    config, rngs, x = setup_attention
    model = Attention(config, rngs)
    
    @nnx.jit
    def forward_fn(m, input_x):
        return m(input_x, use_cache=False, kv_cache=(None, None), cache_index=0)
    
    try:
        out = forward_fn(model, x)
        assert out is not None
    except Exception as e:
        pytest.fail(f"La compilazione JIT è fallita con l'errore: {e}")

def test_standard_attention_fallback(setup_attention):
    config, rngs, x = setup_attention
    config.mla = False
    
    model = Attention(config, rngs)
    out, cache = model(x, use_cache=False, kv_cache=(None, None), cache_index=0)
    
    assert out.shape == x.shape


def test_attention_gqa_broadcasting(setup_attention):
    config, rngs, _ = setup_attention
    
    config.n_heads = 8
    config.kv_heads = 2
    config.head_size = 64
    config.dim = config.n_heads * config.head_size
    
    x_gqa = jax.random.normal(rngs.dropout(), (2, 10, config.dim))
    
    model = Attention(config, rngs)
    out, _ = model(x_gqa, use_cache=False, kv_cache=(None, None), cache_index=0)
    
    assert out.shape == x_gqa.shape

def test_attention_no_sink_gating(setup_attention):
    config, rngs, x = setup_attention
    config.no_sink = True
    
    model = Attention(config, rngs)
    
    config.inference = False
    out_train, _ = model(x, use_cache=False, kv_cache=(None, None), cache_index=0)
    assert out_train.shape == x.shape
    
    config.inference = True
    x_single = jax.random.normal(rngs.dropout(), (2, 1, config.dim))
    out_infer, _ = model(x_single, use_cache=True, kv_cache=(None, None), cache_index=0)
    assert out_infer.shape == x_single.shape


def test_attention_sliding_window(setup_attention):
    config, rngs, x = setup_attention
    config.sliding_window = True
    config.context_window = 4
    
    model = Attention(config, rngs)
    out, _ = model(x, use_cache=False, kv_cache=(None, None), cache_index=0)
    
    assert out.shape == x.shape
    assert not jnp.isnan(out).any()

def test_attention_prefill_and_decode(setup_attention):
    config, rngs, _ = setup_attention
    config.inference = True
    config.mla = True
    model = Attention(config, rngs)
    
    prompt = jax.random.normal(rngs.dropout(), (2, 5, config.dim))
    out_prefill, cache = model(prompt, use_cache=True, kv_cache=(None, None), cache_index=0)
    
    assert out_prefill.shape == (2, 5, config.dim)
    
    token_6 = jax.random.normal(rngs.dropout(), (2, 1, config.dim))
    out_6, cache = model(token_6, use_cache=True, kv_cache=cache, cache_index=5)
    
    assert out_6.shape == (2, 1, config.dim)
    
    token_7 = jax.random.normal(rngs.dropout(), (2, 1, config.dim))
    out_7, cache = model(token_7, use_cache=True, kv_cache=cache, cache_index=6)
    
    assert out_7.shape == (2, 1, config.dim)