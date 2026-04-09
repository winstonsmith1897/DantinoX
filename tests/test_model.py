from core.model import Transformer
import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@dataclass
class Config:
    dim: int = 128
    n_heads: int = 4
    head_size: int = 32
    num_blocks: int = 2
    vocab_size: int = 500
    max_context: int = 64
    kv_heads: int = 2
    weight_tying: bool = True
    activation: str = "gelu"
    gradient_checkpointing: bool = False
    dropout_rate: float = 0.0
    use_swiglu: bool = True
    use_moe: bool = False
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    alpha_balance: float = 0.1
    use_rotary_pos: bool = True
    trainable_pos: bool = False  
    absolute_pos: bool = False    
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = False
    mla: bool = False
    inference: bool = False
    down_dim_q: int = 32
    down_dim_kv: int = 32
    rope_dim: int = 16

@pytest.fixture
def setup_transformer():
    config = Config()
    rngs = nnx.Rngs(0)
    x = jax.random.randint(rngs.dropout(), (2, 10), 0, config.vocab_size)
    return config, rngs, x

def test_transformer_training_forward(setup_transformer):
    config, rngs, x = setup_transformer
    model = Transformer(config, rngs)
    
    logits, kv_caches, bal_loss = model(x, use_cache=False, kv_caches=None, cache_index=0)
    
    assert logits.shape == (2, 10, config.vocab_size)
    assert len(kv_caches) == config.num_blocks
    assert not jnp.isnan(logits).any()

def test_transformer_inference_caching(setup_transformer):
    config, rngs, _ = setup_transformer
    model = Transformer(config, rngs)
    
    x_single = jnp.array([[42], [105]], dtype=jnp.int32) 
    
    logits_1, cache_1, _ = model(x_single, use_cache=True, kv_caches=None, cache_index=0)
    assert logits_1.shape == (2, 1, config.vocab_size)
    
    logits_2, cache_2, _ = model(x_single, use_cache=True, kv_caches=cache_1, cache_index=1)
    assert logits_2.shape == (2, 1, config.vocab_size)

def test_transformer_moe_loss(setup_transformer):
    config, rngs, x = setup_transformer
    config.use_moe = True
    model = Transformer(config, rngs)
    
    _, _, bal_loss = model(x, use_cache=False, kv_caches=None, cache_index=0)
    
    assert bal_loss is not None
    assert float(bal_loss) >= 0.0

def test_transformer_jit_compilation(setup_transformer):
    config, rngs, x = setup_transformer
    model = Transformer(config, rngs)
    
    @nnx.jit
    def forward_fn(m, input_x):
        return m(input_x, use_cache=False, kv_caches=None, cache_index=0)
    
    try:
        logits, _, _ = forward_fn(model, x)
        assert logits is not None
    except Exception as e:
        pytest.fail(f"JIT fallita: {e}")

def test_transformer_weight_tying(setup_transformer):
    config, rngs, x = setup_transformer
    config.weight_tying = True
    model = Transformer(config, rngs)
    
    assert jnp.array_equal(model.lm_head.kernel, model.wte.embedding.T)