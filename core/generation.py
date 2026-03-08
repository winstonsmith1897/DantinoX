from typing import Callable
import jax
import jax.numpy as jnp
from flax import nnx

DecodeFunc = Callable[[jnp.ndarray, jax.Array | None], jnp.ndarray]


def decode(
        probs: jnp.ndarray,
        decoding_func: DecodeFunc,
        key: jax.Array | None
    ) -> tuple[jnp.ndarray, jax.Array]:
    if key is not None:
        key, subkey = jax.random.split(key)
        tok         = decoding_func(probs, subkey)
    else:
        tok         = decoding_func(probs, None)
    
    if tok.ndim == 1:
        tok = tok[:, None]

    return tok, key

@nnx.jit(static_argnames=['decoding_func', 'use_cache'])
def _generate_toks(
    model: nnx.Module,
    x: jnp.ndarray,
    key: jax.Array | None,
    start_pos: int,
    max_generations: int,
    decoding_func: DecodeFunc,
    use_cache: bool = False, 
    ) -> jnp.ndarray:


    def generate_with_kv_cache(i, val):
        x, tok, kv_cache, k  = val
        last_logits, new_kv_cache = model(tok, use_cache, kv_cache, i-1)
        x, k, next_tok_id = _get_tok_id(i, x, k, last_logits)
        return x, next_tok_id, new_kv_cache, k
    
    def prefill_or_no_cache(i, val):
        x, kv_cache, _, k = val
        logits, new_kv_cache = model(x, use_cache, kv_cache, 0)
        x, k, tok = _get_tok_id(i, x, k, logits[:, i-1, :])
        return x, new_kv_cache, tok, k

    def _get_tok_id(i, x, k, last_logits):
        probs  = jax.nn.softmax(last_logits, axis=-1) if k is None else last_logits
        tok, k = decode(probs=probs, decoding_func=decoding_func, key=k)
        x = x.at[:, i].set(tok[:, 0])
        return x,k,tok
    
    init_kv_cache = tuple((None, None) for _ in range(model.num_blocks))
    dummy_tok = jnp.zeros((x.shape[0], 1), dtype=jnp.int32)
    if use_cache is False:
        x, _, _, _   = jax.lax.fori_loop(lower=start_pos, 
                                        upper=start_pos + max_generations, 
                                        body_fun=prefill_or_no_cache, 
                                        init_val=(x, init_kv_cache, dummy_tok, key))
    else:
        x, kv_cache, tok, key = prefill_or_no_cache(start_pos, (x, init_kv_cache, dummy_tok, key))
        x, _, _ , _ = jax.lax.fori_loop(lower=start_pos + 1, upper=start_pos + max_generations, body_fun=generate_with_kv_cache, init_val=(x, tok, kv_cache, key))
    return x


def generate(
        model: nnx.Module, 
        x: jnp.ndarray,
        max_generations: int,
        greedy: bool = False, 
        seed: int = 42, 
        use_cache: bool = True) -> jnp.ndarray:
    
    B, T = x.shape
    to_generate = min(model.max_context, T + max_generations) - T
    
    if to_generate <= 0:
        return x
    x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)
    x_padded = x_padded.at[:, :T].set(x)

    if greedy:
        key = None
        decoding_func = lambda v, key=None: jnp.argmax(v, axis=-1, keepdims=True)
    else:
        key = jax.random.key(seed)
        decoding_func = lambda v, key: jax.random.categorical(key, v, axis=-1)
    x = _generate_toks(model, x_padded, key=key, start_pos=T, max_generations=to_generate, decoding_func=decoding_func, use_cache=use_cache)
    return x[:, :T + to_generate]