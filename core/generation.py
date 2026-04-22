from typing import Callable
import jax
import jax.numpy as jnp
from flax import nnx

DecodeFunc = Callable[[jnp.ndarray, jax.Array | None], jnp.ndarray]


def _greedy_decode(v, key=None):
    return jnp.argmax(v, axis=-1, keepdims=True)

def _sampling_decode(v, key):
    return jax.random.categorical(key, jnp.log(v + 1e-10), axis=-1)

def decode(
        probs: jnp.ndarray,
        decoding_func: DecodeFunc,
        key: jax.Array | None
    ) -> jnp.ndarray:
    
    tok = decoding_func(probs, key)
    
    if tok.ndim == 1:
        tok = tok[:, None]

    return tok

@nnx.jit(static_argnames=['decoding_func', 'use_cache', 'top_k', 'top_p', 'temperature'])
def _generate_toks(
    model: nnx.Module,
    x: jnp.ndarray,
    key: jax.Array | None,
    start_pos: int,
    max_generations: int,
    decoding_func: DecodeFunc,
    use_cache: bool = False,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0
    ) -> jnp.ndarray:

    def __apply_top_k(probs, decoding_func, key, top_k):
        top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k, axis=-1)
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)  
        
        new_key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, probs.shape[0])
        
        def sample_from_top_k(p, k, i):
            sample = decode(probs=p, decoding_func=decoding_func, key=k)
            return i[sample]
            
        toks = jax.vmap(sample_from_top_k)(top_k_probs, batch_keys, top_k_indices)
        return toks, new_key
    
    def __apply_top_p(probs, decoding_func, key, top_p):
        sorted_indices = jnp.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
        
        new_key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, probs.shape[0])
        
        def sample_from_top_p(p_sorted, k, idx_sorted, top_p_val):
            cumulative_probs = jnp.cumsum(p_sorted, axis=-1) 
            mask = (cumulative_probs - p_sorted) < top_p_val   
            masked_probs = jnp.where(mask, p_sorted, 0.0)
            masked_probs = masked_probs / jnp.sum(masked_probs)
            
            sample_idx = decode(probs=masked_probs, decoding_func=decoding_func, key=k)
            return idx_sorted[sample_idx]
            
        toks = jax.vmap(sample_from_top_p, in_axes=(0, 0, 0, None))(
            sorted_probs, batch_keys, sorted_indices, top_p
        )
        return toks, new_key
    
    def generate_with_kv_cache(i, val):
        x, tok, kv_cache, k  = val
        last_logits, new_kv_cache, _ = model(tok, use_cache, kv_cache, i-1, deterministic=True)
        x, k, next_tok_id = _get_tok_id(i, x, k, last_logits[:, -1, :])
        return x, next_tok_id, new_kv_cache, k
    
    def prefill_or_no_cache(i, val):
        x, kv_cache, _, k = val
        logits, new_kv_cache, _ = model(x, use_cache, kv_cache, 0, deterministic=True)
        x, k, tok = _get_tok_id(i, x, k, logits[:, i-1, :])
        return x, new_kv_cache, tok, k

    def _get_tok_id(i, x, k, last_logits):
        last_logits = last_logits / temperature
        probs = jax.nn.softmax(last_logits, axis=-1)
        
        if k is None:
            tok = decode(probs=probs, decoding_func=decoding_func, key=k)
        elif top_k is not None:
            tok, k = __apply_top_k(probs=probs, decoding_func=decoding_func, key=k, top_k=top_k)
        elif top_p is not None:
            tok, k = __apply_top_p(probs=probs, decoding_func=decoding_func, key=k, top_p=top_p)
        else:
            new_key, subkey = jax.random.split(k)
            batch_keys = jax.random.split(subkey, probs.shape[0])
            
            def sample_base(p, ky):
                return decode(probs=p, decoding_func=decoding_func, key=ky)
                
            tok = jax.vmap(sample_base)(probs, batch_keys)
            k = new_key
        tok = tok.reshape(-1, 1)
        x = x.at[:, i].set(tok[:, 0])
        return x, k, tok
    
    init_kv_cache = tuple((None, None) for _ in range(model.num_blocks))
    dummy_tok = jnp.zeros((x.shape[0], 1), dtype=jnp.int32)
    
    if use_cache is False:
        x, _, _, _   = jax.lax.fori_loop(lower=start_pos, 
                                        upper=start_pos + max_generations, 
                                        body_fun=prefill_or_no_cache, 
                                        init_val=(x, init_kv_cache, dummy_tok, key))
    else:
        x, kv_cache, tok, key = prefill_or_no_cache(start_pos, 
                                                    (x, init_kv_cache, dummy_tok, key))
        x, _, _ , _ = jax.lax.fori_loop(lower=start_pos + 1, 
                                        upper=start_pos + max_generations, 
                                        body_fun=generate_with_kv_cache, 
                                        init_val=(x, tok, kv_cache, key))
    return x


def generate(
        model: nnx.Module, 
        x: jnp.ndarray,
        max_generations: int,
        greedy: bool = False, 
        seed: int = 42, 
        use_cache: bool = True, 
        top_p: float | None = None,
        top_k: int | None = None,
        temperature: float = 1.0) -> jnp.ndarray:
    
    B, T = x.shape
    to_generate = min(model.max_context, T + max_generations) - T
    
    if to_generate <= 0:
        return x
        
    x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)
    x_padded = x_padded.at[:, :T].set(x)

    if greedy:
        key = None
        decoding_func = _greedy_decode
    else:
        key = jax.random.key(seed)
        decoding_func = _sampling_decode
        
    x = _generate_toks(model, 
                       x_padded, 
                       key=key, 
                       start_pos=T, 
                       max_generations=to_generate, 
                       decoding_func=decoding_func, 
                       use_cache=use_cache,
                       top_p=top_p,
                       top_k=top_k,
                       temperature=temperature)
                       
    return x[:, :T + to_generate]