from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from .diffusion import (
    DualCache,
    NoiseSchedule,
    confidence_unmask_factor,
    confidence_unmask_threshold,
)

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
    start_pos: int | jax.Array,
    max_generations: int | jax.Array,
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

    num_blocks: int = model.num_blocks  # type: ignore[attr-defined]
    init_kv_cache = tuple((None, None) for _ in range(num_blocks))
    dummy_tok = jnp.zeros((x.shape[0], 1), dtype=jnp.int32)

    if use_cache is False:
        # kv_cache is never updated in this path, so keep it out of the carry
        # to avoid passing Python None values through jax.lax.fori_loop.
        def prefill_no_cache(i, val):
            _x, _k = val
            _cache = tuple((None, None) for _ in range(num_blocks))
            logits, _, _ = model(_x, False, _cache, 0, deterministic=True)
            _x, _k, _ = _get_tok_id(i, _x, _k, logits[:, i - 1, :])
            return _x, _k

        x, _ = jax.lax.fori_loop(
            lower=start_pos,
            upper=start_pos + max_generations,
            body_fun=prefill_no_cache,
            init_val=(x, key),
        )
    else:
        x, kv_cache, tok, key = prefill_or_no_cache(start_pos,
                                                    (x, init_kv_cache, dummy_tok, key))
        x, _, _, _ = jax.lax.fori_loop(lower=start_pos + 1,
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
    to_generate = min(model.max_context, T + max_generations) - T  # type: ignore[attr-defined]

    if to_generate <= 0:
        return x

    x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)  # type: ignore[attr-defined]
    x_padded = x_padded.at[:, :T].set(x)

    decoding_func: DecodeFunc
    if greedy:
        key = None
        decoding_func = _greedy_decode
    else:
        key = jax.random.key(seed)
        decoding_func = _sampling_decode

    # Pass start_pos and max_generations as JAX arrays (not Python ints) so
    # @nnx.jit treats them as dynamic traced values.  A Python int is static
    # from JAX's perspective, which would trigger a separate compilation for
    # every distinct (start_pos, max_generations) pair — e.g. the warmup call
    # with max_new_tokens=1 would compile a different kernel than the real call
    # with max_new_tokens=200, blowing up the apparent tok/s.
    x = _generate_toks(model,
                       x_padded,
                       key=key,
                       start_pos=jnp.array(T, dtype=jnp.int32),
                       max_generations=jnp.array(to_generate, dtype=jnp.int32),
                       decoding_func=decoding_func,
                       use_cache=use_cache,
                       top_p=top_p,
                       top_k=top_k,
                       temperature=temperature)

    return x[:, :T + to_generate]


# ── Diffusion generation ──────────────────────────────────────────────────────

def diffusion_generate(
    model: nnx.Module,
    prefix: jnp.ndarray,
    gen_len: int,
    schedule: NoiseSchedule,
    mask_token_id: int,
    seed: int = 42,
    num_sampling_steps: int = 50,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Simple MDLM reverse-diffusion generation (no block-wise cache).

    Runs ``num_sampling_steps`` denoising steps over the full sequence.
    For faster inference use ``fast_dllm_generate``.
    """
    B   = prefix.shape[0]
    rng = jax.random.key(seed)

    dual_cache: DualCache | None = None
    if prefix.shape[1] > 0:
        dual_cache = model.compute_prefix_cache(prefix)  # type: ignore[attr-defined]

    x_t       = jnp.full((B, gen_len), mask_token_id, dtype=jnp.int32)
    T         = schedule.alpha_bar.shape[0] - 1
    step_size = max(1, T // max(num_sampling_steps, 1))

    for t_val in range(T, 0, -step_size):
        t      = jnp.full((B,), t_val, dtype=jnp.int32)
        output = model(x_t, t, dual_cache=dual_cache, deterministic=True)   # type: ignore[call-arg]
        logits = output.logits / max(temperature, 1e-6)
        probs  = jax.nn.softmax(logits, axis=-1)

        rng, subkey = jax.random.split(rng)
        flat_probs  = probs.reshape(B * gen_len, -1)
        flat_keys   = jax.random.split(subkey, B * gen_len)
        x0_pred     = jax.vmap(
            lambda p, k: jax.random.categorical(k, jnp.log(p + 1e-10))
        )(flat_probs, flat_keys).reshape(B, gen_len)

        t_prev      = max(t_val - step_size, 0)
        alpha_t     = float(schedule.alpha_bar[t_val])
        alpha_prev  = float(schedule.alpha_bar[t_prev])
        unmask_prob = (alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8) if alpha_t < 1.0 else 0.0

        rng, subkey2 = jax.random.split(rng)
        do_unmask    = jax.random.bernoulli(subkey2, float(jnp.clip(unmask_prob, 0, 1)), x_t.shape)
        x_t          = jnp.where((x_t == mask_token_id) & do_unmask, x0_pred, x_t)

    t_zero  = jnp.zeros((B,), dtype=jnp.int32)
    output  = model(x_t, t_zero, dual_cache=dual_cache, deterministic=True)  # type: ignore[call-arg]
    x_t     = jnp.where(x_t == mask_token_id, jnp.argmax(output.logits, axis=-1), x_t)
    return x_t


def fast_dllm_generate(
    model: nnx.Module,
    prefix: jnp.ndarray,
    gen_len: int,
    schedule: NoiseSchedule,
    mask_token_id: int,
    block_size: int = 32,
    steps_per_block: int = 50,
    confidence_threshold: float = 0.9,
    decoding_strategy: str = "threshold",
    factor: float = 1.5,
    use_dual_cache: bool = True,
    refresh_interval: int | None = None,
    seed: int = 42,
) -> jnp.ndarray:
    """Block-wise masked-diffusion generation with Fast-dLLM DualCache.

    Implements Algorithm 1 from *Fast-dLLM* (Wu et al., arXiv:2505.22618):

    .. code-block:: text

        x ← [prefix ; MASK * gen_len]
        Compute DualCache for x (prefix KV + suffix KV)

        for each block k in [0, K):
            s ← T_prefix + k * block_size
            e ← min(s + block_size, T_prefix + gen_len)

            for each step t in [T, 0):
                logits ← decode_block(x[s:e], t, dual_cache)    # fresh block KV only
                x[s:e] ← confidence_unmask(logits, x[s:e])       # reveal high-conf tokens
                # (suffix KV reused unchanged within this inner loop)

            # After block k: refresh DualCache for block k+1
            dual_cache ← compute_block_dual_cache(x, t=0, s=e, e=e+block_size)

    Why DualCache is faster than PrefixCache
    -----------------------------------------
    PrefixCache re-runs the model on ``x[s:]`` (block + all suffix blocks) every
    inner step — the suffix KV is recomputed each time.  DualCache instead caches
    the suffix KV (which barely changes within a block, see Fig. 3) and only
    processes ``x[s:e]`` (current block) per step, giving ~1.4–2.1× additional
    speedup over PrefixCache.

    Confidence-aware parallel decoding (§3.3)
    ------------------------------------------
    Two strategies are supported:

    ``"threshold"``
        Unmask all masked tokens whose ``max_softmax(logits[i]) >= τ``.
        Always unmask at least one token to guarantee forward progress.
        Simple and effective; use τ = 0.9 as default.

    ``"factor"``
        Find the largest n such that ``(n+1)(1 - c_(n)) < f`` where c_(n) is
        the n-th highest confidence.  Theoretically grounded by Theorem 1 —
        this bound ensures greedy parallel decoding equals greedy sequential
        decoding.  Achieves ~1.4–1.5× higher throughput than threshold at
        minor accuracy cost.

    Args:
        model:               Trained ``DiffusionTransformer``.
        prefix:              Prompt token IDs ``[B, T_prefix]``.  Pass
                             ``jnp.zeros((B, 0), jnp.int32)`` for unconditional
                             generation.
        gen_len:             Number of tokens to generate.
        schedule:            Precomputed ``NoiseSchedule``.
        mask_token_id:       Vocabulary ID of ``[MASK]``.
        block_size:          Tokens decoded per block (B in the paper).
                             Paper default: 32.  Larger = faster but more
                             approximation error; smaller = slower but more
                             accurate (optimal at 32 per Fig. 4).
        steps_per_block:     Denoising steps per block (T in the paper).
        confidence_threshold: τ for the threshold strategy.
        decoding_strategy:   ``"threshold"`` or ``"factor"``.
        factor:              f for the factor strategy.
        use_dual_cache:      If ``False``, falls back to prefix-only caching
                             (``model.__call__`` on the full suffix).
        refresh_interval:    Recompute the suffix cache every r inner steps.
                             ``None`` = refresh only at block boundaries
                             (fastest, slightly less accurate).
                             Smaller values trade speed for accuracy.
        seed:                PRNG seed (used only with sampling).

    Returns:
        Generated token IDs ``[B, gen_len]``.
    """
    B        = prefix.shape[0]
    T_prefix = prefix.shape[1]
    T_total  = T_prefix + gen_len
    T_diff   = int(schedule.alpha_bar.shape[0]) - 1

    # Full sequence: [prefix | MASK * gen_len]
    x = jnp.concatenate([
        prefix,
        jnp.full((B, gen_len), mask_token_id, dtype=jnp.int32),
    ], axis=1)

    # Timestep ladder for the inner loop: T → 0 in steps_per_block steps
    step_size = max(1, T_diff // max(steps_per_block, 1))
    inner_steps = list(range(T_diff, 0, -step_size))

    n_blocks = (gen_len + block_size - 1) // block_size

    for k in range(n_blocks):
        block_start = T_prefix + k * block_size
        block_end   = min(T_prefix + (k + 1) * block_size, T_total)
        actual_bs   = block_end - block_start  # last block may be smaller

        # ── Initialise / refresh dual cache ───────────────────────────────
        if use_dual_cache:
            t_init     = jnp.full((B,), T_diff, dtype=jnp.int32)
            dual_cache = model.compute_block_dual_cache(  # type: ignore[attr-defined]
                x, t_init, block_start, block_end
            )

        # ── Inner denoising loop ───────────────────────────────────────────
        for step_idx, t_val in enumerate(inner_steps):
            t       = jnp.full((B,), t_val, dtype=jnp.int32)
            x_block = x[:, block_start:block_end]

            # Periodic suffix refresh (optional, for better accuracy)
            if (
                use_dual_cache
                and refresh_interval is not None
                and step_idx > 0
                and step_idx % refresh_interval == 0
            ):
                dual_cache = model.compute_block_dual_cache(  # type: ignore[attr-defined]
                    x, t, block_start, block_end
                )

            if use_dual_cache:
                logits = model.decode_block(  # type: ignore[attr-defined]
                    x_block, t, dual_cache, block_start
                )
            else:
                # PrefixCache fallback: run model on x[block_start:]
                x_from = x[:, block_start:]
                out    = model(x_from, t, dual_cache=None, deterministic=True)  # type: ignore[call-arg]
                logits = out.logits[:, :actual_bs, :]

            # ── Confidence-aware unmasking ─────────────────────────────────
            if decoding_strategy == "factor":
                x_block_new = confidence_unmask_factor(
                    logits, x_block, mask_token_id, factor=factor
                )
            else:
                x_block_new = confidence_unmask_threshold(
                    logits, x_block, mask_token_id, threshold=confidence_threshold
                )

            x = x.at[:, block_start:block_end].set(x_block_new)

            # Early exit if all tokens in this block are revealed
            if not (x[:, block_start:block_end] == mask_token_id).any():
                break

    # Final greedy cleanup: fill any positions still masked after all blocks
    t_zero  = jnp.zeros((B,), dtype=jnp.int32)
    out     = model(x, t_zero, dual_cache=None, deterministic=True)          # type: ignore[call-arg]
    x       = jnp.where(x == mask_token_id, jnp.argmax(out.logits, axis=-1), x)

    return x[:, T_prefix:]
