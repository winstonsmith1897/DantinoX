"""Decoding loops for all three generation paradigms.

Autoregressive
    ``generate`` — jit-compiled token loop with greedy / temperature /
    top-k / top-p decoding and optional KV cache.
Discrete masked diffusion
    ``diffusion_generate`` / ``stream_diffusion_generate`` — full-sequence
    reverse diffusion; ``fast_dllm_generate`` / ``stream_fast_dllm_generate``
    — block-wise decoding with the Fast-dLLM DualCache.
Continuous flow-matching
    ``flow_generate`` / ``stream_flow_generate`` — Euler ODE (or SDE with
    ``gamma > 0``) integration plus a dedicated decode step.

Design rules kept by this module:

* every eager function drains its streaming twin, so the two can never
  diverge;
* the per-step model forwards are jit-compiled (module-level ``nnx.jit``
  helpers) — shapes are constant within a call, so compilation happens once;
* sampling operates on logits directly (``jax.random.categorical``), never on
  re-logged softmax probabilities.

Models live in model.py / flow.py; unmasking strategies in diffusion.py.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .diffusion import (
    DualCache,
    NoiseSchedule,
    confidence_unmask_factor,
    confidence_unmask_threshold,
)

# Decode functions operate on (possibly filtered) *logits*, not probabilities:
# jax.random.categorical takes logits directly, so no softmax → log round-trip
# is needed anywhere in the sampling path.
DecodeFunc = Callable[[jnp.ndarray, jax.Array | None], jnp.ndarray]


def _greedy_decode(logits, _key=None):
    return jnp.argmax(logits, axis=-1, keepdims=True)

def _sampling_decode(logits, key):
    return jax.random.categorical(key, logits, axis=-1)

def decode(
        logits: jnp.ndarray,
        decoding_func: DecodeFunc,
        key: jax.Array | None
    ) -> jnp.ndarray:

    tok = decoding_func(logits, key)

    if tok.ndim == 1:
        tok = tok[:, None]

    return tok


# ── JIT-compiled forward helpers ──────────────────────────────────────────────
# The AR path compiles its whole loop (`_generate_toks`); the diffusion / flow
# paths run a Python loop but compile the per-step model forward below, which
# dominates the cost.  Shapes are constant within a generation call, so each
# helper compiles once per (batch, length) configuration and is reused across
# steps and calls.

@nnx.jit
def _forward_logits(model: nnx.Module, x: jnp.ndarray, dual_cache: DualCache | None) -> jnp.ndarray:
    return model(x, dual_cache=dual_cache, deterministic=True).logits  # type: ignore[operator]


@nnx.jit(static_argnames=["block_start", "block_end"])
def _block_dual_cache(
    model: nnx.Module, x_full: jnp.ndarray, block_start: int, block_end: int
) -> DualCache:
    return model.compute_block_dual_cache(x_full, block_start, block_end)  # type: ignore[attr-defined]


@nnx.jit
def _decode_block_logits(
    model: nnx.Module, x_block: jnp.ndarray, dual_cache: DualCache, block_start: jnp.ndarray
) -> jnp.ndarray:
    return model.decode_block(x_block, dual_cache, block_start)  # type: ignore[attr-defined]


@nnx.jit
def _flow_step(
    model: nnx.Module,
    z: jnp.ndarray,
    x_prev: jnp.ndarray,
    t: jnp.ndarray,
    w: jnp.ndarray,
    is_decode: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    out = model(z, x_prev, t, w, is_decode, deterministic=True)  # type: ignore[operator]
    return out.x_pred, jnp.argmax(out.logits, axis=-1).astype(jnp.int32)

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

    def __apply_top_k(logits, decoding_func, key, top_k):
        # Sampling from the top-k logits directly is equivalent to sampling
        # from the renormalised top-k probabilities (categorical normalises).
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)

        new_key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, logits.shape[0])

        def sample_from_top_k(lg, k, i):
            sample = decode(logits=lg, decoding_func=decoding_func, key=k)
            return i[sample]

        toks = jax.vmap(sample_from_top_k)(top_k_logits, batch_keys, top_k_indices)
        return toks, new_key

    def __apply_top_p(logits, decoding_func, key, top_p):
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits  = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs   = jax.nn.softmax(sorted_logits, axis=-1)

        new_key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, logits.shape[0])

        def sample_from_top_p(lg_sorted, p_sorted, k, idx_sorted, top_p_val):
            cumulative_probs = jnp.cumsum(p_sorted, axis=-1)
            mask = (cumulative_probs - p_sorted) < top_p_val
            masked_logits = jnp.where(mask, lg_sorted, -jnp.inf)

            sample_idx = decode(logits=masked_logits, decoding_func=decoding_func, key=k)
            return idx_sorted[sample_idx]

        toks = jax.vmap(sample_from_top_p, in_axes=(0, 0, 0, 0, None))(
            sorted_logits, sorted_probs, batch_keys, sorted_indices, top_p
        )
        return toks, new_key

    def generate_with_kv_cache(i, val):
        x, tok, kv_cache, k = val
        out = model(tok, caches=kv_cache, cache_index=i - 1, deterministic=True)
        x, k, next_tok_id = _get_tok_id(i, x, k, out.logits[:, -1, :])
        return x, next_tok_id, out.kv_caches, k

    def prefill_or_no_cache(i, val):
        x, kv_cache, _, k = val
        out = model(x, caches=kv_cache, cache_index=0, deterministic=True)
        x, k, tok = _get_tok_id(i, x, k, out.logits[:, i - 1, :])
        return x, out.kv_caches, tok, k

    def _get_tok_id(i, x, k, last_logits):
        last_logits = last_logits / temperature

        if k is None:
            tok = decode(logits=last_logits, decoding_func=decoding_func, key=k)
        elif top_k is not None:
            tok, k = __apply_top_k(logits=last_logits, decoding_func=decoding_func, key=k, top_k=top_k)
        elif top_p is not None:
            tok, k = __apply_top_p(logits=last_logits, decoding_func=decoding_func, key=k, top_p=top_p)
        else:
            new_key, subkey = jax.random.split(k)
            batch_keys = jax.random.split(subkey, last_logits.shape[0])

            def sample_base(lg, ky):
                return decode(logits=lg, decoding_func=decoding_func, key=ky)

            tok = jax.vmap(sample_base)(last_logits, batch_keys)
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
            out = model(_x, deterministic=True)
            _x, _k, _ = _get_tok_id(i, _x, _k, out.logits[:, i - 1, :])
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
        if top_k is not None or top_p is not None:
            warnings.warn(
                "greedy=True ignores top_k/top_p; pass greedy=False to use them.",
                UserWarning,
                stacklevel=2,
            )
        key = None
        decoding_func = _greedy_decode
    else:
        key = jax.random.key(seed)
        decoding_func = _sampling_decode

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

def stream_diffusion_generate(
    model: nnx.Module,
    prefix: jnp.ndarray | None,
    gen_len: int,
    schedule: NoiseSchedule,
    mask_token_id: int,
    seed: int = 42,
    num_sampling_steps: int = 50,
    temperature: float = 1.0,
    batch_size: int = 1,
    decoding_strategy: str = "sample",
    confidence_threshold: float = 0.9,
    factor: float = 1.5,
):
    """Generator version of ``diffusion_generate``.

    Yields ``(step, total_steps, x_t)`` after each denoising step so the
    caller can inspect / display the partially-unmasked sequence live.
    The final yield is the fully-decoded result.

    Args:
        decoding_strategy: One of:
            ``"sample"``      — temperature-scaled categorical sampling (default).
            ``"greedy"``      — argmax at each step (deterministic).
            ``"confidence"``  — unmask positions whose softmax confidence ≥ ``confidence_threshold``.
            ``"factor"``      — factor-based parallel unmasking (Fast-dLLM Thm. 1).
        confidence_threshold: τ for the ``"confidence"`` strategy.
        factor:               f for the ``"factor"`` strategy.
    """
    B   = prefix.shape[0] if prefix is not None else batch_size
    rng = jax.random.key(seed)

    dual_cache: DualCache | None = None
    if prefix is not None and prefix.shape[1] > 0:
        dual_cache = model.compute_prefix_cache(prefix)  # type: ignore[attr-defined]

    x_t       = jnp.full((B, gen_len), mask_token_id, dtype=jnp.int32)
    T         = schedule.alpha_bar.shape[0] - 1
    step_size = max(1, T // max(num_sampling_steps, 1))
    t_ladder  = list(range(T, 0, -step_size))
    total     = len(t_ladder) + 1  # denoising steps + final greedy cleanup

    for step, t_val in enumerate(t_ladder):
        logits = _forward_logits(model, x_t, dual_cache)

        if decoding_strategy == "confidence":
            x_t = confidence_unmask_threshold(logits, x_t, mask_token_id,
                                              threshold=confidence_threshold)
        elif decoding_strategy == "factor":
            x_t = confidence_unmask_factor(logits, x_t, mask_token_id, factor=factor)
        else:
            # "sample" or "greedy": schedule-based probabilistic unmasking
            scaled = logits / max(temperature, 1e-6)
            if decoding_strategy == "greedy":
                x0_pred = jnp.argmax(scaled, axis=-1)
            else:  # "sample": categorical draws independently per position
                rng, subkey = jax.random.split(rng)
                x0_pred = jax.random.categorical(subkey, scaled, axis=-1)

            # alpha_bar is a host-side numpy array, so these are free of syncs.
            t_prev      = max(t_val - step_size, 0)
            alpha_t     = float(schedule.alpha_bar[t_val])
            alpha_prev  = float(schedule.alpha_bar[t_prev])
            unmask_prob = (alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8) if alpha_t < 1.0 else 0.0
            unmask_prob = min(max(unmask_prob, 0.0), 1.0)
            rng, subkey2 = jax.random.split(rng)
            do_unmask    = jax.random.bernoulli(subkey2, unmask_prob, x_t.shape)
            x_t          = jnp.where((x_t == mask_token_id) & do_unmask, x0_pred, x_t)

        yield step, total, x_t

    # Final greedy cleanup: fill any positions still masked
    logits = _forward_logits(model, x_t, dual_cache)
    x_t    = jnp.where(x_t == mask_token_id, jnp.argmax(logits, axis=-1), x_t)
    yield total - 1, total, x_t


def diffusion_generate(
    model: nnx.Module,
    prefix: jnp.ndarray | None,
    gen_len: int,
    schedule: NoiseSchedule,
    mask_token_id: int,
    seed: int = 42,
    num_sampling_steps: int = 50,
    temperature: float = 1.0,
    batch_size: int = 1,
    decoding_strategy: str = "sample",
    confidence_threshold: float = 0.9,
    factor: float = 1.5,
) -> jnp.ndarray:
    """Simple MDLM reverse-diffusion generation (no block-wise cache).

    Runs ``num_sampling_steps`` denoising steps over the full sequence.
    For faster inference use ``fast_dllm_generate``.
    ``prefix`` may be ``None`` for unconditional generation; in that case
    ``batch_size`` controls the output batch dimension.
    """
    x_t = None
    for _, _, x_t in stream_diffusion_generate(  # noqa: B007
        model, prefix, gen_len, schedule, mask_token_id,
        seed=seed, num_sampling_steps=num_sampling_steps,
        temperature=temperature, batch_size=batch_size,
        decoding_strategy=decoding_strategy,
        confidence_threshold=confidence_threshold,
        factor=factor,
    ):
        pass
    return x_t  # type: ignore[return-value]


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
    seed: int = 42,  # reserved for future stochastic unmasking variants
) -> jnp.ndarray:
    """Block-wise masked-diffusion generation with Fast-dLLM DualCache.

    Implements Algorithm 1 from *Fast-dLLM* (Wu et al., arXiv:2505.22618).

    Args:
        model:               Trained ``DiffusionTransformer``.
        prefix:              Prompt token IDs ``[B, T_prefix]``.
        gen_len:             Number of tokens to generate.
        schedule:            Precomputed ``NoiseSchedule``.
        mask_token_id:       Vocabulary ID of ``[MASK]``.
        block_size:          Tokens decoded per block (default: 32).
        steps_per_block:     Denoising steps per block.
        confidence_threshold: τ for the threshold strategy.
        decoding_strategy:   ``"threshold"`` or ``"factor"``.
        factor:              f for the factor strategy.
        use_dual_cache:      If ``False``, falls back to prefix-only caching.
        refresh_interval:    Recompute the suffix cache every r inner steps.
        seed:                PRNG seed.

    Returns:
        Generated token IDs ``[B, gen_len]``.
    """
    # Single implementation: drain the streaming generator so the eager and
    # streaming paths can never diverge.
    x_gen = None
    for _, _, x_gen in stream_fast_dllm_generate(  # noqa: B007
        model, prefix, gen_len, schedule, mask_token_id,
        block_size=block_size, steps_per_block=steps_per_block,
        confidence_threshold=confidence_threshold,
        decoding_strategy=decoding_strategy, factor=factor,
        use_dual_cache=use_dual_cache, refresh_interval=refresh_interval,
        seed=seed,
    ):
        pass
    return x_gen  # type: ignore[return-value]


def stream_fast_dllm_generate(
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
):
    """Streaming version of ``fast_dllm_generate``.

    Yields ``(step, total, x_gen)`` after every inner denoising step so the
    caller can display the partially-revealed sequence live.  ``x_gen`` is
    the generated portion only (no prefix), shape ``[B, gen_len]``.

    ``total`` is ``n_blocks * steps_per_block + 1`` (upper bound; blocks may
    exit early).  ``step`` counts globally across all blocks and inner steps.
    """
    B        = prefix.shape[0]
    T_prefix = prefix.shape[1]
    T_total  = T_prefix + gen_len
    T_diff   = int(schedule.alpha_bar.shape[0]) - 1

    x = jnp.concatenate([
        prefix,
        jnp.full((B, gen_len), mask_token_id, dtype=jnp.int32),
    ], axis=1)

    step_size   = max(1, T_diff // max(steps_per_block, 1))
    inner_steps = list(range(T_diff, 0, -step_size))
    n_blocks    = (gen_len + block_size - 1) // block_size
    total       = n_blocks * len(inner_steps) + 1  # +1 for final cleanup
    global_step = 0

    for k in range(n_blocks):
        block_start = T_prefix + k * block_size
        block_end   = min(T_prefix + (k + 1) * block_size, T_total)
        actual_bs   = block_end - block_start

        if use_dual_cache:
            dual_cache = _block_dual_cache(model, x, block_start, block_end)

        for step_idx, _t_val in enumerate(inner_steps):
            x_block = x[:, block_start:block_end]

            if (
                use_dual_cache
                and refresh_interval is not None
                and step_idx > 0
                and step_idx % refresh_interval == 0
            ):
                dual_cache = _block_dual_cache(model, x, block_start, block_end)

            if use_dual_cache:
                logits = _decode_block_logits(
                    model, x_block, dual_cache, jnp.asarray(block_start, jnp.int32)
                )
            else:
                # PrefixCache fallback: run the model on x[block_start:]
                logits = _forward_logits(model, x[:, block_start:], None)[:, :actual_bs, :]

            if decoding_strategy == "factor":
                x_block_new = confidence_unmask_factor(
                    logits, x_block, mask_token_id, factor=factor
                )
            else:
                x_block_new = confidence_unmask_threshold(
                    logits, x_block, mask_token_id, threshold=confidence_threshold
                )

            x = x.at[:, block_start:block_end].set(x_block_new)
            yield global_step, total, x[:, T_prefix:]
            global_step += 1

            if not (x[:, block_start:block_end] == mask_token_id).any():
                break

        # Per-block greedy cleanup: fill any positions still masked in this
        # block before moving to the next one, so blocks reveal sequentially.
        if (x[:, block_start:block_end] == mask_token_id).any():
            if use_dual_cache:
                logits = _decode_block_logits(
                    model, x[:, block_start:block_end], dual_cache,
                    jnp.asarray(block_start, jnp.int32),
                )
            else:
                logits = _forward_logits(model, x[:, block_start:], None)[:, :actual_bs, :]
            # Exclude mask_token_id from argmax — the model should never
            # predict the mask token itself as output.
            safe_logits = logits.at[:, :, mask_token_id].set(-jnp.inf)
            x_filled = jnp.where(
                x[:, block_start:block_end] == mask_token_id,
                jnp.argmax(safe_logits, axis=-1),
                x[:, block_start:block_end],
            )
            x = x.at[:, block_start:block_end].set(x_filled)
            yield global_step, total, x[:, T_prefix:]
            global_step += 1

    # Final greedy cleanup for any residual masks across all blocks.
    # Also excludes mask_token_id from argmax.
    logits      = _forward_logits(model, x, None)
    safe_logits = logits.at[:, :, mask_token_id].set(-jnp.inf)
    x           = jnp.where(x == mask_token_id, jnp.argmax(safe_logits, axis=-1), x)
    yield global_step, total, x[:, T_prefix:]


# ── Continuous flow-matching generation ───────────────────────────────────────

def stream_flow_generate(
    model,
    gen_len:    int,
    batch_size: int   = 1,
    n_steps:    int   = 64,
    cfg_scale:  float = 1.0,
    gamma:      float = 0.0,
    seed:       int   = 42,
):
    """Generator version of ``flow_generate``.

    Yields ``(step, total_steps, tokens)`` after each Euler ODE step, where
    *tokens* is the argmax-decoded prediction from the current flow state.
    The final yield is the result of the dedicated t=1 decode step.
    """
    rng = jax.random.PRNGKey(seed)
    B, L  = batch_size, gen_len
    E     = model.config.embed_dim

    rng_z, *rng_steps = jax.random.split(rng, n_steps + 1)

    z      = jax.random.normal(rng_z, (B, L, E))
    x_prev = jnp.zeros_like(z)
    # Host-side schedule: avoids a device sync per step when reading t/dt.
    ts     = np.linspace(0.0, 1.0, n_steps + 1)
    total  = n_steps + 1  # ODE steps + final decode
    w_arr  = jnp.full((B,), cfg_scale)
    is_den = jnp.zeros(B, dtype=bool)

    for i in range(n_steps):
        t_val  = float(ts[i])
        dt_val = float(ts[i + 1] - ts[i])

        if gamma > 0.0:
            alpha  = 1.0 - gamma * dt_val
            z_back = alpha * z + (1.0 - alpha) * jax.random.normal(rng_steps[i], z.shape)
            t_arr  = jnp.full((B,), alpha * t_val)
            x_hat, cur_tokens = _flow_step(model, z_back, x_prev, t_arr, w_arr, is_den)
        else:
            t_arr = jnp.full((B,), t_val)
            x_hat, cur_tokens = _flow_step(model, z, x_prev, t_arr, w_arr, is_den)

        v      = (x_hat - z) / max(1.0 - t_val, 1e-6)
        z      = z + dt_val * v
        x_prev = x_hat

        yield i, total, cur_tokens

    _, tokens = _flow_step(
        model,
        z,
        jnp.zeros_like(z),
        jnp.ones(B),
        w_arr,
        jnp.ones(B, dtype=bool),
    )
    yield n_steps, total, tokens


def flow_generate(
    model,
    gen_len:    int,
    batch_size: int   = 1,
    n_steps:    int   = 64,
    cfg_scale:  float = 1.0,
    gamma:      float = 0.0,
    seed:       int   = 42,
) -> jnp.ndarray:
    """Generate with continuous flow-matching (ELF recipe, Algorithms 5/6).

    Denoises from pure Gaussian noise to clean embeddings using an Euler ODE
    sampler (``gamma=0``) or an SDE-inspired stochastic sampler (``gamma>0``),
    then decodes via the shared unembedding head.

    Parameters
    ----------
    model:      Trained ``FlowMatchingTransformer``.
    gen_len:    Number of tokens to generate.
    batch_size: Number of sequences to generate in parallel.
    n_steps:    Denoising steps before the final decode step.
    cfg_scale:  CFG guidance scale w (≥ 1.0).
    gamma:      SDE noise re-injection scale; ``0.0`` = deterministic ODE.
    seed:       PRNG seed.

    Returns
    -------
    Token IDs ``[batch_size, gen_len]`` int32.
    """
    tokens = None
    for _, _, tokens in stream_flow_generate(  # noqa: B007
        model, gen_len=gen_len, batch_size=batch_size,
        n_steps=n_steps, cfg_scale=cfg_scale, gamma=gamma, seed=seed,
    ):
        pass
    return tokens  # type: ignore[return-value]


# Deprecated ELF-branded aliases (removed in v1.0).
elf_generate        = flow_generate
stream_elf_generate = stream_flow_generate
