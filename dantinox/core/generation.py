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
    logit_normal_schedule,
)

DecodeFunc = Callable[[jnp.ndarray, jax.Array | None], jnp.ndarray]


def _greedy_decode(v, _key=None):
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
) -> jnp.ndarray:
    """Simple MDLM reverse-diffusion generation (no block-wise cache).

    Runs ``num_sampling_steps`` denoising steps over the full sequence.
    For faster inference use ``fast_dllm_generate``.
    ``prefix`` may be ``None`` for unconditional generation; in that case
    ``batch_size`` controls the output batch dimension.
    """
    B   = prefix.shape[0] if prefix is not None else batch_size
    rng = jax.random.key(seed)

    dual_cache: DualCache | None = None
    if prefix is not None and prefix.shape[1] > 0:
        dual_cache = model.compute_prefix_cache(prefix)  # type: ignore[attr-defined]

    x_t       = jnp.full((B, gen_len), mask_token_id, dtype=jnp.int32)
    T         = schedule.alpha_bar.shape[0] - 1
    step_size = max(1, T // max(num_sampling_steps, 1))

    for t_val in range(T, 0, -step_size):
        output = model(x_t, dual_cache=dual_cache, deterministic=True)
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

    output = model(x_t, dual_cache=dual_cache, deterministic=True)
    x_t    = jnp.where(x_t == mask_token_id, jnp.argmax(output.logits, axis=-1), x_t)
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
    step_size  = max(1, T_diff // max(steps_per_block, 1))
    inner_steps = list(range(T_diff, 0, -step_size))

    n_blocks = (gen_len + block_size - 1) // block_size

    for k in range(n_blocks):
        block_start = T_prefix + k * block_size
        block_end   = min(T_prefix + (k + 1) * block_size, T_total)
        actual_bs   = block_end - block_start  # last block may be smaller

        # ── Initialise / refresh dual cache ───────────────────────────────
        if use_dual_cache:
            dual_cache = model.compute_block_dual_cache(  # type: ignore[attr-defined]
                x, block_start, block_end
            )

        # ── Inner denoising loop ───────────────────────────────────────────
        for step_idx, _t_val in enumerate(inner_steps):
            x_block = x[:, block_start:block_end]

            # Periodic suffix refresh (optional, for better accuracy)
            if (
                use_dual_cache
                and refresh_interval is not None
                and step_idx > 0
                and step_idx % refresh_interval == 0
            ):
                dual_cache = model.compute_block_dual_cache(  # type: ignore[attr-defined]
                    x, block_start, block_end
                )

            if use_dual_cache:
                logits = model.decode_block(  # type: ignore[attr-defined]
                    x_block, dual_cache, block_start
                )
            else:
                # PrefixCache fallback: run model on x[block_start:]
                x_from = x[:, block_start:]
                out    = model(x_from, deterministic=True)
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
    out = model(x, deterministic=True)
    x   = jnp.where(x == mask_token_id, jnp.argmax(out.logits, axis=-1), x)

    return x[:, T_prefix:]


# ── ELF generation (continuous flow-matching) ─────────────────────────────────

def elf_generate(
    model,
    gen_len:    int,
    batch_size: int   = 1,
    n_steps:    int   = 64,
    cfg_scale:  float = 1.0,
    gamma:      float = 0.0,
    seed:       int   = 42,
) -> jnp.ndarray:
    """Generate token sequences with ELF continuous diffusion (ELF Algorithm 5/6).

    Denoises from pure Gaussian noise to clean embeddings using an Euler ODE
    sampler (``gamma=0``) or an SDE-inspired stochastic sampler (``gamma>0``),
    then decodes via the shared unembedding head.

    Parameters
    ----------
    model:      Trained ``ELFTransformer``.
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
    rng = jax.random.PRNGKey(seed)
    B, L  = batch_size, gen_len
    E     = model.config.embed_dim

    rng_z, *rng_steps = jax.random.split(rng, n_steps + 1)

    z      = jax.random.normal(rng_z, (B, L, E))
    x_prev = jnp.zeros_like(z)
    # Linear schedule for inference: uniform coverage of [0, 1].
    # The logit-normal schedule used during training concentrates near t≈0.18
    # and is unsuitable for generation (62/65 values would be < 0.5).
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    for i in range(n_steps):
        t_val  = float(ts[i])
        dt_val = float(ts[i + 1] - ts[i])
        w_arr  = jnp.full((B,), cfg_scale)
        is_den = jnp.zeros(B, dtype=bool)

        if gamma > 0.0:
            # SDE: re-inject noise and shift t slightly toward the noisy regime
            alpha  = 1.0 - gamma * dt_val
            z_back = alpha * z + (1.0 - alpha) * jax.random.normal(rng_steps[i], z.shape)
            t_arr  = jnp.full((B,), alpha * t_val)
            x_hat  = model(z_back, x_prev, t_arr, w_arr, is_den).x_pred
        else:
            t_arr = jnp.full((B,), t_val)
            x_hat = model(z, x_prev, t_arr, w_arr, is_den).x_pred

        v      = (x_hat - z) / jnp.clip(1.0 - t_val, 1e-6)
        z      = z + dt_val * v
        x_prev = x_hat

    # Final decode step at t=1: switch to decode mode and return token logits
    out = model(
        z,
        jnp.zeros_like(z),
        jnp.ones(B),
        jnp.full((B,), cfg_scale),
        jnp.ones(B, dtype=bool),
    )
    return jnp.argmax(out.logits, axis=-1).astype(jnp.int32)
