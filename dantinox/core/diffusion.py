"""Masked Discrete Diffusion for DantinoX — LLaDA-style.

Implements masked-diffusion following LLaDA (arXiv:2502.09992):

Forward process
---------------
Each token is independently masked with continuous probability t ∈ (0, 1]:

    x_t[i] = mask_token_id    with probability  p_mask(t)
    x_t[i] = x_0[i]           with probability  1 − p_mask(t)

where p_mask depends on the noise schedule (linear: p_mask=t, etc.).

Loss (LLaDA Eq. 3)
------------------
L = -E_{t~U[0,1], x_t} [ (1/t) * Σ_i 1[x_t^i=M] log p_θ(x_0^i | x_t) ]

The 1/t weight ensures each noise level contributes equally in expectation
and is the correct VLB weight for the linear masking schedule.

Time-free parameterization (LLaDA Eq. 11)
------------------------------------------
The optimal predictor p_θ(x_0 | x_t) depends only on the unmasked tokens,
not on t.  The model therefore receives NO time-step input — standard
bidirectional transformer, no AdaLayerNorm.

DualCache (Fast-dLLM §3.2)
---------------------------
Block-wise inference cache for efficient generation.  The output sequence is
divided into K blocks.  For block k, prefix and suffix KV are cached and
only the current block is recomputed each step.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config

# ── Dual-cache container (Fast-dLLM DualCache) ────────────────────────────────

class DualCache(NamedTuple):
    """Bidirectional KV cache for block-wise diffusion inference (Fast-dLLM).

    Attributes
    ----------
    prefix_kvs:
        Per-layer ``(k, v)`` for the prompt tokens, sliced from the full
        forward pass.  Reused unchanged for all steps of every block.
        Entries are ``None`` for attention variants (MLA) that do not support
        KV injection.
    suffix_kvs:
        Per-layer ``(k, v)`` for the remaining all-MASK blocks **after** the
        current block being decoded.  These are approximately constant within
        a block's inner loop.  Refreshed at each block boundary.
        ``None`` when using prefix-only caching, or before the first block.
    """
    prefix_kvs: tuple
    suffix_kvs: tuple | None = None


# ── Noise schedule ─────────────────────────────────────────────────────────────

class NoiseSchedule(NamedTuple):
    """Discrete masking schedule for reverse-diffusion generation.

    Attributes
    ----------
    schedule:
        Schedule name: ``"linear"`` | ``"cosine"`` | ``"sqrt"``.
    alpha_bar:
        Survival probability array, shape ``[T+1]``.
        ``alpha_bar[t]`` is the fraction of tokens expected to remain
        unmasked at step ``t``.  ``alpha_bar[0] = 1.0`` (clean),
        ``alpha_bar[T] ≈ 0.0`` (fully masked).
        Required by ``diffusion_generate`` / ``fast_dllm_generate``.
    """
    schedule: str
    alpha_bar: np.ndarray | jnp.ndarray | None = None  # shape [T+1]


def make_noise_schedule(config_or_name: Config | str, n_steps: int | None = None) -> NoiseSchedule:
    """Build a :class:`NoiseSchedule` with a precomputed ``alpha_bar`` array.

    Args:
        config_or_name: A :class:`~core.config.Config` (reads ``noise_schedule``
            and ``diffusion_steps``), or a schedule name string directly.
        n_steps: Override the number of discrete steps.  Defaults to
            ``config.diffusion_steps`` when a Config is supplied, else 1000.

    Returns:
        :class:`NoiseSchedule` with ``alpha_bar`` of shape ``[n_steps + 1]``.
    """
    if isinstance(config_or_name, str):
        sched = config_or_name
        T     = n_steps or 1000
    else:
        sched = config_or_name.noise_schedule
        T     = n_steps or int(getattr(config_or_name, "diffusion_steps", 1000))

    ts = np.linspace(0.0, 1.0, T + 1, dtype=np.float32)

    if sched == "cosine":
        s      = 0.008
        alpha  = np.cos(((ts + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
        alpha0 = np.cos((s / (1.0 + s)) * (np.pi / 2.0)) ** 2
        p_mask = np.clip(1.0 - alpha / alpha0, 0.0, 1.0)
    elif sched == "sqrt":
        p_mask = np.sqrt(np.clip(ts, 0.0, 1.0))
    else:  # "linear" (default)
        p_mask = ts

    alpha_bar = np.clip(1.0 - p_mask, 0.0, 1.0).astype(np.float32)
    return NoiseSchedule(schedule=sched, alpha_bar=alpha_bar)


# ── Forward process ────────────────────────────────────────────────────────────

def corrupt(
    x0: jnp.ndarray,
    t: jnp.ndarray,
    rng: jax.Array,
    noise_schedule: str | NoiseSchedule,
    mask_token_id: int,
) -> jnp.ndarray:
    """LLaDA-style forward process: mask each token with probability p_mask(t).

    Args:
        x0:             Clean token IDs, shape ``[B, L]``.
        t:              Per-sample noise level, shape ``[B]``, values in (0, 1].
        rng:            JAX PRNG key.
        noise_schedule: Schedule name (str) or ``NoiseSchedule`` namedtuple.
                        "linear" → p_mask = t (LLaDA default).
                        "cosine" → p_mask = 1 − cos²(πt/2 · scale).
                        "sqrt"   → p_mask = √t.
        mask_token_id:  Vocabulary ID of ``[MASK]``.

    Returns:
        Noisy token sequence ``x_t``, shape ``[B, L]``.
    """
    sched = noise_schedule.schedule if isinstance(noise_schedule, NoiseSchedule) else noise_schedule

    if sched == "linear":
        p_mask = t
    elif sched == "cosine":
        s = 0.008
        alpha = jnp.cos(((t + s) / (1.0 + s)) * (jnp.pi / 2.0)) ** 2
        alpha0 = jnp.cos((s / (1.0 + s)) * (jnp.pi / 2.0)) ** 2
        p_mask = 1.0 - alpha / alpha0
    else:  # "sqrt"
        p_mask = jnp.sqrt(t + 1e-8)

    p_mask = jnp.clip(p_mask, 0.0, 1.0)[:, None]   # [B, 1]
    mask   = jax.random.bernoulli(rng, p_mask, x0.shape)
    return jnp.where(mask, mask_token_id, x0)


# ── Loss ───────────────────────────────────────────────────────────────────────

def masked_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    x_t: jnp.ndarray,
    mask_token_id: int,
    t_float: jnp.ndarray | None = None,
    aux_loss: float | jnp.ndarray = 0.0,
    alpha_balance: float = 0.1,
) -> jnp.ndarray:
    """LLaDA ELBO loss (Eq. 3): (1/t)-weighted masked cross-entropy.

    Args:
        logits:        Model predictions, shape ``[B, L, vocab_size]``.
        targets:       Original clean tokens x_0, shape ``[B, L]``.
        x_t:           Noisy tokens, shape ``[B, L]``.
        mask_token_id: Vocabulary ID of ``[MASK]``.
        t_float:       Per-sample noise level, shape ``[B]``, values in (0, 1].
                       When provided, applies the LLaDA ``1/t`` importance weight.
                       Falls back to plain mean-over-masked-tokens when ``None``.
        aux_loss:      MoE load-balancing term.
        alpha_balance: Weight for aux_loss.
    """
    is_masked  = (x_t == mask_token_id)                                     # [B, L]
    log_probs  = jax.nn.log_softmax(logits, axis=-1)
    nll        = -jnp.sum(log_probs * jax.nn.one_hot(targets, logits.shape[-1]), axis=-1)
    nll_masked = jnp.where(is_masked, nll, 0.0)                             # [B, L]

    if t_float is not None:
        # LLaDA Eq. 3: (1/t) * Σ_masked nll, divided by L for scale invariance.
        # Expected value ≈ avg NLL per token regardless of t.
        L       = logits.shape[1]
        t_safe  = jnp.maximum(t_float, 1e-6)                                # [B]
        per_ex  = nll_masked.sum(axis=-1) / (t_safe * L)                    # [B]
        base_loss = per_ex.mean()
    else:
        n_masked  = jnp.maximum(is_masked.astype(jnp.float32).sum(), 1.0)
        base_loss = nll_masked.sum() / n_masked

    return base_loss + alpha_balance * aux_loss


# ── Confidence-aware parallel decoding helpers (Fast-dLLM §3.3) ───────────────

def confidence_unmask_threshold(
    logits: jnp.ndarray,           # [B, T, V]
    x_t: jnp.ndarray,              # [B, T]
    mask_token_id: int,
    threshold: float = 0.9,
) -> jnp.ndarray:
    """Unmask all masked positions whose max-softmax confidence ≥ τ.

    At least one token is always unmasked (the most confident one) to
    guarantee forward progress (Alg. 1 line 9).

    Returns the updated token sequence.
    """
    B, T = x_t.shape
    probs      = jax.nn.softmax(logits, axis=-1)              # [B, T, V]
    confidence = probs.max(axis=-1)                            # [B, T]
    x0_pred    = jnp.argmax(logits, axis=-1)                  # [B, T]
    is_masked  = (x_t == mask_token_id)                       # [B, T]

    do_unmask = is_masked & (confidence >= threshold)

    # Progress guarantee: if nothing was selected, unmask the most confident masked token
    any_unmasked = do_unmask.any(axis=-1, keepdims=True)       # [B, 1]
    masked_conf  = jnp.where(is_masked, confidence, -1.0)
    best_idx     = jnp.argmax(masked_conf, axis=-1)            # [B]
    forced       = (jnp.arange(T)[None, :] == best_idx[:, None]) & is_masked & ~any_unmasked

    do_unmask = do_unmask | forced
    return jnp.where(do_unmask, x0_pred, x_t)


def confidence_unmask_factor(
    logits: jnp.ndarray,
    x_t: jnp.ndarray,
    mask_token_id: int,
    factor: float = 1.5,
) -> jnp.ndarray:
    """Factor-based parallel decoding (Fast-dLLM Alg. 1, lines 10-13).

    Finds the largest n such that (n+1)(1 - c_(n)) < f, where c_(n) is the
    n-th highest confidence among masked positions.  This is a tighter,
    theoretically grounded variant of the threshold strategy (Theorem 1).

    Fully vectorised over batch and sequence, so it is jit-compatible and
    runs without host round-trips.
    """
    B, T = x_t.shape
    probs      = jax.nn.softmax(logits, axis=-1)
    confidence = probs.max(axis=-1)                            # [B, T]
    x0_pred    = jnp.argmax(logits, axis=-1)                   # [B, T]
    is_masked  = (x_t == mask_token_id)                        # [B, T]

    # Rank masked positions by confidence (descending); unmasked positions get
    # -inf so they sort last and can never enter the selected prefix.
    conf_masked = jnp.where(is_masked, confidence, -jnp.inf)   # [B, T]
    order       = jnp.argsort(-conf_masked, axis=-1)           # [B, T]
    sorted_conf = jnp.take_along_axis(conf_masked, order, axis=-1)

    # The original algorithm extends the prefix while consecutive ranks n
    # satisfy (n+1)(1 - c_(n)) < f, always unmasking at least rank 0.
    ranks  = jnp.arange(T, dtype=sorted_conf.dtype)
    cond   = (ranks + 1.0) * (1.0 - sorted_conf) < factor      # [B, T]
    cond   = cond.at[:, 0].set(True)                            # progress guarantee
    prefix = jnp.cumprod(cond.astype(jnp.int32), axis=-1).astype(bool)

    # Only genuinely masked entries may be revealed (rows with no masks
    # select nothing because their sorted entries are all -inf / unmasked).
    sorted_is_masked = jnp.take_along_axis(is_masked, order, axis=-1)
    select_sorted    = prefix & sorted_is_masked                # [B, T]

    # Scatter the selection back to original positions via the inverse permutation.
    inv_order = jnp.argsort(order, axis=-1)
    do_unmask = jnp.take_along_axis(select_sorted, inv_order, axis=-1)

    return jnp.where(do_unmask, x0_pred, x_t)


# ── Backward-compatibility re-exports ─────────────────────────────────────────
# The continuous flow-matching utilities (corrupt_denoiser, corrupt_decoder,
# sample_*_*, logit_normal_schedule) moved to flow.py, where the rest of the
# flow-matching paradigm lives.  Re-exported here so old imports keep working;
# removed in v1.0.
from .flow import (  # noqa: E402, F401
    corrupt_decoder,
    corrupt_denoiser,
    logit_normal_schedule,
    sample_cfg_scale,
    sample_p_per_token,
    sample_t_logit_normal,
)
