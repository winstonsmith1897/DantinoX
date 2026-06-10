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
    """Continuous masking schedule: maps t ∈ [0,1] → mask probability p_mask(t).

    schedule: "linear" | "cosine" | "sqrt"
    """
    schedule: str


def make_noise_schedule(config: Config) -> NoiseSchedule:
    return NoiseSchedule(schedule=config.noise_schedule)


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
    """
    B, T = x_t.shape
    V           = logits.shape[-1]
    probs       = jax.nn.softmax(logits, axis=-1)
    confidence  = probs.max(axis=-1)                            # [B, T]
    x0_pred     = jnp.argmax(logits, axis=-1)
    is_masked   = (x_t == mask_token_id)

    # Process each batch element independently (Python loop over B — small)
    new_x = x_t
    for b in range(B):
        masked_pos  = jnp.where(is_masked[b])[0]               # positions that are MASK
        if masked_pos.size == 0:
            continue
        conf_masked = confidence[b][masked_pos]
        # Sort descending
        order       = jnp.argsort(-conf_masked)
        sorted_conf = conf_masked[order]
        sorted_pos  = masked_pos[order]

        # Find largest n with (n+1)(1 - c_(n)) < factor
        n_unmask = 1  # always unmask at least 1
        for n in range(1, len(sorted_conf)):
            if (n + 1) * (1.0 - float(sorted_conf[n])) < factor:
                n_unmask = n + 1
            else:
                break

        for idx in range(n_unmask):
            pos   = int(sorted_pos[idx])
            token = int(x0_pred[b, pos])
            new_x = new_x.at[b, pos].set(token)

    return new_x


# ── Continuous flow-matching utilities (ELF) ──────────────────────────────────
# Analogous to corrupt() / masked_cross_entropy() above, but for the continuous
# rectified-flow paradigm used by ELFTransformer.
#
# Forward process:  z_t = t·x + (1−t)·ε,  ε ~ N(0, I),  t ∈ [0, 1]
# Network predicts clean embeddings x̂ (x-prediction).
# At inference, use logit_normal_schedule + ODE/SDE steps (see generation.py).

def sample_t_logit_normal(
    rng:        jax.Array,
    batch_size: int,
    pmean:      float,
    pstd:       float,
) -> jax.Array:
    """Sample t ∈ (0, 1) per example from a logit-normal distribution.

    Draws t' ~ N(pmean, pstd²) then maps t = sigmoid(t').
    Default denoiser params (pmean=-1.5, pstd=0.8) concentrate mass near
    t ≈ 0.18, ensuring the noisy regime is well-trained.
    """
    return jax.nn.sigmoid(jax.random.normal(rng, (batch_size,)) * pstd + pmean)


def sample_p_per_token(
    rng:        jax.Array,
    batch_size: int,
    seq_len:    int,
    pmean:      float,
    pstd:       float,
) -> jax.Array:
    """Per-token corruption level p ∈ (0, 1) for the ELF decoder branch.

    Each token in each sequence gets its own p, encouraging the decoder to
    handle a range of corruption levels within a single context.
    Returns shape [B, L].
    """
    return jax.nn.sigmoid(jax.random.normal(rng, (batch_size, seq_len)) * pstd + pmean)


def sample_cfg_scale(
    rng:       jax.Array,
    batch_size: int,
    scale_min: float,
    scale_max: float,
) -> jax.Array:
    """Sample CFG scale w ∈ [scale_min, scale_max] with a power distribution.

    Uses a quadratic transform biased toward smaller values, matching ELF §B.1.
    Returns shape [B].
    """
    u = jax.random.uniform(rng, (batch_size,))
    return scale_min + (scale_max - scale_min) * u ** 2


def corrupt_denoiser(
    x:           jnp.ndarray,  # [B, L, D] clean normalised embeddings
    t:           jnp.ndarray,  # [B]
    rng:         jax.Array,
    noise_scale: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Denoiser-branch corruption (ELF Appendix B.1).

    z_t = t · x + (1 − t) · (noise_scale · ε),   ε ~ N(0, I)
    v   = x − (noise_scale · ε)

    Returns ``(z_t, v)``.  ``noise_scale=2`` is the ELF default.
    """
    eps = noise_scale * jax.random.normal(rng, x.shape)
    t_b = t[:, None, None]
    z_t = t_b * x + (1.0 - t_b) * eps
    return z_t, x - eps


def corrupt_decoder(
    x:           jnp.ndarray,  # [B, L, D] clean normalised embeddings
    p:           jnp.ndarray,  # [B, L] per-token corruption level
    rng:         jax.Array,
    noise_scale: float = 5.0,
) -> jnp.ndarray:
    """Decoder-branch corruption (ELF Appendix B.1).

    z̃ = p · x + (1 − p) · (noise_scale · ε),   ε ~ N(0, I)

    Per-token p makes the decoder robust to imperfect denoiser outputs.
    """
    eps = noise_scale * jax.random.normal(rng, x.shape)
    p_b = p[:, :, None]
    return p_b * x + (1.0 - p_b) * eps


def logit_normal_schedule(
    n_steps: int,
    pmean:   float = -1.5,
    pstd:    float = 0.8,
    *,
    rng:     jax.Array | None = None,
) -> jnp.ndarray:
    """Logit-normal inference time schedule for ELF (ELF §B.2).

    Samples ``n_steps − 1`` interior time-points from the same logit-normal
    distribution used at training time, sorts them, and bookends with 0 and 1.

    Returns sorted array of shape ``[n_steps + 1]`` with ts[0]=0, ts[-1]=1.
    Smaller intervals near t=0 (noisy) and larger near t=1 (clean).
    """
    import numpy as np
    rng_np = np.random.default_rng(
        int(jax.random.randint(rng, (), 0, 2**31 - 1)) if rng is not None else None
    )
    t_prime   = rng_np.normal(pmean, pstd, size=(n_steps - 1,))
    t_interior = 1.0 / (1.0 + np.exp(-t_prime))
    t_interior.sort()
    return jnp.asarray(np.concatenate([[0.0], t_interior, [1.0]]), dtype=jnp.float32)
