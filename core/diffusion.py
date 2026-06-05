"""Masked Discrete Diffusion for DantinoX — Fast-dLLM KV Cache.

Implements masked-diffusion (MDLM-style) forward/reverse process with the
**block-wise approximate KV cache** from Fast-dLLM (Wu et al., 2025,
arXiv:2505.22618).

Forward process
---------------
Each token is independently replaced by ``[MASK]`` with probability 1 − ᾱ_t:

    x_t[i] = x_0[i]           with probability  ᾱ_t
    x_t[i] = mask_token_id    with probability  1 − ᾱ_t

Reverse process
---------------
A bidirectional ``DiffusionTransformer`` is trained to predict x_0 from
(x_t, t).  Block-wise generation with DualCache is used for efficient
inference.

DualCache (Fast-dLLM §3.2)
---------------------------
The output sequence is divided into K blocks of size B.  For block k:

    [Prompt | Block 0 | … | Block k-1 | **Block k** | Block k+1 | … ]
    |<——— prefix_kvs ————>|              |<——— suffix_kvs ————————>|

  • ``prefix_kvs``: KV for the prompt — computed from the full forward pass
    and reused unchanged for all steps within every block.

  • ``suffix_kvs``: KV for the remaining *all-MASK* blocks after block k —
    also reused within the inner loop and **refreshed** (recomputed) only
    after block k is fully decoded.

For each inner step the model runs **only on x[s:e]** (block k tokens);
its attention context is [prefix_KV | fresh_block_KV | suffix_KV].
This avoids recomputing suffix KV every step and gives 1.4–2.1× speedup
over prefix-only caching (Table 4 in the paper).

Confidence-aware parallel decoding (Fast-dLLM §3.3)
----------------------------------------------------
Instead of unmasking a fixed number of tokens per step, only tokens whose
max-softmax confidence exceeds a threshold τ are revealed.  The *factor*
strategy extends this by finding the largest n satisfying (n+1)(1-c_(n))<f,
providing a theoretically grounded guarantee (Theorem 1).
"""
from __future__ import annotations

from typing import NamedTuple

import flax.nnx as nnx
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
        a block's inner loop (high cosine similarity across adjacent steps,
        Fig. 3 in the paper).  Refreshed at each block boundary.
        ``None`` when using prefix-only caching, or before the first block.
    """
    prefix_kvs: tuple
    suffix_kvs: tuple | None = None


# ── Noise schedule ─────────────────────────────────────────────────────────────

class NoiseSchedule(NamedTuple):
    """Precomputed ᾱ_t for t = 0, …, T.

    ᾱ_t = probability that a token is *unmasked* at step t.
    ᾱ_0 = 1 (clean), ᾱ_T ≈ 0 (fully masked).
    """
    alpha_bar: jnp.ndarray  # shape [T+1]
    schedule: str


def make_noise_schedule(config: Config) -> NoiseSchedule:
    """Build a discrete masking schedule from ``config.noise_schedule``.

    Supported schedules
    -------------------
    ``"cosine"``  Squared-cosine (Nichol & Dhariwal 2021). Slow at boundaries.
    ``"linear"``  ᾱ_t = 1 − t/T.
    ``"sqrt"``    ᾱ_t = 1 − √(t/T).  Masking decelerates over time.
    """
    T = config.diffusion_steps
    t = jnp.arange(T + 1, dtype=jnp.float32) / T

    if config.noise_schedule == "cosine":
        s     = 0.008
        alpha = jnp.cos(((t + s) / (1.0 + s)) * (jnp.pi / 2.0)) ** 2
        alpha = alpha / alpha[0]
    elif config.noise_schedule == "linear":
        alpha = 1.0 - t
    else:  # "sqrt"
        alpha = 1.0 - jnp.sqrt(t + 1e-4)

    return NoiseSchedule(
        alpha_bar=jnp.clip(alpha, 0.0, 1.0),
        schedule=config.noise_schedule,
    )


# ── Forward process ────────────────────────────────────────────────────────────

def corrupt(
    x0: jnp.ndarray,
    t: jnp.ndarray,
    rng: jax.Array,
    schedule: NoiseSchedule,
    mask_token_id: int,
) -> jnp.ndarray:
    """Apply masked-diffusion forward process to a batch.

    Each token is independently replaced by ``mask_token_id`` with
    probability 1 − ᾱ_t.

    Args:
        x0:            Clean token IDs, shape ``[B, T]``.
        t:             Per-sample integer timestep indices, shape ``[B]``.
        rng:           JAX PRNG key.
        schedule:      Precomputed noise schedule.
        mask_token_id: Vocabulary ID of ``[MASK]``.

    Returns:
        Noisy token sequence ``x_t``, shape ``[B, T]``.
    """
    alpha_t = schedule.alpha_bar[t][:, None]  # [B, 1]
    keep    = jax.random.bernoulli(rng, alpha_t, x0.shape)
    return jnp.where(keep, x0, mask_token_id)


# ── Loss ───────────────────────────────────────────────────────────────────────

def masked_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    x_t: jnp.ndarray,
    mask_token_id: int,
    aux_loss: float | jnp.ndarray = 0.0,
    alpha_balance: float = 0.1,
) -> jnp.ndarray:
    """Cross-entropy ELBO loss, evaluated only at masked positions.

    Args:
        logits:        Model predictions, shape ``[B, T, vocab_size]``.
        targets:       Original clean tokens x_0, shape ``[B, T]``.
        x_t:           Noisy tokens, shape ``[B, T]``.
        mask_token_id: Vocabulary ID of ``[MASK]``.
        aux_loss:      MoE load-balancing term.
        alpha_balance: Weight for aux_loss.
    """
    is_masked  = (x_t == mask_token_id)
    log_probs  = jax.nn.log_softmax(logits, axis=-1)
    nll        = -jnp.sum(log_probs * jax.nn.one_hot(targets, logits.shape[-1]), axis=-1)
    nll_masked = jnp.where(is_masked, nll, 0.0)
    n_masked   = jnp.maximum(jnp.sum(is_masked.astype(jnp.float32)), 1.0)
    return jnp.sum(nll_masked) / n_masked + alpha_balance * aux_loss


# ── Time embedding ─────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal timestep embedding, shape ``[B, dim]``."""
    half    = dim // 2
    freqs   = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args    = t[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class TimeEmbedding(nnx.Module):
    """Maps integer diffusion timesteps → continuous embeddings (sinusoidal + 2-layer MLP)."""

    def __init__(self, model_dim: int, emb_dim: int, rngs: nnx.Rngs) -> None:
        self.model_dim = model_dim
        self.fc1 = nnx.Linear(model_dim, emb_dim, rngs=rngs)
        self.fc2 = nnx.Linear(emb_dim,   emb_dim, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """t: [B] integer timesteps → [B, emb_dim]."""
        x = sinusoidal_embedding(t, self.model_dim)
        x = jax.nn.silu(self.fc1(x))
        return self.fc2(x)


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
