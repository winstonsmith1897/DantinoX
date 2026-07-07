"""Continuous flow-matching language modelling in embedding space.

The current implementation follows the ELF recipe ("ELF: Embedded Language
Flows", Hu et al., 2026, arXiv:2605.10938).  ELF is *one* instantiation of
the continuous flow-matching paradigm; the public names in this module are
paradigm-level (``FlowMatchingTransformer``, ``flow_loss``, …) so alternative
formulations can be added without renaming the API.  The old ``ELF*`` names
remain available as deprecated aliases.

Overview
--------
The model operates in a *continuous embedding space* rather than discrete
token space.  The forward process is a rectified flow:

    z_t = t · x + (1 − t) · ε,    ε ~ N(0, I),    t ∈ [0, 1]

where x is the clean normalised embedding (encoded from token IDs) and t=0
is pure noise, t=1 is clean data.  The network uses *x-prediction*: it predicts
the clean embedding x̂ directly; the flow velocity is recovered as
v(z_t, t) = (x̂ − z_t) / (1 − t).

Library integration
-------------------
The flow-matching paradigm follows the same conventions as the rest of DantinoX:

=========================  =================================
AR / masked diffusion      Continuous flow-matching
=========================  =================================
``Transformer``            ``FlowMatchingTransformer``
``ModelConfig``            ``FlowMatchingConfig`` (in config.py)
``ModelOutput``            ``FlowMatchingOutput`` (in output.py)
``corrupt``                ``corrupt_denoiser/decoder``
``masked_cross_entropy``   ``flow_mse_loss / flow_ce_loss``
``diffusion_generate``     ``flow_generate`` (generation.py)
=========================  =================================

Quick-start
-----------
::

    from dantinox.core.config import FlowMatchingConfig
    from dantinox.core.flow import FlowMatchingTransformer, flow_loss
    from dantinox.core.generation import flow_generate

    cfg   = FlowMatchingConfig(embed_dim=512, bottleneck_dim=128,
                      model_dim=768, n_heads=12, head_size=64)
    model = FlowMatchingTransformer(cfg, rngs=nnx.Rngs(42))

    # Training step
    x    = model.encode(tokens)                        # [B, L, E]
    loss, aux = flow_loss(model, x, tokens, rng, cfg)

    # Inference
    tokens = flow_generate(model, gen_len=256, n_steps=64, cfg_scale=2.0)
"""
from __future__ import annotations

import math
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from .block import Block, RMSNorm
from .config import FlowMatchingConfig
from .output import FlowMatchingOutput

# ── Variable types ────────────────────────────────────────────────────────────

class Frozen(nnx.Variable):
    """Non-trainable variable — excluded from optimizer updates.

    The optimizer targets ``nnx.Param``; anything stored as ``Frozen`` is
    carried in the model state but never receives a gradient update.  Used for
    pretrained weights that must not change during fine-tuning (e.g. the T5
    embedding table).
    """


# ── Private helpers ────────────────────────────────────────────────────────────

def _sinusoidal_embed(x: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Map scalar values x [B] to sinusoidal embeddings [B, dim].

    Used to encode the continuous conditioning values (t, w) into the
    control-token prefix prepended to each sequence.
    """
    half  = dim // 2
    freqs = jnp.exp(
        -math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
    )
    angles = x[:, None].astype(jnp.float32) * freqs[None, :]   # [B, half]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # [B, dim]


# ── Forward process and schedules ─────────────────────────────────────────────
# Analogous to diffusion.corrupt() / masked_cross_entropy(), but for the
# continuous rectified-flow paradigm used by FlowMatchingTransformer.
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
    rng_np = np.random.default_rng(
        int(jax.random.randint(rng, (), 0, 2**31 - 1)) if rng is not None else None
    )
    t_prime   = rng_np.normal(pmean, pstd, size=(n_steps - 1,))
    t_interior = 1.0 / (1.0 + np.exp(-t_prime))
    t_interior.sort()
    return jnp.asarray(np.concatenate([[0.0], t_interior, [1.0]]), dtype=jnp.float32)


# ── Flow embedder ─────────────────────────────────────────────────────────────

class FlowEmbedder(nnx.Module, pytree=False):
    """Normalizes pre-computed T5 contextual embeddings to zero-mean, unit-std.

    The actual T5 encoder forward pass happens outside JIT in
    ``utils.t5_encoder.T5ContextualEncoder``.  This module only holds the
    channel-wise normalization statistics (mean, std) and applies them.

    Statistics are initialized to (0, 1) and updated once before training
    starts from a representative sample of encoded training sequences.
    """

    def __init__(self, embed_dim: int, rngs: nnx.Rngs) -> None:  # noqa: ARG002
        # Placeholder stats — updated from real T5 encoder outputs before training
        self.emb_mean = nnx.Variable(jnp.zeros(embed_dim))  # [E]
        self.emb_std  = nnx.Variable(jnp.ones(embed_dim))   # [E]

    def __call__(self, embeddings: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        """embeddings ``[B, L, E]`` float → ``[B, L, E]`` (zero-mean, unit-std)."""
        if normalize:
            return (embeddings - self.emb_mean[...]) / (self.emb_std[...] + 1e-6)
        return embeddings


# ── Flow-matching transformer ─────────────────────────────────────────────────

class FlowMatchingTransformer(nnx.Module, pytree=False):
    """Shared-weight flow-matching denoiser + decoder model (ELF recipe).

    Continuous-flow analogue of ``Transformer`` (model.py).  A single set of
    weights handles both training branches, selected by in-context mode tokens:

    * **Denoiser** (Algorithm 3): MSE on predicted velocity with training-time
      CFG and self-conditioning.
    * **Decoder** (Algorithm 4): cross-entropy at t=1 with per-token corruption.

    Architecture
    ------------
    Input z_t ``[B, L, embed_dim]`` passes through::

        self_cond_proj(concat[z_t, x_prev])       # [B, L, E]   → E
        → in_proj (E → Bd) → SiLU → in_proj2 (Bd → D)  # bottleneck
        → concat([ctrl_tokens, h])                # [B, C+L, D]
        → num_blocks × Block (bidirectional)
        → norm_f → strip ctrl → out_proj (D → E)  # [B, L, E]  = x_pred
        → unembed (E → V)                          # [B, L, V]  = logits

    where C = ``num_ctrl`` = num_time_tokens + num_cfg_tokens + num_mode_tokens.

    Returns
    -------
    ``FlowMatchingOutput(x_pred, logits)``

    * ``x_pred``: predicted clean embeddings ``[B, L, embed_dim]`` — use for
      velocity computation and MSE loss.
    * ``logits``: token logits ``[B, L, vocab_size]`` — use for CE loss and
      final-step argmax decoding.

    Quick-start
    -----------
    ::

        cfg   = FlowMatchingConfig(embed_dim=512, bottleneck_dim=128,
                          model_dim=768, n_heads=12, head_size=64)
        model = FlowMatchingTransformer(cfg, rngs=nnx.Rngs(42))

        # or via the builder
        model = FlowMatchingTransformer.build(
            embed_dim=512, bottleneck_dim=128,
            model_dim=768, n_heads=12, head_size=64,
            num_blocks=12, vocab_size=32_000, max_seq_len=1024,
            rngs=nnx.Rngs(42),
        )
    """

    def __init__(self, config: FlowMatchingConfig, rngs: nnx.Rngs) -> None:
        if config.vocab_size is None:
            raise ValueError(
                "FlowMatchingConfig.vocab_size is not set. "
                "When using Trainer.fit() it is inferred automatically from the tokenizer. "
                "For direct model building pass vocab_size=<n> to FlowMatchingConfig."
            )
        self.config: FlowMatchingConfig = config
        # Not a FlowMatchingConfig field (moved to TrainingConfig): plain
        # mutable attribute defaulting False, flipped on by Trainer.fit()
        # when the user's TrainingConfig requests it.
        self.gradient_checkpointing: bool = getattr(config, "gradient_checkpointing", False)

        D  = config.model_dim
        E  = config.embed_dim
        Bd = config.bottleneck_dim

        # Normalizer for pre-computed T5 contextual embeddings (stats set in trainer)
        self.embedder = FlowEmbedder(E, rngs)

        # Self-conditioning projection: [z_t ‖ x_prev] → z_sc
        self.self_cond_proj = nnx.Linear(2 * E, E, use_bias=True,  rngs=rngs)

        # Bottleneck: E → Bd → D
        self.in_proj  = nnx.Linear(E,  Bd, use_bias=False, rngs=rngs)
        self.in_proj2 = nnx.Linear(Bd, D,  use_bias=True,  rngs=rngs)

        # Output projection: D → E
        self.out_proj = nnx.Linear(D, E, use_bias=True, rngs=rngs)

        # Control-token projections (sinusoidal scalar → model-dim token sequence)
        self.time_proj = nnx.Linear(
            config.time_emb_dim, config.num_time_tokens * D, use_bias=True, rngs=rngs
        )
        self.cfg_proj = nnx.Linear(
            config.time_emb_dim, config.num_cfg_tokens * D,  use_bias=True, rngs=rngs
        )

        # Learnable mode-token prototypes (denoiser vs decoder)
        _normal = nnx.initializers.normal(stddev=0.02)
        self.mode_denoise = nnx.Param(
            _normal(rngs.params(), (config.num_mode_tokens, D), jnp.float32)
        )
        self.mode_decode = nnx.Param(
            _normal(rngs.params(), (config.num_mode_tokens, D), jnp.float32)
        )

        # Bidirectional transformer backbone
        mc = config.to_model_config()
        self.blocks: list[Block] = [Block(mc, rngs=rngs) for _ in range(config.num_blocks)]
        self.norm_f = RMSNorm(D, rngs=rngs)

        # Shared unembedding head (E → V)
        self.unembed = nnx.Linear(E, config.vocab_size, use_bias=False, rngs=rngs)

    # ── Public interface ───────────────────────────────────────────────────────

    def encode(self, embeddings: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        """Normalize pre-computed T5 embeddings ``[B, L, E]`` → ``[B, L, E]``."""
        return self.embedder(embeddings, normalize=normalize)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(
        self,
        z_t:       jnp.ndarray,  # [B, L, embed_dim]
        x_prev:    jnp.ndarray,  # [B, L, embed_dim]  zeros on first call
        t:         jnp.ndarray,  # [B]  in [0, 1]
        cfg_scale: jnp.ndarray,  # [B]
        is_decode: jnp.ndarray,  # [B] bool  True at the t=1 decode step
        *,
        deterministic: bool = False,
    ) -> FlowMatchingOutput:
        """Predict clean embeddings x̂ from noisy embeddings z_t.

        Parameters
        ----------
        z_t:          Noisy embeddings ``[B, L, embed_dim]``.
        x_prev:       Self-conditioning signal ``[B, L, embed_dim]``.
                      Pass ``jnp.zeros_like(z_t)`` on the first call.
        t:            Timestep ∈ [0, 1], shape ``[B]``.
        cfg_scale:    CFG guidance scale w, shape ``[B]``.
        is_decode:    ``True`` at the final t=1 decode step — selects the
                      decode mode tokens instead of denoiser tokens.
        deterministic: Disables dropout (set ``True`` at inference).

        Returns
        -------
        ``FlowMatchingOutput(x_pred, logits)``
        """
        cfg = self.config

        # 1. Self-conditioning: fuse noisy input with previous prediction
        z_sc = self.self_cond_proj(jnp.concatenate([z_t, x_prev], axis=-1))

        # 2. Bottleneck projection into transformer space
        h = jax.nn.silu(self.in_proj(z_sc))   # [B, L, Bd]
        h = self.in_proj2(h)                   # [B, L, D]

        # 3. Build and prepend control tokens [t, w, mode]
        ctrl = self._ctrl_tokens(t, cfg_scale, is_decode)  # [B, C, D]
        h    = jnp.concatenate([ctrl, h], axis=1)           # [B, C+L, D]

        # 4. Bidirectional transformer
        # Rematerialisation only pays off when gradients flow; at inference
        # (deterministic=True) it would recompute every block for nothing.
        use_remat = self.gradient_checkpointing and not deterministic
        if use_remat:
            def _block_fn(block: object, hs: jnp.ndarray) -> tuple:
                return block(hs, deterministic=deterministic)  # type: ignore[call-arg, operator]
            _checkpointed = nnx.remat(_block_fn)
        for block in self.blocks:
            if use_remat:
                h, _, _ = _checkpointed(block, h)  # type: ignore[possibly-undefined]
            else:
                h, _, _ = block(h, deterministic=deterministic)
        h = self.norm_f(h)

        # 5. Strip control prefix and project back to embedding space
        x_pred = self.out_proj(h[:, cfg.num_ctrl:])  # [B, L, E]

        # 6. Unembed to token logits (always materialised; cheap relative to transformer)
        logits = self.unembed(x_pred)                # [B, L, V]

        return FlowMatchingOutput(x_pred=x_pred, logits=logits)

    # ── Builder ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        embed_dim:      int,
        bottleneck_dim: int,
        model_dim:      int,
        n_heads:        int,
        head_size:      int,
        num_blocks:     int,
        vocab_size:     int,
        max_seq_len:    int,
        rngs:           nnx.Rngs,
        **kwargs: Any,
    ) -> FlowMatchingTransformer:
        """Construct an ``FlowMatchingTransformer`` from individual hyperparameters.

        All parameters are forwarded to ``FlowMatchingConfig``.  Keyword arguments that
        are not ``FlowMatchingConfig`` fields are dropped with a warning (so a typo'd
        hyperparameter cannot vanish silently).

        Example::

            model = FlowMatchingTransformer.build(
                embed_dim=512, bottleneck_dim=128,
                model_dim=768, n_heads=12, head_size=64,
                num_blocks=12, vocab_size=tok.vocab_size, max_seq_len=1024,
                dropout=0.1, denoiser_pmean=-1.5,
                rngs=nnx.Rngs(42),
            )
        """
        import warnings
        from dataclasses import fields as _fields

        valid = {f.name for f in _fields(FlowMatchingConfig)}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            warnings.warn(
                f"FlowMatchingTransformer.build: ignoring unknown keyword arguments {unknown} "
                "(not FlowMatchingConfig fields)",
                UserWarning,
                stacklevel=2,
            )
        cfg   = FlowMatchingConfig(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            model_dim=model_dim,
            n_heads=n_heads,
            head_size=head_size,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            **{k: v for k, v in kwargs.items() if k in valid},
        )
        return cls(cfg, rngs=rngs)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ctrl_tokens(
        self,
        t:         jnp.ndarray,  # [B]
        cfg_scale: jnp.ndarray,  # [B]
        is_decode: jnp.ndarray,  # [B] bool
    ) -> jnp.ndarray:             # [B, num_ctrl, model_dim]
        cfg = self.config
        B   = t.shape[0]
        D   = cfg.model_dim

        t_tok = self.time_proj(
            _sinusoidal_embed(t, cfg.time_emb_dim)
        ).reshape(B, cfg.num_time_tokens, D)

        w_tok = self.cfg_proj(
            _sinusoidal_embed(cfg_scale, cfg.time_emb_dim)
        ).reshape(B, cfg.num_cfg_tokens, D)

        m_den = jnp.broadcast_to(self.mode_denoise[...][None], (B, cfg.num_mode_tokens, D))
        m_dec = jnp.broadcast_to(self.mode_decode[...][None],  (B, cfg.num_mode_tokens, D))
        m_tok = jnp.where(is_decode[:, None, None], m_dec, m_den)

        return jnp.concatenate([t_tok, w_tok, m_tok], axis=1)  # [B, num_ctrl, D]


# ── Deprecated ELF-branded aliases ────────────────────────────────────────────
# The paradigm is "continuous flow-matching"; ELF (Hu et al., 2026) is the
# specific recipe currently implemented.  These aliases keep old imports and
# checkpoints working and will be removed in v1.0.

ELFNet         = FlowMatchingTransformer
ELFTransformer = FlowMatchingTransformer
ELFEmbedder    = FlowEmbedder


# ── Loss functions ─────────────────────────────────────────────────────────────

def flow_mse_loss(v_pred: jnp.ndarray, v_target: jnp.ndarray) -> jnp.ndarray:
    """Mean-squared error on flow velocity predictions (ELF Eq. 1)."""
    return jnp.mean((v_pred - v_target) ** 2)


def flow_ce_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Per-token cross-entropy for the decode step (ELF Eq. 2).

    Parameters
    ----------
    logits:  ``[B, L, vocab_size]``
    targets: ``[B, L]`` int token IDs
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot   = jax.nn.one_hot(targets, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def flow_denoiser_loss(
    model:      FlowMatchingTransformer,
    embeddings: jnp.ndarray,    # [B, L, embed_dim] clean normalised embeddings
    rng:        jax.Array,
    config:     FlowMatchingConfig,
) -> jnp.ndarray:
    """Training-time CFG denoiser loss (ELF Algorithm 3).

    Two forward passes per batch:

    1. **No self-conditioning** (x_prev = 0) → x_no_sc
    2. **Self-conditioning** on stop_gradient(x_no_sc) → x_sc

    CFG regression target (ELF Appendix B, Eq. 3)::

        v_target = v + (1 − 1/w) · (v_sc − v_no_sc)

    Each example uses the self-conditioned path with probability
    ``config.self_cond_prob``, and the unconditioned path otherwise.
    """
    B = embeddings.shape[0]
    rng_t, rng_noise, rng_cfg, rng_sc = jax.random.split(rng, 4)

    t   = sample_t_logit_normal(rng_t,  B, config.denoiser_pmean, config.denoiser_pstd)
    w   = sample_cfg_scale(rng_cfg,     B, config.cfg_scale_min,   config.cfg_scale_max)
    z_t, v = corrupt_denoiser(embeddings, t, rng_noise, config.denoiser_noise_scale)

    is_denoise = jnp.zeros(B, dtype=bool)
    zeros      = jnp.zeros_like(embeddings)
    inv_1mt    = 1.0 / jnp.clip(1.0 - t[:, None, None], 1e-6)

    # Pass 1 — unconditioned
    out_no_sc = model(z_t, zeros, t, w, is_denoise)
    v_no_sc   = (out_no_sc.x_pred - z_t) * inv_1mt

    # Pass 2 — self-conditioned on stop_gradient of pass 1
    out_sc = model(z_t, jax.lax.stop_gradient(out_no_sc.x_pred), t, w, is_denoise)
    v_sc   = (out_sc.x_pred - z_t) * inv_1mt

    # CFG target
    v_cfg = v + (1.0 - 1.0 / w[:, None, None]) * (v_sc - v_no_sc)

    # Stochastic self-conditioning mask
    sc_mask  = jax.random.uniform(rng_sc, (B,)) < config.self_cond_prob
    v_pred   = jnp.where(sc_mask[:, None, None], v_sc,  v_no_sc)
    v_target = jax.lax.stop_gradient(
        jnp.where(sc_mask[:, None, None], v_cfg, v)
    )

    return flow_mse_loss(v_pred, v_target)


def flow_decoder_loss(
    model:      FlowMatchingTransformer,
    embeddings: jnp.ndarray,    # [B, L, embed_dim] clean normalised embeddings
    tokens:     jnp.ndarray,    # [B, L] int ground-truth token IDs
    rng:        jax.Array,
    config:     FlowMatchingConfig,
) -> jnp.ndarray:
    """Decoder training step loss (ELF Algorithm 4).

    Corrupts embeddings with per-token noise, runs in decode mode (t=1,
    x_prev=0), then computes cross-entropy against ground-truth tokens.
    """
    B, L, _ = embeddings.shape
    rng_p, rng_noise, rng_cfg = jax.random.split(rng, 3)

    p     = sample_p_per_token(rng_p, B, L, config.decoder_pmean, config.decoder_pstd)
    w     = sample_cfg_scale(rng_cfg,  B,    config.cfg_scale_min,  config.cfg_scale_max)
    z_dec = corrupt_decoder(embeddings, p, rng_noise, config.decoder_noise_scale)

    out = model(
        z_dec,
        jnp.zeros_like(embeddings),
        jnp.ones(B),
        w,
        jnp.ones(B, dtype=bool),
    )
    return flow_ce_loss(out.logits, tokens)


def flow_loss(
    model:      FlowMatchingTransformer,
    embeddings: jnp.ndarray,    # [B, L, embed_dim]
    tokens:     jnp.ndarray,    # [B, L] int
    rng:        jax.Array,
    config:     FlowMatchingConfig,
) -> tuple[jnp.ndarray, dict]:
    """Combined flow-matching training loss (ELF recipe, Algorithm 1).

    Stochastically assigns each example to the denoiser branch
    (probability ``config.denoiser_prob``) or the decoder branch, runs both,
    and returns the batch-weighted sum.

    Returns
    -------
    ``(loss, {"den_loss": scalar, "dec_loss": scalar})``
    """
    B = tokens.shape[0]
    rng_mode, rng_den, rng_dec = jax.random.split(rng, 3)

    is_den = jax.random.uniform(rng_mode, (B,)) < config.denoiser_prob
    n_den  = jnp.sum(is_den).astype(jnp.float32)
    n_dec  = (B - n_den).astype(jnp.float32)

    den_loss = flow_denoiser_loss(model, embeddings, rng_den, config)
    dec_loss = flow_decoder_loss(model, embeddings, tokens, rng_dec, config)

    loss = (n_den * den_loss + n_dec * dec_loss) / B
    return loss, {"den_loss": den_loss, "dec_loss": dec_loss}


# Deprecated ELF-branded loss aliases (removed in v1.0).
elf_mse_loss      = flow_mse_loss
elf_ce_loss       = flow_ce_loss
elf_denoiser_loss = flow_denoiser_loss
elf_decoder_loss  = flow_decoder_loss
elf_loss          = flow_loss
