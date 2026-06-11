from __future__ import annotations

import optax
from flax import nnx

from dantinox.core.config import TrainingConfig
from dantinox.core.lora import LoRAParam


def build_schedule(config: TrainingConfig, total_steps: int) -> optax.Schedule:
    """Return the learning-rate schedule specified by *config.lr_schedule*.

    *total_steps* counts optimizer **updates** (i.e. micro-steps divided by
    ``grad_accum``), since gradient accumulation only advances the inner
    schedule once per accumulated update.

    Schedules
    ---------
    cosine   : linear warmup → cosine decay to 1 % of peak.
    linear   : linear warmup → linear decay to 1 % of peak.
    constant : linear warmup → flat plateau.
    wsd      : linear warmup → 40 % stable → cosine decay (Warm-Stable-Decay).
    """
    warmup = min(
        getattr(config, "warmup_steps", max(int(total_steps * 0.05), 1)),
        max(int(total_steps * 0.3), 1),
    )
    safe_total = max(total_steps, warmup + 1)
    peak = config.lr
    end  = peak * 0.01

    if config.lr_schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak,
            warmup_steps=warmup,
            decay_steps=safe_total,
            end_value=end,
        )

    if config.lr_schedule == "linear":
        up   = optax.linear_schedule(0.0, peak, warmup)
        down = optax.linear_schedule(peak, end, safe_total - warmup)
        return optax.join_schedules([up, down], [warmup])

    if config.lr_schedule == "constant":
        up   = optax.linear_schedule(0.0, peak, warmup)
        flat = optax.constant_schedule(peak)
        return optax.join_schedules([up, flat], [warmup])

    # wsd
    stable_steps = int(safe_total * 0.4)
    decay_steps  = max(safe_total - warmup - stable_steps, 1)
    up     = optax.linear_schedule(0.0, peak, warmup)
    stable = optax.constant_schedule(peak)
    down   = optax.cosine_decay_schedule(peak, decay_steps, alpha=end / peak)
    return optax.join_schedules(
        [up, stable, down],
        [warmup, warmup + stable_steps],
    )


def build_optimizer(
    model: nnx.Module,
    config: TrainingConfig,
    total_steps: int,
) -> nnx.Optimizer:
    """Build an ``nnx.Optimizer`` wrapping the requested optax transformation.

    *total_steps* counts optimizer updates (micro-steps / ``grad_accum``).

    When ``config.grad_accum > 1`` the transformation is wrapped in
    ``optax.MultiSteps`` so gradients are accumulated over that many
    micro-batches before each update.

    When LoRA is active only ``LoRAParam`` variables are updated; all other
    parameters are frozen by zeroing their gradients via a masked transform.

    Supported optimizers: ``adamw`` | ``adafactor`` | ``lion`` | ``adam`` | ``muon``.
    """
    schedule = build_schedule(config, total_steps)

    name = config.optimizer.lower()
    if name == "adamw":
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                         optax.adamw(schedule, weight_decay=0.1))
    elif name == "adafactor":
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                         optax.adafactor(learning_rate=schedule))
    elif name == "lion":
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                         optax.lion(schedule))
    elif name == "adam":
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                         optax.adam(schedule))
    elif name == "muon":
        # Muon handles clipping internally; skip the outer clip chain.
        tx = optax.contrib.muon(learning_rate=schedule)
    else:
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}. "
            "Choose from: adamw, adafactor, lion, adam, muon."
        )

    if _model_has_lora(model):
        tx = _lora_masked_optimizer(tx, model)

    if config.grad_accum > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum)

    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _model_has_lora(model: nnx.Module) -> bool:
    try:
        import jax
        state  = nnx.state(model, LoRAParam)
        leaves = jax.tree_util.tree_leaves(state)
        return len(leaves) > 0
    except Exception:
        return False


def _lora_masked_optimizer(
    tx: optax.GradientTransformation,
    model: nnx.Module,
) -> optax.GradientTransformation:
    """Wrap *tx* so that only LoRAParam variables receive non-zero updates."""

    def _label(path, _leaf):
        path_str = "/".join(str(p) for p in path)
        return "lora" if "lora" in path_str.lower() else "frozen"

    return optax.multi_transform(
        {"lora": tx, "frozen": optax.set_to_zero()},
        _label,
    )
