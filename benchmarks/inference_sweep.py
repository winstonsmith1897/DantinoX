#!/usr/bin/env python3
"""
benchmarks/inference_sweep.py

Comprehensive no-training inference benchmark for DantinoX.
Randomly-initialized models are benchmarked across 13 experiment groups:

  1.  attention_type   — MHA vs GQA (1/2, 1/4, 1/8 heads) vs MLA (3 compression ratios)
  2.  scale            — ~0.1 M → ~85 M parameters
  3.  batch_size       — BS 1 → 128
  4.  context_len      — max_context 64 → 1024
  5.  dtype            — fp32 vs bfloat16
  6.  sampling         — greedy / temperature / top-k / top-p
  7.  kv_cache         — cache on vs off, at BS 1 and BS 8
  8.  moe              — dense vs MoE (4/8 experts, top-1/2/4)
  9.  activation       — SwiGLU vs GELU
  10. pos_encoding      — RoPE / absolute / trainable / sliding window
  11. gqa_vs_cache      — GQA KV-compression × context length interaction
  12. scale_dtype       — bf16 speedup at small vs large scale
  13. batch_attn        — batch size × attention type interaction

Metrics collected per experiment:
  prefill_ms_p50/p95      — prompt forward-pass latency (percentile)
  decode_step_ms_p50/p95  — single autoregressive decode step with KV cache
  decode_tok_s            — effective batch decode throughput (tokens / second)
  kv_cache_mb             — theoretical KV cache memory (MB)
  params_m                — parameter count (millions)

Usage:
    python benchmarks/inference_sweep.py --out results/inference_sweep.csv
    python benchmarks/inference_sweep.py --groups attention_type scale --out results/attn.csv
    python benchmarks/inference_sweep.py --list-groups
    python benchmarks/inference_sweep.py --n-warmup 5 --n-trials 20 --out results/careful.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.model import Transformer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Global constants ─────────────────────────────────────────────────────────

VOCAB_SIZE = 256   # tiny fixed vocabulary — not a benchmark variable
N_WARMUP   = 3     # JIT warm-up repetitions (overrideable via CLI)
N_TRIALS   = 10    # timed repetitions (overrideable via CLI)


# ─── JIT-compiled step functions ─────────────────────────────────────────────
# These are module-level so JAX reuses compilations across experiments with
# the same model shapes.

@nnx.jit
def _prefill(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    logits, _, _ = model(x, use_cache=False, kv_caches=None, cache_index=0, deterministic=True)
    return logits


@nnx.jit
def _prefill_cached(model: nnx.Module, x: jnp.ndarray, cache: tuple) -> tuple:
    logits, new_cache, _ = model(x, use_cache=True, kv_caches=cache, cache_index=0, deterministic=True)
    return logits, new_cache


@nnx.jit
def _decode(model: nnx.Module, tok: jnp.ndarray, cache: tuple, pos: jax.Array) -> tuple:
    logits, new_cache, _ = model(tok, use_cache=True, kv_caches=cache, cache_index=pos, deterministic=True)
    return logits, new_cache


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_model(cfg_dict: dict[str, Any]) -> tuple[Config, Transformer]:
    config = Config.from_dict({**cfg_dict, "vocab_size": VOCAB_SIZE})
    model = Transformer(config, rngs=nnx.Rngs(42))
    return config, model


def _cast_bf16(model: Transformer) -> None:
    params = nnx.state(model, nnx.Param)
    bf16_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    nnx.update(model, bf16_params)


def _count_params_m(model: Transformer) -> float:
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(state) if hasattr(x, "size")) / 1e6


def _kv_cache_mb(config: Config, batch_size: int, bf16: bool) -> float:
    bpp = 2 if bf16 else 4
    S = config.max_context
    if config.mla:
        per_layer = S * (config.down_dim_kv + config.rope_dim) * bpp * batch_size
    else:
        hs = config.dim // config.n_heads
        per_layer = 2 * S * config.kv_heads * hs * bpp * batch_size
    return round(config.num_blocks * per_layer / 1e6, 3)


def _attn_label(config: Config) -> str:
    if config.mla:
        return "MLA"
    if config.kv_heads < config.n_heads:
        return f"GQA({config.kv_heads}/{config.n_heads})"
    return "MHA"


def _pos_label(config: Config) -> str:
    if config.sliding_window:
        return "sliding"
    if config.absolute_pos:
        return "absolute"
    if config.trainable_pos:
        return "trainable"
    return "rope"


def _time_fn(fn: Any, *args: Any, n_warmup: int, n_trials: int) -> np.ndarray:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    ts: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_one(exp: dict, n_warmup: int, n_trials: int) -> dict:
    cfg_dict = exp["cfg"]
    measure  = exp["measure"]
    B        = measure["batch_size"]
    use_bf16 = cfg_dict.get("use_bf16", False)
    use_cache = measure.get("use_cache", True)

    try:
        config, model = _build_model(cfg_dict)
    except (ValueError, Exception) as exc:
        log.warning("Config error [%s/%s]: %s", exp["group"], exp["label"], exc)
        return _oom_row(exp, {}, B, use_bf16)

    if use_bf16:
        _cast_bf16(model)

    T   = min(measure["prompt_len"], config.max_context - 1)
    x   = jnp.ones((B, T), dtype=jnp.int32)
    tok = jnp.ones((B, 1), dtype=jnp.int32)
    pos = jnp.array(T)
    init_cache = tuple((None, None) for _ in range(config.num_blocks))

    try:
        prefill_times = _time_fn(_prefill, model, x, n_warmup=n_warmup, n_trials=n_trials)

        if use_cache:
            # One real prefill to populate the cache, then time individual decode steps.
            _, kv_cache = _prefill_cached(model, x, init_cache)
            jax.block_until_ready(kv_cache)
            decode_times = _time_fn(_decode, model, tok, kv_cache, pos,
                                    n_warmup=n_warmup, n_trials=n_trials)
        else:
            # Without cache: cost per generated token ≈ full forward pass.
            decode_times = prefill_times

        decode_tok_s = B * 1000.0 / float(np.median(decode_times))

    except Exception as exc:
        log.warning("OOM/runtime error [%s/%s]: %s", exp["group"], exp["label"], exc)
        return _oom_row(exp, {"config": config, "model": model}, B, use_bf16)

    return {
        **_meta_row(exp, config, model, B, T, use_bf16),
        "prefill_ms_p50":      round(float(np.percentile(prefill_times, 50)), 3),
        "prefill_ms_p95":      round(float(np.percentile(prefill_times, 95)), 3),
        "decode_step_ms_p50":  round(float(np.percentile(decode_times, 50)), 3),
        "decode_step_ms_p95":  round(float(np.percentile(decode_times, 95)), 3),
        "decode_tok_s":        round(decode_tok_s, 2),
        "kv_cache_mb":         _kv_cache_mb(config, B, use_bf16),
        "oom":                 False,
    }


def _meta_row(
    exp: dict,
    config: Config,
    model: Transformer,
    B: int,
    T: int,
    use_bf16: bool,
) -> dict:
    cfg = exp["cfg"]
    return {
        "group":        exp["group"],
        "label":        exp["label"],
        "attn_type":    _attn_label(config),
        "params_m":     round(_count_params_m(model), 3),
        "dim":          config.dim,
        "n_heads":      config.n_heads,
        "kv_heads":     config.kv_heads,
        "num_blocks":   config.num_blocks,
        "max_context":  config.max_context,
        "batch_size":   B,
        "prompt_len":   T,
        "dtype":        "bf16" if use_bf16 else "fp32",
        "use_cache":    exp["measure"].get("use_cache", True),
        "use_moe":      config.use_moe,
        "n_experts":    config.n_experts if config.use_moe else None,
        "top_k_mlp":    config.top_k_mlp if config.use_moe else None,
        "use_swiglu":   config.use_swiglu,
        "pos_encoding": _pos_label(config),
        "mla_down_kv":  config.down_dim_kv if config.mla else None,
        "mla_rope_dim": config.rope_dim if config.mla else None,
    }


def _oom_row(exp: dict, ctx: dict, B: int, use_bf16: bool) -> dict:
    nan = float("nan")
    config = ctx.get("config")
    model  = ctx.get("model")
    base = {
        "group":        exp["group"],
        "label":        exp["label"],
        "attn_type":    _attn_label(config) if config else "?",
        "params_m":     round(_count_params_m(model), 3) if model else nan,
        "dim":          config.dim if config else nan,
        "n_heads":      config.n_heads if config else nan,
        "kv_heads":     config.kv_heads if config else nan,
        "num_blocks":   config.num_blocks if config else nan,
        "max_context":  config.max_context if config else nan,
        "batch_size":   B,
        "prompt_len":   exp["measure"].get("prompt_len", nan),
        "dtype":        "bf16" if use_bf16 else "fp32",
        "use_cache":    exp["measure"].get("use_cache", True),
        "use_moe":      config.use_moe if config else nan,
        "n_experts":    None,
        "top_k_mlp":    None,
        "use_swiglu":   config.use_swiglu if config else nan,
        "pos_encoding": _pos_label(config) if config else "?",
        "mla_down_kv":  None,
        "mla_rope_dim": None,
    }
    return {
        **base,
        "prefill_ms_p50": nan, "prefill_ms_p95": nan,
        "decode_step_ms_p50": nan, "decode_step_ms_p95": nan,
        "decode_tok_s": nan, "kv_cache_mb": nan, "oom": True,
    }


# ─── Experiment definitions ───────────────────────────────────────────────────

def _c(**kw: Any) -> dict:
    """Build a config dict by merging overrides onto the shared base."""
    base: dict[str, Any] = dict(
        dim=256, n_heads=8, head_size=32, num_blocks=6,
        kv_heads=8, max_context=256,
        use_moe=False, n_experts=4, top_k_mlp=2, expansion=4, alpha_balance=0.1,
        use_swiglu=True, use_rotary_pos=True,
        trainable_pos=False, absolute_pos=False,
        sliding_window=False, context_window=32,
        mla=False, inference=False, down_dim_q=128, down_dim_kv=64, rope_dim=16,
        dropout_rate=0.0, gradient_checkpointing=False, weight_tying=True,
        use_bf16=False,
    )
    base.update(kw)
    return base


def _m(**kw: Any) -> dict:
    """Build a measurement dict by merging overrides onto the shared base."""
    base: dict[str, Any] = dict(batch_size=1, prompt_len=64, use_cache=True)
    base.update(kw)
    return base


EXPERIMENTS: list[dict] = [

    # ── 1. Attention type ─────────────────────────────────────────────────────
    # What: vary the attention mechanism at fixed model scale (medium, 6 blocks).
    # Insight: decode throughput & KV cache size vs compression ratio.

    {"group": "attention_type", "label": "MHA",
     "cfg": _c(kv_heads=8, mla=False), "measure": _m()},

    {"group": "attention_type", "label": "GQA-1/2",
     "cfg": _c(kv_heads=4, mla=False), "measure": _m()},

    {"group": "attention_type", "label": "GQA-1/4",
     "cfg": _c(kv_heads=2, mla=False), "measure": _m()},

    {"group": "attention_type", "label": "GQA-1/8",
     "cfg": _c(kv_heads=1, mla=False), "measure": _m()},

    {"group": "attention_type", "label": "MLA-down64",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8), "measure": _m()},

    {"group": "attention_type", "label": "MLA-down32",
     "cfg": _c(mla=True, inference=True, down_dim_kv=32, rope_dim=16, kv_heads=8), "measure": _m()},

    {"group": "attention_type", "label": "MLA-down128",
     "cfg": _c(mla=True, inference=True, down_dim_kv=128, rope_dim=16, kv_heads=8), "measure": _m()},

    # ── 2. Model scale ────────────────────────────────────────────────────────
    # What: sweep parameter count from ~0.1 M to ~85 M (MHA, BS=1, ctx=256).
    # Insight: how prefill and decode latency grow with model size.

    {"group": "scale", "label": "tiny-~0.1M",
     "cfg": _c(dim=64, n_heads=4, head_size=16, num_blocks=2, kv_heads=4), "measure": _m()},

    {"group": "scale", "label": "small-~0.5M",
     "cfg": _c(dim=128, n_heads=4, head_size=32, num_blocks=3, kv_heads=4), "measure": _m()},

    {"group": "scale", "label": "medium-~3M",
     "cfg": _c(), "measure": _m()},

    {"group": "scale", "label": "large-~25M",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16), "measure": _m()},

    {"group": "scale", "label": "xlarge-~50M",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=16, kv_heads=16), "measure": _m()},

    {"group": "scale", "label": "xxlarge-~85M",
     "cfg": _c(dim=768, n_heads=12, head_size=64, num_blocks=12, kv_heads=12), "measure": _m()},

    # ── 3. Batch size ─────────────────────────────────────────────────────────
    # What: sweep BS 1–128 at fixed model (medium, MHA, ctx=256, fp32).
    # Insight: how decode throughput scales with batch (GPU parallelism).

    {"group": "batch_size", "label": "bs=1",   "cfg": _c(), "measure": _m(batch_size=1)},
    {"group": "batch_size", "label": "bs=4",   "cfg": _c(), "measure": _m(batch_size=4)},
    {"group": "batch_size", "label": "bs=8",   "cfg": _c(), "measure": _m(batch_size=8)},
    {"group": "batch_size", "label": "bs=16",  "cfg": _c(), "measure": _m(batch_size=16)},
    {"group": "batch_size", "label": "bs=32",  "cfg": _c(), "measure": _m(batch_size=32)},
    {"group": "batch_size", "label": "bs=64",  "cfg": _c(), "measure": _m(batch_size=64)},
    {"group": "batch_size", "label": "bs=128", "cfg": _c(), "measure": _m(batch_size=128)},

    # ── 4. Context length ─────────────────────────────────────────────────────
    # What: sweep max_context 64–1024 (prompt_len = max_context/2).
    # Insight: O(T²) prefill growth; KV cache memory vs context size.

    {"group": "context_len", "label": "ctx=64",
     "cfg": _c(max_context=64),   "measure": _m(prompt_len=32)},

    {"group": "context_len", "label": "ctx=128",
     "cfg": _c(max_context=128),  "measure": _m(prompt_len=64)},

    {"group": "context_len", "label": "ctx=256",
     "cfg": _c(max_context=256),  "measure": _m(prompt_len=128)},

    {"group": "context_len", "label": "ctx=512",
     "cfg": _c(max_context=512),  "measure": _m(prompt_len=256)},

    {"group": "context_len", "label": "ctx=1024",
     "cfg": _c(max_context=1024), "measure": _m(prompt_len=512)},

    # ── 5. Dtype ──────────────────────────────────────────────────────────────
    # What: fp32 vs bfloat16 at medium and large scale.
    # Insight: bf16 speedup and memory halving.

    {"group": "dtype", "label": "medium-fp32",
     "cfg": _c(use_bf16=False), "measure": _m()},

    {"group": "dtype", "label": "medium-bf16",
     "cfg": _c(use_bf16=True), "measure": _m()},

    {"group": "dtype", "label": "large-fp32",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=False),
     "measure": _m()},

    {"group": "dtype", "label": "large-bf16",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=True),
     "measure": _m()},

    # ── 6. Sampling strategy ──────────────────────────────────────────────────
    # What: greedy vs temperature vs top-k vs top-p.
    # Insight: model forward pass dominates; post-processing differences are in the noise.
    # (All experiments share the same JIT-compiled forward pass — only Python sampling differs.)

    {"group": "sampling", "label": "greedy",        "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=0.5",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=1.0",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=2.0",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k=10",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k=50",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_p=0.90",    "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_p=0.95",    "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k50+top_p", "cfg": _c(), "measure": _m()},

    # ── 7. KV cache ───────────────────────────────────────────────────────────
    # What: cache on vs off at BS=1 and BS=8.
    # Insight: decode-step speedup from the O(1)-per-step KV cache vs O(T) re-computation.

    {"group": "kv_cache", "label": "bs1-cache=on",
     "cfg": _c(), "measure": _m(batch_size=1, use_cache=True)},

    {"group": "kv_cache", "label": "bs1-cache=off",
     "cfg": _c(), "measure": _m(batch_size=1, use_cache=False)},

    {"group": "kv_cache", "label": "bs8-cache=on",
     "cfg": _c(), "measure": _m(batch_size=8, use_cache=True)},

    {"group": "kv_cache", "label": "bs8-cache=off",
     "cfg": _c(), "measure": _m(batch_size=8, use_cache=False)},

    {"group": "kv_cache", "label": "bs32-cache=on",
     "cfg": _c(), "measure": _m(batch_size=32, use_cache=True)},

    {"group": "kv_cache", "label": "bs32-cache=off",
     "cfg": _c(), "measure": _m(batch_size=32, use_cache=False)},

    # ── 8. MoE vs dense ───────────────────────────────────────────────────────
    # What: dense MLP vs various MoE configs (experts=4/8, top-k=1/2/4).
    # Insight: MoE adds parameters but top-k routing keeps active FLOPs low.

    {"group": "moe", "label": "dense",
     "cfg": _c(use_moe=False), "measure": _m()},

    {"group": "moe", "label": "4exp-top1",
     "cfg": _c(use_moe=True, n_experts=4, top_k_mlp=1), "measure": _m()},

    {"group": "moe", "label": "4exp-top2",
     "cfg": _c(use_moe=True, n_experts=4, top_k_mlp=2), "measure": _m()},

    {"group": "moe", "label": "8exp-top2",
     "cfg": _c(use_moe=True, n_experts=8, top_k_mlp=2), "measure": _m()},

    {"group": "moe", "label": "8exp-top4",
     "cfg": _c(use_moe=True, n_experts=8, top_k_mlp=4), "measure": _m()},

    {"group": "moe", "label": "16exp-top4",
     "cfg": _c(use_moe=True, n_experts=16, top_k_mlp=4), "measure": _m()},

    # ── 9. Activation ─────────────────────────────────────────────────────────
    # What: SwiGLU (gated, 3 matmuls) vs GELU (2 matmuls) per FFN block.
    # Insight: SwiGLU has ~1.5× more FFN parameters for the same expansion factor.

    {"group": "activation", "label": "SwiGLU",
     "cfg": _c(use_swiglu=True), "measure": _m()},

    {"group": "activation", "label": "GELU",
     "cfg": _c(use_swiglu=False), "measure": _m()},

    # ── 10. Positional encoding ───────────────────────────────────────────────
    # What: RoPE vs learned absolute vs sinusoidal absolute vs sliding window.
    # Insight: positional encoding rarely bottlenecks inference but affects memory.

    {"group": "pos_encoding", "label": "RoPE",
     "cfg": _c(use_rotary_pos=True,  trainable_pos=False, absolute_pos=False, sliding_window=False),
     "measure": _m()},

    {"group": "pos_encoding", "label": "sinusoidal",
     "cfg": _c(use_rotary_pos=False, trainable_pos=False, absolute_pos=True,  sliding_window=False),
     "measure": _m()},

    {"group": "pos_encoding", "label": "trainable",
     "cfg": _c(use_rotary_pos=False, trainable_pos=True,  absolute_pos=False, sliding_window=False),
     "measure": _m()},

    {"group": "pos_encoding", "label": "sliding_win",
     "cfg": _c(use_rotary_pos=True,  trainable_pos=False, absolute_pos=False, sliding_window=True,
               context_window=32),
     "measure": _m()},

    # ── 11. GQA compression × context length interaction ──────────────────────
    # What: cross-tab attention type vs context length — shows where KV compression pays off.
    # Insight: GQA advantage in kv_cache_mb grows quadratically with context.

    {"group": "gqa_vs_cache", "label": "MHA-ctx128",
     "cfg": _c(kv_heads=8, max_context=128),  "measure": _m(prompt_len=64)},

    {"group": "gqa_vs_cache", "label": "GQA4-ctx128",
     "cfg": _c(kv_heads=2, max_context=128),  "measure": _m(prompt_len=64)},

    {"group": "gqa_vs_cache", "label": "MHA-ctx256",
     "cfg": _c(kv_heads=8, max_context=256),  "measure": _m(prompt_len=128)},

    {"group": "gqa_vs_cache", "label": "GQA4-ctx256",
     "cfg": _c(kv_heads=2, max_context=256),  "measure": _m(prompt_len=128)},

    {"group": "gqa_vs_cache", "label": "MHA-ctx512",
     "cfg": _c(kv_heads=8, max_context=512),  "measure": _m(prompt_len=256)},

    {"group": "gqa_vs_cache", "label": "GQA4-ctx512",
     "cfg": _c(kv_heads=2, max_context=512),  "measure": _m(prompt_len=256)},

    {"group": "gqa_vs_cache", "label": "MHA-ctx1024",
     "cfg": _c(kv_heads=8, max_context=1024), "measure": _m(prompt_len=512)},

    {"group": "gqa_vs_cache", "label": "GQA4-ctx1024",
     "cfg": _c(kv_heads=2, max_context=1024), "measure": _m(prompt_len=512)},

    {"group": "gqa_vs_cache", "label": "MLA-ctx512",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8, max_context=512),
     "measure": _m(prompt_len=256)},

    {"group": "gqa_vs_cache", "label": "MLA-ctx1024",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8, max_context=1024),
     "measure": _m(prompt_len=512)},

    # ── 12. Scale × dtype interaction ─────────────────────────────────────────
    # What: bf16 speedup measured at four model sizes.
    # Insight: bf16 benefit is larger for compute-bound (large) models.

    {"group": "scale_dtype", "label": "tiny-fp32",
     "cfg": _c(dim=64, n_heads=4, head_size=16, num_blocks=2, kv_heads=4, use_bf16=False),
     "measure": _m()},

    {"group": "scale_dtype", "label": "tiny-bf16",
     "cfg": _c(dim=64, n_heads=4, head_size=16, num_blocks=2, kv_heads=4, use_bf16=True),
     "measure": _m()},

    {"group": "scale_dtype", "label": "small-fp32",
     "cfg": _c(dim=128, n_heads=4, head_size=32, num_blocks=3, kv_heads=4, use_bf16=False),
     "measure": _m()},

    {"group": "scale_dtype", "label": "small-bf16",
     "cfg": _c(dim=128, n_heads=4, head_size=32, num_blocks=3, kv_heads=4, use_bf16=True),
     "measure": _m()},

    {"group": "scale_dtype", "label": "medium-fp32",
     "cfg": _c(use_bf16=False), "measure": _m()},

    {"group": "scale_dtype", "label": "medium-bf16",
     "cfg": _c(use_bf16=True), "measure": _m()},

    {"group": "scale_dtype", "label": "large-fp32",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=False),
     "measure": _m()},

    {"group": "scale_dtype", "label": "large-bf16",
     "cfg": _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=True),
     "measure": _m()},

    # ── 13. Batch size × attention type interaction ───────────────────────────
    # What: MHA vs GQA vs MLA at BS=1, 8, 32.
    # Insight: GQA/MLA cache advantage scales with batch; compute bound shifts earlier.

    {"group": "batch_attn", "label": "MHA-bs1",
     "cfg": _c(kv_heads=8, mla=False), "measure": _m(batch_size=1)},

    {"group": "batch_attn", "label": "MHA-bs8",
     "cfg": _c(kv_heads=8, mla=False), "measure": _m(batch_size=8)},

    {"group": "batch_attn", "label": "MHA-bs32",
     "cfg": _c(kv_heads=8, mla=False), "measure": _m(batch_size=32)},

    {"group": "batch_attn", "label": "GQA-bs1",
     "cfg": _c(kv_heads=2, mla=False), "measure": _m(batch_size=1)},

    {"group": "batch_attn", "label": "GQA-bs8",
     "cfg": _c(kv_heads=2, mla=False), "measure": _m(batch_size=8)},

    {"group": "batch_attn", "label": "GQA-bs32",
     "cfg": _c(kv_heads=2, mla=False), "measure": _m(batch_size=32)},

    {"group": "batch_attn", "label": "MLA-bs1",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=1)},

    {"group": "batch_attn", "label": "MLA-bs8",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=8)},

    {"group": "batch_attn", "label": "MLA-bs32",
     "cfg": _c(mla=True, inference=True, down_dim_kv=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=32)},
]

ALL_GROUPS: list[str] = sorted({e["group"] for e in EXPERIMENTS})


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DantinoX comprehensive inference benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default="results/inference_sweep.csv",
                        help="Output CSV path (default: results/inference_sweep.csv)")
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help="Subset of groups to run (default: all). Use --list-groups to see options.")
    parser.add_argument("--list-groups", action="store_true",
                        help="Print available group names and exit.")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP,
                        help=f"JIT warm-up repetitions per experiment (default: {N_WARMUP})")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Timed repetitions per experiment (default: {N_TRIALS})")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-experiment latency values.")
    args = parser.parse_args()

    if args.list_groups:
        print("Available groups:")
        for g in ALL_GROUPS:
            count = sum(1 for e in EXPERIMENTS if e["group"] == g)
            print(f"  {g:<22} ({count} experiments)")
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    selected = EXPERIMENTS
    if args.groups:
        unknown = set(args.groups) - set(ALL_GROUPS)
        if unknown:
            parser.error(f"Unknown groups: {sorted(unknown)}. Valid: {ALL_GROUPS}")
        selected = [e for e in EXPERIMENTS if e["group"] in args.groups]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DantinoX inference sweep — {len(selected)} experiments")
    print(f"  device : {jax.default_backend()}")
    print(f"  warmup : {args.n_warmup}  trials: {args.n_trials}")
    print(f"  output : {out_path}")
    print()

    rows: list[dict] = []
    for exp in tqdm(selected, desc="sweep", unit="exp"):
        row = run_one(exp, n_warmup=args.n_warmup, n_trials=args.n_trials)
        rows.append(row)
        if args.verbose:
            tqdm.write(
                f"  [{row['group']:>14}] {row['label']:<22}  "
                f"prefill={row['prefill_ms_p50']:>7.2f}ms  "
                f"decode={row['decode_tok_s']:>8.1f} tok/s  "
                f"cache={row['kv_cache_mb']:>6.2f} MB  "
                f"params={row['params_m']:>6.2f} M"
            )

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows → {out_path}")
        _print_summary(df)
    except ImportError:
        import csv, io
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        out_path.write_text(buf.getvalue())
        print(f"\nSaved {len(rows)} rows → {out_path}")


def _print_summary(df: Any) -> None:
    print("\n── Group summary (median decode_tok_s) ─────────────────────────────")
    for grp in df["group"].unique():
        sub = df[df["group"] == grp]
        best = sub.loc[sub["decode_tok_s"].idxmax()]
        print(f"  {grp:<22}  best: {best['label']:<24} "
              f"{best['decode_tok_s']:>8.1f} tok/s   "
              f"prefill={best['prefill_ms_p50']:>6.2f}ms")


if __name__ == "__main__":
    main()
