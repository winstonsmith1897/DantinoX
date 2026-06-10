#!/usr/bin/env python3
"""
benchmarks/inference_sweep.py

Comprehensive no-training inference benchmark for DantinoX.
Randomly-initialized models are benchmarked across 13 experiment groups.
Every group that does not already vary attention type is expanded into three
parallel runs — MHA, GQA-1/4, and MLA — so every plot can split results
by attention mechanism.

Groups:
  1.  attention_type   — MHA vs GQA (1/2, 1/4, 1/8 heads) vs MLA (3 compression ratios)
  2.  scale            — ~0.1 M → ~85 M parameters  × {MHA, GQA, MLA}
  3.  batch_size       — BS 1 → 128  × {MHA, GQA, MLA}
  4.  context_len      — max_context 64 → 1024  × {MHA, GQA, MLA}
  5.  dtype            — fp32 vs bfloat16  × {MHA, GQA, MLA}
  6.  sampling         — greedy / temperature / top-k / top-p (forward pass is attn-agnostic)
  7.  kv_cache         — cache on vs off  × {MHA, GQA, MLA}
  8.  moe              — dense vs MoE variants  × {MHA, GQA, MLA}
  9.  activation       — SwiGLU vs GELU  × {MHA, GQA, MLA}
  10. pos_encoding      — RoPE / absolute / trainable / sliding  × {MHA, GQA, MLA}
  11. gqa_vs_cache      — attention type × context length (already cross-cuts attention)
  12. scale_dtype       — bf16 speedup at different scales (already cross-cuts attention)
  13. batch_attn        — batch size × attention type (already cross-cuts attention)

CSV columns include `attn_variant` ("MHA" | "GQA" | "MLA") for easy grouping in plots.

Metrics per experiment:
  prefill_ms_p50/p95      — prompt forward-pass latency
  decode_step_ms_p50/p95  — single decode step latency (with KV cache)
  decode_tok_s            — batch decode throughput (tokens / second)
  kv_cache_mb             — theoretical KV cache memory (MB)
  params_m                — parameter count (millions)

Usage:
    python benchmarks/inference_sweep.py --out results/inference_sweep.csv
    python benchmarks/inference_sweep.py --groups scale batch_size --out results/ab.csv
    python benchmarks/inference_sweep.py --list-groups
    python benchmarks/inference_sweep.py --n-warmup 5 --n-trials 20 --out results/careful.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # override with --device or env var

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

# ─── Persistent XLA compilation cache ────────────────────────────────────────
# Saves compiled GPU kernels to disk. The first run per unique model config
# still compiles (seconds); subsequent runs load from cache in <100 ms.
_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_bench"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

# ─── Constants ────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256
N_WARMUP   = 3
N_TRIALS   = 10

# GQA always uses 1/4 of n_heads as kv_heads (at least 1)
_GQA_RATIO = 4


# ─── JIT step functions ───────────────────────────────────────────────────────

@nnx.jit
def _prefill(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    logits, _, _ = model(x, deterministic=True)
    return logits


@nnx.jit
def _prefill_cached(model: nnx.Module, x: jnp.ndarray, cache: tuple) -> tuple:
    logits, new_cache, _ = model(x, caches=cache, cache_index=0, deterministic=True)
    return logits, new_cache


@nnx.jit
def _decode(model: nnx.Module, tok: jnp.ndarray, cache: tuple, pos: jax.Array) -> tuple:
    logits, new_cache, _ = model(tok, caches=cache, cache_index=pos, deterministic=True)
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


def _attn_variant(exp: dict, config: Config) -> str:
    """Normalise attn_type to one of MHA / GQA / MLA for grouping in plots."""
    if "attn_variant" in exp:
        return exp["attn_variant"]
    lbl = _attn_label(config)
    if lbl == "MLA":
        return "MLA"
    if "GQA" in lbl:
        return "GQA"
    return "MHA"


def _time_fn(fn: Any, *args: Any, n_warmup: int, n_trials: int, desc: str = "") -> np.ndarray:
    # First call triggers JIT compilation — measure and report it separately.
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    if compile_s > 1.0:
        tqdm.write(f"    compile  {desc:<45} {compile_s:5.1f}s")
    # Remaining warm-up (kernels already compiled, this is steady-state warm-up)
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(fn(*args))
    # Timed trials
    ts: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_one(exp: dict, n_warmup: int, n_trials: int) -> dict:
    cfg_dict  = exp["cfg"]
    measure   = exp["measure"]
    B         = measure["batch_size"]
    use_bf16  = cfg_dict.get("use_bf16", False)
    use_cache = measure.get("use_cache", True)

    try:
        config, model = _build_model(cfg_dict)
    except (ValueError, Exception) as exc:
        log.warning("Config error [%s/%s]: %s", exp["group"], exp["label"], exc)
        return _oom_row(exp, {}, B, use_bf16)

    if use_bf16:
        _cast_bf16(model)

    T        = min(measure["prompt_len"], config.max_context - 1)
    x        = jnp.ones((B, T), dtype=jnp.int32)
    tok      = jnp.ones((B, 1), dtype=jnp.int32)
    pos      = jnp.array(T)
    init_cache = tuple((None, None) for _ in range(config.num_blocks))

    tag = f"{exp['group']}/{exp['label']}[{exp.get('attn_variant', '?')}]"

    try:
        prefill_times = _time_fn(_prefill, model, x,
                                 n_warmup=n_warmup, n_trials=n_trials,
                                 desc=f"prefill  {tag}")

        if use_cache:
            _, kv_cache = _prefill_cached(model, x, init_cache)
            jax.block_until_ready(kv_cache)
            decode_times = _time_fn(_decode, model, tok, kv_cache, pos,
                                    n_warmup=n_warmup, n_trials=n_trials,
                                    desc=f"decode   {tag}")
        else:
            decode_times = prefill_times   # no-cache: full pass per token

        decode_tok_s = B * 1000.0 / float(np.median(decode_times))

    except Exception as exc:
        log.warning("OOM/runtime [%s/%s]: %s", exp["group"], exp["label"], exc)
        return _oom_row(exp, {"config": config, "model": model}, B, use_bf16)

    return {
        **_meta_row(exp, config, model, B, T, use_bf16),
        "prefill_ms_p50":     round(float(np.percentile(prefill_times, 50)), 3),
        "prefill_ms_p95":     round(float(np.percentile(prefill_times, 95)), 3),
        "decode_step_ms_p50": round(float(np.percentile(decode_times, 50)), 3),
        "decode_step_ms_p95": round(float(np.percentile(decode_times, 95)), 3),
        "decode_tok_s":       round(decode_tok_s, 2),
        "kv_cache_mb":        _kv_cache_mb(config, B, use_bf16),
        "oom":                False,
    }


def _meta_row(exp: dict, config: Config, model: Transformer, B: int, T: int, use_bf16: bool) -> dict:
    return {
        "group":        exp["group"],
        "label":        exp["label"],
        "attn_variant": _attn_variant(exp, config),
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
    nan    = float("nan")
    config = ctx.get("config")
    model  = ctx.get("model")
    base   = {
        "group":        exp["group"],
        "label":        exp["label"],
        "attn_variant": exp.get("attn_variant", "?"),
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
    return {**base,
            "prefill_ms_p50": nan, "prefill_ms_p95": nan,
            "decode_step_ms_p50": nan, "decode_step_ms_p95": nan,
            "decode_tok_s": nan, "kv_cache_mb": nan, "oom": True}


# ─── Experiment factory ───────────────────────────────────────────────────────

def _c(**kw: Any) -> dict:
    """Config dict merged onto the shared base."""
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
    """Measurement dict merged onto the shared base."""
    base: dict[str, Any] = dict(batch_size=1, prompt_len=64, use_cache=True)
    base.update(kw)
    return base


def _by_attn(group: str, label: str, cfg: dict, measure: dict) -> list[dict]:
    """
    Expand one logical experiment into three parallel runs:
    MHA, GQA-1/4, and MLA — each tagged with attn_variant.

    GQA uses n_heads // 4 kv_heads (minimum 1).
    MLA uses rope_dim = min(16, head_size), down_dim_kv = min(64, head_size*2).
    """
    n_heads   = cfg.get("n_heads", 8)
    head_size = cfg.get("head_size", 32)
    gqa_kv    = max(1, n_heads // _GQA_RATIO)
    rope_dim  = min(16, head_size)
    mla_dkv   = min(64, head_size * 2)
    mla_dq    = min(64, head_size * 2)

    return [
        {"group": group, "label": label, "attn_variant": "MHA",
         "cfg": {**cfg, "kv_heads": n_heads, "mla": False, "inference": False},
         "measure": measure},

        {"group": group, "label": label, "attn_variant": "GQA",
         "cfg": {**cfg, "kv_heads": gqa_kv, "mla": False, "inference": False},
         "measure": measure},

        {"group": group, "label": label, "attn_variant": "MLA",
         "cfg": {**cfg, "kv_heads": n_heads, "mla": True, "inference": True,
                 "down_dim_kv": mla_dkv, "down_dim_q": mla_dq, "rope_dim": rope_dim},
         "measure": measure},
    ]


# ─── Experiment definitions ───────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [

    # ── 1. Attention type ─────────────────────────────────────────────────────
    # Sweeps the attention mechanism directly — no _by_attn expansion.

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

    # ── 2. Model scale  ×  {MHA, GQA, MLA} ───────────────────────────────────
    # Insight: latency/throughput growth with parameter count per attention type.

    *_by_attn("scale", "tiny-~0.1M",
               _c(dim=64,  n_heads=4,  head_size=16, num_blocks=2,  kv_heads=4),  _m()),
    *_by_attn("scale", "small-~0.5M",
               _c(dim=128, n_heads=4,  head_size=32, num_blocks=3,  kv_heads=4),  _m()),
    *_by_attn("scale", "medium-~3M",   _c(),                                        _m()),
    *_by_attn("scale", "large-~25M",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=8,  kv_heads=16), _m()),
    *_by_attn("scale", "xlarge-~50M",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=16, kv_heads=16), _m()),
    *_by_attn("scale", "xxlarge-~85M",
               _c(dim=768, n_heads=12, head_size=64, num_blocks=12, kv_heads=12), _m()),

    # ── 3. Batch size  ×  {MHA, GQA, MLA} ────────────────────────────────────
    # Insight: how decode throughput scales with batch for each attention type.

    *_by_attn("batch_size", "bs=1",   _c(), _m(batch_size=1)),
    *_by_attn("batch_size", "bs=4",   _c(), _m(batch_size=4)),
    *_by_attn("batch_size", "bs=8",   _c(), _m(batch_size=8)),
    *_by_attn("batch_size", "bs=16",  _c(), _m(batch_size=16)),
    *_by_attn("batch_size", "bs=32",  _c(), _m(batch_size=32)),
    *_by_attn("batch_size", "bs=64",  _c(), _m(batch_size=64)),
    *_by_attn("batch_size", "bs=128", _c(), _m(batch_size=128)),

    # ── 4. Context length  ×  {MHA, GQA, MLA} ────────────────────────────────
    # Insight: O(T²) prefill and linear KV cache growth per attention type.

    *_by_attn("context_len", "ctx=64",   _c(max_context=64),   _m(prompt_len=32)),
    *_by_attn("context_len", "ctx=128",  _c(max_context=128),  _m(prompt_len=64)),
    *_by_attn("context_len", "ctx=256",  _c(max_context=256),  _m(prompt_len=128)),
    *_by_attn("context_len", "ctx=512",  _c(max_context=512),  _m(prompt_len=256)),
    *_by_attn("context_len", "ctx=1024", _c(max_context=1024), _m(prompt_len=512)),

    # ── 5. Dtype  ×  {MHA, GQA, MLA} ─────────────────────────────────────────
    # Insight: bf16 speedup and memory saving per attention type and model size.

    *_by_attn("dtype", "medium-fp32", _c(use_bf16=False), _m()),
    *_by_attn("dtype", "medium-bf16", _c(use_bf16=True),  _m()),
    *_by_attn("dtype", "large-fp32",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=False), _m()),
    *_by_attn("dtype", "large-bf16",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=True),  _m()),

    # ── 6. Sampling strategy (forward pass is attention-agnostic) ─────────────
    # Insight: model forward pass dominates; post-processing cost is negligible.
    # Not expanded by attention type since the JIT-compiled forward pass is identical.

    {"group": "sampling", "label": "greedy",        "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=0.5",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=1.0",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "temp=2.0",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k=10",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k=50",      "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_p=0.90",    "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_p=0.95",    "cfg": _c(), "measure": _m()},
    {"group": "sampling", "label": "top_k50+top_p", "cfg": _c(), "measure": _m()},

    # ── 7. KV cache  ×  {MHA, GQA, MLA} ─────────────────────────────────────
    # Insight: decode speedup from cache differs by attention — GQA/MLA write less.

    *_by_attn("kv_cache", "bs1-cache=on",  _c(), _m(batch_size=1,  use_cache=True)),
    *_by_attn("kv_cache", "bs1-cache=off", _c(), _m(batch_size=1,  use_cache=False)),
    *_by_attn("kv_cache", "bs8-cache=on",  _c(), _m(batch_size=8,  use_cache=True)),
    *_by_attn("kv_cache", "bs8-cache=off", _c(), _m(batch_size=8,  use_cache=False)),
    *_by_attn("kv_cache", "bs32-cache=on", _c(), _m(batch_size=32, use_cache=True)),
    *_by_attn("kv_cache", "bs32-cache=off",_c(), _m(batch_size=32, use_cache=False)),

    # ── 8. MoE vs dense  ×  {MHA, GQA, MLA} ─────────────────────────────────
    # Insight: MoE adds parameters while top-k routing keeps active FLOPs low.

    *_by_attn("moe", "dense",       _c(use_moe=False),                           _m()),
    *_by_attn("moe", "4exp-top1",   _c(use_moe=True, n_experts=4,  top_k_mlp=1), _m()),
    *_by_attn("moe", "4exp-top2",   _c(use_moe=True, n_experts=4,  top_k_mlp=2), _m()),
    *_by_attn("moe", "8exp-top2",   _c(use_moe=True, n_experts=8,  top_k_mlp=2), _m()),
    *_by_attn("moe", "8exp-top4",   _c(use_moe=True, n_experts=8,  top_k_mlp=4), _m()),
    *_by_attn("moe", "16exp-top4",  _c(use_moe=True, n_experts=16, top_k_mlp=4), _m()),

    # ── 9. Activation  ×  {MHA, GQA, MLA} ────────────────────────────────────
    # Insight: SwiGLU has ~1.5× FFN params for the same expansion factor.

    *_by_attn("activation", "SwiGLU", _c(use_swiglu=True),  _m()),
    *_by_attn("activation", "GELU",   _c(use_swiglu=False), _m()),

    # ── 10. Positional encoding  ×  {MHA, GQA, MLA} ──────────────────────────
    # Insight: pos encoding rarely bottlenecks inference — confirms attn type dominates.

    *_by_attn("pos_encoding", "RoPE",
               _c(use_rotary_pos=True,  trainable_pos=False, absolute_pos=False, sliding_window=False), _m()),
    *_by_attn("pos_encoding", "sinusoidal",
               _c(use_rotary_pos=False, trainable_pos=False, absolute_pos=True,  sliding_window=False), _m()),
    *_by_attn("pos_encoding", "trainable",
               _c(use_rotary_pos=False, trainable_pos=True,  absolute_pos=False, sliding_window=False), _m()),
    *_by_attn("pos_encoding", "sliding_win",
               _c(use_rotary_pos=True,  trainable_pos=False, absolute_pos=False,
                  sliding_window=True, context_window=32), _m()),

    # ── 11. GQA compression × context length (already cross-cuts attention) ───
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

    # ── 12. Scale × dtype (already cross-cuts attention via _by_attn) ─────────
    *_by_attn("scale_dtype", "tiny-fp32",
               _c(dim=64,  n_heads=4,  head_size=16, num_blocks=2,  kv_heads=4,  use_bf16=False), _m()),
    *_by_attn("scale_dtype", "tiny-bf16",
               _c(dim=64,  n_heads=4,  head_size=16, num_blocks=2,  kv_heads=4,  use_bf16=True),  _m()),
    *_by_attn("scale_dtype", "small-fp32",
               _c(dim=128, n_heads=4,  head_size=32, num_blocks=3,  kv_heads=4,  use_bf16=False), _m()),
    *_by_attn("scale_dtype", "small-bf16",
               _c(dim=128, n_heads=4,  head_size=32, num_blocks=3,  kv_heads=4,  use_bf16=True),  _m()),
    *_by_attn("scale_dtype", "medium-fp32", _c(use_bf16=False), _m()),
    *_by_attn("scale_dtype", "medium-bf16", _c(use_bf16=True),  _m()),
    *_by_attn("scale_dtype", "large-fp32",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=8,  kv_heads=16, use_bf16=False), _m()),
    *_by_attn("scale_dtype", "large-bf16",
               _c(dim=512, n_heads=16, head_size=32, num_blocks=8,  kv_heads=16, use_bf16=True),  _m()),

    # ── 13. Batch size × attention type (explicit cross-group) ────────────────
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

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DantinoX comprehensive inference benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default="results/inference_sweep.csv",
                        help="Output CSV path (default: results/inference_sweep.csv)")
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help="Run only these groups (default: all).")
    parser.add_argument("--list-groups", action="store_true",
                        help="Print available group names and exit.")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--no-mla", action="store_true",
                        help="Skip all MLA experiments (faster runs for MHA/GQA-only iteration).")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device index or comma-separated list (e.g. '0', '0,1'). "
                             "Overrides CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.list_groups:
        print("Available groups:")
        from collections import Counter
        counts = Counter(e["group"] for e in EXPERIMENTS)
        for g in ALL_GROUPS:
            print(f"  {g:<22} ({counts[g]} experiments)")
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    selected = EXPERIMENTS
    if args.groups:
        unknown = set(args.groups) - set(ALL_GROUPS)
        if unknown:
            parser.error(f"Unknown groups: {sorted(unknown)}. Valid: {ALL_GROUPS}")
        selected = [e for e in EXPERIMENTS if e["group"] in args.groups]
    if args.no_mla:
        selected = [e for e in selected if e.get("attn_variant") != "MLA"
                    and not e["cfg"].get("mla", False)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DantinoX inference sweep — {len(selected)} experiments")
    print(f"  device   : {jax.default_backend()}")
    print(f"  warmup   : {args.n_warmup}   trials: {args.n_trials}")
    print(f"  xla cache: {_XLA_CACHE}")
    print(f"  no-mla   : {args.no_mla}")
    print(f"  output   : {out_path}")
    print()

    rows: list[dict] = []
    for exp in tqdm(selected, desc="sweep", unit="exp"):
        row = run_one(exp, n_warmup=args.n_warmup, n_trials=args.n_trials)
        rows.append(row)
        if args.verbose:
            tqdm.write(
                f"  [{row['group']:>14}] {row['label']:<22} [{row['attn_variant']:<3}]  "
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
        import csv
        import io
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        out_path.write_text(buf.getvalue())
        print(f"\nSaved {len(rows)} rows → {out_path}")


def _print_summary(df: Any) -> None:
    print("\n── Group summary (best decode_tok_s per attention variant) ────────────")
    for grp in df["group"].unique():
        sub = df[df["group"] == grp].dropna(subset=["decode_tok_s"])
        if sub.empty:
            continue
        best = sub.loc[sub["decode_tok_s"].idxmax()]
        print(f"  {grp:<22}  {best['attn_variant']:<4}  {best['label']:<24}  "
              f"{best['decode_tok_s']:>8.1f} tok/s")


if __name__ == "__main__":
    main()
