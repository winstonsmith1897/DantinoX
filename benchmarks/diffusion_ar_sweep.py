#!/usr/bin/env python3
"""
benchmarks/diffusion_ar_sweep.py
=================================

Comprehensive benchmark comparing **Autoregressive (AR)** and **Masked-Diffusion**
transformers across the three attention variants supported by DantinoX:
MHA, GQA, and MLA.  Random-weight models are used so no training is required.

Metrics collected
-----------------
Shared:
  params_m              – parameter count (millions)
  params_mb             – model weight memory (MB, fp32 or bf16)
  peak_mem_mb           – device memory in use after the first forward pass
  forward_ms_p50/p95    – single forward-pass latency (both model types)

Autoregressive only:
  ar_prefill_ms_p50/p95 – forward pass on T-token prompt (no cache)
  ar_decode_ms_p50/p95  – single decode step (KV-cache, 1 new token)
  ar_tok_s              – decode throughput, batch × 1000 / median_decode_ms
  ar_kv_cache_mb        – theoretical KV-cache memory for the full context

Diffusion only:
  diff_step_ms_p50/p95          – one denoising step, no dual-cache
  diff_step_cached_ms_p50/p95   – one denoising step with dual-cache
  diff_cache_build_ms            – one-time prefix-cache build cost
  diff_dual_cache_speedup        – step_ms_p50 / step_cached_ms_p50
  diff_gen_tok_s                 – throughput estimate: B×T / (N_steps × step_s)
  diff_gen_tok_s_cached          – same with dual-cache
  diff_prefix_cache_mb           – dual-cache memory (MB)

Benchmark groups
----------------
  1. model_attn      AR × {MHA,GQA,MLA}  +  Diff × {MHA,GQA,MLA}
  2. scale           size sweep  × {AR,Diff} × {MHA,GQA}
  3. batch_size      BS sweep    × {AR,Diff} × {MHA,GQA}
  4. seq_len         ctx sweep   × {AR,Diff} × {MHA,GQA}
  5. diff_steps      #steps sweep (Diff only) × {MHA,GQA}
  6. dual_cache      prefix_len sweep (Diff only) × {MHA,GQA}
  7. dtype           fp32/bf16   × {AR,Diff} × {MHA,GQA}
  8. noise_schedule  cosine/linear/sqrt (Diff only) × {MHA,GQA}
  9. moe             dense/MoE   × {AR,Diff} × {MHA,GQA}

Usage
-----
  python benchmarks/diffusion_ar_sweep.py
  python benchmarks/diffusion_ar_sweep.py --groups scale batch_size dual_cache
  python benchmarks/diffusion_ar_sweep.py --list-groups
  python benchmarks/diffusion_ar_sweep.py --n-warmup 2 --n-trials 5 --out /tmp/quick.csv
  python benchmarks/diffusion_ar_sweep.py --no-mla --out results/nomla.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.diffusion import DualCache, NoiseSchedule, make_noise_schedule
from core.model import DiffusionTransformer, Transformer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── XLA compilation cache ──────────────────────────────────────────────────────
_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_diff_bench"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

# ── Constants ──────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 256
N_WARMUP     = 3
N_TRIALS     = 10
_GQA_RATIO   = 4     # kv_heads = n_heads // 4 for GQA experiments
_DIFF_STEPS  = 20    # default denoising steps for generation throughput estimate


# ── JIT functions (AR) ─────────────────────────────────────────────────────────

@nnx.jit
def _ar_prefill(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    logits, _, _ = model(x, use_cache=False, kv_caches=None, cache_index=0,
                         deterministic=True)
    return logits


@nnx.jit
def _ar_prefill_cached(
    model: nnx.Module, x: jnp.ndarray, cache: tuple
) -> tuple[jnp.ndarray, tuple]:
    logits, new_cache, _ = model(x, use_cache=True, kv_caches=cache,
                                 cache_index=0, deterministic=True)
    return logits, new_cache


@nnx.jit
def _ar_decode(
    model: nnx.Module, tok: jnp.ndarray, cache: tuple, pos: jax.Array
) -> tuple[jnp.ndarray, tuple]:
    logits, new_cache, _ = model(tok, use_cache=True, kv_caches=cache,
                                 cache_index=pos, deterministic=True)
    return logits, new_cache


# ── JIT functions (Diffusion — full sequence) ────────────────────────────────

@nnx.jit
def _diff_step(
    model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray
) -> jnp.ndarray:
    out = model(x_t, t, dual_cache=None, deterministic=True)  # type: ignore[call-arg]
    return out.logits


@nnx.jit
def _diff_step_cached(
    model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray, dual_cache: DualCache
) -> jnp.ndarray:
    out = model(x_t, t, dual_cache=dual_cache, deterministic=True)  # type: ignore[call-arg]
    return out.logits


# ── JIT functions (Fast-dLLM block-wise) ─────────────────────────────────────

@nnx.jit
def _diff_decode_block(
    model: nnx.Module,
    x_block: jnp.ndarray,
    t: jnp.ndarray,
    dual_cache: DualCache,
    block_start: jax.Array,
) -> jnp.ndarray:
    return model.decode_block(x_block, t, dual_cache, block_start,   # type: ignore[attr-defined]
                              deterministic=True)


def _diff_build_dual_cache(
    model: nnx.Module,
    x_full: jnp.ndarray,
    t: jnp.ndarray,
    block_start: int,
    block_end: int,
) -> DualCache:
    # Not decorated with @nnx.jit: block_start/block_end are Python ints used
    # as slice endpoints inside compute_block_dual_cache — they cannot be traced
    # as JAX values.  XLA still compiles the op graph internally on the first
    # call (visible as compile latency in _time_fn's first trial).
    return model.compute_block_dual_cache(x_full, t, block_start, block_end)  # type: ignore[attr-defined]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _time_fn(
    fn: Any,
    *args: Any,
    n_warmup: int,
    n_trials: int,
    desc: str = "",
) -> np.ndarray:
    """Warm-up + timed trials; returns latencies in milliseconds."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    if compile_s > 1.5:
        tqdm.write(f"    compile  {desc:<50} {compile_s:5.1f}s")
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(fn(*args))
    ts: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1_000.0)
    return np.array(ts)


def _count_params(model: nnx.Module) -> tuple[float, float]:
    """Returns (params_millions, params_mb)."""
    _, state = nnx.split(model)
    leaves   = jax.tree_util.tree_leaves(state)
    n_params = sum(x.size for x in leaves if hasattr(x, "size"))
    n_bytes  = sum(
        x.size * x.dtype.itemsize
        for x in leaves
        if hasattr(x, "size") and hasattr(x, "dtype")
    )
    return n_params / 1e6, n_bytes / 1e6


def _device_mem_mb() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")


def _cast_bf16(model: nnx.Module) -> None:
    params = nnx.state(model, nnx.Param)
    bf16   = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    nnx.update(model, bf16)


def _ar_kv_cache_mb(config: Config, batch_size: int, bf16: bool) -> float:
    bpp = 2 if bf16 else 4
    S   = config.max_context
    if config.mla:
        per_layer = S * (config.down_dim_kv + config.rope_dim) * bpp * batch_size
    else:
        hs        = config.dim // config.n_heads
        per_layer = 2 * S * config.kv_heads * hs * bpp * batch_size
    return round(config.num_blocks * per_layer / 1e6, 3)


def _diff_prefix_cache_mb(
    config: Config, batch_size: int, prefix_len: int, bf16: bool
) -> float:
    """Approximate memory for the dual-cache KV tensors (MHA/GQA only)."""
    if config.mla:
        return float("nan")
    bpp       = 2 if bf16 else 4
    hs        = config.dim // config.n_heads
    per_layer = 2 * prefix_len * config.kv_heads * hs * bpp * batch_size
    return round(config.num_blocks * per_layer / 1e6, 3)


def _attn_label(config: Config) -> str:
    if config.mla:
        return "MLA"
    if config.kv_heads < config.n_heads:
        return f"GQA({config.kv_heads}/{config.n_heads})"
    return "MHA"


def _attn_variant(config: Config) -> str:
    if config.mla:
        return "MLA"
    if config.kv_heads < config.n_heads:
        return "GQA"
    return "MHA"


# ── AR experiment runner ───────────────────────────────────────────────────────

def _run_ar(
    exp: dict,
    n_warmup: int,
    n_trials: int,
) -> dict:
    cfg_dict = exp["cfg"]
    m        = exp["measure"]
    B        = m["batch_size"]
    T        = m["seq_len"]
    bf16     = cfg_dict.get("use_bf16", False)
    tag      = f"AR/{exp['group']}/{exp['label']}[{exp.get('attn_variant','?')}]"

    nan = float("nan")
    try:
        config = Config.from_dict({**cfg_dict, "vocab_size": VOCAB_SIZE,
                                   "model_type": "autoregressive"})
        model  = Transformer(config, rngs=nnx.Rngs(42))
    except Exception as exc:
        log.warning("Config error %s: %s", tag, exc)
        return _oom_row(exp, "AR", {})

    if bf16:
        _cast_bf16(model)

    T         = min(T, config.max_context - 1)
    x         = jnp.ones((B, T), dtype=jnp.int32)
    tok       = jnp.ones((B, 1), dtype=jnp.int32)
    pos       = jnp.array(T, dtype=jnp.int32)
    init_cache = tuple((None, None) for _ in range(config.num_blocks))

    params_m, params_mb = _count_params(model)

    try:
        # Prefill (no cache)
        prefill_ms = _time_fn(_ar_prefill, model, x,
                              n_warmup=n_warmup, n_trials=n_trials,
                              desc=f"ar_prefill {tag}")
        peak_mem = _device_mem_mb()

        # Prefill with cache + single decode step
        _, kv_cache = _ar_prefill_cached(model, x, init_cache)
        jax.block_until_ready(kv_cache)

        decode_ms = _time_fn(_ar_decode, model, tok, kv_cache, pos,
                             n_warmup=n_warmup, n_trials=n_trials,
                             desc=f"ar_decode  {tag}")

        decode_tok_s = B * 1000.0 / float(np.median(decode_ms))

    except Exception as exc:
        log.warning("Runtime error %s: %s", tag, exc)
        return _oom_row(exp, "AR", {"config": config, "params_m": params_m,
                                    "params_mb": params_mb})

    row = _base_row(exp, config, "AR", B, T, bf16, params_m, params_mb, peak_mem)
    row.update({
        "forward_ms_p50":      round(float(np.percentile(prefill_ms, 50)), 3),
        "forward_ms_p95":      round(float(np.percentile(prefill_ms, 95)), 3),
        "ar_prefill_ms_p50":   round(float(np.percentile(prefill_ms, 50)), 3),
        "ar_prefill_ms_p95":   round(float(np.percentile(prefill_ms, 95)), 3),
        "ar_decode_ms_p50":    round(float(np.percentile(decode_ms,  50)), 3),
        "ar_decode_ms_p95":    round(float(np.percentile(decode_ms,  95)), 3),
        "ar_tok_s":            round(decode_tok_s, 2),
        "ar_kv_cache_mb":      _ar_kv_cache_mb(config, B, bf16),
        # Diffusion columns → NaN
        "diff_step_ms_p50": nan, "diff_step_ms_p95": nan,
        "diff_step_cached_ms_p50": nan, "diff_step_cached_ms_p95": nan,
        "diff_cache_build_ms": nan, "diff_dual_cache_speedup": nan,
        "diff_gen_tok_s": nan, "diff_gen_tok_s_cached": nan,
        "diff_prefix_cache_mb": nan, "diff_n_steps": nan,
        "diff_prefix_len": nan,
        "oom": False,
    })
    return row


# ── Diffusion experiment runner ────────────────────────────────────────────────

def _run_diff(
    exp: dict,
    n_warmup: int,
    n_trials: int,
) -> dict:
    cfg_dict    = exp["cfg"]
    m           = exp["measure"]
    B           = m["batch_size"]
    T           = m["seq_len"]
    prefix_len  = m.get("prefix_len", 0)
    n_steps     = m.get("n_diff_steps", _DIFF_STEPS)
    bf16        = cfg_dict.get("use_bf16", False)
    tag         = f"Diff/{exp['group']}/{exp['label']}[{exp.get('attn_variant','?')}]"

    nan = float("nan")
    try:
        config = Config.from_dict({
            **cfg_dict,
            "vocab_size": VOCAB_SIZE,
            "model_type": "diffusion",
            "mask_token_id": 0,
        })
        model    = DiffusionTransformer(config, rngs=nnx.Rngs(42))
        schedule = make_noise_schedule(config)
    except Exception as exc:
        log.warning("Config error %s: %s", tag, exc)
        return _oom_row(exp, "Diff", {})

    if bf16:
        _cast_bf16(model)

    T       = min(T, config.max_context - prefix_len - 1)
    x_t     = jnp.zeros((B, T), dtype=jnp.int32)              # all-mask
    t_step  = jnp.full((B,), config.diffusion_steps // 2, dtype=jnp.int32)

    params_m, params_mb = _count_params(model)

    try:
        # ── Single step, no dual-cache ─────────────────────────────────────
        step_ms = _time_fn(_diff_step, model, x_t, t_step,
                           n_warmup=n_warmup, n_trials=n_trials,
                           desc=f"diff_step  {tag}")
        peak_mem = _device_mem_mb()

        # ── Dual-cache: build prefix KV once ──────────────────────────────
        cache_build_ms = nan
        step_cached_ms = np.full(n_trials, nan)
        dual_cache_speedup = nan
        prefix_cache_mb    = nan

        if prefix_len > 0 and not config.mla:
            prefix = jnp.ones((B, prefix_len), dtype=jnp.int32)

            t0 = time.perf_counter()
            dual_cache = model.compute_prefix_cache(prefix)
            jax.block_until_ready(dual_cache.prefix_kvs)
            # second call to get stable (compiled) timing
            t0 = time.perf_counter()
            dual_cache = model.compute_prefix_cache(prefix)
            jax.block_until_ready(dual_cache.prefix_kvs)
            cache_build_ms = (time.perf_counter() - t0) * 1000.0

            step_cached_ms = _time_fn(
                _diff_step_cached, model, x_t, t_step, dual_cache,
                n_warmup=n_warmup, n_trials=n_trials,
                desc=f"diff_cached {tag}",
            )
            p50_plain  = float(np.percentile(step_ms, 50))
            p50_cached = float(np.percentile(step_cached_ms, 50))
            dual_cache_speedup = round(p50_plain / p50_cached, 3) if p50_cached > 0 else nan
            prefix_cache_mb    = _diff_prefix_cache_mb(config, B, prefix_len, bf16)

        # ── Throughput estimates ───────────────────────────────────────────
        # tokens generated = B × T (all positions, all samples)
        # time = N_steps × step_time
        p50_step     = float(np.percentile(step_ms, 50))
        p50_cached   = float(np.percentile(step_cached_ms, 50)) if not np.isnan(step_cached_ms).all() else nan
        gen_tok_s        = B * T * 1000.0 / (n_steps * p50_step)  if p50_step > 0 else nan
        gen_tok_s_cached = B * T * 1000.0 / (n_steps * p50_cached) if not np.isnan(p50_cached) and p50_cached > 0 else nan

    except Exception as exc:
        log.warning("Runtime error %s: %s", tag, exc)
        return _oom_row(exp, "Diff", {"config": config, "params_m": params_m,
                                      "params_mb": params_mb})

    row = _base_row(exp, config, "Diff", B, T, bf16, params_m, params_mb, peak_mem)
    row.update({
        "forward_ms_p50": round(float(np.percentile(step_ms, 50)), 3),
        "forward_ms_p95": round(float(np.percentile(step_ms, 95)), 3),
        # AR columns → NaN
        "ar_prefill_ms_p50": nan, "ar_prefill_ms_p95": nan,
        "ar_decode_ms_p50": nan, "ar_decode_ms_p95": nan,
        "ar_tok_s": nan, "ar_kv_cache_mb": nan,
        # Diffusion columns
        "diff_step_ms_p50":        round(float(np.percentile(step_ms, 50)), 3),
        "diff_step_ms_p95":        round(float(np.percentile(step_ms, 95)), 3),
        "diff_step_cached_ms_p50": round(float(np.percentile(step_cached_ms, 50)), 3) if not np.isnan(step_cached_ms).all() else nan,
        "diff_step_cached_ms_p95": round(float(np.percentile(step_cached_ms, 95)), 3) if not np.isnan(step_cached_ms).all() else nan,
        "diff_cache_build_ms":     round(cache_build_ms, 3) if not np.isnan(cache_build_ms) else nan,
        "diff_dual_cache_speedup": dual_cache_speedup,
        "diff_gen_tok_s":          round(gen_tok_s, 2) if not np.isnan(gen_tok_s) else nan,
        "diff_gen_tok_s_cached":   round(gen_tok_s_cached, 2) if not np.isnan(gen_tok_s_cached) else nan,
        "diff_prefix_cache_mb":    prefix_cache_mb,
        "diff_n_steps":            n_steps,
        "diff_prefix_len":         prefix_len,
        "oom": False,
    })
    return row


# ── Fast-dLLM block-wise experiment runner ────────────────────────────────────

def _run_diff_blockwise(exp: dict, n_warmup: int, n_trials: int) -> dict:
    """Benchmark Fast-dLLM block-wise decoding primitives.

    Measures three independent latencies (all JIT-compiled):

    1. ``decode_block_ms``    – single call to ``model.decode_block`` (inner loop op)
    2. ``build_cache_ms``     – single call to ``compute_block_dual_cache`` (refresh op)
    3. ``full_step_ms``       – full-sequence forward pass without block-wise cache
                                (baseline for comparison)

    From these we derive:
    - ``block_speedup`` = full_step_ms / decode_block_ms  (inner-loop speedup)
    - ``refresh_overhead`` = build_cache_ms / decode_block_ms  (how many inner steps
                              the refresh costs, i.e. acceptable if > steps_per_block)
    """
    cfg_dict   = exp["cfg"]
    m          = exp["measure"]
    B          = m["batch_size"]
    T_prefix   = m.get("prefix_len", 32)
    block_size = m.get("block_size", 32)
    n_suffix   = m.get("suffix_len", 64)   # tokens in suffix (other MASK blocks)
    bf16       = cfg_dict.get("use_bf16", False)
    tag        = f"DualCache/{exp['group']}/{exp['label']}[{exp.get('attn_variant','?')}]"

    nan = float("nan")
    try:
        config = Config.from_dict({**cfg_dict, "vocab_size": VOCAB_SIZE,
                                   "model_type": "diffusion"})
        model    = DiffusionTransformer(config, rngs=nnx.Rngs(42))
        schedule = make_noise_schedule(config)
    except Exception as exc:
        log.warning("Config error %s: %s", tag, exc)
        return _oom_blockwise_row(exp, {})

    if bf16:
        _cast_bf16(model)

    T_total     = T_prefix + block_size + n_suffix
    block_start = T_prefix
    block_end   = T_prefix + block_size

    if T_total > config.max_context:
        log.warning("Sequence too long %s: %d > %d", tag, T_total, config.max_context)
        return _oom_blockwise_row(exp, {})

    x_full  = jnp.zeros((B, T_total),     dtype=jnp.int32)
    x_block = jnp.zeros((B, block_size),  dtype=jnp.int32)
    t_mid   = jnp.full((B,), config.diffusion_steps // 2, dtype=jnp.int32)
    t_init  = jnp.full((B,), config.diffusion_steps,      dtype=jnp.int32)
    bs_arr  = jnp.asarray(block_start, dtype=jnp.int32)

    params_m, params_mb = _count_params(model)

    # JIT-compile build_cache with block_start/block_end as Python ints captured
    # in the closure — they cannot be traced as JAX values because they are used
    # as static Python slice endpoints inside compute_block_dual_cache.
    _bs, _be = block_start, block_end  # captured as Python ints
    @nnx.jit
    def _build_cache_jit(x_full: jnp.ndarray, t: jnp.ndarray) -> DualCache:
        return model.compute_block_dual_cache(x_full, t, _bs, _be)  # type: ignore[attr-defined]

    try:
        # ── Build dual cache once (for timing of inner decode_block) ───────
        dual_cache = _build_cache_jit(x_full, t_init)
        jax.block_until_ready(dual_cache)
        peak_mem = _device_mem_mb()

        # ── 1. decode_block latency (inner loop) ───────────────────────────
        decode_ms = _time_fn(
            _diff_decode_block, model, x_block, t_mid, dual_cache, bs_arr,
            n_warmup=n_warmup, n_trials=n_trials,
            desc=f"decode_block  {tag}",
        )

        # ── 2. build_dual_cache latency (refresh cost) ─────────────────────
        build_ms = _time_fn(
            _build_cache_jit, x_full, t_init,
            n_warmup=n_warmup, n_trials=n_trials,
            desc=f"build_cache   {tag}",
        )

        # ── 3. Full-sequence step (no block-wise) ──────────────────────────
        x_gen    = jnp.zeros((B, block_size + n_suffix), dtype=jnp.int32)
        full_ms  = _time_fn(
            _diff_step, model, x_gen, t_mid,
            n_warmup=n_warmup, n_trials=n_trials,
            desc=f"full_step     {tag}",
        )

    except Exception as exc:
        log.warning("Runtime error %s: %s", tag, exc)
        return _oom_blockwise_row(exp, {"params_m": params_m, "params_mb": params_mb})

    p50_dec  = float(np.percentile(decode_ms, 50))
    p50_bld  = float(np.percentile(build_ms,  50))
    p50_full = float(np.percentile(full_ms,   50))

    return {
        "group":        exp["group"],
        "label":        exp["label"],
        "model_type":   "DualCache",
        "attn_variant": exp.get("attn_variant", _attn_variant(config)),
        "attn_type":    _attn_label(config),
        "params_m":     round(params_m, 3),
        "params_mb":    round(params_mb, 3),
        "dim":          config.dim,
        "n_heads":      config.n_heads,
        "kv_heads":     config.kv_heads,
        "num_blocks":   config.num_blocks,
        "max_context":  config.max_context,
        "batch_size":   B,
        "seq_len":      block_size,
        "dtype":        "bf16" if bf16 else "fp32",
        "use_moe":      config.use_moe,
        "n_experts":    config.n_experts if config.use_moe else None,
        "peak_mem_mb":  round(peak_mem, 2),
        "prefix_len":   T_prefix,
        "block_size":   block_size,
        "suffix_len":   n_suffix,
        # Latencies
        "decode_block_ms_p50":  round(p50_dec,  3),
        "decode_block_ms_p95":  round(float(np.percentile(decode_ms, 95)), 3),
        "build_cache_ms_p50":   round(p50_bld,  3),
        "build_cache_ms_p95":   round(float(np.percentile(build_ms,  95)), 3),
        "full_step_ms_p50":     round(p50_full, 3),
        "full_step_ms_p95":     round(float(np.percentile(full_ms,   95)), 3),
        # Derived metrics
        "block_speedup":        round(p50_full / p50_dec, 3) if p50_dec > 0 else nan,
        "refresh_overhead_x":   round(p50_bld / p50_dec, 2) if p50_dec > 0 else nan,
        # forward_ms_p50 for unified summary column
        "forward_ms_p50": round(p50_dec, 3),
        "forward_ms_p95": round(float(np.percentile(decode_ms, 95)), 3),
        "oom": False,
    }


def _oom_blockwise_row(exp: dict, ctx: dict) -> dict:
    nan = float("nan")
    return {
        "group": exp["group"], "label": exp["label"],
        "model_type": "DualCache",
        "attn_variant": exp.get("attn_variant", "?"), "attn_type": "?",
        "params_m": ctx.get("params_m", nan), "params_mb": ctx.get("params_mb", nan),
        "dim": nan, "n_heads": nan, "kv_heads": nan, "num_blocks": nan,
        "max_context": nan, "batch_size": exp["measure"]["batch_size"],
        "seq_len": exp["measure"].get("block_size", nan),
        "dtype": "bf16" if exp["cfg"].get("use_bf16") else "fp32",
        "use_moe": exp["cfg"].get("use_moe", False), "n_experts": None,
        "peak_mem_mb": nan, "prefix_len": nan, "block_size": nan, "suffix_len": nan,
        "decode_block_ms_p50": nan, "decode_block_ms_p95": nan,
        "build_cache_ms_p50": nan, "build_cache_ms_p95": nan,
        "full_step_ms_p50": nan, "full_step_ms_p95": nan,
        "block_speedup": nan, "refresh_overhead_x": nan,
        "forward_ms_p50": nan, "forward_ms_p95": nan, "oom": True,
    }


# ── Shared row builders ────────────────────────────────────────────────────────

def _base_row(
    exp: dict,
    config: Config,
    model_type: str,
    B: int,
    T: int,
    bf16: bool,
    params_m: float,
    params_mb: float,
    peak_mem: float,
) -> dict:
    return {
        "group":        exp["group"],
        "label":        exp["label"],
        "model_type":   model_type,
        "attn_variant": exp.get("attn_variant", _attn_variant(config)),
        "attn_type":    _attn_label(config),
        "params_m":     round(params_m, 3),
        "params_mb":    round(params_mb, 3),
        "dim":          config.dim,
        "n_heads":      config.n_heads,
        "kv_heads":     config.kv_heads,
        "num_blocks":   config.num_blocks,
        "max_context":  config.max_context,
        "batch_size":   B,
        "seq_len":      T,
        "dtype":        "bf16" if bf16 else "fp32",
        "use_moe":      config.use_moe,
        "n_experts":    config.n_experts if config.use_moe else None,
        "peak_mem_mb":  round(peak_mem, 2),
    }


def _oom_row(exp: dict, model_type: str, ctx: dict) -> dict:
    nan = float("nan")
    base = {
        "group": exp["group"], "label": exp["label"],
        "model_type": model_type,
        "attn_variant": exp.get("attn_variant", "?"),
        "attn_type": "?",
        "params_m": ctx.get("params_m", nan),
        "params_mb": ctx.get("params_mb", nan),
        "dim": nan, "n_heads": nan, "kv_heads": nan,
        "num_blocks": nan, "max_context": nan,
        "batch_size": exp["measure"]["batch_size"],
        "seq_len": exp["measure"]["seq_len"],
        "dtype": "bf16" if exp["cfg"].get("use_bf16") else "fp32",
        "use_moe": exp["cfg"].get("use_moe", False),
        "n_experts": None, "peak_mem_mb": nan,
        "forward_ms_p50": nan, "forward_ms_p95": nan,
        "ar_prefill_ms_p50": nan, "ar_prefill_ms_p95": nan,
        "ar_decode_ms_p50": nan, "ar_decode_ms_p95": nan,
        "ar_tok_s": nan, "ar_kv_cache_mb": nan,
        "diff_step_ms_p50": nan, "diff_step_ms_p95": nan,
        "diff_step_cached_ms_p50": nan, "diff_step_cached_ms_p95": nan,
        "diff_cache_build_ms": nan, "diff_dual_cache_speedup": nan,
        "diff_gen_tok_s": nan, "diff_gen_tok_s_cached": nan,
        "diff_prefix_cache_mb": nan, "diff_n_steps": nan,
        "diff_prefix_len": nan, "oom": True,
    }
    return base


def run_one(exp: dict, n_warmup: int, n_trials: int) -> dict:
    if exp["model_type"] == "AR":
        return _run_ar(exp, n_warmup, n_trials)
    if exp["model_type"] == "DualCache":
        return _run_diff_blockwise(exp, n_warmup, n_trials)
    return _run_diff(exp, n_warmup, n_trials)


# ── Config factories ───────────────────────────────────────────────────────────

def _c(**kw: Any) -> dict:
    """Base AR/Diff config dict."""
    base: dict[str, Any] = dict(
        dim=256, n_heads=8, head_size=32, num_blocks=6,
        kv_heads=8, max_context=256,
        use_moe=False, n_experts=4, top_k_mlp=2,
        expansion=4, alpha_balance=0.1,
        use_swiglu=True, use_rotary_pos=True,
        trainable_pos=False, absolute_pos=False,
        sliding_window=False, context_window=32,
        mla=False, inference=False,
        down_dim_q=128, down_dim_kv=64, rope_dim=16,
        dropout_rate=0.0, gradient_checkpointing=False,
        weight_tying=True, use_bf16=False,
        diffusion_steps=1000, noise_schedule="cosine",
        mask_token_id=0, num_sampling_steps=50, time_emb_dim=128,
    )
    base.update(kw)
    return base


def _m(**kw: Any) -> dict:
    base: dict[str, Any] = dict(
        batch_size=1, seq_len=64, prefix_len=0, n_diff_steps=_DIFF_STEPS,
    )
    base.update(kw)
    return base


def _by_mode_attn(
    group: str,
    label: str,
    cfg: dict,
    measure: dict,
    include_mla: bool = True,
) -> list[dict]:
    """Expand one logical experiment into AR×MHA, AR×GQA, AR×MLA,
       Diff×MHA, Diff×GQA, Diff×MLA (6 entries)."""
    n_heads   = cfg.get("n_heads", 8)
    head_size = cfg.get("head_size", 32)
    gqa_kv    = max(1, n_heads // _GQA_RATIO)
    rope_dim  = min(16, head_size)
    mla_dkv   = min(64, head_size * 2)
    mla_dq    = min(64, head_size * 2)

    mha_cfg  = {**cfg, "kv_heads": n_heads, "mla": False, "inference": False}
    gqa_cfg  = {**cfg, "kv_heads": gqa_kv, "mla": False, "inference": False}
    mla_cfg  = {**cfg, "kv_heads": n_heads, "mla": True, "inference": True,
                "down_dim_kv": mla_dkv, "down_dim_q": mla_dq, "rope_dim": rope_dim}

    entries = [
        {"group": group, "label": label, "model_type": "AR",   "attn_variant": "MHA",
         "cfg": mha_cfg, "measure": measure},
        {"group": group, "label": label, "model_type": "AR",   "attn_variant": "GQA",
         "cfg": gqa_cfg, "measure": measure},
        {"group": group, "label": label, "model_type": "Diff", "attn_variant": "MHA",
         "cfg": mha_cfg, "measure": measure},
        {"group": group, "label": label, "model_type": "Diff", "attn_variant": "GQA",
         "cfg": gqa_cfg, "measure": measure},
    ]
    if include_mla:
        entries += [
            {"group": group, "label": label, "model_type": "AR",   "attn_variant": "MLA",
             "cfg": mla_cfg, "measure": measure},
            {"group": group, "label": label, "model_type": "Diff", "attn_variant": "MLA",
             "cfg": mla_cfg, "measure": measure},
        ]
    return entries


def _diff_only(
    group: str,
    label: str,
    cfg: dict,
    measure: dict,
    attn_variants: tuple[str, ...] = ("MHA", "GQA"),
) -> list[dict]:
    """Diffusion-only entries, one per attention variant."""
    n_heads   = cfg.get("n_heads", 8)
    gqa_kv    = max(1, n_heads // _GQA_RATIO)

    entries = []
    for v in attn_variants:
        if v == "MHA":
            c = {**cfg, "kv_heads": n_heads, "mla": False}
        elif v == "GQA":
            c = {**cfg, "kv_heads": gqa_kv, "mla": False}
        else:
            rope_dim = min(16, cfg.get("head_size", 32))
            c = {**cfg, "kv_heads": n_heads, "mla": True, "inference": True,
                 "down_dim_kv": min(64, cfg.get("head_size", 32) * 2),
                 "down_dim_q":  min(64, cfg.get("head_size", 32) * 2),
                 "rope_dim": rope_dim}
        entries.append({
            "group": group, "label": label,
            "model_type": "Diff", "attn_variant": v,
            "cfg": c, "measure": measure,
        })
    return entries


# ── Experiment definitions ─────────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [

    # ── 1. Direct model-type × attention-type comparison ─────────────────────
    # Baseline: same medium model, BS=1, seq=64, all 6 combos.

    *_by_mode_attn("model_attn", "medium-bs1", _c(), _m(batch_size=1, seq_len=64)),
    *_by_mode_attn("model_attn", "medium-bs8", _c(), _m(batch_size=8, seq_len=64)),

    # ── 2. Model scale × {AR, Diff} × {MHA, GQA} ─────────────────────────────
    # How does compute cost scale with model size for each paradigm?

    *_by_mode_attn("scale", "tiny-0.1M",
                   _c(dim=64,  n_heads=4,  head_size=16, num_blocks=2),
                   _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("scale", "small-0.5M",
                   _c(dim=128, n_heads=4,  head_size=32, num_blocks=3),
                   _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("scale", "medium-3M",   _c(),
                   _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("scale", "large-25M",
                   _c(dim=512, n_heads=16, head_size=32, num_blocks=8,  kv_heads=16),
                   _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("scale", "xlarge-50M",
                   _c(dim=512, n_heads=16, head_size=32, num_blocks=16, kv_heads=16),
                   _m(batch_size=4, seq_len=64), include_mla=False),

    # ── 3. Batch size × {AR, Diff} × {MHA, GQA} ──────────────────────────────
    # Throughput scaling: AR decode is memory-bandwidth bound; Diff scales with B×T.

    *_by_mode_attn("batch_size", "bs=1",  _c(), _m(batch_size=1,  seq_len=64), include_mla=False),
    *_by_mode_attn("batch_size", "bs=4",  _c(), _m(batch_size=4,  seq_len=64), include_mla=False),
    *_by_mode_attn("batch_size", "bs=8",  _c(), _m(batch_size=8,  seq_len=64), include_mla=False),
    *_by_mode_attn("batch_size", "bs=16", _c(), _m(batch_size=16, seq_len=64), include_mla=False),
    *_by_mode_attn("batch_size", "bs=32", _c(), _m(batch_size=32, seq_len=64), include_mla=False),
    *_by_mode_attn("batch_size", "bs=64", _c(), _m(batch_size=64, seq_len=64), include_mla=False),

    # ── 4. Sequence length × {AR, Diff} × {MHA, GQA} ─────────────────────────
    # AR: decode step is O(1) in T; prefill is O(T²).  Diff: full forward is O(T²).

    *_by_mode_attn("seq_len", "T=32",
                   _c(max_context=64),   _m(batch_size=4, seq_len=32),  include_mla=False),
    *_by_mode_attn("seq_len", "T=64",
                   _c(max_context=128),  _m(batch_size=4, seq_len=64),  include_mla=False),
    *_by_mode_attn("seq_len", "T=128",
                   _c(max_context=256),  _m(batch_size=4, seq_len=128), include_mla=False),
    *_by_mode_attn("seq_len", "T=256",
                   _c(max_context=512),  _m(batch_size=4, seq_len=256), include_mla=False),
    *_by_mode_attn("seq_len", "T=512",
                   _c(max_context=1024), _m(batch_size=4, seq_len=512), include_mla=False),

    # ── 5. Diffusion steps (Diff only) × {MHA, GQA} ──────────────────────────
    # Fewer steps → faster generation; more steps → (typically) higher quality.
    # Metric: estimated total generation time = N_steps × step_ms.

    *_diff_only("diff_steps", "steps=5",
                _c(), _m(batch_size=4, seq_len=64, n_diff_steps=5)),
    *_diff_only("diff_steps", "steps=10",
                _c(), _m(batch_size=4, seq_len=64, n_diff_steps=10)),
    *_diff_only("diff_steps", "steps=20",
                _c(), _m(batch_size=4, seq_len=64, n_diff_steps=20)),
    *_diff_only("diff_steps", "steps=50",
                _c(), _m(batch_size=4, seq_len=64, n_diff_steps=50)),
    *_diff_only("diff_steps", "steps=100",
                _c(), _m(batch_size=4, seq_len=64, n_diff_steps=100)),

    # ── 6. Dual-cache prefix length × {MHA, GQA} (Diff only) ────────────────
    # How much does the prefix KV cache speed up each denoising step?
    # The speedup should grow with prefix_len / seq_len.

    *_diff_only("dual_cache", "prefix=0",
                _c(max_context=256), _m(batch_size=4, seq_len=64, prefix_len=0)),
    *_diff_only("dual_cache", "prefix=16",
                _c(max_context=256), _m(batch_size=4, seq_len=64, prefix_len=16)),
    *_diff_only("dual_cache", "prefix=32",
                _c(max_context=256), _m(batch_size=4, seq_len=64, prefix_len=32)),
    *_diff_only("dual_cache", "prefix=64",
                _c(max_context=256), _m(batch_size=4, seq_len=64, prefix_len=64)),
    *_diff_only("dual_cache", "prefix=128",
                _c(max_context=512), _m(batch_size=4, seq_len=128, prefix_len=128)),

    # ── 7. dtype: fp32 vs bf16 × {AR, Diff} × {MHA, GQA} ───────────────────
    # Memory halved with bf16; speed typically 1.5–2× on tensor-core hardware.

    *_by_mode_attn("dtype", "medium-fp32", _c(use_bf16=False), _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("dtype", "medium-bf16", _c(use_bf16=True),  _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("dtype", "large-fp32",
                   _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=False),
                   _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("dtype", "large-bf16",
                   _c(dim=512, n_heads=16, head_size=32, num_blocks=8, kv_heads=16, use_bf16=True),
                   _m(batch_size=4, seq_len=64), include_mla=False),

    # ── 8. Noise schedule (Diff only) × {MHA, GQA} ───────────────────────────
    # Schedule only affects generation quality; forward-pass cost is identical
    # (the schedule is just used to compute unmasking probabilities externally).
    # This group verifies that all schedules produce the same latency.

    *_diff_only("noise_schedule", "cosine",
                _c(noise_schedule="cosine"),  _m(batch_size=4, seq_len=64)),
    *_diff_only("noise_schedule", "linear",
                _c(noise_schedule="linear"),  _m(batch_size=4, seq_len=64)),
    *_diff_only("noise_schedule", "sqrt",
                _c(noise_schedule="sqrt"),    _m(batch_size=4, seq_len=64)),

    # ── 9. MoE vs dense × {AR, Diff} × {MHA, GQA} ───────────────────────────
    # MoE adds parameters while top-k routing keeps active FLOPs low.

    *_by_mode_attn("moe", "dense",
                   _c(use_moe=False), _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("moe", "4exp-top2",
                   _c(use_moe=True, n_experts=4,  top_k_mlp=2), _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("moe", "8exp-top2",
                   _c(use_moe=True, n_experts=8,  top_k_mlp=2), _m(batch_size=4, seq_len=64), include_mla=False),
    *_by_mode_attn("moe", "8exp-top4",
                   _c(use_moe=True, n_experts=8,  top_k_mlp=4), _m(batch_size=4, seq_len=64), include_mla=False),

    # ── 10. MLA full comparison (AR×MLA vs Diff×MLA) ─────────────────────────
    # Includes both training-path (inference=False) and inference-path (inference=True).

    {"group": "mla_detail", "label": "AR-MLA-train",  "model_type": "AR",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=False, down_dim_kv=64, down_dim_q=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    {"group": "mla_detail", "label": "AR-MLA-infer",  "model_type": "AR",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=True,  down_dim_kv=64, down_dim_q=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    {"group": "mla_detail", "label": "Diff-MLA-train", "model_type": "Diff",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=False, down_dim_kv=64, down_dim_q=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    {"group": "mla_detail", "label": "Diff-MLA-infer", "model_type": "Diff",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=True,  down_dim_kv=64, down_dim_q=64, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    # Sweep MLA latent dimension
    {"group": "mla_detail", "label": "Diff-MLA-dkv32", "model_type": "Diff",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=False, down_dim_kv=32,  down_dim_q=32,  rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    {"group": "mla_detail", "label": "Diff-MLA-dkv64", "model_type": "Diff",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=False, down_dim_kv=64,  down_dim_q=64,  rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},
    {"group": "mla_detail", "label": "Diff-MLA-dkv128", "model_type": "Diff",
     "attn_variant": "MLA",
     "cfg": _c(mla=True, inference=False, down_dim_kv=128, down_dim_q=128, rope_dim=16, kv_heads=8),
     "measure": _m(batch_size=4, seq_len=64)},

    # ── 11. Fast-dLLM block-wise: decode_block vs full-step ──────────────────
    # Measures the three inner primitives of fast_dllm_generate:
    #   decode_block_ms  – inner loop (amortised across steps_per_block)
    #   build_cache_ms   – refresh cost (amortised across all inner steps)
    #   full_step_ms     – baseline (model.__call__ on the generation suffix)
    # block_speedup = full_step_ms / decode_block_ms
    # refresh_overhead_x = build_cache_ms / decode_block_ms  (ideal: > steps_per_block)

    {"group": "block_wise", "label": "MHA-bs1",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=1, block_size=32, prefix_len=64, suffix_len=128)},
    {"group": "block_wise", "label": "MHA-bs4",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=64, suffix_len=128)},
    {"group": "block_wise", "label": "MHA-bs16", "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=16, block_size=32, prefix_len=64, suffix_len=128)},
    {"group": "block_wise", "label": "GQA-bs1",  "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=512),
     "measure": dict(batch_size=1, block_size=32, prefix_len=64, suffix_len=128)},
    {"group": "block_wise", "label": "GQA-bs4",  "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=64, suffix_len=128)},
    {"group": "block_wise", "label": "GQA-bs16", "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=512),
     "measure": dict(batch_size=16, block_size=32, prefix_len=64, suffix_len=128)},

    # ── 12. Block-size sweep × {MHA, GQA} ────────────────────────────────────
    # Replicates Fig. 4 from Fast-dLLM: smaller block = more accurate but higher
    # refresh overhead; larger block = faster but more approximation error.
    # block_speedup and refresh_overhead_x are the key output metrics.

    {"group": "block_size_sweep", "label": "bs=4",   "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=4,   prefix_len=64, suffix_len=192)},
    {"group": "block_size_sweep", "label": "bs=8",   "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=8,   prefix_len=64, suffix_len=192)},
    {"group": "block_size_sweep", "label": "bs=16",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=16,  prefix_len=64, suffix_len=192)},
    {"group": "block_size_sweep", "label": "bs=32",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32,  prefix_len=64, suffix_len=128)},
    {"group": "block_size_sweep", "label": "bs=64",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=64,  prefix_len=64, suffix_len=64)},
    {"group": "block_size_sweep", "label": "bs=128", "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=256),
     "measure": dict(batch_size=4, block_size=128, prefix_len=64, suffix_len=0)},

    # ── 13. Prefix/suffix ratio × {MHA, GQA} ─────────────────────────────────
    # Speedup from DualCache grows with (prefix + suffix) / block_size.
    # This group isolates the suffix contribution (suffix = all remaining MASK blocks).

    {"group": "context_ratio", "label": "suf=0",   "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=256),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=0)},
    {"group": "context_ratio", "label": "suf=32",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=256),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=32)},
    {"group": "context_ratio", "label": "suf=64",  "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=256),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=64)},
    {"group": "context_ratio", "label": "suf=128", "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=128)},
    {"group": "context_ratio", "label": "suf=256", "model_type": "DualCache", "attn_variant": "MHA",
     "cfg": _c(kv_heads=8, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=256)},
    # Same for GQA
    {"group": "context_ratio", "label": "suf=0-GQA",   "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=256),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=0)},
    {"group": "context_ratio", "label": "suf=128-GQA", "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=128)},
    {"group": "context_ratio", "label": "suf=256-GQA", "model_type": "DualCache", "attn_variant": "GQA",
     "cfg": _c(kv_heads=2, mla=False, max_context=512),
     "measure": dict(batch_size=4, block_size=32, prefix_len=32, suffix_len=256)},
]

ALL_GROUPS: list[str] = sorted({e["group"] for e in EXPERIMENTS})


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DantinoX AR vs Diffusion benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default="results/diffusion_ar_sweep.csv",
                        help="Output CSV path")
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help="Run only these groups (default: all).")
    parser.add_argument("--list-groups", action="store_true")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--no-mla", action="store_true",
                        help="Skip MLA experiments (faster iterations for MHA/GQA focus).")
    parser.add_argument("--no-diff", action="store_true",
                        help="Skip Diffusion experiments (AR-only mode).")
    parser.add_argument("--no-ar", action="store_true",
                        help="Skip Autoregressive experiments (Diff-only mode).")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device index (e.g. '0', '1').")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.list_groups:
        from collections import Counter
        counts = Counter(e["group"] for e in EXPERIMENTS)
        print("Available groups:")
        for g in ALL_GROUPS:
            print(f"  {g:<22}  ({counts[g]} experiments)")
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
        selected = [e for e in selected
                    if e.get("attn_variant") != "MLA" and not e["cfg"].get("mla")]
    if args.no_diff:
        selected = [e for e in selected if e["model_type"] != "Diff"]
    if args.no_ar:
        selected = [e for e in selected if e["model_type"] != "AR"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DantinoX AR vs Diffusion sweep — {len(selected)} experiments")
    print(f"  device   : {jax.default_backend()}")
    print(f"  warmup   : {args.n_warmup}   trials: {args.n_trials}")
    print(f"  xla cache: {_XLA_CACHE}")
    print(f"  output   : {out_path}")
    print()

    rows: list[dict] = []
    for exp in tqdm(selected, desc="sweep", unit="exp"):
        row = run_one(exp, n_warmup=args.n_warmup, n_trials=args.n_trials)
        rows.append(row)
        if args.verbose:
            tqdm.write(
                f"  [{row['model_type']:<4}][{row['attn_variant']:<3}] "
                f"{row['group']:<14} {row['label']:<22} "
                f"fwd={row['forward_ms_p50']:>7.2f}ms  "
                f"params={row['params_m']:>6.2f}M  "
                f"mem={row['peak_mem_mb']:>7.1f}MB"
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
    nan = float("nan")
    print("\n── Summary by group ──────────────────────────────────────────────────")
    print(f"  {'group':<20} {'model':<5} {'attn':<5} {'label':<22} "
          f"{'fwd_ms':>8} {'tok/s':>10} {'mem_MB':>8}")
    print("  " + "─" * 78)
    for grp in df["group"].unique():
        sub = df[df["group"] == grp].copy()
        for mt in ("AR", "Diff"):
            for av in ("MHA", "GQA", "MLA"):
                filt = sub[(sub["model_type"] == mt) & (sub["attn_variant"] == av)]
                if filt.empty:
                    continue
                best = filt.loc[filt["forward_ms_p50"].idxmin()]
                tok_s = best.get("ar_tok_s", nan) if mt == "AR" else best.get("diff_gen_tok_s", nan)
                tok_s_str = f"{tok_s:>10.1f}" if tok_s == tok_s else "       n/a"
                print(f"  {grp:<20} {mt:<5} {av:<5} {best['label']:<22} "
                      f"{best['forward_ms_p50']:>8.2f} {tok_s_str} "
                      f"{best['peak_mem_mb']:>8.1f}")


if __name__ == "__main__":
    main()
