#!/usr/bin/env python3
"""
benchmarks/paradigm_bench.py
============================

Three-way inference benchmark: **AR vs Discrete Diffusion vs Continuous
Diffusion** — the complete paradigm comparison for the EMNLP system demo.

All three paradigms share the same transformer backbone tier (dim, heads,
blocks) so differences reflect the *paradigm*, not the architecture:

  AR          Transformer(causal=True)   — prefill + KV-cache greedy decode
  Discrete    Transformer(causal=False)  — LLaDA-style iterative unmasking,
                                           prefix dual-cache for conditioning
  Continuous  ELFTransformer             — flow-matching Euler ODE sampler
                                           (z_t = t·x + (1−t)·ε) + decode step

Metrics per experiment
----------------------
  params_m / params_mb        parameter count and weight memory
  prefill_ms_p50/p95          AR prompt prefill | Discrete prefix-cache build
  step_ms_p50/p95             one decode step (AR) / one denoise step (Diff)
  e2e_ms                      measured wall-clock to generate B×G tokens
  ttft_ms                     time-to-first-token (diffusion: = e2e — tokens
                              materialise all at once)
  tok_s_e2e                   B × G / e2e_s   (end-to-end throughput)
  tok_s_steady                steady-state throughput (excludes prefill)
  prefill_gflops, step_gflops measured XLA FLOPs (cost_analysis) per call
  total_gen_gflops            FLOPs to generate the full batch
  gflops_per_tok              total / (B × G)
  mfu_pct                     total_flops / e2e_time / A100 peak
                              (fp32→TF32 156 TFLOP/s, bf16 312 TFLOP/s)
  peak_mem_mb                 device bytes_in_use after generation
  cache_mb                    AR KV-cache / Discrete prefix-cache memory

Sweep groups
------------
  scale        6 backbone tiers (~0.1 M → ~90 M)      × 3 paradigms
  batch_size   B ∈ {1,2,4,8,16,32,64,128}             × 3 paradigms
  gen_len      G ∈ {32,64,128,256,512,1024}           × 3 paradigms
  diff_steps   steps ∈ {4,8,16,32,64,128}             × {Discrete, Continuous}
               (+ AR reference row)
  dtype        fp32 vs bf16 × {medium, large}         × 3 paradigms
  prompt_len   P ∈ {16,64,256,512}                    × {AR, Discrete}
               (prefill / conditioning-cache scaling; ELF is unconditional)

Usage
-----
  python benchmarks/paradigm_bench.py --out results/paradigm_bench.csv
  python benchmarks/paradigm_bench.py --groups scale batch_size
  python benchmarks/paradigm_bench.py --quick          # smoke test
  python benchmarks/paradigm_bench.py --list-groups
"""
from __future__ import annotations

import argparse
import gc
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

from dantinox.core.config import ELFConfig, ModelConfig
from dantinox.core.diffusion import make_noise_schedule
from dantinox.core.elf import ELFTransformer
from dantinox.core.model import Transformer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_paradigm_bench"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

# ── Constants ──────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 256
MASK_TOKEN_ID = 4
N_WARMUP      = 3
N_TRIALS      = 10
N_E2E_RUNS    = 3          # 1 warm-up + (N-1) timed full-generation runs

# A100 peak throughput used for MFU (XLA uses TF32 tensor cores for fp32 matmul)
PEAK_FLOPS = {"fp32": 156e12, "bf16": 312e12}

# Backbone tiers shared by all three paradigms: (dim, n_heads, head_size, blocks)
SIZES: dict[str, tuple[int, int, int, int]] = {
    "tiny":   (64,  4,  16, 2),
    "small":  (128, 4,  32, 3),
    "medium": (256, 8,  32, 6),
    "large":  (512, 16, 32, 8),
    "xl":     (512, 16, 32, 16),
    "xxl":    (768, 12, 64, 12),
}

_GQA_RATIO = 4   # GQA: kv_heads = n_heads // 4 (min 1)


def _attn_kwargs(size: str, attn: str, ar_cache: bool) -> dict[str, Any]:
    """Attention-variant kwargs shared by ModelConfig and ELFConfig.

    ``ar_cache=True`` enables the MLA absorbed-projection inference path,
    required for AR KV-cache decoding (not used by the diffusion forwards).
    """
    _, n_heads, head_size, _ = SIZES[size]
    if attn == "gqa":
        return {"attention": "gqa", "kv_heads": max(1, n_heads // _GQA_RATIO)}
    if attn == "mla":
        kw: dict[str, Any] = {
            "attention": "mla",
            "down_dim_q":  min(64, head_size * 2),
            "down_dim_kv": min(64, head_size * 2),
            "rope_dim":    min(16, head_size),
        }
        if ar_cache:
            kw["inference_mode"] = True
        return kw
    return {"attention": "mha"}


# ── Step functions (un-jitted cores; jitted wrappers below) ───────────────────

def _ar_prefill_fn(model: Transformer, x: jnp.ndarray, cache: tuple) -> tuple:
    out = model(x, caches=cache, cache_index=0, deterministic=True)
    return out.logits, out.kv_caches


def _ar_decode_fn(
    model: Transformer, tok: jnp.ndarray, cache: tuple, pos: jax.Array
) -> tuple:
    out = model(tok, caches=cache, cache_index=pos, deterministic=True)
    nxt = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)[:, None]
    return nxt, out.kv_caches


def _disc_prefix_fn(model: Transformer, prefix: jnp.ndarray):
    return model.compute_prefix_cache(prefix)


def _disc_step_fn(
    model: Transformer,
    x_t: jnp.ndarray,
    dual_cache: Any,
    key: jax.Array,
    unmask_p: jax.Array,
) -> jnp.ndarray:
    out = model(x_t, dual_cache=dual_cache, deterministic=True)
    k1, k2 = jax.random.split(key)
    x0     = jax.random.categorical(k1, out.logits).astype(jnp.int32)
    reveal = jax.random.bernoulli(k2, unmask_p, x_t.shape)
    return jnp.where((x_t == MASK_TOKEN_ID) & reveal, x0, x_t)


def _disc_final_fn(model: Transformer, x_t: jnp.ndarray, dual_cache: Any) -> jnp.ndarray:
    out = model(x_t, dual_cache=dual_cache, deterministic=True)
    return jnp.where(
        x_t == MASK_TOKEN_ID,
        jnp.argmax(out.logits, axis=-1).astype(jnp.int32),
        x_t,
    )


def _elf_step_fn(
    model: ELFTransformer,
    z: jnp.ndarray,
    x_prev: jnp.ndarray,
    t: jax.Array,       # scalar in [0, 1]
    dt: jax.Array,      # scalar
    w: jnp.ndarray,     # [B]
) -> tuple:
    B     = z.shape[0]
    t_arr = jnp.full((B,), t, dtype=z.dtype)
    out   = model(z, x_prev, t_arr, w, jnp.zeros(B, dtype=bool), deterministic=True)
    v     = (out.x_pred - z) / jnp.clip(1.0 - t, 1e-6)
    return (z + dt * v).astype(z.dtype), out.x_pred


def _elf_decode_fn(model: ELFTransformer, z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    B   = z.shape[0]
    out = model(
        z, jnp.zeros_like(z), jnp.ones(B, dtype=z.dtype), w,
        jnp.ones(B, dtype=bool), deterministic=True,
    )
    return jnp.argmax(out.logits, axis=-1).astype(jnp.int32)


ar_prefill  = nnx.jit(_ar_prefill_fn)
ar_decode   = nnx.jit(_ar_decode_fn)
disc_prefix = nnx.jit(_disc_prefix_fn)
disc_step   = nnx.jit(_disc_step_fn)
disc_final  = nnx.jit(_disc_final_fn)
elf_step    = nnx.jit(_elf_step_fn)
elf_decode  = nnx.jit(_elf_decode_fn)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _time_fn(fn: Any, *args: Any, n_warmup: int, n_trials: int, desc: str = "") -> np.ndarray:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    if compile_s > 1.5:
        tqdm.write(f"    compile  {desc:<52} {compile_s:5.1f}s")
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(fn(*args))
    ts: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts)


def _measured_gflops(fn: Any, model: nnx.Module, *args: Any) -> float:
    """Measured FLOPs of one call via XLA cost analysis (GFLOPs)."""
    graphdef, state = nnx.split(model)

    def pure(state: Any, *a: Any) -> Any:
        m = nnx.merge(graphdef, state)
        return fn(m, *a)

    try:
        compiled = jax.jit(pure).lower(state, *args).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca.get("flops", float("nan"))) / 1e9
    except Exception as exc:  # noqa: BLE001 — cost analysis is best-effort
        log.info("cost_analysis failed: %s", exc)
        return float("nan")


def _count_params(model: nnx.Module) -> tuple[float, float]:
    _, state = nnx.split(model)
    leaves   = jax.tree_util.tree_leaves(state)
    n  = sum(x.size for x in leaves if hasattr(x, "size"))
    nb = sum(x.size * x.dtype.itemsize for x in leaves
             if hasattr(x, "size") and hasattr(x, "dtype"))
    return n / 1e6, nb / 1e6


def _cast_bf16(model: nnx.Module) -> None:
    params = nnx.state(model, nnx.Param)
    bf16 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    nnx.update(model, bf16)


def _device_mem_mb() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")


def _median_ms(fn: Any, n_runs: int, desc: str = "") -> float:
    """Wall-clock a full-generation closure: 1 warm-up + (n_runs-1) timed."""
    t0 = time.perf_counter()
    fn()
    compile_s = time.perf_counter() - t0
    if compile_s > 2.0:
        tqdm.write(f"    e2e warm {desc:<52} {compile_s:5.1f}s")
    ts: list[float] = []
    for _ in range(max(1, n_runs - 1)):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _rng_tokens(key: int, shape: tuple[int, ...]) -> jnp.ndarray:
    toks = jax.random.randint(jax.random.key(key), shape, 5, VOCAB_SIZE, dtype=jnp.int32)
    return toks


# ── Row scaffolding ────────────────────────────────────────────────────────────

_NAN = float("nan")

def _base_row(exp: dict) -> dict:
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    return {
        "group":      exp["group"],
        "label":      exp["label"],
        "paradigm":   exp["paradigm"],
        "size":       exp["size"],
        "attn":       exp["attn"].upper(),
        "dim":        dim,
        "n_heads":    n_heads,
        "num_blocks": blocks,
        "batch_size": exp["B"],
        "prompt_len": exp["P"],
        "gen_len":    exp["G"],
        "n_steps":    exp["steps"],
        "dtype":      "bf16" if exp["bf16"] else "fp32",
        "params_m":   _NAN, "params_mb": _NAN,
        "prefill_ms_p50": _NAN, "prefill_ms_p95": _NAN,
        "step_ms_p50": _NAN, "step_ms_p95": _NAN,
        "e2e_ms": _NAN, "ttft_ms": _NAN,
        "tok_s_e2e": _NAN, "tok_s_steady": _NAN,
        "prefill_gflops": _NAN, "step_gflops": _NAN,
        "total_gen_gflops": _NAN, "gflops_per_tok": _NAN,
        "mfu_pct": _NAN,
        "peak_mem_mb": _NAN, "cache_mb": _NAN,
        "oom": False,
    }


def _finish_row(row: dict, exp: dict, total_gflops: float, e2e_ms: float) -> None:
    B, G  = exp["B"], exp["G"]
    dtype = "bf16" if exp["bf16"] else "fp32"
    row["e2e_ms"]           = round(e2e_ms, 3)
    row["tok_s_e2e"]        = round(B * G * 1e3 / e2e_ms, 2) if e2e_ms > 0 else _NAN
    row["total_gen_gflops"] = round(total_gflops, 3)
    row["gflops_per_tok"]   = round(total_gflops / (B * G), 6)
    if not np.isnan(total_gflops) and e2e_ms > 0:
        row["mfu_pct"] = round(
            100.0 * total_gflops * 1e9 / (e2e_ms / 1e3) / PEAK_FLOPS[dtype], 3
        )
    row["peak_mem_mb"] = round(_device_mem_mb(), 2)


# ── Paradigm runners ───────────────────────────────────────────────────────────

def run_ar(exp: dict, n_warmup: int, n_trials: int, n_e2e: int) -> dict:
    row  = _base_row(exp)
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    B, P, G, bf16 = exp["B"], exp["P"], exp["G"], exp["bf16"]
    tag = f"AR/{exp['group']}/{exp['label']}"

    try:
        cfg = ModelConfig(
            dim=dim, n_heads=n_heads, head_size=head_size, num_blocks=blocks,
            vocab_size=VOCAB_SIZE, max_context=P + G + 1, causal=True, dropout=0.0,
            **_attn_kwargs(exp["size"], exp["attn"], ar_cache=True),
        )
        model = Transformer(cfg, rngs=nnx.Rngs(42))
        if bf16:
            _cast_bf16(model)

        row["params_m"], row["params_mb"] = (round(v, 3) for v in _count_params(model))

        prompt     = _rng_tokens(0, (B, P))
        tok0       = jnp.ones((B, 1), dtype=jnp.int32)
        init_cache = tuple((None, None) for _ in range(blocks))

        # FLOPs (measured, XLA cost analysis)
        prefill_gf = _measured_gflops(_ar_prefill_fn, model, prompt, init_cache)
        _, cache   = ar_prefill(model, prompt, init_cache)
        jax.block_until_ready(cache)
        pos        = jnp.array(P, dtype=jnp.int32)
        decode_gf  = _measured_gflops(_ar_decode_fn, model, tok0, cache, pos)
        row["prefill_gflops"] = round(prefill_gf, 4)
        row["step_gflops"]    = round(decode_gf, 6)

        # Latency
        prefill_ms = _time_fn(ar_prefill, model, prompt, init_cache,
                              n_warmup=n_warmup, n_trials=n_trials,
                              desc=f"prefill {tag}")
        decode_ms  = _time_fn(ar_decode, model, tok0, cache, pos,
                              n_warmup=n_warmup, n_trials=n_trials,
                              desc=f"decode  {tag}")
        row["prefill_ms_p50"] = round(float(np.percentile(prefill_ms, 50)), 3)
        row["prefill_ms_p95"] = round(float(np.percentile(prefill_ms, 95)), 3)
        row["step_ms_p50"]    = round(float(np.percentile(decode_ms, 50)), 3)
        row["step_ms_p95"]    = round(float(np.percentile(decode_ms, 95)), 3)
        row["ttft_ms"]        = row["prefill_ms_p50"]
        row["tok_s_steady"]   = round(B * 1e3 / float(np.median(decode_ms)), 2)

        # End-to-end greedy generation: prefill + G cached decode steps
        def _e2e() -> None:
            _, c = ar_prefill(model, prompt, init_cache)
            t = tok0
            for i in range(G):
                t, c = ar_decode(model, t, c, jnp.array(P + i, dtype=jnp.int32))
            jax.block_until_ready(t)

        e2e_ms = _median_ms(_e2e, n_e2e, desc=f"e2e {tag}")

        bpp = 2 if bf16 else 4
        S   = P + G + 1
        if cfg.mla:   # absorbed latent cache: (down_dim_kv + rope_dim) per position
            per_layer = S * (cfg.down_dim_kv + cfg.rope_dim) * bpp * B
        else:
            per_layer = 2 * S * cfg.kv_heads * head_size * bpp * B
        row["cache_mb"] = round(blocks * per_layer / 1e6, 3)
        total_gf = prefill_gf + G * decode_gf
        _finish_row(row, exp, total_gf, e2e_ms)

    except Exception as exc:  # noqa: BLE001
        log.warning("OOM/error %s: %s", tag, exc)
        row["oom"] = True
    return row


def run_discrete(exp: dict, n_warmup: int, n_trials: int, n_e2e: int) -> dict:
    row  = _base_row(exp)
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    B, P, G, steps, bf16 = exp["B"], exp["P"], exp["G"], exp["steps"], exp["bf16"]
    tag = f"Disc/{exp['group']}/{exp['label']}"

    try:
        cfg = ModelConfig(
            dim=dim, n_heads=n_heads, head_size=head_size, num_blocks=blocks,
            vocab_size=VOCAB_SIZE, max_context=P + G + 1, causal=False,
            dropout=0.0, mask_token_id=MASK_TOKEN_ID,
            **_attn_kwargs(exp["size"], exp["attn"], ar_cache=False),
        )
        model = Transformer(cfg, rngs=nnx.Rngs(42))
        if bf16:
            _cast_bf16(model)

        row["params_m"], row["params_mb"] = (round(v, 3) for v in _count_params(model))

        schedule  = make_noise_schedule("cosine", steps)
        alpha_bar = np.asarray(schedule.alpha_bar, dtype=np.float64)
        prefix    = _rng_tokens(0, (B, P))
        x_mask    = jnp.full((B, G), MASK_TOKEN_ID, dtype=jnp.int32)
        key       = jax.random.key(0)
        p_mid     = jnp.float32(0.05)

        # Conditioning prefix cache ("prefill" analogue).
        # MLA blocks do not expose raw KV (absorbed latents), so the dual-cache
        # path is unavailable — denoise unconditioned, prefill columns stay NaN.
        use_prefix = P > 0 and exp["attn"] != "mla"
        dual = None
        if use_prefix:
            prefill_ms = _time_fn(disc_prefix, model, prefix,
                                  n_warmup=n_warmup, n_trials=n_trials,
                                  desc=f"prefix  {tag}")
            row["prefill_ms_p50"] = round(float(np.percentile(prefill_ms, 50)), 3)
            row["prefill_ms_p95"] = round(float(np.percentile(prefill_ms, 95)), 3)
            row["prefill_gflops"] = round(
                _measured_gflops(_disc_prefix_fn, model, prefix), 4
            )
            dual = disc_prefix(model, prefix)
            jax.block_until_ready(dual.prefix_kvs)
            bpp = 2 if bf16 else 4
            row["cache_mb"] = round(
                blocks * 2 * P * cfg.kv_heads * head_size * bpp * B / 1e6, 3
            )

        # One denoise step (full bidirectional forward over the G masked tokens)
        step_gf = _measured_gflops(_disc_step_fn, model, x_mask, dual, key, p_mid)
        row["step_gflops"] = round(step_gf, 4)

        step_ms = _time_fn(disc_step, model, x_mask, dual, key, p_mid,
                           n_warmup=n_warmup, n_trials=n_trials,
                           desc=f"step    {tag}")
        row["step_ms_p50"] = round(float(np.percentile(step_ms, 50)), 3)
        row["step_ms_p95"] = round(float(np.percentile(step_ms, 95)), 3)
        row["tok_s_steady"] = round(
            B * G * 1e3 / (steps * float(np.median(step_ms))), 2
        )

        # End-to-end LLaDA-style reverse diffusion: steps × denoise + final fill
        def _e2e() -> None:
            d = disc_prefix(model, prefix) if use_prefix else None
            x = x_mask
            k = jax.random.key(1)
            for t in range(steps, 0, -1):
                a_t, a_prev = alpha_bar[t], alpha_bar[t - 1]
                p = (a_prev - a_t) / (1.0 - a_t + 1e-8) if a_t < 1.0 else 0.0
                k, sub = jax.random.split(k)
                x = disc_step(model, x, d, sub, jnp.float32(np.clip(p, 0.0, 1.0)))
            x = disc_final(model, x, d)
            jax.block_until_ready(x)

        e2e_ms = _median_ms(_e2e, n_e2e, desc=f"e2e {tag}")
        row["ttft_ms"] = round(e2e_ms, 3)   # tokens arrive only at the end

        prefill_gf = row["prefill_gflops"] if P > 0 else 0.0
        prefill_gf = 0.0 if np.isnan(prefill_gf) else prefill_gf
        total_gf   = prefill_gf + (steps + 1) * step_gf
        _finish_row(row, exp, total_gf, e2e_ms)

    except Exception as exc:  # noqa: BLE001
        log.warning("OOM/error %s: %s", tag, exc)
        row["oom"] = True
    return row


def run_continuous(exp: dict, n_warmup: int, n_trials: int, n_e2e: int) -> dict:
    row  = _base_row(exp)
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    B, G, steps, bf16 = exp["B"], exp["G"], exp["steps"], exp["bf16"]
    tag = f"Cont/{exp['group']}/{exp['label']}"

    try:
        cfg = ELFConfig(
            embed_dim=dim, bottleneck_dim=max(32, dim // 2),
            model_dim=dim, n_heads=n_heads, head_size=head_size,
            num_blocks=blocks, vocab_size=VOCAB_SIZE, max_seq_len=G,
            gradient_checkpointing=False, dropout=0.0,
            **_attn_kwargs(exp["size"], exp["attn"], ar_cache=False),
        )
        model = ELFTransformer(cfg, rngs=nnx.Rngs(42))
        if bf16:
            _cast_bf16(model)

        row["params_m"], row["params_mb"] = (round(v, 3) for v in _count_params(model))

        dt_np  = 1.0 / steps
        dtype  = jnp.bfloat16 if bf16 else jnp.float32
        z      = jax.random.normal(jax.random.key(0), (B, G, dim), dtype=dtype)
        x_prev = jnp.zeros_like(z)
        w      = jnp.ones((B,), dtype=dtype)
        t_mid  = jnp.float32(0.5)
        dt     = jnp.float32(dt_np)

        # One Euler denoise step
        step_gf = _measured_gflops(_elf_step_fn, model, z, x_prev, t_mid, dt, w)
        row["step_gflops"] = round(step_gf, 4)

        step_ms = _time_fn(elf_step, model, z, x_prev, t_mid, dt, w,
                           n_warmup=n_warmup, n_trials=n_trials,
                           desc=f"step    {tag}")
        row["step_ms_p50"] = round(float(np.percentile(step_ms, 50)), 3)
        row["step_ms_p95"] = round(float(np.percentile(step_ms, 95)), 3)
        row["tok_s_steady"] = round(
            B * G * 1e3 / (steps * float(np.median(step_ms))), 2
        )
        decode_gf = _measured_gflops(_elf_decode_fn, model, z, w)

        # End-to-end ELF generation: Euler ODE from noise + final decode (t=1)
        ts = np.linspace(0.0, 1.0, steps + 1)

        def _e2e() -> None:
            zz, xp = z, x_prev
            for i in range(steps):
                zz, xp = elf_step(
                    model, zz, xp,
                    jnp.float32(ts[i]), jnp.float32(ts[i + 1] - ts[i]), w,
                )
            toks = elf_decode(model, zz, w)
            jax.block_until_ready(toks)

        e2e_ms = _median_ms(_e2e, n_e2e, desc=f"e2e {tag}")
        row["ttft_ms"] = round(e2e_ms, 3)   # tokens decoded all at once at t=1

        decode_gf = 0.0 if np.isnan(decode_gf) else decode_gf
        total_gf  = steps * step_gf + decode_gf
        _finish_row(row, exp, total_gf, e2e_ms)

    except Exception as exc:  # noqa: BLE001
        log.warning("OOM/error %s: %s", tag, exc)
        row["oom"] = True
    return row


RUNNERS = {"AR": run_ar, "Discrete": run_discrete, "Continuous": run_continuous}


# ── Experiment definitions ─────────────────────────────────────────────────────

def _e(group: str, label: str, paradigm: str, **kw: Any) -> dict:
    base = dict(size="medium", B=4, P=64, G=128, steps=32, bf16=False, attn="mha")
    base.update(kw)
    return {"group": group, "label": label, "paradigm": paradigm, **base}


def build_experiments() -> list[dict]:
    """Full sweep, crossed with every attention variant.

    Every logical experiment is repeated for MHA, GQA-1/4 and MLA so each
    paradigm × attention combination is covered on every axis
    (~93 × 3 = 279 experiments).  Use ``--attn`` to run one partition per GPU.
    """
    exps: list[dict] = []

    for a in ("mha", "gqa", "mla"):

        # 1. scale — backbone tier sweep
        for size in SIZES:
            for p in ("AR", "Discrete", "Continuous"):
                exps.append(_e("scale", size, p, size=size, attn=a))

        # 2. batch size
        for b in (1, 2, 4, 8, 16, 32, 64, 128):
            for p in ("AR", "Discrete", "Continuous"):
                exps.append(_e("batch_size", f"B={b}", p, B=b, attn=a))

        # 3. generation length
        for g in (32, 64, 128, 256, 512, 1024):
            for p in ("AR", "Discrete", "Continuous"):
                exps.append(_e("gen_len", f"G={g}", p, G=g, attn=a))

        # 4. diffusion steps (quality–speed knob; AR reference for context)
        exps.append(_e("diff_steps", "AR-ref", "AR", attn=a))
        for s in (4, 8, 16, 32, 64, 128):
            for p in ("Discrete", "Continuous"):
                exps.append(_e("diff_steps", f"steps={s}", p, steps=s, attn=a))

        # 5. dtype
        for size in ("medium", "large"):
            for bf16 in (False, True):
                lbl = f"{size}-{'bf16' if bf16 else 'fp32'}"
                for p in ("AR", "Discrete", "Continuous"):
                    exps.append(_e("dtype", lbl, p, size=size, bf16=bf16, attn=a))

        # 6. prompt length (prefill / conditioning scaling; ELF is unconditional)
        for pl in (16, 64, 256, 512):
            for p in ("AR", "Discrete"):
                exps.append(_e("prompt_len", f"P={pl}", p, P=pl, attn=a))

        # 7–9. Steps ablations: does the AR crossover move with G / B / size?
        # (AR baselines come from the gen_len / batch_size / scale groups.)
        for s in (4, 16, 64, 256):
            for p in ("Discrete", "Continuous"):
                for g in (64, 256, 1024):
                    exps.append(_e("steps_x_genlen", f"s={s},G={g}", p,
                                   steps=s, G=g, attn=a))
                for b in (1, 16, 128):
                    exps.append(_e("steps_x_batch", f"s={s},B={b}", p,
                                   steps=s, B=b, attn=a))
                for size in ("small", "medium", "xl"):
                    exps.append(_e("steps_x_scale", f"s={s},{size}", p,
                                   steps=s, size=size, attn=a))

    return exps


EXPERIMENTS = build_experiments()
ALL_GROUPS  = sorted({e["group"] for e in EXPERIMENTS})


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DantinoX AR vs Discrete vs Continuous diffusion inference benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--out", default="results/paradigm_bench.csv")
    parser.add_argument("--groups", nargs="+", metavar="GROUP")
    parser.add_argument("--attn", nargs="+", choices=["mha", "gqa", "mla"],
                        help="Run only these attention variants "
                             "(partition the sweep across GPUs).")
    parser.add_argument("--list-groups", action="store_true")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-e2e", type=int, default=N_E2E_RUNS)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 warmup, 3 trials, small subset.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.list_groups:
        from collections import Counter
        counts = Counter(e["group"] for e in EXPERIMENTS)
        print("Available groups:")
        for g in ALL_GROUPS:
            print(f"  {g:<14} ({counts[g]} experiments)")
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    selected = EXPERIMENTS
    if args.groups:
        unknown = set(args.groups) - set(ALL_GROUPS)
        if unknown:
            parser.error(f"Unknown groups: {sorted(unknown)}. Valid: {ALL_GROUPS}")
        selected = [e for e in EXPERIMENTS if e["group"] in args.groups]
    if args.attn:
        selected = [e for e in selected if e["attn"] in args.attn]

    if args.quick:
        args.n_warmup, args.n_trials, args.n_e2e = 1, 3, 2
        seen: set[tuple] = set()
        quick: list[dict] = []
        for e in selected:                     # first label of each group × paradigm
            k = (e["group"], e["paradigm"])
            if k not in seen:
                seen.add(k)
                quick.append(e)
        selected = quick

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DantinoX paradigm benchmark — {len(selected)} experiments")
    print(f"  device   : {jax.default_backend()} ({jax.devices()[0].device_kind})")
    print(f"  warmup   : {args.n_warmup}  trials: {args.n_trials}  e2e runs: {args.n_e2e}")
    print(f"  output   : {out_path}")
    print()

    rows: list[dict] = []
    for exp in tqdm(selected, desc="sweep", unit="exp"):
        row = RUNNERS[exp["paradigm"]](exp, args.n_warmup, args.n_trials, args.n_e2e)
        rows.append(row)
        if args.verbose and not row["oom"]:
            tqdm.write(
                f"  [{row['paradigm']:<10}] {row['group']:<11} {row['label']:<14} "
                f"e2e={row['e2e_ms']:>9.1f}ms  tok/s={row['tok_s_e2e']:>9.1f}  "
                f"GF/tok={row['gflops_per_tok']:>8.4f}  mem={row['peak_mem_mb']:>7.1f}MB"
            )
        gc.collect()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")

    ok = df[~df["oom"]]
    if not ok.empty:
        print("\n── Best end-to-end throughput per group ──────────────────────────")
        for grp in ok["group"].unique():
            sub = ok[ok["group"] == grp].dropna(subset=["tok_s_e2e"])
            for p in ("AR", "Discrete", "Continuous"):
                ps = sub[sub["paradigm"] == p]
                if ps.empty:
                    continue
                best = ps.loc[ps["tok_s_e2e"].idxmax()]
                print(f"  {grp:<12} {p:<11} {best['label']:<14} "
                      f"{best['tok_s_e2e']:>10.1f} tok/s  "
                      f"({best['gflops_per_tok']:.4f} GF/tok)")


if __name__ == "__main__":
    main()
