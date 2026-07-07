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
  Continuous  FlowMatchingTransformer             — flow-matching Euler ODE sampler
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
import ctypes
import gc
import logging
import os
import sys
import threading
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

from dantinox.core.config import FlowMatchingConfig, ModelConfig
from dantinox.core.diffusion import make_noise_schedule
from dantinox.core.flow import FlowMatchingTransformer
from dantinox.core.model import Transformer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── NVML power monitoring (ctypes — no pynvml required) ───────────────────────

class _NVMLLib:
    """Lazy-loaded ctypes wrapper around libnvidia-ml.so.1."""

    _inst: Any = None  # None=uninitialised, False=unavailable

    @classmethod
    def get(cls) -> Any:
        if cls._inst is None:
            try:
                lib = ctypes.CDLL("libnvidia-ml.so.1")
                fn  = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
                fn()
                obj = object.__new__(cls)
                obj._lib = lib
                cls._inst = obj
            except Exception:
                cls._inst = False
        return cls._inst if cls._inst is not False else None

    def device_handle(self, idx: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        fn = getattr(self._lib, "nvmlDeviceGetHandleByIndex_v2", None) \
             or self._lib.nvmlDeviceGetHandleByIndex
        fn(idx, ctypes.byref(handle))
        return handle

    def power_mw(self, handle: ctypes.c_void_p) -> int:
        pw = ctypes.c_uint()
        self._lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(pw))
        return pw.value


class _PowerSampler:
    """Samples total GPU power (all visible devices) every ~25 ms in a thread."""

    def __init__(self) -> None:
        nvml = _NVMLLib.get()
        if nvml is None:
            raise RuntimeError("NVML not available")
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        self._handles = [nvml.device_handle(int(i.strip()))
                         for i in vis.split(",") if i.strip().lstrip("-").isdigit()]
        self._nvml   = nvml
        self._stop   = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = sum(self._nvml.power_mw(h) / 1e3 for h in self._handles)
            self._samples.append((time.perf_counter(), w))
            self._stop.wait(0.025)

    def idle_watts(self, secs: float = 0.4) -> float:
        ws: list[float] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            ws.append(sum(self._nvml.power_mw(h) / 1e3 for h in self._handles))
            time.sleep(0.025)
        return float(np.mean(ws)) if ws else float("nan")

    def __enter__(self) -> _PowerSampler:
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def joules(self) -> float:
        if len(self._samples) < 2:
            return float("nan")
        t = np.array([s[0] for s in self._samples])
        w = np.array([s[1] for s in self._samples])
        trap = getattr(np, "trapezoid", None) or np.trapz
        return float(trap(w, t))


def _measure_energy(
    fn: Any, *args: Any, min_window_s: float = 1.5
) -> tuple[float, float]:
    """Run fn repeatedly until ≥ min_window_s of GPU work, return (J/call, mean_W).

    Returns (nan, nan) when NVML is unavailable.  The idle baseline is
    subtracted so only the marginal energy of the workload is reported.
    """
    try:
        ps = _PowerSampler()
    except Exception:
        return float("nan"), float("nan")
    idle = ps.idle_watts()
    jax.block_until_ready(fn(*args))          # ensure XLA kernel is compiled
    t0    = time.perf_counter()
    n_runs = 0
    with ps:
        while time.perf_counter() - t0 < min_window_s or n_runs < 3:
            jax.block_until_ready(fn(*args))
            n_runs += 1
    dt    = time.perf_counter() - t0
    gross = ps.joules()
    net   = max(gross - idle * dt, 0.0)
    mean_w = gross / dt if dt > 0 else float("nan")
    return (net / n_runs if n_runs > 0 else float("nan")), mean_w

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
    """Attention-variant kwargs shared by ModelConfig and FlowMatchingConfig.

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


def _disc_step_topk_fn(
    model: Transformer,
    x_t: jnp.ndarray,    # (B, G)
    dual_cache: Any,
    n_unmask: int,        # Python int — static so jax.lax.top_k can use it
) -> jnp.ndarray:
    """Confident (top-k) unmasking: unmask the n_unmask most confident masked positions.

    This is the mask-predict / confidence-based decoding strategy:
    instead of uniform random unmasking, each step reveals the positions
    where the model assigns the highest max-softmax probability.
    Ties are broken by argmax (deterministic given the same logits).
    """
    out   = model(x_t, dual_cache=dual_cache, deterministic=True)
    x0    = jnp.argmax(out.logits, axis=-1).astype(jnp.int32)          # greedy pred
    probs = jax.nn.softmax(out.logits.astype(jnp.float32), axis=-1)
    conf  = jnp.where(x_t == MASK_TOKEN_ID,
                      probs.max(axis=-1), -jnp.inf)                     # (B, G)
    # top_k returns (values, indices); indices are the positions to unmask
    _, top_idx = jax.lax.top_k(conf, n_unmask)                         # (B, n_unmask)
    reveal = jnp.zeros(x_t.shape, dtype=bool).at[
        jnp.arange(x_t.shape[0])[:, None], top_idx
    ].set(True)
    return jnp.where(reveal, x0, x_t)


def _disc_step_temp_fn(
    model: Transformer,
    x_t: jnp.ndarray,
    dual_cache: Any,
    key: jax.Array,
    unmask_p: jax.Array,
    temperature: jax.Array,    # scalar; 1.0 = standard, <1 = sharper, >1 = flatter
) -> jnp.ndarray:
    """Temperature-scaled diffusion step: softmax(logits / T) before sampling."""
    out = model(x_t, dual_cache=dual_cache, deterministic=True)
    k1, k2 = jax.random.split(key)
    x0     = jax.random.categorical(k1, out.logits / temperature).astype(jnp.int32)
    reveal = jax.random.bernoulli(k2, unmask_p, x_t.shape)
    return jnp.where((x_t == MASK_TOKEN_ID) & reveal, x0, x_t)


def _disc_final_fn(model: Transformer, x_t: jnp.ndarray, dual_cache: Any) -> jnp.ndarray:
    out = model(x_t, dual_cache=dual_cache, deterministic=True)
    return jnp.where(
        x_t == MASK_TOKEN_ID,
        jnp.argmax(out.logits, axis=-1).astype(jnp.int32),
        x_t,
    )


def _flow_step_fn(
    model: FlowMatchingTransformer,
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


def _elf_decode_fn(model: FlowMatchingTransformer, z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    B   = z.shape[0]
    out = model(
        z, jnp.zeros_like(z), jnp.ones(B, dtype=z.dtype), w,
        jnp.ones(B, dtype=bool), deterministic=True,
    )
    return jnp.argmax(out.logits, axis=-1).astype(jnp.int32)


ar_prefill    = nnx.jit(_ar_prefill_fn)
ar_decode     = nnx.jit(_ar_decode_fn)
disc_prefix   = nnx.jit(_disc_prefix_fn)
disc_step     = nnx.jit(_disc_step_fn)
disc_step_temp = nnx.jit(_disc_step_temp_fn)
disc_final    = nnx.jit(_disc_final_fn)
elf_step      = nnx.jit(_flow_step_fn)
elf_decode    = nnx.jit(_elf_decode_fn)

# disc_step_topk is created per-experiment because n_unmask must be a static
# Python int for jax.lax.top_k — use functools.partial + nnx.jit per value.
import functools as _ft


@_ft.lru_cache(maxsize=64)
def _make_disc_step_topk(n_unmask: int) -> Any:
    return nnx.jit(_ft.partial(_disc_step_topk_fn, n_unmask=n_unmask))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _p99(arr: np.ndarray) -> float:
    """p99 from a small sample: use max() when n_trials < 100, else true percentile."""
    return float(np.max(arr)) if len(arr) < 100 else float(np.percentile(arr, 99))


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
    """FLOPs of one call via XLA cost analysis (GFLOPs). Best-effort; unreliable for FlowMatchingTransformer at scale."""
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
    except Exception as exc:  # noqa: BLE001
        log.info("cost_analysis failed: %s", exc)
        return float("nan")


def _analytical_gflops_transformer(
    B: int, seq_len: int, dim: int, n_heads: int, head_size: int, blocks: int,
    kv_heads: int | None = None,
) -> float:
    """Analytical FLOP estimate for one bidirectional transformer forward pass (GFLOPs).

    Uses the standard 2×matmul accounting (multiply-accumulate = 2 FLOPs).
    kv_heads defaults to n_heads (MHA). Covers attention + FFN; ignores layer-norm.
    """
    kv_h = kv_heads if kv_heads is not None else n_heads
    ffn_hidden = 4 * dim

    # Attention: Q/K/V projections + attention scores + output projection
    q_proj   = 2 * B * seq_len * dim * dim
    kv_proj  = 2 * B * seq_len * dim * (kv_h * head_size) * 2   # K and V
    attn_qk  = 2 * B * n_heads * seq_len * seq_len * head_size
    attn_av  = 2 * B * n_heads * seq_len * seq_len * head_size
    out_proj = 2 * B * seq_len * dim * dim
    attn_total = q_proj + kv_proj + attn_qk + attn_av + out_proj

    # FFN: two linear layers (no gating)
    ffn_total = 2 * (2 * B * seq_len * dim * ffn_hidden)

    return blocks * (attn_total + ffn_total) / 1e9


def _analytical_gflops_disc_step(exp: dict, cfg: Any) -> float:
    """Step GFLOPs for one Discrete diffusion denoising step."""
    B, G = exp["B"], exp["G"]
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    kv_heads = getattr(cfg, "kv_heads", n_heads)
    # Bidirectional attention over gen tokens only (prefix KVs are cached)
    return _analytical_gflops_transformer(B, G, dim, n_heads, head_size, blocks, kv_heads)


def _analytical_gflops_flow_step(exp: dict, cfg: Any) -> float:
    """Step GFLOPs for one ELF (Continuous) Euler ODE step.

    FlowMatchingTransformer processes z of shape (B, G, dim) — continuous latents —
    so seq_len = G and the embedding dimension equals dim.
    """
    B, G = exp["B"], exp["G"]
    dim, n_heads, head_size, blocks = SIZES[exp["size"]]
    kv_heads = getattr(cfg, "kv_heads", n_heads)
    return _analytical_gflops_transformer(B, G, dim, n_heads, head_size, blocks, kv_heads)


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
        "step_ms_p50": _NAN, "step_ms_p95": _NAN, "step_ms_p99": _NAN,
        "e2e_ms": _NAN, "ttft_ms": _NAN,
        # itl_ms: inter-token latency (AR only; NaN for parallel paradigms)
        "itl_ms": _NAN,
        # streaming_e2e_ms: wall-clock to deliver all G tokens in a streaming
        # scenario — for AR this is prefill + itl*(G-1); for diffusion/ELF
        # all tokens arrive simultaneously so streaming_e2e_ms == e2e_ms.
        "streaming_e2e_ms": _NAN,
        "tok_s_e2e": _NAN, "tok_s_steady": _NAN,
        "prefill_gflops": _NAN, "step_gflops": _NAN,
        "total_gen_gflops": _NAN, "gflops_per_tok": _NAN,
        "mfu_pct": _NAN,
        "peak_mem_mb": _NAN, "cache_mb": _NAN,
        # cache_overhead_pct: KV/prefix cache as % of total device memory
        "cache_overhead_pct": _NAN,
        # energy columns — populated when --power is active
        "joules": _NAN, "watts": _NAN, "j_per_tok": _NAN,
        # decoding-strategy metadata
        "decoding": exp.get("decoding", "uniform"),
        "temperature": exp.get("temperature", 1.0),
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
    cache = row.get("cache_mb", _NAN)
    peak  = row["peak_mem_mb"]
    if not (np.isnan(cache) or np.isnan(peak)) and peak > 0:
        row["cache_overhead_pct"] = round(100.0 * cache / peak, 2)


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
        row["step_ms_p99"]    = round(_p99(decode_ms), 3)
        row["ttft_ms"]        = row["prefill_ms_p50"]
        row["itl_ms"]         = row["step_ms_p50"]
        row["streaming_e2e_ms"] = round(
            row["prefill_ms_p50"] + row["step_ms_p50"] * (G - 1), 3
        )
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

        if exp.get("measure_power", False):
            joules, watts = _measure_energy(_e2e)
            row["joules"]    = round(joules, 4)
            row["watts"]     = round(watts, 1)
            row["j_per_tok"] = round(joules / (B * G), 7)

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

        # ── Per-step latency (always measured with the default "uniform" strategy
        #    for comparability; the selected decoding affects e2e timing only)
        step_gf = _analytical_gflops_disc_step(exp, cfg)
        row["step_gflops"] = round(step_gf, 4)

        step_ms = _time_fn(disc_step, model, x_mask, dual, key, p_mid,
                           n_warmup=n_warmup, n_trials=n_trials,
                           desc=f"step    {tag}")
        row["step_ms_p50"] = round(float(np.percentile(step_ms, 50)), 3)
        row["step_ms_p95"] = round(float(np.percentile(step_ms, 95)), 3)
        row["step_ms_p99"] = round(_p99(step_ms), 3)
        row["tok_s_steady"] = round(
            B * G * 1e3 / (steps * float(np.mean(step_ms))), 2
        )

        # ── End-to-end generation: decoding strategy selected by exp["decoding"] ──
        decoding    = exp.get("decoding", "uniform")
        temperature = float(exp.get("temperature", 1.0))

        if decoding == "topk":
            # Confident (mask-predict) decoding: unmask n_unmask most confident
            # masked positions per step; final step unmasks all remaining.
            n_unmask_per_step = max(1, G // steps)
            _step_topk = _make_disc_step_topk(n_unmask_per_step)
            _step_topk_final = _make_disc_step_topk(G)  # unmask all remaining

            def _e2e() -> None:
                d = disc_prefix(model, prefix) if use_prefix else None
                x = x_mask
                for _ in range(steps - 1):
                    x = _step_topk(model, x, d)
                x = _step_topk_final(model, x, d)   # clear any leftovers
                jax.block_until_ready(x)

        elif decoding == "temperature":
            temp_jnp = jnp.float32(temperature)

            def _e2e() -> None:
                d = disc_prefix(model, prefix) if use_prefix else None
                x = x_mask
                k = jax.random.key(1)
                for t in range(steps, 0, -1):
                    a_t, a_prev = alpha_bar[t], alpha_bar[t - 1]
                    p = (a_prev - a_t) / (1.0 - a_t + 1e-8) if a_t < 1.0 else 0.0
                    k, sub = jax.random.split(k)
                    x = disc_step_temp(model, x, d, sub,
                                       jnp.float32(np.clip(p, 0.0, 1.0)), temp_jnp)
                x = disc_final(model, x, d)
                jax.block_until_ready(x)

        else:  # "uniform" — standard LLaDA / MDLM schedule
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

        e2e_ms = _median_ms(_e2e, n_e2e, desc=f"e2e[{decoding}] {tag}")
        # Diffusion generates all G tokens simultaneously; the first token is
        # available only after the full e2e pass — no streaming is possible.
        row["ttft_ms"]          = round(e2e_ms, 3)
        row["streaming_e2e_ms"] = round(e2e_ms, 3)

        prefill_gf = row["prefill_gflops"] if P > 0 else 0.0
        prefill_gf = 0.0 if np.isnan(prefill_gf) else prefill_gf
        total_gf   = prefill_gf + (steps + 1) * step_gf
        _finish_row(row, exp, total_gf, e2e_ms)

        if exp.get("measure_power", False):
            joules, watts = _measure_energy(_e2e)
            row["joules"]    = round(joules, 4)
            row["watts"]     = round(watts, 1)
            row["j_per_tok"] = round(joules / (B * G), 7)

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
        cfg = FlowMatchingConfig(
            embed_dim=dim, bottleneck_dim=max(32, dim // 2),
            model_dim=dim, n_heads=n_heads, head_size=head_size,
            num_blocks=blocks, vocab_size=VOCAB_SIZE, max_seq_len=G,
            dropout=0.0,
            **_attn_kwargs(exp["size"], exp["attn"], ar_cache=False),
        )
        model = FlowMatchingTransformer(cfg, rngs=nnx.Rngs(42))
        if bf16:
            _cast_bf16(model)

        row["params_m"], row["params_mb"] = (round(v, 3) for v in _count_params(model))

        dt_np  = 1.0 / steps
        dtype  = jnp.bfloat16 if bf16 else jnp.float32
        z      = jax.random.normal(jax.random.key(0), (B, G, dim), dtype=dtype)
        x_prev = jnp.zeros_like(z)
        w      = jnp.ones((B,), dtype=dtype)
        # Use jnp.array (not jnp.float32 scalar literal) so JAX traces t and dt
        # as dynamic values — avoids a recompile per step count in the diff_steps sweep.
        t_mid  = jnp.array(0.5, dtype=jnp.float32)
        dt     = jnp.array(dt_np, dtype=jnp.float32)

        # One Euler denoise step.
        # Use analytical GFLOPs — XLA cost_analysis returns inconsistent values for
        # FlowMatchingTransformer (e.g. 9x inflated at xxl), while timing data is correct.
        step_gf = _analytical_gflops_flow_step(exp, cfg)
        row["step_gflops"] = round(step_gf, 4)

        step_ms = _time_fn(elf_step, model, z, x_prev, t_mid, dt, w,
                           n_warmup=n_warmup, n_trials=n_trials,
                           desc=f"step    {tag}")
        row["step_ms_p50"] = round(float(np.percentile(step_ms, 50)), 3)
        row["step_ms_p95"] = round(float(np.percentile(step_ms, 95)), 3)
        row["step_ms_p99"] = round(_p99(step_ms), 3)
        row["tok_s_steady"] = round(
            B * G * 1e3 / (steps * float(np.mean(step_ms))), 2
        )

        # End-to-end ELF generation: Euler ODE from noise + final decode (t=1)
        ts     = np.linspace(0.0, 1.0, steps + 1)
        ts_jax = [jnp.array(t, dtype=jnp.float32) for t in ts]
        dts    = [jnp.array(ts[i + 1] - ts[i], dtype=jnp.float32) for i in range(steps)]

        def _e2e() -> None:
            zz, xp = z, x_prev
            for i in range(steps):
                zz, xp = elf_step(model, zz, xp, ts_jax[i], dts[i], w)
            toks = elf_decode(model, zz, w)
            jax.block_until_ready(toks)

        e2e_ms = _median_ms(_e2e, n_e2e, desc=f"e2e {tag}")
        # ELF decodes all tokens simultaneously at the final step (t=1).
        row["ttft_ms"]          = round(e2e_ms, 3)
        row["streaming_e2e_ms"] = round(e2e_ms, 3)

        total_gf = steps * step_gf
        _finish_row(row, exp, total_gf, e2e_ms)

        if exp.get("measure_power", False):
            joules, watts = _measure_energy(_e2e)
            row["joules"]    = round(joules, 4)
            row["watts"]     = round(watts, 1)
            row["j_per_tok"] = round(joules / (B * G), 7)

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

        # 10. Decoding strategy: uniform vs. top-k confident vs. temperature
        #   AR reference + Discrete only (ELF ODE is deterministic at fixed steps).
        exps.append(_e("decoding", "AR-ref",        "AR",       attn=a))
        for dec in ("uniform", "topk"):
            for s in (8, 32):
                exps.append(_e("decoding", f"{dec}-s={s}", "Discrete",
                               steps=s, decoding=dec, attn=a))
        for temp in (0.5, 0.8, 1.0, 1.5):
            exps.append(_e("decoding", f"temp={temp}-s=32", "Discrete",
                           steps=32, decoding="temperature", temperature=temp, attn=a))

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
    parser.add_argument(
        "--power", action=argparse.BooleanOptionalAction, default=False,
        help="Measure GPU power via NVML and report j_per_tok / watts "
             "(adds ~1.5 s overhead per experiment; requires NVML).",
    )
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

    if args.power and _NVMLLib.get() is None:
        print("  [WARNING] --power requested but NVML is not available — j_per_tok will be NaN")

    print(f"DantinoX paradigm benchmark — {len(selected)} experiments")
    print(f"  device   : {jax.default_backend()} ({jax.devices()[0].device_kind})")
    print(f"  warmup   : {args.n_warmup}  trials: {args.n_trials}  e2e runs: {args.n_e2e}")
    print(f"  power    : {'on (NVML)' if args.power else 'off (use --power to enable)'}")
    print(f"  output   : {out_path}")
    print()

    rows: list[dict] = []
    for exp in tqdm(selected, desc="sweep", unit="exp"):
        exp = {**exp, "measure_power": args.power}
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
