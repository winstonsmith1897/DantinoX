#!/usr/bin/env python3
"""
benchmarks/paradigm_ablations.py
================================

EMNLP-grade *architectural* ablations for AR vs Discrete vs Continuous
diffusion inference, run on the **production architecture actually trained
in this project** (dim=512, 8 heads × 64, 12 blocks, vocab 32 128 — the
``*_512d_12b`` runs) rather than toy configs.

Methodology
-----------
All end-to-end generation loops are **fused into a single XLA computation**
(``lax.fori_loop``) for *every* paradigm, so no paradigm pays Python dispatch
overhead while another doesn't.  bf16 weights throughout (as trained).
FLOPs/bytes are read from the compiled executables (XLA cost analysis).

Ablations
---------
  grid     Parity map + roofline over a (batch × gen_len) grid.
           - parity_steps S*(B,G): denoising steps diffusion can afford while
             matching AR fused-generation latency.  Empirical: diffusion step
             time saturates the GPU sub-linearly in B×G.
           - step-level FLOPs *and* bytes accessed → arithmetic intensity →
             A100 roofline placement (AR decode memory-bound, diffusion
             steps compute-bound).
           - OOM cells are themselves findings: diffusion materialises
             B×G×V logits per step, AR only B×1×V.

  stack    Serving-stack waterfall: marginal contribution of each inference
           optimisation, per paradigm (all fused, S=32):
           AR        : no-KV-cache → +KV-cache → +bf16(weights already bf16:
                       fp32 variant included for the dtype delta)
           Discrete  : vanilla [prefix|x_t] forward → +prefix dual-cache
                       → +block-wise DualCache (Fast-dLLM schedule)
           Continuous: fp32 → bf16

  ceiling  Largest concurrent batch one A100-40GB sustains per
           paradigm × attention at G=512 (bf16), with steady tok/s at the
           ceiling.  AR is bounded by KV-cache, diffusion by per-step
           activations/logits.

Usage
-----
  python benchmarks/paradigm_ablations.py grid    --device 0
  python benchmarks/paradigm_ablations.py stack   --device 0
  XLA_PYTHON_CLIENT_MEM_FRACTION=.92 \\
  python benchmarks/paradigm_ablations.py ceiling --device 0
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
from dantinox.core.elf import ELFTransformer
from dantinox.core.generation import generate as ar_generate_lib
from dantinox.core.model import Transformer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_ablations"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

NAN = float("nan")

# ── Production architectures ──────────────────────────────────────────────────
# 512d12b matches runs/*_512d_12b_*; 768d16b matches runs/diff_mha_768d_16b;
# 1024d16b is the extra scale point.  (dim, n_heads, head_size, blocks)

ARCHS: dict[str, tuple[int, int, int, int]] = {
    "512d12b":  (512,  8,  64, 12),
    "768d16b":  (768,  12, 64, 16),
    "1024d16b": (1024, 16, 64, 16),
    "1536d24b": (1536, 24, 64, 24),   # ~0.95 B params — run with --tp 4
}

VOCAB   = 32_128
MASK_ID = 32_099
DIM, N_HEADS, HEAD_SIZE, BLOCKS = ARCHS["512d12b"]
GQA_KV  = 2                        # GQA-1/4
MLA_KW  = dict(down_dim_q=128, down_dim_kv=96, rope_dim=16)


def _set_arch(name: str) -> None:
    global DIM, N_HEADS, HEAD_SIZE, BLOCKS, GQA_KV
    DIM, N_HEADS, HEAD_SIZE, BLOCKS = ARCHS[name]
    GQA_KV = max(1, N_HEADS // 4)


# ── Tensor parallelism (set by --tp in main) ──────────────────────────────────

MESH = None          # jax Mesh when --tp > 1, else None
TP_SIZE = 1


def _mesh_ctx():
    """Context manager that activates the TP mesh (no-op without --tp)."""
    from contextlib import nullcontext
    if MESH is None:
        return nullcontext()
    try:
        return jax.sharding.use_mesh(MESH)     # JAX ≥ 0.5
    except AttributeError:
        return MESH                            # legacy Mesh context manager


def _apply_tp(model: nnx.Module) -> None:
    if MESH is not None:
        from dantinox.core.sharding import apply_tp_sharding
        apply_tp_sharding(model, MESH)


# ── Statistics: median + bootstrap 95% CI ─────────────────────────────────────

def _time_stats(fn: Any, *args: Any, n_trials: int, desc: str = "") -> dict:
    """Median and bootstrap 95% CI of wall-clock ms over n_trials calls."""
    t0 = time.perf_counter()
    with _mesh_ctx():
        jax.block_until_ready(fn(*args))
    if (c := time.perf_counter() - t0) > 2.0:
        tqdm.write(f"    compile {desc:<52} {c:5.1f}s")
    # extra steady-state warmup (post-compile autotuning remnants)
    with _mesh_ctx():
        jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    boot = np.median(
        np.random.default_rng(0).choice(a, size=(1000, len(a)), replace=True),
        axis=1)
    return {"med": float(np.median(a)),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "n": len(a)}


# ── Energy: NVML power sampling around a timed section ────────────────────────

class PowerSampler:
    """Samples GPU power on the visible device(s) every ~25 ms in a thread."""

    def __init__(self) -> None:
        import threading
        import pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(int(i))
                         for i in vis.split(",") if i != ""]
        self._nv = pynvml
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: Any = None
        self._threading = threading

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = sum(self._nv.nvmlDeviceGetPowerUsage(h) / 1e3
                    for h in self._handles)
            self._samples.append((time.perf_counter(), w))
            self._stop.wait(0.025)

    def idle_watts(self, secs: float = 0.5) -> float:
        t0 = time.perf_counter()
        ws = []
        while time.perf_counter() - t0 < secs:
            ws.append(sum(self._nv.nvmlDeviceGetPowerUsage(h) / 1e3
                          for h in self._handles))
            time.sleep(0.02)
        return float(np.mean(ws))

    def __enter__(self) -> "PowerSampler":
        self._samples = []
        self._stop.clear()
        self._thread = self._threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def joules(self) -> float:
        if len(self._samples) < 2:
            return float("nan")
        t = np.array([s[0] for s in self._samples])
        w = np.array([s[1] for s in self._samples])
        trap = getattr(np, "trapezoid", None) or np.trapz
        return float(trap(w, t))


def _energy_of(fn: Any, *args: Any, min_window_s: float = 1.5) -> tuple[float, float]:
    """(net joules per call, mean watts), idle-subtracted.

    Repeats the call until at least ``min_window_s`` of work is sampled, so
    the ~25 ms NVML sampling period cannot dominate short kernels."""
    try:
        ps = PowerSampler()
    except Exception:
        return float("nan"), float("nan")
    idle = ps.idle_watts()
    with _mesh_ctx():
        jax.block_until_ready(fn(*args))         # ensure compiled
        t0 = time.perf_counter()
        n_runs = 0
        with ps:
            while time.perf_counter() - t0 < min_window_s or n_runs < 3:
                jax.block_until_ready(fn(*args))
                n_runs += 1
        dt = time.perf_counter() - t0
    gross = ps.joules()
    net = gross - idle * dt
    return max(net, 0.0) / n_runs, gross / dt if dt > 0 else float("nan")

# A100 peaks for MFU / roofline
PEAK = {"fp32": 156e12, "bf16": 312e12}      # TF32 / BF16 tensor cores
HBM_BW = 1.555e12                            # bytes/s


def _attn_cfg(attn: str, ar_cache: bool) -> dict[str, Any]:
    if attn == "gqa":
        return {"attention": "gqa", "kv_heads": GQA_KV}
    if attn == "mla":
        kw: dict[str, Any] = {"attention": "mla", **MLA_KW}
        if ar_cache:
            kw["inference_mode"] = True
        return kw
    return {"attention": "mha"}


# ── Builders ───────────────────────────────────────────────────────────────────

def build_ar(attn: str, max_context: int, bf16: bool) -> tuple[ModelConfig, Transformer]:
    cfg = ModelConfig(
        dim=DIM, n_heads=N_HEADS, head_size=HEAD_SIZE, num_blocks=BLOCKS,
        vocab_size=VOCAB, max_context=max_context, causal=True, dropout=0.0,
        tp_size=TP_SIZE, **_attn_cfg(attn, ar_cache=True),
    )
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    if bf16:
        _cast_bf16(model)
    _apply_tp(model)
    return cfg, model


def build_disc(attn: str, max_context: int, bf16: bool) -> tuple[ModelConfig, Transformer]:
    cfg = ModelConfig(
        dim=DIM, n_heads=N_HEADS, head_size=HEAD_SIZE, num_blocks=BLOCKS,
        vocab_size=VOCAB, max_context=max_context, causal=False, dropout=0.0,
        mask_token_id=MASK_ID, tp_size=TP_SIZE, **_attn_cfg(attn, ar_cache=False),
    )
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    if bf16:
        _cast_bf16(model)
    _apply_tp(model)
    return cfg, model


def build_elf(attn: str, G: int, bf16: bool) -> tuple[ELFConfig, ELFTransformer]:
    cfg = ELFConfig(
        embed_dim=DIM, bottleneck_dim=128, model_dim=DIM,
        n_heads=N_HEADS, head_size=HEAD_SIZE, num_blocks=BLOCKS,
        vocab_size=VOCAB, max_seq_len=G,
        gradient_checkpointing=False, dropout=0.0,
        **_attn_cfg(attn, ar_cache=False),
    )
    model = ELFTransformer(cfg, rngs=nnx.Rngs(42))
    if bf16:
        _cast_bf16(model)
    _apply_tp(model)   # weight sharding only; GSPMD propagates inside blocks
    return cfg, model


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _cast_bf16(model: nnx.Module) -> None:
    params = nnx.state(model, nnx.Param)
    nnx.update(model, jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    ))


def _device_mem_mb() -> float:
    try:
        return jax.devices()[0].memory_stats().get("bytes_in_use", 0) / 1e6
    except Exception:
        return NAN


def _time_call(fn: Any, *args: Any, n_trials: int = 3, desc: str = "") -> float:
    """Median wall-clock ms of a (possibly fused) call; first call compiles."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    if compile_s > 2.0:
        tqdm.write(f"    compile {desc:<52} {compile_s:5.1f}s")
    ts = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _cost(fn: Any, model: nnx.Module, *args: Any) -> tuple[float, float]:
    """(GFLOPs, GBytes accessed) per call from the compiled executable."""
    graphdef, state = nnx.split(model)

    def pure(s: Any, *a: Any) -> Any:
        return fn(nnx.merge(graphdef, s), *a)

    try:
        compiled = jax.jit(pure).lower(state, *args).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return (float(ca.get("flops", NAN)) / 1e9,
                float(ca.get("bytes accessed", NAN)) / 1e9)
    except Exception as exc:  # noqa: BLE001
        log.info("cost_analysis failed: %s", exc)
        return NAN, NAN


# ── Step functions (for FLOPs / bytes / step latency) ─────────────────────────

def _ar_decode_fn(model: Transformer, tok: jnp.ndarray, cache: tuple,
                  pos: jax.Array) -> tuple:
    out = model(tok, caches=cache, cache_index=pos, deterministic=True)
    nxt = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)[:, None]
    return nxt, out.kv_caches


def _ar_prefill_fn(model: Transformer, x: jnp.ndarray, cache: tuple) -> tuple:
    out = model(x, caches=cache, cache_index=0, deterministic=True)
    return out.logits, out.kv_caches


def _disc_step_fn(model: Transformer, x_t: jnp.ndarray, dual: Any,
                  key: jax.Array, p: jax.Array) -> jnp.ndarray:
    out = model(x_t, dual_cache=dual, deterministic=True)
    k1, k2 = jax.random.split(key)
    x0 = jax.random.categorical(k1, out.logits).astype(jnp.int32)
    reveal = jax.random.bernoulli(k2, p, x_t.shape)
    return jnp.where((x_t == MASK_ID) & reveal, x0, x_t)


def _elf_step_fn(model: ELFTransformer, z: jnp.ndarray, x_prev: jnp.ndarray,
                 t: jax.Array, dt: jax.Array, w: jnp.ndarray) -> tuple:
    B = z.shape[0]
    out = model(z, x_prev, jnp.full((B,), t, dtype=z.dtype), w,
                jnp.zeros(B, dtype=bool), deterministic=True)
    v = (out.x_pred - z) / jnp.clip(1.0 - t, 1e-6)
    # Cast both carries back to the input dtype: fp32 control tokens upcast
    # x_pred, which would break the fori_loop carry signature under bf16.
    return (z + dt * v).astype(z.dtype), out.x_pred.astype(z.dtype)


ar_decode  = nnx.jit(_ar_decode_fn)
ar_prefill = nnx.jit(_ar_prefill_fn)
disc_step  = nnx.jit(_disc_step_fn)
elf_step   = nnx.jit(_elf_step_fn)


# ── Fused end-to-end generators (lax.fori_loop inside one XLA program) ────────

@nnx.jit
def ar_gen_fused(model: Transformer, prompt: jnp.ndarray) -> jnp.ndarray:
    """Library AR path: prefill + KV-cached greedy decode, fully fused."""
    return ar_generate_lib(
        model, prompt,
        max_generations=model.max_context - prompt.shape[1],  # type: ignore[attr-defined]
        greedy=True, use_cache=True,
    )


@nnx.jit
def ar_gen_fused_nocache(model: Transformer, prompt: jnp.ndarray) -> jnp.ndarray:
    return ar_generate_lib(
        model, prompt,
        max_generations=model.max_context - prompt.shape[1],  # type: ignore[attr-defined]
        greedy=True, use_cache=False,
    )


@nnx.jit
def disc_gen_fused(model: Transformer, prefix: jnp.ndarray, x0: jnp.ndarray,
                   key: jax.Array, unmask_ps: jnp.ndarray) -> jnp.ndarray:
    """LLaDA reverse diffusion with prefix dual-cache, fully fused.

    ``unmask_ps[i]`` is the precomputed reveal probability of step *i*
    (from the noise schedule); S = len(unmask_ps) steps + final greedy fill.
    """
    dual = model.compute_prefix_cache(prefix)

    def body(i: jax.Array, val: tuple) -> tuple:
        x, k = val
        k, sub = jax.random.split(k)
        x = _disc_step_fn(model, x, dual, sub, unmask_ps[i])
        return x, k

    x, _ = jax.lax.fori_loop(0, unmask_ps.shape[0], body, (x0, key))
    out = model(x, dual_cache=dual, deterministic=True)
    return jnp.where(x == MASK_ID,
                     jnp.argmax(out.logits, axis=-1).astype(jnp.int32), x)


@nnx.jit
def disc_gen_fused_vanilla(model: Transformer, x0_full: jnp.ndarray,
                           key: jax.Array, unmask_ps: jnp.ndarray) -> jnp.ndarray:
    """Vanilla LLaDA: forward over the full [prefix | x_t] every step."""
    def body(i: jax.Array, val: tuple) -> tuple:
        x, k = val
        k, sub = jax.random.split(k)
        x = _disc_step_fn(model, x, None, sub, unmask_ps[i])
        return x, k

    x, _ = jax.lax.fori_loop(0, unmask_ps.shape[0], body, (x0_full, key))
    out = model(x, deterministic=True)
    return jnp.where(x == MASK_ID,
                     jnp.argmax(out.logits, axis=-1).astype(jnp.int32), x)


@nnx.jit
def elf_gen_fused(model: ELFTransformer, z0: jnp.ndarray, w: jnp.ndarray,
                  ts: jnp.ndarray) -> jnp.ndarray:
    """ELF Euler ODE sampler + final decode, fully fused. ts: [S+1] schedule."""
    B = z0.shape[0]

    def body(i: jax.Array, val: tuple) -> tuple:
        z, xp = val
        t, dt = ts[i], ts[i + 1] - ts[i]
        return _elf_step_fn(model, z, xp, t, dt, w)

    z, _ = jax.lax.fori_loop(0, ts.shape[0] - 1, body, (z0, jnp.zeros_like(z0)))
    out = model(z, jnp.zeros_like(z), jnp.ones(B, dtype=z.dtype), w,
                jnp.ones(B, dtype=bool), deterministic=True)
    return jnp.argmax(out.logits, axis=-1).astype(jnp.int32)


def _unmask_schedule(S: int) -> jnp.ndarray:
    """Per-step reveal probabilities from the cosine alpha-bar schedule."""
    from dantinox.core.diffusion import make_noise_schedule
    ab = np.asarray(make_noise_schedule("cosine", S).alpha_bar, dtype=np.float64)
    ps = [(ab[t - 1] - ab[t]) / (1.0 - ab[t] + 1e-8) if ab[t] < 1.0 else 0.0
          for t in range(S, 0, -1)]
    return jnp.asarray(np.clip(ps, 0.0, 1.0), dtype=jnp.float32)


# ══ Ablation 1+2: parity grid + roofline ═══════════════════════════════════════

GRID_B, GRID_G = (1, 4, 16, 64), (64, 256, 1024)
GRID_STEPS, GRID_P = 32, 64


def _stats_cols(prefix: str, st: dict) -> dict:
    return {f"{prefix}_ms_med": round(st["med"], 4),
            f"{prefix}_ms_lo": round(st["lo"], 4),
            f"{prefix}_ms_hi": round(st["hi"], 4)}


# ── Crash-safe incremental CSV (segfault-tolerant, resumable) ─────────────────
# Pattern: append a marker row (oom=True) BEFORE attempting a point, and the
# full row after success.  A hard allocator segfault leaves the marker on
# disk, so a relaunch skips the lethal point.  Analysis keeps the last row
# per key.

GRID_FIELDS = [
    "arch", "paradigm", "batch_size", "gen_len", "prompt_len", "dtype", "tp",
    "step_ms_med", "step_ms_lo", "step_ms_hi",
    "e2e_ms_med", "e2e_ms_lo", "e2e_ms_hi",
    "step_gflops", "step_gbytes", "step_gflops_xla", "gen_gflops",
    "tok_s_e2e", "joules", "watts", "j_per_tok",
    "parity_steps", "speedup_at_32", "peak_mem_mb", "oom",
]
PARETO_FIELDS = [
    "arch", "paradigm", "label", "n_steps", "batch_size", "gen_len",
    "prompt_len", "dtype", "tp",
    "e2e_ms_med", "e2e_ms_lo", "e2e_ms_hi",
    "tok_s_system", "joules", "j_per_tok", "watts", "peak_mem_mb", "oom",
]


def _append_row(path: Path, row: dict, fields: list[str]) -> None:
    import csv as _csv
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
        fh.flush()


def _load_done(path: Path, key_cols: list[str]) -> dict[tuple, dict]:
    """Last row per key from an existing CSV (resume support)."""
    import csv as _csv
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    with path.open() as fh:
        for r in _csv.DictReader(fh):
            done[tuple(str(r.get(c, "")) for c in key_cols)] = r
    return done


def run_grid(args: argparse.Namespace) -> list[dict]:
    from benchmarks.analytical_cost import CostModel
    from benchmarks.paradigm_bench import disc_prefix  # jitted

    out_path = Path(args.out or f"results/ablation_grid_{args.arch}.csv")
    done = _load_done(out_path, ["paradigm", "batch_size", "gen_len"])
    if getattr(args, "retry_oom", False):
        done = {k: v for k, v in done.items()
                if str(v.get("oom", "")).lower() != "true"}
    unmask_ps = _unmask_schedule(GRID_STEPS)

    def _measure_ar(B: int, G: int, P: int, prompt: jnp.ndarray, cell: dict) -> float:
        cfg, model = build_ar("mha", P + G, bf16=args.precision == "bf16")
        cm = CostModel.from_model(model, cfg)
        init_cache = tuple((None, None) for _ in range(BLOCKS))
        tok0 = jnp.ones((B, 1), dtype=jnp.int32)
        with _mesh_ctx():
            _, cache = ar_prefill(model, prompt, init_cache)
            jax.block_until_ready(cache)
        pos = jnp.array(P, dtype=jnp.int32)
        dec_gf_xla, _ = _cost(_ar_decode_fn, model, tok0, cache, pos)
        st_step = _time_stats(ar_decode, model, tok0, cache, pos,
                              n_trials=args.n_trials)
        del cache
        st_e2e = _time_stats(ar_gen_fused, model, prompt,
                             n_trials=args.n_e2e, desc=f"AR e2e B{B} G{G}")
        joules, watts = _energy_of(ar_gen_fused, model, prompt)
        e2e = st_e2e["med"]
        _append_row(out_path, {**cell, "paradigm": "AR",
                    **_stats_cols("step", st_step), **_stats_cols("e2e", st_e2e),
                    "step_gflops": round(cm.ar_decode_step(B, P + G), 4),
                    "step_gbytes": round(cm.ar_decode_bytes(B, P + G), 5),
                    "step_gflops_xla": round(dec_gf_xla, 4),
                    "gen_gflops": round(cm.ar_generate(B, P, G), 2),
                    "tok_s_e2e": round(B * G * 1e3 / e2e, 2),
                    "joules": round(joules, 3), "watts": round(watts, 1),
                    "j_per_tok": round(joules / (B * G), 6),
                    "peak_mem_mb": round(_device_mem_mb(), 1),
                    "oom": False}, GRID_FIELDS)
        del model
        return e2e

    def _measure_diff(paradigm: str, B: int, G: int, P: int,
                      prompt: jnp.ndarray, cell: dict, ar_e2e: float) -> None:
        if paradigm == "Discrete":
            cfg, model = build_disc("mha", P + G, bf16=args.precision == "bf16")
            cm = CostModel.from_model(model, cfg)
            x_mask = jnp.full((B, G), MASK_ID, dtype=jnp.int32)
            with _mesh_ctx():
                dual = disc_prefix(model, prompt)
                jax.block_until_ready(dual.prefix_kvs)
            gf_xla, _ = _cost(_disc_step_fn, model, x_mask, dual,
                              jax.random.key(0), jnp.float32(0.05))
            st_step = _time_stats(disc_step, model, x_mask, dual,
                                  jax.random.key(0), jnp.float32(0.05),
                                  n_trials=args.n_trials)
            del dual
            gen_args = (model, prompt, x_mask, jax.random.key(1), unmask_ps)
            gen_fn = disc_gen_fused
            an = dict(step_gflops=round(cm.disc_step(B, G, P), 2),
                      step_gbytes=round(cm.disc_step_bytes(B, G, P), 4),
                      gen_gflops=round(cm.disc_generate(B, G, P, GRID_STEPS), 2))
        else:
            cfg, model = build_elf("mha", G, bf16=args.precision == "bf16")
            cm = CostModel.from_model(model, cfg)
            zdt = jnp.bfloat16 if args.precision == "bf16" else jnp.float32
            z = jax.random.normal(jax.random.key(0), (B, G, DIM), dtype=zdt)
            xp = jnp.zeros_like(z)
            w = jnp.ones((B,), dtype=zdt)
            gf_xla, _ = _cost(_elf_step_fn, model, z, xp,
                              jnp.float32(0.5), jnp.float32(1 / GRID_STEPS), w)
            st_step = _time_stats(elf_step, model, z, xp, jnp.float32(0.5),
                                  jnp.float32(1 / GRID_STEPS), w,
                                  n_trials=args.n_trials)
            ts = jnp.linspace(0.0, 1.0, GRID_STEPS + 1, dtype=jnp.float32)
            gen_args = (model, z, w, ts)
            gen_fn = elf_gen_fused
            an = dict(step_gflops=round(cm.elf_step(B, G), 2),
                      step_gbytes=round(cm.elf_step_bytes(B, G), 4),
                      gen_gflops=round(cm.elf_generate(B, G, GRID_STEPS), 2))

        st_e2e = _time_stats(gen_fn, *gen_args, n_trials=args.n_e2e,
                             desc=f"{paradigm} e2e B{B} G{G}")
        joules, watts = _energy_of(gen_fn, *gen_args)
        step_med, e2e = st_step["med"], st_e2e["med"]
        _append_row(out_path, {**cell, "paradigm": paradigm,
                    **_stats_cols("step", st_step), **_stats_cols("e2e", st_e2e),
                    **an, "step_gflops_xla": round(gf_xla, 2),
                    "tok_s_e2e": round(B * G * 1e3 / e2e, 2),
                    "joules": round(joules, 3), "watts": round(watts, 1),
                    "j_per_tok": round(joules / (B * G), 6),
                    "parity_steps": round(ar_e2e / step_med, 2)
                        if ar_e2e == ar_e2e and step_med > 0 else NAN,
                    "speedup_at_32": round(ar_e2e / e2e, 3)
                        if ar_e2e == ar_e2e else NAN,
                    "peak_mem_mb": round(_device_mem_mb(), 1),
                    "oom": False}, GRID_FIELDS)
        del model

    for B, G in tqdm([(b, g) for b in GRID_B for g in GRID_G], desc="grid"):
        P = GRID_P
        prompt = jax.random.randint(jax.random.key(0), (B, P), 5, VOCAB,
                                    dtype=jnp.int32)
        cell = dict(arch=args.arch, batch_size=B, gen_len=G, prompt_len=P,
                    dtype=args.precision, tp=TP_SIZE)

        # AR — also provides the parity baseline for the diffusion cells
        ar_e2e = NAN
        key = ("AR", str(B), str(G))
        if key in done:
            try:
                ar_e2e = float(done[key].get("e2e_ms_med") or "nan")
            except (TypeError, ValueError):
                pass
            tqdm.write(f"  skip AR B{B} G{G} (resumed)")
        else:
            _append_row(out_path, {**cell, "paradigm": "AR", "oom": True},
                        GRID_FIELDS)
            try:
                ar_e2e = _measure_ar(B, G, P, prompt, cell)
            except Exception as exc:  # noqa: BLE001
                log.warning("grid AR B%d G%d: %s", B, G, exc)
            gc.collect()

        for paradigm in ("Discrete", "Continuous"):
            key = (paradigm, str(B), str(G))
            if key in done:
                tqdm.write(f"  skip {paradigm} B{B} G{G} (resumed)")
                continue
            _append_row(out_path, {**cell, "paradigm": paradigm, "oom": True},
                        GRID_FIELDS)
            try:
                _measure_diff(paradigm, B, G, P, prompt, cell, ar_e2e)
            except Exception as exc:  # noqa: BLE001
                log.warning("grid %s B%d G%d: %s", paradigm, B, G, exc)
            gc.collect()

    # Rows live on disk (incremental, crash-safe).
    return []


# ══ Ablation: serving Pareto (latency per request vs system throughput) ════════

PARETO_B = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PARETO_G, PARETO_P = 256, 64
PARETO_STEPS = (8, 32)


def run_pareto(args: argparse.Namespace) -> list[dict]:
    """For each paradigm/setting, sweep batch size and record the
    (per-request latency, aggregate throughput, energy) frontier.
    Crash-safe: marker row before each point, full row on success; on
    relaunch completed points are skipped and a series is abandoned at the
    smallest batch size that previously OOMed/segfaulted."""
    out_path = Path(args.out or f"results/ablation_pareto_{args.arch}.csv")
    done = _load_done(out_path, ["label", "batch_size"])
    dead_b: dict[str, int] = {}
    if getattr(args, "retry_oom", False):
        done = {k: v for k, v in done.items()
                if str(v.get("oom", "")).lower() != "true"}
    else:
        for (lbl, b), r in done.items():
            if str(r.get("oom", "")).lower() == "true":
                dead_b[lbl] = min(dead_b.get(lbl, 1 << 30), int(b))

    P, G = PARETO_P, PARETO_G
    settings: list[tuple[str, str, int | None]] = [("AR", "AR (greedy)", None)]
    settings += [("Discrete", f"Discrete S={s}", s) for s in PARETO_STEPS]
    settings += [("Continuous", f"Continuous S={s}", s) for s in PARETO_STEPS]

    for paradigm, label, S in settings:
        for B in tqdm(PARETO_B, desc=f"pareto {label}", unit="B"):
            key = (label, str(B))
            if key in done and str(done[key].get("oom", "")).lower() != "true":
                tqdm.write(f"  skip {label} B{B} (resumed)")
                continue
            if label in dead_b and B >= dead_b[label]:
                tqdm.write(f"  skip {label} B{B} (series OOM at B={dead_b[label]})")
                continue
            meta = dict(arch=args.arch, paradigm=paradigm, label=label,
                        n_steps=S or G, batch_size=B, gen_len=G, prompt_len=P,
                        dtype=args.precision, tp=TP_SIZE)
            _append_row(out_path, {**meta, "oom": True}, PARETO_FIELDS)
            try:
                prompt = jax.random.randint(jax.random.key(0), (B, P), 5,
                                            VOCAB, dtype=jnp.int32)
                if paradigm == "AR":
                    _, model = build_ar("mha", P + G,
                                        bf16=args.precision == "bf16")
                    fn, fargs = ar_gen_fused, (model, prompt)
                elif paradigm == "Discrete":
                    _, model = build_disc("mha", P + G,
                                          bf16=args.precision == "bf16")
                    x_mask = jnp.full((B, G), MASK_ID, dtype=jnp.int32)
                    fn = disc_gen_fused
                    fargs = (model, prompt, x_mask, jax.random.key(1),
                             _unmask_schedule(S))
                else:
                    _, model = build_elf("mha", G,
                                         bf16=args.precision == "bf16")
                    zdt = jnp.bfloat16 if args.precision == "bf16" else jnp.float32
                    z = jax.random.normal(jax.random.key(0), (B, G, DIM),
                                          dtype=zdt)
                    w = jnp.ones((B,), dtype=zdt)
                    ts = jnp.linspace(0.0, 1.0, S + 1, dtype=jnp.float32)
                    fn, fargs = elf_gen_fused, (model, z, w, ts)

                st = _time_stats(fn, *fargs, n_trials=args.n_e2e,
                                 desc=f"pareto {label} B{B}")
                joules, watts = _energy_of(fn, *fargs)
                _append_row(out_path, {**meta, **_stats_cols("e2e", st),
                            "tok_s_system": round(B * G * 1e3 / st["med"], 2),
                            "joules": round(joules, 3),
                            "j_per_tok": round(joules / (B * G), 6),
                            "watts": round(watts, 1),
                            "peak_mem_mb": round(_device_mem_mb(), 1),
                            "oom": False}, PARETO_FIELDS)
                del model
            except Exception as exc:  # noqa: BLE001
                log.warning("pareto %s B%d: %s", label, B, exc)
                gc.collect()
                break          # larger B will also OOM — stop this series
            gc.collect()
    return []


# ══ Ablation 3: serving-stack waterfall ════════════════════════════════════════

STACK_B, STACK_P, STACK_G, STACK_S = 4, 64, 128, 32
BLOCK_SIZE = 32


def _make_block_runner(bs: int, be: int, inner_steps: int) -> Any:
    """One fused program per block: build dual cache + inner denoise loop."""
    @nnx.jit
    def _run(model: Transformer, x_full: jnp.ndarray,
             x_blk: jnp.ndarray) -> jnp.ndarray:
        dual = model.compute_block_dual_cache(x_full, bs, be)
        start = jnp.asarray(bs, dtype=jnp.int32)

        def body(i: jax.Array, xb: jnp.ndarray) -> jnp.ndarray:
            logits = model.decode_block(xb, dual, start, deterministic=True)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        return jax.lax.fori_loop(0, inner_steps, body, x_blk)
    return _run


def run_stack(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    B, P, G, S = STACK_B, STACK_P, STACK_G, STACK_S
    unmask_ps = _unmask_schedule(S)
    prompt = jax.random.randint(jax.random.key(0), (B, P), 5, VOCAB,
                                dtype=jnp.int32)
    meta = dict(batch_size=B, prompt_len=P, gen_len=G, n_steps=S)

    def add(paradigm: str, variant: str, dtype: str, fn: Any, *fargs: Any) -> None:
        try:
            e2e = _time_call(fn, *fargs, n_trials=args.n_e2e,
                             desc=f"stack {variant}")
            rows.append({**meta, "paradigm": paradigm, "variant": variant,
                         "dtype": dtype, "e2e_ms": round(e2e, 2),
                         "tok_s_e2e": round(B * G * 1e3 / e2e, 2), "oom": False})
        except Exception as exc:  # noqa: BLE001
            log.warning("stack %s: %s", variant, exc)
            rows.append({**meta, "paradigm": paradigm, "variant": variant,
                         "dtype": dtype, "e2e_ms": NAN, "tok_s_e2e": NAN,
                         "oom": True})
        gc.collect()

    # ── AR: no-cache → +KV-cache → dtype ─────────────────────────────────
    _, m_fp32 = build_ar("mha", P + G, bf16=False)
    _, m_bf16 = build_ar("mha", P + G, bf16=True)
    add("AR", "AR/no-cache (fp32)",   "fp32", ar_gen_fused_nocache, m_fp32, prompt)
    add("AR", "AR/+kv-cache (fp32)",  "fp32", ar_gen_fused,         m_fp32, prompt)
    add("AR", "AR/+kv-cache +bf16",   "bf16", ar_gen_fused,         m_bf16, prompt)
    del m_fp32, m_bf16
    gc.collect()

    # ── Discrete: vanilla → +prefix-cache → +block-wise dual-cache ───────
    x_mask = jnp.full((B, G), MASK_ID, dtype=jnp.int32)
    x_full = jnp.concatenate([prompt, x_mask], axis=1)
    _, d_fp32 = build_disc("mha", P + G, bf16=False)
    _, d_bf16 = build_disc("mha", P + G, bf16=True)

    add("Discrete", "Disc/vanilla (fp32)", "fp32",
        disc_gen_fused_vanilla, d_fp32, x_full, jax.random.key(1), unmask_ps)
    add("Discrete", "Disc/+prefix-cache (fp32)", "fp32",
        disc_gen_fused, d_fp32, prompt, x_mask, jax.random.key(1), unmask_ps)

    # Block-wise Fast-dLLM: same S total network calls, each over BLOCK_SIZE
    # tokens (+ one cache rebuild per block).
    n_blocks = G // BLOCK_SIZE
    inner = S // n_blocks
    runners = [_make_block_runner(P + k * BLOCK_SIZE, P + (k + 1) * BLOCK_SIZE, inner)
               for k in range(n_blocks)]
    x_blk = jnp.full((B, BLOCK_SIZE), MASK_ID, dtype=jnp.int32)

    def _blockwise(model: Transformer) -> Any:
        o = None
        for r in runners:
            o = r(model, x_full, x_blk)
        return o

    add("Discrete", "Disc/+dual-cache blockwise (fp32)", "fp32", _blockwise, d_fp32)
    add("Discrete", "Disc/+dual-cache blockwise +bf16",  "bf16", _blockwise, d_bf16)
    del d_fp32, d_bf16
    gc.collect()

    # ── Continuous: fp32 → bf16 ───────────────────────────────────────────
    ts = jnp.linspace(0.0, 1.0, S + 1, dtype=jnp.float32)
    for lbl, bf16 in (("ELF/fp32", False), ("ELF/+bf16", True)):
        _, e_model = build_elf("mha", G, bf16=bf16)
        dt_ = jnp.bfloat16 if bf16 else jnp.float32
        z = jax.random.normal(jax.random.key(0), (B, G, DIM), dtype=dt_)
        w = jnp.ones((B,), dtype=dt_)
        add("Continuous", lbl, "bf16" if bf16 else "fp32",
            elf_gen_fused, e_model, z, w, ts)
        del e_model, z
        gc.collect()

    return rows


# ══ Ablation 4: memory ceiling ═════════════════════════════════════════════════

CEIL_G, CEIL_P = 512, 64
CEIL_START, CEIL_CAP = 32, 16384


def run_ceiling(args: argparse.Namespace) -> list[dict]:
    """Probe-level rows, appended to the CSV after every successful batch
    size so that a hard allocator crash (segfault) only truncates the series
    instead of losing it.  Use ``--series AR:mha`` to probe one combination
    per process (the crash-safe driver pattern)."""
    rows: list[dict] = []
    out_path = Path(args.out or f"results/ablation_ceiling_{args.arch}.csv")
    if not out_path.exists():
        out_path.write_text("paradigm,attn,gen_len,prompt_len,dtype,batch,tok_s\n")

    only = None
    if args.series:
        p_only, a_only = args.series.split(":")
        only = (p_only, a_only.lower())

    for paradigm in ("AR", "Discrete", "Continuous"):
        for attn in ("mha", "gqa", "mla"):
            if only and (paradigm, attn) != only:
                continue
            if paradigm == "AR":
                cfg, model = build_ar(attn, CEIL_P + CEIL_G, bf16=True)
            elif paradigm == "Discrete":
                cfg, model = build_disc(attn, CEIL_P + CEIL_G, bf16=True)
            else:
                cfg, model = build_elf(attn, CEIL_G, bf16=True)

            from benchmarks.paradigm_bench import disc_prefix
            use_prefix = attn != "mla"
            max_b, tok_s_at_max = 0, NAN
            B = CEIL_START
            while B <= CEIL_CAP:
                try:
                    if paradigm == "AR":
                        prompt = jnp.ones((B, CEIL_P), dtype=jnp.int32)
                        init = tuple((None, None) for _ in range(BLOCKS))
                        tok0 = jnp.ones((B, 1), dtype=jnp.int32)
                        _, cache = ar_prefill(model, prompt, init)
                        jax.block_until_ready(cache)
                        ms = _time_call(ar_decode, model, tok0, cache,
                                        jnp.array(CEIL_P, dtype=jnp.int32),
                                        n_trials=3)
                        del cache
                        tps = B * 1e3 / ms
                    elif paradigm == "Discrete":
                        x_mask = jnp.full((B, CEIL_G), MASK_ID, dtype=jnp.int32)
                        dual = None
                        if use_prefix:
                            dual = disc_prefix(
                                model, jnp.ones((B, CEIL_P), dtype=jnp.int32))
                            jax.block_until_ready(dual.prefix_kvs)
                        ms = _time_call(disc_step, model, x_mask, dual,
                                        jax.random.key(0), jnp.float32(0.05),
                                        n_trials=3)
                        del dual, x_mask
                        tps = B * CEIL_G * 1e3 / (32 * ms)
                    else:
                        z = jax.random.normal(jax.random.key(0),
                                              (B, CEIL_G, DIM), dtype=jnp.bfloat16)
                        xp = jnp.zeros_like(z)
                        w = jnp.ones((B,), dtype=jnp.bfloat16)
                        ms = _time_call(elf_step, model, z, xp, jnp.float32(0.5),
                                        jnp.float32(1 / 32), w, n_trials=3)
                        del z, xp
                        tps = B * CEIL_G * 1e3 / (32 * ms)

                    max_b, tok_s_at_max = B, tps
                    tqdm.write(f"  ceiling {paradigm:<10} {attn.upper():<3} "
                               f"B={B:<6} ok  ({tps:,.0f} tok/s steady)")
                    with out_path.open("a") as fh:   # crash-safe incremental log
                        fh.write(f"{paradigm},{attn.upper()},{CEIL_G},{CEIL_P},"
                                 f"bf16,{B},{tps:.1f}\n")
                    B *= 2
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"  ceiling {paradigm:<10} {attn.upper():<3} "
                               f"B={B:<6} OOM ({type(exc).__name__})")
                    break
                finally:
                    gc.collect()

            tqdm.write(f"  → ceiling {paradigm}/{attn.upper()}: "
                       f"max_batch={max_b}  ({tok_s_at_max:,.0f} tok/s)")
            del model
            gc.collect()
    # Probe rows are already on disk (incremental, crash-safe); nothing to
    # rewrite — returning [] tells main() to leave the CSV as-is.
    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation", choices=["grid", "stack", "ceiling", "pareto"])
    parser.add_argument("--arch", default="512d12b", choices=list(ARCHS))
    parser.add_argument("--precision", default="bf16",
                        choices=["f32", "tf32", "bf16"],
                        help="f32 = true fp32 matmuls; tf32 = TF32 tensor cores "
                             "(JAX default on A100); bf16 = bf16 weights+activations.")
    parser.add_argument("--retry-oom", action="store_true",
                        help="Re-attempt points whose last CSV row is an OOM "
                             "marker (run this on a fully free GPU); completed "
                             "points are still skipped.")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel degree (Megatron-style sharding "
                             "across the first N visible GPUs).")
    parser.add_argument("--series", default=None,
                        help="Ceiling only: probe a single 'Paradigm:attn' "
                             "combination (e.g. 'AR:mha') — crash isolation.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-e2e", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    _set_arch(args.arch)

    # Explicit matmul precision: on A100 JAX's default fp32 path uses TF32,
    # so "fp32" must be requested explicitly to mean true fp32.
    if args.precision == "f32":
        jax.config.update("jax_default_matmul_precision", "float32")
    elif args.precision == "tf32":
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    global MESH, TP_SIZE
    if args.tp > 1:
        from dantinox.core.sharding import make_tp_mesh
        if len(jax.devices()) < args.tp:
            parser.error(f"--tp {args.tp} but only {len(jax.devices())} devices visible")
        MESH = make_tp_mesh(args.tp)
        TP_SIZE = args.tp

    out = Path(args.out or f"results/ablation_{args.ablation}_{args.arch}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Paradigm ablation '{args.ablation}' on arch {args.arch} "
          f"({DIM}d × {BLOCKS}b, vocab {VOCAB})  precision={args.precision}  "
          f"tp={TP_SIZE}  [{jax.devices()[0].device_kind} × {len(jax.devices())}]")
    runner = {"grid": run_grid, "stack": run_stack,
              "ceiling": run_ceiling, "pareto": run_pareto}[args.ablation]
    rows = runner(args)

    if not rows:           # ceiling writes its probes incrementally
        print(f"\nIncremental probe rows in {out}")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df["arch"] = args.arch
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    with pd.option_context("display.width", 180, "display.max_columns", 30):
        print(df.to_string(index=False, max_rows=60))


if __name__ == "__main__":
    main()
