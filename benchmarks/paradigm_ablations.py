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
        **_attn_cfg(attn, ar_cache=True),
    )
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    if bf16:
        _cast_bf16(model)
    return cfg, model


def build_disc(attn: str, max_context: int, bf16: bool) -> tuple[ModelConfig, Transformer]:
    cfg = ModelConfig(
        dim=DIM, n_heads=N_HEADS, head_size=HEAD_SIZE, num_blocks=BLOCKS,
        vocab_size=VOCAB, max_context=max_context, causal=False, dropout=0.0,
        mask_token_id=MASK_ID, **_attn_cfg(attn, ar_cache=False),
    )
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    if bf16:
        _cast_bf16(model)
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


def run_grid(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    unmask_ps = _unmask_schedule(GRID_STEPS)
    dtype_lbl = "bf16"

    for B, G in tqdm([(b, g) for b in GRID_B for g in GRID_G], desc="grid"):
        P = GRID_P
        prompt = jax.random.randint(jax.random.key(0), (B, P), 5, VOCAB,
                                    dtype=jnp.int32)
        cell = dict(batch_size=B, gen_len=G, prompt_len=P, dtype=dtype_lbl)

        # ── AR (fused library path) ───────────────────────────────────────
        ar_e2e = NAN
        try:
            cfg, model = build_ar("mha", P + G, bf16=True)
            init_cache = tuple((None, None) for _ in range(BLOCKS))
            tok0 = jnp.ones((B, 1), dtype=jnp.int32)
            _, cache = ar_prefill(model, prompt, init_cache)
            jax.block_until_ready(cache)
            pos = jnp.array(P, dtype=jnp.int32)
            dec_gf, dec_gb = _cost(_ar_decode_fn, model, tok0, cache, pos)
            dec_ms = _time_call(ar_decode, model, tok0, cache, pos,
                                n_trials=args.n_trials)
            del cache
            ar_e2e = _time_call(ar_gen_fused, model, prompt,
                                n_trials=args.n_e2e, desc=f"AR e2e B{B} G{G}")
            rows.append({**cell, "paradigm": "AR",
                         "step_ms_p50": round(dec_ms, 4),
                         "step_gflops": round(dec_gf, 5),
                         "step_gbytes": round(dec_gb, 5),
                         "e2e_ms": round(ar_e2e, 3),
                         "tok_s_e2e": round(B * G * 1e3 / ar_e2e, 2),
                         "parity_steps": NAN, "speedup_at_32": NAN,
                         "peak_mem_mb": round(_device_mem_mb(), 1),
                         "oom": False})
            del model
        except Exception as exc:  # noqa: BLE001
            log.warning("grid AR B%d G%d: %s", B, G, exc)
            rows.append({**cell, "paradigm": "AR", "oom": True})
        gc.collect()

        # ── Discrete (fused) ──────────────────────────────────────────────
        try:
            cfg, model = build_disc("mha", P + G, bf16=True)
            x_mask = jnp.full((B, G), MASK_ID, dtype=jnp.int32)
            from benchmarks.paradigm_bench import disc_prefix  # jitted
            dual = disc_prefix(model, prompt)
            jax.block_until_ready(dual.prefix_kvs)
            gf, gb = _cost(_disc_step_fn, model, x_mask, dual,
                           jax.random.key(0), jnp.float32(0.05))
            step_ms = _time_call(disc_step, model, x_mask, dual,
                                 jax.random.key(0), jnp.float32(0.05),
                                 n_trials=args.n_trials)
            del dual
            e2e = _time_call(disc_gen_fused, model, prompt, x_mask,
                             jax.random.key(1), unmask_ps,
                             n_trials=args.n_e2e, desc=f"Disc e2e B{B} G{G}")
            rows.append({**cell, "paradigm": "Discrete",
                         "step_ms_p50": round(step_ms, 4),
                         "step_gflops": round(gf, 5), "step_gbytes": round(gb, 5),
                         "e2e_ms": round(e2e, 3),
                         "tok_s_e2e": round(B * G * 1e3 / e2e, 2),
                         "parity_steps": round(ar_e2e / step_ms, 2)
                             if ar_e2e == ar_e2e and step_ms > 0 else NAN,
                         "speedup_at_32": round(ar_e2e / e2e, 3)
                             if ar_e2e == ar_e2e else NAN,
                         "peak_mem_mb": round(_device_mem_mb(), 1),
                         "oom": False})
            del model
        except Exception as exc:  # noqa: BLE001
            log.warning("grid Disc B%d G%d: %s", B, G, exc)
            rows.append({**cell, "paradigm": "Discrete", "oom": True})
        gc.collect()

        # ── Continuous (fused) ────────────────────────────────────────────
        try:
            cfg, model = build_elf("mha", G, bf16=True)
            z = jax.random.normal(jax.random.key(0), (B, G, DIM),
                                  dtype=jnp.bfloat16)
            xp = jnp.zeros_like(z)
            w = jnp.ones((B,), dtype=jnp.bfloat16)
            gf, gb = _cost(_elf_step_fn, model, z, xp,
                           jnp.float32(0.5), jnp.float32(1 / GRID_STEPS), w)
            step_ms = _time_call(elf_step, model, z, xp, jnp.float32(0.5),
                                 jnp.float32(1 / GRID_STEPS), w,
                                 n_trials=args.n_trials)
            ts = jnp.linspace(0.0, 1.0, GRID_STEPS + 1, dtype=jnp.float32)
            e2e = _time_call(elf_gen_fused, model, z, w, ts,
                             n_trials=args.n_e2e, desc=f"Cont e2e B{B} G{G}")
            rows.append({**cell, "paradigm": "Continuous",
                         "step_ms_p50": round(step_ms, 4),
                         "step_gflops": round(gf, 5), "step_gbytes": round(gb, 5),
                         "e2e_ms": round(e2e, 3),
                         "tok_s_e2e": round(B * G * 1e3 / e2e, 2),
                         "parity_steps": round(ar_e2e / step_ms, 2)
                             if ar_e2e == ar_e2e and step_ms > 0 else NAN,
                         "speedup_at_32": round(ar_e2e / e2e, 3)
                             if ar_e2e == ar_e2e else NAN,
                         "peak_mem_mb": round(_device_mem_mb(), 1),
                         "oom": False})
            del model, z, xp
        except Exception as exc:  # noqa: BLE001
            log.warning("grid Cont B%d G%d: %s", B, G, exc)
            rows.append({**cell, "paradigm": "Continuous", "oom": True})
        gc.collect()

    return rows


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
    rows: list[dict] = []

    for paradigm in ("AR", "Discrete", "Continuous"):
        for attn in ("mha", "gqa", "mla"):
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
                    B *= 2
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"  ceiling {paradigm:<10} {attn.upper():<3} "
                               f"B={B:<6} OOM ({type(exc).__name__})")
                    break
                finally:
                    gc.collect()

            rows.append({"paradigm": paradigm, "attn": attn.upper(),
                         "gen_len": CEIL_G, "prompt_len": CEIL_P, "dtype": "bf16",
                         "max_batch": max_b, "tok_s_at_max": round(tok_s_at_max, 1),
                         "hit_cap": max_b >= CEIL_CAP})
            del model
            gc.collect()
    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation", choices=["grid", "stack", "ceiling"])
    parser.add_argument("--arch", default="512d12b", choices=list(ARCHS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-e2e", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    _set_arch(args.arch)
    out = Path(args.out or f"results/ablation_{args.ablation}_{args.arch}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Paradigm ablation '{args.ablation}' on arch {args.arch} "
          f"({DIM}d × {BLOCKS}b, vocab {VOCAB})  [{jax.devices()[0].device_kind}]")
    runner = {"grid": run_grid, "stack": run_stack, "ceiling": run_ceiling}[args.ablation]
    rows = runner(args)

    import pandas as pd
    df = pd.DataFrame(rows)
    df["arch"] = args.arch
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    with pd.option_context("display.width", 180, "display.max_columns", 30):
        print(df.to_string(index=False, max_rows=60))


if __name__ == "__main__":
    main()
