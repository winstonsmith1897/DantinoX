#!/usr/bin/env python3
"""
benchmarks/confidence_sweep.py
================================

Confidence-aware parallel decoding sweep for DantinoX Diffusion models.

Sweeps two decoding strategies (Fast-dLLM §3.3) across their key hyper-
parameters, measuring the speed/quality tradeoff on randomly initialised models
(no training required).

Strategies
----------
threshold   Unmask all masked tokens whose max-softmax confidence ≥ τ.
            τ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99}

factor      Find the largest n s.t. (n+1)(1 − c_(n)) < f  (Theorem 1).
            f ∈ {0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0}

For each (strategy, hyper-param) × {MHA, GQA, MLA} × {seq_len ∈ 64, 128, 256}:

Metrics
-------
avg_steps_to_complete   average denoising steps needed to unmask all tokens
                        across N_RUNS independent generation trajectories
                        (lower = faster decoding)
avg_tok_per_step        average masked tokens revealed per step
tok_s                   estimated throughput: (seq_len × 1000) / (steps × step_ms)
step_ms_p50             single denoising step latency (pre-computed from
                        diffusion_ar_sweep or measured here)

Output CSV columns
------------------
  strategy, param, attn_variant, seq_len, model_size,
  avg_steps_to_complete, avg_tok_per_step,
  tok_s, step_ms_p50, params_m

Usage
-----
  python benchmarks/confidence_sweep.py
  python benchmarks/confidence_sweep.py --seq-lens 64 128 --n-runs 20
  python benchmarks/confidence_sweep.py --out results/confidence_sweep.csv
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from dantinox.core.config import Config
from dantinox.core.diffusion import (
    NoiseSchedule,
    confidence_unmask_factor,
    confidence_unmask_threshold,
    corrupt,
    make_noise_schedule,
)
from dantinox.core.model import DiffusionTransformer

log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_conf"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

# ── Sweep parameters ──────────────────────────────────────────────────────────

_THRESHOLD_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
_FACTOR_VALUES    = [0.80, 1.00, 1.20, 1.50, 2.00, 3.00, 5.00]
_SEQ_LENS         = [64, 128, 256]
_N_RUNS           = 30      # trajectories per configuration
_N_WARMUP         = 3
_N_MEASURE        = 10
_VOCAB_SIZE       = 256

# ── Model configs ─────────────────────────────────────────────────────────────

def _make_config(
    attn_variant: str,
    max_context: int,
    *,
    dim: int = 256, n_heads: int = 8, head_size: int = 32, num_blocks: int = 6,
) -> Config:
    base: dict[str, Any] = dict(
        dim=dim, n_heads=n_heads, head_size=head_size, num_blocks=num_blocks,
        vocab_size=_VOCAB_SIZE, max_context=max_context,
        use_moe=False, use_swiglu=True, use_rotary_pos=True,
        mla=False, inference=False, down_dim_q=64, down_dim_kv=64, rope_dim=16,
        dropout_rate=0.0, gradient_checkpointing=False, weight_tying=True,
        diffusion_steps=100, noise_schedule="cosine", mask_token_id=0,
        num_sampling_steps=20, time_emb_dim=128, model_type="diffusion",
    )
    if attn_variant == "GQA":
        base["kv_heads"] = max(1, n_heads // 4)
    elif attn_variant == "MLA":
        base["kv_heads"] = n_heads
        base["mla"]      = True
        base["inference"] = False
    else:  # MHA
        base["kv_heads"] = n_heads
    return Config.from_dict(base)


# ── JIT denoising step ────────────────────────────────────────────────────────

@nnx.jit
def _diff_step(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    out = model(x_t, dual_cache=None, deterministic=True)  # type: ignore[call-arg]
    return out.logits


# ── Step latency measurement ──────────────────────────────────────────────────

def _measure_step_ms(
    model: nnx.Module,
    seq_len: int,
    config: Config,
    n_warmup: int,
    n_trials: int,
) -> float:
    x_t = jnp.zeros((1, seq_len), dtype=jnp.int32)
    t   = jnp.array([config.diffusion_steps // 2], dtype=jnp.int32)

    # First call triggers JIT
    jax.block_until_ready(_diff_step(model, x_t, t))
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(_diff_step(model, x_t, t))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(_diff_step(model, x_t, t))
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ── Simulate one generation trajectory ───────────────────────────────────────

def _simulate_trajectory(
    model: nnx.Module,
    config: Config,
    schedule: NoiseSchedule,
    seq_len: int,
    strategy: str,
    param: float,
    rng: jax.Array,
) -> tuple[int, float]:
    """Return (steps_to_complete, avg_tok_per_step) for one trajectory."""
    T_diff    = config.diffusion_steps
    step_size = max(1, T_diff // config.num_sampling_steps)

    x_t = jnp.full((1, seq_len), config.mask_token_id, dtype=jnp.int32)
    steps   = 0
    total_unmasked = 0

    for t_val in range(T_diff, 0, -step_size):
        if not (x_t == config.mask_token_id).any():
            break
        t    = jnp.array([t_val], dtype=jnp.int32)
        logits = _diff_step(model, x_t, t)

        if strategy == "threshold":
            x_new = confidence_unmask_threshold(logits, x_t, config.mask_token_id, threshold=param)
        else:  # factor
            x_new = confidence_unmask_factor(logits, x_t, config.mask_token_id, factor=param)

        unmasked_this_step = int(((x_t == config.mask_token_id) & (x_new != config.mask_token_id)).sum())
        total_unmasked += unmasked_this_step
        x_t   = x_new
        steps += 1

    # Force-complete any remaining masks
    if (x_t == config.mask_token_id).any():
        t_zero  = jnp.zeros((1,), dtype=jnp.int32)
        logits  = _diff_step(model, x_t, t_zero)
        x_t     = jnp.where(x_t == config.mask_token_id, jnp.argmax(logits, axis=-1), x_t)
        steps  += 1

    avg_per_step = total_unmasked / max(steps, 1)
    return steps, avg_per_step


# ── One experiment ────────────────────────────────────────────────────────────

def run_one(
    strategy: str,
    param: float,
    attn_variant: str,
    seq_len: int,
    n_runs: int,
    n_warmup: int,
    n_measure: int,
) -> dict:
    nan = float("nan")
    max_context = seq_len + 4
    try:
        config = _make_config(attn_variant, max_context)
        model  = DiffusionTransformer(config, rngs=nnx.Rngs(42))
    except Exception as exc:
        log.warning("Config error [%s/%s/%s/%d]: %s", strategy, param, attn_variant, seq_len, exc)
        return _oom_row(strategy, param, attn_variant, seq_len, nan)

    schedule = make_noise_schedule(config)
    _, state = nnx.split(model)
    params_m = sum(x.size for x in jax.tree_util.tree_leaves(state) if hasattr(x, "size")) / 1e6

    # Measure step latency
    try:
        step_ms = _measure_step_ms(model, seq_len, config, n_warmup, n_measure)
    except Exception:
        step_ms = nan

    # Simulate trajectories
    steps_list: list[int]   = []
    tok_per_step: list[float] = []
    rng = jax.random.key(42)
    for i in range(n_runs):
        rng, sub = jax.random.split(rng)
        try:
            s, avg = _simulate_trajectory(model, config, schedule, seq_len, strategy, param, sub)
            steps_list.append(s)
            tok_per_step.append(avg)
        except Exception as exc:
            log.debug("Trajectory %d failed: %s", i, exc)

    if not steps_list:
        return _oom_row(strategy, param, attn_variant, seq_len, params_m)

    avg_steps = float(np.mean(steps_list))
    avg_tok   = float(np.mean(tok_per_step))
    tok_s     = (seq_len * 1_000.0) / (avg_steps * step_ms) if not np.isnan(step_ms) and avg_steps > 0 else nan

    return {
        "strategy":             strategy,
        "param":                param,
        "attn_variant":         attn_variant,
        "seq_len":              seq_len,
        "model_size":           "medium",
        "avg_steps_to_complete": round(avg_steps, 2),
        "avg_tok_per_step":     round(avg_tok, 3),
        "tok_s":                round(tok_s, 2),
        "step_ms_p50":          round(step_ms, 3),
        "params_m":             round(params_m, 3),
        "oom":                  False,
    }


def _oom_row(
    strategy: str, param: float, attn: str, seq_len: int, params_m: float
) -> dict:
    nan = float("nan")
    return {
        "strategy": strategy, "param": param, "attn_variant": attn,
        "seq_len": seq_len, "model_size": "medium",
        "avg_steps_to_complete": nan, "avg_tok_per_step": nan,
        "tok_s": nan, "step_ms_p50": nan, "params_m": params_m, "oom": True,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Confidence-aware decoding sweep for DantinoX Diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out",        default="results/confidence_sweep.csv")
    parser.add_argument("--seq-lens",   nargs="+", type=int, default=_SEQ_LENS)
    parser.add_argument("--n-runs",     type=int, default=_N_RUNS,
                        help=f"Trajectories per config (default: {_N_RUNS})")
    parser.add_argument("--n-warmup",   type=int, default=_N_WARMUP)
    parser.add_argument("--n-measure",  type=int, default=_N_MEASURE)
    parser.add_argument("--attn",       nargs="+", default=["MHA", "GQA", "MLA"],
                        choices=["MHA", "GQA", "MLA"])
    parser.add_argument("--no-mla",     action="store_true")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    attn_variants = [a for a in args.attn if not (args.no_mla and a == "MLA")]

    # Build experiment list
    experiments: list[tuple[str, float, str, int]] = []
    for seq in args.seq_lens:
        for attn in attn_variants:
            for τ in _THRESHOLD_VALUES:
                experiments.append(("threshold", τ, attn, seq))
            for f in _FACTOR_VALUES:
                experiments.append(("factor", f, attn, seq))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DantinoX confidence sweep — {len(experiments)} experiments")
    print(f"  device   : {jax.default_backend()}")
    print(f"  seq_lens : {args.seq_lens}")
    print(f"  attn     : {attn_variants}")
    print(f"  n_runs   : {args.n_runs}")
    print(f"  output   : {out_path}")
    print()

    rows: list[dict] = []
    for strategy, param, attn, seq in tqdm(experiments, desc="sweep", unit="exp"):
        row = run_one(strategy, param, attn, seq, args.n_runs, args.n_warmup, args.n_measure)
        rows.append(row)
        if args.verbose:
            tqdm.write(
                f"  [{strategy:<9}] param={param:<5}  [{attn}] seq={seq:<4}  "
                f"steps={row['avg_steps_to_complete']:<6.1f}  "
                f"tok/step={row['avg_tok_per_step']:<5.2f}  "
                f"tok/s={row['tok_s']:<8.1f}"
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
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        out_path.write_text(buf.getvalue())
        print(f"\nSaved {len(rows)} rows → {out_path}")


def _print_summary(df: Any) -> None:
    nan = float("nan")
    print("\n── Threshold strategy: avg steps to complete (MHA, seq=64) ───────────")
    sub = df[(df["strategy"] == "threshold") & (df["attn_variant"] == "MHA") & (df["seq_len"] == 64)]
    sub = sub.sort_values("param")
    for _, row in sub.iterrows():
        bar = "█" * int(min(40, row["avg_steps_to_complete"]))
        print(f"  τ={row['param']:.2f}  steps={row['avg_steps_to_complete']:<6.1f}  "
              f"tok/s={row['tok_s']:<8.1f}  {bar}")
    print("\n── Factor strategy: avg steps to complete (MHA, seq=64) ──────────────")
    sub = df[(df["strategy"] == "factor") & (df["attn_variant"] == "MHA") & (df["seq_len"] == 64)]
    sub = sub.sort_values("param")
    for _, row in sub.iterrows():
        bar = "█" * int(min(40, row["avg_steps_to_complete"]))
        print(f"  f={row['param']:<5.2f}  steps={row['avg_steps_to_complete']:<6.1f}  "
              f"tok/s={row['tok_s']:<8.1f}  {bar}")


if __name__ == "__main__":
    main()
