#!/usr/bin/env python3
"""
benchmarks/trained_batch_sweep.py
===================================
Decode throughput vs batch size for one representative run per attention type.

Purpose: show the batch-size crossover where MLA's smaller KV cache starts
paying off in throughput (MLA allows more sequences in VRAM).

Outputs
-------
<out-csv>  — CSV with columns:
  run, type, dim, num_blocks, params_m, kv_heads, n_heads,
  down_dim_kv, theoretical_cache_mb,
  batch_size, seq_len, tps, cache_mb_total, oom

Usage
-----
  python benchmarks/trained_batch_sweep.py
  python benchmarks/trained_batch_sweep.py --runs run1 run2
  python benchmarks/trained_batch_sweep.py --batch-sizes 1 2 4 8 --seq-len 256
  python benchmarks/trained_batch_sweep.py --analysis-csv results/benchmark_results.csv
  python benchmarks/trained_batch_sweep.py --device 1

  # Via run_all.py
  python benchmarks/run_all.py --trained

  # Via CLI
  dantinox infbench --trained
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx

from benchmarks.trained_analysis import (
    _attn_type,
    _load_config,
    _load_model,
    _theoretical_kv_cache_mb,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_RUNS = [
    "standard_gqa_256d_12b_Dense_005059",
    "standard_mha_256d_12b_Dense_120714",
    "mla_256d_12b_Dense_110023",
    "standard_gqa_512d_12b_Dense_111603",
    "standard_mha_512d_12b_Dense_080826",
    "mla_512d_12b_Dense_110117",
]

_DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
_N_WARMUP  = 3
_N_MEASURE = 20


# ── Benchmark kernel ─────────────────────────────────────────────────────────

@nnx.jit
def _decode_step(model, tok, cache, idx):
    return model(tok, caches=cache, cache_index=idx)


def _bench_one(
    run_name: str,
    batch_size: int,
    seq_len: int,
    runs_dir: str,
) -> dict | None:
    """Benchmark one (run, batch_size) combination. Returns None to skip."""
    run_path = os.path.join(runs_dir, run_name)
    config   = _load_config(run_path)

    if seq_len > config.max_context:
        return None

    if config.mla:
        config.inference = True

    try:
        model = _load_model(run_path, config)
    except Exception as e:
        print(f"    load failed: {e}")
        return None

    import json
    attn_type = _attn_type(config)
    params_m  = 0
    summary   = os.path.join(run_path, "model_summary.json")
    if os.path.exists(summary):
        try:
            params_m = json.load(open(summary)).get("total_params_M", 0)
        except Exception:
            pass

    tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)

    try:
        cache = None
        _, cache, _ = _decode_step(model, tok, cache, 0)
        jax.block_until_ready(cache)

        dev       = jax.devices()[0]
        mem_stats = dev.memory_stats()
        mem_in_use = mem_stats.get("bytes_in_use", 0)

        for i in range(1, _N_WARMUP + 1):
            _, cache, _ = _decode_step(model, tok, cache, i)
        jax.block_until_ready(cache)

        t0 = time.time()
        for i in range(_N_WARMUP + 1, _N_WARMUP + 1 + _N_MEASURE):
            _, cache, _ = _decode_step(model, tok, cache, min(i, seq_len - 1))
        jax.block_until_ready(cache)
        t1 = time.time()

        tps      = round(_N_MEASURE * batch_size / (t1 - t0), 2)
        cache_mb = round(mem_in_use / 1e6, 2)

    except Exception as e:
        print(f"    OOM or error (bs={batch_size}): {e}")
        return {
            "run": run_name, "type": attn_type,
            "dim": config.dim, "num_blocks": config.num_blocks,
            "params_m": params_m, "kv_heads": config.kv_heads,
            "n_heads": config.n_heads,
            "down_dim_kv": getattr(config, "down_dim_kv", None),
            "theoretical_cache_mb": round(_theoretical_kv_cache_mb(config), 2),
            "batch_size": batch_size, "seq_len": seq_len,
            "tps": float("nan"), "cache_mb_total": float("nan"), "oom": True,
        }
    finally:
        del model, cache
        gc.collect()
        jax.clear_caches()

    return {
        "run": run_name, "type": attn_type,
        "dim": config.dim, "num_blocks": config.num_blocks,
        "params_m": params_m, "kv_heads": config.kv_heads,
        "n_heads": config.n_heads,
        "down_dim_kv": getattr(config, "down_dim_kv", None),
        "theoretical_cache_mb": round(_theoretical_kv_cache_mb(config), 2),
        "batch_size": batch_size, "seq_len": seq_len,
        "tps": tps, "cache_mb_total": cache_mb, "oom": False,
    }


# ── Auto-select runs from analysis CSV ───────────────────────────────────────

def _select_from_csv(analysis_csv: str) -> list[str]:
    """Pick one representative run per attention type from an analysis CSV."""
    try:
        df = pd.read_csv(analysis_csv)
    except Exception as e:
        print(f"[batch_sweep] Could not read {analysis_csv}: {e}", file=sys.stderr)
        return []

    if "run" not in df.columns or "type" not in df.columns:
        return []

    selected = []
    for attn_type, grp in df.groupby("type"):
        # Pick the run with the most params (representative of the family)
        if "params_m" in grp.columns:
            row = grp.loc[grp["params_m"].idxmax()]
        else:
            row = grp.iloc[0]
        selected.append(row["run"])
        print(f"  auto-selected {row['run']}  ({attn_type})")
    return selected


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trained_batch_sweep",
        description="Decode throughput vs batch size for trained DantinoX models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--runs-dir", default="runs", metavar="DIR",
        help="Directory containing run subdirectories (default: runs)",
    )
    parser.add_argument(
        "--runs", nargs="+", metavar="RUN",
        help="Specific run names to sweep (default: built-in list or auto-selected)",
    )
    parser.add_argument(
        "--run-prefix", nargs="+", default=["ar_", "diff_"], metavar="PREFIX",
        help="When auto-selecting from CSV, only include runs with these prefixes "
             "(default: ar_ diff_). Prevents mixing runs trained on different datasets.",
    )
    parser.add_argument(
        "--analysis-csv", default=None, metavar="PATH",
        help="If --runs is not given, auto-select one run per attention type "
             "from this analysis CSV (produced by trained_analysis.py)",
    )
    parser.add_argument(
        "--out-csv", default="results/batch_sweep_results.csv", metavar="PATH",
        help="Output CSV path (default: results/batch_sweep_results.csv)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=_DEFAULT_BATCH_SIZES, metavar="N",
        help="Batch sizes to sweep (default: 1 2 4 8 16 32 64)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, metavar="N",
        help="Fixed context length for the sweep (default: 512)",
    )
    parser.add_argument(
        "--device", default=None, metavar="N",
        help="CUDA device index — sets CUDA_VISIBLE_DEVICES (default: inherit)",
    )

    args = parser.parse_args(argv)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Resolve which runs to sweep
    _prefixes = tuple(p for p in (args.run_prefix or []) if p)
    if args.runs:
        sweep_runs = args.runs
    elif args.analysis_csv:
        print(f"Auto-selecting runs from {args.analysis_csv}:")
        sweep_runs = _select_from_csv(args.analysis_csv)
        if not sweep_runs:
            print("[batch_sweep] No runs found in analysis CSV — falling back to defaults.")
            sweep_runs = _DEFAULT_RUNS
    else:
        sweep_runs = _DEFAULT_RUNS

    # Apply prefix filter
    if _prefixes and not args.runs:
        sweep_runs = [r for r in sweep_runs if any(r.startswith(p) for p in _prefixes)]

    runs_dir = args.runs_dir
    runs_present = [
        r for r in sweep_runs
        if os.path.isdir(os.path.join(runs_dir, r))
        and os.path.exists(os.path.join(runs_dir, r, "model_weights.msgpack"))
    ]
    missing = set(sweep_runs) - set(runs_present)
    if missing:
        print(f"Missing runs (will skip): {missing}")
    if not runs_present:
        print("[batch_sweep] No valid runs found. Exiting.")
        sys.exit(1)

    # Ensure output directory exists
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(runs_present) * len(args.batch_sizes)
    done  = 0

    for run in runs_present:
        print(f"\n{'='*60}")
        print(f"Run: {run}")
        for bs in sorted(args.batch_sizes):
            done += 1
            print(f"  [{done}/{total}] bs={bs:3d} ...", end=" ", flush=True)
            row = _bench_one(run, bs, args.seq_len, runs_dir)
            if row is None:
                print("skip (seq_len > max_context)")
                continue
            results.append(row)
            if row["oom"]:
                print("OOM — stopping batch sweep for this run")
                break
            print(f"tps={row['tps']:.1f}  cache={row['cache_mb_total']:.0f} MB")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved {len(df)} rows → {args.out_csv}")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
