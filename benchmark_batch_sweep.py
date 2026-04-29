"""
benchmark_batch_sweep.py — decode throughput vs batch size for one
representative run per attention type.

Purpose: show the batch-size crossover where MLA's smaller KV cache
starts paying off in throughput (MLA allows more sequences in VRAM).

Outputs: batch_sweep_results.csv
Columns: run, type, dim, num_blocks, params_m, kv_heads, n_heads,
         down_dim_kv, theoretical_cache_mb,
         batch_size, seq_len, tps, cache_mb_total, oom

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 benchmark_batch_sweep.py
"""

import os, time, gc
import dataclasses
import yaml, json, msgpack
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from core.config import Config
from core.model import Transformer
from benchmark_analysis import (
    _load_config, _load_model, _attn_type,
    _theoretical_kv_cache_mb, _detect_actual_vocab,
)

# ── Which runs to sweep ──────────────────────────────────────────────────────
# One per type, same architecture family (dim=256, num_blocks=12, Dense)
SWEEP_RUNS = [
    "standard_gqa_256d_12b_Dense_005059",   # GQA  ~10.7M
    "standard_mha_256d_12b_Dense_120714",   # MHA  ~13.0M
    "mla_256d_12b_Dense_110023",             # MLA  ~13.3M
    # Larger scale (dim=512, num_blocks=12)
    "standard_gqa_512d_12b_Dense_111603",   # GQA  ~39M
    "standard_mha_512d_12b_Dense_080826",   # MHA  ~45M
    "mla_512d_12b_Dense_110117",             # MLA  ~53M
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SEQ_LEN     = 512     # fixed context length for the batch sweep
N_WARMUP    = 3
N_MEASURE   = 20
RUNS_DIR    = "runs"
OUT_CSV     = "batch_sweep_results.csv"


@nnx.jit
def _decode_step(model, tok, cache, idx):
    return model(tok, use_cache=True, kv_caches=cache, cache_index=idx)


def _bench_one(run_name: str, batch_size: int) -> dict | None:
    """Returns a result dict or None on OOM / error."""
    run_path = os.path.join(RUNS_DIR, run_name)
    config   = _load_config(run_path)

    if SEQ_LEN > config.max_context:
        return None

    if config.mla:
        config.inference = True

    try:
        model = _load_model(run_path, config)
    except Exception as e:
        print(f"    load failed: {e}")
        return None

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
        # Initialise KV cache (index=0)
        cache = None
        _, cache, _ = _decode_step(model, tok, cache, 0)
        jax.block_until_ready(cache)

        # Measure cache memory
        dev         = jax.devices()[0]
        mem_stats   = dev.memory_stats()
        mem_in_use  = mem_stats.get("bytes_in_use", 0)

        # Warmup
        for i in range(1, N_WARMUP + 1):
            _, cache, _ = _decode_step(model, tok, cache, i)
        jax.block_until_ready(cache)

        # Timed loop
        t0 = time.time()
        for i in range(N_WARMUP + 1, N_WARMUP + 1 + N_MEASURE):
            _, cache, _ = _decode_step(model, tok, cache,
                                       min(i, SEQ_LEN - 1))
        jax.block_until_ready(cache)
        t1 = time.time()

        tps = round(N_MEASURE * batch_size / (t1 - t0), 2)

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
            "batch_size": batch_size, "seq_len": SEQ_LEN,
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
        "batch_size": batch_size, "seq_len": SEQ_LEN,
        "tps": tps, "cache_mb_total": cache_mb, "oom": False,
    }


if __name__ == "__main__":
    runs_present = [
        r for r in SWEEP_RUNS
        if os.path.isdir(os.path.join(RUNS_DIR, r))
        and os.path.exists(os.path.join(RUNS_DIR, r, "model_weights.msgpack"))
    ]
    missing = set(SWEEP_RUNS) - set(runs_present)
    if missing:
        print(f"Missing runs (will skip): {missing}")

    results = []
    total   = len(runs_present) * len(BATCH_SIZES)
    done    = 0

    for run in runs_present:
        print(f"\n{'='*60}")
        print(f"Run: {run}")
        for bs in BATCH_SIZES:
            done += 1
            print(f"  [{done}/{total}] bs={bs:3d} ...", end=" ", flush=True)
            row = _bench_one(run, bs)
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
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {len(df)} rows → {OUT_CSV}")
    else:
        print("No results collected.")
