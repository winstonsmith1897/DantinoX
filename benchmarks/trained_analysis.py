#!/usr/bin/env python3
"""
benchmarks/trained_analysis.py
================================
Benchmark real DantinoX model runs stored in a runs/ directory.

Measures per run:
  • decode throughput at multiple sequence lengths (tok/s)
  • prefill latency at max_context (ms)
  • KV cache — theoretical (formula) and measured (VRAM delta, MB)
  • FLOPs & arithmetic intensity via XLA cost analysis
  • Final validation loss from training_log.csv

Outputs:
  <out-csv>   — one row per run (default: benchmark_results.csv)
  <out-plot>  — grouped-bar + line + scatter overview figure

Already-benchmarked runs are loaded from <out-csv> and skipped so the
script can be interrupted and resumed incrementally.

Usage:
  python benchmarks/trained_analysis.py
  python benchmarks/trained_analysis.py --runs-dir runs/ --out-csv results/benchmark_results.csv
  python benchmarks/trained_analysis.py --device 1
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import pandas as pd
import yaml
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.model import Transformer

SEQ_LENS  = [64, 128, 256, 512]
N_WARMUP  = 3
N_MEASURE = 20

TYPE_COLORS = {"MLA": "#4C9BE8", "GQA": "#E87B4C", "MHA": "#4CE87B"}
TYPE_ORDER  = ["MLA", "GQA", "MHA"]


# ─── JIT step ────────────────────────────────────────────────────────────────

@nnx.jit
def _decode_step(model: Transformer, tok: jnp.ndarray,
                 cache: tuple | None, idx: int) -> tuple:
    return model(tok, use_cache=True, kv_caches=cache, cache_index=idx)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_config(run_path: str) -> Config:
    with open(os.path.join(run_path, "config.yaml")) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    if not flat:
        flat = raw
    valid = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in flat.items() if k in valid})


def _detect_actual_vocab(state_dict: dict, dim: int) -> int | None:
    def _get(d: Any, key: str) -> Any:
        if not isinstance(d, dict):
            return None
        return d.get(key) or d.get(key.encode() if isinstance(key, str) else key)

    def _unwrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k in ("value", "raw_value", b"value", b"raw_value"):
                if k in obj:
                    return obj[k]
        return obj

    wte = _get(state_dict, "wte")
    if wte is None:
        return None
    emb = _unwrap(_get(wte, "embedding"))
    if emb is None or not hasattr(emb, "shape") or emb.ndim != 2:
        return None
    return int(emb.shape[0]) if emb.shape[1] == dim else (
        int(emb.shape[1]) if emb.shape[0] == dim else None
    )


def _load_model(run_path: str, config: Config) -> Transformer:
    weights_path = os.path.join(run_path, "model_weights.msgpack")
    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack,
                                     strict_map_key=False)
    actual_vocab = _detect_actual_vocab(state_dict, config.dim)
    if actual_vocab is not None and actual_vocab != config.vocab_size:
        config = dataclasses.replace(config, vocab_size=actual_vocab)
    model = Transformer(config, rngs=nnx.Rngs(42))
    nnx.update(model, state_dict)
    return model


def _attn_type(config: Config) -> str:
    if getattr(config, "mla", False):
        return "MLA"
    if getattr(config, "kv_heads", config.n_heads) < config.n_heads:
        return "GQA"
    return "MHA"


def _theoretical_kv_cache_mb(config: Config) -> float:
    S = config.max_context
    if getattr(config, "mla", False):
        per_layer = S * (config.down_dim_kv + config.rope_dim) * 4
    else:
        hs = config.dim // config.n_heads
        per_layer = 2 * S * getattr(config, "kv_heads", config.n_heads) * hs * 4
    return per_layer * config.num_blocks / 1e6


def _val_loss(run_path: str) -> float | None:
    log = os.path.join(run_path, "training_log.csv")
    if not os.path.exists(log):
        return None
    try:
        return float(pd.read_csv(log)["val_loss"].dropna().iloc[-1])
    except Exception:
        return None


def _xla_costs(fn: Any, *args: Any) -> tuple[float, float]:
    try:
        costs = fn.lower(*args).cost_analysis()
        if isinstance(costs, list):
            flops = sum(c.get("flops", 0) for c in costs)
            mem   = sum(c.get("bytes accessed", 0) for c in costs)
        else:
            flops = costs.get("flops", float("nan"))
            mem   = costs.get("bytes accessed", float("nan"))
        return float(flops), float(mem)
    except Exception:
        return float("nan"), float("nan")


def _family_key(row: Any) -> str:
    moe = "MoE" if row["moe"] else "Dense"
    return f"L{row['num_blocks']}_D{row['dim']}_H{row['n_heads']}_C{row['max_context']}_{moe}"


# ─── Per-run benchmark ────────────────────────────────────────────────────────

def run_benchmark(run_name: str, runs_dir: str) -> dict:
    run_path = os.path.join(runs_dir, run_name)
    config   = _load_config(run_path)
    if config.mla:
        config.inference = True
    model = _load_model(run_path, config)

    params_m = 0.0
    summary  = os.path.join(run_path, "model_summary.json")
    if os.path.exists(summary):
        try:
            params_m = json.load(open(summary)).get("total_params_M", 0)
        except Exception:
            pass

    tok    = jnp.zeros((1, 1), dtype=jnp.int32)
    prompt = jnp.zeros((1, config.max_context), dtype=jnp.int32)

    # ── Throughput at each sequence length ────────────────────────────────────
    tps_by_seq: dict[int, float] = {}
    for seq in SEQ_LENS:
        if seq > config.max_context:
            tps_by_seq[seq] = float("nan")
            continue
        cache = None
        _, cache, _ = _decode_step(model, tok, cache, 0)
        for i in range(1, N_WARMUP + 1):
            _, cache, _ = _decode_step(model, tok, cache, i)
        jax.block_until_ready(cache)
        t0 = time.time()
        for i in range(N_WARMUP + 1, N_WARMUP + 1 + N_MEASURE):
            _, cache, _ = _decode_step(model, tok, cache, min(i, seq - 1))
        jax.block_until_ready(cache)
        tps_by_seq[seq] = round(N_MEASURE / (time.time() - t0), 2)

    # ── Prefill latency ───────────────────────────────────────────────────────
    @nnx.jit
    def _prefill(m: Transformer, x: jnp.ndarray) -> tuple:
        return m(x, use_cache=False, kv_caches=None, cache_index=0)

    _prefill(model, prompt)
    jax.block_until_ready(_prefill(model, prompt))
    t0 = time.time()
    jax.block_until_ready(_prefill(model, prompt))
    prefill_ms = round((time.time() - t0) * 1000, 2)

    # ── Measured KV cache VRAM ────────────────────────────────────────────────
    jax.clear_caches()
    dev  = jax.devices()[0]
    vram_before = dev.memory_stats().get("bytes_in_use", 0)
    cache = None
    _, cache, _ = _decode_step(model, tok, cache, 0)
    jax.block_until_ready(cache)
    vram_cache_mb = round(max(0, dev.memory_stats().get("bytes_in_use", 0) - vram_before) / 1e6, 2)

    # ── FLOPs & arithmetic intensity ──────────────────────────────────────────
    mid_idx      = min(config.max_context // 2, config.max_context - 1)
    cache_mid: tuple = tuple((None, None) for _ in range(config.num_blocks))
    _decode_jit = nnx.jit(lambda m, t, c, i: m(t, use_cache=True, kv_caches=c, cache_index=i))
    _prefill_jit = nnx.jit(lambda m, x: m(x, use_cache=False, kv_caches=None, cache_index=0))

    decode_flops,  decode_bytes  = _xla_costs(_decode_jit,  model, tok,    cache_mid, mid_idx)
    prefill_flops, prefill_bytes = _xla_costs(_prefill_jit, model, prompt)

    def _safe_r(v: float, d: float) -> float:
        return round(v / d, 4) if not (np.isnan(v) or np.isnan(d) or d == 0) else float("nan")

    decode_gflops   = _safe_r(decode_flops,  1e9)
    prefill_gflops  = _safe_r(prefill_flops, 1e9)
    best_tps        = tps_by_seq.get(max(s for s in SEQ_LENS if s <= config.max_context), float("nan"))
    decode_tflops_s = _safe_r(decode_gflops * best_tps, 1e3) if not np.isnan(best_tps) else float("nan")

    del model, cache
    gc.collect()

    return {
        "run":                   run_name,
        "type":                  _attn_type(config),
        "params_m":              params_m,
        "moe":                   getattr(config, "use_moe", False),
        "num_blocks":            config.num_blocks,
        "dim":                   config.dim,
        "n_heads":               config.n_heads,
        "kv_heads":              getattr(config, "kv_heads", config.n_heads),
        "max_context":           config.max_context,
        "down_dim_kv":           getattr(config, "down_dim_kv", None),
        "theoretical_cache_mb":  round(_theoretical_kv_cache_mb(config), 2),
        "measured_cache_mb":     vram_cache_mb,
        "prefill_ms":            prefill_ms,
        "val_loss":              _val_loss(run_path),
        "decode_gflops":         decode_gflops,
        "prefill_gflops":        prefill_gflops,
        "decode_arith_int":      _safe_r(decode_flops, decode_bytes),
        "prefill_arith_int":     _safe_r(prefill_flops, prefill_bytes),
        "decode_tflops_s":       decode_tflops_s,
        **{f"tps_{s}": tps_by_seq[s] for s in SEQ_LENS},
    }


# ─── Plots ───────────────────────────────────────────────────────────────────

def _grouped_bar(ax: plt.Axes, df: pd.DataFrame, col: str,
                 ylabel: str, title: str, log: bool = False) -> None:
    families = sorted(df["family"].unique())
    types    = [t for t in TYPE_ORDER if t in df["type"].unique()]
    n_f, n_t = len(families), len(types)
    width    = 0.8 / n_t
    x        = np.arange(n_f)
    for ti, t in enumerate(types):
        sub    = df[df["type"] == t].groupby("family")[col].agg(["mean", "std"])
        means  = [sub.loc[f, "mean"] if f in sub.index else float("nan") for f in families]
        stds   = [sub.loc[f, "std"]  if f in sub.index else 0.0 for f in families]
        stds   = [s if not np.isnan(s) else 0.0 for s in stds]
        offset = (ti - n_t / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width, label=t,
                        color=TYPE_COLORS[t], edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if log:
        ax.set_yscale("log")


def _line_scaling(ax: plt.Axes, df: pd.DataFrame) -> None:
    families   = sorted(df["family"].unique())
    types      = [t for t in TYPE_ORDER if t in df["type"].unique()]
    linestyles = ["-", "--", ":", "-."]
    for fi, fam in enumerate(families):
        sub_fam = df[df["family"] == fam]
        for t in types:
            sub = sub_fam[sub_fam["type"] == t]
            if sub.empty:
                continue
            ys = [sub[f"tps_{s}"].dropna().mean() for s in SEQ_LENS]
            ax.plot(SEQ_LENS, ys, marker="o", linestyle=linestyles[fi % 4],
                    label=f"{fam} / {t}", color=TYPE_COLORS.get(t, "grey"), linewidth=1.8)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput vs sequence length (per family)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def _scatter_params_tps(ax: plt.Axes, df: pd.DataFrame) -> None:
    families = sorted(df["family"].unique())
    markers  = ["o", "s", "^", "D", "v", "P", "X"]
    for fi, fam in enumerate(families):
        sub = df[df["family"] == fam]
        for t in TYPE_ORDER:
            pts = sub[sub["type"] == t]
            if pts.empty:
                continue
            ax.scatter(pts["params_m"], pts[f"tps_{SEQ_LENS[-1]}"],
                       label=f"{fam}/{t}", color=TYPE_COLORS.get(t, "grey"),
                       marker=markers[fi % len(markers)],
                       edgecolors="black", linewidth=0.5, s=80, zorder=3)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel(f"Tokens/sec  (seq={SEQ_LENS[-1]})")
    ax.set_title("Parameters vs throughput", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def _cache_comparison(ax: plt.Axes, df: pd.DataFrame) -> None:
    families = sorted(df["family"].unique())
    markers  = ["o", "s", "^", "D", "v", "P", "X"]
    for fi, fam in enumerate(families):
        sub = df[df["family"] == fam]
        for t in TYPE_ORDER:
            pts = sub[sub["type"] == t]
            if pts.empty:
                continue
            ax.scatter(pts["theoretical_cache_mb"], pts["measured_cache_mb"],
                       label=f"{fam}/{t}", color=TYPE_COLORS.get(t, "grey"),
                       marker=markers[fi % len(markers)],
                       edgecolors="black", linewidth=0.5, s=80, zorder=3)
    lim = max(df["theoretical_cache_mb"].max(), df["measured_cache_mb"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Theoretical cache (MB)")
    ax.set_ylabel("Measured cache (MB)")
    ax.set_title("KV cache: theoretical vs measured", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def make_plots(df: pd.DataFrame, out_plot: str) -> None:
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    if "family" not in df.columns:
        df = df.copy()
        df["family"] = df.apply(_family_key, axis=1)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("MLA vs GQA vs MHA — fair comparison by architecture family",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.38)

    _grouped_bar(fig.add_subplot(gs[0, 0]), df, f"tps_{SEQ_LENS[-1]}",
                 "Tokens/sec", f"Decode throughput (seq={SEQ_LENS[-1]})")
    _grouped_bar(fig.add_subplot(gs[0, 1]), df, "theoretical_cache_mb",
                 "MB", "KV cache size (theoretical)")
    _grouped_bar(fig.add_subplot(gs[0, 2]), df, "prefill_ms",
                 "ms", "Prefill latency")

    ax_loss = fig.add_subplot(gs[1, 0])
    if df["val_loss"].notna().any():
        _grouped_bar(ax_loss, df, "val_loss", "Val loss (NLL)", "Final validation loss")
    else:
        ax_loss.text(0.5, 0.5, "No val-loss data", ha="center", va="center",
                     transform=ax_loss.transAxes, fontsize=11)
        ax_loss.set_title("Final validation loss", fontweight="bold")

    ax_flops    = fig.add_subplot(gs[1, 1])
    ax_tflops_s = fig.add_subplot(gs[1, 2])
    if df["decode_gflops"].notna().any():
        _grouped_bar(ax_flops,    df, "decode_gflops",   "GFLOPs",  "Decode FLOPs (XLA)")
        _grouped_bar(ax_tflops_s, df, "decode_tflops_s", "TFLOP/s", "Achieved TFLOP/s (decode)")
    else:
        for ax, lbl in ((ax_flops, "Decode FLOPs"), (ax_tflops_s, "Achieved TFLOP/s")):
            ax.text(0.5, 0.5, "FLOPs unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(lbl, fontweight="bold")

    _line_scaling(fig.add_subplot(gs[2, :]),   df)
    _scatter_params_tps(fig.add_subplot(gs[3, 0:2]), df)
    _cache_comparison(fig.add_subplot(gs[3, 2]), df)

    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {Path(out_plot).name}")


# ─── Console summary ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    agg = (["theoretical_cache_mb", "measured_cache_mb", "prefill_ms", "val_loss",
             "decode_gflops", "decode_tflops_s"] + [f"tps_{s}" for s in SEQ_LENS])
    agg = [c for c in agg if c in df.columns]
    print("\n── Per-family comparison (apples-to-apples) ────────────────────")
    for fam, grp in df.groupby("family"):
        print(f"\n  [{fam}]  runs: {len(grp)}")
        print(grp.groupby("type")[agg].mean().to_string())
    if df["moe"].any():
        print("\n── Dense vs MoE — throughput ───────────────────────────────────")
        print(df.groupby(["moe", "type"])[[f"tps_{s}" for s in SEQ_LENS]].mean().to_string())


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark real DantinoX model runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir",  default="runs",
                        help="Directory containing run sub-directories (default: runs)")
    parser.add_argument("--out-csv",   default="benchmark_results.csv",
                        help="Output CSV (default: benchmark_results.csv)")
    parser.add_argument("--out-plot",  default="plots/trained_analysis.png",
                        help="Output plot PNG (default: plots/trained_analysis.png)")
    parser.add_argument("--device",    default=None,
                        help="Override CUDA_VISIBLE_DEVICES (default: env or 0)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip generating the summary plot")
    args = parser.parse_args(argv)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    runs_dir = args.runs_dir
    out_csv  = args.out_csv

    if not os.path.isdir(runs_dir):
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Load already-benchmarked runs for incremental operation
    if os.path.exists(out_csv):
        existing_df   = pd.read_csv(out_csv)
        already_done  = set(existing_df["run"].astype(str))
        existing_rows = existing_df.to_dict("records")
        print(f"Loaded {len(already_done)} existing results from {out_csv}")
    else:
        already_done, existing_rows = set(), []

    all_runs = sorted(
        r for r in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, r))
        and os.path.exists(os.path.join(runs_dir, r, "model_weights.msgpack"))
    )
    pending = [r for r in all_runs if r not in already_done]
    print(f"Found {len(all_runs)} runs — {len(already_done)} done, {len(pending)} pending")

    new_results: list[dict] = []
    for i, run in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {run} ...", end=" ", flush=True)
        try:
            row = run_benchmark(run, runs_dir)
            new_results.append(row)
            print(f"OK  type={row['type']}  tps@{SEQ_LENS[-1]}={row[f'tps_{SEQ_LENS[-1]}']}")
        except Exception as exc:
            print(f"SKIP ({exc})")

    all_results = existing_rows + new_results
    if not all_results:
        print("No valid runs found — nothing to save.")
        return

    df = pd.DataFrame(all_results)
    df["family"] = df.apply(_family_key, axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows → {out_csv}")

    print_summary(df)
    if not args.no_plot:
        make_plots(df, args.out_plot)


if __name__ == "__main__":
    main()
