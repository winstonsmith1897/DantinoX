#!/usr/bin/env python3
"""
benchmarks/plot_inference.py

Visualise DantinoX inference benchmark results produced by inference_sweep.py.

Generated plots (saved to --out-dir, default: results/plots/):
  01_attention_type.png    — prefill latency & decode throughput per attention variant
  02_scale.png             — latency / throughput vs parameter count
  03_batch_size.png        — decode throughput vs batch size (with linear-scaling reference)
  04_context_len.png       — prefill latency & KV cache MB vs context length
  05_dtype.png             — fp32 vs bf16 speedup by model size
  06_kv_cache.png          — cache on vs off decode throughput comparison
  07_moe.png               — dense vs MoE variants: latency and parameter count
  08_activation.png        — SwiGLU vs GELU latency & throughput
  09_pos_encoding.png      — positional encoding prefill & decode comparison
  10_gqa_vs_cache.png      — KV cache MB & decode throughput: attn type × context length
  11_scale_dtype.png       — bf16 speedup ratio across model sizes
  12_batch_attn.png        — decode throughput heatmap: batch size × attention type

Usage:
    python benchmarks/plot_inference.py --csv results/inference_sweep.csv
    python benchmarks/plot_inference.py --csv results/inference_sweep.csv --out-dir plots/inference/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PALETTE = {
    "MHA":            "#4C72B0",
    "GQA(4/8)":       "#DD8452",
    "GQA(2/8)":       "#55A868",
    "GQA(1/8)":       "#C44E52",
    "MLA":            "#8172B3",
    "fp32":           "#4C72B0",
    "bf16":           "#DD8452",
    "dense":          "#4C72B0",
    "MoE":            "#DD8452",
    "SwiGLU":         "#4C72B0",
    "GELU":           "#DD8452",
    "cache=on":       "#4C72B0",
    "cache=off":      "#DD8452",
}
DEFAULT_COLOR = "#64B5CD"

plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})


def _bar(ax: plt.Axes, labels: list, values: list, title: str, ylabel: str,
         colors: list | None = None) -> None:
    x = np.arange(len(labels))
    c = colors or [DEFAULT_COLOR] * len(labels)
    bars = ax.bar(x, values, color=c, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.20)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path.name}")


# ─── Individual plot functions ────────────────────────────────────────────────

def plot_attention_type(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "attention_type"].copy()
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _bar(ax1, sub["label"].tolist(), sub["prefill_ms_p50"].tolist(),
         "Prefill latency", "ms (p50)")
    _bar(ax2, sub["label"].tolist(), sub["decode_tok_s"].tolist(),
         "Decode throughput", "tokens / sec")
    fig.suptitle("Attention type comparison", fontweight="bold")
    _save(fig, out / "01_attention_type.png")


def plot_scale(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "scale"].dropna(subset=["params_m", "prefill_ms_p50"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("params_m")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in [
        (axes[0], "prefill_ms_p50",     "Prefill latency (ms)"),
        (axes[1], "decode_step_ms_p50", "Decode step latency (ms)"),
        (axes[2], "decode_tok_s",       "Decode throughput (tok/s)"),
    ]:
        ax.plot(sub["params_m"], sub[col], "o-", color=DEFAULT_COLOR)
        for _, row in sub.iterrows():
            ax.annotate(row["label"].split("-")[0], (row["params_m"], row[col]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.suptitle("Model scale sweep", fontweight="bold")
    _save(fig, out / "02_scale.png")


def plot_batch_size(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "batch_size"].dropna(subset=["batch_size", "decode_tok_s"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("batch_size")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(sub["batch_size"], sub["decode_tok_s"], "o-", color=DEFAULT_COLOR, label="actual")
    bs1_tps = sub.loc[sub["batch_size"] == 1, "decode_tok_s"].iloc[0]
    ax1.plot(sub["batch_size"], bs1_tps * sub["batch_size"], "--",
             color="#aaa", alpha=0.7, label="ideal linear")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Decode throughput (tok/s)")
    ax1.set_title("Decode throughput vs batch size")
    ax1.legend()

    ax2.plot(sub["batch_size"], sub["prefill_ms_p50"], "o-", color="#DD8452")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Prefill latency (ms, p50)")
    ax2.set_title("Prefill latency vs batch size")

    fig.suptitle("Batch size sweep", fontweight="bold")
    _save(fig, out / "03_batch_size.png")


def plot_context_len(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "context_len"].dropna(subset=["max_context"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("max_context")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(sub["max_context"], sub["prefill_ms_p50"], "o-", color=DEFAULT_COLOR)
    axes[0].set_title("Prefill latency vs context")
    axes[0].set_xlabel("max_context")
    axes[0].set_ylabel("ms (p50)")

    axes[1].plot(sub["max_context"], sub["kv_cache_mb"], "o-", color="#DD8452")
    axes[1].set_title("KV cache memory vs context")
    axes[1].set_xlabel("max_context")
    axes[1].set_ylabel("MB")

    axes[2].plot(sub["max_context"], sub["decode_tok_s"], "o-", color="#55A868")
    axes[2].set_title("Decode throughput vs context")
    axes[2].set_xlabel("max_context")
    axes[2].set_ylabel("tok/s")

    for ax in axes:
        ax.set_xticks(sub["max_context"].tolist())

    fig.suptitle("Context length sweep", fontweight="bold")
    _save(fig, out / "04_context_len.png")


def plot_dtype(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "dtype"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title, unit in [
        (axes[0], "prefill_ms_p50",  "Prefill latency", "ms"),
        (axes[1], "decode_tok_s",    "Decode throughput", "tok/s"),
        (axes[2], "kv_cache_mb",     "KV cache MB", "MB"),
    ]:
        colors = [PALETTE.get("bf16" if "bf16" in lbl else "fp32", DEFAULT_COLOR)
                  for lbl in sub["label"]]
        _bar(ax, sub["label"].tolist(), sub[col].tolist(), title, unit, colors=colors)
    fig.suptitle("dtype: fp32 vs bfloat16", fontweight="bold")
    _save(fig, out / "05_dtype.png")


def plot_kv_cache(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "kv_cache"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = [PALETTE.get("cache=on" if "on" in lbl else "cache=off", DEFAULT_COLOR)
              for lbl in sub["label"]]
    _bar(ax1, sub["label"].tolist(), sub["decode_tok_s"].tolist(),
         "Decode throughput (cache on vs off)", "tok/s", colors=colors)
    _bar(ax2, sub["label"].tolist(), sub["prefill_ms_p50"].tolist(),
         "Prefill latency", "ms", colors=colors)
    fig.suptitle("KV cache: on vs off", fontweight="bold")
    _save(fig, out / "06_kv_cache.png")


def plot_moe(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "moe"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, unit in [
        (axes[0], "params_m",        "Parameter count", "M"),
        (axes[1], "prefill_ms_p50",  "Prefill latency", "ms"),
        (axes[2], "decode_tok_s",    "Decode throughput", "tok/s"),
    ]:
        _bar(ax, sub["label"].tolist(), sub[col].tolist(), title, unit)
    fig.suptitle("Dense vs MoE — varying experts and top-k", fontweight="bold")
    _save(fig, out / "07_moe.png")


def plot_activation(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "activation"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = [PALETTE.get(lbl, DEFAULT_COLOR) for lbl in sub["label"]]
    for ax, col, title, unit in [
        (axes[0], "params_m",       "Parameter count", "M"),
        (axes[1], "prefill_ms_p50", "Prefill latency", "ms"),
        (axes[2], "decode_tok_s",   "Decode throughput", "tok/s"),
    ]:
        _bar(ax, sub["label"].tolist(), sub[col].tolist(), title, unit, colors=colors)
    fig.suptitle("Activation: SwiGLU vs GELU", fontweight="bold")
    _save(fig, out / "08_activation.png")


def plot_pos_encoding(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "pos_encoding"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _bar(ax1, sub["label"].tolist(), sub["prefill_ms_p50"].tolist(),
         "Prefill latency", "ms")
    _bar(ax2, sub["label"].tolist(), sub["decode_tok_s"].tolist(),
         "Decode throughput", "tok/s")
    fig.suptitle("Positional encoding comparison", fontweight="bold")
    _save(fig, out / "09_pos_encoding.png")


def plot_gqa_vs_cache(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "gqa_vs_cache"].dropna(subset=["max_context", "kv_cache_mb"]).copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    markers = {"MHA": "o", "GQA(2/8)": "s", "MLA": "^"}
    colors  = {"MHA": PALETTE["MHA"], "GQA(2/8)": PALETTE["GQA(2/8)"], "MLA": PALETTE["MLA"]}

    for attn in sub["attn_type"].unique():
        s = sub[sub["attn_type"] == attn].sort_values("max_context")
        m = markers.get(attn, "o")
        c = colors.get(attn, DEFAULT_COLOR)
        axes[0].plot(s["max_context"], s["kv_cache_mb"],   marker=m, color=c, label=attn)
        axes[1].plot(s["max_context"], s["decode_tok_s"], marker=m, color=c, label=attn)

    for ax, ylabel, title in [
        (axes[0], "KV cache (MB)",  "KV cache memory vs context length"),
        (axes[1], "tok/s",          "Decode throughput vs context length"),
    ]:
        ax.set_xlabel("max_context")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(sorted(sub["max_context"].unique()))

    fig.suptitle("GQA compression × context length", fontweight="bold")
    _save(fig, out / "10_gqa_vs_cache.png")


def plot_scale_dtype(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "scale_dtype"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return

    # Compute speedup ratios within each (model_size) pair
    sub["size_tag"] = sub["label"].str.replace(r"-(fp32|bf16)$", "", regex=True)
    sizes = ["tiny", "small", "medium", "large"]
    fp32  = sub[sub["label"].str.endswith("fp32")].set_index("size_tag")
    bf16  = sub[sub["label"].str.endswith("bf16")].set_index("size_tag")

    ratios_prefill = []
    ratios_decode  = []
    valid_sizes    = []
    for s in sizes:
        if s in fp32.index and s in bf16.index:
            r_p = fp32.loc[s, "prefill_ms_p50"] / max(bf16.loc[s, "prefill_ms_p50"], 1e-6)
            r_d = bf16.loc[s, "decode_tok_s"]   / max(fp32.loc[s, "decode_tok_s"], 1e-6)
            ratios_prefill.append(r_p)
            ratios_decode.append(r_d)
            valid_sizes.append(s)

    if not valid_sizes:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(valid_sizes))
    ax1.bar(x, ratios_prefill, color=DEFAULT_COLOR, edgecolor="white")
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_sizes)
    ax1.set_ylabel("fp32 latency / bf16 latency  (>1 = bf16 faster)")
    ax1.set_title("Prefill speedup from bf16")

    ax2.bar(x, ratios_decode, color="#DD8452", edgecolor="white")
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_sizes)
    ax2.set_ylabel("bf16 tok/s / fp32 tok/s  (>1 = bf16 faster)")
    ax2.set_title("Decode throughput speedup from bf16")

    for ax, ratios in [(ax1, ratios_prefill), (ax2, ratios_decode)]:
        for xi, r in zip(x, ratios):
            ax.text(xi, r + 0.02, f"{r:.2f}×", ha="center", fontsize=9)

    fig.suptitle("bfloat16 speedup by model scale", fontweight="bold")
    _save(fig, out / "11_scale_dtype.png")


def plot_batch_attn(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "batch_attn"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return

    attn_types = [a for a in ["MHA", "GQA(2/8)", "MLA"] if a in sub["attn_type"].values]
    batch_sizes = sorted(sub["batch_size"].unique())

    # Build decode throughput and KV cache MB matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, title, ylabel in [
        (axes[0], "decode_tok_s", "Decode throughput (tok/s)",  "tok/s"),
        (axes[1], "kv_cache_mb",  "KV cache memory (MB)",       "MB"),
    ]:
        for attn in attn_types:
            s = sub[sub["attn_type"] == attn].sort_values("batch_size")
            ax.plot(s["batch_size"], s[col], "o-",
                    color=PALETTE.get(attn, DEFAULT_COLOR), label=attn)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(batch_sizes)
        ax.legend()

    fig.suptitle("Batch size × attention type", fontweight="bold")
    _save(fig, out / "12_batch_attn.png")


# ─── Entry point ─────────────────────────────────────────────────────────────

PLOT_FNS = [
    plot_attention_type, plot_scale, plot_batch_size, plot_context_len,
    plot_dtype, plot_kv_cache, plot_moe, plot_activation, plot_pos_encoding,
    plot_gqa_vs_cache, plot_scale_dtype, plot_batch_attn,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DantinoX inference benchmark results.")
    parser.add_argument("--csv", default="results/inference_sweep.csv",
                        help="Input CSV from inference_sweep.py")
    parser.add_argument("--out-dir", default="results/plots/",
                        help="Directory for output PNG files")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in PLOT_FNS:
        try:
            fn(df, out_dir)
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {exc}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
