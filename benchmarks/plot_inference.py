#!/usr/bin/env python3
"""
benchmarks/plot_inference.py

Visualise DantinoX inference benchmark results produced by inference_sweep.py.
Every group expanded by _by_attn shows three series —
MHA (blue), GQA (orange), MLA (purple) — so attention-type differences are
always visible.

Generated plots (saved to --out-dir, default: results/plots/):
  01_attention_type.png    — prefill latency & decode throughput per attention variant
  02_scale.png             — latency / throughput vs parameter count × attn type
  03_batch_size.png        — decode throughput vs batch size × attn type
  04_context_len.png       — prefill latency & KV cache MB vs context length × attn type
  05_dtype.png             — fp32 vs bf16 grouped by attn type
  06_kv_cache.png          — cache on vs off lines per attn type
  07_moe.png               — dense vs MoE variants × attn type
  08_activation.png        — SwiGLU vs GELU × attn type
  09_pos_encoding.png      — positional encoding × attn type
  10_gqa_vs_cache.png      — KV cache MB & decode throughput: attn type × context length
  11_scale_dtype.png       — bf16 speedup ratio across model sizes × attn type
  12_batch_attn.png        — decode throughput: batch size × attention type
  13_sampling.png          — sampling strategy comparison (attn-agnostic)
  14_3d_params_seq_latency.png    — 3D: params × sequence length × prefill latency
  15_3d_batch_seq_kvcache.png     — 3D: batch size × sequence × KV cache (size=throughput)
  16_3d_flops_latency_throughput.png — 3D: estimated FLOPs × latency × throughput
  17_3d_params_batch_throughput.png  — 3D: params × batch size × decode throughput

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
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

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

# Normalised attention-variant colours (used for all grouped plots)
ATTN_VARIANTS = ["MHA", "GQA", "MLA"]
ATTN_PALETTE  = {"MHA": "#4C72B0", "GQA": "#DD8452", "MLA": "#8172B3"}

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})


# ─── Shared helpers ───────────────────────────────────────────────────────────

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
    valid = [v for v in values if not pd.isna(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.20)


def _grouped_bar(ax: plt.Axes, labels: list, vals_by_variant: dict,
                 title: str, ylabel: str) -> None:
    """Grouped bar chart: x = labels, groups = attn variants (MHA / GQA / MLA)."""
    variants = [v for v in ATTN_VARIANTS if v in vals_by_variant]
    n, ng = len(labels), len(variants)
    w = 0.7 / max(ng, 1)
    x = np.arange(n)
    for i, var in enumerate(variants):
        vals = vals_by_variant[var]
        offset = (i - ng / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=var,
                      color=ATTN_PALETTE.get(var, DEFAULT_COLOR),
                      edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ng > 1:
        ax.legend(fontsize=8)
    all_vals = [v for vs in vals_by_variant.values() for v in vs if not pd.isna(v)]
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.25)


def _vals_by_variant(sub: pd.DataFrame, labels: list, col: str) -> dict:
    """For each attn_variant present in sub, return {variant: [value per label]}."""
    result: dict = {}
    for var in ATTN_VARIANTS:
        lookup = (sub[sub["attn_variant"] == var]
                  .groupby("label")[col].mean()
                  .to_dict())
        vals = [float(lookup.get(l, float("nan"))) for l in labels]
        if any(not pd.isna(v) for v in vals):
            result[var] = vals
    return result


def _attn_legend(ax: plt.Axes) -> None:
    ax.legend(handles=[Patch(facecolor=ATTN_PALETTE[v], label=v) for v in ATTN_VARIANTS],
              fontsize=8)


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
    colors = [ATTN_PALETTE.get(v, DEFAULT_COLOR) for v in sub["attn_variant"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _bar(ax1, sub["label"].tolist(), sub["prefill_ms_p50"].tolist(),
         "Prefill latency", "ms (p50)", colors=colors)
    _bar(ax2, sub["label"].tolist(), sub["decode_tok_s"].tolist(),
         "Decode throughput", "tokens / sec", colors=colors)
    _attn_legend(ax1)
    fig.suptitle("Attention type comparison", fontweight="bold")
    _save(fig, out / "01_attention_type.png")


def plot_scale(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "scale"].dropna(subset=["params_m"]).copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, lbl in [
        (axes[0], "prefill_ms_p50",     "Prefill latency (ms)"),
        (axes[1], "decode_step_ms_p50", "Decode step latency (ms)"),
        (axes[2], "decode_tok_s",       "Decode throughput (tok/s)"),
    ]:
        for var in ATTN_VARIANTS:
            s = sub[sub["attn_variant"] == var].dropna(subset=[col]).sort_values("params_m")
            if s.empty:
                continue
            ax.plot(s["params_m"], s[col], "o-", color=ATTN_PALETTE[var], label=var)
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.suptitle("Model scale sweep × attention type", fontweight="bold")
    _save(fig, out / "02_scale.png")


def plot_batch_size(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "batch_size"].dropna(subset=["batch_size"]).copy()
    if sub.empty:
        return
    all_bs = sorted(sub["batch_size"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var].sort_values("batch_size")
        if s.empty:
            continue
        s1 = s.dropna(subset=["decode_tok_s"])
        s2 = s.dropna(subset=["prefill_ms_p50"])
        if not s1.empty:
            ax1.plot(s1["batch_size"], s1["decode_tok_s"], "o-",
                     color=ATTN_PALETTE[var], label=var)
        if not s2.empty:
            ax2.plot(s2["batch_size"], s2["prefill_ms_p50"], "o-",
                     color=ATTN_PALETTE[var], label=var)
    # Linear reference from MHA bs=1
    mha_bs1 = sub[(sub["attn_variant"] == "MHA") & (sub["batch_size"] == 1)]["decode_tok_s"]
    if not mha_bs1.empty:
        bs1_tps = float(mha_bs1.iloc[0])
        ax1.plot(all_bs, [bs1_tps * b for b in all_bs], "--",
                 color="#aaa", alpha=0.7, label="ideal (MHA)")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Decode throughput (tok/s)")
    ax1.set_title("Decode throughput vs batch size")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Prefill latency (ms, p50)")
    ax2.set_title("Prefill latency vs batch size")
    ax2.legend(fontsize=8)
    fig.suptitle("Batch size sweep × attention type", fontweight="bold")
    _save(fig, out / "03_batch_size.png")


def plot_context_len(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "context_len"].dropna(subset=["max_context"]).copy()
    if sub.empty:
        return
    ctxs = sorted(sub["max_context"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title, ylabel in [
        (axes[0], "prefill_ms_p50", "Prefill latency vs context",   "ms (p50)"),
        (axes[1], "kv_cache_mb",    "KV cache memory vs context",   "MB"),
        (axes[2], "decode_tok_s",   "Decode throughput vs context", "tok/s"),
    ]:
        for var in ATTN_VARIANTS:
            s = sub[sub["attn_variant"] == var].dropna(subset=[col]).sort_values("max_context")
            if s.empty:
                continue
            ax.plot(s["max_context"], s[col], "o-", color=ATTN_PALETTE[var], label=var)
        ax.set_xlabel("max_context")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(ctxs)
        ax.legend(fontsize=8)
    fig.suptitle("Context length sweep × attention type", fontweight="bold")
    _save(fig, out / "04_context_len.png")


def plot_dtype(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "dtype"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    ordered = ["medium-fp32", "medium-bf16", "large-fp32", "large-bf16"]
    labels = [l for l in ordered if l in sub["label"].values] or sub["label"].unique().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, unit in [
        (axes[0], "prefill_ms_p50", "Prefill latency",   "ms"),
        (axes[1], "decode_tok_s",   "Decode throughput", "tok/s"),
        (axes[2], "kv_cache_mb",    "KV cache MB",       "MB"),
    ]:
        _grouped_bar(ax, labels, _vals_by_variant(sub, labels, col), title, unit)
    fig.suptitle("dtype: fp32 vs bfloat16 × attention type", fontweight="bold")
    _save(fig, out / "05_dtype.png")


def plot_kv_cache(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "kv_cache"].copy()
    if sub.empty:
        return
    sub = sub.copy()
    sub["_bs"]       = sub["label"].str.extract(r"bs(\d+)").astype(float)
    sub["_cache_on"] = sub["label"].str.contains("cache=on")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, title, ylabel in [
        (axes[0], "decode_tok_s",   "Decode throughput (cache on vs off)", "tok/s"),
        (axes[1], "prefill_ms_p50", "Prefill latency",                     "ms"),
    ]:
        for var in ATTN_VARIANTS:
            for cache_on, ls, suffix in [(True, "-", "on"), (False, "--", "off")]:
                s = (sub[(sub["attn_variant"] == var) & (sub["_cache_on"] == cache_on)]
                     .dropna(subset=[col]).sort_values("_bs"))
                if s.empty:
                    continue
                ax.plot(s["_bs"], s[col], f"o{ls}", color=ATTN_PALETTE[var],
                        label=f"{var} cache={suffix}", alpha=0.9)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("KV cache: on vs off × attention type", fontweight="bold")
    _save(fig, out / "06_kv_cache.png")


def plot_moe(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "moe"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    ordered = ["dense", "4exp-top1", "4exp-top2", "8exp-top2", "8exp-top4", "16exp-top4"]
    labels = [l for l in ordered if l in sub["label"].values]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title, unit in [
        (axes[0], "params_m",       "Parameter count",   "M"),
        (axes[1], "prefill_ms_p50", "Prefill latency",   "ms"),
        (axes[2], "decode_tok_s",   "Decode throughput", "tok/s"),
    ]:
        _grouped_bar(ax, labels, _vals_by_variant(sub, labels, col), title, unit)
    fig.suptitle("Dense vs MoE × attention type", fontweight="bold")
    _save(fig, out / "07_moe.png")


def plot_activation(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "activation"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    labels = [l for l in ["SwiGLU", "GELU"] if l in sub["label"].values]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, unit in [
        (axes[0], "params_m",       "Parameter count",   "M"),
        (axes[1], "prefill_ms_p50", "Prefill latency",   "ms"),
        (axes[2], "decode_tok_s",   "Decode throughput", "tok/s"),
    ]:
        _grouped_bar(ax, labels, _vals_by_variant(sub, labels, col), title, unit)
    fig.suptitle("Activation: SwiGLU vs GELU × attention type", fontweight="bold")
    _save(fig, out / "08_activation.png")


def plot_pos_encoding(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "pos_encoding"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    ordered = ["RoPE", "sinusoidal", "trainable", "sliding_win"]
    labels = [l for l in ordered if l in sub["label"].values]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title, unit in [
        (ax1, "prefill_ms_p50", "Prefill latency",   "ms"),
        (ax2, "decode_tok_s",   "Decode throughput", "tok/s"),
    ]:
        _grouped_bar(ax, labels, _vals_by_variant(sub, labels, col), title, unit)
    fig.suptitle("Positional encoding × attention type", fontweight="bold")
    _save(fig, out / "09_pos_encoding.png")


def plot_gqa_vs_cache(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "gqa_vs_cache"].dropna(subset=["max_context", "kv_cache_mb"]).copy()
    if sub.empty:
        return
    markers = {"MHA": "o", "GQA(4/8)": "D", "GQA(2/8)": "s", "GQA(1/8)": "P", "MLA": "^"}
    colors  = {k: PALETTE.get(k, DEFAULT_COLOR) for k in markers}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for attn in sub["attn_type"].unique():
        s = sub[sub["attn_type"] == attn].sort_values("max_context")
        axes[0].plot(s["max_context"], s["kv_cache_mb"],  marker=markers.get(attn, "o"),
                     color=colors.get(attn, DEFAULT_COLOR), label=attn)
        axes[1].plot(s["max_context"], s["decode_tok_s"], marker=markers.get(attn, "o"),
                     color=colors.get(attn, DEFAULT_COLOR), label=attn)
    for ax, ylabel, title in [
        (axes[0], "KV cache (MB)", "KV cache memory vs context length"),
        (axes[1], "tok/s",         "Decode throughput vs context length"),
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
    sub["size_tag"] = sub["label"].str.replace(r"-(fp32|bf16)$", "", regex=True)
    sizes = ["tiny", "small", "medium", "large"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for var in ATTN_VARIANTS:
        v = sub[sub["attn_variant"] == var]
        fp32 = v[v["label"].str.endswith("fp32")].groupby("size_tag").first()
        bf16 = v[v["label"].str.endswith("bf16")].groupby("size_tag").first()
        ratios_p, ratios_d, valid = [], [], []
        for s in sizes:
            if s in fp32.index and s in bf16.index:
                r_p = fp32.loc[s, "prefill_ms_p50"] / max(bf16.loc[s, "prefill_ms_p50"], 1e-6)
                r_d = bf16.loc[s, "decode_tok_s"]   / max(fp32.loc[s, "decode_tok_s"],   1e-6)
                ratios_p.append(r_p)
                ratios_d.append(r_d)
                valid.append(s)
        if not valid:
            continue
        x = np.arange(len(valid))
        ax1.plot(x, ratios_p, "o-", color=ATTN_PALETTE[var], label=var)
        ax2.plot(x, ratios_d, "o-", color=ATTN_PALETTE[var], label=var)
        ax1.set_xticks(x); ax1.set_xticklabels(valid)
        ax2.set_xticks(x); ax2.set_xticklabels(valid)
    for ax, title, ylabel in [
        (ax1, "Prefill speedup from bf16",          "fp32 / bf16 latency  (>1 = bf16 faster)"),
        (ax2, "Decode throughput speedup from bf16", "bf16 / fp32 tok/s  (>1 = bf16 faster)"),
    ]:
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("bfloat16 speedup by model scale × attention type", fontweight="bold")
    _save(fig, out / "11_scale_dtype.png")


def plot_batch_attn(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "batch_attn"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    known = {"MHA", "GQA(2/8)", "GQA(4/8)", "MLA"}
    attn_types  = [a for a in sub["attn_type"].unique() if a in known]
    batch_sizes = sorted(sub["batch_size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title, ylabel in [
        (axes[0], "decode_tok_s", "Decode throughput (tok/s)", "tok/s"),
        (axes[1], "kv_cache_mb",  "KV cache memory (MB)",      "MB"),
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


def plot_sampling(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "sampling"].dropna(subset=["decode_tok_s"]).copy()
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _bar(ax1, sub["label"].tolist(), sub["prefill_ms_p50"].tolist(),
         "Prefill latency", "ms")
    _bar(ax2, sub["label"].tolist(), sub["decode_tok_s"].tolist(),
         "Decode throughput", "tok/s")
    fig.suptitle("Sampling strategy comparison", fontweight="bold")
    _save(fig, out / "13_sampling.png")


# ─── 2-D scatter helpers ─────────────────────────────────────────────────────

def _norm_size(s: pd.Series, lo: float = 20, hi: float = 160) -> np.ndarray:
    """Normalise a series to [lo, hi] for use as scatter point sizes."""
    mn, mx = s.min(), s.max()
    if mx <= mn:
        return np.full(len(s), (lo + hi) / 2)
    return (lo + (s - mn) / (mx - mn) * (hi - lo)).values


def _sc2(ax: plt.Axes, x, y, label: str, color: str,
         marker: str = "o", sizes=50, alpha: float = 0.72) -> None:
    ax.scatter(x, y, c=color, marker=marker, label=label,
               s=sizes, alpha=alpha, edgecolors="white", linewidths=0.3)


def _finish2(ax: plt.Axes, xlabel: str, ylabel: str, title: str,
             log_x: bool = False, log_y: bool = False) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=7, markerscale=0.9)


# ─── 3-D helpers ─────────────────────────────────────────────────────────────

def _estimate_flops_g(df: pd.DataFrame) -> pd.Series:
    """Rough GFLOPs estimate for a prefill forward pass.

    = linear-layer FLOPs (2 * params * B * T)
    + quadratic attention FLOPs (4 * B * n_heads * T² * head_size * n_blocks)
    """
    B   = df["batch_size"]
    T   = df["prompt_len"]
    nh  = df["n_heads"].clip(lower=1)
    nb  = df["num_blocks"]
    hs  = (df["dim"] / nh).fillna(32)
    linear   = 2.0 * df["params_m"] * 1e6 * B * T
    attn_q   = 4.0 * B * nh * T * T * hs * nb
    return (linear + attn_q) / 1e9


def _sc3(ax: plt.Axes, x, y, z, label: str, color: str,
         marker: str = "o", sizes=40, alpha: float = 0.75) -> None:
    ax.scatter(x, y, z, c=color, marker=marker, label=label,
               s=sizes, alpha=alpha, edgecolors="white", linewidths=0.3)


def _ax3(fig: plt.Figure, pos: int, xlabel: str, ylabel: str, zlabel: str,
         title: str, elev: int = 22, azim: int = 45) -> plt.Axes:
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_xlabel(xlabel, labelpad=6, fontsize=8)
    ax.set_ylabel(ylabel, labelpad=6, fontsize=8)
    ax.set_zlabel(zlabel, labelpad=6, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return ax


# ─── 3-D plot functions ───────────────────────────────────────────────────────

def plot_3d_params_seq_latency(df: pd.DataFrame, out: Path) -> None:
    """Params × sequence length × prefill latency (size ∝ estimated FLOPs)."""
    sub = (df[df["group"].isin(["scale", "context_len"])]
           .dropna(subset=["params_m", "prompt_len", "prefill_ms_p50"])
           .copy())
    if sub.empty:
        return
    sub["flops_g"] = _estimate_flops_g(sub)
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig = plt.figure(figsize=(14, 5))
    # Left: params × seq → prefill latency
    ax1 = _ax3(fig, 121, "Params (M)", "Seq len (tokens)", "Prefill (ms)",
               "Params × Sequence → Prefill latency", elev=25, azim=50)
    # Right: FLOPs × seq → prefill latency  (shows quadratic regime)
    ax2 = _ax3(fig, 122, "FLOPs (G)", "Seq len (tokens)", "Prefill (ms)",
               "FLOPs × Sequence → Prefill latency", elev=25, azim=210)
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        sz = np.clip(s["flops_g"] / s["flops_g"].max() * 120 + 20, 20, 140)
        _sc3(ax1, s["params_m"],  s["prompt_len"], s["prefill_ms_p50"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz)
        _sc3(ax2, s["flops_g"],   s["prompt_len"], s["prefill_ms_p50"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle("Scale & context: params / FLOPs × sequence × prefill latency",
                 fontweight="bold")
    _save(fig, out / "14_3d_params_seq_latency.png")


def plot_3d_batch_seq_kvcache(df: pd.DataFrame, out: Path) -> None:
    """Batch size × sequence length × KV cache (size ∝ decode throughput)."""
    sub = (df[df["group"].isin(["batch_size", "context_len", "gqa_vs_cache", "batch_attn"])]
           .dropna(subset=["batch_size", "max_context", "kv_cache_mb", "decode_tok_s"])
           .copy())
    if sub.empty:
        return
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig = plt.figure(figsize=(14, 5))
    ax1 = _ax3(fig, 121, "Batch size", "max_context (tokens)", "KV cache (MB)",
               "Batch × Context → KV cache", elev=20, azim=55)
    ax2 = _ax3(fig, 122, "Batch size", "max_context (tokens)", "Decode tok/s",
               "Batch × Context → Throughput", elev=20, azim=230)
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        # size encodes decode throughput (left) or KV cache (right)
        sz_thr = np.clip(s["decode_tok_s"] / sub["decode_tok_s"].max() * 130 + 15, 15, 145)
        sz_kv  = np.clip(s["kv_cache_mb"]  / sub["kv_cache_mb"].max()  * 130 + 15, 15, 145)
        _sc3(ax1, s["batch_size"], s["max_context"], s["kv_cache_mb"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz_thr)
        _sc3(ax2, s["batch_size"], s["max_context"], s["decode_tok_s"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz_kv)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle("Batch size × sequence length × KV cache / throughput",
                 fontweight="bold")
    _save(fig, out / "15_3d_batch_seq_kvcache.png")


def plot_3d_flops_latency_throughput(df: pd.DataFrame, out: Path) -> None:
    """FLOPs × prefill latency × decode throughput — efficiency frontier."""
    sub = (df.dropna(subset=["params_m", "prompt_len", "prefill_ms_p50", "decode_tok_s"])
           .copy())
    sub = sub[~sub["oom"].fillna(False).astype(bool)]
    sub["flops_g"] = _estimate_flops_g(sub)
    sub = sub[sub["flops_g"] > 0]
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig = plt.figure(figsize=(14, 5))
    # Left: linear FLOPs axis; size = params_m
    ax1 = _ax3(fig, 121, "FLOPs (G)", "Prefill (ms)", "Decode tok/s",
               "FLOPs × Latency × Throughput", elev=18, azim=40)
    # Right: log FLOPs and log throughput to reveal structure across scales
    ax2 = _ax3(fig, 122, "log₁₀(FLOPs G)", "Prefill (ms)", "log₁₀(tok/s)",
               "log-scale view", elev=18, azim=220)
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        sz = np.clip(s["params_m"] / sub["params_m"].max() * 120 + 15, 15, 135)
        _sc3(ax1, s["flops_g"],              s["prefill_ms_p50"], s["decode_tok_s"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz)
        _sc3(ax2, np.log10(s["flops_g"] + 1), s["prefill_ms_p50"],
             np.log10(s["decode_tok_s"] + 1),
             var, ATTN_PALETTE[var], markers[var], sizes=sz)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle("Estimated FLOPs × prefill latency × decode throughput  "
                 "(point size ∝ params)", fontweight="bold")
    _save(fig, out / "16_3d_flops_latency_throughput.png")


def plot_3d_params_batch_throughput(df: pd.DataFrame, out: Path) -> None:
    """Params × batch size × decode throughput / KV cache — scaling surface."""
    sub = (df[df["group"].isin(["scale", "batch_size", "batch_attn"])]
           .dropna(subset=["params_m", "batch_size", "decode_tok_s", "kv_cache_mb"])
           .copy())
    if sub.empty:
        return
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig = plt.figure(figsize=(14, 5))
    ax1 = _ax3(fig, 121, "Params (M)", "Batch size", "Decode tok/s",
               "Params × Batch → Throughput", elev=22, azim=50)
    ax2 = _ax3(fig, 122, "Params (M)", "Batch size", "KV cache (MB)",
               "Params × Batch → KV cache", elev=22, azim=230)
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        sz_kv  = np.clip(s["kv_cache_mb"]  / sub["kv_cache_mb"].max()  * 130 + 15, 15, 145)
        sz_thr = np.clip(s["decode_tok_s"] / sub["decode_tok_s"].max() * 130 + 15, 15, 145)
        _sc3(ax1, s["params_m"], s["batch_size"], s["decode_tok_s"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz_kv)
        _sc3(ax2, s["params_m"], s["batch_size"], s["kv_cache_mb"],
             var, ATTN_PALETTE[var], markers[var], sizes=sz_thr)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle("Params × batch size × throughput / KV cache  "
                 "(left: size ∝ KV cache MB; right: size ∝ tok/s)", fontweight="bold")
    _save(fig, out / "17_3d_params_batch_throughput.png")


# ─── 2-D companion plots (pairwise projections of the 3-D plots) ─────────────

def plot_2d_params_seq_latency(df: pd.DataFrame, out: Path) -> None:
    """Three 2D projections of plot 14 (params × seq × prefill latency)."""
    sub = (df[df["group"].isin(["scale", "context_len"])]
           .dropna(subset=["params_m", "prompt_len", "prefill_ms_p50"])
           .copy())
    if sub.empty:
        return
    sub["flops_g"] = _estimate_flops_g(sub)
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        m, c = markers[var], ATTN_PALETTE[var]
        _sc2(axes[0], s["params_m"],   s["prefill_ms_p50"], var, c, m, _norm_size(s["prompt_len"]))
        _sc2(axes[1], s["prompt_len"], s["prefill_ms_p50"], var, c, m, _norm_size(s["params_m"]))
        _sc2(axes[2], s["flops_g"],    s["prefill_ms_p50"], var, c, m, _norm_size(s["prompt_len"]))

    _finish2(axes[0], "Params (M)",        "Prefill latency (ms)",
             "Params → Prefill latency\n(size ∝ seq len)")
    _finish2(axes[1], "Seq len (tokens)",  "Prefill latency (ms)",
             "Seq len → Prefill latency\n(size ∝ params)")
    _finish2(axes[2], "Estimated FLOPs (G)", "Prefill latency (ms)",
             "FLOPs → Prefill latency\n(size ∝ seq len)")
    fig.suptitle("2D: params / seq / FLOPs × prefill latency  [companion to plot 14]",
                 fontweight="bold")
    _save(fig, out / "18_2d_params_seq_latency.png")


def plot_2d_batch_seq_kvcache(df: pd.DataFrame, out: Path) -> None:
    """Four 2D projections of plot 15 (batch × context × KV cache / throughput)."""
    sub = (df[df["group"].isin(["batch_size", "context_len", "gqa_vs_cache", "batch_attn"])]
           .dropna(subset=["batch_size", "max_context", "kv_cache_mb", "decode_tok_s"])
           .copy())
    if sub.empty:
        return
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axf = axes.flat
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        m, c = markers[var], ATTN_PALETTE[var]
        _sc2(axf[0], s["batch_size"],  s["kv_cache_mb"],  var, c, m, _norm_size(s["max_context"]))
        _sc2(axf[1], s["max_context"], s["kv_cache_mb"],  var, c, m, _norm_size(s["batch_size"]))
        _sc2(axf[2], s["batch_size"],  s["decode_tok_s"], var, c, m, _norm_size(s["max_context"]))
        _sc2(axf[3], s["max_context"], s["decode_tok_s"], var, c, m, _norm_size(s["batch_size"]))

    _finish2(axf[0], "Batch size",       "KV cache (MB)",  "Batch → KV cache\n(size ∝ context)",    log_x=True)
    _finish2(axf[1], "Context (tokens)", "KV cache (MB)",  "Context → KV cache\n(size ∝ batch)",    log_x=True)
    _finish2(axf[2], "Batch size",       "Decode tok/s",   "Batch → Throughput\n(size ∝ context)",  log_x=True, log_y=True)
    _finish2(axf[3], "Context (tokens)", "Decode tok/s",   "Context → Throughput\n(size ∝ batch)",  log_x=True)
    fig.suptitle("2D: batch / context × KV cache / throughput  [companion to plot 15]",
                 fontweight="bold")
    _save(fig, out / "19_2d_batch_seq_kvcache.png")


def plot_2d_flops_latency_throughput(df: pd.DataFrame, out: Path) -> None:
    """Three 2D projections of plot 16 (FLOPs × latency × throughput)."""
    sub = (df.dropna(subset=["params_m", "prompt_len", "prefill_ms_p50", "decode_tok_s"])
           .copy())
    sub = sub[~sub["oom"].fillna(False).astype(bool)]
    sub["flops_g"] = _estimate_flops_g(sub)
    sub = sub[sub["flops_g"] > 0]
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        m, c = markers[var], ATTN_PALETTE[var]
        sz = _norm_size(s["params_m"])
        _sc2(axes[0], s["flops_g"],        s["prefill_ms_p50"], var, c, m, sz)
        _sc2(axes[1], s["flops_g"],        s["decode_tok_s"],   var, c, m, sz)
        _sc2(axes[2], s["prefill_ms_p50"], s["decode_tok_s"],   var, c, m, sz)

    _finish2(axes[0], "Estimated FLOPs (G)", "Prefill latency (ms)",
             "FLOPs → Prefill latency\n(size ∝ params)", log_x=True)
    _finish2(axes[1], "Estimated FLOPs (G)", "Decode tok/s",
             "FLOPs → Throughput\n(size ∝ params)", log_x=True, log_y=True)
    _finish2(axes[2], "Prefill latency (ms)", "Decode tok/s",
             "Latency vs Throughput — efficiency frontier\n(size ∝ params)", log_y=True)
    fig.suptitle("2D: FLOPs × latency × throughput — all experiments  [companion to plot 16]",
                 fontweight="bold")
    _save(fig, out / "20_2d_flops_latency_throughput.png")


def plot_2d_params_batch_throughput(df: pd.DataFrame, out: Path) -> None:
    """Four 2D projections of plot 17 (params × batch × throughput / KV cache)."""
    sub = (df[df["group"].isin(["scale", "batch_size", "batch_attn"])]
           .dropna(subset=["params_m", "batch_size", "decode_tok_s", "kv_cache_mb"])
           .copy())
    if sub.empty:
        return
    markers = {"MHA": "o", "GQA": "s", "MLA": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axf = axes.flat
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var]
        if s.empty:
            continue
        m, c = markers[var], ATTN_PALETTE[var]
        _sc2(axf[0], s["params_m"],   s["decode_tok_s"], var, c, m, _norm_size(s["batch_size"]))
        _sc2(axf[1], s["batch_size"], s["decode_tok_s"], var, c, m, _norm_size(s["params_m"]))
        _sc2(axf[2], s["params_m"],   s["kv_cache_mb"],  var, c, m, _norm_size(s["batch_size"]))
        _sc2(axf[3], s["batch_size"], s["kv_cache_mb"],  var, c, m, _norm_size(s["params_m"]))

    _finish2(axf[0], "Params (M)",  "Decode tok/s",  "Params → Throughput\n(size ∝ batch)",  log_y=True)
    _finish2(axf[1], "Batch size",  "Decode tok/s",  "Batch → Throughput\n(size ∝ params)",  log_x=True, log_y=True)
    _finish2(axf[2], "Params (M)",  "KV cache (MB)", "Params → KV cache\n(size ∝ batch)")
    _finish2(axf[3], "Batch size",  "KV cache (MB)", "Batch → KV cache\n(size ∝ params)",    log_x=True)
    fig.suptitle("2D: params × batch × throughput / KV cache  [companion to plot 17]",
                 fontweight="bold")
    _save(fig, out / "21_2d_params_batch_throughput.png")


# ─── Entry point ─────────────────────────────────────────────────────────────

PLOT_FNS = [
    plot_attention_type, plot_scale, plot_batch_size, plot_context_len,
    plot_dtype, plot_kv_cache, plot_moe, plot_activation, plot_pos_encoding,
    plot_gqa_vs_cache, plot_scale_dtype, plot_batch_attn, plot_sampling,
    # 3D
    plot_3d_params_seq_latency, plot_3d_batch_seq_kvcache,
    plot_3d_flops_latency_throughput, plot_3d_params_batch_throughput,
    # 2D companions
    plot_2d_params_seq_latency, plot_2d_batch_seq_kvcache,
    plot_2d_flops_latency_throughput, plot_2d_params_batch_throughput,
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot DantinoX inference benchmark results.")
    parser.add_argument("--csv", default="results/inference_sweep.csv",
                        help="Input CSV from inference_sweep.py")
    parser.add_argument("--out-dir", default="results/plots/",
                        help="Directory for output PNG files")
    args = parser.parse_args(argv)

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
