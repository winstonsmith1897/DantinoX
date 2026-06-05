#!/usr/bin/env python3
"""
benchmarks/plot_diffusion_ar.py
================================

Generate all comparison figures from a ``diffusion_ar_sweep.csv`` produced by
``diffusion_ar_sweep.py``.

Figures generated
-----------------
  01_forward_by_mode_attn.png   – forward-pass latency: AR vs Diff, all attn types
  02_ar_tok_s_by_mode_attn.png  – AR decode throughput by attention type
  03_diff_gen_tok_s.png         – Diffusion generation throughput (no/with dual-cache)
  04_scale_latency.png          – forward-pass latency vs model size
  05_scale_memory.png           – parameters + KV/cache memory vs model size
  06_batch_forward_ms.png       – forward-pass latency vs batch size
  07_batch_tok_s.png            – throughput vs batch size (AR tok/s + Diff gen tok/s)
  08_seqlen_forward_ms.png      – latency vs sequence length
  09_seqlen_memory.png          – KV/cache memory vs sequence length
  10_diff_steps_gentime.png     – estimated generation time vs #denoising steps
  11_dual_cache_speedup.png     – dual-cache speedup vs prefix length
  12_dual_cache_ms.png          – step latency w/ and w/o dual cache vs prefix len
  13_dtype_latency.png          – fp32 vs bf16 latency comparison
  14_dtype_memory.png           – fp32 vs bf16 parameter memory
  15_noise_schedule_latency.png – latency for each noise schedule (should be equal)
  16_moe_latency.png            – dense vs MoE forward-pass latency
  17_moe_params.png             – parameter count dense vs MoE
  18_mla_detail.png             – MLA latent-dim sweep + train vs infer path
  19_peak_mem_heatmap.png       – heatmap: model_type × attn_variant, peak memory
  20_throughput_heatmap.png     – heatmap: model_type × attn_variant, throughput

Usage
-----
  python benchmarks/plot_diffusion_ar.py --csv results/diffusion_ar_sweep.csv
  python benchmarks/plot_diffusion_ar.py --csv results/diffusion_ar_sweep.csv --out plots/
  python benchmarks/plot_diffusion_ar.py --csv results/diffusion_ar_sweep.csv --figs 01 04 11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as exc:
    print(f"Missing dependency: {exc}.\nInstall with: pip install pandas matplotlib")
    sys.exit(1)

# ── Palette ────────────────────────────────────────────────────────────────────

_MODEL_COLOR = {"AR": "#2196F3", "Diff": "#FF9800"}   # blue / orange
_ATTN_COLOR  = {"MHA": "#1565C0", "GQA": "#43A047", "MLA": "#8E24AA"}
_ATTN_MARKER = {"MHA": "o", "GQA": "s", "MLA": "^"}

_STYLE = dict(dpi=150, bbox_inches="tight")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig(title: str, w: float = 8, h: float = 5) -> tuple:
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, **_STYLE)
    plt.close(fig)
    print(f"  → {path.name}")


def _label_bars(ax: Any, fmt: str = "{:.1f}") -> None:
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(
                fmt.format(h),
                xy=(p.get_x() + p.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )


def _combo(df: Any, model_type: str, attn_variant: str) -> Any:
    return df[(df["model_type"] == model_type) & (df["attn_variant"] == attn_variant)]


def _ordered_attn(df: Any) -> list[str]:
    order = ["MHA", "GQA", "MLA"]
    present = df["attn_variant"].unique().tolist()
    return [a for a in order if a in present]


def _ordered_models(df: Any) -> list[str]:
    order = ["AR", "Diff"]
    present = df["model_type"].unique().tolist()
    return [m for m in order if m in present]


def _combo_label(mt: str, av: str) -> str:
    return f"{mt}-{av}"


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_01_forward_by_mode_attn(df: Any, out: Path) -> None:
    grp = df[df["group"] == "model_attn"].copy()
    if grp.empty:
        return
    combos = [
        (mt, av)
        for mt in _ordered_models(grp)
        for av in _ordered_attn(grp)
    ]
    labels = [_combo_label(mt, av) for mt, av in combos]
    values = []
    for mt, av in combos:
        sub = _combo(grp, mt, av)
        values.append(sub["forward_ms_p50"].median() if not sub.empty else float("nan"))

    fig, ax = _fig("Forward-pass latency: AR vs Diffusion × Attention (group=model_attn)")
    colors = [_MODEL_COLOR.get(mt, "#999") for mt, _ in combos]
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    _label_bars(ax, "{:.2f}")
    ax.set_ylabel("Median forward-pass latency (ms)")
    ax.set_xlabel("Model × Attention")
    # legend patch
    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color=c, label=k) for k, c in _MODEL_COLOR.items()]
    ax.legend(handles=legend, title="Model type")
    _save(fig, out / "01_forward_by_mode_attn.png")


def plot_02_ar_tok_s_by_mode_attn(df: Any, out: Path) -> None:
    grp = df[(df["group"] == "model_attn") & (df["model_type"] == "AR")].copy()
    if grp.empty:
        return
    attns  = _ordered_attn(grp)
    values = [grp[grp["attn_variant"] == a]["ar_tok_s"].median() for a in attns]
    fig, ax = _fig("AR decode throughput by attention type (group=model_attn)")
    bars = ax.bar(attns, values,
                  color=[_ATTN_COLOR.get(a, "#777") for a in attns],
                  edgecolor="white")
    _label_bars(ax, "{:.0f}")
    ax.set_ylabel("Tokens / second (decode, BS varies by sub-label)")
    ax.set_xlabel("Attention variant")
    _save(fig, out / "02_ar_tok_s_by_mode_attn.png")


def plot_03_diff_gen_tok_s(df: Any, out: Path) -> None:
    grp = df[(df["group"] == "model_attn") & (df["model_type"] == "Diff")].copy()
    if grp.empty:
        return
    attns = _ordered_attn(grp)
    no_cache   = [grp[grp["attn_variant"] == a]["diff_gen_tok_s"].median()        for a in attns]
    with_cache = [grp[grp["attn_variant"] == a]["diff_gen_tok_s_cached"].median() for a in attns]

    x = np.arange(len(attns))
    w = 0.35
    fig, ax = _fig("Diffusion generation throughput: no-cache vs dual-cache")
    ax.bar(x - w/2, no_cache,   w, label="No dual-cache", color="#FF9800", edgecolor="white")
    ax.bar(x + w/2, with_cache, w, label="Dual-cache",    color="#E65100", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(attns)
    ax.set_ylabel("Generation tok/s  (B × T / (N_steps × step_s))")
    ax.legend()
    _save(fig, out / "03_diff_gen_tok_s.png")


def plot_04_scale_latency(df: Any, out: Path) -> None:
    grp = df[df["group"] == "scale"].copy()
    if grp.empty:
        return
    fig, ax = _fig("Forward-pass latency vs model size", w=9, h=5)
    for mt in _ordered_models(grp):
        for av in _ordered_attn(grp):
            sub = _combo(grp, mt, av).sort_values("params_m")
            if sub.empty:
                continue
            style = "--" if mt == "Diff" else "-"
            ax.plot(sub["params_m"], sub["forward_ms_p50"],
                    marker=_ATTN_MARKER.get(av, "o"),
                    linestyle=style,
                    color=_ATTN_COLOR.get(av, "#777"),
                    alpha=0.8 if mt == "AR" else 0.6,
                    label=f"{mt}-{av}")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    _save(fig, out / "04_scale_latency.png")


def plot_05_scale_memory(df: Any, out: Path) -> None:
    grp = df[df["group"] == "scale"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Memory vs model size", fontweight="bold")
    for ax, col, title in [
        (axes[0], "params_mb",    "Model weights (MB)"),
        (axes[1], "peak_mem_mb",  "Peak device memory (MB)"),
    ]:
        for mt in _ordered_models(grp):
            for av in _ordered_attn(grp):
                sub = _combo(grp, mt, av).sort_values("params_m")
                if sub.empty:
                    continue
                ax.plot(sub["params_m"], sub[col],
                        marker=_ATTN_MARKER.get(av, "o"),
                        linestyle="--" if mt == "Diff" else "-",
                        color=_ATTN_COLOR.get(av, "#777"),
                        label=f"{mt}-{av}")
        ax.set_xlabel("Parameters (M)"); ax.set_ylabel(title)
        ax.set_xscale("log"); ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "05_scale_memory.png")


def plot_06_batch_forward_ms(df: Any, out: Path) -> None:
    grp = df[df["group"] == "batch_size"].copy()
    if grp.empty:
        return
    fig, ax = _fig("Forward-pass latency vs batch size", w=9, h=5)
    for mt in _ordered_models(grp):
        for av in _ordered_attn(grp):
            sub = _combo(grp, mt, av).sort_values("batch_size")
            if sub.empty:
                continue
            ax.plot(sub["batch_size"], sub["forward_ms_p50"],
                    marker=_ATTN_MARKER.get(av, "o"),
                    linestyle="--" if mt == "Diff" else "-",
                    color=_ATTN_COLOR.get(av, "#777"),
                    label=f"{mt}-{av}")
    ax.set_xlabel("Batch size"); ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out / "06_batch_forward_ms.png")


def plot_07_batch_tok_s(df: Any, out: Path) -> None:
    grp = df[df["group"] == "batch_size"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Throughput vs batch size", fontweight="bold")

    for ax, mt, col, ylabel in [
        (axes[0], "AR",   "ar_tok_s",       "AR decode tok/s"),
        (axes[1], "Diff", "diff_gen_tok_s",  "Diff gen tok/s"),
    ]:
        sub_mt = grp[grp["model_type"] == mt]
        for av in _ordered_attn(sub_mt):
            sub = sub_mt[sub_mt["attn_variant"] == av].sort_values("batch_size")
            if sub.empty:
                continue
            ax.plot(sub["batch_size"], sub[col],
                    marker=_ATTN_MARKER.get(av, "o"),
                    color=_ATTN_COLOR.get(av, "#777"),
                    label=av)
        ax.set_xlabel("Batch size"); ax.set_ylabel(ylabel)
        ax.set_title(f"{mt} throughput"); ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "07_batch_tok_s.png")


def plot_08_seqlen_forward_ms(df: Any, out: Path) -> None:
    grp = df[df["group"] == "seq_len"].copy()
    if grp.empty:
        return
    fig, ax = _fig("Forward-pass latency vs sequence length (O(T²) for both)", w=9, h=5)
    for mt in _ordered_models(grp):
        for av in _ordered_attn(grp):
            sub = _combo(grp, mt, av).sort_values("seq_len")
            if sub.empty:
                continue
            ax.plot(sub["seq_len"], sub["forward_ms_p50"],
                    marker=_ATTN_MARKER.get(av, "o"),
                    linestyle="--" if mt == "Diff" else "-",
                    color=_ATTN_COLOR.get(av, "#777"),
                    label=f"{mt}-{av}")
    ax.set_xlabel("Sequence length T"); ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out / "08_seqlen_forward_ms.png")


def plot_09_seqlen_memory(df: Any, out: Path) -> None:
    grp = df[df["group"] == "seq_len"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Memory vs sequence length", fontweight="bold")
    for ax, mt, col, title in [
        (axes[0], "AR",   "ar_kv_cache_mb",  "AR KV-cache (MB)"),
        (axes[1], "Diff", "peak_mem_mb",      "Diff peak device memory (MB)"),
    ]:
        sub_mt = grp[grp["model_type"] == mt]
        for av in _ordered_attn(sub_mt):
            sub = sub_mt[sub_mt["attn_variant"] == av].sort_values("seq_len")
            if sub.empty:
                continue
            ax.plot(sub["seq_len"], sub[col],
                    marker=_ATTN_MARKER.get(av, "o"),
                    color=_ATTN_COLOR.get(av, "#777"),
                    label=av)
        ax.set_xlabel("Sequence length T"); ax.set_ylabel(title)
        ax.set_title(f"{mt} — {title}"); ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "09_seqlen_memory.png")


def plot_10_diff_steps_gentime(df: Any, out: Path) -> None:
    grp = df[df["group"] == "diff_steps"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Diffusion generation throughput vs #denoising steps", fontweight="bold")
    for ax, col, ylabel in [
        (axes[0], "diff_gen_tok_s",        "tok/s (no dual-cache)"),
        (axes[1], "diff_step_ms_p50",      "Step latency (ms) — constant"),
    ]:
        for av in _ordered_attn(grp):
            sub = grp[grp["attn_variant"] == av].sort_values("diff_n_steps")
            if sub.empty:
                continue
            ax.plot(sub["diff_n_steps"], sub[col],
                    marker=_ATTN_MARKER.get(av, "o"),
                    color=_ATTN_COLOR.get(av, "#777"),
                    label=av)
        ax.set_xlabel("#Denoising steps"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "10_diff_steps_gentime.png")


def plot_11_dual_cache_speedup(df: Any, out: Path) -> None:
    grp = df[df["group"] == "dual_cache"].copy()
    if grp.empty:
        return
    fig, ax = _fig("Dual-cache speedup vs prefix length  (step_ms / step_cached_ms)")
    for av in _ordered_attn(grp):
        sub = grp[grp["attn_variant"] == av].sort_values("diff_prefix_len")
        if sub.empty:
            continue
        ax.plot(sub["diff_prefix_len"], sub["diff_dual_cache_speedup"],
                marker=_ATTN_MARKER.get(av, "o"),
                color=_ATTN_COLOR.get(av, "#777"),
                label=av)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="break-even")
    ax.set_xlabel("Prefix length (tokens)"); ax.set_ylabel("Speedup (×)")
    ax.legend()
    _save(fig, out / "11_dual_cache_speedup.png")


def plot_12_dual_cache_ms(df: Any, out: Path) -> None:
    grp = df[df["group"] == "dual_cache"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Diffusion step latency: no-cache vs dual-cache vs build cost", fontweight="bold")
    for ax, av in zip(axes, _ordered_attn(grp)[:2]):
        sub = grp[grp["attn_variant"] == av].sort_values("diff_prefix_len")
        if sub.empty:
            continue
        ax.plot(sub["diff_prefix_len"], sub["diff_step_ms_p50"],
                marker="o", color="#FF9800", label="step (no cache)")
        ax.plot(sub["diff_prefix_len"], sub["diff_step_cached_ms_p50"],
                marker="s", color="#E65100", label="step (dual-cache)")
        ax.plot(sub["diff_prefix_len"], sub["diff_cache_build_ms"],
                marker="^", color="#795548", linestyle=":", label="cache build (one-time)")
        ax.set_xlabel("Prefix length"); ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{av}"); ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out / "12_dual_cache_ms.png")


def plot_13_dtype_latency(df: Any, out: Path) -> None:
    grp = df[df["group"] == "dtype"].copy()
    if grp.empty:
        return
    labels_all = sorted(grp["label"].unique())
    combos = [(mt, av) for mt in _ordered_models(grp) for av in _ordered_attn(grp)]
    x = np.arange(len(combos))
    width = 0.35

    fp32_vals = []
    bf16_vals = []
    combo_labels = []
    for mt, av in combos:
        sub = _combo(grp, mt, av)
        fp32 = sub[sub["dtype"] == "fp32"]["forward_ms_p50"].median()
        bf16 = sub[sub["dtype"] == "bf16"]["forward_ms_p50"].median()
        fp32_vals.append(fp32); bf16_vals.append(bf16)
        combo_labels.append(f"{mt}-{av}")

    fig, ax = _fig("fp32 vs bf16 forward-pass latency", w=10)
    ax.bar(x - width/2, fp32_vals, width, label="fp32", color="#546E7A", edgecolor="white")
    ax.bar(x + width/2, bf16_vals, width, label="bf16", color="#00BCD4", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(combo_labels, rotation=20, ha="right")
    ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend()
    _save(fig, out / "13_dtype_latency.png")


def plot_14_dtype_memory(df: Any, out: Path) -> None:
    grp = df[df["group"] == "dtype"].copy()
    if grp.empty:
        return
    combos = [(mt, av) for mt in _ordered_models(grp) for av in _ordered_attn(grp)]
    x = np.arange(len(combos))
    width = 0.35
    fp32_vals = []; bf16_vals = []; labels = []
    for mt, av in combos:
        sub = _combo(grp, mt, av)
        fp32_vals.append(sub[sub["dtype"] == "fp32"]["params_mb"].median())
        bf16_vals.append(sub[sub["dtype"] == "bf16"]["params_mb"].median())
        labels.append(f"{mt}-{av}")

    fig, ax = _fig("fp32 vs bf16 model weight memory (MB)", w=10)
    ax.bar(x - width/2, fp32_vals, width, label="fp32", color="#546E7A", edgecolor="white")
    ax.bar(x + width/2, bf16_vals, width, label="bf16", color="#00BCD4", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Parameter memory (MB)")
    ax.legend()
    _save(fig, out / "14_dtype_memory.png")


def plot_15_noise_schedule_latency(df: Any, out: Path) -> None:
    grp = df[df["group"] == "noise_schedule"].copy()
    if grp.empty:
        return
    schedules = sorted(grp["label"].unique())
    attns     = _ordered_attn(grp)
    x  = np.arange(len(schedules))
    bw = 0.25

    fig, ax = _fig("Forward-pass latency by noise schedule (should be equal — schedule affects sampling only)")
    for i, av in enumerate(attns):
        vals = [grp[(grp["label"] == s) & (grp["attn_variant"] == av)]["forward_ms_p50"].median()
                for s in schedules]
        ax.bar(x + i * bw, vals, bw, label=av,
               color=_ATTN_COLOR.get(av, "#777"), edgecolor="white")
    ax.set_xticks(x + bw); ax.set_xticklabels(schedules)
    ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend(title="Attention")
    _save(fig, out / "15_noise_schedule_latency.png")


def plot_16_moe_latency(df: Any, out: Path) -> None:
    grp = df[df["group"] == "moe"].copy()
    if grp.empty:
        return
    labels  = sorted(grp["label"].unique(), key=lambda s: (s != "dense", s))
    combos  = [(mt, av) for mt in _ordered_models(grp) for av in _ordered_attn(grp)]
    n_bars  = len(labels)
    x       = np.arange(len(combos))
    bw      = 0.8 / n_bars

    fig, ax = _fig("MoE vs dense forward-pass latency", w=11)
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(labels):
        vals = []
        for mt, av in combos:
            sub = _combo(grp[grp["label"] == lbl], mt, av)
            vals.append(sub["forward_ms_p50"].median() if not sub.empty else float("nan"))
        ax.bar(x + i * bw, vals, bw, label=lbl, color=cmap(i), edgecolor="white")
    ax.set_xticks(x + bw * (n_bars / 2 - 0.5))
    ax.set_xticklabels([f"{mt}-{av}" for mt, av in combos], rotation=20, ha="right")
    ax.set_ylabel("Median forward-pass latency (ms)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out / "16_moe_latency.png")


def plot_17_moe_params(df: Any, out: Path) -> None:
    grp = df[df["group"] == "moe"].copy()
    if grp.empty:
        return
    labels = sorted(grp["label"].unique(), key=lambda s: (s != "dense", s))
    attns  = _ordered_attn(grp)
    x  = np.arange(len(labels))
    bw = 0.8 / len(attns)

    fig, ax = _fig("Parameter count: dense vs MoE variants")
    for i, av in enumerate(attns):
        vals = [grp[(grp["label"] == lbl) & (grp["attn_variant"] == av)]["params_m"].median()
                for lbl in labels]
        ax.bar(x + i * bw, vals, bw, label=av,
               color=_ATTN_COLOR.get(av, "#777"), edgecolor="white")
    ax.set_xticks(x + bw); ax.set_xticklabels(labels)
    ax.set_ylabel("Parameters (M)")
    ax.legend(title="Attention")
    _save(fig, out / "17_moe_params.png")


def plot_18_mla_detail(df: Any, out: Path) -> None:
    grp = df[df["group"] == "mla_detail"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MLA detail: train vs infer path, latent-dim sweep", fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], "forward_ms_p50", "Forward-pass latency (ms)"),
        (axes[1], "params_m",       "Parameters (M)"),
    ]:
        labels = grp["label"].tolist()
        vals   = grp[metric].tolist()
        colors = ["#1565C0" if "AR" in l else "#FF9800" for l in labels]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color="#1565C0", label="AR"),
              mpatches.Patch(color="#FF9800", label="Diff")]
    axes[0].legend(handles=legend)
    plt.tight_layout()
    _save(fig, out / "18_mla_detail.png")


def plot_19_peak_mem_heatmap(df: Any, out: Path) -> None:
    grp = df[df["group"] == "model_attn"].copy()
    if grp.empty:
        return
    models = _ordered_models(grp)
    attns  = _ordered_attn(grp)
    data   = np.full((len(models), len(attns)), float("nan"))
    for i, mt in enumerate(models):
        for j, av in enumerate(attns):
            sub = _combo(grp, mt, av)
            if not sub.empty:
                data[i, j] = sub["peak_mem_mb"].median()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Peak device memory (MB) — model_type × attention (group=model_attn)",
                 fontweight="bold")
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(attns))); ax.set_xticklabels(attns)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(attns)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="MB")
    plt.tight_layout()
    _save(fig, out / "19_peak_mem_heatmap.png")


def plot_20_throughput_heatmap(df: Any, out: Path) -> None:
    grp = df[df["group"] == "batch_size"].copy()
    if grp.empty:
        return
    batch_sizes = sorted(grp["batch_size"].unique())
    combos      = [(mt, av) for mt in _ordered_models(grp) for av in _ordered_attn(grp)]
    combo_labels = [f"{mt}-{av}" for mt, av in combos]

    data = np.full((len(combos), len(batch_sizes)), float("nan"))
    for i, (mt, av) in enumerate(combos):
        for j, bs in enumerate(batch_sizes):
            sub  = _combo(grp, mt, av)
            sub  = sub[sub["batch_size"] == bs]
            col  = "ar_tok_s" if mt == "AR" else "diff_gen_tok_s"
            if not sub.empty:
                v = sub[col].median()
                data[i, j] = v if np.isfinite(v) else float("nan")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Throughput (tok/s) — (model × attn) × batch size", fontweight="bold")
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(batch_sizes))); ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(combos))); ax.set_yticklabels(combo_labels)
    ax.set_xlabel("Batch size"); ax.set_ylabel("Model × Attention")
    for i in range(len(combos)):
        for j in range(len(batch_sizes)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="tok/s")
    plt.tight_layout()
    _save(fig, out / "20_throughput_heatmap.png")


# ── Plot registry ──────────────────────────────────────────────────────────────

PLOTS = {
    "01": plot_01_forward_by_mode_attn,
    "02": plot_02_ar_tok_s_by_mode_attn,
    "03": plot_03_diff_gen_tok_s,
    "04": plot_04_scale_latency,
    "05": plot_05_scale_memory,
    "06": plot_06_batch_forward_ms,
    "07": plot_07_batch_tok_s,
    "08": plot_08_seqlen_forward_ms,
    "09": plot_09_seqlen_memory,
    "10": plot_10_diff_steps_gentime,
    "11": plot_11_dual_cache_speedup,
    "12": plot_12_dual_cache_ms,
    "13": plot_13_dtype_latency,
    "14": plot_14_dtype_memory,
    "15": plot_15_noise_schedule_latency,
    "16": plot_16_moe_latency,
    "17": plot_17_moe_params,
    "18": plot_18_mla_detail,
    "19": plot_19_peak_mem_heatmap,
    "20": plot_20_throughput_heatmap,
    "21": plot_21_block_wise_speedup,
    "22": plot_22_block_size_sweep,
    "23": plot_23_context_ratio,
}


def plot_21_block_wise_speedup(df: Any, out: Path) -> None:
    """Fig 21 — decode_block vs full_step latency by batch size (group=block_wise)."""
    grp = df[df["group"] == "block_wise"].copy()
    if grp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fast-dLLM: decode_block vs full-sequence step (group=block_wise)",
                 fontweight="bold")

    for ax, col, ylabel in [
        (axes[0], ["decode_block_ms_p50", "full_step_ms_p50"], "Latency (ms)"),
        (axes[1], ["block_speedup"], "Speedup (full_step / decode_block)"),
    ]:
        for av in _ordered_attn(grp):
            sub = grp[grp["attn_variant"] == av].sort_values("batch_size")
            if sub.empty:
                continue
            if len(col) > 1:
                ax.plot(sub["batch_size"], sub[col[0]],
                        marker="o", color=_ATTN_COLOR.get(av, "#777"),
                        linestyle="-", label=f"{av} decode_block")
                ax.plot(sub["batch_size"], sub[col[1]],
                        marker="s", color=_ATTN_COLOR.get(av, "#777"),
                        linestyle="--", label=f"{av} full_step")
            else:
                ax.plot(sub["batch_size"], sub[col[0]],
                        marker="o", color=_ATTN_COLOR.get(av, "#777"), label=av)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if len(col) == 1:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="baseline")

    plt.tight_layout()
    _save(fig, out / "21_block_wise_speedup.png")


def plot_22_block_size_sweep(df: Any, out: Path) -> None:
    """Fig 22 — block_speedup and refresh_overhead vs block size (group=block_size_sweep).

    Reproduces Fig. 4 of Fast-dLLM: larger blocks → more speedup but also higher
    approximation error (longer time between cache refreshes).
    """
    grp = df[df["group"] == "block_size_sweep"].copy()
    if grp.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fast-dLLM block-size sweep (replicates Fig. 4)", fontweight="bold")

    for ax, col, ylabel, hline in [
        (axes[0], "decode_block_ms_p50", "decode_block latency (ms)", None),
        (axes[1], "block_speedup",       "Speedup over full-sequence step",  1.0),
        (axes[2], "refresh_overhead_x",  "Cache refresh cost (× decode_block)", None),
    ]:
        for av in _ordered_attn(grp):
            sub = grp[grp["attn_variant"] == av].sort_values("block_size")
            if sub.empty:
                continue
            ax.plot(sub["block_size"], sub[col],
                    marker="o", color=_ATTN_COLOR.get(av, "#777"), label=av)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Block size (tokens)"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, out / "22_block_size_sweep.png")


def plot_23_context_ratio(df: Any, out: Path) -> None:
    """Fig 23 — block_speedup vs suffix length (group=context_ratio).

    Larger suffix → more tokens cached → larger speedup from DualCache.
    The suffix is all-MASK blocks not yet decoded.
    """
    grp = df[df["group"] == "context_ratio"].copy()
    if grp.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fast-dLLM DualCache: speedup vs suffix (cached MASK) length",
                 fontweight="bold")

    for ax, col, ylabel in [
        (axes[0], "block_speedup",      "Speedup (full_step / decode_block)"),
        (axes[1], "refresh_overhead_x", "Refresh cost (× decode_block)"),
    ]:
        for av in _ordered_attn(grp):
            sub = grp[grp["attn_variant"] == av].sort_values("suffix_len")
            if sub.empty:
                continue
            ax.plot(sub["suffix_len"], sub[col],
                    marker="o", color=_ATTN_COLOR.get(av, "#777"), label=av)
        ax.set_xlabel("Suffix length (cached MASK tokens)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)
        if col == "block_speedup":
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="break-even")

    plt.tight_layout()
    _save(fig, out / "23_context_ratio.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot diffusion_ar_sweep results.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Path to diffusion_ar_sweep.csv")
    parser.add_argument("--out", default="plots/diffusion_ar",
                        help="Output directory for PNG files (default: plots/diffusion_ar)")
    parser.add_argument("--figs", nargs="+", metavar="NN",
                        help="Only produce these figure numbers (e.g. --figs 01 04 11).")
    parser.add_argument("--list-figs", action="store_true")
    args = parser.parse_args(argv)

    if args.list_figs:
        print("Available figures:")
        for k, fn in sorted(PLOTS.items()):
            print(f"  {k}  {fn.__name__}")
        return

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    selected = args.figs if args.figs else list(PLOTS.keys())
    unknown  = set(selected) - set(PLOTS.keys())
    if unknown:
        parser.error(f"Unknown figure numbers: {sorted(unknown)}")

    print(f"Generating {len(selected)} figures → {out}/")
    for key in sorted(selected):
        PLOTS[key](df, out)

    print(f"\nDone — {len(selected)} figures saved to {out}/")


if __name__ == "__main__":
    main()
