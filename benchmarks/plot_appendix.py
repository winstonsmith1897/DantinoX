#!/usr/bin/env python3
"""
benchmarks/plot_appendix.py
===========================

Paper-ready appendix figures, distilled from the exploratory plots of
``plot_inference.py`` and ``plot_paradigm_bench.py``. Every figure is
sized for a SINGLE two-column-paper column (no double-column figure*):

  appx_attention_type.pdf/png   bars: prefill latency & decode throughput
                                per attention variant (MLA sorted by down-dim)
  appx_batch_size.pdf/png       decode throughput & prefill latency vs batch
                                size (log2 x)
  appx_context_len.pdf/png      KV cache memory & decode throughput vs
                                context length (log2 x; prefill panel dropped:
                                its 3 ms range is measurement noise)
  appx_paradigm_scaling.pdf/png generation-length scaling (top 3 rows) and
                                batch-size scaling (bottom 3 rows) per
                                paradigm, one shared legend (MHA; GQA/MLA
                                are qualitatively identical, see caption)

Style: no suptitles (captions live in LaTeX), colorblind-safe palettes
(Okabe-Ito for paradigms), fonts sized for a single ~3.3in column.

Usage:
  python benchmarks/plot_appendix.py \
      --sweep-csv results/inference_sweep.csv \
      --bench-csv results/paradigm_bench_full.csv \
      --out results/appendix_figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Attention variants (validated colorblind-safe triple)
ATTN_VARIANTS = ["MHA", "GQA", "MLA"]
ATTN_PALETTE  = {"MHA": "#4C72B0", "GQA": "#C1571F", "MLA": "#6B4FA1"}
ATTN_MARKERS  = {"MHA": "o", "GQA": "s", "MLA": "^"}

# Paradigms (Okabe-Ito: blue / vermillion / bluish green)
PARADIGMS = ["AR", "Discrete", "Continuous"]
PARA_PALETTE = {"AR": "#0072B2", "Discrete": "#D55E00", "Continuous": "#009E73"}
PARA_MARKERS = {"AR": "o", "Discrete": "s", "Continuous": "^"}
PARA_NICE    = {"AR": "AR", "Discrete": "Disc. Diff.", "Continuous": "Cont. Diff."}
ATTN_NICE    = {"MHA": "MHA", "GQA": "GQA-1/4", "MLA": "MLA"}

# Single-column width for a two-column paper (EMNLP/ACL ~3.3in column)
COL_W = 3.3

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.5, "lines.markersize": 4,
})


def _k_fmt(v: float, _pos=None) -> str:
    if v >= 1000:
        s = f"{v / 1000:g}"
        return f"{s}k"
    return f"{v:.0f}"


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# ── inference_sweep.csv figures ───────────────────────────────────────────────

ATTN_LABEL_ORDER = ["MHA", "GQA-1/2", "GQA-1/4", "GQA-1/8",
                    "MLA-down32", "MLA-down64", "MLA-down128"]


def fig_attention_type(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "attention_type"].copy()
    sub["label"] = pd.Categorical(sub["label"], ATTN_LABEL_ORDER, ordered=True)
    sub = sub.sort_values("label")
    colors = [ATTN_PALETTE[v] for v in sub["attn_variant"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for ax, col, ylabel in [(ax1, "prefill_ms_p50", "Prefill latency (ms, p50)"),
                            (ax2, "decode_tok_s",   "Throughput (tok/s)")]:
        x = np.arange(len(sub))
        bars = ax.bar(x, sub[col], color=colors, edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.0f", padding=1.5, fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, sub[col].max() * 1.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=ATTN_PALETTE[v], label=v)
                        for v in ATTN_VARIANTS],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
    _save(fig, out, "appx_attention_type")


def fig_batch_size(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "batch_size"].dropna(subset=["batch_size"]).copy()
    all_bs = sorted(sub["batch_size"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for var in ATTN_VARIANTS:
        s = sub[sub["attn_variant"] == var].sort_values("batch_size")
        if s.empty:
            continue
        ax1.plot(s["batch_size"], s["decode_tok_s"],
                 marker=ATTN_MARKERS[var], color=ATTN_PALETTE[var], label=var)
        ax2.plot(s["batch_size"], s["prefill_ms_p50"],
                 marker=ATTN_MARKERS[var], color=ATTN_PALETTE[var], label=var)
    mha_bs1 = sub[(sub["attn_variant"] == "MHA") & (sub["batch_size"] == 1)]["decode_tok_s"]
    if not mha_bs1.empty:
        ax1.plot(all_bs, [float(mha_bs1.iloc[0]) * b for b in all_bs], "--",
                 color="#999999", linewidth=1.1, label="ideal (MHA)")
    for ax, ylabel in [(ax1, "Decode through. (tok/s)"),
                       (ax2, "Prefill latency (ms, p50)")]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_bs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
    ax2.set_ylim(0, sub["prefill_ms_p50"].max() * 1.15)
    ax1.legend(frameon=False, fontsize=7, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    _save(fig, out, "appx_batch_size")


def fig_context_len(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["group"] == "context_len"].dropna(subset=["max_context"]).copy()
    ctxs = sorted(sub["max_context"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for ax, col, ylabel in [(ax1, "kv_cache_mb",  "KV cache memory (MB)"),
                            (ax2, "decode_tok_s", "Decode through. (tok/s)")]:
        for var in ATTN_VARIANTS:
            s = sub[sub["attn_variant"] == var].dropna(subset=[col]).sort_values("max_context")
            if s.empty:
                continue
            ax.plot(s["max_context"], s[col],
                    marker=ATTN_MARKERS[var], color=ATTN_PALETTE[var], label=var)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctxs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, sub["decode_tok_s"].max() * 1.15)
    ax1.legend(frameon=False, fontsize=7, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    _save(fig, out, "appx_context_len")


# ── paradigm_bench_full.csv: single combined figure ───────────────────────────
# (single attention variant — full 3x3 grids don't fit one column legibly;
#  GQA/MLA verified qualitatively identical, noted in the caption. Both
#  scaling studies share the same 3 paradigm series, so they are drawn as
#  one figure with one shared legend instead of two figures with duplicate
#  legends.)

def _plot_metric_col(ax, sub: pd.DataFrame, x: str, col: str, ylabel: str,
                     logy: bool) -> None:
    vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
    for p in PARADIGMS:
        d = sub[sub["paradigm"] == p].sort_values(x).dropna(subset=[col])
        if d.empty:
            continue
        ax.plot(d[x], d[col], marker=PARA_MARKERS[p],
                color=PARA_PALETTE[p], label=PARA_NICE[p])
    ax.set_xscale("log", base=2)
    if logy and not vals.empty and (vals > 0).all():
        ax.set_yscale("log")
        ax.set_ylim(vals.min() * 0.7, vals.max() * 1.4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_fmt))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    elif not vals.empty:
        ax.set_ylim(0, vals.max() * 1.1)
    ax.set_ylabel(ylabel)


def fig_paradigm_scaling(df: pd.DataFrame, out: Path, attn: str = "MHA") -> None:
    genlen_metrics = [("e2e_ms",      "E2E lat. (ms)",  True),
                      ("tok_s_e2e",   "Through. (tok/s)", True),
                      ("peak_mem_mb", "Device mem (MB)",    False)]
    batch_metrics  = [("tok_s_e2e", "Through. (tok/s)", True),
                      ("e2e_ms",    "E2E lat. (ms)",   True),
                      ("mfu_pct",   "MFU (%)",             False)]

    fig = plt.figure(figsize=(COL_W, 4.1))
    gs = fig.add_gridspec(3, 2, left=0.22, right=0.98, top=0.90, bottom=0.10,
                          hspace=0.15, wspace=0.75)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
    for r in range(1, 3):
        axes[r][0].sharex(axes[0][0])
        axes[r][1].sharex(axes[0][1])

    sub_genlen = df[(df["group"] == "gen_len") & (~df["oom"]) & (df["attn"] == attn)]
    for r, (col, ylabel, logy) in enumerate(genlen_metrics):
        _plot_metric_col(axes[r][0], sub_genlen, "gen_len", col, ylabel, logy)
    axes[0][0].set_title("Gen. length", fontsize=8, pad=3)
    axes[-1][0].set_xlabel("Gen. length")

    sub_batch = df[(df["group"] == "batch_size") & (~df["oom"]) & (df["attn"] == attn)]
    for r, (col, ylabel, logy) in enumerate(batch_metrics):
        _plot_metric_col(axes[r][1], sub_batch, "batch_size", col, ylabel, logy)
    axes[0][1].set_title("Batch size", fontsize=8, pad=3)
    axes[-1][1].set_xlabel("Batch size")

    for r in range(3):
        for c in range(2):
            axes[r][c].tick_params(axis="both", labelsize=7)
            axes[r][c].yaxis.set_label_coords(-0.36, 0.5)
            if r < 2:
                axes[r][c].tick_params(axis="x", labelbottom=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=8, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.005))
    _save(fig, out, "appx_paradigm_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-csv", default="results/inference_sweep.csv")
    parser.add_argument("--bench-csv", default="results/paradigm_bench_full.csv")
    parser.add_argument("--out", default="results/appendix_figs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(args.sweep_csv)
    print(f"inference_sweep: {len(sweep)} rows")
    fig_attention_type(sweep, out)
    fig_batch_size(sweep, out)
    fig_context_len(sweep, out)

    bench = pd.read_csv(args.bench_csv)
    if "attn" not in bench.columns:
        bench["attn"] = "MHA"
    bench["attn"] = bench["attn"].fillna("MHA")
    print(f"paradigm_bench: {len(bench)} rows")
    fig_paradigm_scaling(bench, out)
    print("Done.")


if __name__ == "__main__":
    main()
