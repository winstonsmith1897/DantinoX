#!/usr/bin/env python3
"""
benchmarks/plot_emnlp.py
=========================

Generate EMNLP 2026 submission-quality figures from DantinoX benchmark CSVs.

Figures produced
----------------
  fig1_attention_paradigm_matrix.pdf/png
        3×2 grouped bar chart: rows = MHA/GQA/MLA, cols = AR/Diffusion.
        Each cell shows PPL (from perplexity.csv) and KV-cache MB.
        Uses the Pareto-optimal run (lowest PPL) for each combination.

  fig2_pareto_kv_vs_ppl.pdf/png
        Scatter plot: x = KV-cache MB, y = validation PPL.
        One marker per (run, attn_variant) coloured by paradigm.
        Pareto frontier (lowest PPL for each cache budget) highlighted.

  fig3_pareto_kv_vs_toks.pdf/png
        Scatter plot: x = KV-cache MB, y = throughput (tok/s).
        Pareto frontier (highest tok/s per cache budget) highlighted.

  fig4_diffusion_steps_gentime.pdf/png
        Line chart: x = #diffusion steps, y = estimated generation time (ms).
        One line per attn_variant (from diffusion_ar_sweep.csv group diff_steps).

  fig5_dualcache_speedup.pdf/png
        Bar/line chart: x = block_size, y = block_speedup vs full-sequence step.
        From diffusion_ar_sweep.csv group block_size_sweep.

  fig6_confidence_tradeoff.pdf/png
        Dual-axis line chart: x = τ (threshold) or f (factor),
        y_left = avg_steps_to_complete (quality proxy, lower = faster),
        y_right = tok/s throughput.
        One panel per strategy (threshold / factor).

  fig7_generation_quality_radar.pdf/png
        Radar / spider chart: distinct_1, distinct_2, 1-rep_4, 1-self_bleu_4.
        One polygon per (model_type × attn_variant).

  fig8_ablation_table.pdf/png
        Visual table: rows = ablation conditions (sliding_window, no_sink, moe),
        cols = PPL / tok/s / KV-MB.  Formatted for inclusion in LaTeX via
        tikz/pgfplots or as a standalone PNG.

Input CSVs (all optional — missing CSVs are gracefully skipped)
--------------------------------------------------------------
  --ppl-csv              results/perplexity.csv
  --trained-csv          results/benchmark_results.csv
  --diffusion-ar-csv     results/diffusion_ar_sweep.csv
  --confidence-csv       results/confidence_sweep.csv
  --gen-quality-csv      results/generation_quality.csv

Usage
-----
  python benchmarks/plot_emnlp.py
  python benchmarks/plot_emnlp.py --out-dir results/paper_figures/
  python benchmarks/plot_emnlp.py --figs 1 2 4 6
  python benchmarks/plot_emnlp.py --pdf   # save PDF in addition to PNG
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
except ImportError as exc:
    print(f"Missing dependency: {exc}\nInstall: pip install pandas matplotlib")
    sys.exit(1)

log = logging.getLogger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────

_ATTN_COLOR  = {"MHA": "#1565C0", "GQA": "#2E7D32", "MLA": "#6A1B9A"}
_MODEL_COLOR = {"autoregressive": "#1976D2", "AR": "#1976D2",
                "diffusion": "#E65100",     "Diff": "#E65100"}
_ATTN_MARKER = {"MHA": "o", "GQA": "s", "MLA": "^"}
_MODEL_LS    = {"AR": "-", "autoregressive": "-", "Diff": "--", "diffusion": "--"}

# Use LaTeX-compatible fonts
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     ":",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, pdf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _attn_legend(ax: plt.Axes) -> None:
    handles = [
        mpatches.Patch(color=c, label=a)
        for a, c in _ATTN_COLOR.items()
    ]
    ax.legend(handles=handles, title="Attention", fontsize=8, title_fontsize=8)


def _pareto_front(x: np.ndarray, y: np.ndarray, minimize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] <= x[i]) if minimize_x else (x[j] >= x[i])
            y_better = (y[j] <= y[i]) if minimize_y else (y[j] >= y[i])
            x_strict = (x[j] < x[i])  if minimize_x else (x[j] > x[i])
            y_strict = (y[j] < y[i])  if minimize_y else (y[j] > y[i])
            if x_better and y_better and (x_strict or y_strict):
                dominated[i] = True
                break
    return ~dominated


# ── Figure 1: Attention × Paradigm matrix ────────────────────────────────────

def fig1_attention_paradigm_matrix(
    trained_csv: Path,
    ppl_csv: Path,
    out_dir: Path,
    pdf: bool,
) -> None:
    """Grouped bar chart: rows = attention, cols = paradigm; metrics = PPL + KV-MB."""
    rows_list: list[dict] = []

    if trained_csv.exists():
        df_t = pd.read_csv(trained_csv)
        for _, r in df_t.iterrows():
            attn = str(r.get("type", "?"))
            mtype = str(r.get("model_type", "autoregressive"))
            rows_list.append({
                "attn": attn,
                "paradigm": "AR" if "auto" in mtype else "Diff",
                "val_ppl": float(r.get("val_ppl", np.nan)),
                "kv_mb":   float(r.get("theoretical_cache_mb", np.nan)),
                "tok_s":   float(r.get("tok_s", np.nan)),
            })

    if ppl_csv.exists():
        df_p = pd.read_csv(ppl_csv)
        ext  = df_p[df_p["dataset"] != "train_val"]
        for key, grp in ext.groupby(["attn_variant", "model_type"]):
            attn_v = str(key[0]) if isinstance(key, tuple) else str(key)
            mtype  = str(key[1]) if isinstance(key, tuple) else "unknown"
            mean_ppl = float(grp["ppl"].mean())
            rows_list.append({
                "attn": attn_v,
                "paradigm": "AR" if "auto" in mtype else "Diff",
                "val_ppl": mean_ppl,
                "kv_mb":   np.nan,
                "tok_s":   np.nan,
            })

    if not rows_list:
        log.warning("fig1: no data — skipping")
        return

    df = pd.DataFrame(rows_list)
    attns     = ["MHA", "GQA", "MLA"]
    paradigms = ["AR", "Diff"]
    metrics   = [("val_ppl", "Validation PPL ↓"), ("kv_mb", "KV-cache MB ↓"), ("tok_s", "Throughput (tok/s) ↑")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(attns))
    w = 0.35

    for ax, (col, ylabel) in zip(axes, metrics):
        for i, par in enumerate(paradigms):
            vals = []
            for attn in attns:
                sub = df[(df["attn"] == attn) & (df["paradigm"] == par)][col].dropna()
                vals.append(float(sub.mean()) if len(sub) > 0 else np.nan)
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, width=w,
                          color=_MODEL_COLOR[par], alpha=0.85,
                          label=par, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(attns)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" ")[0])

    handles = [mpatches.Patch(color=_MODEL_COLOR[p], label=p) for p in paradigms]
    axes[0].legend(handles=handles, title="Paradigm", fontsize=8)
    fig.suptitle("DantinoX: Attention Type × Generation Paradigm", fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig1_attention_paradigm_matrix.png", pdf)


# ── Figure 2: Pareto frontier KV-MB vs PPL ───────────────────────────────────

def fig2_pareto_kv_ppl(trained_csv: Path, ppl_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not trained_csv.exists():
        log.warning("fig2: %s not found — skipping", trained_csv)
        return

    df = pd.read_csv(trained_csv)
    if "val_ppl" not in df.columns or "theoretical_cache_mb" not in df.columns:
        log.warning("fig2: required columns missing — skipping")
        return

    df = df.dropna(subset=["val_ppl", "theoretical_cache_mb"])
    if df.empty:
        log.warning("fig2: empty data — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for attn in ["MHA", "GQA", "MLA"]:
        for mtype, ls in [("autoregressive", "-"), ("diffusion", "--")]:
            sub = df[(df["type"] == attn) & (df.get("model_type", pd.Series(["autoregressive"] * len(df))) == mtype)]
            if sub.empty:
                continue
            label = f"{attn} / {'AR' if 'auto' in mtype else 'Diff'}"
            ax.scatter(sub["theoretical_cache_mb"], sub["val_ppl"],
                       color=_ATTN_COLOR.get(attn, "grey"),
                       marker=_ATTN_MARKER.get(attn, "o"),
                       alpha=0.7, s=60, zorder=3, label=label)

    # Pareto frontier across all runs (minimise both KV-MB and PPL)
    x_all = df["theoretical_cache_mb"].values
    y_all = df["val_ppl"].values
    mask  = _pareto_front(x_all, y_all, minimize_x=True, minimize_y=True)
    if mask.sum() > 1:
        idx_sorted = np.argsort(x_all[mask])
        ax.plot(x_all[mask][idx_sorted], y_all[mask][idx_sorted],
                "k--", linewidth=1.5, label="Pareto frontier", zorder=4)
        ax.scatter(x_all[mask], y_all[mask], color="black", s=80, zorder=5, marker="*")

    ax.set_xlabel("KV-cache Memory (MB) ↓")
    ax.set_ylabel("Validation PPL ↓")
    ax.set_title("Memory–Quality Pareto Frontier")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig2_pareto_kv_vs_ppl.png", pdf)


# ── Figure 3: Pareto KV-MB vs throughput ─────────────────────────────────────

def fig3_pareto_kv_toks(trained_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not trained_csv.exists():
        log.warning("fig3: %s not found — skipping", trained_csv)
        return

    df = pd.read_csv(trained_csv).dropna(subset=["tok_s", "theoretical_cache_mb"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for attn in ["MHA", "GQA", "MLA"]:
        sub = df[df["type"] == attn]
        if sub.empty:
            continue
        ax.scatter(sub["theoretical_cache_mb"], sub["tok_s"],
                   color=_ATTN_COLOR.get(attn, "grey"),
                   marker=_ATTN_MARKER.get(attn, "o"),
                   s=60, alpha=0.8, label=attn, zorder=3)

    x_all = df["theoretical_cache_mb"].values
    y_all = df["tok_s"].values
    mask  = _pareto_front(x_all, y_all, minimize_x=True, minimize_y=False)
    if mask.sum() > 1:
        idx = np.argsort(x_all[mask])
        ax.plot(x_all[mask][idx], y_all[mask][idx], "k--", linewidth=1.5,
                label="Pareto frontier", zorder=4)
        ax.scatter(x_all[mask], y_all[mask], color="black", s=80, zorder=5, marker="*")

    ax.set_xlabel("KV-cache Memory (MB) ↓")
    ax.set_ylabel("Throughput (tok/s) ↑")
    ax.set_title("Memory–Throughput Pareto Frontier")
    _attn_legend(ax)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig3_pareto_kv_vs_toks.png", pdf)


# ── Figure 4: Diffusion steps vs generation time ─────────────────────────────

def fig4_diff_steps_gentime(diff_ar_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not diff_ar_csv.exists():
        log.warning("fig4: %s not found — skipping", diff_ar_csv)
        return

    df = pd.read_csv(diff_ar_csv)
    sub = df[(df["group"] == "diff_steps") & (df["model_type"] == "Diff")].dropna(
        subset=["diff_n_steps", "diff_step_ms_p50"]
    )
    if sub.empty:
        log.warning("fig4: no diff_steps data — skipping")
        return

    sub = sub.copy()
    sub["gen_time_ms"] = sub["diff_n_steps"] * sub["diff_step_ms_p50"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for attn in ["MHA", "GQA", "MLA"]:
        pts = sub[sub["attn_variant"] == attn].sort_values("diff_n_steps")
        if pts.empty:
            continue
        ax.plot(pts["diff_n_steps"], pts["gen_time_ms"],
                color=_ATTN_COLOR.get(attn, "grey"), marker=_ATTN_MARKER.get(attn, "o"),
                label=attn, linewidth=2, markersize=6)

    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Estimated Generation Time (ms)")
    ax.set_title("Diffusion Steps vs. Generation Time")
    _attn_legend(ax)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig4_diffusion_steps_gentime.png", pdf)


# ── Figure 5: DualCache block-size speedup ───────────────────────────────────

def fig5_dualcache_speedup(diff_ar_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not diff_ar_csv.exists():
        log.warning("fig5: %s not found — skipping", diff_ar_csv)
        return

    df = pd.read_csv(diff_ar_csv)
    sub = df[df["group"] == "block_size_sweep"].dropna(subset=["block_speedup", "block_size"])
    if sub.empty:
        log.warning("fig5: no block_size_sweep data — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for attn in ["MHA", "GQA"]:
        pts = sub[sub["attn_variant"] == attn].sort_values("block_size")
        if pts.empty:
            continue
        axes[0].plot(pts["block_size"], pts["block_speedup"],
                     color=_ATTN_COLOR.get(attn, "grey"),
                     marker=_ATTN_MARKER.get(attn, "o"),
                     label=attn, linewidth=2, markersize=7)
        axes[1].plot(pts["block_size"], pts["refresh_overhead_x"],
                     color=_ATTN_COLOR.get(attn, "grey"),
                     marker=_ATTN_MARKER.get(attn, "o"),
                     label=attn, linewidth=2, markersize=7)

    axes[0].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Block Size (tokens)")
    axes[0].set_ylabel("Speedup vs Full-Sequence Step  ↑")
    axes[0].set_title("DualCache Block Speedup")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Block Size (tokens)")
    axes[1].set_ylabel("Cache Refresh Overhead (× decode_block)  ↓")
    axes[1].set_title("Cache Refresh Cost")
    axes[1].legend(fontsize=8)

    fig.suptitle("Fast-dLLM Block-wise DualCache (DantinoX)", fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir / "fig5_dualcache_speedup.png", pdf)


# ── Figure 6: Confidence threshold tradeoff ──────────────────────────────────

def fig6_confidence_tradeoff(conf_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not conf_csv.exists():
        log.warning("fig6: %s not found — skipping", conf_csv)
        return

    df = pd.read_csv(conf_csv)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    strategy_info = [
        ("threshold", "τ (confidence threshold)", _THRESHOLD_VALUES),
        ("factor",    "f (factor parameter)",     _FACTOR_VALUES),
    ]

    for ax, (strat, xlabel, _) in zip(axes, strategy_info):
        sub = df[df["strategy"] == strat]
        ax2 = ax.twinx()
        for attn in ["MHA", "GQA", "MLA"]:
            pts = sub[(sub["attn_variant"] == attn) & (sub["seq_len"] == 64)].sort_values("param")
            if pts.empty:
                continue
            c = _ATTN_COLOR.get(attn, "grey")
            ax.plot(pts["param"], pts["avg_steps_to_complete"],
                    color=c, linestyle="-",  marker="o", markersize=5,
                    linewidth=2, label=f"{attn} steps")
            ax2.plot(pts["param"], pts["tok_s"],
                     color=c, linestyle="--", marker="s", markersize=5,
                     linewidth=1.5, alpha=0.7, label=f"{attn} tok/s")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Avg Steps to Complete  ↓", color="#333")
        ax2.set_ylabel("Throughput (tok/s)  ↑", color="#666")
        ax.set_title(f"Strategy: {strat}")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle("Confidence-Aware Parallel Decoding: Speed vs Quality Proxy",
                 fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir / "fig6_confidence_tradeoff.png", pdf)


# ── Figure 7: Generation quality radar ───────────────────────────────────────

def fig7_generation_radar(gen_csv: Path, out_dir: Path, pdf: bool) -> None:
    if not gen_csv.exists():
        log.warning("fig7: %s not found — skipping", gen_csv)
        return

    df = pd.read_csv(gen_csv)
    if df.empty:
        return

    metrics   = ["distinct_1", "distinct_2", "1-rep_4", "1-self_bleu_4"]
    raw_cols  = ["distinct_1", "distinct_2", "rep_4",   "self_bleu_4"]

    # Build a version where higher is always better
    df["1-rep_4"]       = 1.0 - df["rep_4"].clip(0, 1)
    df["1-self_bleu_4"] = 1.0 - df["self_bleu_4"].clip(0, 1)

    groups = df.groupby(["model_type", "attn_variant"])[metrics].mean()
    if groups.empty:
        return

    N   = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)

    for (mtype, attn), row in groups.iterrows():
        vals   = row[metrics].tolist()
        vals  += vals[:1]
        color  = _ATTN_COLOR.get(str(attn), "grey")
        ls     = _MODEL_LS.get(str(mtype), "-")
        paradigm = "AR" if "auto" in str(mtype) else "Diff"
        label  = f"{attn} / {paradigm}"
        ax.plot(angles, vals, linestyle=ls, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title("Generation Quality (higher = better)", pad=20, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir / "fig7_generation_quality_radar.png", pdf)


# ── Figure 8: Summary table ───────────────────────────────────────────────────

def fig8_summary_table(
    trained_csv: Path, ppl_csv: Path, out_dir: Path, pdf: bool
) -> None:
    """Render a visual summary table as a matplotlib figure."""
    rows_data: list[dict] = []

    if trained_csv.exists():
        df_t = pd.read_csv(trained_csv)
        for attn in ["MHA", "GQA", "MLA"]:
            for mtype in ["autoregressive", "diffusion"]:
                sub = df_t[(df_t["type"] == attn) &
                           (df_t.get("model_type", pd.Series(["autoregressive"] * len(df_t))) == mtype)]
                if sub.empty:
                    continue
                row: dict = {
                    "Attn": attn,
                    "Paradigm": "AR" if "auto" in mtype else "Diff",
                    "Val-PPL": f"{sub['val_ppl'].mean():.1f}" if "val_ppl" in sub else "—",
                    "tok/s":   f"{sub['tok_s'].mean():.0f}"  if "tok_s"  in sub else "—",
                    "KV-MB":   f"{sub['theoretical_cache_mb'].mean():.1f}" if "theoretical_cache_mb" in sub else "—",
                    "Params-M": f"{sub['params_m'].mean():.1f}" if "params_m" in sub else "—",
                }
                rows_data.append(row)

    if not rows_data:
        log.warning("fig8: no data for summary table — skipping")
        return

    df_tbl = pd.DataFrame(rows_data)
    cols   = list(df_tbl.columns)
    n_rows = len(df_tbl)
    n_cols = len(cols)

    fig, ax = plt.subplots(figsize=(n_cols * 1.4, n_rows * 0.5 + 1.2))
    ax.axis("off")

    table = ax.table(
        cellText=df_tbl.values.tolist(),
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        if r > 0 and c == 0:
            attn_val = df_tbl.iloc[r - 1]["Attn"]
            cell.set_text_props(color=_ATTN_COLOR.get(str(attn_val), "black"), fontweight="bold")

    ax.set_title("DantinoX — Attention × Paradigm Summary",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    _savefig(fig, out_dir / "fig8_summary_table.png", pdf)


# ── Main ──────────────────────────────────────────────────────────────────────

_FIGS: dict[int, str] = {
    1: "Attention × Paradigm matrix",
    2: "Pareto frontier KV-MB vs PPL",
    3: "Pareto frontier KV-MB vs tok/s",
    4: "Diffusion steps vs gen time",
    5: "DualCache block speedup",
    6: "Confidence threshold tradeoff",
    7: "Generation quality radar",
    8: "Summary table",
}

# Keep a reference to threshold/factor values needed in fig6
_THRESHOLD_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
_FACTOR_VALUES    = [0.80, 1.00, 1.20, 1.50, 2.00, 3.00, 5.00]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="EMNLP 2026 paper figures for DantinoX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir",          default="results/paper_figures/",
                        help="Output directory (default: results/paper_figures/)")
    parser.add_argument("--ppl-csv",          default="results/perplexity.csv")
    parser.add_argument("--trained-csv",      default="results/benchmark_results.csv")
    parser.add_argument("--diffusion-ar-csv", default="results/diffusion_ar_sweep.csv")
    parser.add_argument("--confidence-csv",   default="results/confidence_sweep.csv")
    parser.add_argument("--gen-quality-csv",  default="results/generation_quality.csv")
    parser.add_argument("--figs", nargs="+", type=int,
                        help="Figure numbers to generate (default: all). "
                             f"Available: {list(_FIGS.keys())}")
    parser.add_argument("--pdf", action="store_true",
                        help="Also save PDF versions (for LaTeX inclusion)")
    args = parser.parse_args(argv)

    out_dir      = Path(args.out_dir)
    ppl_csv      = Path(args.ppl_csv)
    trained_csv  = Path(args.trained_csv)
    diff_ar_csv  = Path(args.diffusion_ar_csv)
    conf_csv     = Path(args.confidence_csv)
    gen_csv      = Path(args.gen_quality_csv)

    figs_to_run = set(args.figs) if args.figs else set(_FIGS.keys())

    print(f"DantinoX EMNLP figures → {out_dir}")
    for fig_id, label in _FIGS.items():
        if fig_id not in figs_to_run:
            continue
        print(f"  Figure {fig_id}: {label}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print()

    dispatch = {
        1: lambda: fig1_attention_paradigm_matrix(trained_csv, ppl_csv, out_dir, args.pdf),
        2: lambda: fig2_pareto_kv_ppl(trained_csv, ppl_csv, out_dir, args.pdf),
        3: lambda: fig3_pareto_kv_toks(trained_csv, out_dir, args.pdf),
        4: lambda: fig4_diff_steps_gentime(diff_ar_csv, out_dir, args.pdf),
        5: lambda: fig5_dualcache_speedup(diff_ar_csv, out_dir, args.pdf),
        6: lambda: fig6_confidence_tradeoff(conf_csv, out_dir, args.pdf),
        7: lambda: fig7_generation_radar(gen_csv, out_dir, args.pdf),
        8: lambda: fig8_summary_table(trained_csv, ppl_csv, out_dir, args.pdf),
    }

    for fig_id in sorted(figs_to_run):
        if fig_id not in dispatch:
            log.warning("Unknown figure ID: %d", fig_id)
            continue
        try:
            dispatch[fig_id]()
        except Exception as exc:
            log.warning("Figure %d failed: %s", fig_id, exc)

    from pathlib import Path as _P
    n_files = len(list(out_dir.glob("fig*.png")))
    print(f"\nDone — {n_files} figures in {out_dir}")


if __name__ == "__main__":
    main()
