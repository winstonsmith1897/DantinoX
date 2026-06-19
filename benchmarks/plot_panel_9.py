#!/usr/bin/env python3
"""
benchmarks/plot_panel_9.py
==========================
EMNLP System Demonstration — publication-quality 3 × 3 composite figure.

  Row 0  Serving Pareto frontier  — throughput vs latency, log–log
  Row 1  Energy per generated tok — mJ/tok vs batch size, log–log
  Row 2  Throughput head-to-head  — k tok/s vs batch size, linear–y

  Col 0  512d12b   Col 1  768d16b   Col 2  1024d16b

Output: results/paradigm_bench/fig_panel_9.{pdf,png}
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Design tokens ──────────────────────────────────────────────────────────────

BLUE   = "#2166ac"
RED    = "#d62728"
GREEN  = "#2ca02c"
SALMON = "#f4a582"
LGREEN = "#74c476"

# (label, color, linestyle, marker, linewidth, markersize)
SERIES = [
    ("AR (greedy)",      BLUE,   "-",  "o", 1.8, 4.5),
    ("Discrete S=32",    RED,    "-",  "s", 1.5, 4.0),
    ("Discrete S=8",     SALMON, "--", "s", 1.3, 3.5),
    ("Continuous S=32",  GREEN,  "-",  "^", 1.5, 4.0),
    ("Continuous S=8",   LGREEN, "--", "^", 1.3, 3.5),
]

ARCHS = ["512d12b", "768d16b", "1024d16b"]
COL_TITLES = {
    "512d12b":  "Small (512-d · 12-layer)",
    "768d16b":  "Medium (768-d · 16-layer)",
    "1024d16b": "Large (1024-d · 16-layer)",
}

# ── Global rc — clean, publication-ready ───────────────────────────────────────

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":            8,
    "axes.titlesize":       9.5,
    "axes.labelsize":       8.0,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "axes.linewidth":       0.8,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "axes.grid":            True,
    "grid.alpha":           0.5,
    "grid.color":           "#e5e5e5",
    "grid.linestyle":       "-",
    "grid.linewidth":       0.6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#cccccc",
    "legend.borderpad":     0.5,
})

# ── Data loading ───────────────────────────────────────────────────────────────

def _load(pattern: str, key: list[str]) -> pd.DataFrame | None:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        m = re.search(r"_(\d+d\d+b)", f)
        arch = m.group(1) if m else "512d12b"
        if "arch" not in d.columns:
            d["arch"] = arch
        else:
            d["arch"] = d["arch"].fillna(arch)
        mp = re.search(r"_(tf32|f32)\.csv$", f)
        if mp and "dtype" in d.columns:
            d["dtype"] = d["dtype"].fillna(mp.group(1))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "oom" not in df.columns:
        df["oom"] = False
    df["oom"] = df["oom"].astype(str).str.lower().eq("true")
    key_ok = [k for k in key if k in df.columns]
    return df.drop_duplicates(subset=key_ok, keep="last").reset_index(drop=True)


def _crossover_b(d_ar: pd.DataFrame, d_other: pd.DataFrame) -> float | None:
    m = pd.merge(
        d_ar[["batch_size", "tok_s_system"]],
        d_other[["batch_size", "tok_s_system"]],
        on="batch_size", suffixes=("_ar", "_o"),
    ).sort_values("batch_size")
    for _, r in m.iterrows():
        if r["tok_s_system_ar"] >= r["tok_s_system_o"]:
            return float(r["batch_size"])
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _x_batch_ticks(ax: plt.Axes, data: pd.DataFrame) -> None:
    bs = sorted(data["batch_size"].dropna().unique().astype(int))
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) if b in (1, 4, 16, 64, 256) else ""
                        for b in bs], fontsize=7)


# ── Row 0: Serving Pareto frontier ────────────────────────────────────────────

def _draw_pareto(ax: plt.Axes, par: pd.DataFrame, arch: str, col: int) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return

    for label, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["e2e_ms_med"], d["tok_s_system"],
                color=color, ls=ls, marker=mk, ms=ms, lw=lw,
                label=label, zorder=3, clip_on=True)

    # Crossover: AR vs Discrete S=32 — single annotation, clean style
    d_ar  = d_all[d_all.label == "AR (greedy)"]
    d_d32 = d_all[d_all.label == "Discrete S=32"]
    if not d_ar.empty and not d_d32.empty:
        bx = _crossover_b(d_ar, d_d32)
        if bx is not None:
            r = d_ar[d_ar.batch_size == bx]
            if not r.empty:
                x = float(r["e2e_ms_med"].iloc[0])
                y = float(r["tok_s_system"].iloc[0])
                ax.annotate(
                    f"AR overtakes\nDisc. S=32 @ B={int(bx)}",
                    xy=(x, y),
                    xytext=(x * 2.8, y * 0.55),
                    fontsize=6.5, fontweight="bold", color="#222222",
                    arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                   color="#555555", shrinkA=3, shrinkB=3),
                    ha="left", va="top",
                )

    # SLO guide
    ax.axvline(200, color="#888888", lw=0.9, ls=(0, (4, 3)), zorder=1)
    if col == 0:
        ax.text(175, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e3,
                "200 ms\nSLO", fontsize=6.5, color="#666666",
                ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.set_xlabel("Per-request latency (ms)", labelpad=3)
    if col == 0:
        ax.set_ylabel("System throughput (tok/s)", labelpad=4)


# ── Row 1: Energy per token ────────────────────────────────────────────────────

def _draw_energy(ax: plt.Axes, par: pd.DataFrame, arch: str, col: int) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    d_all = d_all[d_all["j_per_tok"].notna() & (d_all["j_per_tok"] > 0)]
    if d_all.empty:
        return

    for label, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        ymJ = d["j_per_tok"] * 1e3
        ax.plot(d["batch_size"], ymJ,
                color=color, ls=ls, marker=mk, ms=ms, lw=lw,
                label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _x_batch_ticks(ax, d_all)
    ax.set_xlabel("Batch size", labelpad=3)
    if col == 0:
        ax.set_ylabel("Energy / token (mJ)", labelpad=4)


# ── Row 2: System throughput head-to-head ────────────────────────────────────

def _draw_throughput(ax: plt.Axes, par: pd.DataFrame, arch: str, col: int) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return

    for label, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        yk = d["tok_s_system"] / 1e3
        ax.plot(d["batch_size"], yk,
                color=color, ls=ls, marker=mk, ms=ms, lw=lw,
                label=label)

    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    _x_batch_ticks(ax, d_all)
    ax.set_xlabel("Batch size", labelpad=3)
    if col == 0:
        ax.set_ylabel("Gen. throughput (k tok/s)", labelpad=4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out  = root / "results" / "paradigm_bench"
    out.mkdir(parents=True, exist_ok=True)

    par = _load(str(root / "results" / "ablation_pareto_*.csv"),
                ["arch", "label", "batch_size", "dtype"])
    if par is None:
        raise RuntimeError("No ablation_pareto_*.csv found.")
    par = par[par.arch.isin(ARCHS)]

    # ── Figure geometry ───────────────────────────────────────────────────────
    FIG_W        = 7.20    # inches — double-column EMNLP text width
    HEIGHT_RATIOS = [1.30, 1.00, 1.00]
    ROW_H_UNIT   = 2.10    # base inches per ratio unit
    FIG_H        = sum(r * ROW_H_UNIT for r in HEIGHT_RATIOS) + 0.80  # + legend

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Spaziature ottimizzate per avvicinare le figure
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=HEIGHT_RATIOS,
        hspace=0.25,  # Spazio verticale drasticamente ridotto (era 0.45)
        wspace=0.15,  # Spazio orizzontale ulteriormente ridotto (era 0.25)
        left=0.075, right=0.980,
        top=0.905,  bottom=0.115,
    )

    DRAW_FNS = [_draw_pareto, _draw_energy, _draw_throughput]

    axes: list[list[plt.Axes]] = [
        [fig.add_subplot(gs[row, col]) for col in range(3)]
        for row in range(3)
    ]

    # Draw panels
    for row, draw_fn in enumerate(DRAW_FNS):
        for col, arch in enumerate(ARCHS):
            draw_fn(axes[row][col], par, arch, col)

    # ── Column titles ─────────────────────────────────────────────────────────
    for col, arch in enumerate(ARCHS):
        axes[0][col].set_title(COL_TITLES[arch],
                               fontsize=9.0, fontweight="bold", pad=8)

    # ── Shared legend — bottom centre ─────────────────────────────────────────
    handles = [
        matplotlib.lines.Line2D(
            [], [],
            color=c, ls=ls, marker=mk, ms=5.5, lw=lw, label=lbl
        )
        for lbl, c, ls, mk, lw, ms in SERIES
    ]

    fig.legend(
        handles,
        [s[0] for s in SERIES],
        loc="lower center",
        bbox_to_anchor=(0.548, 0.012),
        ncol=5,
        fontsize=8.0,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.5,
        borderpad=0.6,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.text(
        0.548, 0.965,
        "Inference Efficiency: AR vs. Discrete Diffusion (LLaDA) vs. "
        "Continuous Diffusion (ELF)  ·  1×A100-40 GB, bf16, $G=256$ tokens",
        ha="center", va="bottom",
        fontsize=9.0, fontweight="bold", color="#111111",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(out / "fig_panel_9.pdf")
    fig.savefig(out / "fig_panel_9.png", dpi=300)
    plt.close(fig)
    print(f"Saved  {out}/fig_panel_9.{{pdf,png}}")


if __name__ == "__main__":
    main()