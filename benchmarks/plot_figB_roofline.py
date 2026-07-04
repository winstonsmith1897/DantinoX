#!/usr/bin/env python3
"""
benchmarks/plot_figB_roofline.py
=================================
Publication-quality roofline figure — Large backbone (1024-d · 16-layer).

Claim: AR decode is memory-bound (I ≈ 1–34 FLOP/byte); diffusion steps are
compute-bound (I ≈ 63–3 800 FLOP/byte), achieving up to 19 % of bf16 peak.

Output: results/paradigm_bench/figB_roofline_1024d16b.{pdf,png}
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Hardware constants — A100-40 GB SXM4 ──────────────────────────────────────
PEAK_BF16_TF = 312.0          # TF/s (BF16 tensor cores)
HBM_BW_TBS   = 1.555          # TB/s
RIDGE         = PEAK_BF16_TF / HBM_BW_TBS   # ~200.6 FLOP/byte

# ── Palette (consistent with fig_panel_9) ─────────────────────────────────────
C_AR   = "#1f78b4"
C_DISC = "#d62728"
C_CONT = "#2ca02c"

PARADIGM_STYLE = {
    "AR":         dict(color=C_AR,   marker="o", label="AR"),
    "Discrete":   dict(color=C_DISC, marker="s", label="Discrete diff."),
    "Continuous": dict(color=C_CONT, marker="^", label="Continuous diff."),
}

# ── rc — matches fig_panel_9 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         7,
    "axes.labelsize":    7.0,
    "legend.fontsize":   6.0,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
    "xtick.major.size":  3.0,
    "ytick.major.size":  3.0,
    "xtick.minor.size":  1.6,
    "ytick.minor.size":  1.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "axes.linewidth":    0.7,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         False,          # manual grid for cleaner look
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.axisbelow":    True,
    "legend.frameon":    True,
    "legend.framealpha": 0.96,
    "legend.edgecolor":  "#bbbbbb",
    "legend.borderpad":  0.45,
})

# ── Data ──────────────────────────────────────────────────────────────────────

def _load_grid(root: Path, arch: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "results" / "ablation_grid_*.csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        m = re.search(r"_(\d+d\d+b)", f)
        d["arch"] = m.group(1) if m else "?"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["oom"] = df["oom"].astype(str).str.lower().eq("true")
    df = df[(df.arch == arch) & (~df.oom)].dropna(
        subset=["step_gflops", "step_gbytes", "step_ms_med"])
    df = df.copy()
    df["intensity"] = df["step_gflops"] / df["step_gbytes"]
    df["achieved"]  = df["step_gflops"] / df["step_ms_med"]   # TF/s
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_minor(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(mticker.LogLocator(subs="auto"))
        axis.set_minor_formatter(mticker.NullFormatter())


def _light_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#dddddd", lw=0.4, zorder=0)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.25, zorder=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out  = root / "results" / "paradigm_bench"
    out.mkdir(parents=True, exist_ok=True)

    arch = "1024d16b"
    df   = _load_grid(root, arch)
    if df.empty:
        raise RuntimeError(f"No data found for {arch}")

    fig, ax = plt.subplots(figsize=(3.33, 2.05))
    fig.subplots_adjust(left=0.148, right=0.975, top=0.962, bottom=0.130)

    # ── Regime shading ────────────────────────────────────────────────────────
    ax.axvspan(0.3,    RIDGE, alpha=0.055, color=C_AR,   lw=0, zorder=0)
    ax.axvspan(RIDGE, 1.2e4,  alpha=0.055, color=C_DISC, lw=0, zorder=0)

    # ── Reference lines ───────────────────────────────────────────────────────
    xs = np.logspace(-0.5, 4.2, 400)
    roof = np.minimum(xs * HBM_BW_TBS, PEAK_BF16_TF)
    ax.plot(xs, roof, color="#111111", lw=1.3, zorder=4)

    # Peak and 10 % ceilings
    for frac, lbl, yoff in (
        (1.0, f"BF16 peak ({PEAK_BF16_TF:.0f} TF/s)", 1.08),
        (0.1, "10% peak", 1.08),
    ):
        yval = PEAK_BF16_TF * frac
        ax.axhline(yval, color="#aaaaaa", lw=0.55, ls=(0, (5, 4)), zorder=1)
        ax.text(0.48, yval * yoff, lbl, fontsize=5.5, color="#444444", va="bottom")

    # Ridge line + label placed BELOW the roofline kink (in data coords)
    ax.axvline(RIDGE, color="#aaaaaa", lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.text(RIDGE * 1.12, PEAK_BF16_TF * 0.40,
            f"Ridge\n{RIDGE:.0f} FLOP/B",
            fontsize=5.5, color="#444444",
            va="center", ha="left", style="italic")

    # ── Data scatter + trajectory ──────────────────────────────────────────────
    for p, st in PARADIGM_STYLE.items():
        d = df[df.paradigm == p].sort_values("intensity").copy()
        if d.empty:
            continue
        # Trajectory line (dashed, behind markers)
        ax.plot(d["intensity"], d["achieved"],
                color=st["color"], lw=0.8, ls="--", alpha=0.45, zorder=3)
        # Scatter: marker size ∝ log(B × G) = tokens in flight
        s = 14 + 3.0 * np.log2(d.batch_size * d.gen_len)
        ax.scatter(
            d["intensity"], d["achieved"],
            s=s, color=st["color"], marker=st["marker"],
            edgecolors="white", linewidths=0.35,
            alpha=0.92, zorder=5, label=st["label"],
        )

    # ── Cluster annotations ───────────────────────────────────────────────────
    d_ar = df[df.paradigm == "AR"]
    # AR annotation: text top-left, arrow down-right into cluster
    ax.annotate(
        "AR decode:\nmemory-bound\n($I \\ll$ 201 FLOP/byte)",
        xy=(float(d_ar.intensity.median()), float(d_ar.achieved.median())),
        xytext=(0.03, 0.53),
        textcoords="axes fraction",
        fontsize=6, color="#222222",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_AR,
                  lw=0.6, alpha=0.94),
        arrowprops=dict(arrowstyle="-|>", lw=0.65, color=C_AR,
                        shrinkA=2, shrinkB=4,
                        connectionstyle="arc3,rad=0.25"),
        ha="left", va="center", zorder=7,
    )

    d_df = df[df.paradigm != "AR"]
    best_achieved = float(d_df.achieved.max())
    best_pct      = 100 * best_achieved / PEAK_BF16_TF
    best_row      = d_df.loc[d_df.achieved.idxmax()]
    # Diffusion annotation: text just below the cluster, arrow up to best point
    ax.annotate(
        f"Diffusion: compute-bound\nbest {best_achieved:.0f} TF/s"
        f" ({best_pct:.0f}% of BF16 peak)",
        xy=(float(best_row.intensity), best_achieved),
        xytext=(870, 1.0),
        textcoords="data",
        fontsize=6, color="#222222",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_DISC,
                  lw=0.6, alpha=0.94),
        arrowprops=dict(arrowstyle="-|>", lw=0.45, color=C_DISC,
                        shrinkA=2, shrinkB=10,
                        connectionstyle="arc3,rad=0.18"),
        ha="center", va="top", zorder=7,
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.38, 1.2e4)
    ax.set_ylim(0.012, 700)

    _log_minor(ax)
    _light_grid(ax)

    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))

    ax.set_xlabel("Arithmetic intensity (FLOP/byte, per step)", labelpad=3)
    ax.set_ylabel("Achieved throughput (TF/s, per step)", labelpad=4)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    # Add a size-legend entry hinting at "marker ∝ tokens in flight"
    ax.legend(
        handles, labels,
        loc="lower right",
        fontsize=6.0,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.35,
        # title="Paradigm  (size $\\propto$ $B{\\times}G$)",
        # title_fontsize=5.5,
        frameon=True, framealpha=0.96, edgecolor="#bbbbbb",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fname = f"figB_roofline_{arch}"
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png", dpi=300)
    plt.close(fig)
    print(f"Saved  {out}/{fname}.{{pdf,png}}")


if __name__ == "__main__":
    main()
