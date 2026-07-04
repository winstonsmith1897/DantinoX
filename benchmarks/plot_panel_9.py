#!/usr/bin/env python3
"""
benchmarks/plot_panel_9.py
==========================
EMNLP System Demonstration — single-column 3-panel figure.

  Panel (a)  Serving Pareto frontier  — throughput vs latency, log–log
  Panel (b)  Energy per generated tok — mJ/tok vs batch size, log–log
  Panel (c)  Throughput head-to-head  — k tok/s vs batch size, linear–y

  Architecture: Large (1024-d · 16-layer)

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
import matplotlib.transforms as mtransforms
import pandas as pd

# ── Palette ────────────────────────────────────────────────────────────────────
# Hue encodes paradigm; solid/dashed + saturation encodes step count.
# Chosen for readability at single-column size and colorblind safety.

BLUE   = "#1f78b4"   # AR
RED    = "#d62728"   # Discrete  S=32  (deep red)
ORANGE = "#fc8d59"   # Discrete  S=8   (warm orange — clearly ≠ red)
GREEN  = "#2ca02c"   # Continuous S=32 (deep green)
TEAL   = "#17becf"   # Continuous S=8  (teal — clearly ≠ green)

# (display_label, csv_label, color, ls, marker, lw, ms)
SERIES = [
    ("AR",                  "AR (greedy)",     BLUE,   "-",  "o", 1.9, 4.2),
    ("Disc. diff., $S=32$", "Discrete S=32",   RED,    "-",  "s", 1.5, 3.6),
    ("Disc. diff., $S=8$",  "Discrete S=8",    ORANGE, "--", "s", 1.3, 3.4),
    ("Cont. diff., $S=32$", "Continuous S=32", GREEN,  "-",  "^", 1.5, 3.6),
    ("Cont. diff., $S=8$",  "Continuous S=8",  TEAL,   "--", "^", 1.3, 3.4),
]

# Legend order: group by paradigm across 3 columns
#   Row 1:  AR          | Disc. S=32 | Cont. S=32
#   Row 2:  (spacer)    | Disc. S=8  | Cont. S=8
LEGEND_ORDER = [0, 1, 3, 2, 4]   # indices into SERIES

ARCHS = ["1024d16b"]

# ── Global rc ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         7,
    "axes.titlesize":    8.0,
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
    "axes.grid":         True,
    "grid.alpha":        0.38,
    "grid.color":        "#cccccc",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.45,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.axisbelow":    True,
    "legend.frameon":    True,
    "legend.framealpha": 0.96,
    "legend.edgecolor":  "#bbbbbb",
    "legend.borderpad":  0.45,
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
        d["arch"] = d["arch"].fillna(arch) if "arch" in d.columns else arch
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


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _log_minor_ticks(ax: plt.Axes, which: str = "both") -> None:
    nf = mticker.NullFormatter()
    if which in ("x", "both"):
        ax.xaxis.set_minor_locator(mticker.LogLocator(subs="auto"))
        ax.xaxis.set_minor_formatter(nf)
    if which in ("y", "both"):
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs="auto"))
        ax.yaxis.set_minor_formatter(nf)


def _x_batch_ticks(ax: plt.Axes, data: pd.DataFrame) -> None:
    bs = sorted(data["batch_size"].dropna().unique().astype(int))
    ax.set_xticks(bs)
    ax.set_xticklabels(
        [str(b) if b in (1, 4, 16, 64, 256) else "" for b in bs],
        fontsize=6.5,
    )


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.88, 0.472, text,
        transform=ax.transAxes,
        fontsize=7.5, fontweight="bold", va="top", ha="left", color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        zorder=5,
    )


# ── Panel (a): Serving Pareto frontier ────────────────────────────────────────

def _draw_pareto(ax: plt.Axes, par: pd.DataFrame, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return

    for disp, csv, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == csv].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["e2e_ms_med"], d["tok_s_system"],
                color=color, ls=ls, marker=mk, ms=ms, lw=lw,
                label=disp, zorder=3, clip_on=True)

    # Crossover annotation — AR vs Discrete S=32
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
                    f"AR $\\geq$ Disc. ($S\\!=\\!32$)\nat $B={int(bx)}$",
                    xy=(x, y),
                    xytext=(x * 2.4, y * 0.38),
                    fontsize=5.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#aaaaaa", lw=0.5, alpha=0.93),
                    arrowprops=dict(
                        arrowstyle="-|>", lw=0.7, color="#777777",
                        shrinkA=2, shrinkB=3,
                        connectionstyle="arc3,rad=0.18",
                    ),
                    ha="left", va="top", zorder=6,
                )

    # SLO guideline — drawn after data so xlim is not perturbed
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(labelOnlyBase=True))
    _log_minor_ticks(ax, "both")
    ax.autoscale(enable=True, axis="x", tight=False)

    ax.axvline(200, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=2)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(178, 0.04, "200 ms SLO", transform=trans,
            fontsize=5.5, color="#777777", ha="right", va="bottom",
            style="italic")

    ax.set_xlabel("Per-request latency (ms)", labelpad=3)
    ax.set_ylabel("System throughput (tok/s)", labelpad=4)


# ── Panel (b): Energy per token ───────────────────────────────────────────────

def _draw_energy(ax: plt.Axes, par: pd.DataFrame, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    d_all = d_all[d_all["j_per_tok"].notna() & (d_all["j_per_tok"] > 0)]
    if d_all.empty:
        return

    for _, csv, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == csv].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["batch_size"], d["j_per_tok"] * 1e3,
                color=color, ls=ls, marker=mk, ms=ms, lw=lw)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _x_batch_ticks(ax, d_all)
    _log_minor_ticks(ax, "y")
    ax.set_xlabel("Batch size", labelpad=3)
    ax.set_ylabel("Energy / token (mJ)", labelpad=4)


# ── Panel (c): System throughput ──────────────────────────────────────────────

def _draw_throughput(ax: plt.Axes, par: pd.DataFrame, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return

    for _, csv, color, ls, mk, lw, ms in SERIES:
        d = d_all[d_all.label == csv].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["batch_size"], d["tok_s_system"] / 1e3,
                color=color, ls=ls, marker=mk, ms=ms, lw=lw)

    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    _x_batch_ticks(ax, d_all)
    ax.set_xlabel("Batch size", labelpad=3)
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

    # ── Figure geometry — single column EMNLP (3.33") ────────────────────────
    HEIGHT_RATIOS = [1.28, 1.00, 1.00]
    FIG_H = sum(r * 1.35 for r in HEIGHT_RATIOS) + 0.15

    fig = plt.figure(figsize=(3.33, FIG_H))
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=HEIGHT_RATIOS,
        hspace=0.35,
        left=0.15, right=0.99,
        top=0.978,  bottom=0.162,
    )

    arch       = ARCHS[0]
    draw_fns   = [_draw_pareto, _draw_energy, _draw_throughput]
    panel_lbls = ["(a)", "(b)", "(c)"]

    axs = [fig.add_subplot(gs[row, 0]) for row in range(3)]
    for ax, fn, lbl in zip(axs, draw_fns, panel_lbls):
        fn(ax, par, arch)
        _panel_label(ax, lbl)

    # ── Legend — paradigm-grouped, 3 columns ──────────────────────────────────
    ordered = [SERIES[i] for i in LEGEND_ORDER]
    handles = [
        matplotlib.lines.Line2D(
            [], [], color=c, ls=ls, marker=mk, ms=4.5, lw=lw, label=disp,
        )
        for disp, _, c, ls, mk, lw, ms in ordered
    ]
    fig.legend(
        handles, [s[0] for s in ordered],
        loc="lower center",
        bbox_to_anchor=(0.565, 0.016),
        ncol=3,
        fontsize=5.8,
        handlelength=1.8, handletextpad=0.35, columnspacing=0.7,
        borderpad=0.5,
        frameon=True, framealpha=0.97, edgecolor="#bbbbbb",
    )

    fig.savefig(out / "fig_panel_9.pdf")
    fig.savefig(out / "fig_panel_9.png", dpi=300)
    plt.close(fig)
    print(f"Saved  {out}/fig_panel_9.{{pdf,png}}")


if __name__ == "__main__":
    main()
