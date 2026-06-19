#!/usr/bin/env python3
"""
benchmarks/plot_headline.py
===========================

Head-to-head headline figures (Fast-dLLM / LLaDA / serving-paper style):
one metric per figure, the three paradigms side by side, speedup
multipliers written on the data.  Outputs to results/paradigm_bench/:

  H1_throughput_{arch}    tok/s vs batch — 5 curves, endpoint values
  H2_speedup_{arch}       speedup vs AR (bars, ×N labels) at B ∈ {1,8,64,256}
  H3_latency_{arch}       e2e latency vs generation length (B=4)
  H4_scorecard_{arch}     4-panel bars at reference configs: tok/s, latency,
                          mJ/token, max concurrent batch

Usage: python benchmarks/plot_headline.py [--arch 512d12b ...]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"AR": "#1f77b4", "Discrete": "#d62728", "Continuous": "#2ca02c"}
LIGHT  = {"Discrete": "#f4a582", "Continuous": "#a6dba0"}
NICE   = {"AR": "Autoregressive", "Discrete": "Discrete Diff. (LLaDA)",
          "Continuous": "Continuous Diff. (ELF)"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf")


def _load(pattern: str, key: list[str]) -> pd.DataFrame | None:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        m = re.search(r"_(\d+d\d+b)", f)
        if "arch" not in d.columns:
            d["arch"] = m.group(1) if m else "512d12b"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "oom" not in df.columns:
        df["oom"] = False
    df["oom"] = df["oom"].astype(str).str.lower().eq("true")
    return df.drop_duplicates(subset=[k for k in key if k in df.columns],
                              keep="last").reset_index(drop=True)


def _kfmt(v: float) -> str:
    return f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


# ── H1: throughput vs batch ───────────────────────────────────────────────────

def h1_throughput(par: pd.DataFrame, out: Path, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return
    series = [("AR (greedy)",     COLORS["AR"],         "-",  "o", 2.2),
              ("Discrete S=32",   COLORS["Discrete"],   "-",  "s", 1.8),
              ("Discrete S=8",    LIGHT["Discrete"],    "--", "s", 1.5),
              ("Continuous S=32", COLORS["Continuous"], "-",  "^", 1.8),
              ("Continuous S=8",  LIGHT["Continuous"],  "--", "^", 1.5)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, c, ls, mk, lw in series:
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["batch_size"], d["tok_s_system"] / 1e3, color=c, ls=ls,
                marker=mk, ms=5, lw=lw, label=label)
        last = d.iloc[-1]
        ax.annotate(_kfmt(last["tok_s_system"]),
                    (last["batch_size"], last["tok_s_system"] / 1e3),
                    textcoords="offset points", xytext=(6, 0), fontsize=8,
                    color=c, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (concurrent sequences)")
    ax.set_ylabel("Generation throughput (k tok/s)")
    ax.set_title(f"Throughput head-to-head — {arch}, G=256, bf16, 1×A100")
    ax.legend(loc="upper left")
    _save(fig, out, f"H1_throughput_{arch}")


# ── H2: speedup vs AR (bars with ×N) ─────────────────────────────────────────

def h2_speedup(par: pd.DataFrame, out: Path, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return
    Bs = [b for b in (1, 8, 64, 256)
          if not d_all[(d_all.label == "AR (greedy)")
                       & (d_all.batch_size == b)].empty]
    bars = [("Discrete S=8",    LIGHT["Discrete"],    ""),
            ("Discrete S=32",   COLORS["Discrete"],   ""),
            ("Continuous S=8",  LIGHT["Continuous"],  ""),
            ("Continuous S=32", COLORS["Continuous"], "")]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = np.arange(len(Bs))
    width = 0.8 / len(bars)
    for i, (label, color, hatch) in enumerate(bars):
        vals = []
        for b in Bs:
            ar = d_all[(d_all.label == "AR (greedy)") & (d_all.batch_size == b)]
            o  = d_all[(d_all.label == label) & (d_all.batch_size == b)]
            vals.append(float(o["tok_s_system"].iloc[0] / ar["tok_s_system"].iloc[0])
                        if not ar.empty and not o.empty else np.nan)
        pos = xs + (i - (len(bars) - 1) / 2) * width
        rects = ax.bar(pos, vals, width, color=color, hatch=hatch,
                       edgecolor="black", linewidth=0.5, label=label)
        for r, v in zip(rects, vals):
            if v == v:
                ax.text(r.get_x() + r.get_width() / 2, v,
                        f"{v:.1f}×", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
    ax.axhline(1.0, color=COLORS["AR"], lw=1.4, ls="--")
    ax.text(len(Bs) - 0.42, 1.02, "AR baseline", color=COLORS["AR"], fontsize=8.5)
    ax.set_xticks(xs, [f"B={b}" for b in Bs])
    ax.set_ylabel("Throughput speedup over AR (×)")
    ax.set_title(f"Diffusion speedup over AR by serving regime — {arch} "
                 f"(G=256, bf16, 1×A100)")
    ax.legend(fontsize=8)
    _save(fig, out, f"H2_speedup_{arch}")


# ── H3: latency vs generation length ─────────────────────────────────────────

def h3_latency(grid: pd.DataFrame, out: Path, arch: str, B: int = 4) -> None:
    g = grid[(grid.arch == arch) & (~grid.oom) & (grid.batch_size == B)]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for p in ("AR", "Discrete", "Continuous"):
        d = g[g.paradigm == p].sort_values("gen_len")
        if d.empty:
            continue
        lbl = NICE[p] if p == "AR" else f"{NICE[p]} (S=32)"
        ax.plot(d["gen_len"], d["e2e_ms_med"], color=COLORS[p], marker="o",
                ms=5, lw=1.8, label=lbl)
        last = d.iloc[-1]
        ax.annotate(f"{last['e2e_ms_med']:.0f} ms",
                    (last["gen_len"], last["e2e_ms_med"]),
                    textcoords="offset points", xytext=(6, -2),
                    fontsize=8, color=COLORS[p], fontweight="bold")
    ax.set_xlabel("Tokens generated (G)")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"Latency vs generation length — {arch}, B={B}\n"
                 f"AR: one pass per token (linear in G) · "
                 f"diffusion: 32 passes total (flat-ish)")
    ax.set_xticks(sorted(g["gen_len"].unique()))
    ax.legend()
    _save(fig, out, f"H3_latency_{arch}")


# ── H4: scorecard ─────────────────────────────────────────────────────────────

def h4_scorecard(par: pd.DataFrame, ceil: pd.DataFrame | None, out: Path,
                 arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return
    ref = {"AR": "AR (greedy)", "Discrete": "Discrete S=32",
           "Continuous": "Continuous S=32"}

    def metric(label: str, b: int, col: str) -> float:
        r = d_all[(d_all.label == label) & (d_all.batch_size == b)]
        return float(r[col].iloc[0]) if not r.empty else np.nan

    panels = []
    panels.append(("Peak throughput\n(tok/s, best batch)", "{:,.0f}",
                   {p: d_all[d_all.label == lbl]["tok_s_system"].max()
                    for p, lbl in ref.items()}, False))
    panels.append(("Single-request latency\n(ms, B=1, G=256)", "{:,.0f}",
                   {p: metric(lbl, 1, "e2e_ms_med") for p, lbl in ref.items()},
                   True))
    panels.append(("Energy at saturation\n(mJ/token, B=64)", "{:,.1f}",
                   {p: metric(lbl, 64, "j_per_tok") * 1e3
                    for p, lbl in ref.items()}, True))
    if ceil is not None:
        c = ceil[ceil.arch == arch]
        panels.append(("Max concurrent batch\n(G=512, dedicated GPU)", "{:,.0f}",
                       {p: float(c[(c.paradigm == p) & (c.attn == "MHA")]["batch"].max())
                        if not c[(c.paradigm == p) & (c.attn == "MHA")].empty
                        else np.nan for p in ref}, False))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.2))
    for ax, (title, fmt, vals, lower_better) in zip(axes, panels):
        ps = list(vals)
        v = [vals[p] for p in ps]
        rects = ax.bar(range(len(ps)), v, color=[COLORS[p] for p in ps],
                       edgecolor="black", linewidth=0.5)
        for r, x in zip(rects, v):
            if x == x:
                ax.text(r.get_x() + r.get_width() / 2, x, fmt.format(x),
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")
        ax.set_xticks(range(len(ps)), ["AR", "Disc", "Cont"])
        ax.set_title(title + ("  ↓" if lower_better else "  ↑"), fontsize=9)
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Paradigm scorecard — {arch} (bf16, 1×A100; diffusion S=32)",
                 y=1.04, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, f"H4_scorecard_{arch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", default=None)
    parser.add_argument("--out", default="results/paradigm_bench")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    par  = _load("results/ablation_pareto_*.csv", ["arch", "label", "batch_size", "dtype"])
    grid = _load("results/ablation_grid_*.csv",
                 ["arch", "paradigm", "batch_size", "gen_len", "dtype", "tp"])
    ceil = _load("results/ablation_ceiling_*.csv", ["arch", "paradigm", "attn", "batch"])

    archs = args.arch or sorted(par.arch.unique() if par is not None else [])
    for arch in archs:
        if par is not None:
            h1_throughput(par, out, arch)
            h2_speedup(par, out, arch)
            h4_scorecard(par, ceil, out, arch)
        if grid is not None:
            h3_latency(grid, out, arch)
    print("Done.")


if __name__ == "__main__":
    main()
