#!/usr/bin/env python3
"""
Publication-quality figures for EMNLP 2026 System Demo.

Generates:
  results/paper_figs/fig1_architecture.{png,pdf}   — system overview
  results/paper_figs/fig2_comparison.{png,pdf}      — vs existing systems
  results/paper_figs/fig3_pareto.{png,pdf}          — serving frontier
  results/paper_figs/fig4_quality.{png,pdf}         — quality + memory
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── global style (ACL-compatible) ──────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# colour palette — consistent across all figures
C = {
    "AR":         "#2166ac",
    "Discrete":   "#d6604d",
    "Continuous": "#4dac26",
    "MHA":        "#555555",
    "GQA":        "#9970ab",
    "MLA":        "#e08214",
}
LIGHT = {  # lighter variants for S=8 curves / secondary bars
    "AR":         "#92c5de",
    "Discrete":   "#f4a582",
    "Continuous": "#a1d76a",
}

ROOT = Path(__file__).parent.parent
DATA = ROOT / "results"
OUT  = DATA / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved  {name}.{{png,pdf}}")


# ── low-level drawing helpers (used only by fig1 / fig2) ───────────────────
def _rbox(ax, x, y, w, h, fc, ec="#333333", lw=1.2, pad=0.06):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        clip_on=False, zorder=2)
    ax.add_patch(p)


def _t(ax, x, y, s, size=8, bold=False, color="#1a1a1a",
       ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight="bold" if bold else "normal",
            clip_on=False, zorder=3)


def _arr(ax, x1, y1, x2, y2, color="#555555", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=3)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1  —  Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig1_architecture() -> None:
    """
    Top-down flow:
      [Shared Backbone (3 attention variants inside)]
               ↓    ↓    ↓
      [AR]  [Masked Diff]  [Continuous ELF]   ← paradigm (3 choices)
               ↓    ↓    ↓
      [Unified CLI — train · eval · benchmark]
    """
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # ── title ─────────────────────────────────────────────────────────────
    _t(ax, 5.0, 5.0,
       "DantinoX — Unified Language Modelling Research Platform (JAX / Flax NNX)",
       size=9.5, bold=True)

    # ── shared backbone (blue) — spans full width ──────────────────────────
    # Outer frame: y ∈ [3.2, 4.7]
    _rbox(ax, 0.2, 3.2, 9.6, 1.5, "#cde8f5", ec="#1d6fa4", lw=2.0, pad=0.08)
    _t(ax, 5.0, 4.55, "Shared Transformer Backbone",
       size=9, bold=True, color="#0d4a6e")
    _t(ax, 5.0, 4.24,
       "Pre-LayerNorm  ·  RoPE positional encoding  ·  SwiGLU feed-forward",
       size=7.5, color="#0d4a6e")

    # three attention-variant sub-boxes inside backbone
    # each box width 2.85, gap 0.2; start x: 0.35
    attn_cfg = [
        (0.35,  "#e2e2e2", "#555555", "MHA",   "full KV-cache"),
        (3.40,  "#e8d6f5", "#9970ab", "GQA-¼", "4× KV reduction"),
        (6.45,  "#f5e6cc", "#e08214", "MLA",   "12× KV reduction"),
    ]
    arrow_xs = []
    for xi, fc, ec, aname, asub in attn_cfg:
        _rbox(ax, xi, 3.29, 2.85, 0.78, fc, ec=ec, lw=1.3, pad=0.04)
        _t(ax, xi + 1.425, 3.83, aname, size=8.5, bold=True, color=ec)
        _t(ax, xi + 1.425, 3.50, asub, size=7, color=ec)
        arrow_xs.append(xi + 1.425)

    # ── arrows — backbone bottom → paradigm boxes ──────────────────────────
    for ax_x in arrow_xs:
        _arr(ax, ax_x, 3.2, ax_x, 3.0, lw=1.1)

    # ── three paradigm boxes ───────────────────────────────────────────────
    par_cfg = [
        (0.35,  "#dceefb", "#2166ac",
         "Autoregressive (AR)",
         "Causal LM · greedy / sampling\nKV-cache enables fast batching"),
        (3.40,  "#fde8e4", "#d6604d",
         "Masked Diffusion (LLaDA)",
         "Absorbing-state diffusion · S steps\nParallel decoding — no KV-cache"),
        (6.45,  "#e6f4e6", "#4dac26",
         "Continuous Flow (ELF)",
         "T5-encoder conditioning\nFlow-matching denoising · S steps"),
    ]
    for xi, fc, ec, pname, psub in par_cfg:
        _rbox(ax, xi, 1.55, 2.85, 1.35, fc, ec=ec, lw=1.5, pad=0.06)
        _t(ax, xi + 1.425, 2.72, pname, size=8.5, bold=True, color=ec)
        _t(ax, xi + 1.425, 2.32, psub, size=7.5, color="#333333")

    # ── arrows — paradigm boxes bottom → CLI ──────────────────────────────
    for ax_x in arrow_xs:
        _arr(ax, ax_x, 1.55, ax_x, 1.35, lw=1.1)

    # ── unified CLI footer ─────────────────────────────────────────────────
    _rbox(ax, 0.2, 0.25, 9.6, 0.98, "#f0f0f0", ec="#777777", lw=1.5, pad=0.06)
    _t(ax, 5.0, 0.90, "Unified CLI", size=8.5, bold=True, color="#222222")
    _t(ax, 5.0, 0.57,
       "dantinoX train  ·  dantinoX eval  ·  dantinoX benchmark",
       size=7.5, color="#444444")

    fig.tight_layout(pad=0.2)
    _save(fig, "fig1_architecture")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2  —  Feature Comparison Matrix
# ══════════════════════════════════════════════════════════════════════════════
def fig2_comparison() -> None:
    """
    Checkmark matrix: DantinoX vs related systems across key capabilities.
    """
    features = [
        "Autoregressive LM",
        "Masked Diffusion LM",
        "Continuous Diffusion (ELF)",
        "GQA / MLA attention variants",
        "Unified train + eval API",
        "Cross-paradigm benchmarking",
        "Configurable diffusion steps (S)",
        "Open-source (Apache-2.0)",
    ]
    systems  = ["DantinoX\n(this work)", "nanoGPT", "MDLM", "LLaDA", "MDT"]

    # rows = features, cols = systems
    matrix = np.array([
        [1, 1, 0, 0, 0],   # AR
        [1, 0, 1, 1, 0],   # Masked diff
        [1, 0, 0, 0, 1],   # Continuous
        [1, 0, 0, 0, 0],   # GQA/MLA
        [1, 0, 0, 0, 0],   # Unified API
        [1, 0, 0, 0, 0],   # Cross-paradigm bench
        [1, 0, 1, 1, 1],   # Configurable S
        [1, 1, 1, 1, 1],   # Open source
    ], dtype=float)

    nf, ns = matrix.shape
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_xlim(-0.5, ns - 0.5)
    ax.set_ylim(-0.5, nf + 0.65)
    ax.axis("off")

    # Column headers only (no title in figure — goes in LaTeX caption)
    for j, sys_name in enumerate(systems):
        color = "#2166ac" if j == 0 else "#333333"
        bold  = (j == 0)
        ax.text(j, nf + 0.55, sys_name, ha="center", va="top",
                fontsize=8, fontweight="bold" if bold else "normal", color=color)

    # Thin horizontal rule below headers
    ax.axhline(nf - 0.5, color="#888888", lw=0.8, xmin=0.0, xmax=1.0)

    # Row labels (right-aligned, leftmost)
    for i, feat in enumerate(features):
        ax.text(-0.65, nf - 1 - i, feat, ha="right", va="center",
                fontsize=7.5, color="#222222")

    # Grid cells — use scatter markers instead of Unicode glyphs
    for i in range(nf):
        for j in range(ns):
            ri  = nf - 1 - i   # row index (bottom=0)
            val = matrix[i, j]
            bg  = "#e8f4d9" if val == 1 else "#fce8e4"
            if j == 0 and val == 1:
                bg = "#c6e6f5"
            rect = plt.Rectangle((j - 0.48, ri - 0.45), 0.96, 0.90,
                                  facecolor=bg, edgecolor="#bbbbbb",
                                  linewidth=0.5, zorder=1)
            ax.add_patch(rect)
            if val == 1:
                clr    = "#0d5a8e" if j == 0 else "#1a6e1a"
                marker = "o"
                ms     = 9
                mfc    = clr
                mec    = clr
            else:
                clr, marker, ms, mfc, mec = "#b22222", "x", 8, "none", "#b22222"
            ax.plot(j, ri, marker, ms=ms, mfc=mfc, mec=mec, mew=1.8,
                    zorder=2, color=clr)

    # Vertical separator after DantinoX column
    ax.axvline(0.5, color="#aaaaaa", lw=0.8, ls="--", zorder=0)

    # Legend note at bottom
    ax.text((ns-1)/2, -0.48, "filled circle = supported   x = not supported",
            ha="center", va="bottom", fontsize=7, color="#555555")

    fig.tight_layout(pad=0.3)
    _save(fig, "fig2_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3  —  Serving Pareto Frontier  (latency × throughput)
# ══════════════════════════════════════════════════════════════════════════════
PARETO_STYLE = {
    "AR (greedy)":     dict(color=C["AR"],            ls="-",  lw=2.0, marker="o", ms=5),
    "Discrete S=8":    dict(color=LIGHT["Discrete"],   ls="--", lw=1.6, marker="s", ms=4),
    "Discrete S=32":   dict(color=C["Discrete"],       ls="-",  lw=1.6, marker="s", ms=4),
    "Continuous S=8":  dict(color=LIGHT["Continuous"], ls="--", lw=1.6, marker="^", ms=4),
    "Continuous S=32": dict(color=C["Continuous"],     ls="-",  lw=1.6, marker="^", ms=4),
}


def fig3_pareto() -> None:
    """Latency–throughput Pareto; each point is a batch size B."""
    csv = DATA / "ablation_pareto_512d12b.csv"
    if not csv.exists():
        print(f"  SKIP fig3_pareto — {csv} not found")
        return

    par = pd.read_csv(csv)
    d_all = par[~par["oom"] & (par["dtype"] == "bf16")].copy()

    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    for label, st in PARETO_STYLE.items():
        d = d_all[d_all["label"] == label].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["e2e_ms_med"], d["tok_s_system"], label=label,
                zorder=3, **st)
        # B=1 label: only on Discrete S=8 (leftmost point, tells the latency story)
        if label == "Discrete S=8":
            b1 = d[d["batch_size"] == 1]
            if not b1.empty:
                ax.annotate("B=1",
                            (float(b1["e2e_ms_med"].iloc[0]),
                             float(b1["tok_s_system"].iloc[0])),
                            textcoords="offset points", xytext=(-5, 8),
                            fontsize=6.5, color=st["color"], ha="right")

    # — headline annotation: 6.4× lower latency at B=1 ——————————————
    ar_b1  = d_all[(d_all["label"] == "AR (greedy)")   & (d_all["batch_size"] == 1)]
    ds8_b1 = d_all[(d_all["label"] == "Discrete S=8")  & (d_all["batch_size"] == 1)]
    if not ar_b1.empty and not ds8_b1.empty:
        x_ar  = float(ar_b1["e2e_ms_med"].iloc[0])
        y_d8  = float(ds8_b1["tok_s_system"].iloc[0])
        x_d8  = float(ds8_b1["e2e_ms_med"].iloc[0])
        ratio = x_ar / x_d8
        ax.annotate(
            f"{ratio:.1f}x lower latency\n(Disc. S=8 vs AR at B=1)",
            xy=(x_d8, y_d8), xytext=(280, 18000),
            fontsize=7.5, fontweight="bold", color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0,
                            connectionstyle="arc3,rad=-0.25"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#999999", alpha=0.95),
        )

    # SLO reference line
    ax.axvline(200, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.text(200, 1100, " 200 ms\n SLO", fontsize=6.5, color="#999999",
            va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Per-request latency (ms, G=256 tokens, A100 bf16)")
    ax.set_ylabel("System throughput (tok/s)")
    ax.set_title("Serving frontier — 512d12b, bf16, 1×A100", fontsize=8.5)

    legend = ax.legend(loc="lower right", fontsize=7,
                       framealpha=0.9, edgecolor="#cccccc")
    ax.set_xlim(20, 2500)
    ax.set_ylim(800, 80000)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    fig.tight_layout(pad=0.3)
    _save(fig, "fig3_pareto")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4  —  Quality & Memory Efficiency
# ══════════════════════════════════════════════════════════════════════════════
def fig4_quality() -> None:
    """
    Left panel: bpb (wikitext-103) for Masked-Diffusion MHA/GQA/MLA at two sizes.
    Right panel: KV-cache footprint (MB/seq) for AR at three attention variants.

    Key messages:
      • GQA and MLA match MHA quality (iso-quality) while halving/decimating the cache.
      • MLA delivers 12× smaller KV-cache with zero quality penalty.
    """
    ppl = pd.read_csv(DATA / "perplexity.csv")
    diff_wiki = ppl[
        (ppl["model_type"] == "diffusion") &
        (ppl["dataset"] == "wikitext-103")
    ].copy()

    # ── left panel data ────────────────────────────────────────────────────
    # group by dim × attn_variant
    dims   = [512, 768]
    attns  = ["MHA", "GQA", "MLA"]
    attn_colors = [C["MHA"], C["GQA"], C["MLA"]]
    attn_hatch  = ["", "//", ".."]

    # pivot: bpb indexed by (dim, attn_variant)
    bpb_map: dict[tuple, float] = {}
    for _, row in diff_wiki.iterrows():
        key = (int(row["dim"]), row["attn_variant"])
        bpb_map[key] = float(row["bpb"])

    # ── right panel data ───────────────────────────────────────────────────
    # KV-cache for AR xl, B=4, G=128 — from benchmarks
    kv_cache = {
        "MHA": 50.594,   # MB
        "GQA": 12.648,
        "MLA":  3.953,
    }

    # ── figure layout ──────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                                      gridspec_kw={"wspace": 0.38})

    # ── LEFT: bpb grouped bar ─────────────────────────────────────────────
    n_dims   = len(dims)
    n_attns  = len(attns)
    bar_w    = 0.22
    group_gap = 0.25
    x_centers = np.arange(n_dims) * (n_attns * bar_w + group_gap)

    for ai, (aname, acolor, ahatch) in enumerate(
            zip(attns, attn_colors, attn_hatch)):
        xs   = x_centers + (ai - 1) * bar_w
        vals = [bpb_map.get((d, aname), np.nan) for d in dims]
        bars = ax_l.bar(xs, vals, width=bar_w, color=acolor, hatch=ahatch,
                        edgecolor="white", linewidth=0.5,
                        label=aname, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax_l.text(bar.get_x() + bar.get_width() / 2,
                           v + 0.0015, f"{v:.4f}",
                           ha="center", va="bottom", fontsize=5.5,
                           color="#333333")

    ax_l.set_xticks(x_centers)
    ax_l.set_xticklabels(["512d  (70M)" , "768d  (180M)"], fontsize=7.5)
    ax_l.set_ylabel("Bits-per-byte (bpb) ↓", fontsize=8)
    ax_l.set_title("Language modelling quality\n(Masked Diffusion, WikiText-103)",
                    fontsize=8)
    ax_l.set_ylim(0.36, 0.425)
    ax_l.legend(title="Attention", fontsize=7, title_fontsize=7,
                framealpha=0.8, edgecolor="#cccccc")
    ax_l.spines["left"].set_visible(True)
    ax_l.spines["bottom"].set_visible(True)
    # add horizontal reference lines
    ax_l.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax_l.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    # ── RIGHT: KV-cache bar ───────────────────────────────────────────────
    xs_r = np.arange(n_attns)
    vals_r = [kv_cache[a] for a in attns]
    bars_r = ax_r.bar(xs_r, vals_r,
                       color=attn_colors, edgecolor="white", linewidth=0.5,
                       hatch=attn_hatch, zorder=3)

    # annotate reduction factor vs MHA
    ref = vals_r[0]
    for bi, (bar, v, aname) in enumerate(zip(bars_r, vals_r, attns)):
        bx = bar.get_x() + bar.get_width() / 2
        # MB label inside bar near top
        ax_r.text(bx, v - 2.0, f"{v:.1f} MB",
                  ha="center", va="top", fontsize=7.5,
                  fontweight="bold", color="white")
        if bi > 0:
            fold = ref / v
            # annotation below the bar with an upward arrow
            ax_r.annotate(f"{fold:.0f}x\nsmaller",
                          xy=(bx, v), xytext=(bx, v + 6),
                          ha="center", va="bottom", fontsize=7.5,
                          fontweight="bold", color=attn_colors[bi],
                          arrowprops=dict(arrowstyle="-|>",
                                         color=attn_colors[bi], lw=1.0))

    ax_r.set_xticks(xs_r)
    ax_r.set_xticklabels(attns, fontsize=8)
    ax_r.set_ylabel("KV-cache footprint (MB / sequence)", fontsize=8)
    ax_r.set_title("KV-cache memory\n(AR xl ~70M params, B=4, G=128)",
                    fontsize=8)
    ax_r.spines["left"].set_visible(True)
    ax_r.spines["bottom"].set_visible(True)
    ax_r.set_ylim(0, 70)   # headroom for annotations above GQA and MLA bars

    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 1])
    _save(fig, "fig4_quality")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.chdir(ROOT)
    print("Generating EMNLP paper figures →", OUT)
    fig1_architecture()
    fig2_comparison()
    fig3_pareto()
    fig4_quality()
    print("Done.")
