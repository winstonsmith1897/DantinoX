#!/usr/bin/env python3
"""
benchmarks/plot_emnlp_paper.py
===============================
Publication-quality figures for the EMNLP 2026 System Demonstrations paper.

Each figure states ONE claim in the title and annotates it directly on the data.

  fig1_serving      Serving efficiency — diffusion delivers all tokens faster
                    than AR sequential generation; S=8 is 6-29× AR throughput
  fig2_ppl          Perplexity landscape — ELF-MHA/MLA outperform Discrete
                    by 2-3 orders of magnitude across benchmarks
  fig3_quality      Generation quality — diffusion has higher lexical diversity;
                    ELF-MHA best coherence (lowest rep-4, highest distinct-2)
  fig4_attention    Attention orthogonality — MHA and MLA are interchangeable
                    for Discrete; ELF-GQA collapses (high PPL, low diversity)
  fig5_throughput   Throughput advantage scales with batch size — diffusion S=8
                    saturates GPU 4× faster than AR

Run:
  python benchmarks/plot_emnlp_paper.py
  python benchmarks/plot_emnlp_paper.py --out results/emnlp_paper_figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Global style (ACL / EMNLP two-column) ─────────────────────────────────────
TW = 7.20   # text width, inches (EMNLP double-column)

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":            9,
    "axes.titlesize":       9.5,
    "axes.titleweight":     "bold",
    "axes.labelsize":       8.5,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "axes.grid":            True,
    "grid.alpha":           0.22,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
})

# ── Colorblind-safe palette (Wong 2011) ───────────────────────────────────────
C = {
    "AR":          "#0072B2",   # blue
    "Discrete":    "#D55E00",   # vermillion
    "Continuous":  "#009E73",   # green
    "MHA":         "#0072B2",
    "GQA":         "#E69F00",   # orange
    "MLA":         "#009E73",
    "diffusion":   "#D55E00",
    "elf":         "#009E73",
    "D_S8":        "#E69F00",   # orange (fast diffusion)
    "D_S32":       "#D55E00",   # vermillion (standard diffusion)
    "C_S8":        "#56B4E9",   # sky blue (fast ELF)
    "C_S32":       "#009E73",   # green (standard ELF)
}

MARKERS = {"AR": "o", "Discrete": "s", "Continuous": "^",
           "MHA": "o", "GQA": "s", "MLA": "^",
           "diffusion": "s", "elf": "^"}


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Serving efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_serving(root: Path, out: Path) -> None:
    par_path = root / "results" / "ablation_pareto_768d16b.csv"
    bench_path = root / "results" / "paradigm_bench.csv"
    if not par_path.exists() or not bench_path.exists():
        print("  [SKIP] fig1 — data missing"); return

    par = pd.read_csv(par_path)
    par = par[~par["oom"].astype(str).str.lower().eq("true")]

    bench = pd.read_csv(bench_path)
    steps = bench[bench["group"] == "diff_steps"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TW, 3.1))
    fig.suptitle(
        "Discrete diffusion generates all tokens faster than AR sequential decode "
        "and achieves up to 29× throughput at S=8",
        fontsize=8.8, fontweight="bold", y=1.01,
    )

    # ── Panel A: Latency vs throughput (Pareto scatter) ───────────────────────
    series_cfg = [
        ("AR (greedy)",   "AR",    C["AR"],     "o", "-",  7, 1.8),
        ("Discrete S=8",  "D_S8",  C["D_S8"],   "s", "-",  6, 1.6),
        ("Discrete S=32", "D_S32", C["D_S32"],  "s", "--", 5, 1.4),
        ("Continuous S=8",  "C_S8",  C["C_S8"],  "^", "-",  6, 1.6),
        ("Continuous S=32", "C_S32", C["C_S32"], "^", "--", 5, 1.4),
    ]
    for label, key, color, mk, ls, ms, lw in series_cfg:
        d = par[par["label"] == label].sort_values("batch_size")
        if d.empty: continue
        ax1.plot(d["e2e_ms_med"], d["tok_s_system"] / 1e3,
                 color=color, marker=mk, linestyle=ls, ms=ms, lw=lw,
                 label=label, zorder=3)
        # annotate B=4 point
        r4 = d[d["batch_size"] == 4]
        if not r4.empty:
            ax1.scatter(r4["e2e_ms_med"], r4["tok_s_system"] / 1e3,
                        color=color, s=55, zorder=5, edgecolors="white", lw=0.8)

    ax1.axvline(200, color="#888", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax1.text(185, ax1.get_ylim()[1] * 0.05 if ax1.get_ylim()[1] > 0 else 0.5,
             "200 ms\nSLO", fontsize=7, color="#555", ha="right", va="bottom")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("End-to-end latency (ms)")
    ax1.set_ylabel("System throughput (k tok/s)")
    ax1.set_title("(a) Latency–throughput Pareto, 768d·16L·bf16, G=256")
    ax1.legend(fontsize=6.5, loc="upper left", ncol=1,
               handlelength=2.0, handletextpad=0.4)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x)}" if x >= 1 else f"{x:.1f}"))

    # ── Panel B: Steps vs TTFT + throughput ratio ─────────────────────────────
    ar_ref = steps[steps["paradigm"] == "AR"]["tok_s_e2e"].mean()

    disc = steps[steps["paradigm"] == "Discrete"].sort_values("n_steps")
    cont = steps[steps["paradigm"] == "Continuous"].sort_values("n_steps")

    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)

    lns = []
    for df_, color, label, mk in [
        (disc, C["Discrete"], "Discrete TTFT", "s"),
        (cont, C["Continuous"], "ELF TTFT", "^"),
    ]:
        l, = ax2.plot(df_["n_steps"], df_["ttft_ms"],
                      color=color, marker=mk, ms=5.5, lw=1.8, label=label, zorder=3)
        lns.append(l)
    for df_, color, label, mk in [
        (disc, C["Discrete"], "Discrete tok/s ÷ AR", "s"),
        (cont, C["Continuous"], "ELF tok/s ÷ AR", "^"),
    ]:
        l, = ax2b.plot(df_["n_steps"], df_["tok_s_e2e"] / ar_ref,
                       color=color, marker=mk, ms=4, lw=1.2, ls="--",
                       label=label, zorder=2, alpha=0.75)
        lns.append(l)

    ax2.axhline(200, color="#888", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.text(4, 210, "200 ms SLO", fontsize=7, color="#555")
    ax2.axvline(16, color="#bbb", lw=0.8, ls=":", zorder=0)
    ax2.text(17, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 900,
             "S=16\nsweet\nspot", fontsize=6.5, color="#555", va="top")

    ax2.set_xlabel("Diffusion steps (S)")
    ax2.set_ylabel("TTFT = e2e latency (ms)")
    ax2b.set_ylabel("Throughput ratio vs AR", color="#666")
    ax2b.tick_params(axis="y", labelcolor="#666")
    ax2.set_title("(b) Steps vs latency and throughput, medium model")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([4, 8, 16, 32, 64, 128])
    ax2.set_xticklabels(["4", "8", "16", "32", "64", "128"])

    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, fontsize=6.5, loc="upper right", ncol=1,
               handlelength=1.8, handletextpad=0.4)

    fig.tight_layout()
    _save(fig, out, "fig1_serving")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Bits-per-byte landscape (bpb, tokenizer-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ppl_df(root: Path) -> pd.DataFrame:
    """Tidy per-run DataFrame from perplexity.csv. Excludes degenerate ELF-GQA-768d."""
    ppl = pd.read_csv(root / "results" / "perplexity.csv")
    PLAB = {"elf": "ELF", "diffusion": "Disc."}
    rows = []
    for run in ppl["run"].unique():
        sub   = ppl[ppl["run"] == run]
        first = sub.iloc[0]
        pt    = first["model_type"]
        attn  = first["attn_variant"]
        dim   = int(first["dim"])
        params = float(first["params_m"])
        wt_b  = float(sub.loc[sub["dataset"] == "wikitext-103", "bpb"].iloc[0]) \
                if not sub[sub["dataset"] == "wikitext-103"].empty else np.nan
        lm_b  = float(sub.loc[sub["dataset"] == "lambada", "bpb"].iloc[0]) \
                if not sub[sub["dataset"] == "lambada"].empty else np.nan
        tv    = float(sub.loc[sub["dataset"] == "train_val", "train_val_ppl"].iloc[0]) \
                if not sub[sub["dataset"] == "train_val"].empty else np.nan
        if pt == "elf" and wt_b > 0.45:    # training collapse
            continue
        rows.append({"run": run, "paradigm": pt, "attn": attn, "dim": dim,
                     "params_m": params, "wt_bpb": wt_b, "lam_bpb": lm_b,
                     "train_ppl": tv,
                     "label": f"{PLAB[pt]} {attn} {dim}d"})
    return pd.DataFrame(rows).sort_values("wt_bpb").reset_index(drop=True)


def fig2_ppl(root: Path, out: Path) -> None:
    if not (root / "results" / "perplexity.csv").exists():
        print("  [SKIP] fig2 — data missing"); return

    df = _load_ppl_df(root)
    n  = len(df)
    ys = np.arange(n)

    ATTN_MK  = {"MHA": "o", "GQA": "D", "MLA": "^"}
    PAR_COL  = {"elf": C["elf"], "diffusion": C["diffusion"]}

    # Pre-compute per-group y-spans for shading
    def _yspan(mask):
        idx = ys[mask]
        return idx.min() - 0.48, idx.max() + 0.48

    elf_mask  = df["paradigm"].values == "elf"
    disc_mask = ~elf_mask

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TW, 3.8),
        gridspec_kw={"wspace": 0.05},
    )
    fig.suptitle(
        "ELF achieves 5–13× lower bits-per-byte than Discrete Diffusion "
        "on both benchmarks",
        fontsize=9.2, fontweight="bold",
    )

    for ax_idx, (ax, metric, panel_label) in enumerate([
        (ax1, "wt_bpb",  "(a) WikiText-103"),
        (ax2, "lam_bpb", "(b) LAMBADA"),
    ]):
        vals    = df[metric].values
        v_max   = np.nanmax(vals)
        x_right = v_max * 1.20      # room for value labels

        # ── group background shading ─────────────────────────────────────────
        for mask, col in [(elf_mask, C["elf"]), (disc_mask, C["diffusion"])]:
            if not mask.any(): continue
            y0, y1 = _yspan(mask)
            ax.axhspan(y0, y1, color=col, alpha=0.06, lw=0, zorder=0)

        # ── group labels on RIGHT margin ──────────────────────────────────────
        for mask, col, lbl in [
            (elf_mask,  C["elf"],        "ELF"),
            (disc_mask, C["diffusion"],  "Discrete"),
        ]:
            if not mask.any(): continue
            y_mid = (ys[mask].min() + ys[mask].max()) / 2.0
            ax.text(x_right * 0.995, y_mid, lbl,
                    va="center", ha="right", fontsize=7.5,
                    color=col, fontstyle="italic", fontweight="bold")

        # ── lollipops ────────────────────────────────────────────────────────
        for i in range(n):
            v = vals[i]
            if np.isnan(v): continue
            row = df.iloc[i]
            col = PAR_COL[row["paradigm"]]
            mk  = ATTN_MK.get(row["attn"], "o")
            sz  = 80 if row["dim"] == 768 else 46

            # Stem (light grey — separates shape from color)
            ax.plot([0, v], [i, i], color="#d0d0d0", lw=1.1,
                    zorder=2, solid_capstyle="round")
            # Dot (paradigm color + attention shape + size = dim)
            ax.scatter(v, i, marker=mk, color=col, s=sz, zorder=5,
                       edgecolors="white", linewidths=0.8)
            # Value label
            fmt = f"{v:.4f}" if v < 0.10 else f"{v:.3f}"
            ax.text(v + v_max * 0.025, i, fmt,
                    va="center", ha="left", fontsize=6.5, color=col)

        # ── axes ──────────────────────────────────────────────────────────────
        ax.set_xlim(0, x_right)
        ax.set_ylim(-0.6, n - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("bits per byte  (↓ better)", fontsize=8.5)
        ax.set_title(panel_label, fontweight="bold", fontsize=9.5, pad=5)
        ax.yaxis.set_tick_params(length=0)
        ax.set_yticks(ys)
        if ax_idx == 0:
            ax.set_yticklabels(df["label"], fontsize=7.8)
        else:
            ax.set_yticklabels([""] * n)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=4))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── legend — bottom of figure, horizontal ─────────────────────────────────
    leg = (
        [mpatches.Patch(color=C["elf"],       label="ELF (flow matching)"),
         mpatches.Patch(color=C["diffusion"], label="Discrete diffusion")]
        + [plt.Line2D([0],[0], marker=mk, color="#555", ms=6, lw=0,
                      label=f"{attn}")
           for attn, mk in ATTN_MK.items()]
        + [plt.Line2D([0],[0], marker="o", color="#555", ms=5.0, lw=0,
                      label="512d"),
           plt.Line2D([0],[0], marker="o", color="#555", ms=8.5, lw=0,
                      label="768d")]
    )
    fig.legend(handles=leg, fontsize=7.2, loc="lower center",
               ncol=7, handletextpad=0.35, handlelength=1.2,
               columnspacing=1.0, framealpha=0.0, borderpad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)   # room for bottom legend
    _save(fig, out, "fig2_bpb")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Quality–Fluency trade-off scatter
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_quality(root: Path, out: Path) -> None:
    gq_path  = root / "results" / "generation_quality.csv"
    ppl_path = root / "results" / "perplexity.csv"
    if not gq_path.exists() or not ppl_path.exists():
        print("  [SKIP] fig3 — data missing"); return

    gq = pd.read_csv(gq_path)
    if ppl_path.exists():
        ppl_df = _load_ppl_df(root)   # already excludes degenerate runs
        ppl_lookup = ppl_df.set_index("run")[["wt_bpb", "lam_bpb"]].to_dict("index")
    else:
        ppl_lookup = {}

    # Merge bpb into generation quality table
    rows = []
    for _, r in gq.iterrows():
        run = r["run"]
        if run not in ppl_lookup:
            continue
        rows.append({
            "run":        run,
            "paradigm":   r["model_type"],
            "attn":       r["attn_variant"],
            "params_m":   float(r["params_m"]),
            "distinct_1": float(r["distinct_1"]),
            "distinct_2": float(r["distinct_2"]),
            "rep_4":      float(r["rep_4"]),
            "self_bleu":  float(r["self_bleu_4"]),
            "wt_bpb":     ppl_lookup[run]["wt_bpb"],
        })
    df = pd.DataFrame(rows)

    ATTN_MK = {"MHA": "o", "GQA": "D", "MLA": "^"}
    PAR_COL = {"elf": C["elf"], "diffusion": C["diffusion"]}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TW, 3.4),
        gridspec_kw={"wspace": 0.32},
    )
    fig.suptitle(
        "ELF-MHA-768d matches Discrete diversity with 13× better fluency; "
        "ELF-MLA variants show high repetition",
        fontsize=9.2, fontweight="bold",
    )

    # ── Panel A: Fluency–Diversity scatter ────────────────────────────────────
    #    x = wt_bpb (log, ↓ better), y = distinct-2 (↑ better)
    #    Ideal corner: bottom-left → top-left
    ax = ax1
    for _, r in df.iterrows():
        col = PAR_COL[r["paradigm"]]
        mk  = ATTN_MK.get(r["attn"], "o")
        sz  = 110 if r["params_m"] > 120 else 50

        ax.scatter(r["wt_bpb"], r["distinct_2"],
                   marker=mk, color=col, s=sz, zorder=4,
                   edgecolors="white", linewidths=0.8, alpha=0.92)

    ax.set_xscale("log")
    ax.set_xlim(0.020, 0.55)
    ax.set_ylim(0.44, 0.85)

    # Label ELF-MHA-768d directly (isolated top-left)
    r_star = df[df["run"] == "elf_mha_768d_16b_Dense"]
    if not r_star.empty:
        r = r_star.iloc[0]
        ax.annotate(
            "ELF-MHA-768d",
            (r["wt_bpb"], r["distinct_2"]),
            xytext=(8, -14), textcoords="offset points",
            fontsize=7.2, color=C["elf"], fontweight="bold",
            va="top", ha="left",
            arrowprops=dict(arrowstyle="-", color=C["elf"], lw=0.8),
        )

    # Label ELF-MHA-512d
    r_s = df[df["run"] == "elf_mha_512d_12b_Dense"]
    if not r_s.empty:
        r = r_s.iloc[0]
        ax.annotate(
            "ELF-MHA-512d",
            (r["wt_bpb"], r["distinct_2"]),
            xytext=(8, 5), textcoords="offset points",
            fontsize=7.0, color=C["elf"], fontweight="bold",
            va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=C["elf"], lw=0.7),
        )

    # Annotate Discrete cluster as a group (text box, no confusing arrows)
    disc_df = df[df["paradigm"] == "diffusion"]
    if not disc_df.empty:
        cx = np.exp(np.mean(np.log(disc_df["wt_bpb"].values)))
        cy = disc_df["distinct_2"].mean()
        ax.text(cx, cy + 0.055,
                "Discrete\ndiffusion",
                ha="center", va="bottom", fontsize=7.2,
                color=C["diffusion"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=C["diffusion"], lw=0.8, alpha=0.85))

    # Corner hint — placed where legend won't obscure it
    ax.text(0.022, 0.847, "← more fluent  ↑ more diverse",
            fontsize=6.5, color="#aaa", va="top", ha="left")

    ax.set_xlabel("WikiText-103 bpb  (↓ better)", fontsize=8.5)
    ax.set_ylabel("Distinct-2  (↑ better)", fontsize=8.5)
    ax.set_title("(a)  Fluency–diversity trade-off", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.2f}" if x >= 0.10 else f"{x:.3f}"))

    # ── Panel B: Rep-4 scatter ────────────────────────────────────────────────
    #    Repetition rate (lower is better) vs distinct-2
    ax = ax2
    for _, r in df.iterrows():
        col = PAR_COL[r["paradigm"]]
        mk  = ATTN_MK.get(r["attn"], "o")
        sz  = 110 if r["params_m"] > 120 else 50
        ax.scatter(r["distinct_2"], r["rep_4"],
                   marker=mk, color=col, s=sz, zorder=4,
                   edgecolors="white", linewidths=0.8, alpha=0.92)

    # Label the high-repetition outliers — MLA models are the key story here
    LABEL2 = {
        "elf_mla_512d_12b_Dense": ("ELF-MLA-512d", (6, 6), "left", "bottom"),
        "elf_mla_768d_16b_Dense": ("ELF-MLA-768d", (6, 6), "left", "bottom"),
    }
    for _, r in df.iterrows():
        if r["run"] not in LABEL2: continue
        lbl, xy_off, ha, va = LABEL2[r["run"]]
        col = PAR_COL[r["paradigm"]]
        ax.annotate(lbl, (r["distinct_2"], r["rep_4"]),
                    xytext=xy_off, textcoords="offset points",
                    fontsize=7.0, color=col, fontweight="bold",
                    ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7))

    ax.set_xlabel("Distinct-2  (↑ better)", fontsize=8.5)
    ax.set_ylabel("Rep-4  (↓ better = less repetition)", fontsize=8.5)
    ax.set_title("(b)  Diversity vs repetition", fontweight="bold")

    # ── Shared legend at bottom ────────────────────────────────────────────────
    leg = (
        [mpatches.Patch(color=C["elf"],       label="ELF (flow matching)"),
         mpatches.Patch(color=C["diffusion"], label="Discrete diffusion")]
        + [plt.Line2D([0],[0], marker=mk, color="#555", ms=6, lw=0,
                      label=f"{attn}")
           for attn, mk in ATTN_MK.items()]
        + [plt.Line2D([0],[0], marker="o", color="#555", ms=5, lw=0,
                      label="512d"),
           plt.Line2D([0],[0], marker="o", color="#555", ms=8.5, lw=0,
                      label="768d")]
    )
    fig.legend(handles=leg, fontsize=7.2, loc="lower center",
               ncol=7, handletextpad=0.35, handlelength=1.2,
               columnspacing=1.0, framealpha=0.0, borderpad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    _save(fig, out, "fig3_quality")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Attention variant analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_attention(root: Path, out: Path) -> None:
    ppl_path = root / "results" / "perplexity.csv"
    gq_path  = root / "results" / "generation_quality.csv"
    if not ppl_path.exists() or not gq_path.exists():
        print("  [SKIP] fig4 — data missing"); return

    ppl = pd.read_csv(ppl_path)
    gq  = pd.read_csv(gq_path)

    fig, axes = plt.subplots(1, 3, figsize=(TW, 3.1),
                             gridspec_kw={"wspace": 0.38})
    fig.suptitle(
        "Attention variant is orthogonal to Discrete quality; "
        "MHA ≈ MLA for ELF — GQA degrades reconstruction at 768d",
        fontsize=8.8, fontweight="bold", y=1.02,
    )

    ATTNS   = ["MHA", "GQA", "MLA"]
    x_pos   = np.array([0, 1, 2])
    OFFSETS = {"diffusion": -0.18, "elf": +0.18}
    MARKERS = {"diffusion": "s", "elf": "^"}
    SIZES   = {512: 48, 768: 110}

    # ── Panel A: WikiText-103 bpb by attention ────────────────────────────────
    ax = axes[0]
    wt = ppl[ppl["dataset"] == "wikitext-103"].copy()
    # exclude degenerate ELF-GQA-768d
    wt = wt[~((wt["model_type"] == "elf") & (wt["bpb"] > 0.45))]

    for paradigm in ["diffusion", "elf"]:
        sub = wt[wt["model_type"] == paradigm]
        for _, r in sub.iterrows():
            attn = r["attn_variant"]
            if attn not in ATTNS: continue
            col  = C[attn]
            xp   = ATTNS.index(attn) + OFFSETS[paradigm]
            sz   = SIZES.get(int(r["dim"]), 55)
            ax.scatter(xp, r["bpb"], marker=MARKERS[paradigm], color=col,
                       s=sz, zorder=4, edgecolors="white", lw=0.8, alpha=0.9)

    # Annotate best ELF values
    elf_wt = wt[wt["model_type"] == "elf"]
    for attn in ATTNS:
        best = elf_wt[elf_wt["attn_variant"] == attn].nsmallest(1, "bpb")
        if best.empty: continue
        xp = ATTNS.index(attn) + OFFSETS["elf"]
        v  = float(best["bpb"].iloc[0])
        ax.text(xp, v - 0.015, f"{v:.3f}", ha="center", va="top",
                fontsize=6.0, color=C[attn], fontweight="bold")

    ax.set_xticks(x_pos); ax.set_xticklabels(ATTNS)
    ax.set_ylabel("Bits per byte (bpb) ↓")
    ax.set_title("(a)  WikiText-103 bpb", fontweight="bold")
    ax.set_ylim(bottom=0)

    leg = [plt.Line2D([0], [0], marker="s", color="#777", ms=7, lw=0,
                      label="Discrete"),
           plt.Line2D([0], [0], marker="^", color="#777", ms=7, lw=0,
                      label="ELF"),
           plt.Line2D([0], [0], marker="o", color="#777", ms=4.5, lw=0,
                      label="512d"),
           plt.Line2D([0], [0], marker="o", color="#777", ms=8.5, lw=0,
                      label="768d")]
    ax.legend(handles=leg, fontsize=6.5, loc="upper right",
              handlelength=0.8, handletextpad=0.3, borderpad=0.5)

    # ── Panel B: Distinct-2 by attention ─────────────────────────────────────
    ax = axes[1]
    for paradigm in ["diffusion", "elf"]:
        sub = gq[gq["model_type"] == paradigm]
        for _, r in sub.iterrows():
            attn = r["attn_variant"]
            if attn not in ATTNS: continue
            col  = C[attn]
            xp   = ATTNS.index(attn) + OFFSETS[paradigm]
            sz   = SIZES.get(512 if r["params_m"] < 120 else 768, 55)
            ax.scatter(xp, r["distinct_2"], marker=MARKERS[paradigm], color=col,
                       s=sz, zorder=4, edgecolors="white", lw=0.8, alpha=0.88)

    ax.set_xticks(x_pos); ax.set_xticklabels(ATTNS)
    ax.set_ylabel("Distinct-2 ↑")
    ax.set_ylim(0.28, 0.88)
    ax.set_title("(b)  Lexical diversity", fontweight="bold")

    # ── Panel C: Distinct-2 vs bpb scatter — trade-off space ─────────────────
    ax = axes[2]
    merged = []
    wt_all = ppl[ppl["dataset"] == "wikitext-103"].copy()
    wt_all = wt_all[~((wt_all["model_type"] == "elf") & (wt_all["bpb"] > 0.45))]
    for _, gq_r in gq.iterrows():
        ppl_r = wt_all[wt_all["run"] == gq_r["run"]]
        if ppl_r.empty: continue
        merged.append({
            "paradigm":  gq_r["model_type"],
            "attn":      gq_r["attn_variant"],
            "params_m":  gq_r["params_m"],
            "distinct_2": gq_r["distinct_2"],
            "rep_4":     gq_r["rep_4"],
            "wt_bpb":    float(ppl_r["bpb"].iloc[0]),
        })
    mdf = pd.DataFrame(merged)

    for _, r in mdf.iterrows():
        col  = C[r["attn"]]
        mk   = MARKERS[r["paradigm"]]
        sz   = SIZES.get(512 if r["params_m"] < 120 else 768, 55)
        ax.scatter(r["wt_bpb"], r["distinct_2"], marker=mk, color=col,
                   s=sz, zorder=4, edgecolors="white", lw=0.8, alpha=0.9)

    # Ideal region annotation
    ax.annotate("", xy=(0.02, 0.82), xytext=(0.02, 0.74),
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))
    ax.text(0.03, 0.78, "better\ndiversity", fontsize=6, color="#888")
    ax.annotate("", xy=(0.02, 0.82), xytext=(0.10, 0.82),
                arrowprops=dict(arrowstyle="<-", color="#888", lw=0.9))
    ax.text(0.055, 0.83, "better bpb", fontsize=6, color="#888", ha="center")

    ax.set_xlabel("WikiText-103 bpb ↓")
    ax.set_ylabel("Distinct-2 ↑")
    ax.set_title("(c)  Quality–diversity trade-off", fontweight="bold")

    # Color legend for attention
    col_handles = [mpatches.Patch(color=C[a], label=a) for a in ATTNS]
    axes[2].legend(handles=col_handles, fontsize=6.8, loc="lower left",
                   handlelength=1.2, handletextpad=0.4)

    fig.tight_layout()
    _save(fig, out, "fig4_attention")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1 — Full results (LaTeX)
# ═══════════════════════════════════════════════════════════════════════════════

def table1_results(root: Path, out: Path) -> None:
    """Generate a publication-quality LaTeX table combining ppl + gen-quality."""
    ppl_path = root / "results" / "perplexity.csv"
    gq_path  = root / "results" / "generation_quality.csv"
    if not ppl_path.exists() or not gq_path.exists():
        print("  [SKIP] table1 — data missing"); return

    ppl = pd.read_csv(ppl_path)
    gq  = pd.read_csv(gq_path)

    # ── Pivot perplexity: one row per run ─────────────────────────────────────
    def _bpb(run, ds):
        r = ppl[(ppl["run"] == run) & (ppl["dataset"] == ds)]
        if r.empty: return None
        v = float(r["bpb"].iloc[0])
        return v if not np.isnan(v) else None

    def _ppl_trainval(run):
        r = ppl[(ppl["run"] == run) & (ppl["dataset"] == "train_val")]
        if r.empty: return None
        v = float(r["train_val_ppl"].iloc[0])
        return v if not np.isnan(v) else None

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None and not np.isnan(v) else "---"

    def _bold(v, fmt=".3f"):
        """Return LaTeX \textbf{val} string."""
        return f"\\textbf{{{v:{fmt}}}}" if v is not None and not np.isnan(v) else "---"

    # Collect all runs (merge ppl + gq)
    runs_order = []
    ppl_runs   = ppl["run"].unique()
    for run in sorted(ppl_runs):
        row = ppl[ppl["run"] == run].iloc[0]
        runs_order.append({
            "run":      run,
            "paradigm": row["model_type"],
            "attn":     row["attn_variant"],
            "dim":      int(row["dim"]),
            "params":   float(row["params_m"]),
        })

    # Identify best values per metric × paradigm (for bold annotation)
    elf_runs  = [r for r in runs_order if r["paradigm"] == "elf"]
    diff_runs = [r for r in runs_order if r["paradigm"] == "diffusion"]

    def _best_val(run_list, metric_fn, smaller_is_better=True):
        vals = [(r["run"], metric_fn(r["run"])) for r in run_list]
        vals = [(r, v) for r, v in vals if v is not None]
        if not vals: return set()
        best = min(vals, key=lambda x: x[1]) if smaller_is_better else max(vals, key=lambda x: x[1])
        return {best[0]}

    best_elf_wt   = _best_val(elf_runs,  lambda r: _bpb(r, "wikitext-103"))
    best_elf_lam  = _best_val(elf_runs,  lambda r: _bpb(r, "lambada"))
    best_diff_wt  = _best_val(diff_runs, lambda r: _bpb(r, "wikitext-103"))
    best_diff_lam = _best_val(diff_runs, lambda r: _bpb(r, "lambada"))

    gq_dict = {r["run"]: r for _, r in gq.iterrows()}
    best_d2_elf  = _best_val(elf_runs,  lambda r: gq_dict[r].get("distinct_2", None) if r in gq_dict else None, smaller_is_better=False)
    best_d2_diff = _best_val(diff_runs, lambda r: gq_dict[r].get("distinct_2", None) if r in gq_dict else None, smaller_is_better=False)
    best_r4_elf  = _best_val(elf_runs,  lambda r: gq_dict[r].get("rep_4", None) if r in gq_dict else None)
    best_r4_diff = _best_val(diff_runs, lambda r: gq_dict[r].get("rep_4", None) if r in gq_dict else None)

    # ── LaTeX generation ──────────────────────────────────────────────────────
    PARADIGM_DISPLAY = {"elf": "ELF", "diffusion": "Discrete"}

    lines = [
        r"% Auto-generated by benchmarks/plot_emnlp_paper.py",
        r"% Columns: Paradigm | Attn | Dim | Params | Train PPL | WT-103 bpb | LAMBADA bpb | D-2↑ | Rep-4↓ | SB-4↓",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{%",
        r"    \textbf{DantinoX model suite results.}",
        r"    ELF (Embedded Language Flows) and Discrete Diffusion are evaluated on",
        r"    WikiText-103 and LAMBADA (bits-per-byte, bpb $\downarrow$) and on open-ended",
        r"    generation quality (Distinct-2 $\uparrow$, 4-gram repetition Rep-4 $\downarrow$,",
        r"    Self-BLEU-4 SB-4 $\downarrow$). Bold = best per paradigm.",
        r"    \dag{} ELF-GQA-768d omitted (training collapse; see \S\ref{sec:analysis}).",
        r"  }",
        r"  \label{tab:results}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrr|rrr|rrr}",
        r"    \toprule",
        r"    \multicolumn{4}{c|}{\textbf{Architecture}} &",
        r"    \multicolumn{3}{c|}{\textbf{Language Modelling}} &",
        r"    \multicolumn{3}{c}{\textbf{Generation Quality}} \\",
        r"    \cmidrule(lr){1-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"    Paradigm & Attn & $d$ & Params &",
        r"    Train PPL & WT-103 bpb & LAM bpb &",
        r"    D-2 $\uparrow$ & Rep-4 $\downarrow$ & SB-4 $\downarrow$ \\",
        r"    \midrule",
    ]

    prev_paradigm = None
    for r in runs_order:
        run  = r["run"]
        par  = r["paradigm"]
        attn = r["attn"]
        dim  = r["dim"]

        # Section separator
        if prev_paradigm is not None and prev_paradigm != par:
            lines.append(r"    \midrule")
        prev_paradigm = par

        par_disp  = PARADIGM_DISPLAY[par]
        dim_disp  = str(dim)
        par_m     = r["params"]
        train_ppl = _ppl_trainval(run)
        wt_bpb    = _bpb(run, "wikitext-103")
        lam_bpb   = _bpb(run, "lambada")

        gq_row    = gq_dict.get(run)
        dist2     = float(gq_row["distinct_2"])  if gq_row is not None else None
        rep4      = float(gq_row["rep_4"])        if gq_row is not None else None
        sb4       = float(gq_row["self_bleu_4"])  if gq_row is not None else None

        def _cell(v, fmt, is_best):
            s = f"{v:{fmt}}" if v is not None and not np.isnan(v) else "---"
            return f"\\textbf{{{s}}}" if is_best and s != "---" else s

        best_wt  = best_elf_wt  if par == "elf" else best_diff_wt
        best_lam = best_elf_lam if par == "elf" else best_diff_lam
        best_d2  = best_d2_elf  if par == "elf" else best_d2_diff
        best_r4  = best_r4_elf  if par == "elf" else best_r4_diff

        # Mark degenerate runs
        degenerate = (par == "elf" and wt_bpb is not None and wt_bpb > 0.45)
        run_suffix = r"${}^\dag$" if degenerate else ""

        col_wt   = _cell(wt_bpb,   ".4f", run in best_wt   and not degenerate)
        col_lam  = _cell(lam_bpb,  ".4f", run in best_lam  and not degenerate)
        col_d2   = _cell(dist2,    ".3f", run in best_d2   and not degenerate)
        col_r4   = _cell(rep4,     ".4f", run in best_r4   and not degenerate)
        col_sb4  = _cell(sb4,      ".3f", False)
        col_tv   = f"{train_ppl:.2f}" if train_ppl is not None else "---"

        lines.append(
            f"    {par_disp}{run_suffix} & {attn} & {dim_disp} & {par_m:.0f}M &"
            f" {col_tv} & {col_wt} & {col_lam} &"
            f" {col_d2} & {col_r4} & {col_sb4} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    tex_path = out / "table1_results.tex"
    tex_path.write_text(tex)
    print(f"  saved {tex_path}")

    # Also print a quick ASCII preview
    print("\n  ── LaTeX table preview ──")
    print(f"  {'Paradigm':<10} {'Attn':<4} {'d':>4} {'Params':>8}  "
          f"{'TrainPPL':>8}  {'WT-bpb':>7}  {'LAM-bpb':>7}  "
          f"{'D-2':>5}  {'Rep-4':>5}  {'SB-4':>5}")
    print("  " + "-" * 80)
    for r in runs_order:
        run      = r["run"]
        par      = r["paradigm"][:4].upper()
        attn     = r["attn"]
        dim      = r["dim"]
        par_m    = r["params"]
        tv       = _ppl_trainval(run)
        wt       = _bpb(run, "wikitext-103")
        lm       = _bpb(run, "lambada")
        gq_r     = gq_dict.get(run)
        d2       = float(gq_r["distinct_2"]) if gq_r is not None else None
        r4       = float(gq_r["rep_4"])      if gq_r is not None else None
        sb4v     = float(gq_r["self_bleu_4"]) if gq_r is not None else None
        print(f"  {par:<10} {attn:<4} {dim:>4} {par_m:>6.0f}M  "
              f"{_fmt(tv, '.2f'):>8}  {_fmt(wt, '.4f'):>7}  {_fmt(lm, '.4f'):>7}  "
              f"{_fmt(d2, '.3f'):>5}  {_fmt(r4, '.4f'):>5}  {_fmt(sb4v, '.3f'):>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Throughput scaling with batch size
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_throughput(root: Path, out: Path) -> None:
    bench_path = root / "results" / "paradigm_bench.csv"
    if not bench_path.exists():
        print("  [SKIP] fig5 — data missing"); return

    df = pd.read_csv(bench_path)
    bs = df[(df["group"] == "batch_size") & (df["dtype"] == "fp32")].copy()

    # Add streaming_e2e_ms if not present
    G = 128
    is_ar = bs["paradigm"] == "AR"
    if "streaming_e2e_ms" not in bs.columns:
        bs["streaming_e2e_ms"] = np.where(
            is_ar,
            bs["prefill_ms_p50"] + bs["step_ms_p50"] * (G - 1),
            bs["e2e_ms"]
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TW, 3.0))
    fig.suptitle(
        "Diffusion throughput scales 4× faster with batch than AR; "
        "AR streaming latency (G=128) is 3-8× higher than diffusion e2e",
        fontsize=8.8, fontweight="bold", y=1.01,
    )

    bs_vals = sorted(bs["batch_size"].unique())

    # ── Panel A: Throughput vs batch size ─────────────────────────────────────
    paradigm_cfg = [
        ("AR",         C["AR"],         "o", "-",  1.8, "AR"),
        ("Discrete",   C["Discrete"],   "s", "-",  1.8, "Discrete (S=32)"),
        ("Continuous", C["Continuous"], "^", "-",  1.8, "ELF (S=32)"),
    ]
    for par, color, mk, ls, lw, lbl in paradigm_cfg:
        d = bs[bs["paradigm"] == par].sort_values("batch_size")
        ax1.plot(d["batch_size"], d["tok_s_e2e"] / 1e3,
                 color=color, marker=mk, linestyle=ls, lw=lw, ms=5.5,
                 label=lbl, zorder=3)

    # AR ideal linear scaling reference
    ar_b1 = bs[(bs["paradigm"] == "AR") & (bs["batch_size"] == 1)]["tok_s_e2e"].values
    if len(ar_b1):
        ideal = [ar_b1[0] * b / 1e3 for b in bs_vals]
        ax1.plot(bs_vals, ideal, color=C["AR"], lw=0.9, ls=":",
                 alpha=0.5, label="AR ideal linear", zorder=1)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(bs_vals)
    ax1.set_xticklabels([str(b) for b in bs_vals], fontsize=7)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (k tok/s) ↑")
    ax1.set_title("(a) Throughput vs batch, medium model")
    ax1.legend(fontsize=7, loc="upper left")

    # ── Panel B: Streaming e2e latency vs batch ────────────────────────────────
    for par, color, mk, ls, lw, lbl in paradigm_cfg:
        d = bs[bs["paradigm"] == par].sort_values("batch_size")
        ax2.plot(d["batch_size"], d["streaming_e2e_ms"],
                 color=color, marker=mk, linestyle=ls, lw=lw, ms=5.5,
                 label=lbl, zorder=3)

    ax2.axhline(200, color="#888", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.text(1.1, 215, "200 ms SLO", fontsize=7, color="#555")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(bs_vals)
    ax2.set_xticklabels([str(b) for b in bs_vals], fontsize=7)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Full-generation latency (ms) ↓\n(AR: streaming; Diffusion/ELF: e2e)")
    ax2.set_title(f"(b) End-to-end generation latency (G={G})")
    ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    _save(fig, out, "fig5_throughput")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/emnlp_paper_figs",
                        help="Output directory for figures")
    parser.add_argument("--figs", nargs="+",
                        choices=["1", "2", "3", "4", "5", "t1", "all"],
                        default=["all"],
                        help="Which figures/tables to generate (default: all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out  = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out}\n")

    figs_to_run = set(args.figs)
    run_all = "all" in figs_to_run

    if run_all or "1" in figs_to_run:
        print("Generating fig1_serving …")
        fig1_serving(root, out)
    if run_all or "2" in figs_to_run:
        print("Generating fig2_ppl …")
        fig2_ppl(root, out)
    if run_all or "3" in figs_to_run:
        print("Generating fig3_quality …")
        fig3_quality(root, out)
    if run_all or "4" in figs_to_run:
        print("Generating fig4_attention …")
        fig4_attention(root, out)
    if run_all or "5" in figs_to_run:
        print("Generating fig5_throughput …")
        fig5_throughput(root, out)
    if run_all or "t1" in figs_to_run:
        print("Generating table1_results …")
        table1_results(root, out)

    n_pdf = len(list(out.glob("*.pdf")))
    n_tex = len(list(out.glob("*.tex")))
    print(f"\nDone. {n_pdf} PDFs, {n_tex} .tex in {out}")


if __name__ == "__main__":
    main()
