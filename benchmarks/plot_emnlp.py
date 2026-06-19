#!/usr/bin/env python3
"""
benchmarks/plot_emnlp.py
========================
Publication figures for the EMNLP 2026 System Demonstrations paper.

Design rule: each figure asserts EXACTLY ONE claim, stated in the title and
annotated directly on the data. Four figures cover the full story:

  fig_emnlp_1_pareto    Serving frontier — diffusion is 3-7x faster than AR
                        at interactive latency; AR recovers at high batch.
  fig_emnlp_2_quality   Quality-efficiency scatter — diff achieves competitive
                        bpb at 3-7x the throughput of AR.
  fig_emnlp_3_attn      Attention orthogonality — GQA/MLA cut KV cache 4-6x
                        with <0.002 bpb change on masked diffusion.
  fig_emnlp_4_scaling   Throughput advantage holds from 0.2M to 114M params.

Run:
  python benchmarks/plot_emnlp.py
  python benchmarks/plot_emnlp.py --out results/emnlp_figs

Safe to re-run; updates automatically once results/perplexity_ar.csv appears.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "legend.fontsize":   7.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        200,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.20,
    "grid.linestyle":    ":",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colorblind-safe palette (Wong 2011)
C = {
    "AR":         "#0072B2",
    "Discrete":   "#D55E00",
    "Continuous": "#009E73",
}
LIGHT = {
    "AR":         "#56B4E9",
    "Discrete":   "#E69F00",
    "Continuous": "#CC79A7",
}
MARKER = {"AR": "o", "Discrete": "s", "Continuous": "^"}
LABEL  = {
    "AR":         "Autoregressive",
    "Discrete":   "Masked Diffusion (LLaDA)",
    "Continuous": "Continuous Diffusion (ELF)",
}


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  saved  {name}.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Serving Pareto frontier
# Claim: "At interactive latency (<=200 ms), diffusion fits; AR does not.
#         Discrete S=8 is ~6x faster than AR at B=1."
# ══════════════════════════════════════════════════════════════════════════════

def fig1_pareto(par_path: Path, out: Path) -> None:
    if not par_path.exists():
        print(f"  [SKIP] fig1 — missing {par_path.name}")
        return

    par = pd.read_csv(par_path)
    par = par[~par["oom"]].copy()

    # Series definition: (paradigm_key, linestyle, linewidth, marker, markersize)
    SERIES = {
        "AR (greedy)":     ("AR",         "-",  2.0, "o", 5.5),
        "Discrete S=32":   ("Discrete",   "-",  1.8, "s", 4.5),
        "Discrete S=8":    ("Discrete",   "--", 1.4, "s", 4.0),
        "Continuous S=32": ("Continuous", "-",  1.8, "^", 4.5),
        "Continuous S=8":  ("Continuous", "--", 1.4, "^", 4.0),
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for lbl, (paradigm, ls, lw, mk, ms) in SERIES.items():
        d = par[par["label"] == lbl].sort_values("batch_size")
        if d.empty:
            continue
        col = C[paradigm] if ("S=32" in lbl or paradigm == "AR") else LIGHT[paradigm]
        ax.plot(d["e2e_ms_med"], d["tok_s_system"],
                color=col, ls=ls, lw=lw, marker=mk, ms=ms,
                zorder=3, label=lbl, alpha=0.90)
        for _, r in d.iterrows():
            if r["batch_size"] in (1, 256):
                xoff = 5 if r["e2e_ms_med"] < 500 else -32
                ax.annotate(f"B={int(r['batch_size'])}",
                            (r["e2e_ms_med"], r["tok_s_system"]),
                            textcoords="offset points", xytext=(xoff, 5),
                            fontsize=7, color=col)

    # 200 ms interactive SLO guideline
    ax.autoscale_view()
    ax.axvline(200, color="#888888", lw=1.0, ls=":", zorder=1)
    ax.text(200, ax.get_ylim()[0] * 2.5, "200 ms\nSLO", fontsize=7, color="#666666",
            ha="left", va="bottom")

    # Annotate B=1 latency speedup (AR vs Discrete S=8)
    d_ar = par[par["label"] == "AR (greedy)"]
    d_d8 = par[par["label"] == "Discrete S=8"]
    if not d_ar.empty and not d_d8.empty:
        r_ar = d_ar[d_ar["batch_size"] == 1].iloc[0]
        r_d8 = d_d8[d_d8["batch_size"] == 1].iloc[0]
        speedup_lat = r_ar["e2e_ms_med"] / r_d8["e2e_ms_med"]
        speedup_tok = r_d8["tok_s_system"] / r_ar["tok_s_system"]
        ax.annotate(
            f"B=1: Discrete S=8 is\n{speedup_lat:.1f}x lower latency\n"
            f"{speedup_tok:.1f}x higher throughput",
            xy=(r_d8["e2e_ms_med"], r_d8["tok_s_system"]),
            xytext=(60, 15000),
            fontsize=8, fontweight="bold", color="#444444",
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#666666"),
        )

    # Crossover annotation
    d_d32 = par[par["label"] == "Discrete S=32"]
    if not d_ar.empty and not d_d32.empty:
        m = pd.merge(
            d_ar[["batch_size", "tok_s_system"]],
            d_d32[["batch_size", "tok_s_system"]],
            on="batch_size", suffixes=("_ar", "_d"),
        ).sort_values("batch_size")
        for _, row in m.iterrows():
            if row["tok_s_system_ar"] >= row["tok_s_system_d"]:
                bx = int(row["batch_size"])
                rx = d_ar[d_ar["batch_size"] == bx].iloc[0]
                ax.annotate(
                    f"AR overtakes Discrete S=32 at B={bx}",
                    xy=(rx["e2e_ms_med"], rx["tok_s_system"]),
                    xytext=(rx["e2e_ms_med"] * 1.8, rx["tok_s_system"] * 0.4),
                    fontsize=7.5, fontweight="bold", color="#444444",
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#666666"),
                )
                break

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Per-request latency  (ms, G=256 tokens, bf16, A100 40 GB)")
    ax.set_ylabel("System throughput  (tok/s, all sequences combined)")
    ax.set_title(
        "Serving frontier (512d · 12-layer · ~70M params): diffusion fits\n"
        "the 200 ms interactive SLO; AR requires 235 ms at B=1.",
        pad=8,
    )
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    _save(fig, out, "fig_emnlp_1_pareto")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Quality-Efficiency scatter
# Claim: "Masked diffusion achieves competitive bpb at 3-4x AR throughput."
# ══════════════════════════════════════════════════════════════════════════════

# Throughput: tok/s from paradigm_bench_full.csv, group=scale, size=xl/xxl,
# attn=MHA, B=4, fp32 — matched to trained model sizes by dim
_TOK_S = {
    ("AR",         512): 156.79,
    ("Discrete",   512): 547.66,
    ("Continuous", 512): 529.55,
    ("AR",         768): 252.26,
    ("Discrete",   768): 985.95,
    ("Continuous", 768): 990.71,
}


def _load_perplexity(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["dataset"] == "wikitext-103"].copy()

    def _par(run: str) -> str:
        if run.startswith("ar_"):    return "AR"
        if run.startswith("diff_"): return "Discrete"
        if run.startswith("elf_"):  return "Continuous"
        return "Unknown"

    def _dim(run: str) -> int:
        for tok in run.split("_"):
            if tok.endswith("d") and tok[:-1].isdigit():
                return int(tok[:-1])
        return 0

    def _attn(run: str) -> str:
        for a in ("mha", "gqa", "mla"):
            if f"_{a}_" in run.lower():
                return a.upper()
        return "MHA"

    df["paradigm"]     = df["run"].apply(_par)
    df["dim"]          = df["run"].apply(_dim)
    df["attn_variant"] = df.get("attn_variant", pd.Series(dtype=str)).combine_first(
        df["run"].apply(_attn)
    )
    df["tok_s"] = df.apply(
        lambda r: _TOK_S.get((r["paradigm"], r["dim"]), np.nan), axis=1
    )
    return df[df["tok_s"].notna() & df["bpb"].notna()].copy()


def fig2_quality(ppl_paths: list[Path], out: Path) -> None:
    df = _load_perplexity(ppl_paths)
    if df.empty:
        print("  [SKIP] fig2 — no perplexity data yet")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    SIZE_S = {512: 100, 768: 200}
    EDGE   = {"MHA": "#333333", "GQA": "#00aa00", "MLA": "#cc0000"}

    for _, row in df.iterrows():
        par  = row["paradigm"]
        col  = C.get(par, "grey")
        sz   = SIZE_S.get(int(row["dim"]), 100)
        mk   = MARKER.get(par, "o")
        attn = row.get("attn_variant", "MHA")
        ec   = EDGE.get(attn, "#333333")
        # ELF is a different metric — make it semi-transparent
        alpha = 0.50 if par == "Continuous" else 0.90
        ax.scatter(row["tok_s"], row["bpb"],
                   s=sz, c=col, marker=mk, edgecolors=ec,
                   linewidths=1.3, zorder=4, alpha=alpha)

    # ELF annotation
    elf = df[df["paradigm"] == "Continuous"]
    if not elf.empty:
        ax.axhline(elf["bpb"].max() + 0.015,
                   color=C["Continuous"], ls=":", lw=0.8, alpha=0.5)
        ax.text(df["tok_s"].min() * 1.03, elf["bpb"].max() + 0.018,
                "ELF bpb = reconstruction quality at t=1\n"
                "(not directly comparable to AR / Masked Diffusion)",
                fontsize=6.5, color=C["Continuous"], alpha=0.75)

    # Throughput speedup annotation
    ar  = df[(df["paradigm"] == "AR") & (df["attn_variant"] == "MHA")]
    di  = df[(df["paradigm"] == "Discrete") & (df["attn_variant"] == "MHA")]
    if not ar.empty and not di.empty:
        for dim in [512, 768]:
            ar_r = ar[ar["dim"] == dim]
            di_r = di[di["dim"] == dim]
            if ar_r.empty or di_r.empty:
                continue
            ratio = float(di_r["tok_s"].iloc[0]) / float(ar_r["tok_s"].iloc[0])
            ax.annotate(
                f"{ratio:.1f}x\nthroughput",
                xy=(float(di_r["tok_s"].iloc[0]),
                    float(di_r["bpb"].iloc[0])),
                xytext=(float(di_r["tok_s"].iloc[0]) * 0.55,
                        float(di_r["bpb"].iloc[0]) - 0.012),
                fontsize=7.5, fontweight="bold",
                color=C["Discrete"],
                arrowprops=dict(arrowstyle="-|>",
                                color=C["Discrete"], lw=0.9),
            )

    # Legend
    par_handles = [
        mpatches.Patch(color=C[p], label=LABEL[p])
        for p in ["AR", "Discrete", "Continuous"]
        if p in df["paradigm"].values
    ]
    size_handles = [
        plt.scatter([], [], s=100, c="grey", label="512d (~70M params)"),
        plt.scatter([], [], s=200, c="grey", label="768d (~114M params)"),
    ]
    edge_handles = [
        plt.scatter([], [], s=70, c="grey", edgecolors=ec,
                    linewidths=1.5, label=f"Attn: {a}")
        for a, ec in EDGE.items()
    ]
    ax.legend(handles=par_handles + size_handles + edge_handles,
              fontsize=7, loc="lower right", framealpha=0.92, ncol=2)

    ax.invert_yaxis()   # lower bpb = better → top
    ax.set_xlabel("Inference throughput  (tok/s · B=4 · G=128 · fp32 · A100)")
    ax.set_ylabel("WikiText-103 bpb  (lower = better quality)")
    ax.set_title(
        "Quality–efficiency: masked diffusion achieves competitive bpb\n"
        "at 3–4x the throughput of autoregressive decoding.",
        pad=8,
    )
    _save(fig, out, "fig_emnlp_2_quality")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Attention orthogonality (two panels)
# Claim: "GQA/MLA cut KV cache 4-6x with <0.002 bpb change on diffusion."
# ══════════════════════════════════════════════════════════════════════════════

def fig3_attention(ppl_paths: list[Path], bench_path: Path, out: Path) -> None:
    df    = _load_perplexity(ppl_paths)
    bench = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ── (a) bpb by attention variant — diffusion only ────────────────────────
    ax = axes[0]
    diff_df = df[df["paradigm"] == "Discrete"].copy() if not df.empty else pd.DataFrame()

    if not diff_df.empty:
        ATTN  = ["MHA", "GQA", "MLA"]
        DIMS  = sorted(diff_df["dim"].unique())
        x     = np.arange(len(ATTN))
        W     = 0.32
        DCOL  = {512: "#4393c3", 768: "#08306b"}

        for i, dim in enumerate(DIMS):
            sub  = diff_df[diff_df["dim"] == dim]
            bpbs = []
            for attn in ATTN:
                r = sub[sub["attn_variant"] == attn]
                bpbs.append(float(r["bpb"].iloc[0]) if not r.empty else np.nan)
            offset = (i - (len(DIMS) - 1) / 2) * W
            bars = ax.bar(x + offset, bpbs, W,
                          color=DCOL.get(dim, "grey"),
                          label=f"{dim}d / ~{70 if dim==512 else 180}M",
                          zorder=3)
            for bar, v in zip(bars, bpbs):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v + 4e-4, f"{v:.4f}",
                            ha="center", va="bottom", fontsize=6.5)

        # Annotate max spread
        spread_vals = []
        for dim in DIMS:
            sub  = diff_df[diff_df["dim"] == dim]
            vals = [float(sub[sub["attn_variant"] == a]["bpb"].iloc[0])
                    for a in ATTN if not sub[sub["attn_variant"] == a].empty]
            if len(vals) >= 2:
                spread_vals.append(max(vals) - min(vals))
        if spread_vals:
            ax.text(0.5, 0.97,
                    f"Max bpb spread = {max(spread_vals):.4f}  (MHA / GQA / MLA)",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", alpha=0.8))

        ax.set_xticks(x, ATTN)
        # Zoom y-axis to the tight bpb range
        all_bpb = diff_df["bpb"].dropna()
        ax.set_ylim(all_bpb.min() - 0.006, all_bpb.max() + 0.012)
        ax.set_ylabel("WikiText-103 bpb (lower = better)")
        ax.set_title("(a) Attention variant does NOT hurt\ngeneration quality (Masked Diffusion)",
                     pad=6)
        ax.legend(fontsize=7.5, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Perplexity data missing\n(run benchmarks/perplexity_eval.py)",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    # ── (b) KV cache vs throughput per paradigm × attention ──────────────────
    ax2 = axes[1]
    if not bench.empty:
        xl = bench[
            (bench["group"] == "scale") &
            (bench["size"] == "xl") &
            (~bench["oom"])
        ].copy()

        attn_mk = {"MHA": "o", "GQA": "s", "MLA": "^"}
        plotted = set()
        for _, row in xl.iterrows():
            par  = row.get("paradigm", "")
            attn = row.get("attn", "MHA")
            cache = row.get("cache_mb", np.nan)
            tok_s = row.get("tok_s_e2e", np.nan)
            if par not in C or np.isnan(tok_s):
                continue
            # ELF has no KV cache — plot at cache=0
            if par == "Continuous" and (np.isnan(cache) or cache == 0):
                cache = 0.0
            if np.isnan(cache):
                continue
            col = C[par]
            mk  = attn_mk.get(attn, "o")
            label_key = (par, attn)
            lbl = f"{LABEL[par][:14]}… / {attn}" if label_key not in plotted else "_nolegend_"
            plotted.add(label_key)
            ax2.scatter(cache, tok_s, s=130, c=col, marker=mk,
                        edgecolors="white", linewidths=0.8,
                        zorder=4, alpha=0.92)
            ax2.annotate(f"{par[:2]}/{attn}",
                         (cache, tok_s),
                         textcoords="offset points",
                         xytext=(6, 4), fontsize=7, color=col)

        # Cache reduction arrows (MHA → GQA for AR)
        for par in ["AR", "Discrete"]:
            sub = xl[xl["paradigm"] == par]
            mha = sub[sub["attn"] == "MHA"]
            gqa = sub[sub["attn"] == "GQA"]
            if mha.empty or gqa.empty:
                continue
            x0, y0 = float(mha["cache_mb"].iloc[0]), float(mha["tok_s_e2e"].iloc[0])
            x1, y1 = float(gqa["cache_mb"].iloc[0]), float(gqa["tok_s_e2e"].iloc[0])
            ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                         color=C[par], alpha=0.5))
            ratio = x0 / x1 if x1 > 0 else 1
            ax2.text((x0 + x1) / 2, (y0 + y1) / 2 + 8,
                     f"{ratio:.0f}x less cache",
                     fontsize=6.5, color=C[par], ha="center")

        handles = [mpatches.Patch(color=C[p], label=LABEL[p])
                   for p in ["AR", "Discrete", "Continuous"] if p in xl["paradigm"].values]
        handles += [plt.scatter([], [], s=80, c="grey", marker=m, label=f"Attn: {a}")
                    for a, m in attn_mk.items()]
        ax2.legend(handles=handles, fontsize=7, loc="upper right")
        ax2.set_xlabel("KV cache per sequence  (MB, G=512 tokens)")
        ax2.set_ylabel("Inference throughput  (tok/s, B=4)")
        ax2.set_title("(b) KV cache vs throughput:\nGQA gives 4x memory reduction for AR",
                      pad=6)
    else:
        ax2.text(0.5, 0.5, "paradigm_bench_full.csv not found",
                 ha="center", va="center", transform=ax2.transAxes, color="grey")

    fig.suptitle(
        "Attention orthogonality: MHA / GQA / MLA leave generation quality unchanged "
        "while GQA cuts KV cache memory 4x",
        y=1.02, fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, "fig_emnlp_3_attention")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Scaling: throughput advantage holds at all model sizes
# ══════════════════════════════════════════════════════════════════════════════

def fig4_scaling(bench_path: Path, out: Path) -> None:
    if not bench_path.exists():
        print(f"  [SKIP] fig4 — missing {bench_path.name}")
        return

    bench = pd.read_csv(bench_path)
    scale = bench[
        (bench["group"] == "scale") &
        (bench["attn"] == "MHA") &
        (~bench["oom"])
    ].sort_values("params_m")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ── (a) absolute throughput ───────────────────────────────────────────────
    ax = axes[0]
    for par in ["AR", "Discrete", "Continuous"]:
        sub = scale[scale["paradigm"] == par]
        if sub.empty:
            continue
        ax.plot(sub["params_m"], sub["tok_s_e2e"],
                color=C[par], marker=MARKER[par],
                lw=2.0, ms=6.5, zorder=3, label=LABEL[par])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters  (M)")
    ax.set_ylabel("Inference throughput  (tok/s, B=4, G=128, fp32)")
    ax.set_title("(a) Absolute throughput vs model size:\ndiffusion consistently 3-4x faster than AR",
                 pad=6)
    ax.legend(fontsize=7.5)

    # ── (b) throughput ratio (Diffusion / AR) ────────────────────────────────
    ax2 = axes[1]
    ar_sub = scale[scale["paradigm"] == "AR"].set_index("params_m")["tok_s_e2e"]

    for par in ["Discrete", "Continuous"]:
        sub = scale[scale["paradigm"] == par].copy()
        xs, ys = [], []
        for _, row in sub.iterrows():
            nearest = ar_sub.index[np.argmin(np.abs(ar_sub.index - row["params_m"]))]
            xs.append(row["params_m"])
            ys.append(row["tok_s_e2e"] / ar_sub[nearest])
        if xs:
            ax2.plot(xs, ys, color=C[par], marker=MARKER[par],
                     lw=2.0, ms=6.5, zorder=3, label=LABEL[par])

    ax2.axhline(1.0, color=C["AR"], lw=1.5, ls="--",
                label="AR baseline (1x)", alpha=0.7)
    # Shade the 3-4x band
    ax2.axhspan(3.0, 4.0, alpha=0.06, color="#888888",
                label="3-4x speedup region")

    ax2.set_xscale("log")
    ax2.set_xlabel("Model parameters  (M)")
    ax2.set_ylabel("Throughput ratio  (vs AR at same size)")
    ax2.set_title("(b) Speedup ratio over AR:\nconsistently in the 3-4x band across all sizes",
                  pad=6)
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Scaling: the diffusion throughput advantage is robust — "
        "3-4x at all model sizes from 0.2 M to 114 M parameters",
        y=1.02, fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, "fig_emnlp_4_scaling")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate EMNLP 2026 paper figures")
    ap.add_argument("--root", default=".", help="Project root directory")
    ap.add_argument("--out",  default="results/emnlp_figs",
                    help="Output directory for figures")
    args = ap.parse_args()

    root = Path(args.root)
    out  = root / args.out

    par_path   = root / "results" / "ablation_pareto_512d12b.csv"
    bench_path = root / "results" / "paradigm_bench_full.csv"
    ppl_paths  = [
        root / "results" / "perplexity.csv",
        root / "results" / "perplexity_ar.csv",   # auto-added when AR eval finishes
    ]

    print("\nDantinoX — EMNLP 2026 figures")
    print("=" * 50)
    fig1_pareto(par_path, out)
    fig2_quality(ppl_paths, out)
    fig3_attention(ppl_paths, bench_path, out)
    fig4_scaling(bench_path, out)
    print(f"\nFigures saved to  {out}/\n")


if __name__ == "__main__":
    main()
