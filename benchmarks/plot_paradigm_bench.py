#!/usr/bin/env python3
"""
benchmarks/plot_paradigm_bench.py
=================================

EMNLP system-demonstration figures for the AR vs Discrete-Diffusion vs
Continuous-Diffusion inference benchmark (``paradigm_bench.py``).

Layout convention: every comparison is a grid whose **columns are the three
attention variants (MHA / GQA-1/4 / MLA)** and whose rows are the metrics;
each panel shows the three paradigms, so paradigm behaviour can be compared
within and across attention mechanisms at a glance.

Outputs (PDF + PNG) in ``results/paradigm_bench/``:

  fig1_scale.pdf        latency / throughput / GFLOPs-per-token vs model size
  fig2_batch.pdf        throughput / latency / MFU vs batch size
  fig3_genlen.pdf       latency / throughput / memory vs generation length
  fig4_steps.pdf        diffusion-steps knob vs AR baseline
  fig5_dtype.pdf        bf16 speedup bars
  fig6_prefill.pdf      prompt-processing cost and TTFT vs prompt length
  fig7_attention_summary.pdf  cross-regime bars (paradigm × attention)
  fig7b_cache.pdf       inference-cache memory by attention
  summary.md            headline numbers

Usage:
  python benchmarks/plot_paradigm_bench.py --csv results/paradigm_bench_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARADIGMS = ["AR", "Discrete", "Continuous"]
ATTNS     = ["MHA", "GQA", "MLA"]
COLORS    = {"AR": "#1f77b4", "Discrete": "#d62728", "Continuous": "#2ca02c"}
MARKERS   = {"AR": "o", "Discrete": "s", "Continuous": "^"}
NICE      = {"AR": "Autoregressive", "Discrete": "Discrete Diffusion (LLaDA)",
             "Continuous": "Continuous Diffusion (ELF)"}
ATTN_NICE = {"MHA": "MHA", "GQA": "GQA-1/4", "MLA": "MLA"}
ATTN_HATCH = {"MHA": "", "GQA": "//", "MLA": ".."}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf")


# ── Generic grid: rows = metrics, columns = attention variants ────────────────

def grid_by_attn(
    df: pd.DataFrame,
    group: str,
    x: str,
    metrics: list[tuple[str, str, bool]],   # (column, ylabel, logy)
    suptitle: str,
    fname: str,
    out: Path,
    logx: bool = True,
    logx_base: int = 2,
    paradigms: list[str] | None = None,
) -> None:
    paradigms = paradigms or PARADIGMS
    sub = df[(df["group"] == group) & (~df["oom"])]
    n_rows, n_cols = len(metrics), len(ATTNS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 2.6 * n_rows),
                             squeeze=False, sharex=True)

    for r, (col, ylabel, logy) in enumerate(metrics):
        # Shared y-range per metric row so attention columns are comparable
        vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
        for c, a in enumerate(ATTNS):
            ax = axes[r][c]
            d_attn = sub[sub["attn"] == a]
            for p in paradigms:
                d = d_attn[d_attn["paradigm"] == p].sort_values(x)
                d = d.dropna(subset=[col])
                if d.empty:
                    continue
                ax.plot(d[x], d[col], marker=MARKERS[p], color=COLORS[p],
                        label=NICE[p], markersize=4, linewidth=1.4)
            if logx:
                ax.set_xscale("log", base=logx_base)
            if logy and not vals.empty and (vals > 0).all():
                ax.set_yscale("log")
            if not vals.empty and logy:
                ax.set_ylim(vals.min() * 0.7, vals.max() * 1.4)
            elif not vals.empty:
                ax.set_ylim(0, vals.max() * 1.1)
            if r == 0:
                ax.set_title(ATTN_NICE[a])
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == n_rows - 1:
                ax.set_xlabel(x.replace("_", " "))

    axes[0][0].legend()
    fig.suptitle(suptitle, y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, fname)


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_scale(df: pd.DataFrame, out: Path) -> None:
    grid_by_attn(
        df, "scale", "params_m",
        [("e2e_ms",         "End-to-end latency (ms)", True),
         ("tok_s_e2e",      "Throughput (tok/s)",      True),
         ("gflops_per_tok", "GFLOPs / token",          True)],
        "Model-size scaling — columns: attention variant  (B=4, G=128, 32 steps, fp32, A100)",
        "fig1_scale", out, logx=True, logx_base=10,
    )


def fig_batch(df: pd.DataFrame, out: Path) -> None:
    grid_by_attn(
        df, "batch_size", "batch_size",
        [("tok_s_e2e", "Throughput (tok/s)",        True),
         ("e2e_ms",    "End-to-end latency (ms)",   True),
         ("mfu_pct",   "MFU (%)",                   False)],
        "Batch-size scaling — columns: attention variant  (medium ~6.6 M, G=128, fp32)",
        "fig2_batch", out,
    )


def fig_genlen(df: pd.DataFrame, out: Path) -> None:
    grid_by_attn(
        df, "gen_len", "gen_len",
        [("e2e_ms",      "End-to-end latency (ms)", True),
         ("tok_s_e2e",   "Throughput (tok/s)",      True),
         ("peak_mem_mb", "Device memory (MB)",      False)],
        "Generation-length scaling — columns: attention variant  (medium, B=4, 32 steps)",
        "fig3_genlen", out,
    )


def fig_steps(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["group"] == "diff_steps") & (~df["oom"])]
    metrics = [("tok_s_e2e", "Throughput (tok/s)", True),
               ("e2e_ms",    "End-to-end latency (ms)", True)]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.2), squeeze=False, sharex=True)

    for r, (col, ylabel, _) in enumerate(metrics):
        vals = sub[col].dropna()
        for c, a in enumerate(ATTNS):
            ax = axes[r][c]
            d_attn = sub[sub["attn"] == a]
            for p in ("Discrete", "Continuous"):
                d = d_attn[d_attn["paradigm"] == p].sort_values("n_steps")
                ax.plot(d["n_steps"], d[col], marker=MARKERS[p], color=COLORS[p],
                        label=NICE[p], markersize=4, linewidth=1.4)
            ar = d_attn[d_attn["paradigm"] == "AR"]
            if not ar.empty:
                ax.axhline(float(ar[col].iloc[0]), color=COLORS["AR"],
                           linestyle="--", linewidth=1.2,
                           label="AR baseline (G=128 sequential steps)")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if not vals.empty:
                ax.set_ylim(vals.min() * 0.7, vals.max() * 1.4)
            if r == 0:
                ax.set_title(ATTN_NICE[a])
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("Denoising steps")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Diffusion speed knob: steps vs throughput/latency per attention "
                 "(medium, B=4, G=128)", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out, "fig4_steps")


def fig_dtype(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["group"] == "dtype") & (~df["oom"])]
    sizes = ["medium", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), sharey=True)

    for c, a in enumerate(ATTNS):
        ax = axes[c]
        d_attn = sub[sub["attn"] == a]
        xs = np.arange(len(sizes))
        width = 0.25
        for i, p in enumerate(PARADIGMS):
            speedups = []
            for s in sizes:
                fp = d_attn[(d_attn["paradigm"] == p) & (d_attn["label"] == f"{s}-fp32")]
                bf = d_attn[(d_attn["paradigm"] == p) & (d_attn["label"] == f"{s}-bf16")]
                speedups.append(
                    float(fp["e2e_ms"].iloc[0]) / float(bf["e2e_ms"].iloc[0])
                    if not fp.empty and not bf.empty else np.nan
                )
            ax.bar(xs + (i - 1) * width, speedups, width, color=COLORS[p], label=NICE[p])
        ax.axhline(1.0, color="grey", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.capitalize() for s in sizes])
        ax.set_title(ATTN_NICE[a])
        if c == 0:
            ax.set_ylabel("bf16 e2e speedup (×)")
    axes[0].legend(fontsize=7)
    fig.suptitle("bf16 vs fp32 end-to-end speedup per attention (B=4, G=128)",
                 y=1.04, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "fig5_dtype")


def fig_prefill(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["group"] == "prompt_len") & (~df["oom"])]
    metrics = [("prefill_ms_p50", "Prompt-processing cost (ms)", False),
               ("ttft_ms",        "Time to first token (ms)",    True)]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.2), squeeze=False, sharex=True)

    for r, (col, ylabel, logy) in enumerate(metrics):
        vals = sub[col].dropna()
        for c, a in enumerate(ATTNS):
            ax = axes[r][c]
            d_attn = sub[sub["attn"] == a]
            for p in ("AR", "Discrete"):
                d = d_attn[d_attn["paradigm"] == p].sort_values("prompt_len")
                d = d.dropna(subset=[col])
                if d.empty:
                    continue
                lbl = ("AR prefill" if col == "prefill_ms_p50" and p == "AR"
                       else "Discrete prefix-cache build"
                       if col == "prefill_ms_p50" else NICE[p])
                ax.plot(d["prompt_len"], d[col], marker=MARKERS[p],
                        color=COLORS[p], label=lbl, markersize=4, linewidth=1.4)
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
            if not vals.empty:
                ax.set_ylim((vals.min() * 0.7) if logy else 0, vals.max() * 1.4)
            if r == 0:
                ax.set_title(ATTN_NICE[a])
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("Prompt length P")
    axes[0][0].legend(fontsize=7)
    axes[1][0].legend(fontsize=7)
    fig.suptitle("Prompt conditioning per attention — Discrete+MLA has no prefix cache "
                 "(no raw KV); ELF is unconditional (medium, B=4, G=128)",
                 y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "fig6_prefill")


def fig_steps_ablation(df: pd.DataFrame, out: Path) -> None:
    """Steps ablations: where does the AR crossover sit as G / B / size vary?

    AR baselines (dashed) are taken from the matching rows of the gen_len,
    batch_size and scale groups (MHA partition).
    """
    ok = df[(~df["oom"]) & (df["attn"] == "MHA")]
    ablations = [
        ("steps_x_genlen", "G",    [64, 256, 1024],
         lambda v: ok[(ok.group == "gen_len") & (ok.label == f"G={v}")],
         "fig4b_steps_x_genlen",
         "Steps × generation length (medium, B=4, MHA)"),
        ("steps_x_batch", "B",     [1, 16, 128],
         lambda v: ok[(ok.group == "batch_size") & (ok.label == f"B={v}")],
         "fig4c_steps_x_batch",
         "Steps × batch size (medium, G=128, MHA)"),
        ("steps_x_scale", "size",  ["small", "medium", "xl"],
         lambda v: ok[(ok.group == "scale") & (ok.label == v)],
         "fig4d_steps_x_scale",
         "Steps × model size (B=4, G=128, MHA)"),
    ]

    for group, var, values, ar_rows, fname, title in ablations:
        sub = ok[ok["group"] == group]
        if sub.empty:
            print(f"  (no {group} rows — skipping {fname})")
            continue
        fig, axes = plt.subplots(1, len(values), figsize=(3.4 * len(values), 2.9),
                                 sharey=True)
        vals_all = sub["tok_s_e2e"].dropna()
        for ax, v in zip(axes, values):
            if var == "G":
                d_v = sub[sub["gen_len"] == v]
            elif var == "B":
                d_v = sub[sub["batch_size"] == v]
            else:
                d_v = sub[sub["size"] == v]
            for p in ("Discrete", "Continuous"):
                d = d_v[d_v["paradigm"] == p].sort_values("n_steps")
                ax.plot(d["n_steps"], d["tok_s_e2e"], marker=MARKERS[p],
                        color=COLORS[p], label=NICE[p], markersize=4, linewidth=1.4)
            ar = ar_rows(v)
            ar = ar[ar["paradigm"] == "AR"]
            if not ar.empty:
                ax.axhline(float(ar["tok_s_e2e"].iloc[0]), color=COLORS["AR"],
                           linestyle="--", linewidth=1.2, label="AR baseline")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{var}={v}" if var != "size" else v)
            ax.set_xlabel("Denoising steps")
        axes[0].set_ylabel("Throughput (tok/s)")
        axes[0].legend(fontsize=7)
        fig.suptitle(title, y=1.04, fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, out, fname)


def fig_attention_summary(df: pd.DataFrame, out: Path) -> None:
    """Cross-regime overview bars: paradigm × attention in four regimes."""
    ok = df[~df["oom"]]
    panels = [
        ("batch_size", "B=4",    "(a) Medium, B=4, G=128"),
        ("scale",      "large",  "(b) Large, B=4, G=128"),
        ("gen_len",    "G=512",  "(c) Long generation — G=512"),
        ("batch_size", "B=32",   "(d) Large batch — B=32"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    for ax, (grp, label, title) in zip(axes, panels):
        d = ok[(ok["group"] == grp) & (ok["label"] == label)]
        xs = np.arange(len(PARADIGMS))
        width = 0.25
        for i, a in enumerate(ATTNS):
            vals = []
            for p in PARADIGMS:
                r = d[(d["paradigm"] == p) & (d["attn"] == a)]
                vals.append(float(r["tok_s_e2e"].iloc[0]) if not r.empty else np.nan)
            ax.bar(xs + (i - 1) * width, vals, width,
                   color=[COLORS[p] for p in PARADIGMS],
                   hatch=ATTN_HATCH[a], edgecolor="black", linewidth=0.4,
                   alpha=0.55 + 0.15 * i)
        ax.set_xticks(xs)
        ax.set_xticklabels(["AR", "Discrete", "Continuous"])
        if ax is axes[0]:
            ax.set_ylabel("tok/s (end-to-end)")
        ax.set_title(title)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="lightgrey", edgecolor="black",
                     hatch=ATTN_HATCH[a], label=ATTN_NICE[a]) for a in ATTNS]
    axes[-1].legend(handles=handles, title="Attention", loc="upper right")
    fig.suptitle("Overview: paradigm × attention across regimes "
                 "(hatching = MHA / GQA-1/4 / MLA)", y=1.06, fontsize=10.5)
    _save(fig, out, "fig7_attention_summary")

    # Cache memory companion
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    d = ok[(ok["group"] == "gen_len") & (ok["label"] == "G=512")]
    xs = np.arange(len(ATTNS))
    width = 0.3
    for i, p in enumerate(("AR", "Discrete")):
        vals = []
        for a in ATTNS:
            r = d[(d["paradigm"] == p) & (d["attn"] == a)]
            vals.append(float(r["cache_mb"].iloc[0]) if not r.empty else np.nan)
        ax.bar(xs + (i - 0.5) * width, vals, width, color=COLORS[p],
               label="AR KV-cache" if p == "AR" else "Discrete prefix-cache")
    ax.set_xticks(xs)
    ax.set_xticklabels([ATTN_NICE[a] for a in ATTNS])
    ax.set_ylabel("Cache memory (MB)")
    ax.set_title("Inference cache size by attention (G=512, B=4)\n"
                 "MLA: AR caches absorbed latents; Discrete has no raw KV to cache")
    ax.legend()
    _save(fig, out, "fig7b_cache")


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_md(df: pd.DataFrame, out: Path) -> None:
    ok = df[~df["oom"]]
    lines = [
        "# AR vs Discrete vs Continuous Diffusion — inference benchmark summary",
        "",
        f"Hardware: NVIDIA A100-PCIE-40GB · JAX/XLA · {len(df)} experiments "
        f"({int(df['oom'].sum())} OOM)",
        "",
        "## Headline comparison (medium ~6.6 M backbone, MHA, B=4, G=128, 32 steps, fp32)",
        "",
        "| Paradigm | Params (M) | e2e (ms) | tok/s | TTFT (ms) | GFLOPs/tok | step ms | MFU % | Mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    base = ok[(ok["group"] == "batch_size") & (ok["label"] == "B=4")
              & (ok["attn"] == "MHA")]
    for p in PARADIGMS:
        r = base[base["paradigm"] == p]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(
            f"| {NICE[p]} | {r['params_m']:.2f} | {r['e2e_ms']:.1f} | "
            f"{r['tok_s_e2e']:.0f} | {r['ttft_ms']:.1f} | {r['gflops_per_tok']:.4f} | "
            f"{r['step_ms_p50']:.2f} | {r['mfu_pct']:.2f} | {r['peak_mem_mb']:.0f} |"
        )

    lines += ["", "## Attention variants (medium, B=4, G=128, fp32)", "",
              "| Paradigm | Attention | e2e (ms) | tok/s | step ms | Cache (MB) |",
              "|---|---|---|---|---|---|"]
    b4 = ok[(ok["group"] == "batch_size") & (ok["label"] == "B=4")]
    for p in PARADIGMS:
        for a in ATTNS:
            r = b4[(b4["paradigm"] == p) & (b4["attn"] == a)]
            if r.empty:
                continue
            r = r.iloc[0]
            cache = "—" if pd.isna(r["cache_mb"]) else f"{r['cache_mb']:.1f}"
            lines.append(f"| {NICE[p]} | {ATTN_NICE[a]} | {r['e2e_ms']:.1f} | "
                         f"{r['tok_s_e2e']:.0f} | {r['step_ms_p50']:.2f} | {cache} |")

    lines += ["", "## Best throughput per paradigm (any setting)", "",
              "| Paradigm | Setting | tok/s | e2e ms |", "|---|---|---|---|"]
    for p in PARADIGMS:
        d = ok[(ok["paradigm"] == p)].dropna(subset=["tok_s_e2e"])
        if d.empty:
            continue
        b = d.loc[d["tok_s_e2e"].idxmax()]
        lines.append(f"| {NICE[p]} | {b['group']}/{b['label']} · {b['attn']} "
                     f"(B={b['batch_size']}, G={b['gen_len']}) | "
                     f"{b['tok_s_e2e']:.0f} | {b['e2e_ms']:.1f} |")

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("  saved summary.md")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+",
                        default=["results/paradigm_bench_full.csv"],
                        help="One or more benchmark CSVs (concatenated).")
    parser.add_argument("--out", default="results/paradigm_bench")
    args = parser.parse_args()

    frames = [pd.read_csv(c) for c in args.csv]
    df = pd.concat(frames, ignore_index=True)
    if "attn" not in df.columns:
        df["attn"] = "MHA"
    df["attn"] = df["attn"].fillna("MHA")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {len(df)} rows → {out}/")
    fig_scale(df, out)
    fig_batch(df, out)
    fig_genlen(df, out)
    fig_steps(df, out)
    fig_steps_ablation(df, out)
    fig_dtype(df, out)
    fig_prefill(df, out)
    fig_attention_summary(df, out)
    summary_md(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
