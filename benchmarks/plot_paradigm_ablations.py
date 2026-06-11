#!/usr/bin/env python3
"""
benchmarks/plot_paradigm_ablations.py
=====================================

Figures for the EMNLP architectural ablations (``paradigm_ablations.py``),
run on the production architecture (512d × 12 blocks, vocab 32 128, bf16,
fully fused generation loops for all paradigms).

  figA_parity_map.pdf   Heatmaps over (B × G): denoising-step budget S* at
                        which diffusion matches AR fused end-to-end latency,
                        and realised speedup at S=32.  OOM cells hatched.
  figB_roofline.pdf     A100 roofline: arithmetic intensity (FLOPs/byte,
                        XLA-measured) vs achieved TFLOP/s per step function.
                        AR decode sits on the memory roof; diffusion steps
                        climb toward the compute roof as B×G grows.
  figC_stack.pdf        Serving-stack waterfall: marginal tok/s contribution
                        of each inference optimisation per paradigm.
  figD_ceiling.pdf      Max concurrent batch per paradigm × attention on one
                        A100-40GB (G=512, bf16) + steady tok/s at the ceiling.
  ablation_summary.md   Headline numbers.

Usage:
  python benchmarks/plot_paradigm_ablations.py \\
      --grid results/ablation_grid.csv \\
      --stack results/ablation_stack.csv \\
      --ceiling results/ablation_ceiling.csv \\
      --out results/paradigm_bench
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS  = {"AR": "#1f77b4", "Discrete": "#d62728", "Continuous": "#2ca02c"}
MARKERS = {"AR": "o", "Discrete": "s", "Continuous": "^"}
NICE    = {"AR": "Autoregressive", "Discrete": "Discrete Diffusion (LLaDA)",
           "Continuous": "Continuous Diffusion (ELF)"}

PEAK_BF16 = 312e12        # A100 bf16 tensor-core FLOP/s
HBM_BW    = 1.555e12      # A100 HBM2e bytes/s

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf")


# ── A. Parity map ──────────────────────────────────────────────────────────────

def fig_parity(grid: pd.DataFrame, out: Path) -> None:
    Bs = sorted(grid["batch_size"].unique())
    Gs = sorted(grid["gen_len"].unique())

    def matrix(paradigm: str, col: str) -> np.ndarray:
        m = np.full((len(Bs), len(Gs)), np.nan)
        for i, b in enumerate(Bs):
            for j, g in enumerate(Gs):
                r = grid[(grid.paradigm == paradigm) & (grid.batch_size == b)
                         & (grid.gen_len == g) & (~grid.oom)]
                if not r.empty:
                    m[i, j] = float(r[col].iloc[0])
        return m

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    panels = [("Discrete", "parity_steps", "(a) Discrete — step budget S* at AR parity"),
              ("Continuous", "parity_steps", "(b) Continuous — step budget S* at AR parity"),
              ("Discrete", "speedup_at_32", "(c) Discrete — speedup vs AR at S=32"),
              ("Continuous", "speedup_at_32", "(d) Continuous — speedup vs AR at S=32")]

    for ax, (p, col, title) in zip(axes.flat, panels):
        m = matrix(p, col)
        im = ax.imshow(m, cmap="RdYlGn", aspect="auto",
                       norm=matplotlib.colors.LogNorm(
                           vmin=max(np.nanmin(m), 0.5), vmax=np.nanmax(m))
                       if np.isfinite(m).any() else None)
        for i in range(len(Bs)):
            for j in range(len(Gs)):
                if np.isnan(m[i, j]):
                    ax.text(j, i, "OOM", ha="center", va="center",
                            fontsize=8, color="dimgrey", style="italic")
                else:
                    v = m[i, j]
                    ax.text(j, i, f"{v:.0f}" if col == "parity_steps" else f"{v:.1f}×",
                            ha="center", va="center", fontsize=9,
                            fontweight="bold", color="black")
        ax.set_xticks(range(len(Gs)))
        ax.set_xticklabels([f"G={g}" for g in Gs])
        ax.set_yticks(range(len(Bs)))
        ax.set_yticklabels([f"B={b}" for b in Bs])
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Parity with fused AR generation — production arch (512d×12b, "
                 "vocab 32k, bf16, A100)\nS*: denoising steps diffusion can spend "
                 "and still match AR end-to-end latency", y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "figA_parity_map")


# ── B. Roofline ────────────────────────────────────────────────────────────────

def fig_roofline(grid: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Roof: memory slope and bf16 compute ceiling
    xs = np.logspace(-1, 4, 200)
    roof = np.minimum(xs * HBM_BW, PEAK_BF16)
    ax.plot(xs, roof / 1e12, color="black", linewidth=1.4)
    ridge = PEAK_BF16 / HBM_BW
    ax.axvline(ridge, color="grey", linewidth=0.7, linestyle=":")
    ax.text(ridge * 1.1, 0.02, f"ridge ≈ {ridge:.0f} FLOP/B",
            fontsize=7, color="grey")

    ok = grid[~grid.oom].dropna(subset=["step_gflops", "step_gbytes", "step_ms_p50"])
    for p in ("AR", "Discrete", "Continuous"):
        d = ok[ok.paradigm == p]
        if d.empty:
            continue
        intensity = d.step_gflops / d.step_gbytes
        achieved  = d.step_gflops / d.step_ms_p50 / 1e3        # TFLOP/s
        size = 18 + 4 * np.log2(d.batch_size * d.gen_len)
        ax.scatter(intensity, achieved, s=size, color=COLORS[p],
                   marker=MARKERS[p], alpha=0.8, edgecolor="black",
                   linewidth=0.4, label=NICE[p], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte, XLA-measured)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title("A100 roofline — one decode/denoise step, production arch (bf16)\n"
                 "marker size ∝ tokens in flight (B×G); AR decode is pinned to "
                 "the memory roof", fontsize=9)
    ax.legend(loc="upper left")
    _save(fig, out, "figB_roofline")


# ── C. Serving-stack waterfall ────────────────────────────────────────────────

def fig_stack(stack: pd.DataFrame, out: Path) -> None:
    ok = stack[~stack.oom].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    labels, values, colors = [], [], []
    for p in ("AR", "Discrete", "Continuous"):
        d = ok[ok.paradigm == p]
        for _, r in d.iterrows():
            labels.append(r["variant"])
            values.append(r["tok_s_e2e"])
            colors.append(COLORS[p])
        labels.append("")          # spacer
        values.append(0)
        colors.append("white")
    if labels and labels[-1] == "":
        labels, values, colors = labels[:-1], values[:-1], colors[:-1]

    ys = np.arange(len(labels))
    ax.barh(ys, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end throughput (tok/s) — B=4, G=128, S=32, fused")

    # Annotate multiplier vs the first (baseline) bar of each paradigm
    base = None
    for y, (lbl, v) in enumerate(zip(labels, values)):
        if lbl == "":
            base = None
            continue
        if base is None:
            base = v
        if v > 0:
            ax.text(v, y, f"  {v:,.0f}  ({v / base:.2f}×)",
                    va="center", fontsize=8)

    ax.set_title("Serving-stack ablation — marginal effect of each inference "
                 "optimisation (production arch, A100)", fontsize=9.5)
    _save(fig, out, "figC_stack")


# ── D. Memory ceiling ─────────────────────────────────────────────────────────

def fig_ceiling(ceil: pd.DataFrame, out: Path) -> None:
    attns = ["MHA", "GQA", "MLA"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

    xs = np.arange(len(attns))
    width = 0.25
    for i, p in enumerate(("AR", "Discrete", "Continuous")):
        mb, ts = [], []
        for a in attns:
            r = ceil[(ceil.paradigm == p) & (ceil.attn == a)]
            mb.append(float(r["max_batch"].iloc[0]) if not r.empty else np.nan)
            ts.append(float(r["tok_s_at_max"].iloc[0]) if not r.empty else np.nan)
        b = axes[0].bar(xs + (i - 1) * width, mb, width, color=COLORS[p], label=NICE[p])
        axes[0].bar_label(b, fmt="%.0f", fontsize=7, padding=1)
        b = axes[1].bar(xs + (i - 1) * width, ts, width, color=COLORS[p])
        axes[1].bar_label(b, fmt="%.0f", fontsize=6, padding=1, rotation=90)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(attns)
        ax.set_yscale("log", base=2)
    axes[0].set_ylabel("Max concurrent batch (G=512)")
    axes[0].set_title("(a) Memory ceiling on one A100-40GB")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Steady tok/s at the ceiling")
    axes[1].set_title("(b) Throughput at the ceiling (Diff @ S=32)")

    fig.suptitle("Serving capacity — AR bounded by KV-cache, diffusion by "
                 "per-step activations/logits (bf16)", y=1.04, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "figD_ceiling")


# ── Summary ────────────────────────────────────────────────────────────────────

def summary(grid: pd.DataFrame | None, stack: pd.DataFrame | None,
            ceil: pd.DataFrame | None, out: Path) -> None:
    lines = ["# Architectural ablations — production arch (512d×12b, vocab 32k, "
             "bf16, fused loops, A100)", ""]
    if grid is not None:
        ok = grid[~grid.oom]
        lines += ["## Parity step budget S* (denoising steps affordable at AR-parity latency)", "",
                  "| B | G | S* Discrete | S* Continuous | AR e2e (ms) |", "|---|---|---|---|---|"]
        for (b, g), sub in ok.groupby(["batch_size", "gen_len"]):
            ar = sub[sub.paradigm == "AR"]
            di = sub[sub.paradigm == "Discrete"]
            co = sub[sub.paradigm == "Continuous"]
            lines.append(
                f"| {b} | {g} | "
                + (f"{di.parity_steps.iloc[0]:.0f}" if not di.empty else "—") + " | "
                + (f"{co.parity_steps.iloc[0]:.0f}" if not co.empty else "—") + " | "
                + (f"{ar.e2e_ms.iloc[0]:.0f}" if not ar.empty else "—") + " |")
        lines.append("")
    if stack is not None:
        ok = stack[~stack.oom]
        lines += ["## Serving stack (tok/s, B=4, G=128, S=32)", "",
                  "| Variant | tok/s |", "|---|---|"]
        for _, r in ok.iterrows():
            lines.append(f"| {r['variant']} | {r['tok_s_e2e']:,.0f} |")
        lines.append("")
    if ceil is not None:
        lines += ["## Memory ceiling (max batch, G=512, bf16)", "",
                  "| Paradigm | MHA | GQA | MLA |", "|---|---|---|---|"]
        for p in ("AR", "Discrete", "Continuous"):
            row = [p]
            for a in ("MHA", "GQA", "MLA"):
                r = ceil[(ceil.paradigm == p) & (ceil.attn == a)]
                row.append(f"{int(r['max_batch'].iloc[0])}" if not r.empty else "—")
            lines.append("| " + " | ".join(row) + " |")
    (out / "ablation_summary.md").write_text("\n".join(lines) + "\n")
    print("  saved ablation_summary.md")


def _load_many(pattern: str) -> pd.DataFrame | None:
    """Concatenate all per-arch CSVs matching a glob; None if none exist."""
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if "arch" not in df.columns:
        df["arch"] = "512d12b"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="results/ablation_grid_*.csv")
    parser.add_argument("--stack", default="results/ablation_stack_*.csv")
    parser.add_argument("--ceiling", default="results/ablation_ceiling_*.csv")
    parser.add_argument("--out", default="results/paradigm_bench")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = _load_many(args.grid)
    stack = _load_many(args.stack)
    ceil = _load_many(args.ceiling)

    archs = sorted(set(
        ([] if grid is None else list(grid["arch"].unique()))
        + ([] if stack is None else list(stack["arch"].unique()))
        + ([] if ceil is None else list(ceil["arch"].unique()))
    ))
    print(f"Architectures found: {archs}")

    for arch in archs:
        sub_out = out
        suffix = f"_{arch}"
        g = grid[grid["arch"] == arch] if grid is not None else None
        s = stack[stack["arch"] == arch] if stack is not None else None
        c = ceil[ceil["arch"] == arch] if ceil is not None else None

        # Temporarily wrap _save to append the arch suffix
        global _save
        orig_save = _save

        def _save(fig, o, name, _orig=orig_save, _sfx=suffix):  # type: ignore[no-redef]
            _orig(fig, o, name + _sfx)

        try:
            if g is not None and not g.empty:
                fig_parity(g, sub_out)
                fig_roofline(g, sub_out)
            if s is not None and not s.empty:
                fig_stack(s, sub_out)
            if c is not None and not c.empty:
                fig_ceiling(c, sub_out)
        finally:
            _save = orig_save

        summary(g, s, c, sub_out)
        sm = sub_out / "ablation_summary.md"
        if sm.exists():
            sm.rename(sub_out / f"ablation_summary{suffix}.md")
    print("Done.")


if __name__ == "__main__":
    main()
