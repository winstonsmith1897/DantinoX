#!/usr/bin/env python3
"""
benchmarks/plot_paradigm_ablations.py
=====================================

Publication figures for the v3 paradigm ablations.  Design rule: every
figure asserts ONE claim, stated in the title and annotated on the data —
crossovers marked, regimes labelled, redundant panels merged.

  figE_pareto_{arch}    Serving frontier: diffusion owns the low-latency
                        regime, AR wins at batch saturation (crossover
                        annotated, SLO guide line).
  figA_parity_{arch}    Step budget S* at AR parity + speedup@32 in one row
                        (Discrete shown; Continuous ≡ within ±5%, noted).
  figB_roofline_{arch}  Mechanism: AR decode memory-bound vs diffusion
                        compute-bound (clusters annotated, MFU ticks).
  figF_energy_{arch}    mJ/token: who is cheaper, where (minima annotated).
  figG_precision        TF32/bf16 vs true fp32 at saturation (bars, deltas).
  figC_stack_{arch}     Serving-stack waterfall (unchanged).
  figD_ceiling          Memory ceiling (horizontal annotated bars).
  ablation_summary_{arch}.md

CSV conventions: incremental crash-safe files — last row per key wins.
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

PARADIGMS = ["AR", "Discrete", "Continuous"]
COLORS  = {"AR": "#1f77b4", "Discrete": "#d62728", "Continuous": "#2ca02c"}
LIGHT   = {"AR": "#9ecae1", "Discrete": "#fc9272", "Continuous": "#a1d99b"}
MARKERS = {"AR": "o", "Discrete": "s", "Continuous": "^"}
NICE    = {"AR": "Autoregressive", "Discrete": "Discrete Diff. (LLaDA)",
           "Continuous": "Continuous Diff. (ELF)"}

PEAK_BF16 = 312e12
HBM_BW    = 1.555e12

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
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
    key = [k for k in key if k in df.columns]
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


# ══ figE: serving Pareto frontier ══════════════════════════════════════════════

PARETO_STYLE = {
    "AR (greedy)":     dict(color=COLORS["AR"], ls="-",  lw=2.0, marker="o", ms=5),
    "Discrete S=32":   dict(color=COLORS["Discrete"], ls="-",  lw=1.6, marker="s", ms=4),
    "Discrete S=8":    dict(color=LIGHT["Discrete"], ls="--", lw=1.4, marker="s", ms=4),
    "Continuous S=32": dict(color=COLORS["Continuous"], ls="-",  lw=1.6, marker="^", ms=4),
    "Continuous S=8":  dict(color=LIGHT["Continuous"], ls="--", lw=1.4, marker="^", ms=4),
}


def _crossover_b(d_ar: pd.DataFrame, d_other: pd.DataFrame) -> float | None:
    """Batch size where AR's system throughput first exceeds the other's."""
    merged = pd.merge(d_ar[["batch_size", "tok_s_system"]],
                      d_other[["batch_size", "tok_s_system"]],
                      on="batch_size", suffixes=("_ar", "_o")).sort_values("batch_size")
    for _, r in merged.iterrows():
        if r["tok_s_system_ar"] >= r["tok_s_system_o"]:
            return float(r["batch_size"])
    return None


def fig_pareto(par: pd.DataFrame, out: Path, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    if d_all.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for label, st in PARETO_STYLE.items():
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["e2e_ms_med"], d["tok_s_system"], label=label,
                zorder=3, **st)
        for _, r in d.iterrows():
            if r["batch_size"] in (1, 256) or r["batch_size"] == d["batch_size"].max():
                ax.annotate(f"B={int(r['batch_size'])}",
                            (r["e2e_ms_med"], r["tok_s_system"]),
                            textcoords="offset points", xytext=(5, -9),
                            fontsize=6.5, color=st["color"])

    # Crossover annotation: AR vs Discrete S=32
    d_ar = d_all[d_all.label == "AR (greedy)"]
    d_d32 = d_all[d_all.label == "Discrete S=32"]
    if not d_ar.empty and not d_d32.empty:
        bx = _crossover_b(d_ar, d_d32)
        if bx:
            r = d_ar[d_ar.batch_size == bx]
            if not r.empty:
                x, y = float(r["e2e_ms_med"].iloc[0]), float(r["tok_s_system"].iloc[0])
                ax.annotate(f"AR overtakes Discrete S=32\nat B={int(bx)}",
                            (x, y), textcoords="offset points", xytext=(12, -34),
                            fontsize=8, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", lw=1.0))

    # Interactive-SLO guide
    ax.axvline(200, color="grey", lw=0.8, ls="--", zorder=1)
    ax.text(200, 0.03, " 200 ms SLO", fontsize=7, color="grey", rotation=90,
            va="bottom", transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Per-request latency (ms, full generation of G=256 tokens)")
    ax.set_ylabel("System throughput (tok/s)")
    ax.set_title(f"Serving frontier — {arch}: diffusion owns the low-latency "
                 f"regime,\nAR wins once batching saturates the GPU "
                 f"(bf16, fused, A100)")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, out, f"figE_pareto_{arch}")


# ══ figA: parity heatmaps (merged Discrete; Continuous noted) ═════════════════

def fig_parity(grid: pd.DataFrame, out: Path, arch: str) -> None:
    g = grid[(grid.arch == arch)]
    Bs = sorted(g["batch_size"].unique())
    Gs = sorted(g["gen_len"].unique())

    def matrix(paradigm: str, col: str) -> np.ndarray:
        m = np.full((len(Bs), len(Gs)), np.nan)
        for i, b in enumerate(Bs):
            for j, gl in enumerate(Gs):
                r = g[(g.paradigm == paradigm) & (g.batch_size == b)
                      & (g.gen_len == gl) & (~g.oom)]
                if not r.empty and pd.notna(r[col].iloc[0]):
                    m[i, j] = float(r[col].iloc[0])
        return m

    m_s = matrix("Discrete", "parity_steps")
    m_x = matrix("Discrete", "speedup_at_32")
    m_s_c = matrix("Continuous", "parity_steps")
    dev = (np.nanmax(np.abs(m_s - m_s_c) / np.maximum(m_s, 1)) * 100
           if np.isfinite(m_s).any() and np.isfinite(m_s_c).any() else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    panels = [(m_s, "(a) Step budget S* at AR-parity latency", "{:.0f}", "RdYlGn"),
              (m_x, "(b) Realised speedup vs AR at S=32", "{:.1f}×", "RdYlGn")]
    for ax, (m, title, fmt, cmap) in zip(axes, panels):
        norm = (matplotlib.colors.LogNorm(vmin=max(np.nanmin(m), 0.2),
                                          vmax=np.nanmax(m))
                if np.isfinite(m).any() else None)
        im = ax.imshow(m, cmap=cmap, aspect="auto", norm=norm)
        for i in range(len(Bs)):
            for j in range(len(Gs)):
                if np.isnan(m[i, j]):
                    ax.text(j, i, "OOM", ha="center", va="center", fontsize=8,
                            color="dimgrey", style="italic")
                else:
                    ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                            fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(Gs)), [f"G={gl}" for gl in Gs])
        ax.set_yticks(range(len(Bs)), [f"B={b}" for b in Bs])
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)

    note = (f"Continuous ≡ Discrete within {dev:.0f}%"
            if dev == dev else "Continuous data pending")
    fig.suptitle(f"When can diffusion afford its steps? — {arch}, bf16, A100.  "
                 f"Long single-stream generation: yes (S*≈60); "
                 f"batched serving: no (S*≈4).\n[{note} — equivalence of the "
                 f"two diffusion paradigms is itself a finding]",
                 y=1.06, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, f"figA_parity_{arch}")


# ══ figB: roofline with annotated regimes ═════════════════════════════════════

def fig_roofline(grid: pd.DataFrame, out: Path, arch: str) -> None:
    g = grid[(grid.arch == arch) & (~grid.oom)].dropna(
        subset=["step_gflops", "step_gbytes", "step_ms_med"])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    xs = np.logspace(-1, 4, 200)
    ax.plot(xs, np.minimum(xs * HBM_BW, PEAK_BF16) / 1e12, color="black", lw=1.5)
    ridge = PEAK_BF16 / HBM_BW
    for frac, lbl in ((1.0, "bf16 peak 312 TF/s"), (0.1, "10% peak"),
                      (0.01, "1% peak")):
        ax.axhline(PEAK_BF16 * frac / 1e12, color="grey", lw=0.5, ls=":")
        ax.text(1.3e-1, PEAK_BF16 * frac / 1e12 * 1.15, lbl, fontsize=6.5,
                color="grey")

    for p in PARADIGMS:
        d = g[g.paradigm == p].copy()
        if d.empty:
            continue
        d["intensity"] = d.step_gflops / d.step_gbytes
        d["achieved"] = d.step_gflops / d.step_ms_med
        d = d.sort_values("intensity")
        ax.plot(d["intensity"], d["achieved"], color=COLORS[p], lw=0.7,
                alpha=0.5, zorder=2)
        ax.scatter(d["intensity"], d["achieved"],
                   s=18 + 4 * np.log2(d.batch_size * d.gen_len),
                   color=COLORS[p], marker=MARKERS[p], alpha=0.9,
                   edgecolor="black", linewidth=0.4, label=NICE[p], zorder=3)

    # Regime annotations
    d_ar = g[g.paradigm == "AR"]
    if not d_ar.empty:
        ax.annotate("AR decode:\nmemory-bound\n(reads all weights+cache\nper 1 token)",
                    (float((d_ar.step_gflops / d_ar.step_gbytes).median()),
                     float((d_ar.step_gflops / d_ar.step_ms_med).median())),
                    textcoords="offset points", xytext=(-10, 38), fontsize=7.5,
                    ha="right", arrowprops=dict(arrowstyle="->", lw=0.8))
    d_df = g[g.paradigm != "AR"]
    if not d_df.empty:
        best = (d_df.step_gflops / d_df.step_ms_med).max()
        ax.annotate(f"diffusion steps: compute-bound\nbest {best:.0f} TF/s "
                    f"= {100 * best * 1e12 / PEAK_BF16:.0f}% of peak",
                    (float((d_df.step_gflops / d_df.step_gbytes).median()), best),
                    textcoords="offset points", xytext=(-120, 14), fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e4)
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte, analytical)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title(f"Why diffusion can spend 17× more FLOPs and still win — "
                 f"{arch} (A100 roofline,\none decode/denoise step; "
                 f"marker size ∝ tokens in flight)")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, out, f"figB_roofline_{arch}")


# ══ figF: energy per token ════════════════════════════════════════════════════

def fig_energy(par: pd.DataFrame, out: Path, arch: str) -> None:
    d_all = par[(par.arch == arch) & (~par.oom) & (par.dtype == "bf16")]
    d_all = d_all[d_all["j_per_tok"].notna() & (d_all["j_per_tok"] > 0)]
    if d_all.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    minima = []
    for label, st in PARETO_STYLE.items():
        d = d_all[d_all.label == label].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(d["batch_size"], d["j_per_tok"] * 1e3, label=label, **st)
        i = d["j_per_tok"].idxmin()
        minima.append((label, float(d.loc[i, "batch_size"]),
                       float(d.loc[i, "j_per_tok"]) * 1e3, st["color"]))
    for label, b, v, c in minima:
        ax.annotate(f"{v:.1f} mJ", (b, v), textcoords="offset points",
                    xytext=(4, -11), fontsize=7, color=c, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Energy per generated token (mJ, idle-subtracted)")
    ax.set_title(f"Energy — {arch} (NVML, G=256): batching amortises everyone;"
                 f"\nminima annotated — fewer denoising steps ⇒ "
                 f"proportionally cheaper tokens")
    ax.legend(fontsize=7.5)
    _save(fig, out, f"figF_energy_{arch}")


# ══ figG: precision study (bars at saturation) ════════════════════════════════

def fig_precision(par: pd.DataFrame, out: Path) -> None:
    B_REF = 64
    d_all = par[(par.arch == "512d12b") & (~par.oom)
                & (par.batch_size == B_REF)]
    precisions = [p for p in ("f32", "tf32", "bf16") if (d_all.dtype == p).any()]
    if len(precisions) < 2:
        return
    labels = ["AR (greedy)", "Discrete S=32", "Continuous S=32"]
    pcolors = {"f32": "#999999", "tf32": "#ff7f0e", "bf16": "#9467bd"}

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(labels))
    width = 0.8 / len(precisions)
    base_vals: dict[str, float] = {}
    for i, prec in enumerate(precisions):
        vals = []
        for lbl in labels:
            r = d_all[(d_all.label == lbl) & (d_all.dtype == prec)]
            vals.append(float(r["tok_s_system"].iloc[0]) if not r.empty else np.nan)
        bars = ax.bar(xs + (i - (len(precisions) - 1) / 2) * width, vals, width,
                      color=pcolors[prec], edgecolor="black", linewidth=0.4,
                      label={"f32": "true FP32", "tf32": "TF32 (JAX default)",
                             "bf16": "BF16"}[prec])
        for j, (lbl, v) in enumerate(zip(labels, vals)):
            if prec == precisions[0]:
                base_vals[lbl] = v
            if v == v:
                mult = v / base_vals[lbl] if base_vals.get(lbl) else np.nan
                ax.text(xs[j] + (i - (len(precisions) - 1) / 2) * width, v,
                        f"{mult:.1f}×" if mult == mult else "",
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs, labels)
    ax.set_ylabel(f"System throughput (tok/s) @ B={B_REF}, G=256")
    ax.set_title("Diffusion is tensor-core-bound: TF32/bf16 give ~3× over true "
                 "fp32;\nAR decode barely benefits (512d12b, A100) — "
                 "× relative to true FP32")
    ax.legend(fontsize=7.5)
    _save(fig, out, "figG_precision")


# ══ figC: serving-stack waterfall ═════════════════════════════════════════════

def fig_stack(stack: pd.DataFrame, out: Path, arch: str) -> None:
    d_all = stack[(stack.arch == arch) & (~stack.oom)]
    if d_all.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels, values, colors = [], [], []
    for p in PARADIGMS:
        d = d_all[d_all.paradigm == p]
        for _, r in d.iterrows():
            labels.append(r["variant"])
            values.append(r["tok_s_e2e"])
            colors.append(COLORS[p])
        if not d.empty:
            labels.append("")
            values.append(0)
            colors.append("white")
    if labels and labels[-1] == "":
        labels, values, colors = labels[:-1], values[:-1], colors[:-1]
    ys = np.arange(len(labels))
    ax.barh(ys, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end throughput (tok/s) — B=4, G=128, S=32, fused")
    base = None
    for y, (lbl, v) in enumerate(zip(labels, values)):
        if lbl == "":
            base = None
            continue
        if base is None:
            base = v
        if v > 0:
            ax.text(v, y, f"  {v:,.0f}  ({v / base:.2f}×)", va="center", fontsize=8)
    ax.set_title(f"What each serving optimisation buys — {arch} (A100)")
    _save(fig, out, f"figC_stack_{arch}")


# ══ figD: memory ceiling ══════════════════════════════════════════════════════

def fig_ceiling(ceil: pd.DataFrame, out: Path) -> None:
    archs = sorted(ceil["arch"].unique())
    attns = ["MHA", "GQA", "MLA"]
    fig, axes = plt.subplots(1, len(archs), figsize=(3.5 * len(archs), 3.4),
                             sharey=True, squeeze=False)
    for ax, arch in zip(axes[0], archs):
        d_arch = ceil[ceil.arch == arch]
        xs = np.arange(len(attns))
        width = 0.26
        for i, p in enumerate(PARADIGMS):
            vals = []
            for a in attns:
                r = d_arch[(d_arch.paradigm == p) & (d_arch.attn == a)]
                vals.append(float(r["batch"].max()) if not r.empty else np.nan)
            b = ax.bar(xs + (i - 1) * width, vals, width, color=COLORS[p],
                       edgecolor="black", linewidth=0.4,
                       label=NICE[p] if arch == archs[0] else None)
            ax.bar_label(b, fmt="%.0f", fontsize=6.5, padding=1)
        ax.set_xticks(xs, attns)
        ax.set_yscale("log", base=2)
        ax.set_title(arch, fontsize=9.5)
    axes[0][0].set_ylabel("Max concurrent batch (G=512)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Serving capacity per A100-40GB (bf16, ×2 search resolution): "
                 "AR is KV-cache-bound → MLA lifts it 8×;\nDiscrete is bound by "
                 "per-step B×G×V logits → identical ceiling everywhere",
                 y=1.08, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "figD_ceiling")


# ── Summary tables ─────────────────────────────────────────────────────────────

def summary(grid, par, ceil, out: Path, arch: str) -> None:
    lines = [f"# v3 ablation summary — {arch} (vocab 32k, fused loops, bf16, A100)", ""]
    if grid is not None:
        g = grid[(grid.arch == arch) & (~grid.oom)]
        if not g.empty:
            lines += ["## Parity step budget S* (AR-parity denoising steps)", "",
                      "| B | G | S* Disc | S* Cont | AR e2e ms | Disc J/tok | AR J/tok |",
                      "|---|---|---|---|---|---|---|"]
            for (b, gl), sub in g.groupby(["batch_size", "gen_len"]):
                def _v(p, c, fmt="{:.0f}"):
                    r = sub[sub.paradigm == p]
                    return (fmt.format(float(r[c].iloc[0]))
                            if not r.empty and pd.notna(r[c].iloc[0]) else "—")
                lines.append(
                    f"| {b} | {gl} | {_v('Discrete', 'parity_steps')} | "
                    f"{_v('Continuous', 'parity_steps')} | {_v('AR', 'e2e_ms_med')} | "
                    f"{_v('Discrete', 'j_per_tok', '{:.4f}')} | "
                    f"{_v('AR', 'j_per_tok', '{:.4f}')} |")
            lines.append("")
    if ceil is not None:
        c = ceil[ceil.arch == arch]
        if not c.empty:
            lines += ["## Memory ceiling (max batch, G=512, dedicated GPU)", "",
                      "| Paradigm | MHA | GQA | MLA |", "|---|---|---|---|"]
            for p in PARADIGMS:
                row = [p]
                for a in ("MHA", "GQA", "MLA"):
                    r = c[(c.paradigm == p) & (c.attn == a)]
                    row.append(str(int(r["batch"].max())) if not r.empty else "—")
                lines.append("| " + " | ".join(row) + " |")
    (out / f"ablation_summary_{arch}.md").write_text("\n".join(lines) + "\n")
    print(f"  saved ablation_summary_{arch}.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/paradigm_bench")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = _load("results/ablation_grid_*.csv",
                 ["arch", "paradigm", "batch_size", "gen_len", "dtype", "tp"])
    par  = _load("results/ablation_pareto_*.csv",
                 ["arch", "label", "batch_size", "dtype"])
    stk  = _load("results/ablation_stack_*.csv", ["arch", "variant"])
    ceil = _load("results/ablation_ceiling_*.csv",
                 ["arch", "paradigm", "attn", "batch"])

    archs = sorted(set(
        ([] if grid is None else list(grid.arch.unique()))
        + ([] if par is None else list(par.arch.unique()))))
    print(f"Architectures: {archs}")

    for arch in archs:
        if grid is not None:
            fig_parity(grid, out, arch)
            fig_roofline(grid, out, arch)
        if stk is not None:
            fig_stack(stk, out, arch)
        if par is not None:
            fig_pareto(par, out, arch)
            fig_energy(par, out, arch)
        summary(grid, par, ceil, out, arch)
    if ceil is not None:
        fig_ceiling(ceil, out)
    if par is not None:
        fig_precision(par, out)
    print("Done.")


if __name__ == "__main__":
    main()
