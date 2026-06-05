#!/usr/bin/env python3
"""
benchmarks/run_all.py — DantinoX end-to-end benchmark suite
=============================================================

Four independent pipelines, each run as isolated subprocesses:

  ── Inference pipeline (random AR models) ──────────────────────────
  Stage 1  inference_sweep.py      →  <out-csv>               (data)
  Stage 2  plot_inference.py       →  <out-dir>/*.png         (21 figures)

  ── AR vs Diffusion pipeline (random models) ───────────────────────
  Stage 3  diffusion_ar_sweep.py   →  <diff-ar-csv>           (data)
  Stage 4  plot_diffusion_ar.py    →  <out-dir>/*.png         (20 figures)

  ── Trained-model pipeline (real runs/) ────────────────────────────
  Stage 5  trained_analysis.py     →  <trained-csv>           (per-run)
  Stage 6  trained_batch_sweep.py  →  <batch-csv>             (tps vs bs)

  ── Evaluation pipeline (quality metrics on trained runs/) ─────────
  Stage 7  perplexity_eval.py      →  <ppl-csv>               (PPL)
  Stage 8  confidence_sweep.py     →  <confidence-csv>        (τ/f sweep)
  Stage 9  generation_quality.py   →  <gen-quality-csv>       (diversity)
  Stage 10 plot_emnlp.py           →  <paper-dir>/*.png       (paper figs)

Stages 3–4   run when --diff-ar   is passed.
Stages 5–6   run when --trained   is passed.
Stages 7–10  run when --eval      is passed (implies --trained).

Usage
-----
  # Inference pipeline only (default)
  python benchmarks/run_all.py

  # Add AR vs Diffusion comparison
  python benchmarks/run_all.py --diff-ar

  # Add trained-model analysis
  python benchmarks/run_all.py --trained

  # Full evaluation pipeline (all stages)
  python benchmarks/run_all.py --diff-ar --eval

  # Skip inference; run only evaluation on trained runs
  python benchmarks/run_all.py --eval --inference-off

  # Quick smoke-test
  python benchmarks/run_all.py --n-warmup 1 --n-trials 3

  # Via the CLI
  dantinox infbench --diff-ar --eval
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # benchmarks/
_ROOT = _HERE.parent                      # repo root


# ─── helpers ─────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a subprocess; propagate non-zero exit codes."""
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged, cwd=str(_ROOT))
    if result.returncode != 0:
        print(
            f"\n[run_all] command failed (exit {result.returncode}):\n"
            f"  {' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def _count_plots(out_dir: str) -> int:
    p = Path(out_dir)
    return len(list(p.glob("*.png"))) if p.exists() else 0


# ─── inference pipeline stages ───────────────────────────────────────────────

def stage_sweep(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  inference sweep  (data collection)")
    cmd = [
        sys.executable,
        str(_HERE / "inference_sweep.py"),
        "--out",       args.out_csv,
        "--n-warmup",  str(args.n_warmup),
        "--n-trials",  str(args.n_trials),
    ]
    if args.groups:
        cmd += ["--groups"] + args.groups
    if args.verbose:
        cmd += ["--verbose"]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  sweep done in {time.perf_counter() - t0:.1f}s  →  {args.out_csv}")


def stage_plot(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  plot generation  (21 figures)")
    cmd = [
        sys.executable,
        str(_HERE / "plot_inference.py"),
        "--csv",     args.out_csv,
        "--out-dir", args.out_dir,
    ]
    t0 = time.perf_counter()
    _run(cmd)
    n = _count_plots(args.out_dir)
    print(f"\n  plotting done in {time.perf_counter() - t0:.1f}s  →  {args.out_dir}  ({n} figures)")


# ─── trained-model pipeline stages ───────────────────────────────────────────

def stage_trained_analysis(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  trained-model analysis  (per-run metrics)")
    cmd = [
        sys.executable,
        str(_HERE / "trained_analysis.py"),
        "--runs-dir", args.runs_dir,
        "--out-csv",  args.trained_csv,
        "--out-plot", args.trained_plot,
    ]
    if args.device is not None:
        cmd += ["--device", args.device]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  analysis done in {time.perf_counter() - t0:.1f}s  →  {args.trained_csv}")


def stage_batch_sweep(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  batch sweep  (tps vs batch size)")
    cmd = [
        sys.executable,
        str(_HERE / "trained_batch_sweep.py"),
        "--runs-dir",     args.runs_dir,
        "--out-csv",      args.batch_csv,
        "--seq-len",      str(args.batch_seq_len),
        "--analysis-csv", args.trained_csv,
    ]
    if args.batch_sizes:
        cmd += ["--batch-sizes"] + [str(b) for b in args.batch_sizes]
    if args.device is not None:
        cmd += ["--device", args.device]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  batch sweep done in {time.perf_counter() - t0:.1f}s  →  {args.batch_csv}")


# ─── AR vs Diffusion pipeline stages ─────────────────────────────────────────

def stage_diff_ar_sweep(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  AR vs Diffusion sweep  (data collection)")
    cmd = [
        sys.executable,
        str(_HERE / "diffusion_ar_sweep.py"),
        "--out",       args.diff_ar_csv,
        "--n-warmup",  str(args.n_warmup),
        "--n-trials",  str(args.n_trials),
    ]
    if getattr(args, "diff_ar_groups", None):
        cmd += ["--groups"] + args.diff_ar_groups
    if getattr(args, "no_mla", False):
        cmd += ["--no-mla"]
    if args.verbose:
        cmd += ["--verbose"]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  diff-ar sweep done in {time.perf_counter() - t0:.1f}s  →  {args.diff_ar_csv}")


def stage_diff_ar_plot(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  AR vs Diffusion plots  (20 figures)")
    cmd = [
        sys.executable,
        str(_HERE / "plot_diffusion_ar.py"),
        "--csv",     args.diff_ar_csv,
        "--out-dir", args.out_dir,
    ]
    t0 = time.perf_counter()
    _run(cmd)
    n = _count_plots(args.out_dir)
    print(f"\n  diff-ar plotting done in {time.perf_counter() - t0:.1f}s  →  {args.out_dir}  ({n} figures)")


# ─── Evaluation pipeline stages ──────────────────────────────────────────────

def stage_perplexity(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  perplexity evaluation  (WikiText-103 / PTB / C4 / LAMBADA)")
    cmd = [
        sys.executable,
        str(_HERE / "perplexity_eval.py"),
        "--runs-dir",    args.runs_dir,
        "--out",         args.ppl_csv,
        "--max-windows", str(getattr(args, "ppl_max_windows", 200)),
    ]
    if getattr(args, "ppl_datasets", None):
        cmd += ["--datasets"] + args.ppl_datasets
    if getattr(args, "local_text", None):
        cmd += ["--local-text", args.local_text]
    if args.device is not None:
        cmd += ["--device", args.device]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  perplexity eval done in {time.perf_counter() - t0:.1f}s  →  {args.ppl_csv}")


def stage_confidence_sweep(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  confidence sweep  (τ/f × MHA/GQA/MLA)")
    cmd = [
        sys.executable,
        str(_HERE / "confidence_sweep.py"),
        "--out",       args.confidence_csv,
        "--n-runs",    str(getattr(args, "conf_n_runs", 30)),
        "--n-warmup",  str(args.n_warmup),
        "--n-measure", str(args.n_trials),
    ]
    if getattr(args, "no_mla", False):
        cmd += ["--no-mla"]
    if args.device is not None:
        cmd += ["--device", args.device]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  confidence sweep done in {time.perf_counter() - t0:.1f}s  →  {args.confidence_csv}")


def stage_gen_quality(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  generation quality  (Distinct / Self-BLEU / Rep-4)")
    cmd = [
        sys.executable,
        str(_HERE / "generation_quality.py"),
        "--runs-dir",  args.runs_dir,
        "--out",       args.gen_quality_csv,
        "--n-samples", str(getattr(args, "genq_n_samples", 100)),
        "--gen-len",   str(getattr(args, "genq_gen_len", 128)),
    ]
    if args.device is not None:
        cmd += ["--device", args.device]

    env = {"CUDA_VISIBLE_DEVICES": args.device} if args.device is not None else {}
    t0 = time.perf_counter()
    _run(cmd, env=env)
    print(f"\n  gen quality done in {time.perf_counter() - t0:.1f}s  →  {args.gen_quality_csv}")


def stage_emnlp_plots(args: argparse.Namespace, n_stages: int, stage_idx: int) -> None:
    _banner(f"STAGE {stage_idx} / {n_stages}  —  EMNLP paper figures  (8 figures)")
    cmd = [
        sys.executable,
        str(_HERE / "plot_emnlp.py"),
        "--out-dir",          args.paper_dir,
        "--ppl-csv",          args.ppl_csv,
        "--trained-csv",      args.trained_csv,
        "--diffusion-ar-csv", args.diff_ar_csv,
        "--confidence-csv",   args.confidence_csv,
        "--gen-quality-csv",  args.gen_quality_csv,
    ]
    if getattr(args, "pdf", False):
        cmd += ["--pdf"]
    t0 = time.perf_counter()
    _run(cmd)
    n = _count_plots(args.paper_dir)
    print(f"\n  EMNLP figures done in {time.perf_counter() - t0:.1f}s  →  {args.paper_dir}  ({n} figures)")


# ─── entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="run_all",
        description="DantinoX end-to-end benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── inference pipeline ────────────────────────────────────────────────────
    io = parser.add_argument_group("inference pipeline — output paths")
    io.add_argument("--out-csv",  default="results/inference_sweep.csv", metavar="PATH")
    io.add_argument("--out-dir",  default="results/plots/",              metavar="DIR")

    sw = parser.add_argument_group("inference pipeline — sweep options")
    sw.add_argument("--groups",    nargs="+",  metavar="GROUP")
    sw.add_argument("--n-warmup",  type=int,   default=3,  metavar="N")
    sw.add_argument("--n-trials",  type=int,   default=10, metavar="N")

    # ── AR vs Diffusion pipeline ──────────────────────────────────────────────
    da = parser.add_argument_group("AR vs Diffusion pipeline")
    da.add_argument("--diff-ar", action="store_true",
                    help="Run the AR vs Diffusion sweep (stages 3–4)")
    da.add_argument("--diff-ar-csv", default="results/diffusion_ar_sweep.csv", metavar="PATH")
    da.add_argument("--diff-ar-groups", nargs="+", metavar="GROUP",
                    help="Restrict diffusion_ar_sweep to these groups (default: all)")
    da.add_argument("--no-mla", action="store_true",
                    help="Skip MLA experiments in diffusion_ar_sweep and confidence_sweep")

    # ── trained-model pipeline ────────────────────────────────────────────────
    tr = parser.add_argument_group("trained-model pipeline")
    tr.add_argument("--trained", action="store_true",
                    help="Run trained-model analysis (stages 5–6)")
    tr.add_argument("--runs-dir",     default="runs",                               metavar="DIR")
    tr.add_argument("--trained-csv",  default="results/benchmark_results.csv",      metavar="PATH")
    tr.add_argument("--trained-plot", default="results/plots/trained_analysis.png", metavar="PATH")
    tr.add_argument("--batch-csv",    default="results/batch_sweep_results.csv",    metavar="PATH")
    tr.add_argument("--batch-sizes",  nargs="+", type=int,  metavar="N")
    tr.add_argument("--batch-seq-len", type=int, default=512, metavar="N")

    # ── evaluation pipeline ───────────────────────────────────────────────────
    ev = parser.add_argument_group("evaluation pipeline (--eval)")
    ev.add_argument("--eval", action="store_true",
                    help="Run the full quality evaluation pipeline (stages 7–10; implies --trained)")
    ev.add_argument("--ppl-csv",          default="results/perplexity.csv",       metavar="PATH")
    ev.add_argument("--confidence-csv",   default="results/confidence_sweep.csv", metavar="PATH")
    ev.add_argument("--gen-quality-csv",  default="results/generation_quality.csv", metavar="PATH")
    ev.add_argument("--paper-dir",        default="results/paper_figures/",       metavar="DIR")
    ev.add_argument("--ppl-datasets",     nargs="+", metavar="DS",
                    help="Datasets for perplexity eval (default: all available)")
    ev.add_argument("--ppl-max-windows",  type=int, default=200, metavar="N")
    ev.add_argument("--local-text",       default=None, metavar="PATH",
                    help="Local text file as fallback when HF datasets are unavailable")
    ev.add_argument("--conf-n-runs",      type=int, default=30, metavar="N",
                    help="Trajectories per confidence config (default: 30)")
    ev.add_argument("--genq-n-samples",   type=int, default=100, metavar="N")
    ev.add_argument("--genq-gen-len",     type=int, default=128, metavar="N")
    ev.add_argument("--pdf", action="store_true",
                    help="Save EMNLP figures as PDF in addition to PNG")

    # ── pipeline control ──────────────────────────────────────────────────────
    ctl = parser.add_argument_group("pipeline control")
    ctl.add_argument("--device",       default=None, metavar="N")
    ctl.add_argument("--sweep-only",   action="store_true")
    ctl.add_argument("--plot-only",    action="store_true")
    ctl.add_argument("--inference-off", action="store_true")
    ctl.add_argument("--verbose",      action="store_true")

    args = parser.parse_args(argv)

    # --eval implies --trained (need trained models for perplexity / gen quality)
    if args.eval:
        args.trained = True

    # Validation
    if args.sweep_only and args.plot_only:
        parser.error("--sweep-only and --plot-only are mutually exclusive")
    if args.inference_off and not (args.trained or args.diff_ar or args.eval):
        parser.error("--inference-off requires at least one of --trained, --diff-ar, --eval")
    if args.plot_only and not Path(args.out_csv).exists():
        parser.error(
            f"--plot-only requires an existing inference CSV.\n"
            f"  Not found: {args.out_csv}"
        )

    # Count active stages
    run_inference = not args.inference_off
    run_diff_ar   = args.diff_ar
    run_trained   = args.trained
    run_eval      = args.eval

    n_stages = 0
    if run_inference:
        n_stages += (0 if args.plot_only else 1) + (0 if args.sweep_only else 1)
    if run_diff_ar:
        n_stages += 2
    if run_trained:
        n_stages += 2
    if run_eval:
        n_stages += 4  # ppl + confidence + gen_quality + emnlp_plots

    t_total = time.perf_counter()
    print("\nDantinoX benchmark suite")
    if run_inference:
        print(f"  inference CSV     : {args.out_csv}")
        print(f"  inference plots   : {args.out_dir}")
    if run_diff_ar:
        print(f"  diff-ar CSV       : {args.diff_ar_csv}")
    if run_trained:
        print(f"  trained CSV       : {args.trained_csv}")
        print(f"  batch-sweep CSV   : {args.batch_csv}")
        print(f"  runs dir          : {args.runs_dir}")
    if run_eval:
        print(f"  perplexity CSV    : {args.ppl_csv}")
        print(f"  confidence CSV    : {args.confidence_csv}")
        print(f"  gen-quality CSV   : {args.gen_quality_csv}")
        print(f"  paper figures dir : {args.paper_dir}")

    stage_idx = 1

    # ── Stage 1–2: inference pipeline ────────────────────────────────────────
    if run_inference:
        if not args.plot_only:
            stage_sweep(args, n_stages, stage_idx)
            stage_idx += 1
        if not args.sweep_only:
            stage_plot(args, n_stages, stage_idx)
            stage_idx += 1

    # ── Stage 3–4: AR vs Diffusion pipeline ──────────────────────────────────
    if run_diff_ar:
        stage_diff_ar_sweep(args, n_stages, stage_idx)
        stage_idx += 1
        stage_diff_ar_plot(args, n_stages, stage_idx)
        stage_idx += 1

    # ── Stage 5–6: trained-model pipeline ────────────────────────────────────
    if run_trained:
        stage_trained_analysis(args, n_stages, stage_idx)
        stage_idx += 1
        stage_batch_sweep(args, n_stages, stage_idx)
        stage_idx += 1

    # ── Stage 7–10: evaluation pipeline ──────────────────────────────────────
    if run_eval:
        stage_perplexity(args, n_stages, stage_idx)
        stage_idx += 1
        stage_confidence_sweep(args, n_stages, stage_idx)
        stage_idx += 1
        stage_gen_quality(args, n_stages, stage_idx)
        stage_idx += 1
        stage_emnlp_plots(args, n_stages, stage_idx)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'═' * 60}")
    print(f"  All done  ({elapsed:.1f}s total)")
    if run_inference and not args.plot_only:
        print(f"  Inference CSV     : {args.out_csv}")
    if run_inference and not args.sweep_only:
        print(f"  Inference plots   : {args.out_dir}  ({_count_plots(args.out_dir)} figures)")
    if run_diff_ar:
        print(f"  Diff-AR CSV       : {args.diff_ar_csv}")
    if run_trained:
        print(f"  Trained CSV       : {args.trained_csv}")
        print(f"  Batch-sweep CSV   : {args.batch_csv}")
    if run_eval:
        print(f"  Perplexity CSV    : {args.ppl_csv}")
        print(f"  Confidence CSV    : {args.confidence_csv}")
        print(f"  Gen-quality CSV   : {args.gen_quality_csv}")
        print(f"  Paper figures     : {args.paper_dir}  ({_count_plots(args.paper_dir)} figures)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
