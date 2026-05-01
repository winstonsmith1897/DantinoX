#!/usr/bin/env python3
"""
benchmarks/run_all.py — DantinoX end-to-end benchmark suite
=============================================================

Two independent pipelines, each run as isolated subprocesses:

  ── Inference pipeline (random models) ─────────────────────────────
  Stage 1  inference_sweep.py    →  <out-csv>          (data collection)
  Stage 2  plot_inference.py     →  <out-dir>/*.png    (21 figures)

  ── Trained-model pipeline (real runs/) ────────────────────────────
  Stage 3  trained_analysis.py   →  <trained-csv>      (per-run metrics)
  Stage 4  trained_batch_sweep.py→  <batch-csv>        (tps vs batch size)

Each stage is a subprocess so JAX is initialised fresh with the right
CUDA device and no compiled-function conflicts between stages.

Usage
-----
  # Full inference pipeline (default)
  python benchmarks/run_all.py

  # Also run the trained-model pipeline
  python benchmarks/run_all.py --trained

  # Trained pipeline only
  python benchmarks/run_all.py --trained --inference-off

  # Re-plot from an existing inference CSV, skip the sweep
  python benchmarks/run_all.py --plot-only

  # Collect inference data only, no plots
  python benchmarks/run_all.py --sweep-only

  # Restrict the sweep to a subset of groups (faster iteration)
  python benchmarks/run_all.py --groups attention_type scale batch_size

  # Quick smoke-test
  python benchmarks/run_all.py --n-warmup 1 --n-trials 3

  # Select GPU (default: CUDA_VISIBLE_DEVICES env, fallback 0)
  python benchmarks/run_all.py --device 2

  # Via the CLI
  dantinox infbench --groups scale --n-trials 5
  dantinox infbench --trained

  # Via Makefile
  make infbench
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


# ─── entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="run_all",
        description="DantinoX end-to-end benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io = parser.add_argument_group("inference pipeline — output paths")
    io.add_argument(
        "--out-csv", default="results/inference_sweep.csv", metavar="PATH",
        help="CSV written by the sweep stage (default: results/inference_sweep.csv)",
    )
    io.add_argument(
        "--out-dir", default="results/plots/", metavar="DIR",
        help="Directory for inference plot PNGs (default: results/plots/)",
    )

    sweep = parser.add_argument_group("inference pipeline — sweep options")
    sweep.add_argument(
        "--groups", nargs="+", metavar="GROUP",
        help="Restrict sweep to these experiment groups (default: all 13)",
    )
    sweep.add_argument(
        "--n-warmup", type=int, default=3, metavar="N",
        help="JIT warm-up repetitions per experiment (default: 3)",
    )
    sweep.add_argument(
        "--n-trials", type=int, default=10, metavar="N",
        help="Timed repetitions per experiment (default: 10)",
    )

    tr = parser.add_argument_group("trained-model pipeline")
    tr.add_argument(
        "--trained", action="store_true",
        help="Also run the trained-model pipeline (stages 3 and 4)",
    )
    tr.add_argument(
        "--inference-off", action="store_true",
        help="Skip the inference pipeline (stages 1 and 2); requires --trained",
    )
    tr.add_argument(
        "--runs-dir", default="runs", metavar="DIR",
        help="Directory containing trained run subdirectories (default: runs)",
    )
    tr.add_argument(
        "--trained-csv", default="results/benchmark_results.csv", metavar="PATH",
        help="Output CSV for trained-model analysis (default: results/benchmark_results.csv)",
    )
    tr.add_argument(
        "--trained-plot", default="results/plots/trained_analysis.png", metavar="PATH",
        help="Output PNG for trained-model analysis (default: results/plots/trained_analysis.png)",
    )
    tr.add_argument(
        "--batch-csv", default="results/batch_sweep_results.csv", metavar="PATH",
        help="Output CSV for batch sweep (default: results/batch_sweep_results.csv)",
    )
    tr.add_argument(
        "--batch-sizes", nargs="+", type=int, metavar="N",
        help="Batch sizes for the batch sweep (default: 1 2 4 8 16 32 64)",
    )
    tr.add_argument(
        "--batch-seq-len", type=int, default=512, metavar="N",
        help="Fixed sequence length for the batch sweep (default: 512)",
    )

    ctl = parser.add_argument_group("pipeline control")
    ctl.add_argument(
        "--device", default=None, metavar="N",
        help="CUDA device index — sets CUDA_VISIBLE_DEVICES for all stages "
             "(default: inherit from environment)",
    )
    ctl.add_argument(
        "--sweep-only", action="store_true",
        help="Inference pipeline: collect data only, skip plot generation",
    )
    ctl.add_argument(
        "--plot-only", action="store_true",
        help="Inference pipeline: skip sweep, re-plot an existing --out-csv",
    )
    ctl.add_argument(
        "--verbose", action="store_true",
        help="Print per-experiment metrics during the inference sweep",
    )

    args = parser.parse_args(argv)

    # Validation
    if args.sweep_only and args.plot_only:
        parser.error("--sweep-only and --plot-only are mutually exclusive")
    if args.inference_off and not args.trained:
        parser.error("--inference-off requires --trained")
    if args.plot_only and not Path(args.out_csv).exists():
        parser.error(
            f"--plot-only requires an existing CSV.\n"
            f"  Not found: {args.out_csv}\n"
            f"  Run without --plot-only first to generate it."
        )

    # Count stages for banner labels
    run_inference = not args.inference_off
    run_trained   = args.trained
    n_stages = (
        (0 if args.plot_only else 1) + (0 if args.sweep_only else 1)
        if run_inference else 0
    ) + (2 if run_trained else 0)

    t_total = time.perf_counter()
    print(f"\nDantinoX benchmark suite")
    if run_inference:
        print(f"  inference CSV  : {args.out_csv}")
        print(f"  inference plots: {args.out_dir}")
        if args.groups:
            print(f"  groups         : {', '.join(args.groups)}")
    if run_trained:
        print(f"  trained CSV    : {args.trained_csv}")
        print(f"  batch-sweep CSV: {args.batch_csv}")
        print(f"  runs dir       : {args.runs_dir}")

    stage_idx = 1

    if run_inference:
        if not args.plot_only:
            stage_sweep(args, n_stages, stage_idx)
            stage_idx += 1
        if not args.sweep_only:
            stage_plot(args, n_stages, stage_idx)
            stage_idx += 1

    if run_trained:
        stage_trained_analysis(args, n_stages, stage_idx)
        stage_idx += 1
        stage_batch_sweep(args, n_stages, stage_idx)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'═' * 60}")
    print(f"  All done  ({elapsed:.1f}s total)")
    if run_inference:
        if not args.plot_only:
            print(f"  Inference CSV  : {args.out_csv}")
        if not args.sweep_only:
            print(f"  Inference plots: {args.out_dir}  ({_count_plots(args.out_dir)} figures)")
    if run_trained:
        print(f"  Trained CSV    : {args.trained_csv}")
        print(f"  Batch-sweep CSV: {args.batch_csv}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
