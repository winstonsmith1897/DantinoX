"""
dantinox CLI
============
Entry point registered as the ``dantinox`` command by pyproject.toml.

Subcommands
-----------
  train       Train a model from a config file and a text corpus.
  generate    Generate text from a trained checkpoint.
  sweep       Run a W&B Bayesian hyperparameter sweep.
  benchmark   Benchmark all (or selected) runs in a directory.

Examples
--------
  dantinox train --config configs/default_config.yaml --data_path data/corpus.txt
  dantinox generate --run_dir runs/run_20260101 --prompt "Nel mezzo del cammin "
  dantinox sweep --config configs/sweep.yaml --data_path data/corpus.txt
  dantinox benchmark --runs_dir runs --out_csv results.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from core.config import Config
from dantinox import __version__

# ─── helpers ────────────────────────────────────────────────────────────────

def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    """Add one --<field> flag for every Config field."""
    for field in dataclasses.fields(Config):
        flag = f"--{field.name}"
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, type=type(field.default) if field.default is not dataclasses.MISSING else str, default=None)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Write any non-None CLI overrides onto the config object."""
    for field in dataclasses.fields(Config):
        val = getattr(args, field.name, None)
        if val is not None:
            setattr(config, field.name, val)
    return config


# ─── subcommand handlers ────────────────────────────────────────────────────

def _cmd_train(args: argparse.Namespace) -> None:
    config = Config.from_yaml(args.config)
    config = _apply_overrides(config, args)

    from dantinox.trainer import Trainer
    trainer = Trainer(config)
    run_dir = trainer.fit(
        args.data_path,
        run_dir=getattr(args, "run_dir", None),
        wandb_project=getattr(args, "wandb_project", None),
        resume=getattr(args, "resume", False),
    )
    print(f"\nRun saved to: {run_dir}")


def _cmd_generate(args: argparse.Namespace) -> None:
    import time

    from dantinox.generator import Generator

    gen = Generator(args.run_dir, seed=args.seed)

    print(f"\nRun: {args.run_dir}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    # Warmup / compile pass
    gen.generate(args.prompt, max_new_tokens=1)

    t0 = time.time()
    text = gen.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        use_cache=not args.no_cache,
    )
    elapsed = time.time() - t0

    prompt_tokens = len(gen.tokenizer.encode(args.prompt))
    total_tokens  = len(gen.tokenizer.encode(text))
    new_tokens    = total_tokens - prompt_tokens

    print(text)
    print("-" * 40)
    print(f"Generated {new_tokens} tokens in {elapsed:.2f}s "
          f"({new_tokens / elapsed:.1f} tok/s)")


def _cmd_sweep(args: argparse.Namespace) -> None:
    """Launch a W&B sweep agent using the existing train_sweep entry point."""
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Install it with: pip install wandb", file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(args.sweep_config) as f:
        sweep_cfg = yaml.safe_load(f)

    project = getattr(args, "wandb_project", None) or "DantinoX"
    sweep_id = wandb.sweep(sweep_cfg, project=project)  # type: ignore[attr-defined]
    print(f"Sweep ID: {sweep_id}  (project: {project})")

    def _agent_fn() -> None:
        from dantinox.trainer import Trainer

        run = wandb.init()  # type: ignore[attr-defined]
        wc = dict(run.config)

        base = Config.from_yaml(args.config) if args.config else Config()
        for k, v in wc.items():
            if hasattr(base, k):
                setattr(base, k, v)

        trainer = Trainer(base)
        trainer.fit(args.data_path, wandb_project=None)
        wandb.finish()  # type: ignore[attr-defined]

    wandb.agent(sweep_id, function=_agent_fn, count=getattr(args, "count", None))  # type: ignore[attr-defined]


def _cmd_find_lr(args: argparse.Namespace) -> None:
    config = Config.from_yaml(args.config)
    config = _apply_overrides(config, args)

    from dantinox.trainer import Trainer
    trainer = Trainer(config)
    suggested_lr, lr_hist, loss_hist = trainer.find_lr(
        args.data_path,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps,
    )
    print(f"\nSuggested learning rate: {suggested_lr:.2e}")
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(lr_hist, loss_hist)
            ax.axvline(suggested_lr, color="red", linestyle="--", label=f"suggested: {suggested_lr:.2e}")
            ax.set_xscale("log")
            ax.set_xlabel("Learning rate")
            ax.set_ylabel("Smoothed loss")
            ax.set_title("LR Range Test")
            ax.legend()
            plt.tight_layout()
            out = args.plot_out or "lr_finder.png"
            fig.savefig(out, dpi=150)
            print(f"Plot saved to: {out}")
        except ImportError:
            print("matplotlib not installed — skipping plot (pip install matplotlib)")


def _cmd_push(args: argparse.Namespace) -> None:
    from dantinox.hub import push
    url = push(
        args.run_dir,
        args.repo,
        private=args.private,
        token=args.token,
        commit_message=args.message,
    )
    print(f"Uploaded to: {url}")


def _cmd_pull(args: argparse.Namespace) -> None:
    from dantinox.hub import pull
    run_dir = pull(
        args.repo,
        local_dir=args.local_dir,
        token=args.token,
        revision=args.revision,
    )
    print(f"Downloaded to: {run_dir}")


def _cmd_plot(args: argparse.Namespace) -> None:
    from dantinox.plotting import Plotter
    groups = args.groups if args.groups else None
    plotter = Plotter(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
    )
    results = plotter.run(groups=groups)
    total = sum(len(v) for v in results.values())
    print(f"\nDone — {total} figures written to {args.out_dir}/")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from dantinox.bench import BenchmarkRunner
    runner = BenchmarkRunner(args.runs_dir)
    run_names = args.runs if args.runs else None
    df = runner.run(run_names, out_csv=args.out_csv)

    if not df.empty:
        cols = ["run", "type", "params_m", "theoretical_cache_mb", "prefill_ms"]
        cols = [c for c in cols if c in df.columns]
        print("\n" + df[cols].to_string(index=False))


def _cmd_infbench(args: argparse.Namespace) -> None:
    """Delegate to benchmarks/run_all.py (subprocess keeps JAX state isolated)."""
    import subprocess
    from pathlib import Path

    run_all = Path(__file__).resolve().parent.parent / "benchmarks" / "run_all.py"
    if not run_all.exists():
        print(f"Error: {run_all} not found — is the repo intact?", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(run_all),
           "--out-csv", args.out_csv,
           "--out-dir", args.out_dir,
           "--n-warmup", str(args.n_warmup),
           "--n-trials", str(args.n_trials)]
    if args.groups:
        cmd += ["--groups"] + args.groups
    if getattr(args, "device", None):
        cmd += ["--device", args.device]
    if args.sweep_only:
        cmd += ["--sweep-only"]
    if args.plot_only:
        cmd += ["--plot-only"]
    if args.verbose:
        cmd += ["--verbose"]
    if getattr(args, "trained", False):
        cmd += ["--trained"]
    if getattr(args, "inference_off", False):
        cmd += ["--inference-off"]
    if getattr(args, "runs_dir", None):
        cmd += ["--runs-dir", args.runs_dir]
    if getattr(args, "trained_csv", None):
        cmd += ["--trained-csv", args.trained_csv]
    if getattr(args, "trained_plot", None):
        cmd += ["--trained-plot", args.trained_plot]
    if getattr(args, "batch_csv", None):
        cmd += ["--batch-csv", args.batch_csv]
    if getattr(args, "batch_sizes", None):
        cmd += ["--batch-sizes"] + [str(b) for b in args.batch_sizes]
    if getattr(args, "batch_seq_len", None):
        cmd += ["--batch-seq-len", str(args.batch_seq_len)]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ─── argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dantinox",
        description="DantinoX — JAX/Flax Transformer library CLI",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", default="configs/default_config.yaml",
                         help="Path to a YAML config file")
    p_train.add_argument("--data_path", help="Path to the training corpus")
    p_train.add_argument("--run_dir", help="Output run directory (auto-generated if omitted)")
    p_train.add_argument("--wandb_project", help="W&B project name for logging")
    p_train.add_argument("--resume", action="store_true",
                         help="Resume training from the last checkpoint in --run_dir")
    _add_config_overrides(p_train)

    # ── generate ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint")
    p_gen.add_argument("--run_dir", required=True, help="Run directory with config + weights")
    p_gen.add_argument("--prompt", default="Nel mezzo del cammin ", help="Input prompt")
    p_gen.add_argument("--max_new_tokens", type=int, default=150)
    p_gen.add_argument("--greedy", action="store_true")
    p_gen.add_argument("--top_k", type=int, default=None)
    p_gen.add_argument("--top_p", type=float, default=None)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--no_cache", action="store_true", help="Disable KV cache")
    p_gen.add_argument("--seed", type=int, default=42)

    # ── sweep ──────────────────────────────────────────────────────────────
    p_sweep = sub.add_parser("sweep", help="Run a W&B hyperparameter sweep")
    p_sweep.add_argument("--sweep_config", default="configs/sweep.yaml",
                         help="W&B sweep YAML configuration")
    p_sweep.add_argument("--config", default="configs/default_config.yaml",
                         help="Base model config (overridden by sweep params)")
    p_sweep.add_argument("--data_path", required=True, help="Path to the training corpus")
    p_sweep.add_argument("--wandb_project", default="DantinoX")
    p_sweep.add_argument("--count", type=int, default=None,
                         help="Maximum number of sweep runs (default: unlimited)")

    # ── benchmark ──────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Benchmark run directories")
    p_bench.add_argument("--runs_dir", default="runs", help="Directory containing run sub-dirs")
    p_bench.add_argument("--runs", nargs="*", help="Specific run names to benchmark (default: all)")
    p_bench.add_argument("--out_csv", default=None, help="Write results to this CSV file")

    # ── find-lr ────────────────────────────────────────────────────────────────
    p_flr = sub.add_parser("find-lr", help="Run the LR range test and suggest a learning rate")
    p_flr.add_argument("--config", default="configs/default_config.yaml",
                       help="Path to a YAML config file")
    p_flr.add_argument("--data_path", required=True, help="Path to the training corpus")
    p_flr.add_argument("--min_lr", type=float, default=1e-7, help="Start LR (default 1e-7)")
    p_flr.add_argument("--max_lr", type=float, default=1.0, help="End LR (default 1.0)")
    p_flr.add_argument("--num_steps", type=int, default=100, help="Sweep steps (default 100)")
    p_flr.add_argument("--plot", action="store_true", help="Save a loss-vs-LR PNG")
    p_flr.add_argument("--plot_out", default=None, help="Output PNG path (default: lr_finder.png)")
    _add_config_overrides(p_flr)

    # ── push ───────────────────────────────────────────────────────────────────
    p_push = sub.add_parser("push", help="Upload a checkpoint to HuggingFace Hub")
    p_push.add_argument("--run_dir", required=True, help="Local run directory to upload")
    p_push.add_argument("--repo", required=True, help="Hub repo id (e.g. my-org/my-model)")
    p_push.add_argument("--private", action="store_true", help="Create a private repository")
    p_push.add_argument("--token", default=None, help="HuggingFace access token")
    p_push.add_argument("--message", default=None, help="Commit message")

    # ── pull ───────────────────────────────────────────────────────────────────
    p_pull = sub.add_parser("pull", help="Download a checkpoint from HuggingFace Hub")
    p_pull.add_argument("--repo", required=True, help="Hub repo id (e.g. my-org/my-model)")
    p_pull.add_argument("--local_dir", default=None, help="Where to save the files")
    p_pull.add_argument("--token", default=None, help="HuggingFace access token")
    p_pull.add_argument("--revision", default=None, help="Branch, tag, or commit SHA")

    # ── infbench ───────────────────────────────────────────────────────────────
    p_ib = sub.add_parser(
        "infbench",
        help="Run the full benchmark suite: inference sweep + optional trained-model pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Benchmark pipeline:\n"
            "  Stage 1  benchmarks/inference_sweep.py   →  CSV          (random-model sweep)\n"
            "  Stage 2  benchmarks/plot_inference.py    →  21 PNG plots\n"
            "  Stage 3  benchmarks/trained_analysis.py  →  CSV          (real trained runs)\n"
            "  Stage 4  benchmarks/trained_batch_sweep.py→  CSV         (tps vs batch size)\n\n"
            "Stages 3-4 only run when --trained is passed.\n\n"
            "Examples:\n"
            "  dantinox infbench\n"
            "  dantinox infbench --trained\n"
            "  dantinox infbench --trained --inference-off\n"
            "  dantinox infbench --groups attention_type scale --n-trials 5\n"
            "  dantinox infbench --plot-only --out-csv results/inference_sweep.csv\n"
            "  dantinox infbench --device 1"
        ),
    )
    p_ib.add_argument("--out-csv", default="results/inference_sweep.csv", metavar="PATH",
                      help="CSV output path (default: results/inference_sweep.csv)")
    p_ib.add_argument("--out-dir", default="results/plots/", metavar="DIR",
                      help="Directory for plot PNGs (default: results/plots/)")
    p_ib.add_argument("--groups", nargs="+", metavar="GROUP",
                      help="Restrict sweep to these groups (default: all 13)")
    p_ib.add_argument("--n-warmup", type=int, default=3, metavar="N",
                      help="Warm-up reps per experiment (default: 3)")
    p_ib.add_argument("--n-trials", type=int, default=10, metavar="N",
                      help="Measured reps per experiment (default: 10)")
    p_ib.add_argument("--device", default=None, metavar="N",
                      help="CUDA device index for CUDA_VISIBLE_DEVICES (default: env)")
    p_ib.add_argument("--sweep-only", action="store_true",
                      help="Run sweep only, skip plotting")
    p_ib.add_argument("--plot-only", action="store_true",
                      help="Skip sweep, re-plot existing --out-csv")
    p_ib.add_argument("--verbose", action="store_true",
                      help="Print per-experiment metrics during the sweep")
    p_ib.add_argument("--trained", action="store_true",
                      help="Also run the trained-model pipeline (stages 3 and 4)")
    p_ib.add_argument("--inference-off", action="store_true",
                      help="Skip inference pipeline; requires --trained")
    p_ib.add_argument("--runs-dir", default="runs", metavar="DIR",
                      help="Directory of trained run subdirs (default: runs)")
    p_ib.add_argument("--trained-csv", default="results/benchmark_results.csv", metavar="PATH",
                      help="Output CSV for trained-model analysis (default: results/benchmark_results.csv)")
    p_ib.add_argument("--trained-plot", default="results/plots/trained_analysis.png", metavar="PATH",
                      help="Output PNG for trained-model analysis")
    p_ib.add_argument("--batch-csv", default="results/batch_sweep_results.csv", metavar="PATH",
                      help="Output CSV for batch sweep (default: results/batch_sweep_results.csv)")
    p_ib.add_argument("--batch-sizes", nargs="+", type=int, metavar="N",
                      help="Batch sizes for the batch sweep (default: 1 2 4 8 16 32 64)")
    p_ib.add_argument("--batch-seq-len", type=int, default=512, metavar="N",
                      help="Fixed sequence length for the batch sweep (default: 512)")

    # ── plot ────────────────────────────────────────────────────────────────
    p_plot = sub.add_parser("plot", help="Generate benchmark plots from a results CSV")
    p_plot.add_argument("--in_csv", default="benchmark_results.csv",
                        help="Benchmark results CSV (output of 'dantinox benchmark')")
    p_plot.add_argument("--out_dir", default="plots",
                        help="Output directory for PNG files (default: plots/)")
    p_plot.add_argument("--batch_csv", default=None,
                        help="Optional batch sweep CSV for the batch throughput figure")
    p_plot.add_argument("--groups", nargs="*",
                        metavar="GROUP",
                        help="Plot groups to generate: insights perf 3d 3d_dkv (default: all)")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "train":     _cmd_train,
        "generate":  _cmd_generate,
        "sweep":     _cmd_sweep,
        "benchmark": _cmd_benchmark,
        "infbench":  _cmd_infbench,
        "plot":      _cmd_plot,
        "find-lr":   _cmd_find_lr,
        "push":      _cmd_push,
        "pull":      _cmd_pull,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
