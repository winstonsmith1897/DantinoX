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
import sys

from core.config import Config

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


def _cmd_plot(args: argparse.Namespace) -> None:
    from dantinox.plotting import Plotter
    groups = args.groups if args.groups else None
    plotter = Plotter(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        batch_csv=args.batch_csv,
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


# ─── argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dantinox",
        description="DantinoX — JAX/Flax Transformer library CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", default="configs/default_config.yaml",
                         help="Path to a YAML config file")
    p_train.add_argument("--data_path", help="Path to the training corpus")
    p_train.add_argument("--run_dir", help="Output run directory (auto-generated if omitted)")
    p_train.add_argument("--wandb_project", help="W&B project name for logging")
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
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "train":     _cmd_train,
        "generate":  _cmd_generate,
        "sweep":     _cmd_sweep,
        "benchmark": _cmd_benchmark,
        "plot":      _cmd_plot,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
