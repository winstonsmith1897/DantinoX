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
from pathlib import Path

from core.config import Config
from dantinox import __version__


# Persistent XLA compilation cache — compiled GPU kernels are saved to disk so
# subsequent calls with the same model architecture skip recompilation entirely.
# Must be set before any JAX operation (lazy import keeps this safe).
def _init_jax_cache() -> None:
    import jax
    _cache = Path.home() / ".cache" / "jax_xla" / "dantinox"
    _cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_cache))

# ─── helpers ────────────────────────────────────────────────────────────────

def _str2bool(v: str) -> bool:
    """Convert 'true'/'false'/'1'/'0'/'yes'/'no' to bool (argparse type helper)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "on"):
        return True
    if v.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    """Add one --<field> flag for every Config field.

    Bool fields use _str2bool so that --flag true/false/1/0 all work correctly.
    (argparse's default type=bool treats any non-empty string as True.)
    """
    for field in dataclasses.fields(Config):
        flag = f"--{field.name}"
        if flag in parser._option_string_actions:
            continue
        if field.default is dataclasses.MISSING:
            parser.add_argument(flag, type=str, default=None)
        elif isinstance(field.default, bool):
            parser.add_argument(flag, type=_str2bool, default=None, metavar="BOOL")
        else:
            parser.add_argument(flag, type=type(field.default), default=None)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Write any non-None CLI overrides onto the config object."""
    for field in dataclasses.fields(Config):
        val = getattr(args, field.name, None)
        if val is not None:
            setattr(config, field.name, val)
    return config


# ─── subcommand handlers ────────────────────────────────────────────────────

def _cmd_train(args: argparse.Namespace) -> None:
    _init_jax_cache()   # persist XLA-compiled kernels across runs
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
    _init_jax_cache()
    import os
    import time
    import yaml

    # Read config.yaml to determine model_type
    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as _f:
        _raw = yaml.safe_load(_f)
    _flat: dict = {}
    for _v in _raw.values():
        if isinstance(_v, dict):
            _flat.update(_v)
    if not _flat:
        _flat = _raw
    model_type = _flat.get("model_type", "autoregressive")

    print(f"\nRun: {args.run_dir}")
    print(f"Model type: {model_type}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    if model_type == "autoregressive":
        from dantinox.generator import Generator

        gen = Generator(args.run_dir, seed=args.seed)

        sampling = dict(
            greedy=args.greedy,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        if args.stream:
            print(args.prompt, end="", flush=True)
            t0 = time.time()
            n = 0
            for chunk in gen.stream(args.prompt, max_new_tokens=args.max_new_tokens, **sampling):
                print(chunk, end="", flush=True)
                n += 1
            elapsed = time.time() - t0
            print(f"\n{'-' * 40}")
            print(f"Generated {n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")
        else:
            gen.generate(args.prompt, max_new_tokens=1)
            t0 = time.time()
            text = gen.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_cache,
                **sampling,
            )
            elapsed = time.time() - t0
            prompt_tokens = len(gen.tokenizer.encode(args.prompt))
            new_tokens    = len(gen.tokenizer.encode(text)) - prompt_tokens
            print(text)
            print("-" * 40)
            print(f"Generated {new_tokens} tokens in {elapsed:.2f}s "
                  f"({new_tokens / elapsed:.1f} tok/s)")

    elif model_type == "diffusion":
        import msgpack
        import jax.numpy as jnp
        from flax import nnx
        from flax.serialization import _msgpack_ext_unpack
        from core.config import Config
        from core.diffusion import make_noise_schedule
        from core.model import DiffusionTransformer
        from core.generation import diffusion_generate, fast_dllm_generate
        from transformers import AutoTokenizer

        cfg = Config.from_dict(_flat)
        for _fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
            _wp = os.path.join(args.run_dir, _fname)
            if os.path.exists(_wp):
                break
        else:
            print(f"Error: no weights file found in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        with open(_wp, "rb") as _f:
            _state = msgpack.unpackb(_f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
        _model = DiffusionTransformer(cfg, rngs=nnx.Rngs(args.seed))
        nnx.update(_model, _state)

        tokenizer_type = _flat.get("tokenizer_type", "")
        tokenizer_path = _flat.get("tokenizer_path", None)
        if tokenizer_type == "bpe" and tokenizer_path:
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            _tokenizer = AutoTokenizer.from_pretrained("t5-base")

        _schedule = make_noise_schedule(cfg)
        _prefix = jnp.zeros((1, 0), dtype=jnp.int32)

        t0 = time.time()
        _out = fast_dllm_generate(
            _model,
            _prefix,
            args.max_new_tokens,
            _schedule,
            mask_token_id=cfg.mask_token_id,
            block_size=args.block_size,
            steps_per_block=args.n_steps,
            confidence_threshold=args.confidence_threshold,
            use_dual_cache=args.use_dual_cache,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        token_ids = _out[0].tolist()
        text = _tokenizer.decode(token_ids, skip_special_tokens=True)
        print(text)
        print("-" * 40)
        print(f"Generated {len(token_ids)} tokens in {elapsed:.2f}s "
              f"({len(token_ids) / elapsed:.1f} tok/s)")

    elif model_type == "elf":
        import msgpack
        from flax import nnx
        from flax.serialization import _msgpack_ext_unpack
        from core.config import Config
        from core.elf import ELFTransformer
        from core.generation import elf_generate
        from transformers import AutoTokenizer

        cfg = Config.from_dict(_flat)
        for _fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
            _wp = os.path.join(args.run_dir, _fname)
            if os.path.exists(_wp):
                break
        else:
            print(f"Error: no weights file found in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        with open(_wp, "rb") as _f:
            _state = msgpack.unpackb(_f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
        _model = ELFTransformer(cfg.to_elf_config(), rngs=nnx.Rngs(args.seed))
        nnx.update(_model, _state)

        tokenizer_type = _flat.get("tokenizer_type", "")
        tokenizer_path = _flat.get("tokenizer_path", None)
        if tokenizer_type == "bpe" and tokenizer_path:
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            _tokenizer = AutoTokenizer.from_pretrained("t5-base")

        t0 = time.time()
        _out = elf_generate(
            _model,
            gen_len=args.max_new_tokens,
            batch_size=1,
            n_steps=args.n_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        token_ids = _out[0].tolist()
        text = _tokenizer.decode(token_ids, skip_special_tokens=True)
        print(text)
        print("-" * 40)
        print(f"Generated {len(token_ids)} tokens in {elapsed:.2f}s "
              f"({len(token_ids) / elapsed:.1f} tok/s)")

    else:
        print(f"Error: unknown model_type {model_type!r} in config.yaml", file=sys.stderr)
        sys.exit(1)


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
    import os
    import pandas as pd
    from dantinox.visualization import Visualizer

    _GROUP_TO_CHARTS = {
        "perf":     ["throughput", "throughput_batch", "latency"],
        "insights": ["pareto"],
        "3d":       ["pareto"],
        "3d_dkv":   ["throughput"],
    }
    all_groups = list(_GROUP_TO_CHARTS)

    if not os.path.exists(args.in_csv):
        print(f"Error: benchmark CSV not found: {args.in_csv}", file=sys.stderr)
        print("Run 'dantinox benchmark --out_csv ...' first.", file=sys.stderr)
        sys.exit(1)

    selected = list(args.groups) if args.groups else all_groups
    unknown  = [g for g in selected if g not in _GROUP_TO_CHARTS]
    if unknown:
        print(f"Error: unknown plot group(s): {unknown}. Valid: {all_groups}", file=sys.stderr)
        sys.exit(1)

    chart_names: list[str] = []
    for g in selected:
        chart_names.extend(_GROUP_TO_CHARTS[g])
    chart_names = list(dict.fromkeys(chart_names))

    df    = pd.read_csv(args.in_csv)
    paths = Visualizer().render(df, charts=chart_names, out_dir=args.out_dir)
    print(f"\nDone — {len(paths)} figures written to {args.out_dir}/")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    import os
    import traceback
    import pandas as pd
    from core.config import Config
    from core.model import Transformer
    from flax import nnx
    from dantinox.paradigms.ar import ARParadigm
    from dantinox.benchmarking import BenchmarkConfig, BenchmarkSuite, ThroughputTask, LatencyTask
    from dantinox.exceptions import BenchmarkError

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    run_names = args.runs or [
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]

    bench_cfg = BenchmarkConfig()
    rows: list[dict] = []
    for name in run_names:
        path = os.path.join(runs_dir, name)
        logging.getLogger(__name__).info("Benchmarking run: %s", name)
        try:
            model    = Transformer.from_pretrained(path, rngs=nnx.Rngs(42))
            cfg      = getattr(model, "config", None)
            if cfg is None:
                raise BenchmarkError(f"Cannot read config from {name}")
            paradigm = ARParadigm(cfg.to_model_config() if hasattr(cfg, "to_model_config") else cfg)
            suite    = BenchmarkSuite(tasks=[ThroughputTask(), LatencyTask()], config=bench_cfg)
            report   = suite.run(paradigm, model)
            row      = {"run": name, **report.to_dataframe().to_dict("records")[0]}
            rows.append(row)
        except BenchmarkError as exc:
            logging.getLogger(__name__).error("  Skipped %s: %s", name, exc)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "  Unexpected error for %s: %s\n%s", name, exc, traceback.format_exc()
            )

    df = pd.DataFrame(rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Results saved to {args.out_csv}")

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
    if getattr(args, "diff_ar", False):
        cmd += ["--diff-ar"]
    if getattr(args, "eval", False):
        cmd += ["--eval"]
    if getattr(args, "inference_off", False):
        cmd += ["--inference-off"]
    if getattr(args, "no_mla", False):
        cmd += ["--no-mla"]
    if getattr(args, "pdf", False):
        cmd += ["--pdf"]
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


def _cmd_merge_lora(args: argparse.Namespace) -> None:
    """Merge LoRA adapters into base weights and save to a new directory."""
    import os
    import shutil
    import msgpack
    import yaml
    from flax import nnx
    from flax.serialization import _msgpack_ext_unpack
    from core.config import Config
    from core.lora import LoRALinear

    run_dir = args.run_dir
    out_dir = args.out_dir

    if not os.path.isdir(run_dir):
        print(f"Error: run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(out_dir) and not args.overwrite:
        print(f"Error: out_dir already exists: {out_dir}  (use --overwrite to force)", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    if not flat:
        flat = raw
    cfg = Config.from_dict(flat)

    for fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
        weights_path = os.path.join(run_dir, fname)
        if os.path.exists(weights_path):
            break
    else:
        print(f"Error: no weights file found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)

    model_type = flat.get("model_type", "autoregressive")
    rngs = nnx.Rngs(42)
    if model_type == "elf":
        from core.elf import ELFTransformer
        model = ELFTransformer(cfg.to_elf_config(), rngs=rngs)
    elif model_type == "diffusion":
        from core.model import DiffusionTransformer
        model = DiffusionTransformer(cfg, rngs=rngs)
    else:
        from core.model import Transformer
        model = Transformer(cfg, rngs=rngs)
    nnx.update(model, state_dict)

    # Merge LoRA adapters: fuse each LoRALinear's low-rank delta into the base kernel
    for path, module in nnx.iter_modules(model):
        if isinstance(module, LoRALinear):
            fused = module.merge_weights()
            module.base.kernel.value = fused

    # Extract merged state and serialise
    _, merged_state = nnx.split(model)
    from flax.serialization import _msgpack_ext_pack
    packed = msgpack.packb(merged_state, default=_msgpack_ext_pack, strict_types=True)

    os.makedirs(out_dir, exist_ok=True)
    out_weights = os.path.join(out_dir, "best_model_weights.msgpack")
    with open(out_weights, "wb") as f:
        f.write(packed)

    shutil.copy(config_path, os.path.join(out_dir, "config.yaml"))
    print(f"Merged weights saved to: {out_weights}")
    print(f"Config copied to:        {os.path.join(out_dir, 'config.yaml')}")


def _cmd_profile(args: argparse.Namespace) -> None:
    """Print parameter count and FLOPs for a checkpoint or config."""
    import yaml
    from flax import nnx
    from core.config import Config
    from dantinox.profiling.counter import count_flops

    if args.run_dir:
        import os
        config_path = os.path.join(args.run_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: config.yaml not found in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    elif args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
    else:
        print("Error: one of --run_dir or --config is required", file=sys.stderr)
        sys.exit(1)

    flat: dict = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    if not flat:
        flat = raw
    cfg = Config.from_dict(flat)

    model_type = flat.get("model_type", "autoregressive")
    rngs = nnx.Rngs(42)
    if model_type == "elf":
        from core.elf import ELFTransformer
        model = ELFTransformer(cfg.to_elf_config(), rngs=rngs)
    elif model_type == "diffusion":
        from core.model import DiffusionTransformer
        model = DiffusionTransformer(cfg, rngs=rngs)
    else:
        from core.model import Transformer
        model = Transformer(cfg, rngs=rngs)

    param_leaves = nnx.state(model, nnx.Param)
    import jax
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(param_leaves))

    print(f"\n{'─' * 50}")
    print(f"  Model type  : {model_type}")
    print(f"  dim         : {cfg.dim}")
    print(f"  num_blocks  : {cfg.num_blocks}")
    print(f"  n_heads     : {cfg.n_heads}")
    print(f"  vocab_size  : {cfg.vocab_size}")
    print(f"  Parameters  : {total_params:,}  ({total_params / 1e6:.2f} M)")

    try:
        model_cfg = cfg.to_model_config() if hasattr(cfg, "to_model_config") else cfg
        flops = count_flops(model_cfg, seq_len=args.seq_len, batch_size=args.batch_size)
        print(f"\n  seq_len={args.seq_len}  batch_size={args.batch_size}")
        print(f"  {flops}")
    except Exception as exc:
        print(f"\n  FLOPs: unavailable ({exc})")
    print(f"{'─' * 50}\n")


def _cmd_eval(args: argparse.Namespace) -> None:
    """Delegate to scripts/test_generation_quality.py via subprocess."""
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "test_generation_quality.py"
    if not script.exists():
        print(f"Error: {script} not found — is the repo intact?", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(script),
        "--run-dir", args.run_dir,
        "--n-samples", str(args.n_samples),
        "--gen-len", str(args.gen_len),
        "--seed", str(args.seed),
    ]
    if args.out_csv:
        cmd += ["--out-csv", args.out_csv]

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
    p_gen.add_argument("--stream", action="store_true",
                       help="Stream tokens to stdout as they are produced")
    p_gen.add_argument("--seed", type=int, default=42)
    # Diffusion-specific arguments
    p_gen.add_argument("--n_steps", type=int, default=50,
                       help="Denoising steps (diffusion/ELF only)")
    p_gen.add_argument("--block_size", type=int, default=32,
                       help="Block size for fast_dllm_generate (diffusion only)")
    p_gen.add_argument("--use_dual_cache", action="store_true", default=True,
                       help="Use DualCache in fast_dllm_generate (diffusion only)")
    p_gen.add_argument("--confidence_threshold", type=float, default=0.9,
                       help="Confidence threshold for unmasking (diffusion only)")
    # ELF-specific arguments
    p_gen.add_argument("--cfg_scale", type=float, default=1.5,
                       help="CFG guidance scale (ELF only)")

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
                      help="Run trained-model analysis (stages 5–6)")
    p_ib.add_argument("--diff-ar", action="store_true",
                      help="Run the AR vs Diffusion sweep (stages 3–4)")
    p_ib.add_argument("--eval", action="store_true",
                      help="Run quality evaluation pipeline: PPL + confidence + gen-quality + paper figures (implies --trained)")
    p_ib.add_argument("--inference-off", action="store_true",
                      help="Skip inference pipeline; requires at least one of --trained/--diff-ar/--eval")
    p_ib.add_argument("--no-mla", action="store_true",
                      help="Skip MLA experiments in diffusion_ar and confidence sweeps")
    p_ib.add_argument("--pdf", action="store_true",
                      help="Save EMNLP paper figures as PDF in addition to PNG")
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

    # ── merge-lora ─────────────────────────────────────────────────────────────
    p_merge = sub.add_parser("merge-lora", help="Merge LoRA adapters into base weights and save")
    p_merge.add_argument("--run_dir", required=True, help="Run directory with LoRA checkpoint")
    p_merge.add_argument("--out_dir", required=True, help="Output directory for merged weights")
    p_merge.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists")

    # ── profile ────────────────────────────────────────────────────────────────
    p_profile = sub.add_parser("profile", help="Print parameter count and FLOPs for a checkpoint")
    p_profile_src = p_profile.add_mutually_exclusive_group()
    p_profile_src.add_argument("--run_dir", default=None, help="Checkpoint directory (reads config.yaml)")
    p_profile_src.add_argument("--config", default=None, help="YAML config file (if --run_dir not given)")
    p_profile.add_argument("--seq_len", type=int, default=512, help="Sequence length for FLOPs (default 512)")
    p_profile.add_argument("--batch_size", type=int, default=1, help="Batch size for FLOPs (default 1)")

    # ── eval ───────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate generation quality for a checkpoint")
    p_eval.add_argument("--run_dir", required=True, help="Run directory with checkpoint")
    p_eval.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate (default 50)")
    p_eval.add_argument("--gen_len", type=int, default=128, help="Generation length in tokens (default 128)")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    p_eval.add_argument("--out_csv", default=None, help="Save metrics to this CSV file")

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
        "train":      _cmd_train,
        "generate":   _cmd_generate,
        "sweep":      _cmd_sweep,
        "benchmark":  _cmd_benchmark,
        "infbench":   _cmd_infbench,
        "plot":       _cmd_plot,
        "find-lr":    _cmd_find_lr,
        "push":       _cmd_push,
        "pull":       _cmd_pull,
        "merge-lora": _cmd_merge_lora,
        "profile":    _cmd_profile,
        "eval":       _cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
