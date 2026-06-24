"""
DantinoX — JAX/Flax transformer library for AR, discrete diffusion, and
continuous flow-matching models.

All paradigms share a single ``ModelConfig`` and a single ``Paradigm`` class.
The ``paradigm`` field in ``ModelConfig`` selects which strategy to use.

Three levels of API
-------------------

**Level 1 — ultra low-code (one-liners)**::

    import dantinox as dx

    run_dir = dx.fit("ar", "data/corpus.txt", dim=512, n_heads=8,
                     num_blocks=12, vocab_size=32_000)
    tokens  = dx.quick_generate(run_dir, "Once upon a time")

**Level 2 — explicit paradigm objects**::

    import dantinox as dx

    # Autoregressive
    p = dx.Paradigm(dx.ModelConfig(paradigm="ar", dim=512, n_heads=8, num_blocks=12))

    # Discrete diffusion (LLaDA)
    p = dx.Paradigm(dx.ModelConfig(paradigm="discrete", dim=512, n_heads=8, num_blocks=12,
                                    noise_schedule="cosine"))

    # ELF continuous flow-matching
    p = dx.Paradigm(dx.ModelConfig(paradigm="continuous", dim=512, n_heads=8, num_blocks=12,
                                    embed_dim=768))

    # Contrastive embedder (SimCSE)
    p = dx.Paradigm(dx.ModelConfig(paradigm="embedder", dim=512, n_heads=8, num_blocks=12,
                                    dropout=0.1))

    run_dir = dx.Trainer(p, dx.TrainingConfig(lr=3e-4, epochs=5)).fit("data/corpus.txt")

**Level 3 — full control**::

    from dantinox.core.config       import ModelConfig
    from dantinox.core.model        import Transformer
    from dantinox.paradigms.ar      import ARParadigm
    from dantinox.training.trainer  import Trainer
    from dantinox.profiling         import LatencyTracker, count_flops
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version("dantinox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

# ── Config re-exports ─────────────────────────────────────────────────────────
from dantinox.core.config import Config, ModelConfig, TrainingConfig

# ── Core model ────────────────────────────────────────────────────────────────
from dantinox.core.model import Transformer

# ── Paradigms ─────────────────────────────────────────────────────────────────
from dantinox.paradigms import (
    Paradigm,
    EmbedderParadigm,
    info_nce_loss,
    # advanced / backward compat
    ARParadigm,
    DiscreteParadigm,
    ContinuousParadigm,
)

# ── Training ──────────────────────────────────────────────────────────────────
from dantinox.training import Trainer, build_optimizer, build_schedule

# ── Profiling ─────────────────────────────────────────────────────────────────
from dantinox.profiling import (
    FLOPsBreakdown,
    LatencyTracker,
    ProfilingResult,
    count_flops,
    profile_fn,
)

# ── Benchmarking ─────────────────────────────────────────────────────────────
from dantinox.benchmarking import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    LatencyTask,
    PerplexityTask,
    SuiteReport,
    ThroughputTask,
)

# ── Visualization ─────────────────────────────────────────────────────────────
from dantinox.visualization import (
    Chart,
    LatencyChart,
    ParetoChart,
    RadarChart,
    RenderConfig,
    ThroughputBatchChart,
    ThroughputChart,
    TrainingCurveChart,
    Visualizer,
)

# ── Embedding / RAG ──────────────────────────────────────────────────────────
from dantinox.embedder import Embedder
from dantinox.embedder_trainer import EmbedderTrainer

# ── Legacy high-level helpers (backward compat) ───────────────────────────────
from dantinox.generator import Generator
from dantinox.bench import BenchmarkRunner
from dantinox.hub import pull, push, resolve_checkpoint
from dantinox.plotting import Plotter
from dantinox.exceptions import (
    BenchmarkError,
    CheckpointError,
    ConfigError,
    DantinoXError,
    PlotError,
)

# ── Internal builders ─────────────────────────────────────────────────────────
# Defined before the public API functions that reference them.


def _build_ar(config, kwargs):
    if config is None:
        config = ModelConfig(**{**kwargs, "paradigm": "ar"})
    return Paradigm(config)


def _build_discrete(config, kwargs):
    if config is None:
        config = ModelConfig(**{**kwargs, "paradigm": "discrete"})
    return Paradigm(config)


def _build_continuous(config, kwargs):
    if config is None:
        config = ModelConfig(**{**kwargs, "paradigm": "continuous"})
    return Paradigm(config)


def _build_embedder(config, kwargs):
    import warnings
    if "pooling" in kwargs:
        warnings.warn(
            "The 'pooling' kwarg is deprecated; use embed_pooling= on ModelConfig instead.",
            DeprecationWarning, stacklevel=4,
        )
        kwargs["embed_pooling"] = kwargs.pop("pooling")
    if "temperature" in kwargs:
        warnings.warn(
            "The 'temperature' kwarg is deprecated; use embed_temperature= on ModelConfig instead.",
            DeprecationWarning, stacklevel=4,
        )
        kwargs["embed_temperature"] = kwargs.pop("temperature")
    if config is None:
        config = ModelConfig(**{**kwargs, "paradigm": "embedder"})
    return Paradigm(config)


_PARADIGM_MAP = {
    "ar":         _build_ar,
    "discrete":   _build_discrete,
    "continuous": _build_continuous,
    "embedder":   _build_embedder,
}


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split **kwargs into (model_kwargs, training_kwargs) by field membership."""
    from dataclasses import fields as _fields
    model_fields = {f.name for f in _fields(ModelConfig)}
    train_fields = {f.name for f in _fields(TrainingConfig)}
    model_kw = {k: v for k, v in kwargs.items() if k in model_fields}
    train_kw = {k: v for k, v in kwargs.items() if k in train_fields}
    return model_kw, train_kw


# ── Low-code functional API ───────────────────────────────────────────────────


def build(
    paradigm: str,
    config: ModelConfig | None = None,
    **model_kwargs,
) -> Paradigm:
    """Construct a Paradigm from a string name and optional ``ModelConfig``.

    This is a thin wrapper around ``Paradigm(ModelConfig(paradigm=..., ...))``.
    For new code, prefer the explicit form::

        p = dx.Paradigm(dx.ModelConfig(paradigm="ar", dim=512, n_heads=8,
                                        num_blocks=12, vocab_size=32_000))

    Args:
        paradigm      : ``"ar"`` | ``"discrete"`` | ``"continuous"`` | ``"embedder"``
        config        : A ``ModelConfig``.  When omitted, *model_kwargs* are
                        forwarded to ``ModelConfig``.
        **model_kwargs : Forwarded to ``ModelConfig`` when *config* is None.

    Returns:
        A ready-to-use :class:`Paradigm` instance.

    Example::

        p = dx.build("ar", dim=512, n_heads=8, num_blocks=12, vocab_size=32_000)

        p = dx.build("continuous", dim=256, n_heads=4, num_blocks=4,
                     embed_dim=768, bottleneck_dim=128)
    """
    if paradigm not in _PARADIGM_MAP:
        raise ValueError(
            f"Unknown paradigm {paradigm!r}. "
            f"Choose from: {list(_PARADIGM_MAP)}"
        )
    return _PARADIGM_MAP[paradigm](config, model_kwargs)


def train(
    paradigm: Paradigm,
    data_source: str | None = None,
    *,
    run_dir: str | None = None,
    training_config: TrainingConfig | None = None,
    **training_kwargs,
) -> str:
    """Train *paradigm* on *data_source* and return the run directory.

    Args:
        paradigm        : Any :class:`Paradigm` instance.
        data_source     : Path to a text file, or a HuggingFace dataset name.
                          May be omitted when *training_config* sets
                          ``dataset_source="huggingface"`` and ``dataset_name``.
        run_dir         : Output directory (auto-generated when omitted).
        training_config : A ready-made :class:`TrainingConfig`.  When supplied,
                          *training_kwargs* are ignored.
        **training_kwargs : Forwarded to ``TrainingConfig`` — e.g.
                            ``lr=3e-4, epochs=10, batch_size=64``.

    Returns:
        Absolute path to the run directory with the best checkpoint.

    Example::

        run_dir = dx.train(paradigm, "data/wiki.txt", lr=1e-4, epochs=3)

        cfg = dx.TrainingConfig(lr=1e-4, epochs=3, dataset_source="huggingface",
                                dataset_name="wikitext")
        run_dir = dx.train(paradigm, training_config=cfg)
    """
    cfg     = training_config or (TrainingConfig(**training_kwargs) if training_kwargs else TrainingConfig())
    trainer = Trainer(paradigm, cfg)
    return trainer.fit(data_source, run_dir=run_dir)


_LEGACY_PARADIGM_ALIASES = {
    "autoregressive": "ar",
    "diffusion": "discrete",
    "elf": "continuous",
}


def fit(
    paradigm: str,
    data_source: str,
    *,
    run_dir: str | None = None,
    training_config: TrainingConfig | None = None,
    **kwargs,
) -> str:
    """One-call shortcut: build paradigm, train, return run directory.

    Keyword arguments that match ``ModelConfig`` fields are forwarded to the
    config constructor; everything else goes to ``TrainingConfig``.

    Example::

        run_dir = dx.fit("ar", "data/wiki.txt",
                         dim=512, n_heads=8, head_size=64, num_blocks=12,
                         vocab_size=32_000, lr=3e-4, epochs=5)
    """
    import warnings
    if paradigm in _LEGACY_PARADIGM_ALIASES:
        canonical = _LEGACY_PARADIGM_ALIASES[paradigm]
        warnings.warn(
            f"Paradigm string {paradigm!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning, stacklevel=2,
        )
        paradigm = canonical
    model_kw, train_kw = _split_kwargs(kwargs)
    p   = build(paradigm, **model_kw)
    cfg = training_config or TrainingConfig(**train_kw)
    return Trainer(p, cfg).fit(data_source, run_dir=run_dir)


def profile(
    config: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    *,
    n_warmup: int = 5,
    n_runs: int = 20,
    model=None,
) -> ProfilingResult:
    """Profile a model: FLOPs + latency + throughput.

    When *model* is provided, a real JAX forward pass is timed.
    Otherwise only the analytical FLOPs estimate is returned (latency = 0).

    Example::

        cfg    = dx.ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12,
                                vocab_size=32_000)
        report = dx.profile(cfg, seq_len=512, batch_size=4)
        print(report.flops)
        print(report.latency)
    """
    from dantinox.profiling.counter import count_flops
    from dantinox.profiling.tracker import LatencyTracker

    flops   = count_flops(config, seq_len, batch_size)
    tracker = LatencyTracker()

    if model is not None:
        import jax
        rng = jax.random.PRNGKey(0)
        x   = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)

        for _ in range(n_warmup):
            _ = model(x)

        for _ in range(n_runs):
            with tracker.measure(n_tokens=batch_size * seq_len):
                _ = model(x)

    result = tracker.result()
    result.flops = flops
    return result


def load(run_dir: str, paradigm: Paradigm | None = None):
    """Load the best checkpoint from *run_dir* and return the NNX model.

    When *paradigm* is omitted, ``config.yaml`` in *run_dir* is read to
    reconstruct the ``ModelConfig`` and infer the paradigm automatically.
    Pass *paradigm* explicitly to override or when no ``config.yaml`` is
    present (legacy checkpoints).

    Example::

        model = dx.load("runs/20240101_120000")
        model = dx.load("runs/20240101_120000", paradigm=my_paradigm)
    """
    import os
    from flax import nnx

    ckpt_path = os.path.join(run_dir, "checkpoint_best.msgpack")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    if paradigm is None:
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(cfg_path):
            model_cfg = ModelConfig.from_yaml(cfg_path)
            paradigm  = Paradigm(model_cfg)

    if paradigm is not None:
        from dantinox.training.trainer import _msgpack_load
        model = paradigm.build_model(nnx.Rngs(0))
        raw   = _msgpack_load(ckpt_path)
        state = nnx.state(model, nnx.Not(nnx.RngState))
        state.replace_by_pure_dict(raw)
        nnx.update(model, state)
        return model

    from dantinox.core.model import Transformer
    return Transformer.from_pretrained(run_dir, rngs=nnx.Rngs(0))


def sweep(
    paradigm,
    sweep_yaml: str,
    data_source: str | None = None,
    *,
    wandb_project: str = "DantinoX",
    count: int | None = None,
    training_config: "TrainingConfig | None" = None,
    **training_kwargs,
) -> str:
    """One-call sweep shortcut: build paradigm, create trainer, run W&B sweep.

    Args:
        paradigm        : Paradigm instance **or** string ``"ar"`` / ``"discrete"``
                          / ``"continuous"`` (same values as :func:`build`).
        sweep_yaml      : Path to a W&B sweep YAML file.
        data_source     : Training corpus (path or HF dataset name).  Optional
                          when TrainingConfig already specifies a HF dataset.
        wandb_project   : W&B project name (default ``"DantinoX"``).
        count           : Max agent runs (unlimited if ``None``).
        training_config : Base :class:`TrainingConfig`.  Built from
                          *training_kwargs* when omitted.
        **training_kwargs : Forwarded to ``TrainingConfig`` (e.g. ``lr``,
                            ``epochs``, ``batch_size``).

    Returns:
        The W&B sweep ID string.

    Example::

        import dantinox as dx

        sweep_id = dx.sweep(
            "ar", "configs/sweep.yaml", "data/wiki.txt",
            wandb_project="my-project",
            dim=256, n_heads=4, head_size=64, num_blocks=4, vocab_size=200,
            epochs=3,
        )
    """
    if isinstance(paradigm, str):
        model_kw, train_kw = _split_kwargs(training_kwargs)
        paradigm = build(paradigm, **model_kw)
        training_kwargs = train_kw

    cfg     = training_config or TrainingConfig(**training_kwargs)
    trainer = Trainer(paradigm, cfg)
    return trainer.sweep(
        sweep_yaml,
        data_source,
        wandb_project=wandb_project,
        count=count,
    )


def quick_generate(
    run_dir: str,
    prompt: str,
    *,
    paradigm: Paradigm | None = None,
    tokenizer=None,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
) -> str:
    """Load checkpoint and generate text — no boilerplate required.

    When *paradigm* is given the checkpoint is restored through
    ``paradigm.build_model()`` and decoded with ``paradigm.generate``;
    *tokenizer* (or the run's saved ``tokenizer.json``) handles text ↔ ids.
    Otherwise the run directory is loaded with :class:`Generator`.

    Example::

        print(dx.quick_generate("runs/20240101_120000", "Once upon a time"))
    """
    if paradigm is None:
        gen = Generator(run_dir)
        return gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    import os
    import jax
    import jax.numpy as jnp

    if tokenizer is None:
        from dantinox.utils.tokenizer import load_tokenizer_from_file
        tok_path = os.path.join(run_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"No tokenizer.json in {run_dir!r} — pass tokenizer= explicitly."
            )
        tokenizer = load_tokenizer_from_file(tok_path)

    model = load(run_dir, paradigm=paradigm)
    ids   = jnp.asarray([tokenizer.encode(prompt)], dtype=jnp.int32)
    rng   = jax.random.PRNGKey(0)

    out = paradigm.generate(model, ids, rng,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature)
    return tokenizer.decode([int(t) for t in out[0]])


__all__ = [
    "__version__",
    # configs
    "Config",
    "ModelConfig",
    "TrainingConfig",
    # model
    "Transformer",
    # paradigms
    "Paradigm",
    "EmbedderParadigm",
    "info_nce_loss",
    # training
    "Trainer",
    "build_optimizer",
    "build_schedule",
    # profiling
    "FLOPsBreakdown",
    "LatencyTracker",
    "ProfilingResult",
    "count_flops",
    "profile_fn",
    # benchmarking
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkTask",
    "BenchmarkSuite",
    "SuiteReport",
    "ThroughputTask",
    "LatencyTask",
    "PerplexityTask",
    # visualization
    "Chart",
    "RenderConfig",
    "Visualizer",
    "TrainingCurveChart",
    "ThroughputChart",
    "ThroughputBatchChart",
    "LatencyChart",
    "RadarChart",
    "ParetoChart",
    # embedding / RAG
    "Embedder",
    "EmbedderTrainer",
    # low-code functional API
    "build",
    "train",
    "fit",
    "sweep",
    "profile",
    "load",
    "quick_generate",
    # hub
    "push",
    "pull",
    "resolve_checkpoint",
    # exceptions
    "DantinoXError",
    "ConfigError",
    "CheckpointError",
    "BenchmarkError",
    "PlotError",
]
