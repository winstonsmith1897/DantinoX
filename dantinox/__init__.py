"""
DantinoX — JAX/Flax transformer library for AR, discrete diffusion, and
continuous flow-matching models.

Three levels of API
-------------------

**Level 1 — ultra low-code (one-liners)**::

    import dantinox as dx

    run_dir = dx.fit("ar", "data/corpus.txt", dim=512, n_heads=8,
                     head_size=64, num_blocks=12, vocab_size=32_000)
    tokens  = dx.quick_generate(run_dir, "Once upon a time")

**Level 2 — explicit paradigm objects**::

    import dantinox as dx

    paradigm = dx.ARParadigm(dx.ModelConfig(dim=512, n_heads=8, head_size=64,
                                             num_blocks=12, vocab_size=32_000))
    trainer  = dx.Trainer(paradigm, dx.TrainingConfig(lr=3e-4, epochs=5))
    run_dir  = trainer.fit("data/corpus.txt")

    model    = dx.load(run_dir)
    tokens   = paradigm.generate(model, prompt, rng)

**Level 3 — full control**::

    from dantinox.core.config    import ModelConfig
    from dantinox.core.model     import Transformer
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
from dantinox.core.config import Config, ELFConfig, ModelConfig, TrainingConfig

# ── Core model ────────────────────────────────────────────────────────────────
from dantinox.core.model import Transformer

# ── Paradigms ─────────────────────────────────────────────────────────────────
from dantinox.paradigms import (
    ARParadigm,
    ContinuousParadigm,
    DiscreteConfig,
    DiscreteParadigm,
    Paradigm,
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
        config = ModelConfig(**{**kwargs, "causal": True})
    return ARParadigm(config)


def _build_discrete(config, kwargs):
    diff_kw = {
        k: kwargs.pop(k)
        for k in ("noise_schedule", "mask_token_id")
        if k in kwargs
    }
    if config is None:
        config = ModelConfig(**{**kwargs, "causal": False})
    return DiscreteParadigm(config, DiscreteConfig(**diff_kw) if diff_kw else None)


def _build_continuous(config, kwargs):
    if config is None:
        config = ELFConfig(**kwargs)
    return ContinuousParadigm(config)


_PARADIGM_MAP = {
    "ar":         _build_ar,
    "discrete":   _build_discrete,
    "continuous": _build_continuous,
}


# ── Low-code functional API ───────────────────────────────────────────────────


def build(
    paradigm: str,
    config: ModelConfig | ELFConfig | None = None,
    **model_kwargs,
) -> Paradigm:
    """Construct a Paradigm from a string name and optional config.

    Args:
        paradigm    : ``"ar"`` | ``"discrete"`` | ``"continuous"``
        config      : A ``ModelConfig`` (AR/discrete) or ``ELFConfig``
                      (continuous).  When omitted, *model_kwargs* are forwarded
                      to the appropriate config constructor.
        **model_kwargs : Forwarded to ``ModelConfig`` or ``ELFConfig`` when
                         *config* is None.

    Returns:
        A ready-to-use :class:`Paradigm` instance.

    Example::

        p = dx.build("ar", dim=512, n_heads=8, head_size=64,
                     num_blocks=12, vocab_size=32_000)
    """
    if paradigm not in _PARADIGM_MAP:
        raise ValueError(
            f"Unknown paradigm {paradigm!r}. "
            f"Choose from: {list(_PARADIGM_MAP)}"
        )
    return _PARADIGM_MAP[paradigm](config, model_kwargs)


def train(
    paradigm: Paradigm,
    data_source: str,
    *,
    run_dir: str | None = None,
    **training_kwargs,
) -> str:
    """Train *paradigm* on *data_source* and return the run directory.

    Args:
        paradigm        : Any :class:`Paradigm` instance.
        data_source     : Path to a text file, or a HuggingFace dataset name.
        run_dir         : Output directory (auto-generated when omitted).
        **training_kwargs : Forwarded to ``TrainingConfig`` — e.g.
                            ``lr=3e-4, epochs=10, batch_size=64``.

    Returns:
        Absolute path to the run directory with the best checkpoint.

    Example::

        run_dir = dx.train(paradigm, "data/wiki.txt", lr=1e-4, epochs=3)
    """
    cfg     = TrainingConfig(**training_kwargs) if training_kwargs else TrainingConfig()
    trainer = Trainer(paradigm, cfg)
    return trainer.fit(data_source, run_dir=run_dir)


def fit(
    paradigm: str,
    data_source: str,
    *,
    run_dir: str | None = None,
    training_config: TrainingConfig | None = None,
    **kwargs,
) -> str:
    """One-call shortcut: build paradigm, train, return run directory.

    Keyword arguments that match ``ModelConfig`` / ``ELFConfig`` fields are
    forwarded to the config constructor; everything else goes to
    ``TrainingConfig``.

    Example::

        run_dir = dx.fit("ar", "data/wiki.txt",
                         dim=512, n_heads=8, head_size=64, num_blocks=12,
                         vocab_size=32_000, lr=3e-4, epochs=5)
    """
    from dataclasses import fields as _fields

    model_fields = {f.name for f in _fields(ModelConfig)}
    elf_fields   = {f.name for f in _fields(ELFConfig)}
    train_fields = {f.name for f in _fields(TrainingConfig)}

    model_kw = {k: v for k, v in kwargs.items() if k in model_fields or k in elf_fields}
    train_kw = {k: v for k, v in kwargs.items() if k in train_fields}

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

    When *paradigm* is supplied its ``build_model()`` is called to construct
    the model skeleton before weights are restored.  Otherwise falls back to
    ``Transformer.from_pretrained()``.

    Example::

        model = dx.load("runs/20240101_120000", paradigm=my_paradigm)
    """
    import os
    import flax.serialization
    from flax import nnx

    ckpt_path = os.path.join(run_dir, "checkpoint_best.msgpack")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    if paradigm is not None:
        from dantinox.training.trainer import _msgpack_load
        model = paradigm.build_model(nnx.Rngs(0))
        raw = _msgpack_load(ckpt_path)
        state = nnx.state(model, nnx.Not(nnx.RngState))
        state.replace_by_pure_dict(raw)
        nnx.update(model, state)
        return model

    from dantinox.core.model import Transformer
    return Transformer.from_pretrained(run_dir, rngs=nnx.Rngs(0))


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

    # Paradigms name their length/temperature knobs differently (AR:
    # max_new_tokens, diffusion: gen_len, ELF: gen_len only) — forward what
    # this paradigm actually accepts.
    import inspect
    params = inspect.signature(paradigm.generate).parameters
    kwargs = {}
    if "max_new_tokens" in params:
        kwargs["max_new_tokens"] = max_new_tokens
    elif "gen_len" in params:
        kwargs["gen_len"] = max_new_tokens
    if "temperature" in params:
        kwargs["temperature"] = temperature

    out = paradigm.generate(model, ids, rng, **kwargs)
    return tokenizer.decode([int(t) for t in out[0]])


__all__ = [
    "__version__",
    # configs
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "ELFConfig",
    # model
    "Transformer",
    # paradigms
    "Paradigm",
    "ARParadigm",
    "DiscreteConfig",
    "DiscreteParadigm",
    "ContinuousParadigm",
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
    # low-code functional API
    "build",
    "train",
    "fit",
    "profile",
    "load",
    "quick_generate",
    # legacy
    "Generator",
    "BenchmarkRunner",
    "Plotter",
    "push",
    "pull",
    "resolve_checkpoint",
    "DantinoXError",
    "ConfigError",
    "CheckpointError",
    "BenchmarkError",
    "PlotError",
]
