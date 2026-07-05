"""Multi-run sequential profiler with config-key filtering.

Discovers all runs under a root directory, optionally filters them by
config key-value pairs, and runs the requested metrics sequentially:

    profiler = RunsProfiler(
        runs="runs/",
        filter_config={"model_type": "elf", "dim": 768},
        metrics=["latency", "throughput", "flops"],
        batch_sizes=[1, 4, 16, 64],
        seq_len=256,
        n_warmup=5,
        n_measure=50,
    )
    report = profiler.run(save_csv="results/profile_768d.csv")
    print(report.summary())
    df = report.to_dataframe()

Supported metrics
-----------------
``latency``    : Per-request latency (mean, p50, p95, p99) at batch_size=1.
``throughput`` : Token/s sweep over *batch_sizes* at fixed *seq_len*.
``energy``     : GPU energy per token via NVML (requires pynvml).
``flops``      : Analytical FLOPs + hardware efficiency (requires elapsed time
                 from the latency measurement).
``perplexity`` : Model perplexity on a held-out token array (*data_path* required).
``entropy``    : Per-token output entropy (*data_path* required, AR only).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger(__name__)

_ALL_METRICS = ("latency", "throughput", "energy", "flops", "perplexity", "entropy")


# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class RunProfile:
    """Profiling results for a single run directory.

    Each metric field is ``None`` when not requested or when the measurement
    failed.  ``config`` contains the flat key-value config of the run.
    """

    run_name:   str
    run_dir:    str
    config:     dict[str, Any]
    latency:    Any | None = None   # LatencyResult
    throughput: Any | None = None   # ThroughputResult
    energy:     Any | None = None   # EnergyResult
    flops:      Any | None = None   # FLOPsResult
    perplexity: Any | None = None   # PerplexityResult
    entropy:    Any | None = None   # EntropyResult
    error:      str | None = None   # set if model loading failed

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_name": self.run_name,
            "run_dir":  self.run_dir,
        }
        # selected config fields surfaced to the dataframe
        for key in ("model_type", "dim", "num_blocks", "n_heads", "kv_heads",
                    "mla", "optimizer", "batch_size", "lr", "attention_type"):
            if key in self.config:
                d[key] = self.config[key]
        for metric in ("latency", "throughput", "energy", "flops", "perplexity", "entropy"):
            result = getattr(self, metric)
            if result is not None and hasattr(result, "to_dict"):
                d.update(result.to_dict())
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class MultiRunReport:
    """Aggregated profiling results across all matched runs.

    Attributes
    ----------
    profiles     : One :class:`RunProfile` per run that was processed.
    total_time_s : Wall-clock seconds for the full run.
    metrics      : Metric names that were requested.
    filter_used  : The config filter that was applied.
    """

    profiles:     list[RunProfile]
    total_time_s: float = 0.0
    metrics:      list[str] = field(default_factory=list)
    filter_used:  dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """Return a ``pandas.DataFrame`` with one row per run."""
        import pandas as pd
        return pd.DataFrame([p.to_dict() for p in self.profiles])

    def save_csv(self, path: str) -> None:
        """Save the report to *path* as a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        log.info("Saved profiling report to %s", path)

    def summary(self) -> str:
        lines = [
            f"MultiRunReport — {len(self.profiles)} run(s)  "
            f"({self.total_time_s:.1f} s total)",
            f"  metrics : {', '.join(self.metrics)}",
        ]
        if self.filter_used:
            lines.append(
                "  filter  : "
                + "  ".join(f"{k}={v}" for k, v in self.filter_used.items())
            )
        for p in self.profiles:
            parts = [f"  [{p.run_name}]"]
            if p.error:
                parts.append(f"ERROR: {p.error}")
            else:
                if p.latency:
                    parts.append(f"lat={p.latency.mean_ms:.1f}ms")
                if p.throughput:
                    parts.append(f"peak={p.throughput.peak_tps:,.0f}tok/s")
                if p.perplexity:
                    parts.append(f"ppl={p.perplexity.perplexity:.3f}")
                if p.energy:
                    parts.append(f"E={p.energy.j_per_tok*1e3:.3f}mJ/tok")
            lines.append("  ".join(parts))
        return "\n".join(lines)

    def plot(
        self,
        metric:  str,
        kind:    str = "3d",
        mode:    str = "surface",
        title:   str | None = None,
        show:    bool = True,
        output:  str | None = None,
    ):
        """Render an interactive plot from this report.

        Args:
            metric  : Metric to visualise — e.g. ``"tps"``, ``"latency"``,
                      ``"perplexity"``, ``"p99"``.  See
                      :mod:`dantinox.profiling.plots` for all aliases.
            kind    : ``"3d"`` for a 3D surface/scatter comparison across runs,
                      ``"bar"`` for a scalar bar chart.
            mode    : ``"surface"`` or ``"scatter"`` (only for ``kind="3d"``).
            title   : Figure title.
            show    : Open in browser.
            output  : Write HTML to this path.

        Returns:
            ``plotly.graph_objects.Figure``
        """
        from dantinox.profiling.plots import plot_3d_compare, plot_bar_compare
        if kind == "3d":
            return plot_3d_compare(
                self, metric=metric, mode=mode, title=title, show=show, output=output,
            )
        return plot_bar_compare(
            self, metric=metric, title=title, show=show, output=output,
        )

    def __repr__(self) -> str:
        return (
            f"MultiRunReport(runs={len(self.profiles)}, "
            f"metrics={self.metrics}, time={self.total_time_s:.1f}s)"
        )


# ── Main profiler ─────────────────────────────────────────────────────────────


class RunsProfiler:
    """Sequential multi-run profiler with config-based filtering.

    Args:
        runs          : Root directory to search for run sub-directories, or an
                        explicit list of run directory paths.
        filter_config : Key-value pairs that must ALL match the run's
                        ``config.yaml``.  Missing keys are treated as no-match.
                        Example: ``{"model_type": "elf", "dim": 768}``.
        metrics       : Subset of metrics to run.  Defaults to
                        ``["latency", "throughput", "flops"]``.
        batch_sizes   : Batch sizes for the throughput sweep.
        seq_len       : Sequence length used for latency / throughput / energy.
        n_warmup      : JIT warm-up calls per configuration.
        n_measure     : Timed calls per configuration.
        data_path     : Path to a pre-tokenised ``.npy`` array (shape ``[N]``,
                        dtype ``int32``).  Required for ``perplexity`` and
                        ``entropy``.
        energy_device : NVML device index for the energy metric.
        gpu_peak_tflops : Peak BF16 TFLOPS of the target GPU (default A100=312).

    Example::

        profiler = RunsProfiler(
            runs="runs/",
            filter_config={"model_type": "elf", "dim": 768},
            metrics=["latency", "throughput", "flops", "perplexity"],
            data_path="data/wikitext_wikitext-103-raw-v1_t5.npy",
        )
        report = profiler.run(save_csv="results/elf_768d_profile.csv")
        print(report.summary())
    """

    def __init__(
        self,
        runs:             str | list[str],
        filter_config:    dict[str, Any] | None = None,
        metrics:          list[str] | None      = None,
        batch_sizes:      list[int]             = None,
        seq_len:          int                   = 256,
        n_warmup:         int                   = 5,
        n_measure:        int                   = 50,
        data_path:        str | None            = None,
        energy_device:    int                   = 0,
        gpu_peak_tflops:  float                 = 312.0,
    ) -> None:
        self.runs_root      = runs
        self.filter_config  = filter_config or {}
        self.metrics        = metrics or ["latency", "throughput", "flops"]
        self.batch_sizes    = batch_sizes or [1, 4, 16, 64, 128, 256]
        self.seq_len        = seq_len
        self.n_warmup       = n_warmup
        self.n_measure      = n_measure
        self.data_path      = data_path
        self.energy_device  = energy_device
        self.gpu_peak_tflops = gpu_peak_tflops

        unknown = set(self.metrics) - set(_ALL_METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}.  Choose from {_ALL_METRICS}.")

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self) -> list[str]:
        """Return all run directories that match *filter_config*.

        A directory qualifies when:
        1. It contains a ``config.yaml``.
        2. Every key in *filter_config* is present in the flattened config and
           its value matches (strict equality after type coercion).
        """
        candidates: list[str] = []

        if isinstance(self.runs_root, list):
            candidates = list(self.runs_root)
        elif os.path.isdir(self.runs_root):
            for name in sorted(os.listdir(self.runs_root)):
                d = os.path.join(self.runs_root, name)
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.yaml")):
                    candidates.append(d)
        else:
            raise FileNotFoundError(f"runs path not found: {self.runs_root!r}")

        if not self.filter_config:
            return candidates

        matched: list[str] = []
        for run_dir in candidates:
            cfg = _load_flat_config(run_dir)
            if _matches(cfg, self.filter_config):
                matched.append(run_dir)
        return matched

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, save_csv: str | None = None) -> MultiRunReport:
        """Profile all matching runs sequentially.

        Args:
            save_csv : If given, write the report CSV to this path.

        Returns:
            A :class:`MultiRunReport` with one :class:`RunProfile` per run.
        """
        run_dirs   = self.discover()
        profiles:  list[RunProfile] = []
        suite_t0   = time.perf_counter()

        log.info(
            "[RunsProfiler] %d run(s) matched — metrics: %s",
            len(run_dirs), self.metrics,
        )

        data = self._load_data() if any(m in self.metrics for m in ("perplexity", "entropy")) else None

        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            log.info("[RunsProfiler] profiling '%s' …", run_name)
            t0 = time.perf_counter()

            cfg = _load_flat_config(run_dir)
            profile = RunProfile(run_name=run_name, run_dir=run_dir, config=cfg)

            try:
                adapter = _load_adapter(run_dir, cfg)
            except Exception as exc:
                log.error("[RunsProfiler] failed to load '%s': %s", run_name, exc)
                profile.error = str(exc)
                profiles.append(profile)
                continue

            self._run_metrics(profile, adapter, data)
            elapsed = time.perf_counter() - t0
            log.info("[RunsProfiler] '%s' done in %.1f s", run_name, elapsed)
            profiles.append(profile)

        report = MultiRunReport(
            profiles=profiles,
            total_time_s=time.perf_counter() - suite_t0,
            metrics=list(self.metrics),
            filter_used=dict(self.filter_config),
        )

        if save_csv:
            report.save_csv(save_csv)

        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_data(self):
        if not self.data_path:
            log.warning("[RunsProfiler] data_path required for perplexity/entropy — skipping")
            return None
        import numpy as np
        try:
            arr = np.load(self.data_path)
            log.info("[RunsProfiler] loaded data from %s (%d tokens)", self.data_path, len(arr))
            return arr
        except Exception as exc:
            log.error("[RunsProfiler] could not load data: %s", exc)
            return None

    def _run_metrics(self, profile: RunProfile, adapter: _ModelAdapter, data: Any) -> None:
        import jax

        lat_elapsed: float | None = None

        if "latency" in self.metrics:
            try:
                from dantinox.profiling.tracker import LatencyMetric
                lat = LatencyMetric(n_warmup=self.n_warmup, n_measure=self.n_measure)
                fn  = adapter.make_forward_fn(batch_size=1, seq_len=self.seq_len)
                profile.latency = lat.measure(fn, n_tokens=self.seq_len)
                lat_elapsed = profile.latency.mean_ms / 1_000
                log.info("  latency: %s", profile.latency)
            except Exception as exc:
                log.error("  latency failed: %s", exc)

        if "throughput" in self.metrics:
            try:
                from dantinox.profiling.tracker import ThroughputMetric
                thr = ThroughputMetric(
                    n_warmup=self.n_warmup,
                    n_measure=self.n_measure,
                    batch_sizes=self.batch_sizes,
                )
                profile.throughput = thr.measure(
                    get_batch_fn=adapter.get_batch_fn(),
                    model_fn=adapter.model_fn(),
                    seq_len=self.seq_len,
                )
                log.info("  throughput: %s", profile.throughput)
            except Exception as exc:
                log.error("  throughput failed: %s", exc)

        if "energy" in self.metrics:
            try:
                from dantinox.profiling.tracker import EnergyMetric
                eng = EnergyMetric(device_idx=self.energy_device)
                fn  = adapter.make_forward_fn(batch_size=1, seq_len=self.seq_len)
                profile.energy = eng.measure(fn, n_tokens=self.seq_len)
                log.info("  energy: %s", profile.energy)
            except Exception as exc:
                log.error("  energy failed: %s", exc)

        if "flops" in self.metrics:
            try:
                from dantinox.profiling.tracker import FLOPsMetric
                fm  = FLOPsMetric(gpu_peak_tflops=self.gpu_peak_tflops)
                cfg_obj = adapter.model_config()
                if cfg_obj is not None:
                    profile.flops = fm.measure(
                        cfg_obj,
                        seq_len=self.seq_len,
                        batch_size=1,
                        elapsed_s=lat_elapsed,
                    )
                    log.info("  flops: %s", profile.flops)
            except Exception as exc:
                log.error("  flops failed: %s", exc)

        if "perplexity" in self.metrics:
            try:
                from dantinox.profiling.tracker import PerplexityMetric
                if data is None:
                    log.warning("  perplexity: no data — skipped")
                else:
                    ppl = PerplexityMetric(
                        data=data, seq_len=self.seq_len, batch_size=4, n_batches=50,
                    )
                    profile.perplexity = ppl.measure(
                        loss_fn=adapter.loss_fn(),
                        rng=jax.random.PRNGKey(0),
                    )
                    log.info("  perplexity: %s", profile.perplexity)
            except Exception as exc:
                log.error("  perplexity failed: %s", exc)

        if "entropy" in self.metrics:
            try:
                from dantinox.profiling.tracker import EntropyMetric
                if data is None:
                    log.warning("  entropy: no data — skipped")
                else:
                    ent = EntropyMetric(
                        data=data, seq_len=self.seq_len, batch_size=4, n_batches=20,
                    )
                    profile.entropy = ent.measure(
                        logit_fn=adapter.logit_fn(),
                        rng=jax.random.PRNGKey(1),
                    )
                    log.info("  entropy: %s", profile.entropy)
            except Exception as exc:
                log.error("  entropy failed: %s", exc)


# ── Model adapters ────────────────────────────────────────────────────────────


class _ModelAdapter:
    """Internal ABC: wraps a loaded model for use by the metrics."""

    def make_forward_fn(self, batch_size: int, seq_len: int):
        """Return zero-arg callable for one forward pass."""
        raise NotImplementedError

    def get_batch_fn(self):
        """Return ``(batch_size, seq_len) → input`` factory."""
        raise NotImplementedError

    def model_fn(self):
        """Return ``(input) → output`` callable."""
        raise NotImplementedError

    def loss_fn(self):
        """Return ``(batch, rng) → (loss, aux)`` callable."""
        raise NotImplementedError

    def logit_fn(self):
        """Return ``(batch) → logits [B, T, V]`` callable."""
        raise NotImplementedError

    def model_config(self):
        """Return the ModelConfig object (for FLOPs computation)."""
        return None


class _ARAdapter(_ModelAdapter):
    def __init__(self, config, model) -> None:
        self._config = config
        self._model  = model

    def model_config(self):
        return self._config

    def get_batch_fn(self):
        import jax
        vocab = self._config.vocab_size
        def _make(bs: int, sl: int):
            return jax.random.randint(jax.random.PRNGKey(0), (bs, sl), 0, vocab)
        return _make

    def model_fn(self):
        import jax
        model = self._model
        def _fn(x):
            return jax.block_until_ready(model(x))
        return _fn

    def make_forward_fn(self, batch_size: int, seq_len: int):
        import jax
        model = self._model
        x = self.get_batch_fn()(batch_size, seq_len)
        def _fn():
            return jax.block_until_ready(model(x))
        return _fn

    def loss_fn(self):
        import jax
        import jax.numpy as jnp
        model = self._model

        def _loss(batch, rng):
            x  = batch[:, :-1]
            y  = batch[:, 1:]
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            log_p  = jax.nn.log_softmax(logits, axis=-1)
            nll    = -log_p[jnp.arange(y.shape[0])[:, None],
                             jnp.arange(y.shape[1])[None, :], y]
            return float(nll.mean()), None

        return _loss

    def logit_fn(self):
        model = self._model

        def _fn(x):
            out = model(x)
            return out.logits if hasattr(out, "logits") else out

        return _fn


class _DiffusionAdapter(_ModelAdapter):
    def __init__(self, config, model) -> None:
        self._config = config
        self._model  = model

    def model_config(self):
        return self._config

    def get_batch_fn(self):
        import jax
        vocab     = self._config.vocab_size
        mask_id   = getattr(self._config, "mask_token_id", 4)
        def _make(bs: int, sl: int):
            import jax.numpy as jnp
            x = jax.random.randint(jax.random.PRNGKey(0), (bs, sl), 0, vocab)
            mask = jax.random.uniform(jax.random.PRNGKey(1), (bs, sl)) < 0.15
            return jnp.where(mask, mask_id, x)
        return _make

    def model_fn(self):
        import jax
        model = self._model
        def _fn(x):
            return jax.block_until_ready(model(x))
        return _fn

    def make_forward_fn(self, batch_size: int, seq_len: int):
        import jax
        model = self._model
        x = self.get_batch_fn()(batch_size, seq_len)
        def _fn():
            return jax.block_until_ready(model(x))
        return _fn

    def loss_fn(self):
        import jax
        import jax.numpy as jnp
        model   = self._model
        mask_id = getattr(self._config, "mask_token_id", 4)

        def _loss(batch, rng):
            x  = batch[:, :-1]
            rng, sub = jax.random.split(rng)
            mask = jax.random.uniform(sub, x.shape) < 0.15
            x_t  = jnp.where(mask, mask_id, x)
            out  = model(x_t, deterministic=True)
            logits = out.logits if hasattr(out, "logits") else out
            log_p  = jax.nn.log_softmax(logits, axis=-1)
            y      = batch[:, 1:]
            nll    = -log_p[jnp.arange(y.shape[0])[:, None],
                             jnp.arange(y.shape[1])[None, :], y]
            masked = nll * mask.astype(nll.dtype)
            return float(masked.sum() / mask.sum().clip(1)), None

        return _loss

    def logit_fn(self):
        model   = self._model
        mask_id = getattr(self._config, "mask_token_id", 4)
        import jax.numpy as jnp

        def _fn(x):
            mask = jnp.zeros(x.shape, dtype=bool)  # no masking for entropy
            x_t  = jnp.where(mask, mask_id, x)
            out  = model(x_t, deterministic=True)
            return out.logits if hasattr(out, "logits") else out

        return _fn


class _FlowAdapter(_ModelAdapter):
    def __init__(self, config, model, t5_encoder) -> None:
        self._config     = config
        self._model      = model
        self._t5_encoder = t5_encoder

    def model_config(self):
        return self._config

    def get_batch_fn(self):
        import jax
        vocab = self._config.vocab_size

        def _make(bs: int, sl: int):
            return jax.random.randint(jax.random.PRNGKey(0), (bs, sl), 0, vocab)

        return _make

    def model_fn(self):
        import jax
        import jax.numpy as jnp
        model     = self._model
        t5_encode = self._t5_encoder.encode

        def _fn(x):
            emb   = t5_encode(x)                          # [B, L, E]
            emb_n = model.encode(emb)                     # normalise
            b     = x.shape[0]
            # FlowMatchingTransformer.__call__ requires (z_t, x_prev, t,
            # cfg_scale, is_decode); a representative denoiser-mode step is
            # enough to profile latency/FLOPs (the scalar values of t /
            # cfg_scale / is_decode don't change the forward pass's cost).
            x_prev    = jnp.zeros_like(emb_n)
            t         = jnp.zeros(b)
            cfg_scale = jnp.ones(b)
            is_decode = jnp.zeros(b, dtype=bool)
            return jax.block_until_ready(
                model(emb_n, x_prev, t, cfg_scale, is_decode, deterministic=True)
            )

        return _fn

    def make_forward_fn(self, batch_size: int, seq_len: int):
        import jax
        model     = self._model
        t5_encode = self._t5_encoder.encode
        x         = self.get_batch_fn()(batch_size, seq_len)
        emb       = t5_encode(x)

        def _fn():
            emb_n = model.encode(emb)
            return jax.block_until_ready(model(emb_n, deterministic=True))

        return _fn

    def loss_fn(self):
        import jax

        from dantinox.training.flow_loss import flow_loss

        model     = self._model
        t5_encode = self._t5_encoder.encode
        elf_cfg   = getattr(self._config, "to_flow_config", lambda: self._config)()

        def _loss(batch, rng):
            x   = batch[:, :-1]
            emb = t5_encode(x)
            emb_n = model.encode(emb)
            rng, sub = jax.random.split(rng)
            loss, aux = flow_loss(model, emb_n, x, sub, elf_cfg)
            return float(loss), aux

        return _loss

    def logit_fn(self):
        model     = self._model
        t5_encode = self._t5_encoder.encode

        def _fn(x):
            emb   = t5_encode(x)
            emb_n = model.encode(emb)
            out   = model(emb_n, deterministic=True)
            return out.logits if hasattr(out, "logits") else out

        return _fn


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_adapter(run_dir: str, cfg: dict[str, Any]) -> _ModelAdapter:
    """Load the model from *run_dir* and wrap it in the right adapter."""
    import msgpack
    from flax import nnx
    from flax.serialization import _msgpack_ext_unpack

    from dantinox.core.config import Config

    model_type = cfg.get("model_type", "autoregressive")
    config     = Config.from_dict(cfg)
    rngs       = nnx.Rngs(42)

    # ── Build model skeleton ──────────────────────────────────────────────────
    if model_type == "elf":
        from dantinox.core.flow import FlowMatchingTransformer
        elf_cfg = config.to_flow_config()
        model   = FlowMatchingTransformer(elf_cfg, rngs=rngs)
    elif model_type == "diffusion":
        from dantinox.core.model import DiffusionTransformer
        model = DiffusionTransformer(config, rngs=rngs)
    else:
        from dantinox.core.model import Transformer
        if config.mla:
            config.inference = True
        model = Transformer(config, rngs=rngs)

    # ── Load weights ──────────────────────────────────────────────────────────
    weights_path = None
    for fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
        candidate = os.path.join(run_dir, fname)
        if os.path.exists(candidate):
            weights_path = candidate
            break

    if weights_path is not None:
        log.info("[RunsProfiler] loading weights from %s", weights_path)
        with open(weights_path, "rb") as f:
            state_dict = msgpack.unpackb(
                f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False,
            )
        _load_state_best_effort(model, state_dict, run_dir)
    else:
        log.warning("[RunsProfiler] no weights found in %s — random init", run_dir)

    # ── Wrap in adapter ───────────────────────────────────────────────────────
    if model_type == "elf":
        t5_enc = _load_t5_encoder(cfg.get("t5_model_name", "t5-base"))
        return _FlowAdapter(config, model, t5_enc)
    elif model_type == "diffusion":
        return _DiffusionAdapter(config, model)
    else:
        return _ARAdapter(config, model)


def _load_t5_encoder(model_name: str):
    """Load the frozen T5 encoder (same pattern as trainer.py)."""
    import logging as _logging
    for _noisy in ("httpx", "transformers", "huggingface_hub"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)
    import jax.numpy as jnp
    from transformers import AutoTokenizer, FlaxT5EncoderModel

    class _T5Enc:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model     = FlaxT5EncoderModel.from_pretrained(model_name)

        def encode(self, token_ids) -> jnp.ndarray:
            import numpy as np
            ids_np = np.array(token_ids, dtype=np.int32)
            out    = self.model(input_ids=ids_np)
            return jnp.array(out.last_hidden_state)

    return _T5Enc()


# ── Weight loading helpers ────────────────────────────────────────────────────


def _load_state_best_effort(model: Any, state_dict: dict, run_dir: str) -> None:
    """Load weights into *model*, silently skipping keys absent in the model.

    Old checkpoints may have been saved with a different model version
    (e.g. biased LayerNorm vs biasless, different attention structure).
    This function loads only the intersection of saved keys and current
    model keys, so the runner stays robust across architecture changes.
    """
    from flax import nnx

    state = nnx.state(model, nnx.Not(nnx.RngState))
    current = state.to_pure_dict()

    def _merge(cur: Any, saved: Any) -> Any:
        if not isinstance(cur, dict) or not isinstance(saved, dict):
            return saved
        result = dict(cur)
        for k, v in saved.items():
            if k in cur:
                result[k] = _merge(cur[k], v)
        return result

    merged  = _merge(current, state_dict)
    missing = _count_extra_keys(state_dict, current)
    if missing:
        log.warning(
            "[RunsProfiler] %s: skipped %d saved key(s) not in current model "
            "(architecture version mismatch)",
            os.path.basename(run_dir), missing,
        )

    state.replace_by_pure_dict(merged)
    nnx.update(model, state)


def _count_extra_keys(saved: Any, current: Any, depth: int = 0) -> int:
    """Count leaf keys present in *saved* but absent in *current*."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return 0
    count = 0
    for k, v in saved.items():
        if k not in current:
            count += 1 if not isinstance(v, dict) else _count_extra_keys(v, {})
        else:
            count += _count_extra_keys(v, current[k], depth + 1)
    return count


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_flat_config(run_dir: str) -> dict[str, Any]:
    """Load and flatten the config.yaml from *run_dir*."""
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    flat: dict[str, Any] = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    if not flat:
        flat = raw
    return flat


def _matches(cfg: dict[str, Any], filt: dict[str, Any]) -> bool:
    """Return True iff every key-value in *filt* matches *cfg*."""
    for k, expected in filt.items():
        actual = cfg.get(k)
        if actual is None:
            return False
        # coerce both sides to str for simple equality (handles int vs str "768")
        if str(actual) != str(expected):
            return False
    return True
