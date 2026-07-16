# `dantinox` — Top-Level API

The top-level `dantinox` package exposes a three-level API. Import `dantinox as dx` to access all public symbols.

---

## Low-code functional API

These functions are the Level-1 entry points — no class instantiation required.

```{eval-rst}
.. autofunction:: dantinox.fit
```

```{eval-rst}
.. autofunction:: dantinox.train
```

```{eval-rst}
.. autofunction:: dantinox.build
```

```{eval-rst}
.. autofunction:: dantinox.profile
```

```{eval-rst}
.. autofunction:: dantinox.load
```

```{eval-rst}
.. autofunction:: dantinox.quick_generate
```

---

## Re-exported symbols

The following symbols are importable directly from `dantinox`:

### Configs
- `ModelConfig` — model architecture configuration
- `TrainingConfig` — training hyperparameters
- `Config` — legacy unified config (backward-compat)

`ELFConfig` (flow-matching architecture config, aliased to `FlowMatchingConfig`)
is **not** re-exported at the top level — import it from the submodule
instead: `from dantinox.core.config import FlowMatchingConfig` (or the
deprecated alias `ELFConfig` from the same module).

### Paradigms
- `Paradigm` — unified paradigm dispatcher (selects AR/discrete/continuous/embedder from `ModelConfig.paradigm`)
- `ARParadigm` — autoregressive
- `DiscreteParadigm` — masked diffusion (LLaDA formulation)
- `ContinuousParadigm` — continuous flow-matching
- `EmbedderParadigm` — contrastive text-embedding paradigm

`DiscreteConfig` (masked-diffusion hyperparameters) is **not** re-exported at
the top level — import it directly: `from dantinox.paradigms.diffusion.discrete import DiscreteConfig`.

### Training
- `Trainer` — paradigm-agnostic training harness
- `build_optimizer` — optimizer factory
- `build_schedule` — LR schedule factory

### Profiling
- `count_flops` — analytical FLOPs estimator
- `FLOPsBreakdown` — per-component FLOPs result
- `LatencyTracker` — wall-clock latency measurement
- `ProfilingResult` — aggregated latency statistics
- `profile_fn` — functional latency wrapper

### Benchmarking
- `BenchmarkSuite` — task orchestrator
- `BenchmarkTask` — plugin base class
- `BenchmarkConfig` — suite configuration
- `BenchmarkResult` — per-task result
- `SuiteReport` — aggregated report
- `ThroughputTask`, `LatencyTask`, `PerplexityTask` — built-in tasks

### Visualization
- `Visualizer` — chart registry and renderer
- `Chart` — chart ABC
- `RenderConfig` — rendering options
- `TrainingCurveChart`, `ThroughputChart`, `ThroughputBatchChart` — built-in charts
- `LatencyChart`, `RadarChart`, `ParetoChart` — built-in charts
