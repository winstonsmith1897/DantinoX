# `dantinox` — Top-Level API

The top-level `dantinox` package exposes a three-level API. Import `dantinox as dx` to access all public symbols.

---

## Low-code functional API

These functions are the Level-1 entry points — no class instantiation required.

::: dantinox.fit
    options:
      show_source: true
      heading_level: 3

::: dantinox.train
    options:
      show_source: true
      heading_level: 3

::: dantinox.build
    options:
      show_source: true
      heading_level: 3

::: dantinox.profile
    options:
      show_source: true
      heading_level: 3

::: dantinox.load
    options:
      show_source: true
      heading_level: 3

::: dantinox.quick_generate
    options:
      show_source: true
      heading_level: 3

---

## Re-exported symbols

The following symbols are importable directly from `dantinox`:

### Configs
- `ModelConfig` — model architecture configuration
- `TrainingConfig` — training hyperparameters
- `ELFConfig` — ELF flow-matching configuration
- `Config` — legacy unified config (backward-compat)

### Paradigms
- `Paradigm` — abstract base
- `ARParadigm` — autoregressive
- `DiscreteParadigm` — LLaDA-style masked diffusion
- `DiscreteConfig` — diffusion hyperparameters
- `ContinuousParadigm` — ELF flow-matching

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
