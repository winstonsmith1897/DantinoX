# Profiling & Benchmarking Architecture

This page explains how `dantinox/profiling/` and `dantinox/benchmarking/` are designed and how they interact.

---

## Profiling

The profiling layer has two standalone components with no dependencies on training or paradigms.

### `count_flops` — analytical FLOPs

```python
from dantinox.profiling import count_flops, FLOPsBreakdown

flops = count_flops(config, seq_len=512, batch_size=4)
# FLOPsBreakdown(attention=..., ffn=..., embedding=..., total=...)
print(flops)   # human-readable with GFLOPs / TFLOPs scaling
```

No JAX, no model instance. Pure arithmetic on `ModelConfig` fields.

**Formulas** (per forward pass):

$$\text{Attention} = \left(4 \cdot 2BTD^2 + 2BT^2D\right) \cdot L$$
$$\text{FFN} = \left(2BT \cdot D \cdot (E \cdot D) \cdot s_\text{swiglu} + 2BT \cdot (E \cdot D) \cdot D\right) \cdot L$$
$$\text{Embedding} = 2BT \cdot V \cdot D$$

where $s_\text{swiglu} = 2$ if `use_swiglu` else $1$.

### `LatencyTracker` — wall-clock timing

`LatencyTracker` uses `jax.effects_barrier()` to block until all pending JAX operations flush before starting and stopping the timer. This gives accurate wall-clock measurements, not XLA dispatch latency.

```python
tracker = LatencyTracker(window=10_000)  # rolling window of last N samples

with tracker.measure(n_tokens=batch * seq_len):
    _ = model(x)

result = tracker.result()
# ProfilingResult: mean_ms, p50_ms, p99_ms, throughput_tps, n_samples
```

**Functional wrapper:**

```python
from dantinox.profiling import profile_fn

fast_generate = profile_fn(model.generate, tracker, n_tokens=256)
output = fast_generate(prompt, rng)  # records one sample automatically
```

---

## Benchmarking

The benchmarking system is a plugin framework: tasks are independent classes, the suite orchestrates them.

### `BenchmarkTask` — the plugin interface

```python
from dantinox.benchmarking import BenchmarkTask, BenchmarkResult

class MyTask(BenchmarkTask):
    name = "my_task"

    def run(self, paradigm, model, config, rng) -> BenchmarkResult:
        score = evaluate_something(model, config)
        return BenchmarkResult(task=self.name, metrics={"score": score})
```

### `BenchmarkSuite` — orchestrator

```python
from dantinox.benchmarking import BenchmarkSuite, BenchmarkConfig

suite = BenchmarkSuite(
    tasks=[ThroughputTask(), LatencyTask(), MyTask()],
    config=BenchmarkConfig(seq_lens=[128, 256, 512], n_measure=30),
)
report = suite.run(paradigm, model, save_csv="results.csv")
```

`BenchmarkSuite.default()` — returns `[ThroughputTask, LatencyTask, PerplexityTask]`.
`BenchmarkSuite.throughput_only()` — single-task variant for quick hardware checks.

### Built-in tasks

| Task | What it measures | Key metrics |
| :--- | :--- | :--- |
| `ThroughputTask` | tok/s vs seq-len sweep + batch-size sweep | `tps_seq{L}`, `peak_tps` |
| `LatencyTask` | Prefill latency + AR decode latency | `prefill_mean_ms`, `prefill_p99_ms`, `decode_tps` |
| `PerplexityTask` | Cross-entropy loss on validation data | `perplexity`, `eval_loss` |

### `SuiteReport` — the result type

```python
report.summary()          # human-readable string
report.to_dataframe()     # pandas DataFrame — one row per task
report.save("out.csv")    # CSV export
```

---

## Visualization

Charts are registered class-globally and auto-discovered by `Visualizer`:

```python
@Visualizer.register
class MyChart(Chart):
    name    = "my_chart"
    accepts = pd.DataFrame

    def _render_mpl(self, data, config, fig, ax):
        ax.plot(...)

Visualizer().render(df, charts=["my_chart"], out_dir="plots/")
```

`RenderConfig` controls backend (`"matplotlib"` | `"plotly"`), format (`"png"` | `"pdf"` | `"svg"`), resolution, size, and style preset (`"publication"` | `"dark"` | `"minimal"`).

Built-in charts: `TrainingCurveChart`, `ThroughputChart`, `ThroughputBatchChart`, `LatencyChart`, `ParetoChart`, `RadarChart`.
