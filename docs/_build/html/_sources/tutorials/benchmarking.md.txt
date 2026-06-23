# Tutorial: Benchmarking a Model

This tutorial covers the complete benchmarking workflow: profiling FLOPs, measuring latency, evaluating perplexity, and visualizing results.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/03_benchmarking.ipynb)

---

## Setup

```python
import dantinox as dx
from flax import nnx
import jax

# Build a small AR model for demonstration
cfg      = dx.ModelConfig(dim=256, n_heads=4, head_size=64, num_blocks=4,
                           vocab_size=8_000, causal=True)
paradigm = dx.ARParadigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))

print(f"Parameters: {paradigm.num_parameters(model):,}")
```

---

## Step 1: Analytical FLOPs

No model warmup or GPU required — just the config:

```python
flops = dx.profile(cfg, seq_len=512, batch_size=4)
print(flops)
```

```
FLOPs breakdown:
  attention : 1.34 GFLOPs
  ffn       : 2.68 GFLOPs
  embedding : 0.02 GFLOPs
  total     : 4.04 GFLOPs
```

---

## Step 2: Wall-clock latency

```python
from dantinox.profiling import LatencyTracker
import jax.numpy as jnp

tracker = LatencyTracker()
x       = jax.random.randint(jax.random.PRNGKey(0), (4, 512), 0, 8_000)

# Warmup (important — first call triggers XLA compilation)
for _ in range(5):
    _ = model(x)

# Measure
for _ in range(20):
    with tracker.measure(n_tokens=4 * 512):
        _ = model(x)

result = tracker.result()
print(result)
```

```
Profiling (20 samples, 40,960 tokens):
  latency mean : 12.4 ms
  latency p50  : 12.1 ms
  latency p99  : 14.8 ms
  throughput   : 330,000 tokens/s
```

---

## Step 3: Full benchmark suite

```python
from dantinox.benchmarking import BenchmarkSuite

report = BenchmarkSuite.default().run(paradigm, model, save_csv="benchmark.csv")
print(report.summary())
```

The default suite runs:
1. **ThroughputTask** — tok/s vs sequence length (batch=1) and batch size (fixed length)
2. **LatencyTask** — prefill latency (all paradigms) + decode latency (AR only)
3. **PerplexityTask** — cross-entropy loss over random token batches

---

## Step 4: Custom benchmark config

```python
from dantinox.benchmarking import BenchmarkConfig, BenchmarkSuite

config = BenchmarkConfig(
    seq_lens    = [64, 128, 256, 512, 1024],
    batch_sizes = [1, 4, 16, 32],
    n_warmup    = 10,
    n_measure   = 50,
    eval_batches= 100,
)
report = BenchmarkSuite.default(config).run(paradigm, model)
```

---

## Step 5: Visualize results

```python
from dantinox.visualization import Visualizer
import pandas as pd

df = pd.read_csv("benchmark.csv")
paths = Visualizer().render(df, out_dir="plots/")
print(f"Saved {len(paths)} figures")
```

Or via CLI:

```bash
dantinox plot --in_csv benchmark.csv --out_dir plots/ --groups perf insights
```

---

## Step 6: Compare multiple models

```python
import pandas as pd
from dantinox.visualization import Visualizer

# Collect results from multiple runs
rows = []
for dim in [128, 256, 512]:
    cfg_i    = dx.ModelConfig(dim=dim, n_heads=4, head_size=dim // 4,
                               num_blocks=4, vocab_size=8_000, causal=True)
    par_i    = dx.ARParadigm(cfg_i)
    model_i  = par_i.build_model(nnx.Rngs(0))
    report_i = BenchmarkSuite.throughput_only().run(par_i, model_i)
    for result in report_i.results:
        rows.append({"dim": dim, **result.metrics})

df = pd.DataFrame(rows)
Visualizer().render(df, charts=["throughput"], out_dir="comparison_plots/")
```

---

## Next steps

- [API Reference: Benchmarking](../api/benchmarking.md) — full API documentation
- [Architecture: Profiling & Benchmarking](../architecture/profiling.md) — system design
- [Developer Guide: Custom Task](../guides/new-benchmark.md) — add your own metrics
