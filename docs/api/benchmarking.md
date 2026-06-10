# `dantinox.benchmarking`

The benchmarking module is a plugin framework: tasks are independent classes, the suite orchestrates them, and results aggregate into a structured report.

---

## Suite orchestrator

::: dantinox.benchmarking.suite.BenchmarkSuite
    options:
      show_source: true
      members:
        - __init__
        - run
        - default
        - throughput_only

---

## Plugin base class

::: dantinox.benchmarking.base.BenchmarkTask
    options:
      show_source: true
      members:
        - run

---

## Result types

::: dantinox.benchmarking.base.BenchmarkResult
    options:
      show_source: true

::: dantinox.benchmarking.base.SuiteReport
    options:
      show_source: true
      members:
        - to_dataframe
        - save
        - summary

::: dantinox.benchmarking.base.BenchmarkConfig
    options:
      show_source: true

---

## Built-in tasks

::: dantinox.benchmarking.tasks.throughput.ThroughputTask
    options:
      show_source: true
      members:
        - name
        - run

::: dantinox.benchmarking.tasks.latency.LatencyTask
    options:
      show_source: true
      members:
        - name
        - run

::: dantinox.benchmarking.tasks.perplexity.PerplexityTask
    options:
      show_source: true
      members:
        - name
        - run

---

## Quick reference

```python
from dantinox.benchmarking import BenchmarkSuite, BenchmarkConfig

# Default suite (Throughput + Latency + Perplexity)
report = BenchmarkSuite.default().run(paradigm, model)

# Custom suite
from dantinox.benchmarking.tasks.perplexity import PerplexityTask
suite  = BenchmarkSuite(
    tasks=[PerplexityTask("data/val.txt")],
    config=BenchmarkConfig(eval_batches=100, eval_seq_len=512),
)
report = suite.run(paradigm, model, save_csv="results.csv")
print(report.summary())
df = report.to_dataframe()
```

### Metrics produced by built-in tasks

| Task | Metric key | Description |
| :--- | :--- | :--- |
| `ThroughputTask` | `tps_seq{L}` | Tokens/s at sequence length L, batch=1 |
| `ThroughputTask` | `tps_bs{B}` | Tokens/s at batch size B |
| `ThroughputTask` | `peak_tps` | Maximum observed tokens/s |
| `LatencyTask` | `prefill_mean_ms` | Mean prefill latency |
| `LatencyTask` | `prefill_p99_ms` | 99th-percentile prefill latency |
| `LatencyTask` | `decode_mean_ms` | Mean single-step decode latency (AR only) |
| `LatencyTask` | `decode_tps` | Decode throughput (AR only) |
| `PerplexityTask` | `perplexity` | `exp(mean_ce_loss)` |
| `PerplexityTask` | `eval_loss` | Mean cross-entropy loss |
