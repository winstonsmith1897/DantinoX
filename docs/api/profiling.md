# `dantinox.profiling`

The profiling module has no dependencies on training or paradigms. Both utilities can be used standalone.

---

## FLOPs estimation

::: dantinox.profiling.counter.count_flops
    options:
      show_source: true
      heading_level: 3

::: dantinox.profiling.counter.FLOPsBreakdown
    options:
      show_source: true
      heading_level: 3

### FLOPs formulas

$$
\text{Attention} = \left(4 \cdot 2BTD^2 + 2BT^2D\right) \times L
$$
$$
\text{FFN} = \left(2BT \cdot D \cdot ED \cdot s_\text{swiglu} + 2BT \cdot ED \cdot D\right) \times L
$$
$$
\text{Embedding} = 2BT \cdot V \cdot D
$$

where $B$ = batch, $T$ = seq len, $D$ = dim, $E$ = expansion, $L$ = layers, $V$ = vocab, $s_\text{swiglu} = 2$ if SwiGLU else $1$.

---

## Latency tracking

::: dantinox.profiling.tracker.LatencyTracker
    options:
      show_source: true
      members:
        - __init__
        - measure
        - record
        - result
        - reset

::: dantinox.profiling.tracker.ProfilingResult
    options:
      show_source: true
      heading_level: 3

::: dantinox.profiling.tracker.profile_fn
    options:
      show_source: true
      heading_level: 3

---

## Usage example

```python
from dantinox.profiling import LatencyTracker, count_flops, profile_fn
from core.config import ModelConfig

# --- Analytical FLOPs (no model instance needed) ---
cfg   = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32_000)
flops = count_flops(cfg, seq_len=512, batch_size=4)
print(flops)
# FLOPs breakdown:
#   attention : 12.88 GFLOPs
#   ffn       : 25.77 GFLOPs
#   embedding : 0.13  GFLOPs
#   total     : 38.78 GFLOPs

# --- Wall-clock latency (JAX barrier-accurate) ---
tracker = LatencyTracker()

with tracker.measure(n_tokens=4 * 512):
    _ = model(x)

result = tracker.result()
print(f"mean: {result.latency_mean_ms:.1f} ms")
print(f"p99:  {result.latency_p99_ms:.1f} ms")
print(f"tps:  {result.throughput_tps:,.0f} tok/s")

# --- Functional wrapper ---
instrumented_generate = profile_fn(model.generate, tracker, n_tokens=256)
output = instrumented_generate(prompt, rng)
```

!!! warning "JAX synchronization"
    `LatencyTracker.measure()` calls `jax.effects_barrier()` before and after the measured call. This ensures all XLA-compiled operations have completed before the timer stops. Without this, JAX's asynchronous dispatch would cause the measured time to reflect only dispatch latency, not actual computation.
