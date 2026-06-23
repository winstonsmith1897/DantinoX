# Architecture

DantinoX is organized in three decoupled layers. Understanding this layering is the key to extending the library effectively.

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 1 API  dx.fit() · dx.train() · dx.quick_generate()       │
├─────────────────────────────────────────────────────────────────┤
│  Paradigms    ARParadigm · DiscreteParadigm · ContinuousParadigm│
│               (own: loss_fn, generate, build_model)             │
├─────────────────────────────────────────────────────────────────┤
│  Training     Trainer · build_optimizer · build_schedule        │
│               (paradigm-agnostic: calls loss_fn, nothing else)  │
├─────────────────────────────────────────────────────────────────┤
│  Profiling &  LatencyTracker · count_flops · BenchmarkSuite     │
│  Benchmarking BenchmarkTask plugins · Visualizer chart registry │
├─────────────────────────────────────────────────────────────────┤
│  Core         Transformer · Attention (MHA/GQA/MLA)             │
│               MLP · MoE · LoRA · ELFTransformer · sharding      │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Core layer

`core/` contains the raw neural-network primitives. Nothing in `core/` knows about training objectives or paradigms — it is purely forward-pass logic.

| Module | Contents |
| :--- | :--- |
| `core/config.py` | `Config`, `ModelConfig`, `TrainingConfig`, `ELFConfig` — single source of truth |
| `core/model.py` | `Transformer`: embedding → blocks → output norm → LM head |
| `core/attention.py` | `Attention`: MHA, GQA, MLA with RoPE, KV cache, Flash Attention |
| `core/block.py` | `Block`: pre-norm residual · Attention + FFN |
| `core/mlp.py` | Dense MLP with SwiGLU/GELU |
| `core/moe.py` | Sparse MoE with top-K routing and load-balancing loss |
| `core/lora.py` | `LoRALinear`, `LoRAParam` — type-level weight freezing |
| `core/elf.py` | `ELFTransformer`, `ELFEmbedder` — continuous flow-matching |
| `core/diffusion.py` | Noise schedules, `corrupt()`, `masked_cross_entropy()` |
| `core/generation.py` | AR decode loop, diffusion reverse pass, ELF denoising |
| `core/sharding.py` | `make_mesh`, `replicate`, `shard_batch` — multi-GPU SPMD |

For the deep-dive on individual layers (MLA math, RoPE, Flash Attention, LoRA, multi-GPU), see [Core Layers](architecture/core.md).

---

## The Paradigm layer

A `Paradigm` is a thin wrapper that defines *how to train and generate* with a core model. It exposes exactly three methods:

```python
class Paradigm(ABC):
    def build_model(self, rngs: nnx.Rngs) -> Any:
        """Construct the NNX model — called once by the Trainer."""

    def loss_fn(self, model, batch: jnp.ndarray, rng) -> tuple[jnp.ndarray, dict]:
        """Compute scalar loss + metrics dict — differentiated by the Trainer."""

    def generate(self, model, prompt: jnp.ndarray, rng, **kwargs) -> jnp.ndarray:
        """Decode a token sequence from a prompt prefix."""
```

The Trainer calls *only* `loss_fn` and nothing else about the model. This is the key design invariant: **all paradigm-specific logic (masking, noise schedules, ELF branches, CFG) lives in the Paradigm, never in the Trainer.**

!!! note "Why `model` is passed explicitly to `loss_fn`"
    `nnx.value_and_grad` differentiates with respect to the first argument. By accepting `model` explicitly, `loss_fn` is directly differentiable without the `Paradigm` needing to be an NNX module or store the model as state.

    ```python
    # Inside Trainer._step:
    def _loss(m):
        return paradigm.loss_fn(m, batch, rng)

    (loss, metrics), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
    ```

### Built-in paradigms

| Paradigm | Training objective | Noise / corruption |
| :--- | :--- | :--- |
| `ARParadigm` | Cross-entropy on shifted targets (teacher-forcing) | None |
| `DiscreteParadigm` | `(1/t)`-weighted masked cross-entropy (LLaDA) | Random token masking at rate `p(t)` |
| `ContinuousParadigm` | Flow-matching MSE + CE (ELF) | `z_t = t·x + (1−t)·ε`, ε ~ N(0,I) |

See [Paradigm System](architecture/paradigm-system.md) for the full design rationale, and [Generation Paradigms](paradigms/index.md) for usage documentation.

---

## The Training layer

`dantinox/training/` contains the paradigm-agnostic infrastructure:

### `Trainer`

The `Trainer` owns:

- data loading (`_load_tokens` supports local files + HuggingFace `datasets`)
- model construction via `paradigm.build_model()`
- JIT-compiled training step (`@nnx.jit` — fuses grad + update in one XLA kernel)
- multi-device replication (`core.sharding`)
- checkpointing (best + latest `checkpoint_*.msgpack`)
- CSV metric logging

```
Trainer.fit(data_source)
  ├── _load_tokens()          → flat list[int]
  ├── paradigm.build_model()  → NNX model
  ├── build_optimizer()       → nnx.Optimizer
  ├── for epoch:
  │     for step:
  │       _step(model, optimizer, batch, rng)
  │         ├── nnx.value_and_grad(paradigm.loss_fn)(model)
  │         └── optimizer.update(grads)
  └── _save_checkpoint()
```

### `build_optimizer`

```python
from dantinox.training.optimizer import build_optimizer

optimizer = build_optimizer(model, config, total_steps)
# config.optimizer: "adamw" | "adafactor" | "lion" | "adam" | "muon"
# config.lr_schedule: "cosine" | "linear" | "constant" | "wsd"
```

When LoRA is active, `build_optimizer` automatically masks gradients so only `LoRAParam` variables are updated — base weights are frozen at the type level.

For full optimizer and schedule documentation, see [Optimizers & Schedules](training/optimizers.md).

---

## The Profiling & Benchmarking layer

### Profiling

Two standalone utilities with no training dependencies:

**`count_flops(config, seq_len, batch_size)`** — analytical FLOPs estimate, returns `FLOPsBreakdown(attention, ffn, embedding, total)`. No model instance required.

**`LatencyTracker`** — accumulates wall-clock measurements with `jax.effects_barrier()` synchronization for accuracy. Computes mean, p50, p99 latencies and tokens/s throughput.

```python
tracker = LatencyTracker()
with tracker.measure(n_tokens=batch * seq_len):
    _ = model(x)
print(tracker.result())   # ProfilingResult
```

### Benchmarking

The benchmarking system is built around two ABCs:

**`BenchmarkTask`** — one task, one `run()` method, one `BenchmarkResult`:

```python
class MyTask(BenchmarkTask):
    name = "my_task"
    def run(self, paradigm, model, config, rng) -> BenchmarkResult:
        ...
```

**`BenchmarkSuite`** — orchestrates a list of tasks, owns timing/logging/CSV export:

```python
report = BenchmarkSuite.default().run(paradigm, model, save_csv="results.csv")
```

Built-in tasks: `ThroughputTask`, `LatencyTask`, `PerplexityTask`.

### Visualization

Charts are registered via a class-level decorator and auto-discovered by `Visualizer`:

```python
@Visualizer.register
class MyChart(Chart):
    name    = "my_chart"
    accepts = pd.DataFrame

    def _render_mpl(self, data, config, fig, ax):
        ax.plot(data["step"], data["loss"])
```

```python
Visualizer().render(df, charts=["my_chart", "throughput"], out_dir="plots/")
```

For deeper documentation see [Architecture: Profiling & Benchmarking](architecture/profiling.md).

---

## Data flow summary

```
data_source (str)
      │
      ▼
 _load_tokens()  ──────────────────────────────► list[int]
                                                     │
paradigm.build_model(rngs)                           │ _sample_batch()
      │                                              ▼
      ▼                                        jnp.ndarray [B, T+1]
  NNX model ──────────────────────────────────────── │
      │                                              │
      └──── nnx.value_and_grad(paradigm.loss_fn) ◄──┘
                        │
                        ▼
                  (loss, metrics), grads
                        │
                        ▼
               optimizer.update(grads)
                        │
                        ▼
               _save_checkpoint() ──► runs/<timestamp>/checkpoint_best.msgpack
```
