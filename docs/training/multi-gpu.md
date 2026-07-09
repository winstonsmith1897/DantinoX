---
title: Multi-GPU Training
---

# Multi-GPU Training: Data and Tensor Parallelism

DantinoX uses JAX's SPMD mesh sharding for multi-GPU training, and can
compose **Data Parallelism (DP)** with **Tensor Parallelism (TP)** on a
single 2-D mesh. No `pmap`, no manual AllReduce — XLA handles gradient
synchronisation and (for TP) the intra-layer all-reduce.

---

## Configuration

```yaml
training:
  n_devices: 2      # 0 = use all available GPUs, 1 = single device
  batch_size: 64    # must be divisible by n_devices
  use_bf16: true    # recommended for multi-GPU runs
```

```bash
dantinox train \
  --config configs/default_config.yaml \
  --n_devices 2 \
  --batch_size 64
```

---

## Sharding Strategy

| Tensor | Sharding |
|---|---|
| Model weights | Replicated on every device |
| Input batch | Sharded along axis 0 (each device gets `batch / n_devices` samples) |
| Gradients | AllReduced automatically by XLA |

Each device computes a full forward + backward pass on its shard.
XLA fuses the gradient AllReduce into the compiled program — zero
manual collective calls.

---

## GPU Selection

Control which GPUs are used via `CUDA_VISIBLE_DEVICES`:

```bash
# Use GPUs 0 and 1 for training, GPU 2 for benchmarks
CUDA_VISIBLE_DEVICES=0,1 dantinox train --config configs/diffusion_base.yaml
```

The training suite scripts set this automatically:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ar_suite.sh
```

---

## Scaling Rules

When increasing `n_devices`, scale `batch_size` proportionally to keep
the per-device batch size (and thus gradient noise) constant:

| `n_devices` | `batch_size` | Per-device batch | Effective LR |
|---|---|---|---|
| 1 | 32 | 32 | base LR |
| 2 | 64 | 32 | base LR |
| 4 | 128 | 32 | base LR |
| 8 | 256 | 32 | base LR (or ×√8 with linear scaling) |

---

## Low-level API

```python
from dantinox.core.sharding import make_mesh, replicate, shard_batch, num_devices

mesh = make_mesh(n_devices=4)
print(f"Training on {num_devices(mesh)} GPUs")

# Replicate any pytree to all devices
replicated_state = replicate(nnx.state((model, optimizer)), mesh)

# Shard a batch along axis 0
x_sharded = shard_batch(x, mesh)   # [batch, seq_len]
```

---

## Tensor Parallelism (DP × TP)

Set `tp_size > 1` (on `ModelConfig`, `TrainingConfig`, or the legacy `Config`)
to shard individual weight matrices across devices, Megatron-style, in
addition to (or instead of) data parallelism:

```python
train_cfg = dx.TrainingConfig(
    lr=3e-4, epochs=5,
    n_devices=8,   # total devices
    tp_size=2,     # 2-way tensor parallel → n_dp = n_devices / tp_size = 4
)
```

```bash
dantinox train --config configs/large.yaml --n_devices 8 --tp_size 2
```

**Mesh shape.** With `tp_size=1` (the default), the `Trainer` builds the same
1-D data-parallel mesh described above via `make_mesh`. With `tp_size > 1`,
it instead builds a 2-D `(data, model)` mesh via
`core.sharding.make_tp_mesh(n_tp=tp_size, n_dp=n_devices // tp_size)`, and
requires `n_devices` to be evenly divisible by `tp_size` (falls back to the
largest divisor otherwise, with a warning).

**Weight sharding.** `core.sharding.apply_tp_sharding(model, mesh)` shards
the backbone's linear layers Megatron-style, *before* the first JIT-compiled
forward pass:

| Layer | Parallelism | Sharded axis |
|---|---|---|
| `qkv`, `up_proj` | Column-parallel | Output (kernel's last axis) |
| `o_proj`, `down_proj` | Row-parallel | Input (kernel's first axis) |

Row-parallel biases are pre-scaled by `1/tp_size` so that the intra-layer
all-reduce (summing each device's partial output) reconstructs the correct
bias exactly once rather than `tp_size` times. `core.sharding.in_mesh_context()`
is what row-parallel attention/FFN layers check at trace time to decide
whether to emit that all-reduce at all — outside a TP mesh (`tp_size=1`),
it's skipped entirely and layers behave as plain dense layers.

**When to use TP vs. DP.** Data parallelism replicates the full model on
every device and only shards the batch — simple and usually fastest, but
bounded by how big a model fits on one device. Tensor parallelism shards the
weights themselves, letting you train models too large for a single device's
memory, at the cost of the extra intra-layer communication. `tp_size` and
`n_devices` compose: `n_devices=8, tp_size=2` gives 4-way data parallelism
over 2-way tensor-parallel model replicas.
