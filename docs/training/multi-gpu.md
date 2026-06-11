---
title: Multi-GPU Training
---

# Multi-GPU Data-Parallel Training

DantinoX uses JAX's SPMD mesh sharding for data-parallel training.
No `pmap`, no manual AllReduce — XLA handles gradient synchronisation.

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
