---
title: FAQ & Troubleshooting
---

# FAQ & Troubleshooting

---

## Installation

**`pip install "jax[cuda12]"` fails — which version of CUDA do I need?**

JAX requires CUDA 12.x and cuDNN 8.9+. Check your driver and toolkit versions:

```bash
nvidia-smi          # shows driver version and maximum supported CUDA
nvcc --version      # shows installed CUDA toolkit version
```

If your driver supports CUDA 11 only, install `jax[cuda11_pip]` instead. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the full compatibility matrix.

---

**`jax.devices()` returns only CPU even after installing `jax[cuda12]`.**

This usually means JAX cannot find the CUDA libraries at runtime. Try:

```bash
# Verify the JAX CUDA plugin is installed
pip show jaxlib | grep Location
ls $(python -c "import jaxlib; print(jaxlib.__file__.rsplit('/',1)[0])")/cuda*

# Force GPU selection
CUDA_VISIBLE_DEVICES=0 python -c "import jax; print(jax.devices())"
```

If running inside a container, ensure the container has access to the GPU (`--gpus all` for Docker).

---

## Training

**`ValueError: dim (512) must equal n_heads * head_size`**

The constraint `dim = n_heads × head_size` is always enforced. For example:

| `dim` | `n_heads` | `head_size` | Valid |
| :--- | :--- | :--- | :--- |
| 512 | 16 | 32 | ✓ |
| 512 | 8 | 64 | ✓ |
| 512 | 12 | 32 | ✗ (12 × 32 = 384 ≠ 512) |

---

**`ValueError: n_heads must be divisible by kv_heads`**

For GQA, `n_heads` must be an integer multiple of `kv_heads`. Common valid combinations:

| `n_heads` | `kv_heads` | Ratio |
| :--- | :--- | :--- |
| 16 | 4 | 4× |
| 16 | 8 | 2× |
| 16 | 1 | 16× |

Set `kv_heads = n_heads` for standard MHA.

---

**Training loss diverges or becomes NaN.**

Most common causes:

1. **Learning rate too high** — run `dantinox find-lr` first and pick the LR just before the minimum.
2. **No gradient clipping** — set `grad_clip: 1.0` in the config (enabled by default).
3. **bfloat16 overflow** — bfloat16 has a smaller dynamic range than float32; lower the LR by 2–4× when enabling `use_bf16: true`.
4. **Dataset encoding issue** — ensure the corpus is UTF-8 encoded and not empty.

---

**Training is much slower than expected.**

- **XLA compilation** — the first few steps are slow because JAX is tracing and compiling the computation graph. This is expected. Subsequent steps use the cached XLA executable.
- **Set `JAX_COMPILATION_CACHE_DIR`** to persist XLA compilations across runs: `export JAX_COMPILATION_CACHE_DIR=~/.jax_cache`
- **Flash Attention** — enable `use_flash_attention: true` (MHA/GQA only, JAX ≥ 0.4.25). This fuses the attention kernel and reduces memory bandwidth pressure.
- **Gradient checkpointing** — `gradient_checkpointing: true` trades compute for memory; disable it if you have enough VRAM.
- **n_devices** — set `n_devices: 0` to use all available GPUs automatically.

---

**`OOM` (out of memory) on GPU.**

Try these in order:

1. Reduce `batch_size` and increase `grad_accum` proportionally (effective batch is unchanged).
2. Enable `use_bf16: true` — halves parameter memory.
3. Enable `gradient_checkpointing: true` — recomputes activations instead of caching them.
4. Reduce `max_context` — KV-cache size scales linearly with sequence length.
5. Switch to GQA (`kv_heads < n_heads`) or MLA (`attention_type: mla`) — both reduce KV-cache memory significantly.

---

## Inference

**`Generator` is slow on the first call.**

The first call triggers XLA JIT compilation of the generation graph. Subsequent calls use the cached executable and are fast. Compilation time scales with model size and sequence length; for a 256d/12-layer model it takes ~30 seconds on the first call.

To amortise this across processes:

```bash
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
```

---

**Streaming generation (`Generator.stream()`) produces tokens with variable latency.**

This is expected: XLA compiles a single-step function that runs at constant latency, but Python overhead from yielding tokens and printing to the terminal adds variability. For throughput measurement, use `generate_batch` instead.

---

**`Generator.from_pretrained` raises `FileNotFoundError`.**

Ensure the run directory contains all three required files: `config.yaml`, `weights.msgpack`, and `tokenizer.json`. If any is missing, re-run training or `dantinox pull` to restore the checkpoint.

---

## MLA / Attention

**What is the difference between `mla: true` and `attention_type: mla`?**

They are equivalent. `attention_type: mla` is the newer, explicit field. `mla: true` is the legacy boolean flag maintained for backward compatibility. `Config.__post_init__` synchronises them automatically: setting either one updates the other.

---

**When should I set `inference: true` in the MLA config?**

Only when running generation (not training). `inference: true` activates weight absorption — pre-fusing $W_{UQ}^\top W_{UK}$ and $W_{UV} W_O$ so that the full multi-head $K$/$V$ tensors are never materialised. The saved weights are identical; only the computation graph changes.

```yaml
# Training config
mla:
  mla: true
  inference: false   # ← default; use during training

# Inference config (reload the same weights.msgpack)
mla:
  mla: true
  inference: true    # ← activates weight absorption
```

---

## HuggingFace Hub

**`dantinox push` fails with `401 Unauthorized`.**

Your token has expired or lacks `write` scope. Re-authenticate:

```bash
huggingface-cli login
```

Or pass the token explicitly:

```bash
dantinox push --run_dir runs/my_run --repo my-org/model --token hf_...
```

---

**The repository does not appear in my HuggingFace profile after `push`.**

By default `push` creates a **public** repository. If you passed `--private true`, the repository exists but is only visible when logged in. Check [huggingface.co/my-org](https://huggingface.co/my-org) while authenticated.

---

## Further Help

- **GitHub Issues**: [github.com/winstonsmith1897/DantinoX/issues](https://github.com/winstonsmith1897/DantinoX/issues)
- **Changelog**: [What changed in the latest release](changelog.md)
- **API Reference**: [Full API documentation](api.md)
