---
title: Masked Diffusion LM
---

# Masked Diffusion Language Model

This tutorial trains a Masked Diffusion Language Model (MDLM) — a non-autoregressive Transformer that generates text by iteratively denoising a fully masked sequence. Unlike autoregressive models, diffusion LMs attend bidirectionally and support native **infilling** without retraining.

---

## How Masked Diffusion Works

### Forward process

During training, tokens are masked according to a noise schedule. At time step $t \in [0, T]$, each token is independently replaced with `[MASK]` with probability $\bar{\alpha}_t$:

$$q(x_t \mid x_0) = \text{Bernoulli}(1 - \bar{\alpha}_t)$$

Three noise schedules are available: `cosine` (default), `linear`, and `sqrt`.

### Reverse process

The model $p_\theta(x_0 \mid x_t, t)$ learns to predict the original token at every masked position simultaneously. Training minimises the masked cross-entropy loss over `[MASK]` positions only:

$$\mathcal{L} = -\mathbb{E}_{t, x_t} \left[ \sum_{i: x_t^i = \text{[MASK]}} \log p_\theta(x_0^i \mid x_t, t) \right]$$

### Generation

Starting from a fully masked sequence $x_T$, the model iteratively denoises over `num_sampling_steps` steps (default: 50), which is much fewer than the full $T = 1000$ training steps. DantinoX supports the **Fast-dLLM DualCache** strategy that reduces latency by ~2.1× over the naive sampler.

---

## 1. Config

Switch to diffusion mode with a single field:

```yaml
# configs/diffusion_tutorial.yaml

model:
  model_type: diffusion    # ← key difference from AR
  dim: 256
  n_heads: 8
  head_size: 32
  num_blocks: 6
  max_context: 128
  kv_heads: 8              # diffusion uses full attention (no GQA needed)
  norm_type: rmsnorm
  use_swiglu: true

  diffusion_steps: 1000        # forward-process steps T
  noise_schedule: cosine       # "cosine" | "linear" | "sqrt"
  num_sampling_steps: 50       # reverse steps at inference (DDIM-style)
  time_emb_dim: 256            # time-embedding MLP output dimension
  mask_token_id: 0             # vocabulary ID reserved for [MASK]

attention:
  use_rotary_pos: true
  use_flash_attention: false   # Flash Attention requires causal mask; diffusion uses full attn

training:
  optimizer: adamw
  lr: 3e-4
  lr_schedule: cosine
  warmup_steps: 200
  grad_clip: 1.0
  use_bf16: true
  batch_size: 64
  grad_accum: 4
  epochs: 3

tokenizer:
  tokenizer_type: char

data:
  dataset_source: local
```

!!! warning "Flash Attention and diffusion"
    Flash Attention (`use_flash_attention: true`) uses a causal mask and is only valid for autoregressive models. Diffusion transformers use **full bidirectional attention** — leave `use_flash_attention: false`.

---

## 2. Train

```bash
dantinox train \
  --config configs/diffusion_tutorial.yaml \
  --data_path data/corpus.txt
```

Or via the Python API:

```python
from core.config import Config
from dantinox.trainer import Trainer

config  = Config.from_yaml("configs/diffusion_tutorial.yaml")
run_dir = Trainer(config).fit("data/corpus.txt")
```

Diffusion training is typically slower per step than AR because the loss is computed over all masked positions (not just the next token). With `num_blocks=6` and `dim=256`, one epoch on a ~1 M-token corpus takes about 15 minutes on a T4.

---

## 3. Generation

### Simple MDLM sampler

```python
from core.model import DiffusionTransformer
from core.generation import diffusion_generate
from core.config import Config
import jax

config = Config.from_yaml(f"{run_dir}/config.yaml")
model  = DiffusionTransformer.from_pretrained(run_dir)

tokens = diffusion_generate(
    model,
    config,
    prompt_tokens=None,          # None = generate from scratch
    seq_len=128,
    num_steps=config.num_sampling_steps,
    key=jax.random.PRNGKey(0),
)
```

### Fast-dLLM DualCache (recommended)

Fast-dLLM reduces decoding latency by maintaining two KV caches: one for the stable prefix (tokens with high confidence that have already converged) and one for the active denoising region. This avoids recomputing attention for positions that are already determined.

```python
from core.generation import fast_dllm_generate

tokens = fast_dllm_generate(
    model,
    config,
    seq_len=128,
    num_steps=config.num_sampling_steps,
    confidence_threshold=0.9,    # a token is "frozen" once its confidence exceeds this
    key=jax.random.PRNGKey(0),
)
```

### Infilling

Diffusion LMs support native infilling: provide a prefix and suffix and let the model fill in the middle.

```python
from utils.tokenizer import load_tokenizer_from_file

tokenizer = load_tokenizer_from_file(f"{run_dir}/tokenizer.json")

prefix = "The quick brown"
suffix = "over the lazy dog."

prefix_ids = tokenizer.encode(prefix)
suffix_ids = tokenizer.encode(suffix)

tokens = fast_dllm_generate(
    model,
    config,
    prefix_tokens=prefix_ids,
    suffix_tokens=suffix_ids,
    seq_len=len(prefix_ids) + 5 + len(suffix_ids),   # 5 tokens to fill
    key=jax.random.PRNGKey(42),
)
print(tokenizer.decode(tokens))
# → "The quick brown fox jumps over the lazy dog."
```

Autoregressive models require re-prompting or fine-tuning for infilling. Diffusion models handle it natively because they condition on the full context bidirectionally.

---

## 4. Comparing AR and Diffusion

| Property | Autoregressive | Masked Diffusion |
| :--- | :--- | :--- |
| Decoding | Sequential (1 token/step) | Parallel (all positions/step) |
| Throughput at large batch | Medium | High |
| Latency at batch size 1 | Low | Medium (50+ steps) |
| Infilling | Requires re-prompting | Native |
| Long-range coherence | Unidirectional | Bidirectional |
| KV cache | Standard | DualCache (Fast-dLLM) |

See [AR vs. Diffusion](../paradigms/comparison.md) for detailed benchmark results.

---

## Next Steps

| Goal | Reference |
| :--- | :--- |
| Full technical details on Fast-dLLM | [Fast-dLLM DualCache](../paradigms/fast-dllm.md) |
| Benchmark AR vs. Diffusion throughput | [Benchmarks](../benchmarks.md) |
| Push the model to HuggingFace Hub | [Pushing to HuggingFace Hub](hub.md) |
