---
title: Diffusion Generation
---

# Diffusion Generation

## Fast-dLLM (recommended)

Block-wise generation with DualCache — 1.4–2.1× faster than prefix-only:

```python
from core.model import DiffusionTransformer
from core.generation import fast_dllm_generate
from core.diffusion import make_noise_schedule

model    = DiffusionTransformer.from_pretrained("runs/diff_mha_256d_12b_Dense")
schedule = make_noise_schedule(model.config)   # or: Config.from_yaml(...)

tokens = fast_dllm_generate(
    model,
    prefix    = prefix_ids,      # [B, T_prefix] — pass zeros for unconditional
    gen_len   = 256,
    schedule  = schedule,
    mask_token_id = 0,

    # Block-wise parameters
    block_size      = 32,        # default: 32
    steps_per_block = 20,        # denoising steps per block

    # Confidence-aware unmasking
    decoding_strategy    = "threshold",   # "threshold" | "factor"
    confidence_threshold = 0.9,
    factor               = 1.5,

    seed = 42,
)
# tokens: [B, gen_len]
```

---

## Simple MDLM Sampler

Full-sequence denoising without block-wise optimisation — slower but simpler:

```python
from core.generation import diffusion_generate

tokens = diffusion_generate(
    model, prefix, gen_len=128,
    schedule      = schedule,
    mask_token_id = 0,
    num_sampling_steps = 50,
    temperature   = 1.0,
    seed          = 42,
)
```

---

## Unconditional Generation

Pass an empty prefix (`T_prefix = 0`):

```python
import jax.numpy as jnp

prefix = jnp.zeros((batch_size, 0), dtype=jnp.int32)
tokens = fast_dllm_generate(model, prefix, gen_len=256, ...)
```

---

## Infilling

Diffusion supports native infilling: mask the positions you want filled,
condition on the rest.

```python
# x0: known tokens with 0 (MASK) at positions to fill
x0_masked = x0.at[:, 50:80].set(0)   # fill positions 50–79

tokens = diffusion_generate(
    model, prefix=x0_masked[:, :50],
    gen_len=30,   # only fill the masked region
    ...
)
```

---

## Decode Steps vs Quality

More `steps_per_block` improves generation quality at the cost of speed:

| `steps_per_block` | Relative quality | Relative speed |
|---|---|---|
| 5 | — | 4× baseline |
| **20** | ✓✓ | **2× baseline** |
| 50 | ✓✓✓ | baseline |
| 100 | ✓✓✓✓ | 0.5× baseline |

---

## Decoding to Text

Use the tokenizer saved in the run directory:

```python
from utils.tokenizer import load_tokenizer_from_file

tokenizer = load_tokenizer_from_file("runs/diff_mha_256d_12b_Dense/tokenizer.json")
text = tokenizer.decode(tokens[0].tolist())
print(text)
```
