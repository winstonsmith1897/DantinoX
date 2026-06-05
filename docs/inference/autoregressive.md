---
title: AR Generation
---

# Autoregressive Generation

## Generator (high-level)

`Generator` wraps a trained checkpoint and exposes three generation modes:

```python
from dantinox import Generator

gen = Generator("runs/ar_mha_256d_12b_Dense")   # local run dir
gen = Generator("my-org/dantinox-model")         # HuggingFace Hub
```

### Single prompt

```python
text = gen.generate(
    "Nel mezzo del cammin ",
    max_new_tokens = 200,
    temperature    = 1.0,
    top_p          = 0.9,
    use_cache      = True,
)
```

### Batched

```python
texts = gen.generate_batch(
    ["Prompt A", "Prompt B", "Prompt C"],
    max_new_tokens = 128,
)
```

### Streaming

```python
for chunk in gen.stream("Nel mezzo del cammin ", max_new_tokens=200):
    print(chunk, end="", flush=True)
```

---

## CLI

```bash
dantinox generate \
  --run_dir runs/ar_mha_256d_12b_Dense \
  --prompt "Nel mezzo del cammin " \
  --max_new_tokens 200 \
  --temperature 1.0 \
  --top_p 0.9 \
  --stream
```

---

## Sampling Strategies

| Strategy | CLI flags | API kwargs |
|---|---|---|
| Greedy | `--greedy` | `greedy=True` |
| Temperature | `--temperature 0.8` | `temperature=0.8` |
| Top-p (nucleus) | `--top_p 0.9` | `top_p=0.9` |
| Top-k | `--top_k 50` | `top_k=50` |

Strategies can be combined: `top_k=50, top_p=0.9` first restricts to top-50 tokens, then applies nucleus sampling.

---

## Low-level API

```python
from core.generation import generate
import jax.numpy as jnp

tokens = generate(
    model,
    prompt_ids,            # [B, T_prompt]  int32
    max_generations = 128,
    use_cache       = True,
    top_p           = 0.9,
    temperature     = 1.0,
    seed            = 42,
)
# tokens: [B, T_prompt + max_generations]
```
