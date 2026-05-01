# Inference & Generation

## Overview

`Generator` wraps a saved checkpoint and exposes three generation modes:

| Mode | Method | Use case |
| :--- | :--- | :--- |
| **Single prompt** | `generate(prompt)` | Interactive, one at a time |
| **Batched** | `generate_batch(prompts)` | Eval sets, data augmentation, throughput |
| **Streaming** | `stream(prompt)` | CLI UX, web SSE endpoints |

All modes use the static pre-allocated KV cache — zero dynamic shapes, zero recompilation after the first call.

---

## Loading a Checkpoint

`Generator` accepts either a **local run directory** or a **HuggingFace Hub repo ID**:

```python
from dantinox import Generator

gen = Generator("runs/run_20260101_120000")          # local
gen = Generator("my-org/dantinox-dante")             # Hub — downloads automatically
gen = Generator("my-org/private-model", token="hf_…")  # private Hub repo
```

`Generator` loads `config.yaml`, `tokenizer.json`, and `model_weights.msgpack`. The tokenizer vocabulary is read from the saved JSON — the original corpus is **not required**.

```bash
# CLI (local only — pull from Hub first if needed)
dantinox pull --repo my-org/dantinox-dante --local_dir runs/pulled
dantinox generate --run_dir runs/pulled --prompt "Nel mezzo del cammin "
```

!!! note "MLA checkpoints"
    For MLA models, `inference = True` is set automatically on load, activating weight absorption for the decode path. No flags needed.

---

## Single-Prompt Generation

```python
text = gen.generate(
    "Nel mezzo del cammin ",
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
)
print(text)
```

### CLI reference

```bash
dantinox generate \
  --run_dir runs/<run> \
  --prompt "Nel mezzo del cammin " \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 40 \
  --greedy          # override sampling with greedy decoding
  --no_cache        # disable KV cache (slower, useful for debugging)
```

### Argument reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | — | Input prefix |
| `max_new_tokens` | `int` | `150` | Tokens to generate |
| `greedy` | `bool` | `False` | Pick the argmax token at each step |
| `temperature` | `float` | `1.0` | Logit scale before softmax (lower = sharper) |
| `top_k` | `int` | `None` | Keep only the top-k tokens |
| `top_p` | `float` | `None` | Nucleus sampling — keep tokens summing to p |
| `use_cache` | `bool` | `True` | Enable static KV cache |

---

## Batched Generation

Run multiple prompts through the model in a **single forward pass**. Shorter prompts are left-padded with zeros so all share the same tensor shape. Throughput scales linearly with batch size on GPU.

```python
texts = gen.generate_batch(
    [
        "Nel mezzo del cammin di nostra vita",
        "Lasciate ogni speranza",
        "Per me si va ne la città dolente",
    ],
    max_new_tokens=100,
    temperature=0.9,
    top_p=0.95,
)
for t in texts:
    print(t)
    print("---")
```

!!! tip "Left-padding and quality"
    Shorter prompts are padded at the **left** with token id `0`. The causal attention mask prevents the model from "seeing" future tokens, but it will attend backward over the padding. For best quality, use prompts of similar length or prefer single-prompt generation when prompts vary greatly in length.

---

## Streaming Generation

`stream()` yields one decoded token at a time as soon as it is available — ideal for interactive CLI output or web server-sent events (SSE).

```python
print("", end="")
for chunk in gen.stream(
    "Nel mezzo del cammin ",
    max_new_tokens=300,
    temperature=0.85,
    top_k=50,
):
    print(chunk, end="", flush=True)
print()  # newline at end
```

### How it works

Unlike `generate()` which runs the entire token loop inside a single `jax.lax.fori_loop` (and cannot yield mid-loop), `stream()` uses two JIT-compiled step functions:

1. **`_stream_prefill`** — one full forward pass over the prompt, populating the KV cache.
2. **`_stream_decode`** — one token per call, reading from and writing into the cache at the current position.

Both are compiled once on the first call and reused for all subsequent tokens (the KV cache shape is fixed at `max_context`).

```python
# Conceptual pseudocode
logits, kv_cache = _stream_prefill(model, x_padded, init_kv_cache)
tok_id = sample(logits[:, T - 1, :])
yield tokenizer.decode([tok_id])

for pos in range(T, T + max_new_tokens - 1):
    logits, kv_cache = _stream_decode(model, [[tok_id]], kv_cache, pos)
    tok_id = sample(logits[:, 0, :])
    yield tokenizer.decode([tok_id])
```

!!! note "BPE streaming"
    With BPE tokenizers, each yielded chunk is a subword, not a character. The usual BPE byte replacements (`Ġ` → space, etc.) are applied per token, so output is readable as it streams, though subword boundaries may look unusual mid-word.

---

## HuggingFace Hub

### Direct loading (recommended)

Pass a Hub repo ID anywhere you would pass a run directory — `Generator` and `Transformer.from_pretrained` download the checkpoint automatically:

```python
from dantinox import Generator
from core import Transformer

# Generator — download + load in one call
gen = Generator("my-org/dantinox-dante")
print(gen.generate("Nel mezzo del cammin "))

# Private repo — pass a token
gen = Generator("my-org/private-model", token="hf_…")

# Specific branch/tag/commit
gen = Generator("my-org/dantinox-dante", revision="v1.0")

# Raw model for custom inference or fine-tuning
model = Transformer.from_pretrained("my-org/dantinox-dante")
model = Transformer.from_pretrained("my-org/dantinox-dante", token="hf_…", revision="v1.0")
```

```bash
# CLI — pull first, then generate
dantinox pull --repo my-org/dantinox-dante --local_dir runs/pulled
dantinox generate --run_dir runs/pulled --prompt "Nel mezzo del cammin "
```

### Sharing a checkpoint

```bash
# Upload from CLI
dantinox push --run_dir runs/run_20260101_120000 --repo my-org/dantinox-dante

# Or from Python
from dantinox import push
push("runs/run_20260101_120000", "my-org/dantinox-dante", private=False)
```

### Low-level resolver

`resolve_checkpoint` is exposed for custom pipelines — it returns a local path for either a local directory or a Hub repo ID:

```python
from dantinox import resolve_checkpoint

local_path = resolve_checkpoint("my-org/dantinox-dante")          # downloads if needed
local_path = resolve_checkpoint("runs/run_20260101_120000")        # returns unchanged
local_path = resolve_checkpoint("my-org/model", token="hf_…", revision="v1.0")
```

---

## Generation Pipeline: Deep Dive

### KV Cache Architecture

The cache is a **fixed pre-allocated buffer** of shape `(B, kv_heads, 1, max_context, head_size)` per layer. New K/V values are written with `jax.lax.dynamic_update_slice` at the `cache_index` position — no dynamic allocation, no shape changes across decode steps.

| Phase | Input shape | Cache operation | Complexity |
| :--- | :--- | :--- | :--- |
| **Prefill** | `[B, T]` | Allocate + fill positions `0…T-1` | $O(T^2)$ |
| **Decode** | `[B, 1]` | Insert at `cache_index = pos` | $O(S)$ |

### Static Padding (`generate`)

XLA requires fixed shapes at trace time. The `generate()` wrapper pre-allocates a zero buffer of `max_context` tokens and inserts the prompt:

```python
x_padded = jnp.zeros((B, model.max_context), dtype=jnp.int32)
x_padded = x_padded.at[:, :T].set(x)
```

The JIT-compiled core receives a fixed-shape array on every call — no recompilation.

### Token Loop (`_generate_toks`)

Decorated with `@nnx.jit`. Sampling flags (`use_cache`, `top_k`, `top_p`, `temperature`) are `static_argnames` so JAX compiles a specialised kernel per unique combination, eliminating dead branches from the computation graph.

=== "Without KV-Cache"

    Full sequence re-processed at every step (useful for debugging):

    ```python
    x, _, _, _ = jax.lax.fori_loop(
        lower=start_pos, upper=start_pos + max_generations,
        body_fun=prefill_or_no_cache,
        init_val=(x, init_kv_cache, dummy_tok, key),
    )
    ```

=== "With KV-Cache (default)"

    Prefill populates the cache once; the decode loop feeds one token per step:

    ```python
    x, kv_cache, tok, key = prefill_or_no_cache(start_pos, (x, init_kv_cache, dummy_tok, key))
    x, _, _, _ = jax.lax.fori_loop(
        lower=start_pos + 1, upper=start_pos + max_generations,
        body_fun=generate_with_kv_cache,
        init_val=(x, tok, kv_cache, key),
    )
    ```

### Sampling Strategies

All sampling is vectorised across the batch dimension with `jax.vmap`. Because JAX requires static shapes, rejected probabilities are zeroed and renormalised in-place rather than truncated.

#### Top-K

```python
top_k_probs, top_k_idx = jax.lax.top_k(probs, k=top_k)
top_k_probs /= top_k_probs.sum(axis=-1, keepdims=True)
tok = top_k_idx[jax.random.categorical(key, jnp.log(top_k_probs))]
```

#### Top-P (Nucleus)

```python
sorted_idx = jnp.argsort(probs)[::-1]
cumsum = jnp.cumsum(probs[sorted_idx])
mask = (cumsum - probs[sorted_idx]) < top_p
filtered = jnp.where(mask, probs[sorted_idx], 0.0)
tok = sorted_idx[jax.random.categorical(key, jnp.log(filtered / filtered.sum()))]
```

!!! note "First-call latency"
    The first call to any generation method triggers XLA compilation. Subsequent calls with the same shapes reuse the compiled kernel. The `dantinox generate` CLI runs a single warmup pass (`max_new_tokens=1`) before timing, so the reported tok/s reflects steady-state throughput.
