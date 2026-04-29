
# Inference & Generation

## Generation Engine

DantinoX implements a two-phase autoregressive pipeline compiled with `@nnx.jit` and driven by `jax.lax.fori_loop`:

| Phase | Input | Complexity | Cache operation |
| :--- | :--- | :--- | :--- |
| **Prefill** | Full prompt (length $T$) | $O(T^2)$ | Allocate + fill |
| **Decode** | Single token | $O(S)$ where $S$ = cached length | Insert at `cache_index` |

Supported decoding strategies: **greedy**, **temperature**, **Top-K**, **Top-P (nucleus)**. All sampling is vectorised over the batch dimension via `jax.vmap`.

```python
output_ids = generate(
    model, input_ids,
    max_generations=150, temperature=1.3, top_p=0.9, use_cache=True
)
```

!!! warning "MLA models require `inference=True`"
    Before calling `generate()` with an MLA checkpoint, set `config.inference = True`.
    This activates weight absorption — the decode path operates directly on the cached latent
    $\mathbf{c}_{KV}$ without materialising full $K$ and $V$ tensors, reducing both memory
    bandwidth and effective cache size.

    ```python
    config = Config.from_yaml("runs/<run>/config.yaml")
    config.inference = True    # enable weight absorption for decode
    model = load_model(config, "runs/<run>/model_weights.msgpack")
    ```

---

## Running Inference

After training, you can generate text using the `generate.py` script. The script automatically loads the model configuration and weights from a specified run directory.

### Basic Usage

Generate text using the default parameters stored in the run directory:

```bash
python generate.py --run_dir runs/run_YYYYMMDD_HHMMSS --prompt "Nel mezzo del cammin "
```

### Advanced Examples

**Deterministic (Greedy) Decoding:**
Ideal for checking the model's most likely output.
```bash
python generate.py --run_dir runs/run_xxx --greedy true --max_new_tokens 100
```

**Creative Sampling (Top-P & Temperature):**
Use Nucleus sampling with a higher temperature for more varied and poetic results.
```bash
python generate.py --run_dir runs/run_xxx \
  --temperature 1.5 \
  --top_p 0.9 \
  --max_new_tokens 200
```

**Constrained Sampling (Top-K):**
Limit the vocabulary to the top 50 candidates per step.
```bash
python generate.py --run_dir runs/run_xxx --top_k 50 --temperature 1.2
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run_dir` | `str` | **Required** | Path to the folder containing `config.yaml` and `model_weights.msgpack`. |
| `--prompt` | `str` | `"Nel mezzo..."`| Initial text to start generation. |
| `--max_new_tokens` | `int` | `150` | Maximum number of new tokens to generate. |
| `--greedy` | `bool` | `false` | If `true`, ignores sampling and picks the most likely token. |
| `--temperature` | `float` | `1.0` | Controls randomness (higher is more creative). |
| `--top_p` | `float` | `null` | Nucleus sampling threshold. |
| `--top_k` | `int` | `null` | Limits sampling to the top K most likely tokens. |
| `--use_cache` | `bool` | `true` | Toggles the static KV-cache for faster generation. |

!!! note "First-token latency"
    The first call to `generate()` triggers XLA compilation. Subsequent calls reuse the compiled kernel. The script reports **tokens per second (tok/s)** throughput at the end of each run.

---


## Deep Dive: The Generation Pipeline

The pipeline separates dynamic Python logic (shape handling, strategy dispatch) from the static JIT-compiled core (`_generate_toks`).

### 1. Static Padding

XLA requires all array shapes to be fixed at trace time. The public `generate()` wrapper pre-allocates a zero buffer of size `(B, max_context)` and inserts the prompt at position 0:

```python
x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)
x_padded = x_padded.at[:, :T].set(x)
```

### 2. JIT-Compiled Core (`_generate_toks`)

Decorated with `@nnx.jit`. Boolean flags (`use_cache`, `top_k`, `top_p`) are declared as `static_argnames` so JAX compiles a specialised kernel for each unique combination, eliminating dead branches from the computation graph.

#### Control Flow (`jax.lax.fori_loop`)

Python `for` loops introduce dynamic control flow that breaks XLA tracing. `fori_loop` carries a fixed-schema state tuple across iterations:

=== "Without KV-Cache"

    The full sequence is re-processed at every step (useful for debugging):

    ```python
    x, _, _, _ = jax.lax.fori_loop(
        lower=start_pos, upper=start_pos + max_generations,
        body_fun=prefill_or_no_cache,
        init_val=(x, init_kv_cache, dummy_tok, key)
    )
    ```

=== "With KV-Cache (default)"

    Prefill populates the cache once; the decode loop feeds a single token per step:

    ```python
    x, kv_cache, tok, key = prefill_or_no_cache(start_pos, (x, init_kv_cache, dummy_tok, key))
    x, _, _, _ = jax.lax.fori_loop(
        lower=start_pos + 1, upper=start_pos + max_generations,
        body_fun=generate_with_kv_cache,
        init_val=(x, tok, kv_cache, key)
    )
    ```
```

#### 2.2 The Body Functions
These functions dictate what happens inside the loop. Note the difference in how they extract logits:

* **`prefill_or_no_cache`:** Extracts the logits for the token at position `i-1` from the full sequence.
    ```python
    logits, new_kv_cache, _ = model(x, use_cache, kv_cache, 0, deterministic=True)
    x, k, tok = _get_tok_id(i, x, k, logits[:, i-1, :])
    ```
* **`generate_with_kv_cache`:** Extracts only the last token's logits `[:, -1, :]` because the input `tok` is just a single token, avoiding redundant computation.
    ```python
    last_logits, new_kv_cache, _ = model(tok, use_cache, kv_cache, i-1, deterministic=True)
    x, k, next_tok_id = _get_tok_id(i, x, k, last_logits[:, -1, :])
    ```

#### 2.3 Logit Routing & Temperature (`_get_tok_id`)
Before sampling, the raw logits are scaled by the `temperature`. Higher temperatures flatten the distribution (more random), while lower temperatures sharpen it (more deterministic).
```python
last_logits = last_logits / temperature
probs = jax.nn.softmax(last_logits, axis=-1)
```
After computing `probs`, this function acts as a router, directing the probabilities to standard decoding, Top-K, or Top-P functions. Finally, it uses `x.at[:, i].set(tok[:, 0])` to inject the new token into the static padded array.

### 3. Advanced Sampling Algorithms
Because array sizes must remain static in JAX, we cannot simply truncate probability arrays. Instead, we must zero out the rejected probabilities and renormalize. All sampling is vectorized across the batch size using `jax.vmap`.

#### 3.1 Top-K Sampling
`jax.lax.top_k` extracts the top $K$ probabilities and their original indices. We renormalize these top probabilities so they sum to 1.
```python
def __apply_top_k(probs, decoding_func, key, top_k):
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k, axis=-1)
    top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)  
    
    # Split PRNG keys for the batch
    new_key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, probs.shape[0])
    
    def sample_from_top_k(p, k, i):
        sample = decode(probs=p, decoding_func=decoding_func, key=k)
        return i[sample] # Map the local Top-K index back to the global vocab index
        
    toks = jax.vmap(sample_from_top_k)(top_k_probs, batch_keys, top_k_indices)
    return toks, new_key
```

#### 3.2 Top-P (Nucleus) Sampling
Nucleus sampling is mathematically more complex in a static graph. It requires sorting the probabilities, computing a cumulative sum, and masking.
```python
def __apply_top_p(probs, decoding_func, key, top_p):
    # Sort probabilities in descending order
    sorted_indices = jnp.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
    
    new_key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, probs.shape[0])
    
    def sample_from_top_p(p_sorted, k, idx_sorted, top_p_val):
        cumulative_probs = jnp.cumsum(p_sorted, axis=-1) 
        
        # Create a boolean mask where the cumulative sum hasn't exceeded Top-P
        mask = (cumulative_probs - p_sorted) < top_p_val   
        
        # Zero out rejected probabilities and renormalize
        masked_probs = jnp.where(mask, p_sorted, 0.0)
        masked_probs = masked_probs / jnp.sum(masked_probs)
        
        sample_idx = decode(probs=masked_probs, decoding_func=decoding_func, key=k)
        return idx_sorted[sample_idx] # Map back to global vocab index
        
    toks = jax.vmap(sample_from_top_p, in_axes=(0, 0, 0, None))(
        sorted_probs, batch_keys, sorted_indices, top_p
    )
    return toks, new_key
```

### 4. Helper Output Formatting (`decode`)
A tiny but critical utility function. Whether `jax.random.categorical` or `jnp.argmax` is used, the output dimension might differ depending on the batching. This helper forces the sampled token into a strict 2D shape `(batch, 1)` to prevent shape mismatch errors when updating the `x` array.
```python
def decode(probs: jnp.ndarray, decoding_func: DecodeFunc, key: jax.Array | None) -> jnp.ndarray:
    tok = decoding_func(probs, key)
    if tok.ndim == 1:
        tok = tok[:, None]
    return tok
```