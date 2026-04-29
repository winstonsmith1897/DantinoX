
# Inference & Generation

## Generation Engine

DantinoX implements an autoregressive generation pipeline utilizing `jax.lax.fori_loop` and `nnx.jit` to maintain execution efficiency across extended sequences.

### Technical Implementation
* **Static KV Caching:** Prevents redundant computation of previously processed tokens. The model transitions from a quadratic complexity prefill stage to a linear complexity decoding phase.
* **Decoding Strategies:**
    * **Greedy Search:** Deterministic selection of the highest-probability token.
    * **Top-K & Top-P (Nucleus) Sampling:** Stochastic sampling methods to control distribution entropy and semantic coherence.
    * **Temperature Scaling:** Adjusts the logit distribution before the softmax layer to modulate output variance.


### Usage Logic

The generation process is encapsulated in the `_generate_toks` function, which handles state management for both the initial context processing and the subsequent token generation loop:

```python
# Functional interface for sequence generation
output_ids = generate(
    model, 
    input_ids, 
    max_generations=150, 
    temperature=1.3, 
    top_p=0.9, 
    use_cache=True
)
```

> **Note on Vectorization:** The sampling logic uses `jax.vmap` to perform batch-level operations for Top-K and Top-P filtering, ensuring that probability masking and token selection do not introduce bottlenecks during the inference cycle.

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

> **Note on Performance:** The first token generation might experience a slight delay due to JIT compilation. Subsequent tokens and runs will benefit from the optimized XLA kernel, and the script will report the **tokens per second (tok/s)** throughput at the end of the run.

---


## Deep Dive: The Generation Pipeline

The pipeline is split into an uncompiled public wrapper (`generate`) that handles dynamic shapes, and a highly optimized, strictly static JIT-compiled core (`_generate_toks`).

### 1. The Public API & Static Padding (`generate`)
The entry point of the module calculates how many tokens to generate and prepares the tensors for XLA compilation.

```python
B, T = x.shape
to_generate = min(model.max_context, T + max_generations) - T

if to_generate <= 0:
    return x
```
**Static Padding:** XLA cannot compile functions where the input sizes change at runtime. To solve this, DantinoX creates a static buffer `x_padded` of size `(B, max_context)` filled with zeros, and drops the input prompt `x` into the beginning of it.
```python
x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)
x_padded = x_padded.at[:, :T].set(x)
```

**Decoding Strategy Setup:** Depending on the `greedy` flag, the engine defines a lambda function. Greedy decoding uses `jnp.argmax`, while stochastic decoding uses `jax.random.categorical`.
```python
if greedy:
    key = None
    decoding_func = lambda v, key=None: jnp.argmax(v, axis=-1, keepdims=True)
else:
    key = jax.random.key(seed)
    decoding_func = lambda v, key: jax.random.categorical(key, jnp.log(v + 1e-10), axis=-1)
```

### 2. The JIT-Compiled Engine (`_generate_toks`)
This function is decorated with `@nnx.jit`. Crucially, control-flow flags like `use_cache`, `top_k`, and `top_p` are passed as `static_argnames`. This tells JAX to compile a specific, optimized C++ graph for the exact sampling strategy requested, stripping away all unused `if/else` branches.

#### 2.1 The Control Flow (`jax.lax.fori_loop`)
JAX forbids native Python loops for dynamic state updates. Instead, we use `fori_loop`, which carries a "state tuple" (`init_val`) across iterations.

**Scenario A: Without KV-Cache**
If caching is disabled, the model recalculates the entire sequence at every step.
```python
if use_cache is False:
    x, _, _, _   = jax.lax.fori_loop(lower=start_pos, 
                                     upper=start_pos + max_generations, 
                                     body_fun=prefill_or_no_cache, 
                                     init_val=(x, init_kv_cache, dummy_tok, key))
```

**Scenario B: With KV-Cache (Standard)**
This is split into two phases. First, a manual call to `prefill_or_no_cache` processes the initial prompt and populates the KV-cache. Then, the `fori_loop` takes over using `generate_with_kv_cache`, feeding only the last generated token back into the model.
```python
else:
    x, kv_cache, tok, key = prefill_or_no_cache(start_pos, (x, init_kv_cache, dummy_tok, key))
    x, _, _ , _ = jax.lax.fori_loop(lower=start_pos + 1, 
                                    upper=start_pos + max_generations, 
                                    body_fun=generate_with_kv_cache, 
                                    init_val=(x, tok, kv_cache, key))
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