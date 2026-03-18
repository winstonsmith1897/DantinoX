
---

## Inference & Generation Engine

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

## 🔬 Deep Dive: JAX Generation Pipeline

Below is a detailed breakdown of how the `generation.py` module achieves high-throughput, XLA-compliant text generation.

### 1. Static Padding & JIT Compilation
To satisfy XLA's static shape requirements, the initial input sequence is padded with zeros up to `max_context` before entering the compiled generation loop. 

```python
# From core/generation.py - generate()
B, T = x.shape
to_generate = min(model.max_context, T + max_generations) - T

# Pad the input tensor to the maximum sequence length statically
x_padded = jnp.zeros((B, model.max_context), dtype=x.dtype)
x_padded = x_padded.at[:, :T].set(x)
```
The core token-by-token generation logic is then wrapped in `@nnx.jit`. By declaring parameters like `use_cache`, `top_k`, and `temperature` as `static_argnames`, JAX compiles an optimized graph specifically for the chosen sampling strategy, avoiding runtime branching penalties.

### 2. The Prefill and Decode Phases
When `use_cache=True`, the engine intelligently splits the generation into two phases: **Prefill** (processing the given prompt all at once) and **Decode** (generating one token at a time using the static KV-cache).

```python
# From core/generation.py - _generate_toks()

# PHASE 1: Prefill
# Process the initial prompt to compute the first token and populate the KV-cache
x, kv_cache, tok, key = prefill_or_no_cache(start_pos, (x, init_kv_cache, dummy_tok, key))

# PHASE 2: Decode
# Use jax.lax.fori_loop for highly optimized, compiled execution on device
x, _, _ , _ = jax.lax.fori_loop(lower=start_pos + 1, 
                                upper=start_pos + max_generations, 
                                body_fun=generate_with_kv_cache, 
                                init_val=(x, tok, kv_cache, key))
```
Inside `generate_with_kv_cache`, only the newly generated token (`tok`) is passed to the model, rather than the whole sequence, reducing compute complexity per step from $O(T^2)$ to $O(1)$ relative to sequence length.

### 3. Vectorized Stochastic Sampling (`vmap`)
Sampling from a batch of probability distributions requires careful handling of JAX's pseudo-random number generator (PRNG). The PRNG state must be explicitly split and passed along.

DantinoX uses `jax.vmap` to vectorize the sampling operation across the batch dimension without writing explicit loops.

```python
# Temperature scaling
last_logits = last_logits / temperature
probs = jax.nn.softmax(last_logits, axis=-1)

# Splitting PRNG keys for batched generation
new_key, subkey = jax.random.split(k)
batch_keys = jax.random.split(subkey, probs.shape[0])

# Vectorized decoding across the batch
def sample_base(p, ky):
    return decode(probs=p, decoding_func=decoding_func, key=ky)
    
tok = jax.vmap(sample_base)(probs, batch_keys)
```
*(Where `decoding_func` is `jax.random.categorical` for stochastic sampling or `jnp.argmax` for greedy decoding).*

### 4. Advanced Top-K and Top-P (Nucleus) Sampling
For advanced decoding, the engine manipulates the probability distributions on-the-fly.

* **Top-K:** `jax.lax.top_k` extracts the top $K$ probabilities, renormalizes them, and samples.
* **Top-P (Nucleus):** Sorts the probabilities descendingly, computes the cumulative sum, and masks out any probabilities that push the sum beyond the $P$ threshold.

```python
# Top-P Nucleus Masking Logic
cumulative_probs = jnp.cumsum(p_sorted, axis=-1) 
mask = (cumulative_probs - p_sorted) < top_p_val   

masked_probs = jnp.where(mask, p_sorted, 0.0)
masked_probs = masked_probs / jnp.sum(masked_probs) # Renormalize

sample_idx = decode(probs=masked_probs, decoding_func=decoding_func, key=k)
```
Because array sizes must remain static in JAX, the unselected probabilities are zeroed out (`0.0`) rather than truncated. This ensures the shape of the array remains constant, avoiding XLA recompilation errors while mathematically preventing those tokens from being sampled.