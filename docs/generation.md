
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