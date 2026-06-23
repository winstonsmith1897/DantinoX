# `dantinox.trainer` — Training API Reference

This page is the complete reference for the `Trainer` class in
`dantinox/trainer.py`. Every parameter, every internal step, and every
helper function is explained here in detail.

---

## Overview

`Trainer` is a single class that handles the full training lifecycle for all
three DantinoX model families: autoregressive (`Transformer`), masked-diffusion
(`DiffusionTransformer`), and continuous flow-matching (`ELFTransformer`).

```
Trainer.fit(data_path)
    │
    ├─ 1. Dataset loading          → tokenised .npy cache
    ├─ 2. max_train_tokens cap      → fixed compute budget
    ├─ 3. 90 / 10 train/val split
    ├─ 4. _build_optimizer          → schedule + gradient clipping
    ├─ 5. _create_model             → Transformer | DiffusionTransformer | ELFTransformer
    ├─ 6. _cast_params(bfloat16)    → before JIT compilation
    ├─ 7. nnx.Optimizer(wrt=...)    → LoRA vs full-param training
    ├─ 8. make_mesh / shard_batch   → multi-GPU data parallelism
    ├─ 9. resume logic              → training_cursor.json + model_weights.msgpack
    ├─ 10. JIT train_step           → AR | Diffusion | ELF dispatch
    ├─ 11. gradient accumulation    → acc pattern, division by grad_accum
    ├─ 12. estimate_loss            → stratified t for diffusion
    ├─ 13. checkpointing            → best_model_weights.msgpack
    └─ 14. early stopping           → patience counter
```

---

## `Trainer.__init__(config)`

```python
trainer = Trainer(config)
```

`__init__` stores the `Config` instance and does nothing else — no model
allocation, no JIT compilation, no data loading. All of that happens inside
`fit()`.

### `config` — what it is and what it stores

`config` is a `core.config.Config` dataclass (the monolithic flat config kept
for backward compatibility). It holds every knob for both the model architecture
and the training procedure in one flat namespace. The relevant fields are:

**Architecture fields** (passed to `_create_model`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dim` | `int` | `512` | Hidden dimension of the transformer |
| `n_heads` | `int` | `16` | Number of query attention heads |
| `head_size` | `int` | `32` | Dimension per head; must satisfy `dim == n_heads * head_size` |
| `num_blocks` | `int` | `20` | Number of transformer blocks |
| `vocab_size` | `int` | `200` | Vocabulary size (updated after tokenizer training) |
| `max_context` | `int` | `512` | Maximum sequence length |
| `kv_heads` | `int` | `4` | KV heads for GQA; set equal to `n_heads` for MHA |
| `model_type` | `str` | `"autoregressive"` | `"autoregressive"`, `"diffusion"`, or `"elf"` |
| `attention_type` | `str` | `"auto"` | `"mha"`, `"gqa"`, `"mla"`, or `"auto"` (derived) |
| `norm_type` | `str` | `"layernorm"` | `"layernorm"` or `"rmsnorm"` |
| `use_swiglu` | `bool` | `True` | Use SwiGLU activation in FFN |
| `gradient_checkpointing` | `bool` | `True` | Wrap blocks with `nnx.remat` |
| `use_lora` | `bool` | `False` | Enable LoRA adapter training |
| `lora_rank` | `int` | `8` | LoRA rank `r` |
| `lora_alpha` | `float` | `16.0` | LoRA scaling factor `α` |

**Training fields** (used inside `fit`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `lr` | `float` | `0.005` | Peak learning rate |
| `batch_size` | `int` | `128` | Total batch size per optimizer step |
| `grad_accum` | `int` | `16` | Gradient accumulation steps |
| `epochs` | `int` | `1000` | Training epochs |
| `warmup_steps` | `int` | `420` | Linear warmup length; capped at 30 % of `total_steps` |
| `lr_schedule` | `str` | `"cosine"` | `"cosine"`, `"linear"`, `"constant"`, or `"wsd"` |
| `optimizer` | `str` | `"adamw"` | Optimizer name (see table below) |
| `grad_clip` | `float` | `1.0` | Global gradient norm clipping; `0` disables |
| `patience` | `int` | `0` | Early-stopping evaluations without improvement; `0` disables |
| `eval_iters` | `int` | `20` | Evaluation batches per `estimate_loss` call |
| `use_bf16` | `bool` | `False` | Cast parameters to bfloat16 before JIT |
| `seed` | `int` | `42` | PRNG seed for model init, data batching, and diffusion noise |
| `n_devices` | `int` | `0` | GPUs to use; `0` = all available, `1` = single-device |
| `checkpoint_every` | `int` | `2000` | Steps between resume checkpoints |
| `max_train_tokens` | `int` | `10_000_000` | Token cap on the dataset |

---

## `Trainer.fit(...)`

```python
run_dir = trainer.fit(
    data_path,           # (1)!
    *,
    run_dir=None,        # (2)!
    wandb_project=None,  # (3)!
    resume=False,        # (4)!
)
```

1. Path to a UTF-8 text file, or `None` to fall back to `config.dataset_name`
   (which can be a local path or a HuggingFace dataset id when
   `config.dataset_source == "huggingface"`).
2. Directory where checkpoints and logs are written. When omitted,
   `fit` creates `runs/run_YYYYMMDD_HHMMSS/` automatically.
3. If not `None`, metrics are streamed to Weights & Biases under this project
   name. Requires `wandb` to be installed and authenticated.
4. If `True` and both `training_cursor.json` and `model_weights.msgpack` exist
   inside `run_dir`, training resumes from the saved step. Optimizer state is
   *not* preserved — the optimizer is re-initialised from scratch.

Returns the `run_dir` string so callers can locate the checkpoint:

```python
run_dir = trainer.fit("data/wiki.txt")
print("Best checkpoint:", run_dir + "/best_model_weights.msgpack")
```

---

### Step 1 — Dataset loading and the tokenised-array cache

The first time a `(dataset_name, dataset_config, tokenizer_type)` combination
is used, downloading and tokenising a large corpus can take tens of seconds or
more. To avoid repeating this work on every run, `fit` builds a two-file cache
under the `data/` directory that lives one level above `run_dir`:

```
data/
  <dataset_slug>_<config>_<tok_type>.npy   ← token ID array, dtype int32
  <dataset_slug>_<config>_<tok_type>.json  ← serialised tokenizer
```

**Cache key.** The key is formed by concatenating:

```python
f"{config.dataset_name.replace('/', '_')}_{config.dataset_config or 'default'}_{config.tokenizer_type}"
```

For example, `wikitext_wikitext-103-raw-v1_bpe` or `corpus_default_char`.

**Fast path** (cache hit): `fit` calls `load_tokenizer_from_file` on the cached
JSON and loads the `.npy` with `numpy.load`. This typically takes under two
seconds regardless of corpus size.

**Slow path** (cache miss):

1. If `config.dataset_source == "huggingface"`, HuggingFace `datasets` is
   imported and `load_dataset` is called with `config.dataset_name`,
   `config.dataset_config`, `config.dataset_split`, and
   `config.dataset_text_field`. When `config.streaming=True` the dataset is
   iterated as an `IterableDataset` and materialised to a single string;
   otherwise the full split is loaded.
2. If `dataset_source == "local"`, the file at `data_path` (or
   `config.dataset_name`) is opened directly.
3. The raw text is normalised with `_format_text` (groups lines into 3-line
   paragraphs, removes blank lines).
4. A tokenizer is created with `get_tokenizer(config.tokenizer_type)`:
   - `char`: character-level; vocabulary is trained from the text.
   - `bpe`: byte-pair encoding; vocabulary is trained to `config.vocab_size`.
   - `t5`: pre-trained SentencePiece tokenizer from the T5 family; no training
     needed.
5. Token IDs are written to the `.npy` cache and the tokenizer is saved to the
   `.json` cache.

After loading, `config.vocab_size` is updated in place from `tokenizer.vocab_size`.

!!! tip "Why the cache exists"
    Re-tokenising WikiText-103 (~100M chars) takes ~60 s with a BPE tokenizer.
    With the cache it takes ~2 s. The cache persists across runs and across
    different `run_dir` values as long as the dataset and tokenizer type are
    the same.

---

### Step 2 — `max_train_tokens` cap

```python
_max_tok = getattr(config, "max_train_tokens", 10_000_000)
if _max_tok > 0 and len(full_data) > _max_tok:
    full_data = full_data[:_max_tok]
```

This is a simple prefix slice. The default cap is 10 million tokens, which
corresponds to roughly 822 optimizer steps at `batch_size=128`,
`max_context=512`, `grad_accum=16` on WikiText-103. On two A100s this takes
about seven minutes — a sensible budget for an exploratory run.

Set `max_train_tokens=0` or a very large number to disable the cap and use
the entire corpus.

---

### Step 3 — Train / validation split

```python
n = int(0.9 * len(full_data))
train_data, val_data = full_data[:n], full_data[n:]
```

A deterministic 90/10 split on the token array. No shuffling is performed at
this stage — shuffling happens implicitly through the random batch sampling in
`get_batch` (which draws random starting positions at each step).

The validation set is the *last* 10 % of the token sequence. For most natural
language corpora this is a representative held-out slice.

---

### Step 4 — Optimizer and schedule construction

`fit` calls `_build_optimizer(config, total_steps)` which in turn calls
`_build_schedule(config, total_steps)`.

#### `_build_schedule`

```python
def _build_schedule(config, total_steps):
    warmup_steps = min(
        getattr(config, "warmup_steps", int(total_steps * 0.1)),
        int(total_steps * 0.3),   # hard cap at 30 %
    )
    peak = config.lr
    end  = peak * 0.01            # final LR = 1 % of peak
    ...
```

The warmup length is `config.warmup_steps`, but it is clamped to a maximum of
30 % of `total_steps` so that very long warmups do not dominate short runs.
All schedules share the same `end` value: 1 % of the peak learning rate.

=== "cosine"

    Standard warmup followed by cosine decay, implemented by
    `optax.warmup_cosine_decay_schedule`.

    ```
    lr
    peak ─────────────╮
                       ╲___cosine decay___
    end  ─────────────────────────────────── step
          warmup       decay_steps
    ```

=== "linear"

    Linear warmup from 0 → peak, then linear decay from peak → end.

    ```python
    warmup = optax.linear_schedule(0.0, peak, warmup_steps)
    decay  = optax.linear_schedule(peak, end, total_steps - warmup_steps)
    return optax.join_schedules([warmup, decay], [warmup_steps])
    ```

=== "constant"

    Linear warmup from 0 → peak, then a flat plateau at `peak` for the
    remainder of training. Useful for short experiments where you want to
    stay at maximum learning rate.

=== "wsd"

    Warmup → Stable → Decay, a three-phase schedule described in the WSD paper:

    - Warmup: linear from 0 → peak over `warmup_steps`.
    - Stable: constant at peak for 40 % of `total_steps`.
    - Decay: cosine from peak → end over the remaining steps.

    ```
    lr
    peak ──────────────╮
          warmup  stable╰──cosine decay──
    end  ───────────────────────────────── step
    ```

    WSD is particularly effective for large models because the stable plateau
    lets the optimizer converge at peak LR before the final cool-down.

#### `_build_optimizer`

The schedule is wrapped in one of five optimizer variants:

| `config.optimizer` | Optax call | Notes |
|---|---|---|
| `"adamw"` (default) | `optax.adamw(lr=schedule)` | AdamW with default weight decay 0.1; standard choice |
| `"adam"` | `optax.adam(lr=schedule)` | Adam without weight decay |
| `"adafactor"` | `optax.adafactor(lr=schedule)` | Memory-efficient; no second-moment per-parameter; good for very large models on memory-constrained devices |
| `"lion"` | `optax.lion(lr=schedule)` | Sign-gradient update; requires a lower LR than AdamW (typically 3–10× lower) |
| `"muon"` | `optax.contrib.muon(lr=schedule)` | Applies Newton-Schulz orthogonalization to 2-D parameter gradients; falls back to Adam for biases and norms; available since optax 0.2.6 |

**Gradient clipping.** When `config.grad_clip > 0` (default `1.0`), the
optimizer is wrapped with `optax.clip_by_global_norm`:

```python
if grad_clip > 0:
    return optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
```

The chain applies clipping *before* the optimizer update, which is the
standard convention. When `grad_clip=0.0`, no clipping is applied.

---

### Step 5 — Model construction

```python
model = _create_model(config, rngs)
```

`_create_model` routes on `config.model_type`:

| `config.model_type` | Class instantiated |
|---|---|
| `"autoregressive"` (default) | `core.model.Transformer(config, rngs=rngs)` |
| `"diffusion"` | `core.model.DiffusionTransformer(config, rngs=rngs)` |
| `"elf"` | `core.elf.ELFTransformer(config.to_elf_config(), rngs=rngs)` |

For ELF, `config.to_elf_config()` first converts the flat `Config` into an
`ELFConfig` dataclass that contains only ELF-relevant fields.

---

### Step 6 — bfloat16 casting (`_cast_params`)

```python
if getattr(config, "use_bf16", False):
    _cast_params(model, jnp.bfloat16)
```

`_cast_params` extracts all `nnx.Param` variables from the model, casts every
floating-point array to `bfloat16`, and writes them back with `nnx.update`:

```python
def _cast_params(model, dtype):
    params = nnx.state(model, nnx.Param)
    nnx.update(
        model,
        jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        ),
    )
```

**Why before JIT?** JAX traces the computation graph at the first call to a
`@nnx.jit`-decorated function and specialises on the concrete dtypes of the
inputs. If the cast happened inside the JIT boundary, XLA would see mixed
float32/bfloat16 and insert extra conversion ops. Casting before compilation
ensures the entire compiled graph operates uniformly in bfloat16, maximising
throughput on hardware with fast bfloat16 units (A100, H100, TPU).

---

### Step 7 — LoRA vs full-parameter training

```python
wrt_type = LoRAParam if getattr(config, "use_lora", False) else nnx.Param
optimizer = nnx.Optimizer(model, tx, wrt=wrt_type)
```

`nnx.Optimizer(model, tx, wrt=T)` only tracks and updates variables of type
`T`. This is the key mechanism that makes LoRA work at the type system level:

- **Full-param training** (`wrt=nnx.Param`): every `nnx.Param` in the model
  receives gradient updates. This includes all weights in all layers.
- **LoRA training** (`wrt=LoRAParam`): only `LoRAParam` variables — the adapter
  matrices A and B inside every `LoRALinear` — receive updates. The base weights
  (stored as `nnx.Param` inside `LoRALinear.base`) are frozen; they still
  participate in the forward pass but XLA never computes or accumulates
  gradients for them during the backward pass.

`LoRAParam` is defined in `core/lora.py` as a trivial subclass of `nnx.Variable`:

```python
class LoRAParam(nnx.Variable):
    """Trainable LoRA variable — distinct type so base nnx.Param weights stay frozen."""
    pass
```

The type distinction is the entire freezing mechanism. No masking, no
`stop_gradient`, no manual gradient zeroing.

!!! note "LoRA targets"
    `config.lora_targets` controls which layers use `LoRALinear` instead of
    `nnx.Linear`:
    - `"attention"` (default): only the QKV and output projections.
    - `"mlp"`: only the FFN up/down projections.
    - `"all"`: both attention and FFN layers.

---

### Step 8 — Multi-GPU data parallelism

```python
n_dev_cfg = getattr(config, "n_devices", 0)
n_local = len(jax.local_devices())
use_multi_gpu = (n_dev_cfg != 1) and n_local > 1
mesh = make_mesh(n_dev_cfg) if use_multi_gpu else None
```

When `n_devices != 1` and more than one local device is visible, DantinoX
creates a data-parallel mesh via `core.sharding.make_mesh`. A `jax.sharding.Mesh`
is a logical grid of devices; here it is a 1-D mesh with a single `"data"` axis.

**Batch divisibility requirement.** The total `batch_size` must be exactly
divisible by the number of devices. `fit` raises `ConfigError` if this is not
satisfied:

```python
if config.batch_size % n_dev != 0:
    raise ConfigError(
        f"batch_size ({config.batch_size}) must be divisible by "
        f"n_devices ({n_dev}) for data-parallel training."
    )
```

This is required because `shard_batch` splits the batch tensor along the first
axis into `n_dev` equal shards, one per device.

After building the mesh, model, optimizer, and metrics state are replicated
across all devices once:

```python
state = replicate(nnx.state((model, optimizer, metrics)), mesh)
nnx.update((model, optimizer, metrics), state)
```

At each training step, `shard_batch(x, mesh)` distributes the batch tensor
so each device sees only its own shard. Because the model is replicated, each
device runs the same computation on a different slice of the batch. Gradients
are all-reduced across devices inside the JIT-compiled step by JAX's SPMD
machinery.

Set `n_devices=0` to automatically use all available GPUs, or `n_devices=1`
to force single-device mode even when multiple GPUs are present.

---

### Step 9 — Resume logic

When `resume=True`, `fit` looks for two files in `run_dir`:

| File | Contents |
|---|---|
| `training_cursor.json` | `{"step": N, "best_val_loss": L}` — last saved step |
| `model_weights.msgpack` | Full parameter state at step N, msgpack-serialised |

```python
if resume and os.path.exists(cursor_path) and os.path.exists(resume_weights):
    with open(cursor_path) as f:
        cursor = json.load(f)
    start_step = int(cursor.get("step", 0)) + 1
    with open(resume_weights, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, ...)
    nnx.update(model, state_dict)
```

Training then starts from `start_step` instead of `0`. Note that the
optimizer state is *not* restored — the optimizer starts fresh. This means
Adam's first and second moment estimates are zero at the start of resumed
training, which causes a brief "warmup" effect in the loss even if the LR
schedule does not have a warmup. This is a known limitation.

The `training_cursor.json` file is *deleted* at the end of a successful run
so a completed run is not confused with an interrupted one.

---

### Step 10 — The three JIT-compiled `train_step` functions

`fit` defines and JIT-compiles a `train_step` function whose implementation
differs across the three model families. The dispatch is a Python `if/elif/else`
block, so only one `train_step` is compiled per `fit` call.

=== "Autoregressive (AR)"

    ```python
    @nnx.jit
    def train_step(model, opt, metrics, full_x, full_y, _unused_key=None):
        xs = full_x.reshape(config.grad_accum, micro_bs, -1)
        ys = full_y.reshape(config.grad_accum, micro_bs, -1)

        def _loss(model, x, y):
            out = model(x)
            loss = compute_loss(out.logits, y)
            if getattr(model, "use_moe", False):
                loss = loss + model.alpha_balance * out.aux_loss
            return loss, out.aux_loss
        ...
    ```

    The input is a shifted token pair `(x, y)` where `y = x[1:]` (next-token
    prediction). `compute_loss` is cross-entropy over the vocabulary.
    For MoE models, the load-balancing auxiliary loss is added to the total
    loss with weight `model.alpha_balance`.

=== "Diffusion (LLaDA)"

    ```python
    @nnx.jit
    def train_step(model, opt, metrics, full_x, _unused_y, key):
        xs = full_x.reshape(config.grad_accum, micro_bs, -1)

        def _loss(model, x, key):
            key, sub_t = jax.random.split(key)
            t_batch = jax.random.uniform(sub_t, (micro_bs,), minval=_t_min, maxval=1.0)
            key, sub_c = jax.random.split(key)
            x_t = corrupt(x, t_batch, sub_c, _noise_schedule, config.mask_token_id)
            out = model(x_t, deterministic=False)
            loss = masked_cross_entropy(
                out.logits, x, x_t, config.mask_token_id,
                t_float=t_batch, aux_loss=out.aux_loss,
                alpha_balance=model.alpha_balance,
            )
            return loss, out.aux_loss
        ...
    ```

    Follows the LLaDA formulation (arXiv:2502.09992). At each step:
    1. A per-sequence `t` is sampled from `U[t_min, 1]`.
    2. `corrupt` replaces each token with the mask token with probability
       `p_mask(t)` (linear schedule: `p_mask = t`).
    3. The time-free model predicts the original token for every masked position.
    4. The loss is `(1/t) × cross_entropy_on_masked_tokens` (ELBO weight, Eq. 3
       of the LLaDA paper).

    `t_min` is set to `max(1/max_context, 0.05)`. The lower bound of 0.05
    prevents extreme gradient variance: at `t = 1/L ≈ 0.002` (one masked token
    per sequence), the `1/t` weight would be ~500, completely dominating the
    gradient. Flooring at 0.05 corresponds to approximately 26 masked tokens per
    512-token sequence, keeping the gradient well-conditioned.

=== "ELF (continuous flow-matching)"

    ```python
    @nnx.jit
    def train_step(model, opt, metrics, full_emb, full_x, key):
        E    = full_emb.shape[-1]
        embs = full_emb.reshape(config.grad_accum, micro_bs, -1, E)
        xs   = full_x.reshape(config.grad_accum, micro_bs, -1)

        def _loss(model, emb_i, x_i, key):
            embeddings = model.encode(emb_i)   # channel-wise normalise
            loss, aux  = elf_loss(model, embeddings, x_i, key, _elf_config)
            return loss, aux["den_loss"]
        ...
    ```

    The ELF step takes a *pre-computed T5 embedding* `full_emb` instead of raw
    tokens. T5 is a large encoder-only model that is kept *outside* JIT — its
    forward pass runs on a separate XLA computation (line 665 of trainer.py:
    `emb = _t5_encoder.encode(x)` before the JIT call). This avoids retracing
    T5's large computation graph every step and allows it to run independently.

    Before training starts, `fit` computes channel-wise mean/std statistics from
    four batches and stores them in `model.embedder.emb_mean` and
    `model.embedder.emb_std`. The `model.encode(emb_i)` call inside the JIT
    step uses these statistics to normalise each channel, ensuring the
    flow-matching loss is on a consistent scale.

---

### Step 11 — Gradient accumulation

All three `train_step` variants share the same accumulation pattern:

```python
grad_fn = nnx.value_and_grad(_loss, argnums=DiffState(0, _wrt), has_aux=True)
acc = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, _wrt))  # (1)!
total_loss = jnp.array(0.0)

for i in range(config.grad_accum):               # (2)!
    (loss, aux), grads = grad_fn(model, ...)
    acc = jax.tree_util.tree_map(
        lambda a, g: a + g / config.grad_accum,  # (3)!
        acc, grads
    )
    total_loss += loss / config.grad_accum        # (4)!

opt.update(model, acc)                            # (5)!
```

1. `acc` is a pytree of zeros with the same structure as the trainable
   parameters. It acts as the gradient accumulation buffer.
2. The loop runs `grad_accum` times, each time on a different micro-batch of
   size `micro_bs = batch_size // grad_accum`.
3. Each micro-batch gradient is divided by `grad_accum` before being added to
   `acc`. After all iterations, `acc` holds the average gradient over the full
   logical batch.
4. The loss is similarly averaged. The reported `total_loss` is the true mean
   loss over the full batch.
5. A single `opt.update` call is made at the end with the accumulated gradient.
   This is equivalent to one forward/backward pass over a batch of size
   `batch_size`, but uses only `micro_bs` activation memory at a time.

Because the loop is inside `@nnx.jit`, XLA unrolls it at compile time. All
`grad_accum` forward/backward graphs coexist in the compiled program
simultaneously. When `gradient_checkpointing=True`, `nnx.remat` wraps each
block's internals and discards intermediate activations, recomputing them on
the backward pass — this keeps peak VRAM manageable even with large
`grad_accum` values.

---

### Step 12 — Evaluation loop (`estimate_loss`)

`estimate_loss` is called every 500 training steps. It returns a dict with
`"train"`, `"val"`, `"train_bal"`, and `"val_bal"` keys.

=== "Autoregressive"

    Plain cross-entropy evaluated on `eval_iters` random batches of size 1
    from both splits:

    ```python
    for split, d in [("train", train_data), ("val", val_data)]:
        for _ in range(config.eval_iters):
            x, y = get_batch(d, 1, config.max_context, sub)
            loss_val, b, _ = eval_step(model, x, y)
    ```

=== "Diffusion — stratified t"

    The diffusion evaluation uses stratified sampling of the noise level `t`:

    ```python
    n = config.eval_iters
    t_low = _t_min    # same floor as training: max(1/max_context, 0.05)
    t_strata = [
        jnp.full((eval_bs,), t_low + (1.0 - t_low) * (i + 0.5) / n)
        for i in range(n)
    ]
    ```

    Each of the `n` evaluation batches is evaluated at a different, evenly
    spaced noise level. This ensures coverage of the entire `[t_min, 1]`
    range. The `t_min` floor is the same as training so that val and train
    losses are on the same scale — using `t` values below `t_min` during
    evaluation would inflate val loss because the model was never trained
    at those noise levels and the `1/t` weight is very large there.

=== "ELF"

    T5 embeddings are computed outside JIT (`emb = _t5_encoder.encode(x)`)
    and then passed to `eval_step`. The ELF loss is averaged over
    `config.eval_iters` batches.

---

### Step 13 — Checkpointing

Two checkpoint files are managed:

| File | When written | Contents |
|---|---|---|
| `best_model_weights.msgpack` | Every time `val_loss < best_val_loss` | Best parameter state seen so far |
| `model_weights.msgpack` | Every `checkpoint_every` steps (default 2000) | Current parameter state for resuming |
| `training_cursor.json` | Together with `model_weights.msgpack` | `{"step": N, "best_val_loss": L}` |

Only the *best* checkpoint is saved in `best_model_weights.msgpack`. There is
no rolling window of recent checkpoints; at any point during training the file
contains the single best set of weights seen. This minimises disk usage while
ensuring the best model is always recoverable.

The resume checkpoint (`model_weights.msgpack` + `training_cursor.json`) is a
separate pair written periodically so that interrupted runs can be restarted
without losing all progress.

**msgpack format.** `_save_weights` calls `flax.serialization.msgpack_serialize`
on a plain Python dict of numpy arrays (the output of `nnx.state(model, nnx.Param).to_pure_dict()`). This is the native Flax NNX serialisation format.

To load a checkpoint manually:

```python
import msgpack
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

with open("runs/my_run/best_model_weights.msgpack", "rb") as f:
    state_dict = msgpack.unpackb(
        f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
    )
nnx.update(model, state_dict)
```

---

### Step 14 — Early stopping

```python
patience = getattr(config, "patience", 0)
best_val_loss = float("inf")
no_improve = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    no_improve = 0
    _save_weights(model, best_weights_path)
else:
    no_improve += 1
    if patience > 0 and no_improve >= patience:
        break
```

`patience` counts the number of consecutive evaluation checkpoints
(every 500 steps) without improvement in `val_loss`. When `no_improve`
reaches `patience`, training stops. Set `patience=0` (the default) to
disable early stopping and always run for the full `total_steps`.

---

### Weights & Biases logging

```python
if wandb_project is not None:
    wandb.log({"train_loss": losses["train"], "val_loss": val_loss, "step": step})
```

Three scalars are logged at every evaluation point: `train_loss`, `val_loss`,
and `step`. The run is initialised with `wandb.init(project=wandb_project,
config=config.to_dict())` so all hyperparameters appear in the W&B run
configuration panel.

---

## `Trainer.find_lr(data_path, min_lr, max_lr, num_steps, smoothing)`

Implements the **LR Range Test** (Smith 2015, arXiv:1506.01186). The idea is to
train for a short period while exponentially increasing the learning rate from
`min_lr` to `max_lr`, observe where the loss starts to diverge, and pick the
LR at the point of steepest loss descent as a good starting point for full
training.

```python
suggested_lr, lr_history, loss_history = trainer.find_lr(
    "data/corpus.txt",
    min_lr=1e-7,   # (1)!
    max_lr=1.0,    # (2)!
    num_steps=100, # (3)!
    smoothing=0.9, # (4)!
)
```

1. Starting learning rate for the sweep (default `1e-7`).
2. Maximum learning rate (default `1.0`). The sweep is exponential so even
   `max_lr=1.0` is reached smoothly.
3. Total number of training steps (default `100`). More steps give a smoother
   curve; fewer steps are faster.
4. Exponential moving average factor for loss smoothing (default `0.9`). Higher
   values produce smoother curves but delay responsiveness.

### Exponential LR sweep

The learning rate at step `s` is:

$$\text{lr}(s) = \text{min\_lr} \times \exp\!\left(s \times \frac{\ln(\text{max\_lr} / \text{min\_lr})}{\text{num\_steps} - 1}\right)$$

This means `lr(0) = min_lr` and `lr(num_steps - 1) = max_lr`.

```python
log_multiplier = math.log(max_lr / min_lr) / max(1, num_steps - 1)

def _lr_fn(step):
    return min_lr * jnp.exp(step * log_multiplier)
```

### Smoothed loss and bias correction

The raw loss at each step is noisy. `find_lr` applies an exponential moving
average (EMA) and corrects the bias introduced by the EMA being initialised at
zero:

```python
smooth_loss = smoothing * smooth_loss + (1 - smoothing) * loss_val
debiased    = smooth_loss / (1 - smoothing ** (step + 1))
```

This is the same formula used in Adam's bias-corrected moment estimates.

### Early stopping criterion

```python
if debiased > 4 * best_loss:
    break
```

If the bias-corrected loss exceeds four times the best loss seen so far, the
learning rate has clearly diverged and the sweep is stopped early. This prevents
wasting time on LR values that are far too large.

### Suggested LR selection

```python
slopes = [loss_history[i+1] - loss_history[i] for i in range(len(loss_history) - 1)]
suggested_lr = lr_history[min(range(len(slopes)), key=lambda i: slopes[i])]
```

The suggested LR is the learning rate at the step with the **steepest negative
slope** in the smoothed loss curve — i.e., where the loss is falling fastest.
This is the standard heuristic from the Smith paper. A common rule of thumb is
to use `suggested_lr / 10` as the peak LR for actual training to maintain a
safety margin.

### Returns

`find_lr` returns a 3-tuple:

| Element | Type | Description |
|---|---|---|
| `suggested_lr` | `float` | LR at the point of steepest descent |
| `lr_history` | `list[float]` | LR value at each step |
| `loss_history` | `list[float]` | Bias-corrected smoothed loss at each step |

Plot `loss_history` against `lr_history` (log scale on the x-axis) to visually
inspect the curve and validate the suggestion.

!!! warning "find_lr uses only Transformer"
    The current implementation always instantiates a `Transformer` (AR model)
    regardless of `config.model_type`. Use `find_lr` to tune the LR for AR runs
    and use the result as a starting point for diffusion/ELF runs with appropriate
    scaling.

---

## Helper functions

### `_build_schedule(config, total_steps)`

Described in [Step 4](#step-4--optimizer-and-schedule-construction) above.
Returns an `optax.Schedule` callable `lr_fn(step) -> float`.

### `_build_optimizer(config, total_steps)`

Described in [Step 4](#step-4--optimizer-and-schedule-construction) above.
Returns an `optax.GradientTransformation`.

### `_model_summary(model, config, optimizer)`

Computes a memory and parameter count summary and writes it to
`model_summary.json` in `run_dir`.

### `_save_weights(model, path)`

```python
def _save_weights(model, path):
    state_dict = nnx.state(model, nnx.Param).to_pure_dict()
    with open(path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(state_dict))
```

Extracts the `nnx.Param` state tree, converts it to a plain Python dict of
numpy arrays, and serialises it with Flax's msgpack serialiser. The file is
a self-contained binary that can be deserialized without access to the Python
model class.

---

## `model_summary.json` — field reference

Written to `<run_dir>/model_summary.json` before training begins. Example:

```json
{
    "total_params_M": 124.44,
    "dtype": "bfloat16",
    "weights_mem_MB": 248.88,
    "optimizer_mem_MB": 497.76,
    "est_activations_MB": 1342.18
}
```

| Field | Formula | Meaning |
|---|---|---|
| `total_params_M` | `Σ param.size / 1e6` | Total number of trainable parameters in millions |
| `dtype` | `"bfloat16"` if `use_bf16` else `"float32"` | Data type of stored weights |
| `weights_mem_MB` | `total_params × bpp / 1e6` | Memory consumed by model weights; `bpp=2` for bfloat16, `4` for float32 |
| `optimizer_mem_MB` | `Σ opt_state.size × bpp / 1e6` | Memory for optimizer state (e.g. Adam's two moment tensors double the parameter count) |
| `est_activations_MB` | `micro_bs × max_context × dim × num_blocks × 8 × bpp × grad_accum / 1e6` | Estimated activation memory |

**Activation memory formula explained.**

```python
act = micro_bs * config.max_context * config.dim * config.num_blocks * 8 * bpp * grad_accum
```

The factor `8` is an empirical constant that accounts for the multiple
intermediate tensors stored during a transformer forward pass per block: the
pre-norm activations, QKV projections, attention weights, context vector, FFN
input/output, and residual connections. The multiplication by `grad_accum`
reflects the fact that the loop inside `@nnx.jit` is fully unrolled by XLA —
all `grad_accum` micro-batch activation graphs coexist simultaneously in the
compiled program. When `gradient_checkpointing=True`, block internals are
recomputed on the backward pass, reducing the effective activation footprint
significantly.

!!! note "Estimates, not exact figures"
    `est_activations_MB` is a rough upper bound. Actual VRAM usage will
    differ because of XLA buffer reuse, operator fusion, and the effect of
    `gradient_checkpointing`. Use profiling tools (e.g. `jax.profiler`) for
    precise measurements.

---

## Files written to `run_dir`

| File | Written when | Contents |
|---|---|---|
| `config.yaml` | Immediately at start | Exact `Config` used (reproducible) |
| `tokenizer.json` | After tokenizer creation or cache copy | Serialised tokenizer for inference |
| `model_summary.json` | Before training loop | Parameter count, memory estimates |
| `training_log.csv` | Every 500 steps | `step, train_loss, val_loss, train_bal, val_bal, ms_per_step` |
| `best_model_weights.msgpack` | On every new best val loss | Best parameter state |
| `model_weights.msgpack` | Every `checkpoint_every` steps | Resume checkpoint |
| `training_cursor.json` | Together with `model_weights.msgpack` | `{"step": N, "best_val_loss": L}` |

`training_cursor.json` is removed at the end of a successful run.

---

## See also

- [Configuration Reference](../configuration.md) — all `Config` fields
- [Architecture: Core Layers](../architecture/core.md) — attention, FFN, LoRA internals
- [CLI Reference](../cli.md) — `dantinox train` and `dantinox find-lr`
- [Cookbook](../cookbook.md) — end-to-end training recipes
