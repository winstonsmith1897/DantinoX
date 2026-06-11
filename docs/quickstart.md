# Quickstart

This guide takes you from zero to a working language model in a few minutes.
Every step is explained: installation, training, generation, and all three available paradigms.

---

## 1. Installation

### Prerequisites

| Requirement | Minimum version | Notes |
|:------------|:---------------:|:------|
| Python | 3.10 | Required for type annotations |
| JAX | 0.4.25 | Provides XLA and JIT compilation |
| Flax NNX | 0.8 | Mutable-state API (different from Linen) |
| CUDA | 12.x | NVIDIA GPU only |

### From source (recommended for research)

```bash title="Terminal"
# 1. Clone the repository
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

# 2. (Optional but recommended) Create a dedicated virtual environment
conda create -n dantinox python=3.10 -y
conda activate dantinox

# 3. Install JAX with CUDA 12 GPU support
pip install -U "jax[cuda12]" jaxlib

# 4. Install DantinoX in editable mode with all dependencies
pip install -e ".[all]"
```

!!! note "CPU-only"
    If you have no GPU or want to run on CPU, replace `jax[cuda12]` with `jax[cpu]`.
    All code works identically, just slower.

!!! tip "Verify the installation"
    After installing, confirm JAX can see your GPU:
    ```python
    import jax
    print(jax.devices())   # should print [CudaDevice(id=0), ...]
    ```

### From PyPI

```bash
pip install dantinox                   # core only
pip install "dantinox[data]"          # + HuggingFace datasets
pip install "dantinox[benchmark]"     # + pandas, matplotlib, scipy
pip install "dantinox[all]"           # everything including dev and doc tools
```

---

## 2. Your first model in 10 lines

DantinoX provides three levels of abstraction. The `dx.fit` function is the highest level: it handles everything automatically.

```python title="first_model.py"
import dantinox as dx

# dx.fit builds the model, trains it, and saves the checkpoint
run_dir = dx.fit(
    "ar",                               # paradigm: "ar" | "discrete" | "continuous"
    "data/wiki.txt",                    # text file used for training
    dim=512,                            # embedding dimension (latent space size)
    n_heads=8,                          # number of attention heads
    head_size=64,                       # dimension per head — MUST satisfy: dim = n_heads × head_size
    num_blocks=12,                      # number of Transformer layers
    vocab_size=32_000,                  # vocabulary size
    lr=3e-4,                            # initial learning rate (Adam)
    epochs=5,                           # number of training epochs
)

# run_dir is the saved folder, e.g. "runs/20260611_142301"
print(dx.quick_generate(run_dir, "Once upon a time"))
```

**What happens internally:**

1. `dx.fit` constructs a `Transformer` with the specified configuration
2. Instantiates a `CharTokenizer` (or BPE if `tokenizer_type="bpe"`)
3. Creates a `Trainer` with `AdamW` and a cosine learning-rate schedule
4. Trains for `epochs` epochs, saving the best checkpoint to `runs/<timestamp>/best_model_weights.msgpack`
5. Returns the path to the run folder

!!! warning "Key constraint"
    `dim` must equal exactly `n_heads × head_size`.
    With `n_heads=8` and `head_size=64`, you must use `dim=512`.
    If the values do not match, the constructor raises a `ValueError`.

---

## 3. The three paradigms

DantinoX supports three different ways to generate text, all sharing the same base Transformer architecture.
Only the training objective and the generation procedure differ.

### Paradigm 1 — Autoregressive (AR)

The classic paradigm: generates one token at a time, left to right.
Each generated token depends on all previous tokens.

```python
run_dir = dx.fit(
    "ar",
    "data/wiki.txt",
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
    causal=True,          # applies a causal (lower-triangular) attention mask
    lr=3e-4,
    epochs=5,
)
```

**When to use it:** The simplest paradigm to train and the fastest at inference with KV-cache. Good as a baseline.

### Paradigm 2 — Masked Diffusion (LLaDA / Discrete)

The model is trained to denoise: during training, a fraction of tokens is replaced with a `[MASK]` token, and the model learns to predict all masked positions simultaneously.
At generation time, it starts from a fully masked sequence and unmasks tokens iteratively.

```python
run_dir = dx.fit(
    "discrete",
    "data/wiki.txt",
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
    causal=False,             # bidirectional attention (sees the whole sequence)
    noise_schedule="cosine",  # schedule that controls how many tokens to mask
    mask_token_id=4,          # vocabulary ID of the [MASK] token
    lr=3e-4,
    epochs=20,                # requires more epochs than AR
)
```

**When to use it:** Produces more coherent and diverse outputs than AR on certain tasks.
Inference requires multiple steps but can be accelerated with Fast-dLLM (see Generation section below).

### Paradigm 3 — ELF (Continuous Flow-Matching)

The model operates in the continuous embedding space rather than on discrete tokens.
It transforms Gaussian noise into clean token embeddings using an Euler ODE solver.

```python
run_dir = dx.fit(
    "continuous",
    "data/wiki.txt",
    embed_dim=768,     # dimension of the continuous embedding space
    model_dim=512,     # internal Transformer dimension
    n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_128,
    elf_cfg_scale=1.5, # Classifier-Free Guidance scale (0 = no guidance)
    lr=1e-4,
    epochs=30,
)
```

**When to use it:** Experimental paradigm for research on discrete flow-matching.
Requires more data and more training epochs than AR or diffusion.

---

## 4. Explicit API (Level 2)

If you need more control — for example to customise the optimiser or access the model directly — use the explicit paradigm API.

```python title="explicit_training.py"
import dantinox as dx
from flax import nnx

# Separate architecture config from training config
model_cfg    = dx.ModelConfig(
    dim=512, n_heads=8, head_size=64,
    num_blocks=12, vocab_size=32_000,
    attention_type="gqa",   # use Grouped-Query Attention instead of MHA
    kv_heads=2,             # 2 KV heads shared across 8 query heads
)

training_cfg = dx.TrainingConfig(
    lr=3e-4,
    batch_size=64,
    grad_accum=4,           # effective batch = 64 × 4 = 256
    optimizer="adamw",
    lr_schedule="cosine",
    warmup_steps=400,
    epochs=5,
)

# Build paradigm and model
paradigm = dx.ARParadigm(model_cfg)
model    = paradigm.build_model(nnx.Rngs(params=42))

# Train
run_dir = dx.Trainer(paradigm, training_cfg).fit("data/wiki.txt")

# Load and generate
model  = dx.load(run_dir, paradigm=paradigm)
tokens = paradigm.generate(model, prompt_ids, rng=nnx.Rngs(0))
```

---

## 5. Generation

### AR — autoregressive generation

```python title="generate_ar.py"
from dantinox.generator import Generator

gen    = Generator("runs/ar_512d_12b")
output = gen.generate(
    "In the beginning",
    max_new_tokens=200,
    top_p=0.9,          # nucleus sampling: keep tokens covering 90% of probability mass
    temperature=0.8,    # lower value = less random output
    use_cache=True,     # use static KV-cache for 3-4× faster inference
)
print(output)
```

### Diffusion — generation with Fast-dLLM

```python title="generate_diffusion.py"
from core.generation import fast_dllm_generate
from core.diffusion import make_noise_schedule
from core.config import Config
from core.model import DiffusionTransformer
from flax import nnx

# Load config and model
cfg      = Config.from_yaml("runs/diffusion_512d/config.yaml")
schedule = make_noise_schedule(cfg)
model    = DiffusionTransformer(cfg, rngs=nnx.Rngs(0))
# ... load weights ...

tokens = fast_dllm_generate(
    model,
    prefix=prefix_ids,
    gen_len=128,
    schedule=schedule,
    mask_token_id=cfg.mask_token_id,
    block_size=32,              # decode 32 tokens per block
    use_dual_cache=True,        # dual cache: ~1.8× faster
    confidence_threshold=0.9,  # commit a token once confidence exceeds 90%
)
```

### ELF — generation with flow-matching

```python title="generate_elf.py"
from core.generation import elf_generate

tokens = elf_generate(
    model,
    gen_len=128,
    batch_size=4,
    n_steps=64,       # Euler ODE steps (more steps = higher quality)
    cfg_scale=1.5,    # guidance strength
    seed=42,
)
```

---

## 6. CLI

Every Python operation is also accessible from the command line. Useful for training scripts and automation.

```bash title="Terminal"
# Train using a YAML config file
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt

# Override parameters inline without editing the YAML
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --model_type diffusion \
    --lr 1e-4 \
    --use_bf16 true \
    --n_devices 4

# Generate text from a saved checkpoint
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "In the beginning" \
    --top_p 0.9 \
    --max_new_tokens 300 \
    --stream              # print tokens as they are generated

# Find the optimal learning rate before training
dantinox find-lr \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --plot

# Print parameter count and FLOPs for a checkpoint
dantinox profile --run_dir runs/ar_512d_12b

# Evaluate generation quality (distinct-1, distinct-2, rep-4)
dantinox eval \
    --run_dir runs/ar_512d_12b \
    --n_samples 50 \
    --gen_len 128

# Merge LoRA adapter weights into the base model (for deployment)
dantinox merge-lora \
    --run_dir runs/lora_finetune \
    --out_dir runs/lora_merged
```

See the [CLI Reference](cli.md) for the full list of commands and all their arguments.

---

## 7. Training output structure

When you run a training job, DantinoX saves everything in a structured folder:

```
runs/
└── 20260611_142301/                    ← auto-generated name (date + time)
    ├── config.yaml                     ← exact copy of the config used (fully reproducible)
    ├── best_model_weights.msgpack      ← checkpoint with the best validation loss
    ├── training_log.csv                ← step-by-step log: loss, lr, grad_norm, …
    └── model_summary.json             ← architecture summary (parameter count, FLOPs, …)
```

The `config.yaml` file lets you reproduce the exact same training run in the future,
or resume from where it stopped with `--resume`.

---

## 8. Next steps

<div class="grid cards" markdown>

-   :material-book-open-variant: **Architecture**

    Understand the internal layers: MHA, GQA, MLA, SwiGLU, MoE, RoPE, LoRA.

    [Architecture →](architecture.md)

-   :material-blur: **Masked Diffusion (LLaDA)**

    Forward process, cosine noise schedule, ELBO loss, iterative unmasking.

    [Diffusion Paradigm →](paradigms/diffusion.md)

-   :material-tune: **Training Guide**

    Optimisers (Muon, AdamW, Lion), multi-GPU, gradient accumulation, W&B sweeps.

    [Training →](training/index.md)

-   :material-chef-hat: **Cookbook**

    Copy-paste recipes for every scenario: training, generation, LoRA, Hub, benchmarks.

    [Cookbook →](cookbook.md)

-   :material-console: **CLI Reference**

    All 12 subcommands with complete argument tables.

    [CLI →](cli.md)

-   :material-file-cog: **Configuration**

    Every field of `ModelConfig`, `TrainingConfig`, `Config`, and `ELFConfig` explained in detail.

    [Configuration →](configuration.md)

</div>
