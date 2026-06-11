---
title: Cookbook
---

# Cookbook

Short, copy-paste recipes for the most common DantinoX patterns.

<div class="grid cards" markdown>

-   :material-play-circle: [Train AR model](#1-train-an-ar-model-on-a-local-file)
-   :material-blur: [Train Diffusion model](#2-train-a-masked-diffusion-llada-model)
-   :material-wave: [Train ELF model](#3-train-an-elf-continuous-flow-model)
-   :material-restore: [Resume training](#4-resume-interrupted-training)
-   :material-text-box-outline: [Generate (AR)](#5-generate-text-from-ar)
-   :material-blur-radial: [Generate (Diffusion)](#6-generate-text-from-diffusion)
-   :material-tune: [LoRA fine-tuning](#7-lora-fine-tuning)
-   :material-code-braces: [Load for inference](#8-load-a-model-for-inference)
-   :material-cloud-upload: [Hub push/pull](#9-push--pull-to-huggingface-hub)
-   :material-magnify: [LR range test](#10-find-the-optimal-learning-rate)
-   :material-chart-bar: [Benchmark](#11-benchmark-trained-checkpoints)
-   :material-counter: [Parameter count & FLOPs](#12-parameter-count-and-flops)
-   :material-server-network: [Multi-GPU training](#13-multi-gpu-training)
-   :material-swap-horizontal: [Convert configs](#14-convert-between-config-apis)

</div>

---

## 1. Train an AR model on a local file

=== "Python"

    ```python
    import dantinox as dx

    run_dir = dx.fit(
        "ar",
        "data/corpus.txt",
        dim=256, n_heads=8, head_size=32, num_blocks=6,
        vocab_size=200, max_context=512,
        lr=3e-4, epochs=10, batch_size=32,
    )
    print("Checkpoint:", run_dir)
    ```

=== "CLI"

    ```bash
    dantinox train \
        --config configs/default_config.yaml \
        --data_path data/corpus.txt
    ```

---

## 2. Train a Masked Diffusion (LLaDA) model

=== "Python"

    ```python
    import dantinox as dx

    run_dir = dx.fit(
        "diffusion",
        "data/corpus.txt",
        dim=256, n_heads=8, head_size=32, num_blocks=6,
        vocab_size=32000, max_context=512,
        model_type="diffusion",
        diffusion_steps=1000,
        noise_schedule="cosine",
        tokenizer_type="bpe",
        tokenizer_path="t5-base",
        lr=1e-4, epochs=20, batch_size=16,
    )
    ```

=== "CLI"

    ```bash
    dantinox train \
        --config configs/diffusion_base.yaml \
        --data_path wiki.txt \
        --model_type diffusion \
        --noise_schedule cosine \
        --tokenizer_type bpe
    ```

---

## 3. Train an ELF (continuous flow) model

=== "Python"

    ```python
    import dantinox as dx

    run_dir = dx.fit(
        "elf",
        "data/corpus.txt",
        model_type="elf",
        dim=256, n_heads=8, head_size=32, num_blocks=6,
        vocab_size=32000, max_context=256,
        embed_dim=256, bottleneck_dim=64,
        elf_n_steps=64, elf_cfg_scale=1.5,
        tokenizer_type="bpe", tokenizer_path="t5-base",
        lr=1e-4, epochs=30,
    )
    ```

---

## 4. Resume interrupted training

=== "CLI"

    ```bash
    dantinox train \
        --config configs/default_config.yaml \
        --data_path wiki.txt \
        --run_dir runs/ar_mha_512d_12b \
        --resume
    ```

=== "Python"

    ```python
    from dantinox.trainer import Trainer
    from dantinox.core.config import Config

    cfg     = Config.from_yaml("runs/ar_mha_512d_12b/config.yaml")
    trainer = Trainer(cfg)
    trainer.fit("wiki.txt", run_dir="runs/ar_mha_512d_12b", resume=True)
    ```

---

## 5. Generate text from AR

=== "CLI — streaming"

    ```bash
    dantinox generate \
        --run_dir runs/ar_mha_512d_12b \
        --prompt "In the beginning" \
        --stream --top_p 0.9
    ```

=== "CLI — batch"

    ```bash
    dantinox generate \
        --run_dir runs/ar_mha_512d_12b \
        --prompt "In the beginning" \
        --top_p 0.9 --temperature 0.8 \
        --max_new_tokens 300
    ```

=== "Python"

    ```python
    from dantinox.generator import Generator

    gen  = Generator("runs/ar_mha_512d_12b")
    text = gen.generate("In the beginning", max_new_tokens=200, top_p=0.9)
    print(text)

    # Token-by-token streaming
    for chunk in gen.stream("In the beginning", max_new_tokens=200):
        print(chunk, end="", flush=True)
    ```

---

## 6. Generate text from Diffusion

```python
import yaml, msgpack
import jax.numpy as jnp
from flax import nnx
from flax.serialization import _msgpack_ext_unpack
from dantinox.core.config import Config
from dantinox.core.model import DiffusionTransformer
from dantinox.core.generation import diffusion_generate
from dantinox.core.diffusion import make_noise_schedule

# Load config and model
with open("runs/diff_mha_512d/config.yaml") as f:
    cfg = Config.from_dict(yaml.safe_load(f))

model = DiffusionTransformer(cfg, rngs=nnx.Rngs(42))
with open("runs/diff_mha_512d/best_model_weights.msgpack", "rb") as f:
    state = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
nnx.update(model, state)

# Generate (iterative unmasking)
schedule = make_noise_schedule(cfg)
prefix   = jnp.zeros((1, 0), dtype=jnp.int32)    # empty prefix = unconditional
tokens   = diffusion_generate(
    model, prefix,
    gen_len=128,
    schedule=schedule,
    mask_token_id=cfg.mask_token_id,
    seed=42,
)
```

!!! tip "Fast-dLLM DualCache"
    For 1.4–2.1× faster generation, use `fast_dllm_generate` with `block_size=32`. See [Fast-dLLM DualCache](paradigms/fast-dllm.md).

---

## 7. LoRA fine-tuning

```python
import yaml
from dantinox.core.config import Config
from dantinox.core.model import Transformer
from flax import nnx
import msgpack
from flax.serialization import _msgpack_ext_unpack
from dantinox.trainer import Trainer

# Load the base checkpoint config and inject LoRA
with open("runs/ar_base/config.yaml") as f:
    raw = yaml.safe_load(f)

lora_cfg = Config.from_dict({
    **raw,
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_targets": "attention",
    "lr": 1e-3,    # higher LR fine for adapters — base weights are frozen
    "epochs": 5,
})

# Build model and load pretrained weights
model = Transformer(lora_cfg, rngs=nnx.Rngs(42))
with open("runs/ar_base/best_model_weights.msgpack", "rb") as f:
    state = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
nnx.update(model, state)

# Fine-tune — only LoRA adapters are updated
ft_run = Trainer(lora_cfg).fit("data/new_domain.txt")
```

Merge adapters into base weights before deployment:

```python
from dantinox.core.lora import merge_lora
merged = merge_lora(model)    # pure base architecture, no LoRA overhead
```

---

## 8. Load a model for inference

```python
import yaml, msgpack
from flax import nnx
from flax.serialization import _msgpack_ext_unpack
from dantinox.core.config import Config
from dantinox.core.model import Transformer

def load_model(run_dir: str):
    with open(f"{run_dir}/config.yaml") as f:
        cfg = Config.from_dict(yaml.safe_load(f))
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    for fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
        try:
            with open(f"{run_dir}/{fname}", "rb") as f:
                state = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
            nnx.update(model, state)
            return model, cfg
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No weights found in {run_dir}")

model, cfg = load_model("runs/ar_mha_512d_12b")
```

---

## 9. Push / pull to HuggingFace Hub

=== "CLI"

    ```bash
    # Upload
    dantinox push \
        --run_dir runs/ar_mha_512d \
        --repo my-org/dantinox-ar-medium \
        --private

    # Download
    dantinox pull \
        --repo my-org/dantinox-ar-medium \
        --local_dir runs/ar_from_hub
    ```

=== "Python"

    ```python
    from dantinox.hub import push, pull

    push("runs/ar_mha_512d", "my-org/dantinox-ar-medium", private=True)
    pull("my-org/dantinox-ar-medium", local_dir="runs/ar_from_hub")
    ```

---

## 10. Find the optimal learning rate

=== "CLI"

    ```bash
    dantinox find-lr \
        --config configs/default_config.yaml \
        --data_path wiki.txt \
        --plot
    # → Suggested learning rate: 3.47e-04
    # → Plot saved to: lr_finder.png
    ```

=== "Python"

    ```python
    from dantinox.trainer import Trainer
    from dantinox.core.config import Config

    cfg  = Config.from_yaml("configs/default_config.yaml")
    t    = Trainer(cfg)
    lr, _, _ = t.find_lr("wiki.txt", min_lr=1e-7, max_lr=1.0, num_steps=150)
    print(f"Suggested LR: {lr:.2e}")
    ```

---

## 11. Benchmark trained checkpoints

=== "All runs"

    ```bash
    dantinox infbench \
        --trained \
        --runs-dir runs \
        --trained-csv results/my_benchmark.csv \
        --n-trials 20
    ```

=== "Selected runs only"

    ```bash
    dantinox benchmark \
        --runs_dir runs \
        --runs ar_mha_512d diff_mha_512d \
        --out_csv results/comparison.csv
    ```

---

## 12. Parameter count and FLOPs

```python
import jax
from flax import nnx
from dantinox.core.config import ModelConfig
from dantinox.core.model import Transformer
from dantinox.profiling import count_flops

cfg    = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32000)
model  = Transformer(cfg, rngs=nnx.Rngs(0))

params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Parameters: {params / 1e6:.1f}M")

flops = count_flops(cfg, seq_len=512, batch_size=1)
print(f"FLOPs (seq=512): {flops.total / 1e9:.2f} GFLOPs")
```

---

## 13. Multi-GPU training

```bash
dantinox train \
    --config configs/large.yaml \
    --data_path wiki.txt \
    --n_devices 4 \
    --grad_accum 8 \
    --batch_size 32 \
    --use_bf16 true
```

!!! info "Effective batch size"
    With the flags above: `32 × 8 × 4 = 1024` tokens per step. JAX SPMD replicates the model on all 4 devices and reduces gradients automatically — no code changes needed.

---

## 14. Convert between config APIs

```python
from dantinox.core.config import Config

cfg = Config.from_yaml("configs/default_config.yaml")

# To the new split API
model_cfg    = cfg.to_model_config()   # → ModelConfig
elf_cfg      = cfg.to_elf_config()     # → ELFConfig (when model_type="elf")

# Use with Paradigm API
from dantinox.paradigms.ar import ARParadigm
paradigm = ARParadigm(model_cfg)
```

---

!!! tip "More examples"
    See the [Notebooks](notebooks/index.md) for interactive, runnable versions of these recipes on Google Colab.
