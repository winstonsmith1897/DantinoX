# API Reference

Auto-generated from source docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

---

## High-level API

The `dantinox` package exposes five classes and two functions that cover the full lifecycle — training, generation, benchmarking, plotting, and Hub sharing — without touching internal modules.

### Trainer

::: dantinox.trainer.Trainer
    options:
      show_source: true
      members:
        - __init__
        - fit
        - find_lr

---

### Generator

::: dantinox.generator.Generator
    options:
      show_source: true
      members:
        - __init__
        - generate
        - generate_batch
        - stream

---

### BenchmarkRunner

::: dantinox.bench.BenchmarkRunner
    options:
      show_source: true
      members:
        - __init__
        - run

---

### Plotter

::: dantinox.plotting.Plotter
    options:
      show_source: true
      members:
        - __init__
        - run

---

### Hub

Push, pull, and directly load checkpoints from HuggingFace Hub.

!!! tip "Optional dependency"
    Install with `pip install "dantinox[hub]"` or `pip install huggingface-hub`.

!!! example "Direct loading — no pull step needed"
    ```python
    from dantinox import Generator
    from core import Transformer

    gen   = Generator("my-org/dantinox-dante")                    # downloads + loads
    model = Transformer.from_pretrained("my-org/dantinox-dante")  # same, no tokenizer
    ```

::: dantinox.hub.resolve_checkpoint

---

::: dantinox.hub.push

---

::: dantinox.hub.pull

---

## Core Modules

Internal implementation. Import directly when you need low-level access.

### Model Architecture

Core Transformer components — `Transformer`, `Block`, `Attention` (MHA/GQA/MLA), `MoE`, and `MLP`.

::: core.model
    options:
      members_order: alphabetical
      show_source: true

---

### Normalisation

`RMSNorm` is the alternative to `nnx.LayerNorm` selected when `norm_type = "rmsnorm"`.

::: core.block.RMSNorm
    options:
      show_source: true

---

### Model Output

`Transformer.__call__` returns a `ModelOutput` NamedTuple — supports both attribute access and positional unpacking.

::: core.output.ModelOutput
    options:
      show_source: true

---

### LoRA Adapters

`LoRAParam` is a distinct NNX variable type that freezes base weights at the type level. `LoRALinear` is a drop-in replacement for `nnx.Linear` with a trainable low-rank delta.

::: core.lora.LoRAParam
    options:
      show_source: true

::: core.lora.LoRALinear
    options:
      show_source: true
      members:
        - __init__
        - __call__
        - merge_weights

---

### Sharding Utilities

SPMD data-parallel helpers built on `jax.sharding`. Pass `n_devices` in `Config` to activate automatically, or call these directly for custom sharding strategies.

::: core.sharding
    options:
      show_source: true
      members:
        - make_mesh
        - replicate
        - shard_batch
        - num_devices

---

### Configuration

The `Config` dataclass is the single source of truth for all architectural and training hyperparameters.

::: core.config
    options:
      show_root_heading: true

---

### Generation Engine

Autoregressive inference with static KV-cache management, `jax.lax.fori_loop` token loop, and sampling strategies (greedy, Top-K, Top-P).

::: core.generation
    options:
      show_source: true

---

### Tokenizers

Character-level and Byte-Level BPE tokenizers with save/load support.

::: utils.tokenizer
    options:
      show_source: true
      members:
        - Tokenizer
        - CharTokenizer
        - BPETokenizer
        - get_tokenizer
        - load_tokenizer_from_file

---

## CLI Reference

The `dantinox` command provides eight subcommands:

| Subcommand | Description |
| :--- | :--- |
| `train` | Train a model from a config and corpus |
| `generate` | Generate text from a checkpoint |
| `find-lr` | Run the LR range test and suggest a learning rate |
| `push` | Upload a checkpoint to HuggingFace Hub |
| `pull` | Download a checkpoint from HuggingFace Hub |
| `sweep` | Run a W&B Bayesian hyperparameter sweep |
| `benchmark` | Benchmark throughput and FLOPs for run directories |
| `plot` | Generate figures from benchmark results |

```bash
dantinox --version
dantinox --help
dantinox train --help
dantinox find-lr --help
dantinox push --help
```

### `train`

```
dantinox train
  --config PATH          YAML config file (default: configs/default_config.yaml)
  --data_path PATH       Training corpus
  --run_dir PATH         Output directory (auto-generated if omitted)
  --wandb_project NAME   W&B project for logging
  --resume               Resume from last checkpoint in --run_dir
  --<field> VALUE        Override any Config field (e.g. --lr 3e-4 --use_bf16 True)
```

### `generate`

```
dantinox generate
  --run_dir PATH         Run directory with config + weights (required)
  --prompt TEXT          Input prefix (default: "Nel mezzo del cammin ")
  --max_new_tokens N     Tokens to generate (default: 150)
  --greedy               Greedy decoding
  --temperature FLOAT    Softmax temperature (default: 1.0)
  --top_k INT            Top-k sampling
  --top_p FLOAT          Nucleus sampling threshold
  --no_cache             Disable KV cache
  --seed INT             RNG seed (default: 42)
```

### `find-lr`

```
dantinox find-lr
  --config PATH          YAML config file
  --data_path PATH       Training corpus (required)
  --min_lr FLOAT         Start LR (default: 1e-7)
  --max_lr FLOAT         End LR (default: 1.0)
  --num_steps INT        Sweep steps (default: 100)
  --plot                 Save a lr_finder.png loss curve
  --plot_out PATH        Custom output path for the PNG
  --<field> VALUE        Override any Config field
```

### `push`

```
dantinox push
  --run_dir PATH         Local run directory to upload (required)
  --repo NAME            Hub repo id, e.g. my-org/my-model (required)
  --private              Create a private repository
  --token TOKEN          HuggingFace access token
  --message TEXT         Commit message
```

### `pull`

```
dantinox pull
  --repo NAME            Hub repo id (required)
  --local_dir PATH       Where to save the files
  --token TOKEN          HuggingFace access token
  --revision REF         Branch, tag, or commit SHA
```
