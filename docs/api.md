# API Reference

Auto-generated from source docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

---

## High-level API

The `dantinox` package exposes five classes and two functions that cover the full lifecycle — training, generation, benchmarking, plotting, and Hub sharing — without touching internal modules.

### Trainer

```{eval-rst}
.. autoclass:: dantinox.trainer.Trainer
   :members:
```

---

### Generator

```{eval-rst}
.. autoclass:: dantinox.generator.Generator
   :members:
```

---

### BenchmarkRunner

```{eval-rst}
.. autoclass:: dantinox.bench.BenchmarkRunner
   :members:
```

---

### Plotter

```{eval-rst}
.. autoclass:: dantinox.plotting.Plotter
   :members:
```

---

### Hub

Push, pull, and directly load checkpoints from HuggingFace Hub.

:::{admonition} Optional dependency
:class: tip

Install with `pip install "dantinox[hub]"` or `pip install huggingface-hub`.
:::

:::{admonition} Direct loading — no pull step needed
:class: example

```python
from dantinox import Generator
from dantinox.core.model import Transformer

gen   = Generator("my-org/dantinox-dante")                    # downloads + loads
model = Transformer.from_pretrained("my-org/dantinox-dante")  # same, no tokenizer
```
:::

```{eval-rst}
.. autofunction:: dantinox.hub.resolve_checkpoint
```

---

```{eval-rst}
.. autofunction:: dantinox.hub.push
```

---

```{eval-rst}
.. autofunction:: dantinox.hub.pull
```

---

## Core Modules

Internal implementation. Import directly when you need low-level access.

### Model Architecture

Core Transformer components — `Transformer`, `Block`, `Attention` (MHA/GQA/MLA), `MoE`, and `MLP`.

```{eval-rst}
.. automodule:: dantinox.core.model
   :members:
```

---

### Normalisation

`RMSNorm` is the alternative to `nnx.LayerNorm` selected when `norm_type = "rmsnorm"`.

```{eval-rst}
.. autoclass:: dantinox.core.block.RMSNorm
   :members:
```

---

### Model Output

Each model family returns its own NamedTuple: `ModelOutput` (`Transformer`, AR and diffusion — supports both attribute access and positional unpacking), `FlowMatchingOutput` (`FlowMatchingTransformer` — `x_pred` + `logits`), and `EmbeddingOutput` (`Transformer.encode_hidden` — pooled sentence embeddings + per-token hidden states).

```{eval-rst}
.. autoclass:: dantinox.core.output.ModelOutput
   :members:
```

```{eval-rst}
.. autoclass:: dantinox.core.output.FlowMatchingOutput
   :members:
```

```{eval-rst}
.. autoclass:: dantinox.core.output.EmbeddingOutput
   :members:
```

---

### LoRA Adapters

`LoRAParam` is a distinct NNX variable type that freezes base weights at the type level. `LoRALinear` is a drop-in replacement for `nnx.Linear` with a trainable low-rank delta.

```{eval-rst}
.. autoclass:: dantinox.core.lora.LoRAParam
   :members:
```

```{eval-rst}
.. autoclass:: dantinox.core.lora.LoRALinear
   :members:
```

---

### Sharding Utilities

SPMD helpers built on `jax.sharding` — 1-D data-parallel meshes plus 2-D data × model meshes with Megatron-style tensor parallelism. Pass `n_devices` / `tp_size` in the config to activate automatically, or call these directly for custom sharding strategies (see [Multi-GPU Training](training/multi-gpu.md)).

```{eval-rst}
.. automodule:: dantinox.core.sharding
   :members:
```

---

### Checkpoint Loading

Single checkpoint-loading path for run directories: config-format detection (`Config` / `ModelConfig` / `FlowMatchingConfig`), weight-file resolution across current and legacy filenames, msgpack decoding, and in-place weight restoration. Shared by `Transformer.from_pretrained`, the `pipeline()` helper, and StableHLO export.

```{eval-rst}
.. automodule:: dantinox.core.checkpoint
   :members:
```

---

### Inference Pipeline

One-call inference helper: builds the right model from a run directory and dispatches to AR decoding, reverse diffusion, or flow-matching ODE integration. For richer control (streaming, decoding strategies) prefer the `Generator` class.

```{eval-rst}
.. automodule:: dantinox.core.pipeline
   :members:
```

---

### StableHLO Export

Ahead-of-time export of a checkpoint to a portable StableHLO binary for Python-free inference — the implementation behind the `dantinox export` CLI subcommand.

```{eval-rst}
.. automodule:: dantinox.core.export
   :members:
```

---

### Configuration

The `Config` dataclass is the single source of truth for all architectural and training hyperparameters.

```{eval-rst}
.. automodule:: dantinox.core.config
   :members:
```

---

### Generation Engine

Autoregressive inference with static KV-cache management, `jax.lax.fori_loop` token loop, and sampling strategies (greedy, Top-K, Top-P).

```{eval-rst}
.. automodule:: dantinox.core.generation
   :members:
```

---

### Tokenizers

Character-level and Byte-Level BPE tokenizers with save/load support.

```{eval-rst}
.. automodule:: dantinox.utils.tokenizer
   :members:
```

---

## CLI Reference

The `dantinox` command provides 14 subcommands (full argument tables in the [CLI Reference](cli.md)):

| Subcommand | Description |
| :--- | :--- |
| `train` | Train a model from a config and corpus |
| `generate` | Generate text from a checkpoint |
| `sweep` | Run a W&B Bayesian hyperparameter sweep |
| `benchmark` | Benchmark throughput and FLOPs for run directories |
| `find-lr` | Run the LR range test and suggest a learning rate |
| `push` | Upload a checkpoint to HuggingFace Hub |
| `pull` | Download a checkpoint from HuggingFace Hub |
| `infbench` | Full inference benchmark suite (random-model sweep + trained pipeline) |
| `merge-lora` | Merge LoRA adapters into base weights and save |
| `profile` | Print parameter count and FLOPs for a checkpoint |
| `run` | Declarative training from a workflow YAML |
| `export` | Export a checkpoint to a StableHLO binary |
| `eval` | Evaluate generation quality for a checkpoint |
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
