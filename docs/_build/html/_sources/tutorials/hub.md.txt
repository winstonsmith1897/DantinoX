---
title: Pushing to HuggingFace Hub
---

# Pushing to HuggingFace Hub

This tutorial covers publishing a trained DantinoX checkpoint to the HuggingFace Hub, loading it on any machine, and loading it directly into `Generator` by repository name.

**Prerequisites:** a trained checkpoint (see [Training Your First Model](first-model.md)) and a HuggingFace account.

---

## 1. Authentication

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace access token when prompted. Tokens can be created at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `write` scope.

---

## 2. Push a Checkpoint

### CLI

```bash
dantinox push \
  --run_dir runs/run_20260101_120000 \
  --repo my-org/dantinox-shakespeare \
  --private false
```

### Python API

```python
from dantinox.hub import push

push(
    run_dir="runs/run_20260101_120000",
    repo_id="my-org/dantinox-shakespeare",
    private=False,
    token=None,   # uses the token from `huggingface-cli login`
)
```

The following files are uploaded:

| File | Description |
| :--- | :--- |
| `config.yaml` | Full model and training configuration |
| `weights.msgpack` | Model weights (Flax serialization) |
| `tokenizer.json` | Fitted tokenizer (char-level or BPE) |
| `model_summary.json` | Parameter count, FLOPs, architecture summary |
| `training_log.csv` | Per-epoch train/val loss history |

---

## 3. Load from the Hub

Once uploaded, the checkpoint is immediately loadable on any machine:

```python
from dantinox.generator import Generator

# Public repository
gen = Generator("my-org/dantinox-shakespeare")
print(gen.generate("To be, or not to be,", max_new_tokens=200))

# Private repository
gen_private = Generator("my-org/private-model", token="hf_...")
```

`Generator` downloads the checkpoint to a local cache the first time it is called and reuses it on subsequent calls.

---

## 4. Pull a Checkpoint Locally

To download the checkpoint to a specific directory (for fine-tuning, benchmarking, etc.):

```bash
dantinox pull \
  --repo my-org/dantinox-shakespeare \
  --local_dir runs/pulled
```

```python
from dantinox.hub import pull

local_dir = pull(
    repo_id="my-org/dantinox-shakespeare",
    local_dir="runs/pulled",
    token=None,
)
```

The downloaded directory has the same structure as a local run directory and can be passed directly to `Trainer`, `Generator`, or `Transformer.from_pretrained`.

---

## 5. Versioning

HuggingFace Hub uses git under the hood. Every `push` call creates a new commit. To load a specific version:

```python
gen = Generator("my-org/dantinox-shakespeare", revision="v1.0")
```

Tag a release on the Hub web interface or via:

```python
from huggingface_hub import HfApi
HfApi().create_tag("my-org/dantinox-shakespeare", tag="v1.0")
```

---

## 6. Writing a Good Model Card

HuggingFace Hub repositories display a `README.md` as a model card. A well-written model card improves discoverability and reproducibility. A minimal template:

```markdown
---
language: en
license: mit
tags:
  - jax
  - transformer
  - language-model
  - dantinox
---

# my-org/dantinox-shakespeare

A DantinoX GQA Transformer trained on TinyShakespeare.

## Model details

| | |
|---|---|
| Architecture | GQA Transformer (dim=256, layers=6, heads=8) |
| Parameters | ~8 M |
| Training data | TinyShakespeare (~1 M tokens) |
| Context length | 256 tokens |

## Usage

\```python
from dantinox.generator import Generator

gen = Generator("my-org/dantinox-shakespeare")
print(gen.generate("To be, or not to be,", max_new_tokens=200))
\```

## Training

Trained with DantinoX using the `wsd` LR schedule and bfloat16 mixed precision.
See `config.yaml` for full hyperparameters.
```

Create the file at `runs/run_20260101_120000/README.md` before calling `push` — it will be uploaded alongside the weights.

---

## Next Steps

| Goal | Reference |
| :--- | :--- |
| Adapt the pushed model to a new domain | [LoRA Fine-Tuning](lora-fine-tuning.md) |
| Benchmark the pushed model | [Benchmarks](../benchmarks.md) |
| Full Hub API reference | [API Reference — Hub](../api.md) |
