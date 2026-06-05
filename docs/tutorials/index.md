---
title: Tutorials
---

# Tutorials

Step-by-step guides that take you from installation to a working model. Each tutorial is self-contained and builds on the previous one.

## Prerequisites

- Python 3.12+
- JAX with CUDA 12: `pip install -U "jax[cuda12]"`
- DantinoX: `pip install -e ".[all]"` (from the repository root)

If you are new to JAX, the [JAX quickstart](https://jax.readthedocs.io/en/latest/quickstart.html) covers the key concepts (JIT compilation, functional transforms, device arrays) in about 15 minutes.

---

## Available Tutorials

| Tutorial | What you will learn |
| :--- | :--- |
| [Training Your First Model](first-model.md) | Train a character-level AR Transformer on a text corpus, evaluate it, and generate text |
| [LoRA Fine-Tuning](lora-fine-tuning.md) | Adapt a pretrained checkpoint to a new domain using LoRA adapters |
| [Masked Diffusion LM](diffusion-lm.md) | Train a Masked Diffusion Language Model and use it for generation and infilling |
| [Pushing to HuggingFace Hub](hub.md) | Publish a trained checkpoint, load it on any machine, and share it publicly |

---

## Choosing the Right Starting Point

**I want to train a basic model quickly.**  
→ Start with [Training Your First Model](first-model.md). A small GQA model on a local text file finishes in under 10 minutes on a single GPU.

**I have a pretrained checkpoint and want to specialise it.**  
→ Go to [LoRA Fine-Tuning](lora-fine-tuning.md). LoRA trains ~0.2 % of parameters, so it is much faster than full fine-tuning.

**I want to explore non-autoregressive generation.**  
→ See [Masked Diffusion LM](diffusion-lm.md). Diffusion models enable native infilling and bidirectional context.

**I want to share my model.**  
→ See [Pushing to HuggingFace Hub](hub.md). A single `dantinox push` command packages and uploads the checkpoint.
