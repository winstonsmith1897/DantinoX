---
title: DantinoX vs HuggingFace Transformers
---

# DantinoX vs HuggingFace Transformers

DantinoX and HuggingFace Transformers serve different goals. This page helps you choose, and shows how familiar HF patterns translate to DantinoX code.

---

## At a glance

| | **DantinoX** | **HuggingFace Transformers** |
|---|---|---|
| **Framework** | JAX + Flax NNX | PyTorch (primary) |
| **Generation paradigms** | AR + Masked Diffusion + ELF | Primarily AR |
| **Training abstraction** | `Paradigm.loss_fn` owns the objective | `Trainer` + model `.forward()` |
| **State management** | Functional (`nnx.state` / `nnx.update`) | Stateful (`model.parameters()`) |
| **JIT / compilation** | XLA JIT via `jax.jit` | `torch.compile` |
| **Attention variants** | MHA, GQA, MLA, Flash | MHA, GQA (via SDPA) |
| **KV cache** | Static pre-allocated, DualCache | Dynamic |
| **LoRA** | Built-in (`use_lora=True`) | PEFT library |
| **Hub integration** | `dantinox push` / `dantinox pull` | `model.push_to_hub()` |
| **Multi-GPU** | JAX SPMD data parallelism | DDP / FSDP |
| **Focus** | Research: architecture experiments, paradigm comparison | Production: pretrained model ecosystem |

---

## Defining a model

=== "DantinoX"

    ```python
    from core.config import ModelConfig
    from core.model import Transformer
    from flax import nnx

    cfg   = ModelConfig(
        dim=512, n_heads=8, head_size=64,
        num_blocks=12, vocab_size=32000,
        attention="gqa", kv_heads=2,
    )
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    ```

=== "HuggingFace"

    ```python
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg   = GPT2Config(
        n_embd=512, n_head=8, n_layer=12,
        vocab_size=32000,
    )
    model = GPT2LMHeadModel(cfg)
    ```

---

## Training loop

=== "DantinoX"

    ```python
    from dantinox.paradigms.ar import ARParadigm
    from dantinox.trainer import Trainer
    from core.config import Config

    cfg     = Config.from_yaml("configs/default_config.yaml")
    trainer = Trainer(cfg)
    run_dir = trainer.fit("wiki.txt")      # full training loop in one call
    ```

    Under the hood, `Trainer` calls `paradigm.loss_fn(model, batch)` at every step — the loss function is owned by the paradigm, not the model.

=== "HuggingFace"

    ```python
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir="./runs",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
    trainer.train()
    ```

---

## Manual training step

=== "DantinoX (JAX)"

    ```python
    import jax, optax
    from flax import nnx

    tx      = optax.adamw(3e-4)
    opt_st  = tx.init(nnx.state(model, nnx.Param))

    @jax.jit
    def step(model, opt_state, batch):
        def loss_fn(params):
            nnx.update(model, params)
            loss, _ = paradigm.loss_fn(model, batch)
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(nnx.state(model, nnx.Param))
        updates, opt_state = tx.update(grads, opt_state)
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss, opt_state
    ```

=== "HuggingFace (PyTorch)"

    ```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    ```

---

## Loading weights

=== "DantinoX"

    ```python
    import msgpack
    from flax import nnx
    from flax.serialization import _msgpack_ext_unpack

    with open("runs/my_run/best_model_weights.msgpack", "rb") as f:
        state = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack,
                                strict_map_key=False)
    nnx.update(model, state)
    ```

=== "HuggingFace"

    ```python
    # From the Hub
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # From a local directory
    model = GPT2LMHeadModel.from_pretrained("./my_model")
    ```

---

## AR generation

=== "DantinoX"

    ```python
    from core.generation import generate

    tokens = generate(
        model, prompt_ids,
        max_generations=200,
        top_p=0.9, temperature=0.8,
        use_cache=True,
    )
    ```

=== "HuggingFace"

    ```python
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9, temperature=0.8,
    )
    ```

---

## Masked Diffusion generation (DantinoX-exclusive)

HuggingFace has no built-in support for non-autoregressive discrete diffusion or continuous flow-matching. DantinoX provides both:

```python
from core.generation import diffusion_generate, fast_dllm_generate
from core.diffusion import make_noise_schedule

schedule = make_noise_schedule(cfg)

# Standard iterative unmasking
tokens = diffusion_generate(
    model, prefix, gen_len=128,
    schedule=schedule, mask_token_id=cfg.mask_token_id,
)

# Fast-dLLM DualCache: 1.4–2.1× faster
tokens = fast_dllm_generate(
    model, prefix, gen_len=256,
    schedule=schedule, mask_token_id=cfg.mask_token_id,
    block_size=32, steps_per_block=20,
    confidence_threshold=0.9,
)
```

---

## LoRA fine-tuning

=== "DantinoX"

    ```python
    cfg = Config.from_dict({
        **base_cfg_dict,
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_targets": "attention",
    })
    # Base weights are frozen automatically — no manual filtering.
    model = Transformer(cfg, rngs=nnx.Rngs(42))
    ```

=== "HuggingFace + PEFT"

    ```python
    from peft import get_peft_model, LoraConfig, TaskType

    peft_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_cfg)
    ```

---

## Hub push / pull

=== "DantinoX"

    ```bash
    dantinox push --run_dir runs/ar_mha_512d --repo my-org/my-model
    dantinox pull --repo my-org/my-model --local_dir runs/downloaded
    ```

=== "HuggingFace"

    ```python
    model.push_to_hub("my-org/my-model")
    model = GPT2LMHeadModel.from_pretrained("my-org/my-model")
    ```

---

## When to choose each

!!! success "Use DantinoX when:"
    - You are researching non-autoregressive generation (masked diffusion, flow matching).
    - You need to compare AR vs. Diffusion vs. ELF with identical architecture and training.
    - You need fine-grained control over attention variant (MHA/GQA/MLA), KV-cache type, or noise schedule.
    - Your training loop is JAX-native and you want zero-overhead SPMD parallelism.
    - You need the systematic benchmark suite for reproducible throughput and quality numbers.

!!! info "Use HuggingFace when:"
    - You want to fine-tune one of thousands of pretrained models in the Hub ecosystem.
    - Your task requires an existing tokenizer, feature extractor, or architecture (BERT, T5, Llama, …).
    - Your team is PyTorch-native and wants minimal friction.
    - You need production integrations: ONNX export, TorchScript, Inference API.
