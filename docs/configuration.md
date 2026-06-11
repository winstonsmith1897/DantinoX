---
title: Configuration Reference
---

# Configuration Reference

<div class="grid cards" markdown>

-   :material-cpu-64-bit: **ModelConfig**

    Architecture-only config for `Transformer`.
    Use with the new Paradigm API.

    [Jump to ModelConfig →](#modelconfig)

-   :material-lightning-bolt: **TrainingConfig**

    Training hyper-parameters, dataset, and device settings.
    Completely independent of architecture.

    [Jump to TrainingConfig →](#trainingconfig)

-   :material-file-cog: **Config** (monolithic)

    Legacy flat config used by the CLI and YAML files.
    Combines all fields from both classes above.

    [Jump to Config →](#config-monolithic)

-   :material-wave: **ELFConfig**

    Architecture config for `ELFTransformer`.
    Use when constructing ELF models directly.

    [Jump to ELFConfig →](#elfconfig)

</div>

!!! abstract "Key constraint"
    `dim` must always equal `n_heads × head_size`. This is validated in `__post_init__` and will raise `ValueError` if violated.

    ```python
    ModelConfig(dim=512, n_heads=8, head_size=64)  # ✓  512 = 8 × 64
    ModelConfig(dim=512, n_heads=8, head_size=32)  # ✗  raises ValueError
    ```

---

## ModelConfig

Architecture specification for `Transformer` (AR and Diffusion). Everything here describes *what the model is* — not how it trains.

```python
from core.config import ModelConfig
from core.model import Transformer
from flax import nnx

cfg   = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32000)
model = Transformer(cfg, rngs=nnx.Rngs(42))
```

### Core dimensions

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `dim` | `int` | `512` | Model hidden dimension. Must equal `n_heads × head_size`. |
| `n_heads` | `int` | `16` | Number of query attention heads. |
| `head_size` | `int` | `32` | Per-head key/value dimension. `dim = n_heads × head_size`. |
| `num_blocks` | `int` | `12` | Number of transformer layers. |
| `vocab_size` | `int` | `200` | Vocabulary size including all special tokens. |
| `max_context` | `int` | `512` | Maximum sequence length for positional encoding and KV cache. |

### Architecture choices

| Field | Type | Default | Valid values | Description |
|:------|:-----|:-------:|:-------------|:------------|
| `attention` | `str` | `"mha"` | `"mha"` · `"gqa"` · `"mla"` | Attention variant. |
| `ffn` | `str` | `"mlp"` | `"mlp"` · `"moe"` | Feed-forward variant. |
| `norm` | `str` | `"rmsnorm"` | `"rmsnorm"` · `"layernorm"` | Normalisation type. |
| `pos_encoding` | `str` | `"rotary"` | `"rotary"` · `"absolute"` · `"learned"` · `"none"` | Positional encoding. |
| `causal` | `bool` | `True` | — | `True` = AR (causal mask); `False` = bidirectional (diffusion). |

### Regularisation

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `dropout` | `float` | `0.0` | Dropout probability after attention and FFN. |
| `weight_tying` | `bool` | `True` | Tie input embedding and output projection (reduces parameters). |
| `gradient_checkpointing` | `bool` | `False` | Recompute activations in backward pass — saves memory, costs extra FLOPs. |

### Attention settings

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `kv_heads` | `int\|None` | `None` | KV heads for GQA. `None` → same as `n_heads` (standard MHA). |
| `use_flash` | `bool` | `False` | Flash Attention kernel (requires Ampere GPU or newer). |
| `rope_scale` | `float` | `1.0` | RoPE frequency scaling. Values > 1 extend the effective context window. |
| `sliding_window` | `bool` | `False` | Limit attention to a local sliding window. |
| `context_window` | `int` | `4` | Number of blocks in the sliding window (used when `sliding_window=True`). |
| `no_sink` | `bool` | `False` | Disable attention sink token. |

??? note "MLA-specific fields"
    Only relevant when `attention="mla"`. Skip if using MHA or GQA.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `down_dim_q` | `int` | `256` | Query latent compression dimension. |
    | `down_dim_kv` | `int` | `256` | Key/value latent compression dimension. |
    | `rope_dim` | `int` | `32` | RoPE subspace dimension. Must be ≤ `head_size`. |
    | `inference_mode` | `bool` | `False` | Absorb KV projection at inference for reduced compute. |

### Feed-forward

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `expansion` | `int` | `4` | FFN hidden width = `dim × expansion`. |
| `use_swiglu` | `bool` | `True` | SwiGLU gate activation (gated linear unit). |
| `activation` | `str` | `"gelu"` | Activation when `use_swiglu=False`. |

??? note "MoE fields (ffn=&quot;moe&quot; only)"
    Only relevant when `ffn="moe"`. Skip for dense models.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `n_experts` | `int` | `4` | Number of expert FFN heads. |
    | `top_k` | `int` | `2` | Active experts per token. |
    | `moe_balance_coeff` | `float` | `0.1` | Load-balancing auxiliary loss coefficient. |

### LoRA

!!! tip "LoRA fine-tuning"
    Set `use_lora=True` to inject adapter matrices. The `Trainer` automatically freezes base `nnx.Param` weights — only `LoRAParam` weights are updated. See [Cookbook → LoRA fine-tuning](cookbook.md#7-lora-fine-tuning).

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `use_lora` | `bool` | `False` | Inject LoRA adapters. When `True`, base weights are frozen. |
| `lora_rank` | `int` | `8` | Rank `r`. Each projection gains `2 × r × d` trainable parameters. |
| `lora_alpha` | `float` | `16.0` | Scaling factor; effective LR multiplier = `alpha / rank`. |
| `lora_dropout` | `float` | `0.0` | Dropout inside LoRA adapters. |
| `lora_targets` | `str` | `"attention"` | Where to inject: `"attention"` (Q/K/V/O), `"ffn"`, or `"all"`. |

---

## TrainingConfig

Training hyperparameters, dataset, and hardware settings. Completely independent of model architecture — mix and match with any `ModelConfig`.

```python
from core.config import TrainingConfig

cfg = TrainingConfig(lr=3e-4, batch_size=32, epochs=100, optimizer="adamw")
```

### Optimisation

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `lr` | `float` | `3e-4` | Peak learning rate (after warmup). |
| `batch_size` | `int` | `32` | Micro-batch size per device per step. |
| `grad_accum` | `int` | `1` | Gradient accumulation steps. Effective batch = `batch_size × grad_accum × n_devices`. |
| `epochs` | `int` | `100` | Maximum training epochs. |
| `warmup_steps` | `int` | `400` | Linear warmup steps before LR schedule begins. |
| `lr_schedule` | `str` | `"cosine"` | LR schedule after warmup: `"cosine"` · `"linear"` · `"constant"` · `"wsd"`. |
| `optimizer` | `str` | `"adamw"` | Optimizer: `"adamw"` · `"adafactor"` · `"lion"` · `"adam"` · `"muon"`. |
| `grad_clip` | `float` | `1.0` | Global gradient-norm clipping threshold. |
| `patience` | `int` | `0` | Early stopping patience in epochs. `0` = disabled. |
| `eval_iters` | `int` | `20` | Validation batches per evaluation. |
| `seed` | `int` | `42` | Random seed for weight initialisation and data shuffling. |
| `use_bf16` | `bool` | `False` | bfloat16 mixed precision. Requires NVIDIA Ampere+. |

### Hardware

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `n_devices` | `int` | `0` | GPU count. `0` = use all available devices. |

### Dataset

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `dataset_source` | `str` | `"local"` | `"local"` (plain text file) or `"huggingface"` (HF Datasets). |
| `dataset_name` | `str` | `""` | HuggingFace dataset identifier (e.g. `"wikitext"`). |
| `dataset_config` | `str` | `""` | HuggingFace dataset config (e.g. `"wikitext-103-raw-v1"`). |
| `dataset_text_field` | `str` | `"text"` | Column containing text in HF datasets. |
| `dataset_split` | `str` | `"train"` | Dataset split to use. |
| `max_train_tokens` | `int` | `10_000_000` | Token budget for training. Corpus is truncated if larger. |
| `streaming` | `bool` | `False` | HF streaming mode — avoids downloading the full dataset. |

### Tokenizer

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `tokenizer_type` | `str` | `"char"` | `"char"` (character-level) or `"bpe"` (HuggingFace BPE). |
| `tokenizer_path` | `str\|None` | `None` | HF tokenizer identifier (e.g. `"t5-base"`) or local path. |

### Diffusion training

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `noise_schedule` | `str` | `"linear"` | Masking schedule: `"linear"` · `"cosine"` · `"sqrt"`. |

---

## Config (monolithic)

The `Config` class is the **legacy flat config** used by the CLI and YAML files. It combines all `ModelConfig` and `TrainingConfig` fields, plus ELF-specific fields.

!!! tip "Prefer the split API for new experiments"
    Use `ModelConfig` + `TrainingConfig` for new code — the split is cleaner and more composable. Use `Config` when working with the CLI, existing YAML files, or `Trainer` directly.

```python
cfg = Config.from_yaml("configs/medium_gqa.yaml")

# Convert to split APIs when needed
model_cfg = cfg.to_model_config()
elf_cfg   = cfg.to_elf_config()     # only when model_type="elf"
```

### Architecture fields (Config-specific names)

The field names below are what Config uses; where the name differs from ModelConfig the equivalent is noted.

| Field | Type | Default | Notes |
|:------|:-----|:-------:|:------|
| `dim` | `int` | `512` | |
| `n_heads` | `int` | `16` | |
| `head_size` | `int` | `32` | |
| `num_blocks` | `int` | `20` | |
| `vocab_size` | `int` | `200` | |
| `max_context` | `int` | `512` | |
| `kv_heads` | `int` | `4` | GQA KV heads. |
| `model_type` | `str` | `"autoregressive"` | `"autoregressive"` · `"diffusion"` · `"elf"` |
| `attention_type` | `str` | `"auto"` | `"mha"` · `"gqa"` · `"mla"` · `"auto"` (derived) |
| `norm_type` | `str` | `"layernorm"` | `"layernorm"` · `"rmsnorm"` |
| `dropout_rate` | `float` | `0.15` | Equivalent to `ModelConfig.dropout` |
| `weight_tying` | `bool` | `True` | |
| `use_swiglu` | `bool` | `True` | |
| `activation` | `str` | `"gelu"` | |
| `gradient_checkpointing` | `bool` | `True` | |

### Diffusion (LLaDA) fields

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `diffusion_steps` | `int` | `1000` | Total noise levels `T`. |
| `noise_schedule` | `str` | `"cosine"` | `"cosine"` · `"linear"` · `"sqrt"`. |
| `mask_token_id` | `int` | `4` | Token ID used as the MASK symbol. |
| `num_sampling_steps` | `int` | `50` | Denoising steps at inference (≤ `diffusion_steps`). |
| `time_emb_dim` | `int` | `256` | Time-step embedding dimension for `AdaLayerNorm`. |

### ELF (continuous flow) fields

??? note "ELF-specific fields — expand for details"
    Only relevant when `model_type="elf"`. The shared fields (`dim`, `n_heads`, etc.) are reused from the architecture section above.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `embed_dim` | `int` | `512` | Token embedding / flow-space dimension. |
    | `bottleneck_dim` | `int` | `128` | Bottleneck between embed space and transformer. |
    | `num_time_tokens` | `int` | `4` | Control tokens encoding timestep `t`. |
    | `num_cfg_tokens` | `int` | `4` | Control tokens encoding CFG scale `w`. |
    | `num_mode_tokens` | `int` | `4` | Control tokens encoding denoiser/decode mode. |
    | `denoiser_pmean` | `float` | `-1.5` | Logit-normal time sampling mean (training). |
    | `denoiser_pstd` | `float` | `0.8` | Logit-normal time sampling std. |
    | `denoiser_noise_scale` | `float` | `2.0` | ε corruption scale — denoiser branch. |
    | `decoder_pmean` | `float` | `0.8` | Logit-normal p mean — decoder branch. |
    | `decoder_pstd` | `float` | `0.8` | Logit-normal p std — decoder branch. |
    | `decoder_noise_scale` | `float` | `5.0` | ε corruption scale — decoder branch. |
    | `denoiser_prob` | `float` | `0.8` | Fraction of training steps using the denoiser branch. |
    | `self_cond_prob` | `float` | `0.5` | Probability of using self-conditioning. |
    | `cfg_scale_min` | `float` | `0.5` | Min CFG scale during training. |
    | `cfg_scale_max` | `float` | `5.0` | Max CFG scale during training. |
    | `elf_cfg_scale` | `float` | `1.0` | CFG scale at inference time. |
    | `elf_n_steps` | `int` | `64` | Euler ODE steps at inference time. |
    | `t5_model_name` | `str` | `"t5-base"` | Frozen T5 variant used as embedding oracle. |

### MoE fields (Config)

??? note "MoE fields — expand for details"
    Only relevant when `use_moe=True`.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `use_moe` | `bool` | `False` | Enable Mixture-of-Experts FFN. |
    | `n_experts` | `int` | `4` | Number of experts. |
    | `top_k_mlp` | `int` | `2` | Active experts per token. |
    | `expansion` | `int` | `4` | FFN expansion factor. |
    | `alpha_balance` | `float` | `0.1` | Load-balancing loss coefficient. |

### Attention & position (Config)

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `use_rotary_pos` | `bool` | `True` | RoPE positional encoding. |
| `trainable_pos` | `bool` | `False` | Learned positional embeddings. |
| `absolute_pos` | `bool` | `False` | Sinusoidal absolute PE. |
| `rope_scale_factor` | `float` | `1.0` | RoPE frequency scaling (>1 extends effective context). |
| `use_flash_attention` | `bool` | `False` | Flash Attention kernel. |
| `sliding_window` | `bool` | `False` | Sliding-window attention. |
| `context_window` | `int` | `4` | Window blocks for sliding-window attention. |
| `no_sink` | `bool` | `True` | Disable attention sink. |

??? note "MLA fields (Config) — expand for details"
    Only relevant when `attention_type="mla"`.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `mla` | `bool` | `False` | Use Multi-Latent Attention. Overridden by `attention_type="mla"`. |
    | `inference` | `bool` | `False` | MLA inference mode (absorbed KV projection). |
    | `down_dim_q` | `int` | `256` | Query latent dim. |
    | `down_dim_kv` | `int` | `256` | KV latent dim. |
    | `rope_dim` | `int` | `32` | RoPE subspace dim. Must be ≤ `head_size`. |

### Training (Config)

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `lr` | `float` | `0.005` | Peak learning rate. |
| `batch_size` | `int` | `128` | Micro-batch size per device. |
| `grad_accum` | `int` | `16` | Gradient accumulation steps. |
| `epochs` | `int` | `1000` | Max training epochs. |
| `warmup_steps` | `int` | `420` | Warmup steps. |
| `lr_schedule` | `str` | `"cosine"` | LR schedule: `"cosine"` · `"linear"` · `"constant"` · `"wsd"`. |
| `optimizer` | `str` | `"adamw"` | Optimizer. |
| `grad_clip` | `float` | `1.0` | Gradient clipping. |
| `patience` | `int` | `0` | Early stopping patience. |
| `use_bf16` | `bool` | `False` | bfloat16 precision. |
| `eval_iters` | `int` | `20` | Validation batches per eval. |
| `seed` | `int` | `42` | Random seed. |

### LoRA (Config)

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `use_lora` | `bool` | `False` | Inject LoRA adapters. |
| `lora_rank` | `int` | `8` | LoRA rank `r`. |
| `lora_alpha` | `float` | `16.0` | LoRA scaling factor. |
| `lora_dropout` | `float` | `0.0` | LoRA dropout. |
| `lora_targets` | `str` | `"attention"` | `"attention"` · `"mlp"` · `"all"`. |

### Hardware & logging (Config)

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `n_devices` | `int` | `0` | GPU count (`0` = all). |
| `tp_size` | `int` | `1` | Tensor-parallel factor. |
| `log_file` | `str` | `"training_log.csv"` | CSV log path (relative to run dir). |
| `summary_file` | `str` | `"model_summary.json"` | JSON architecture summary. |

---

## ELFConfig

Dedicated architecture config for `ELFTransformer`. Use this class when instantiating ELF models directly rather than through the `Config` + `Trainer` pipeline.

```python
from core.config import ELFConfig
from core.elf import ELFTransformer
from flax import nnx

cfg   = ELFConfig(
    embed_dim=512, bottleneck_dim=128,
    model_dim=768, n_heads=12, head_size=64,
    num_blocks=12, vocab_size=32128,
)
model = ELFTransformer(cfg, rngs=nnx.Rngs(42))
```

!!! abstract "Key constraint"
    `model_dim` must equal `n_heads × head_size`, just like in `ModelConfig`.

| Field | Type | Default | Description |
|:------|:-----|:-------:|:------------|
| `embed_dim` | `int` | `512` | Token embedding and flow-space dimension. |
| `bottleneck_dim` | `int` | `128` | Bottleneck between embed space and transformer hidden dim. |
| `model_dim` | `int` | `768` | Transformer hidden dim. Must equal `n_heads × head_size`. |
| `n_heads` | `int` | `12` | Attention heads. |
| `head_size` | `int` | `64` | Per-head dimension. |
| `num_blocks` | `int` | `12` | Transformer layers. |
| `vocab_size` | `int` | `32000` | Vocabulary size. |
| `max_seq_len` | `int` | `1024` | Max sequence length (excluding control tokens). |
| `pos_encoding` | `str` | `"rotary"` | Positional encoding. |
| `norm` | `str` | `"rmsnorm"` | Normalisation type. |
| `dropout` | `float` | `0.0` | Dropout rate. |
| `gradient_checkpointing` | `bool` | `True` | Recompute activations in backward pass. |
| `time_emb_dim` | `int` | `256` | Sinusoidal embedding dim for `t` and `w`. |
| `num_time_tokens` | `int` | `4` | Time control tokens. |
| `num_cfg_tokens` | `int` | `4` | CFG scale control tokens. |
| `num_mode_tokens` | `int` | `4` | Mode control tokens. |
| `sde_gamma` | `float` | `1.0` | SDE noise re-injection at inference (`0` = pure ODE). |
| `t5_model_name` | `str` | `"t5-base"` | Frozen T5 embedding oracle. `vocab_size` must match. |

??? note "Training-time ELF fields — expand for details"
    These control the denoiser/decoder dual-branch training procedure.

    | Field | Type | Default | Description |
    |:------|:-----|:-------:|:------------|
    | `denoiser_pmean` | `float` | `-1.5` | Logit-normal time sampling mean. |
    | `denoiser_pstd` | `float` | `0.8` | Logit-normal time sampling std. |
    | `denoiser_noise_scale` | `float` | `2.0` | Noise corruption scale (denoiser branch). |
    | `decoder_pmean` | `float` | `0.8` | Logit-normal p mean (decoder branch). |
    | `decoder_pstd` | `float` | `0.8` | Logit-normal p std (decoder branch). |
    | `decoder_noise_scale` | `float` | `5.0` | Noise corruption scale (decoder branch). |
    | `denoiser_prob` | `float` | `0.8` | Denoiser branch fraction. |
    | `self_cond_prob` | `float` | `0.5` | Self-conditioning probability. |
    | `cfg_scale_min` | `float` | `0.5` | Min CFG training scale. |
    | `cfg_scale_max` | `float` | `5.0` | Max CFG training scale. |

---

## Complete YAML example

```yaml title="configs/medium_gqa.yaml"
# Architecture
dim: 512
n_heads: 8
head_size: 64
num_blocks: 12
vocab_size: 32000
max_context: 1024
attention_type: gqa
kv_heads: 2
use_rotary_pos: true
norm_type: rmsnorm
use_swiglu: true
gradient_checkpointing: true

# Training paradigm
model_type: autoregressive

# Optimiser
lr: 3e-4
batch_size: 64
grad_accum: 4
epochs: 500
warmup_steps: 400
lr_schedule: cosine
optimizer: adamw
grad_clip: 1.0
use_bf16: true

# Dataset
dataset_source: huggingface
dataset_name: wikitext
dataset_config: wikitext-103-raw-v1
dataset_text_field: text
tokenizer_type: bpe
tokenizer_path: t5-base
```

---

## See also

<div class="grid cards" markdown>

-   :material-console: **CLI Reference**

    All 9 CLI subcommands with full argument tables.

    [→ CLI Reference](cli.md)

-   :material-chef-hat: **Cookbook**

    Copy-paste recipes for common patterns.

    [→ Cookbook](cookbook.md)

-   :material-school: **Tutorials**

    Step-by-step guides for training, diffusion, LoRA.

    [→ Tutorials](tutorials/index.md)

</div>
