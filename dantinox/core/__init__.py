"""``dantinox.core`` — the model substrate shared by all paradigms.

Map of the package (one responsibility per module):

======================  ========================================================
Module                  Responsibility
======================  ========================================================
``config.py``           All serialisable configuration (``ModelConfig``,
                        ``TrainingConfig``, ``FlowMatchingConfig``, legacy
                        ``Config`` + lossless bridging between them).
``attention.py``        Attention variants (MHA / GQA / MLA), typed KV-cache
                        containers, RoPE and masking plumbing.
``mlp.py``              Dense feed-forward network (SwiGLU / GELU).
``moe.py``              Sparse Mixture-of-Experts feed-forward network.
``block.py``            Pre-norm transformer block (attention + FFN) and
                        normalisation layers.
``model.py``            The unified ``Transformer`` backbone (AR, masked
                        diffusion, and embedding via one ``causal`` flag).
``flow.py``             Continuous flow-matching paradigm (ELF recipe): model,
                        forward process, schedules, and losses.
``diffusion.py``        Discrete masked diffusion (LLaDA recipe): forward
                        process, noise schedules, loss, unmasking strategies,
                        and the Fast-dLLM ``DualCache``.
``generation.py``       Decoding loops for all three paradigms (eager wrappers
                        drain their streaming twins).
``lora.py``             LoRA adapters and weight merging.
``output.py``           NamedTuple return types of the model forward passes.
``sharding.py``         Device meshes and data/tensor-parallel helpers.
``checkpoint.py``       Run-directory loading: config detection, weight-file
                        resolution, msgpack decoding, model restoration.
``pipeline.py``         One-call text-generation inference from a checkpoint.
``export.py``           StableHLO export of trained models.
``elf.py``              Deprecated shim → ``flow.py`` (removed in v1.0).
======================  ========================================================
"""

# Attention
from .attention import (
    BaseAttention,
    GQAAttention,
    KVCache,
    MHAAttention,
    MLAAttention,
    MLACache,
    build_attention,
)

# Blocks
from .block import (
    AdaLayerNorm,
    ARBlock,
    Block,  # backward-compat alias for ARBlock
    DiffusionBlock,
    RMSNorm,
    build_block,
)

# Checkpoint loading
from .checkpoint import load_config, load_model, restore_model

# Config
from .config import (
    Config,
    ELFConfig,  # deprecated alias for FlowMatchingConfig
    FlowMatchingConfig,
    ModelConfig,
    TrainingConfig,
)

# Discrete diffusion utilities (LLaDA-style masked diffusion)
from .diffusion import (
    DualCache,
    NoiseSchedule,
    confidence_unmask_factor,
    confidence_unmask_threshold,
    corrupt,
    make_noise_schedule,
    masked_cross_entropy,
)

# Continuous flow-matching (ELF recipe)
from .flow import (
    ELFNet,  # deprecated alias for FlowMatchingTransformer
    ELFTransformer,  # deprecated alias for FlowMatchingTransformer
    FlowEmbedder,
    FlowMatchingTransformer,
    corrupt_decoder,
    corrupt_denoiser,
    flow_ce_loss,
    flow_decoder_loss,
    flow_denoiser_loss,
    flow_loss,
    flow_mse_loss,
    logit_normal_schedule,
    sample_cfg_scale,
    sample_p_per_token,
    sample_t_logit_normal,
)

# Generation
from .generation import (
    decode,
    diffusion_generate,
    elf_generate,  # deprecated alias for flow_generate
    fast_dllm_generate,
    flow_generate,
    generate,
    stream_flow_generate,
)

# LoRA
from .lora import LoRALinear, LoRAParam

# Models
from .model import DiffusionTransformer, Transformer

# Output
from .output import ELFOutput, EmbeddingOutput, FlowMatchingOutput, ModelOutput

# Sharding
from .sharding import in_mesh_context, make_mesh, num_devices, replicate, shard_batch

__all__ = [
    # Attention building blocks
    "BaseAttention",
    "MHAAttention",
    "GQAAttention",
    "MLAAttention",
    "build_attention",
    "KVCache",
    "MLACache",
    # Transformer blocks
    "RMSNorm",
    "AdaLayerNorm",
    "ARBlock",
    "DiffusionBlock",
    "build_block",
    "Block",
    # Checkpoint loading
    "load_config",
    "load_model",
    "restore_model",
    # Config
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "FlowMatchingConfig",
    "ELFConfig",
    # Continuous flow-matching model
    "FlowEmbedder",
    "FlowMatchingTransformer",
    "ELFTransformer",
    "ELFNet",
    # Generation
    "decode",
    "generate",
    "diffusion_generate",
    "fast_dllm_generate",
    "flow_generate",
    "stream_flow_generate",
    "elf_generate",
    # Discrete diffusion utilities
    "DualCache",
    "NoiseSchedule",
    "make_noise_schedule",
    "corrupt",
    "masked_cross_entropy",
    "confidence_unmask_threshold",
    "confidence_unmask_factor",
    # Flow-matching utilities and losses
    "corrupt_denoiser",
    "corrupt_decoder",
    "sample_t_logit_normal",
    "sample_p_per_token",
    "sample_cfg_scale",
    "logit_normal_schedule",
    "flow_loss",
    "flow_denoiser_loss",
    "flow_decoder_loss",
    "flow_mse_loss",
    "flow_ce_loss",
    # LoRA
    "LoRALinear",
    "LoRAParam",
    # Models
    "Transformer",
    "DiffusionTransformer",
    # Output types
    "ModelOutput",
    "FlowMatchingOutput",
    "ELFOutput",
    "EmbeddingOutput",
    # Sharding
    "in_mesh_context",
    "make_mesh",
    "num_devices",
    "replicate",
    "shard_batch",
]
