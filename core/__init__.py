# Attention
from .attention import (
    BaseAttention,
    MHAAttention,
    GQAAttention,
    MLAAttention,
    build_attention,
)

# Blocks
from .block import (
    RMSNorm,
    AdaLayerNorm,
    ARBlock,
    DiffusionBlock,
    build_block,
    Block,          # backward-compat alias for ARBlock
)

# Config
from .config import Config, ModelConfig, ELFConfig

# Discrete diffusion utilities (LLaDA-style masked diffusion)
from .diffusion import (
    DualCache,
    NoiseSchedule,
    make_noise_schedule,
    corrupt,
    masked_cross_entropy,
    confidence_unmask_threshold,
    confidence_unmask_factor,
    # Continuous flow-matching utilities (ELF)
    sample_t_logit_normal,
    sample_p_per_token,
    sample_cfg_scale,
    corrupt_denoiser,
    corrupt_decoder,
    logit_normal_schedule,
)

# ELF: Embedded Language Flows (continuous flow-matching diffusion)
from .elf import (
    ELFEmbedder,
    ELFTransformer,
    ELFNet,          # backward-compat alias for ELFTransformer
    elf_mse_loss,
    elf_ce_loss,
    elf_denoiser_loss,
    elf_decoder_loss,
    elf_loss,
)

# Generation
from .generation import (
    decode,
    generate,
    diffusion_generate,
    fast_dllm_generate,
    elf_generate,
)

# LoRA
from .lora import LoRALinear, LoRAParam

# Models
from .model import Transformer, DiffusionTransformer

# Output
from .output import ModelOutput, ELFOutput

# Sharding
from .sharding import make_mesh, num_devices, replicate, shard_batch

__all__ = [
    # Attention
    "BaseAttention",
    "MHAAttention",
    "GQAAttention",
    "MLAAttention",
    "build_attention",
    # Blocks
    "RMSNorm",
    "AdaLayerNorm",
    "ARBlock",
    "DiffusionBlock",
    "build_block",
    "Block",
    # Config
    "Config",
    "ModelConfig",
    "ELFConfig",
    # Discrete diffusion
    "DualCache",
    "NoiseSchedule",
    "make_noise_schedule",
    "corrupt",
    "masked_cross_entropy",
    "confidence_unmask_threshold",
    "confidence_unmask_factor",
    # Continuous flow-matching (ELF)
    "sample_t_logit_normal",
    "sample_p_per_token",
    "sample_cfg_scale",
    "corrupt_denoiser",
    "corrupt_decoder",
    "logit_normal_schedule",
    # ELF model
    "ELFEmbedder",
    "ELFTransformer",
    "ELFNet",
    "elf_mse_loss",
    "elf_ce_loss",
    "elf_denoiser_loss",
    "elf_decoder_loss",
    "elf_loss",
    # Generation
    "decode",
    "generate",
    "diffusion_generate",
    "fast_dllm_generate",
    "elf_generate",
    # LoRA
    "LoRALinear",
    "LoRAParam",
    # Models
    "Transformer",
    "DiffusionTransformer",
    # Output
    "ModelOutput",
    "ELFOutput",
    # Sharding
    "make_mesh",
    "num_devices",
    "replicate",
    "shard_batch",
]
