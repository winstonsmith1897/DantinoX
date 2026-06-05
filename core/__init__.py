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
from .config import Config

# Diffusion utilities
from .diffusion import (
    DualCache,
    NoiseSchedule,
    TimeEmbedding,
    make_noise_schedule,
    corrupt,
    masked_cross_entropy,
    sinusoidal_embedding,
    confidence_unmask_threshold,
    confidence_unmask_factor,
)

# Generation
from .generation import decode, generate, diffusion_generate, fast_dllm_generate

# LoRA
from .lora import LoRALinear, LoRAParam

# Models
from .model import Transformer, DiffusionTransformer

# Output
from .output import ModelOutput

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
    # Diffusion
    "DualCache",
    "NoiseSchedule",
    "TimeEmbedding",
    "make_noise_schedule",
    "corrupt",
    "masked_cross_entropy",
    "sinusoidal_embedding",
    # Generation
    "decode",
    "generate",
    "diffusion_generate",
    "fast_dllm_generate",
    # Diffusion helpers
    "confidence_unmask_threshold",
    "confidence_unmask_factor",
    # LoRA
    "LoRALinear",
    "LoRAParam",
    # Models
    "Transformer",
    "DiffusionTransformer",
    # Output
    "ModelOutput",
    # Sharding
    "make_mesh",
    "num_devices",
    "replicate",
    "shard_batch",
]
