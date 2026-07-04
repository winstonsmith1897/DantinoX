"""Deprecated module — renamed to :mod:`dantinox.core.flow`.

The continuous flow-matching paradigm is implemented in ``core/flow.py``;
ELF (Hu et al., 2026) is the recipe it currently follows, not the paradigm
itself, so the module and its public names were de-branded.  This shim keeps
``from dantinox.core.elf import ...`` working and will be removed in v1.0.
"""

from __future__ import annotations

import warnings

from .flow import (  # noqa: F401
    ELFEmbedder,
    ELFNet,
    ELFTransformer,
    FlowEmbedder,
    FlowMatchingTransformer,
    Frozen,
    elf_ce_loss,
    elf_decoder_loss,
    elf_denoiser_loss,
    elf_loss,
    elf_mse_loss,
    flow_ce_loss,
    flow_decoder_loss,
    flow_denoiser_loss,
    flow_loss,
    flow_mse_loss,
)

warnings.warn(
    "dantinox.core.elf is deprecated; import from dantinox.core.flow instead "
    "(FlowMatchingTransformer, flow_loss, ...). Removed in v1.0.",
    DeprecationWarning,
    stacklevel=2,
)
