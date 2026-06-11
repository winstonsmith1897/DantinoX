"""Deprecated top-level ``core`` package — moved to ``dantinox.core``.

This shim keeps ``from core.config import ModelConfig`` and friends working
for one release cycle. It will be removed in DantinoX 0.5.0; switch to::

    from dantinox.core.config import ModelConfig
"""

import importlib
import sys
import warnings

warnings.warn(
    "The top-level 'core' package is deprecated and will be removed in "
    "DantinoX 0.5.0 — import from 'dantinox.core' instead "
    "(e.g. 'from dantinox.core.config import ModelConfig').",
    DeprecationWarning,
    stacklevel=2,
)

# Alias every submodule so `import core.config` resolves to the *same* module
# object as `dantinox.core.config` (avoids duplicate dataclass definitions).
_SUBMODULES = (
    "attention", "block", "config", "diffusion", "elf", "generation",
    "lora", "mlp", "model", "moe", "output", "sharding",
)
for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"dantinox.core.{_name}"
    )

from dantinox.core import *          # noqa: F401,F403,E402
from dantinox.core import __all__    # noqa: F401,E402
