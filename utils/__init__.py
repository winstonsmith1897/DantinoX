"""Deprecated top-level ``utils`` package — moved to ``dantinox.utils``.

This shim keeps ``from utils.tokenizer import get_tokenizer`` and friends
working for one release cycle. It will be removed in DantinoX 0.5.0; switch
to::

    from dantinox.utils.tokenizer import get_tokenizer
"""

import importlib
import sys
import warnings

warnings.warn(
    "The top-level 'utils' package is deprecated and will be removed in "
    "DantinoX 0.5.0 — import from 'dantinox.utils' instead "
    "(e.g. 'from dantinox.utils.tokenizer import get_tokenizer').",
    DeprecationWarning,
    stacklevel=2,
)

_SUBMODULES = ("helpers", "t5_encoder", "tokenizer")
for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"dantinox.utils.{_name}"
    )
