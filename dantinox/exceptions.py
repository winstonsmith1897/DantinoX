"""
DantinoX exception hierarchy.

Catch ``DantinoXError`` to handle any library error; use the sub-classes for
finer-grained control.

    DantinoXError
    ├── ConfigError        — invalid or inconsistent Config
    ├── CheckpointError    — missing or corrupt checkpoint files
    ├── BenchmarkError     — failure during benchmarking
    └── PlotError          — failure during plot generation
"""

from __future__ import annotations


class DantinoXError(Exception):
    """Base class for all DantinoX exceptions."""


class ConfigError(DantinoXError):
    """Raised when a :class:`~core.config.Config` is invalid or inconsistent."""


class CheckpointError(DantinoXError):
    """Raised when a run directory, config file, or weights file cannot be loaded."""


class BenchmarkError(DantinoXError):
    """Raised when benchmarking a run fails."""


class PlotError(DantinoXError):
    """Raised when plot generation fails."""
