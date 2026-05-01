"""
DantinoX — a decoder-only Transformer library built in JAX and Flax NNX.

Quick start
-----------
>>> from dantinox import Config, Transformer, Trainer, Generator
>>>
>>> config = Config.from_yaml("configs/default_config.yaml")
>>> trainer = Trainer(config)
>>> run_dir = trainer.fit("data/corpus.txt")
>>>
>>> gen = Generator(run_dir)
>>> print(gen.generate("Nel mezzo del cammin "))
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("dantinox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from core.config import Config
from core.generation import decode, generate
from core.model import Transformer
from dantinox.bench import BenchmarkRunner
from dantinox.exceptions import (
    BenchmarkError,
    CheckpointError,
    ConfigError,
    DantinoXError,
    PlotError,
)
from dantinox.generator import Generator
from dantinox.hub import pull, push, resolve_checkpoint
from dantinox.plotting import Plotter
from dantinox.trainer import Trainer

__all__ = [
    # version
    "__version__",
    # core
    "Config",
    "Transformer",
    "generate",
    "decode",
    # high-level API
    "Trainer",
    "Generator",
    "BenchmarkRunner",
    "Plotter",
    "push",
    "pull",
    "resolve_checkpoint",
    # exceptions
    "DantinoXError",
    "ConfigError",
    "CheckpointError",
    "BenchmarkError",
    "PlotError",
]
