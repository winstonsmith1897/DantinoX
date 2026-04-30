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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("dantinox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from core.config import Config
from core.model import Transformer
from core.generation import generate, decode

from dantinox.exceptions import (
    DantinoXError,
    ConfigError,
    CheckpointError,
    BenchmarkError,
    PlotError,
)
from dantinox.trainer import Trainer
from dantinox.generator import Generator
from dantinox.bench import BenchmarkRunner
from dantinox.plotting import Plotter

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
    # exceptions
    "DantinoXError",
    "ConfigError",
    "CheckpointError",
    "BenchmarkError",
    "PlotError",
]
