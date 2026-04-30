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

from core.config import Config
from core.model import Transformer
from core.generation import generate, decode

from dantinox.trainer import Trainer
from dantinox.generator import Generator
from dantinox.bench import BenchmarkRunner
from dantinox.plotting import Plotter

__all__ = [
    "Config",
    "Transformer",
    "generate",
    "decode",
    "Trainer",
    "Generator",
    "BenchmarkRunner",
    "Plotter",
]

__version__ = "0.1.0"
