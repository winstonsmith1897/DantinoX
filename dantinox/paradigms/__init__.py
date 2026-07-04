# Internal implementations — importable for advanced use / backward compat
from dantinox.paradigms.ar import ARParadigm
from dantinox.paradigms.diffusion import ContinuousParadigm, DiscreteParadigm
from dantinox.paradigms.embedder import EmbedderParadigm, info_nce_loss
from dantinox.paradigms.paradigm import Paradigm

__all__ = [
    "Paradigm",
    "EmbedderParadigm",
    "info_nce_loss",
    # Advanced / backward compat
    "ARParadigm",
    "DiscreteParadigm",
    "ContinuousParadigm",
]
