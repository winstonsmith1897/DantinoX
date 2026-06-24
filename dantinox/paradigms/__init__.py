from dantinox.paradigms.paradigm import Paradigm
from dantinox.paradigms.embedder import EmbedderParadigm, info_nce_loss

# Internal implementations — importable for advanced use / backward compat
from dantinox.paradigms.ar import ARParadigm
from dantinox.paradigms.diffusion import DiscreteParadigm, ContinuousParadigm

__all__ = [
    "Paradigm",
    "EmbedderParadigm",
    "info_nce_loss",
    # Advanced / backward compat
    "ARParadigm",
    "DiscreteParadigm",
    "ContinuousParadigm",
]
