DantinoX
========

A research-grade JAX/Flax NNX library for language model research.
Three generation paradigms — Autoregressive, Masked Diffusion, and ELF —
on the same Transformer architecture, with a single trainer and zero boilerplate.

.. image:: https://img.shields.io/badge/JAX-000000?style=flat-square&logo=google&logoColor=white
   :target: https://github.com/google/jax
   :alt: JAX

.. image:: https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white
   :target: https://github.com/google/flax
   :alt: Flax NNX

.. image:: https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white
   :target: https://www.python.org/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/License-MIT-16a34a?style=flat-square
   :target: https://opensource.org/licenses/MIT
   :alt: License MIT

.. image:: https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square
   :target: https://dantinox.readthedocs.io
   :alt: Docs

----

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   quickstart
   cookbook
   notebooks/index
   vs-transformers

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Models

   architecture
   architecture/core
   architecture/paradigm-system
   paradigms/index
   paradigms/autoregressive
   paradigms/diffusion
   paradigms/elf
   paradigms/fast-dllm
   paradigms/confidence
   paradigms/comparison

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Training

   training/index
   training/autoregressive
   training/diffusion
   training/optimizers
   training/sweeps
   training/multi-gpu
   training/emnlp-suite

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Inference

   inference/index
   inference/autoregressive
   inference/diffusion
   inference/kv-cache

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/index
   tutorials/first-model
   tutorials/diffusion-lm
   tutorials/lora-fine-tuning
   tutorials/benchmarking
   tutorials/hub

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Benchmarks

   benchmarks
   ablation_studies
   paper
   architecture/profiling

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index
   api/dantinox
   api/paradigms
   api/training
   api/profiling
   api/benchmarking
   api/visualization

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   guides/index
   guides/new-layer
   guides/new-paradigm
   guides/new-benchmark
   guides/new-chart

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   configuration
   cli
   contributing
   faq
   changelog


Overview
--------

DantinoX was created to answer a single question: **how do different
generation paradigms — autoregressive, masked diffusion, and flow-matching —
compare when trained on the same architecture with the same training code?**

The library targets three audiences:

* **Researchers** who want a reproducible comparison of AR vs. Diffusion vs. ELF.
* **Students** who want to read the internals of a modern Transformer.
* **Engineers** who need architectural variants (GQA, MLA, MoE, LoRA) without
  rewriting the trainer.


Three Generation Paradigms
--------------------------

**Autoregressive (AR)**
    The classical left-to-right paradigm. Generates one token at a time using
    a causal (masked) attention pattern and a static pre-allocated KV-cache.
    See :doc:`paradigms/autoregressive`.

**Masked Diffusion (LLaDA)**
    Generates all tokens in parallel from a fully masked sequence and
    iteratively unmasks them over multiple diffusion steps. Attention is
    bidirectional. Optionally accelerated by Fast-dLLM DualCache.
    See :doc:`paradigms/diffusion`.

**ELF — Continuous Flow Matching**
    Operates in the continuous embedding space. Transforms Gaussian noise into
    clean token embeddings via an Euler ODE solver.
    See :doc:`paradigms/elf`.


Quick Install
-------------

.. code-block:: bash

   pip install dantinox          # CPU / GPU (JAX auto-detected)

Or from source::

   git clone https://github.com/winstonsmith1897/DantinoX
   cd DantinoX && pip install -e ".[dev]"


One-Liner Usage
---------------

.. code-block:: python

   import dantinox as dx

   run_dir = dx.fit("ar", "data/wiki.txt",
                    dim=512, n_heads=8, head_size=64,
                    num_blocks=12)

   print(dx.quick_generate(run_dir, "In the beginning"))


Citation
--------

.. code-block:: bibtex

   @software{dantinox2026,
     author  = {Simoni, Marco},
     title   = {DantinoX: A Unified {JAX}/Flax Framework for {AR},
                Masked Diffusion, and Flow-Matching Language Models},
     year    = {2026},
     url     = {https://github.com/winstonsmith1897/DantinoX},
   }
