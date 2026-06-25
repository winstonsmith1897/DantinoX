---
title: Notebooks
hide:
  - toc
---

# Interactive Notebooks

All notebooks are self-contained and runnable on **Google Colab** (free GPU tier).
Each cell installs DantinoX automatically — no local setup required.

## At a glance

| # | Notebook | Topic | Est. time (T4) |
|---|----------|-------|:--------------:|
| 01 | [Quickstart](#01--quickstart) | AR model, attention/FFN/norm variants, Generator | ~10 min |
| 02 | [Discrete Diffusion](#02--discrete-diffusion-llada) | LLaDA masked diffusion, block-wise generation, DualCache | ~20 min |
| 03 | [ELF Flow-Matching](#03--elf-continuous-flow-matching) | Rectified flow in T5 embedding space, CFG guidance | ~25 min |
| 04 | [Benchmarking & Profiling](#04--benchmarking--profiling) | FLOPs, latency, BenchmarkSuite, Visualizer | ~15 min |
| 05 | [LoRA Fine-Tuning](#05--lora-fine-tuning) | Adapters, rank ablation, merge, domain adaptation | ~20 min |
| 06 | [Paradigm Profiling](#06--paradigm-profiling) | AR vs Discrete vs ELF — 2D + interactive 3D Plotly | 10–40 min |
| 07 | [Diffusion Cache Profiling](#07--diffusion-cache-profiling) | No Cache vs Prefix Cache vs Dual Cache (Fast-dLLM) | 5–25 min |
| 08 | [Retrievers & Embedder Training](#08--retrievers--embedder-training) | SimCSE, contrastive fine-tuning, FAISS, LangChain, ChromaDB | ~25 min |

---

<div class="grid cards" markdown>

-   :material-rocket-launch: **01 — Quickstart**

    ---

    From zero to a trained AR model in under 10 minutes. Covers the Level-1 one-liner API (`dx.fit`, `dx.quick_generate`), Level-2 explicit `Paradigm`, attention/FFN/norm/positional-encoding variants, and `Generator` decoding strategies (greedy · top-k · nucleus · streaming).

    **~10 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/01_quickstart.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/01_quickstart.ipynb){ .md-button }

-   :material-blur: **02 — Discrete Diffusion (LLaDA)**

    ---

    Train a masked-diffusion LM end-to-end. Covers `Paradigm(ModelConfig(paradigm="discrete"))`, (1/t)-weighted loss, iterative unmasking with four decoding strategies, noise schedule comparison (`linear` · `cosine` · `sqrt`), and block-wise generation with optional DualCache.

    **~20 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/02_discrete_diffusion.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/02_discrete_diffusion.ipynb){ .md-button }

-   :material-wave: **03 — ELF Continuous Flow-Matching**

    ---

    Train an ELF model with rectified flow in T5 embedding space. Covers `ELFTransformer`, logit-normal time schedule, Euler ODE generation, CFG guidance scale ablation (`w = 1.0 … 5.0`), and model-size sweep (`embed_dim=768` fixed, `dim` and `num_blocks` free).

    **~25 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/03_elf_flow_matching.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/03_elf_flow_matching.ipynb){ .md-button }

-   :material-speedometer: **04 — Benchmarking & Profiling**

    ---

    Measure FLOPs analytically with `count_flops`, wall-clock latency with `LatencyTracker`, and run a full `BenchmarkSuite` sweep over sequence lengths and batch sizes. Visualise results with `Visualizer`.

    **~15 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/04_benchmarking.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/04_benchmarking.ipynb){ .md-button }

-   :material-tune: **05 — LoRA Fine-Tuning**

    ---

    Fine-tune a pretrained DantinoX checkpoint with Low-Rank Adaptation. Covers `use_lora=True`, adapter initialisation, `lora_targets` (`attention` · `ffn` · `all`), rank ablation, and `merge_lora()` for zero-overhead inference. Demonstrates domain shift (Shakespeare → Bible).

    **~20 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/05_lora_fine_tuning.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/05_lora_fine_tuning.ipynb){ .md-button }

-   :material-chart-bar: **06 — Paradigm Profiling**

    ---

    Profile AR, Discrete Diffusion, and ELF side-by-side across scale, batch, dtype, and diffusion-step sweeps. Produces five 2D matplotlib benchmark figures and six interactive 3D Plotly surfaces. Exposes a `QUICK` flag for fast (~10 min) vs. full (~40 min) runs.

    **10–40 min &nbsp;·&nbsp; GPU (T4 / A100)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/06_paradigm_profiling.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/06_paradigm_profiling.ipynb){ .md-button }

-   :material-cached: **07 — Diffusion Cache Profiling**

    ---

    Compare **No Cache**, **Prefix Cache**, and **Dual Cache** (Fast-dLLM) across all `dantinox.profiling` metrics: latency (mean/p50/p95/p99), throughput, FLOPs, energy (NVML), perplexity, and entropy. Proves caching is lossless. Exposes a `QUICK` flag.

    **5–25 min &nbsp;·&nbsp; GPU (T4 / A100)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/07_diffusion_cache_profiling.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/07_diffusion_cache_profiling.ipynb){ .md-button }

-   :material-database-search: **08 — Retrievers & Embedder Training**

    ---

    Train dense retrievers with `EmbedderParadigm` using **SimCSE** (unsupervised, dropout-based) and **supervised contrastive fine-tuning** (`dx.EmbedderTrainer`). Covers `dx.Embedder.from_run()`, FAISS vector search, LangChain and ChromaDB drop-in integrations, and injecting pretrained AR/Discrete weights into an embedder head.

    **~25 min &nbsp;·&nbsp; GPU (T4)**

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/08_retrievers_training.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/08_retrievers_training.ipynb){ .md-button }

</div>

---

## Requirements

All notebooks install DantinoX from GitHub in the first cell:

```bash
pip install "dantinox[all] @ git+https://github.com/winstonsmith1897/DantinoX.git"
```

A **free Colab T4** is sufficient for all notebooks.
Notebooks 06 and 07 expose a `QUICK` flag — set `QUICK = False` for larger sweeps that benefit from an A100.

---

## Running locally

```bash
pip install "dantinox[all] @ git+https://github.com/winstonsmith1897/DantinoX.git" notebook
jupyter notebook docs/notebooks/
```

Or with JupyterLab:

```bash
pip install jupyterlab
jupyter lab docs/notebooks/
```
