---
title: Notebooks
hide:
  - toc
---

# Interactive Notebooks

All notebooks are self-contained and runnable on **Google Colab** (free GPU tier). Each cell installs DantinoX automatically — no local setup required.

<div class="grid cards" markdown>

-   :material-rocket-launch: **01 — Quickstart**

    ---

    From zero to a trained AR model in under 10 minutes. Covers the Level-1 one-liner API, Level-2 explicit `Paradigm`, and a FLOPs estimate.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/01_quickstart.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/01_quickstart.ipynb){ .md-button }

-   :material-blur: **02 — Discrete Diffusion (LLaDA)**

    ---

    Train a masked-diffusion LM end-to-end. Covers `DiscreteParadigm`, (1/t)-weighted loss, iterative unmasking generation, and noise schedule comparison.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/02_discrete_diffusion.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/02_discrete_diffusion.ipynb){ .md-button }

-   :material-wave: **03 — ELF Continuous Flow-Matching**

    ---

    Train an ELF model with rectified flow in continuous embedding space. Covers `ELFTransformer`, logit-normal time schedule, Euler ODE generation, and CFG guidance.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/03_elf_flow_matching.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/03_elf_flow_matching.ipynb){ .md-button }

-   :material-speedometer: **04 — Benchmarking & Profiling**

    ---

    Measure FLOPs analytically with `count_flops`, wall-clock latency with `LatencyTracker`, and run a full `BenchmarkSuite`. Visualise results with `Visualizer`.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/04_benchmarking.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/04_benchmarking.ipynb){ .md-button }

-   :material-tune: **05 — LoRA Fine-Tuning**

    ---

    Fine-tune a pretrained DantinoX checkpoint with Low-Rank Adaptation. Covers `use_lora` flag, adapter initialisation, selective parameter freezing, and checkpoint merging.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/winstonsmith1897/DantinoX/blob/main/docs/notebooks/05_lora_fine_tuning.ipynb){ .md-button .md-button--primary }
    [:fontawesome-brands-github: View on GitHub](https://github.com/winstonsmith1897/DantinoX/blob/main/docs/notebooks/05_lora_fine_tuning.ipynb){ .md-button }

</div>

---

## What you need

All notebooks install DantinoX from GitHub in the first cell:

```bash
!pip install -q git+https://github.com/winstonsmith1897/DantinoX.git#egg=dantinox[all]
```

A **free Colab GPU** (T4) is sufficient for all notebooks. For notebooks 03 and 05, a GPU is recommended but not strictly required.

---

## Running locally

```bash
pip install dantinox[all] notebook
jupyter notebook docs/notebooks/
```

Or with JupyterLab:

```bash
pip install jupyterlab
jupyter lab docs/notebooks/
```
