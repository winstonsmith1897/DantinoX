# `dantinox.paradigms`

Paradigms define the training objective and generation strategy. The `Trainer` only ever calls `loss_fn` — all paradigm-specific logic is self-contained.

---

## Base

::: dantinox.paradigms.base.Paradigm
    options:
      show_source: true
      members:
        - build_model
        - loss_fn
        - generate
        - num_parameters

---

## Autoregressive

::: dantinox.paradigms.ar.ARParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

## Discrete Diffusion (LLaDA)

::: dantinox.paradigms.diffusion.discrete.DiscreteConfig
    options:
      show_source: true

::: dantinox.paradigms.diffusion.discrete.DiscreteParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

## Continuous Flow-Matching (ELF)

::: dantinox.paradigms.diffusion.continuous.ContinuousParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - build_embedder
        - loss_fn
        - generate
        - num_parameters
