---
title: Continuous Flow-Matching Generation
---

# Continuous Flow-Matching Generation

The continuous flow-matching paradigm (`FlowMatchingTransformer`, following the
ELF recipe — Hu et al., 2026) generates by integrating an Euler ODE from pure
Gaussian noise in a frozen T5 encoder's embedding space, then decoding to
tokens only at the final step. Unlike AR or discrete diffusion, every position
evolves simultaneously and continuously across all `n_steps` — there is no
discrete `[MASK]` token and no KV-cache.

---

## High-level API — `Generator.stream()`

The paradigm-agnostic entry point auto-dispatches to the flow-matching
sampler when the checkpoint's `ModelConfig.paradigm == "continuous"`:

```python
from dantinox.generator import Generator

gen = Generator("runs/continuous_mha_512d_12b", seed=42)

for chunk in gen.stream(max_new_tokens=128, n_steps=64):
    print(chunk, end="", flush=True)
```

Each yielded chunk is a full in-place rewrite of the current decoded
sequence (`\r[step/total] <current text>`), not a single new token — every
position is being refined simultaneously at every step, so there's no
"next token" to append the way there is for AR or block-wise diffusion.

`n_steps` defaults to `self.config.flow_n_steps` (`32` by default) when
omitted. The Classifier-Free Guidance scale and SDE noise-reinjection amount
are **not** `stream()` keyword arguments — they come from the checkpoint's
`ModelConfig.flow_cfg_scale` / `ModelConfig.sde_gamma` fields. To override
them at inference time, either edit those fields before saving the config,
or drop down to the lower-level `flow_generate` API below.

---

## Lower-level API — `flow_generate` / `stream_flow_generate`

For direct control over `cfg_scale` and `gamma` per call:

```python
from dantinox.core.generation import flow_generate

tokens = flow_generate(
    model,
    gen_len    = 128,
    batch_size = 1,
    n_steps    = 64,      # Euler ODE steps before the final decode step
    cfg_scale  = 1.5,     # Classifier-Free Guidance scale w (>= 1.0)
    gamma      = 0.0,     # 0.0 = deterministic ODE; > 0.0 = SDE-style noise re-injection
    seed       = 42,
)
# tokens: [batch_size, gen_len] int32
```

The streaming variant yields `(step, total_steps, tokens)` after every ODE
step, where `tokens` is the current argmax-decoded prediction (useful for
visualising how the sequence sharpens over the course of denoising):

```python
from dantinox.core.generation import stream_flow_generate

for step, total, tokens in stream_flow_generate(
    model, gen_len=128, n_steps=64, cfg_scale=1.5, seed=42,
):
    print(f"[{step + 1}/{total}]", tokenizer.decode(tokens[0].tolist()))
```

`total = n_steps + 1`: the last yield is a dedicated `t=1` decode step, not
an ODE integration step.

:::{admonition} Deprecated ELF-branded aliases
:class: note

`elf_generate` and `stream_elf_generate` are deprecated aliases of
`flow_generate` and `stream_flow_generate` — they still work but will be
removed in v1.0.
:::

---

## Euler ODE vs. SDE sampling

```
z(0) ~ N(0, I)                        # pure Gaussian noise
for i in 0 .. n_steps-1:
    x_hat = model(z, t=t_i, w=cfg_scale)   # predicted clean embedding
    v     = (x_hat - z) / (1 - t_i)        # velocity field
    z     = z + dt * v                     # Euler step
x = model(z, t=1, w=cfg_scale)         # final decode step -> tokens
```

Setting `gamma > 0.0` mixes in fresh Gaussian noise at each step
(`z_back = alpha*z + (1-alpha)*noise`, with `alpha = 1 - gamma*dt`) before
computing the velocity field — an SDE-style stochastic sampler that trades
determinism for potentially better sample diversity. `gamma=0.0` (the
default) is the plain deterministic ODE sampler.

---

## Classifier-Free Guidance

`cfg_scale` (`w`) interpolates between the conditional and unconditional
velocity prediction inside the model's forward pass; `w = 1.0` disables
guidance. Higher values push generations more strongly toward the training
distribution at some cost to diversity — the paper's ablation sweeps
`w` from 1.0 to 5.0 (see [Notebooks](../notebooks/index.md)).

---

## What's not yet supported

Prefix/conditional generation (continuing a given prompt rather than
generating unconditionally) is **not yet exposed** for this paradigm — the
ELF formulation supports it in principle, but the current DantinoX
implementation only generates unconditionally from pure noise. This is why
conditional-generation metrics (BLEU-4<sub>cond</sub>) are not reported for
Flow-Matching in the paper's evaluation (Table 2) — see the paper's
Limitations for details.

---

## Decoding to text

Same as the other paradigms — use the tokenizer saved alongside the
checkpoint:

```python
from dantinox.utils.tokenizer import load_tokenizer_from_file

tokenizer = load_tokenizer_from_file("runs/continuous_mha_512d_12b/tokenizer.json")
text = tokenizer.decode(tokens[0].tolist())
print(text)
```

---

## See also

- [Paradigm System](../architecture/paradigm-system.md#continuous-continuous-flow-matching) — `ContinuousParadigm`, `build_embedder`, training-time loss
- [Configuration Reference](../configuration.md#flowmatchingconfig) — `flow_n_steps`, `flow_cfg_scale`, `sde_gamma`, and all other flow-matching fields
- [AR Generation](autoregressive.md) · [Diffusion Generation](diffusion.md) — the other two paradigms
