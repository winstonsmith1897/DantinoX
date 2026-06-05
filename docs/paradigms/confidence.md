---
title: Confidence-Aware Decoding
---

# Confidence-Aware Parallel Decoding

Standard masked diffusion unmasks a fixed number of tokens per step, leading
to wasted computation when the model is very confident about some positions
and uncertain about others.  DantinoX implements the two **confidence-aware**
strategies from Fast-dLLM §3.3 that adapt the number of unmasked tokens to
the model's actual confidence.

Both strategies guarantee **forward progress**: at least one token is always
unmasked per step.

---

## Strategy 1 — Threshold (`"threshold"`)

Unmask all masked positions whose max-softmax confidence exceeds $\tau$:

$$
\text{unmask}_i = \mathbb{1}\!\left[\max_v p_\theta(v \mid x_t, t)_i \geq \tau\right]
$$

If no position meets the threshold, the most confident masked position is
unconditionally revealed (progress guarantee, Algorithm 1 line 9).

```python
from core.diffusion import confidence_unmask_threshold

x_new = confidence_unmask_threshold(
    logits,
    x_t,
    mask_token_id = 0,
    threshold     = 0.9,   # τ
)
```

### Choosing $\tau$

| $\tau$ | Avg tokens/step | Quality | Recommended for |
|---|---|---|---|
| 0.50 | very high | low | speed benchmarks |
| 0.70 | high | medium | fast generation |
| **0.90** | medium | **good** | **default** |
| 0.95 | low | high | quality-critical |
| 0.99 | very low | best | near-sequential |

---

## Strategy 2 — Factor (`"factor"`)

Find the largest $n$ such that revealing the top-$n$ confident tokens
satisfies the theoretical bound from Theorem 1:

$$
(n+1)(1 - c_{(n)}) < f
$$

where $c_{(n)}$ is the $n$-th highest confidence among masked positions.
This bound ensures greedy parallel decoding is equivalent to sequential
decoding up to the $f$-factor slack.

```python
from core.diffusion import confidence_unmask_factor

x_new = confidence_unmask_factor(
    logits,
    x_t,
    mask_token_id = 0,
    factor        = 1.5,   # f
)
```

### Choosing $f$

| $f$ | Behaviour | Avg tokens/step |
|---|---|---|
| 0.8 | conservative, near-sequential | ~1.2 |
| 1.0 | balanced | ~2.5 |
| **1.5** | **recommended** | **~4** |
| 2.0 | aggressive | ~7 |
| 5.0 | very aggressive | ~15 |

The factor strategy gives ~1.4–1.5× higher throughput than threshold at
minor accuracy cost (see [Confidence Sweep benchmark](../benchmarks.md)).

---

## Comparison

| | Threshold | Factor |
|---|---|---|
| Parameter | $\tau \in (0, 1)$ | $f > 0$ |
| Theoretical guarantee | — | ✓ Theorem 1 |
| Tuning difficulty | Low — linear effect | Medium — non-linear |
| Typical speedup vs sequential | 3–8× | 5–12× |
| Quality degradation | Minimal ($\tau \geq 0.9$) | Minimal ($f \leq 2$) |

---

## Using in `fast_dllm_generate`

```python
from core.generation import fast_dllm_generate

# Threshold strategy (default)
tokens = fast_dllm_generate(
    model, prefix, gen_len=256, schedule=schedule, mask_token_id=0,
    decoding_strategy    = "threshold",
    confidence_threshold = 0.9,
)

# Factor strategy
tokens = fast_dllm_generate(
    model, prefix, gen_len=256, schedule=schedule, mask_token_id=0,
    decoding_strategy = "factor",
    factor            = 1.5,
)
```

---

## Visualising the Tradeoff

The [confidence sweep benchmark](../benchmarks.md) measures
`avg_steps_to_complete` and `tok/s` for every $(\tau, f)$ value across
MHA / GQA / MLA attention types.  The key insight from the sweep:

- **Threshold**: steps decrease sharply as $\tau$ drops below 0.9; quality
  degrades gently — the sweet spot is $\tau = 0.9$.
- **Factor**: throughput peaks at $f \approx 1.5$–$2.0$; beyond $f = 3$ the
  gains plateau while accuracy declines.
- **Attention type does not significantly affect the optimal hyper-parameter**:
  the same $\tau$ or $f$ works across MHA, GQA, and MLA.
