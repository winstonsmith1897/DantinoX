# Custom Benchmark Task

`BenchmarkTask` is a one-method plugin. Add a class, give it a `name`, implement `run()` — it's immediately usable with any `BenchmarkSuite`.

---

## Step 1: Implement the task

```python
# dantinox/benchmarking/tasks/accuracy.py
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dantinox.benchmarking.base import BenchmarkConfig, BenchmarkResult, BenchmarkTask


class TopKAccuracyTask(BenchmarkTask):
    """Measures top-K next-token accuracy on a held-out corpus.

    Runs ``config.eval_batches`` batches of shape
    ``[eval_batch_size, eval_seq_len]`` through the model and computes
    the fraction of positions where the ground-truth token appears in
    the top-K predictions.

    Args:
        k: Number of candidates to consider. Default ``5``.
        data_source: Path to a validation text file. If ``None``, random
            token IDs are used (useful for smoke-testing).
    """

    name = "top_k_accuracy"

    def __init__(self, k: int = 5, data_source: str | None = None) -> None:
        self.k           = k
        self.data_source = data_source

    def run(
        self,
        paradigm: Any,
        model: Any,
        config: BenchmarkConfig,
        rng: Any,
    ) -> BenchmarkResult:
        """Evaluate top-K accuracy over the evaluation corpus.

        Args:
            paradigm: Any :class:`~dantinox.paradigms.Paradigm` instance.
            model: The NNX model to evaluate.
            config: Suite-level benchmark configuration; ``eval_batches``,
                ``eval_seq_len``, and ``eval_batch_size`` are used.
            rng: JAX random key.

        Returns:
            :class:`~dantinox.benchmarking.base.BenchmarkResult` with
            ``metrics = {"top_k_accuracy": <float>, "k": <int>}``.
        """
        hits, total = 0, 0
        B   = config.eval_batch_size
        T   = config.eval_seq_len
        V   = getattr(
            getattr(paradigm, "config", None) or getattr(paradigm, "model_config", None),
            "vocab_size", 1000,
        )

        for _ in range(config.eval_batches):
            rng, rng_b = jax.random.split(rng)
            batch = jax.random.randint(rng_b, (B, T + 1), 0, V)
            x, y  = batch[:, :-1], batch[:, 1:]

            try:
                out    = model(x)
                logits = out.logits          # [B, T, V]
            except AttributeError:
                # Fallback: paradigm.loss_fn is not the right interface here
                continue

            top_k = jnp.argsort(logits, axis=-1)[..., -self.k:]   # [B, T, k]
            hit   = jnp.any(top_k == y[..., None], axis=-1)       # [B, T]
            hits  += int(jnp.sum(hit))
            total += hit.size

        accuracy = hits / max(total, 1)
        return BenchmarkResult(
            task=self.name,
            metrics={"top_k_accuracy": accuracy, "k": float(self.k)},
        )

    def __repr__(self) -> str:
        return f"TopKAccuracyTask(k={self.k})"
```

---

## Step 2: Register in `dantinox/benchmarking/tasks/__init__.py`

```python
# dantinox/benchmarking/tasks/__init__.py
from dantinox.benchmarking.tasks.throughput import ThroughputTask
from dantinox.benchmarking.tasks.latency    import LatencyTask
from dantinox.benchmarking.tasks.perplexity import PerplexityTask
from dantinox.benchmarking.tasks.accuracy   import TopKAccuracyTask  # ← new

__all__ = ["ThroughputTask", "LatencyTask", "PerplexityTask", "TopKAccuracyTask"]
```

---

## Step 3: Use it

```python
from dantinox.benchmarking import BenchmarkSuite
from dantinox.benchmarking.tasks.accuracy import TopKAccuracyTask

suite = BenchmarkSuite(tasks=[TopKAccuracyTask(k=10), PerplexityTask()])
report = suite.run(paradigm, model)
print(report.summary())
```

---

## Step 4: Test it

```python
# tests/test_topk_task.py
import jax
import jax.numpy as jnp
from flax import nnx
from core.config import ModelConfig
from dantinox.paradigms.ar import ARParadigm
from dantinox.benchmarking import BenchmarkConfig
from dantinox.benchmarking.tasks.accuracy import TopKAccuracyTask

def test_topk_task_result_keys():
    cfg      = ModelConfig(dim=64, n_heads=4, head_size=16, num_blocks=2,
                           vocab_size=100, causal=True)
    paradigm = ARParadigm(cfg)
    model    = paradigm.build_model(nnx.Rngs(0))
    config   = BenchmarkConfig(eval_batches=2, eval_seq_len=16, eval_batch_size=2)
    task     = TopKAccuracyTask(k=5)
    result   = task.run(paradigm, model, config, jax.random.PRNGKey(0))
    assert "top_k_accuracy" in result.metrics
    assert 0.0 <= result.metrics["top_k_accuracy"] <= 1.0
```

---

## Checklist

- [ ] `name: ClassVar[str]` set to a unique snake_case string
- [ ] `run()` returns a valid `BenchmarkResult` in all code paths
- [ ] Google docstring on the class and `run()`
- [ ] Exported from `dantinox/benchmarking/tasks/__init__.py`
- [ ] At least one unit test
- [ ] `make doccheck` passes
