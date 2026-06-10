# API Reference

The API reference is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/). Every public class, method, and function is documented here.

---

## Modules

| Module | Contents |
| :--- | :--- |
| [`dantinox`](dantinox.md) | Top-level `fit`, `train`, `build`, `profile`, `load`, `quick_generate` |
| [`dantinox.paradigms`](paradigms.md) | `Paradigm`, `ARParadigm`, `DiscreteParadigm`, `ContinuousParadigm`, `DiscreteConfig` |
| [`dantinox.training`](training.md) | `Trainer`, `build_optimizer`, `build_schedule`, `TrainingConfig` |
| [`dantinox.profiling`](profiling.md) | `count_flops`, `FLOPsBreakdown`, `LatencyTracker`, `ProfilingResult`, `profile_fn` |
| [`dantinox.benchmarking`](benchmarking.md) | `BenchmarkSuite`, `BenchmarkTask`, `BenchmarkResult`, `SuiteReport`, built-in tasks |
| [`dantinox.visualization`](visualization.md) | `Visualizer`, `Chart`, `RenderConfig`, built-in charts |

---

## Docstring style

All public symbols use **Google-style docstrings**:

```python
def my_function(x: int, y: float = 1.0) -> str:
    """One-line summary.

    Longer description if needed. Multiple paragraphs are fine.

    Args:
        x: Description of x.
        y: Description of y. Default ``1.0``.

    Returns:
        A string representation.

    Raises:
        ValueError: If ``x`` is negative.

    Example:
        >>> my_function(5)
        '5.00'
    """
```

## Enforcement

Docstring coverage is enforced by [interrogate](https://interrogate.readthedocs.io/):

```bash
make doccheck          # fails if coverage < 100 %
```

This runs in CI on every pull request. See [Contributing](../contributing.md) for details.
