# Developer Guide

DantinoX is designed to be extended without touching the core library. Every major seam — core layers, training paradigms, benchmark tasks, and visualization charts — has a clean plugin interface.

---

## Extension points

| What to add | Interface | Guide |
| :--- | :--- | :--- |
| New attention variant, MLP type, or normalization layer | `core/` NNX module + `Config` field | [Adding a Core Layer](new-layer.md) |
| New training objective (e.g. contrastive, RLHF) | `Paradigm` subclass | [Custom Paradigm](new-paradigm.md) |
| New evaluation metric or dataset | `BenchmarkTask` subclass | [Custom Benchmark Task](new-benchmark.md) |
| New chart type | `Chart` subclass + `@Visualizer.register` | [Custom Chart](new-chart.md) |

---

## General principles

- **Extend, don't modify.** New functionality belongs in new files, not edits to existing ones (unless fixing a bug).
- **Typed contracts.** Use `ClassVar[str]` for `name`, typed signatures for `run()` and `loss_fn()` — the type checker will catch mismatches.
- **No global state.** Paradigms and tasks are stateless; all mutable state lives in the model or optimizer.
- **Test your addition.** Add a test in `tests/` before opening a PR. See [Contributing](../contributing.md).
