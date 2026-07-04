from dantinox.profiling.counter import FLOPsBreakdown, count_flops
from dantinox.profiling.plots import (
    plot_3d_compare,
    plot_3d_from_df,
    # 3D interactive (Plotly)
    plot_3d_surface,
    # scalar bar chart
    plot_bar_compare,
    plot_batch,
    plot_composite,
    plot_dtype,
    # 2D matplotlib
    plot_scale,
    plot_steps,
)
from dantinox.profiling.runner import (
    MultiRunReport,
    RunProfile,
    RunsProfiler,
)
from dantinox.profiling.tracker import (
    EnergyMetric,
    EnergyResult,
    EntropyMetric,
    EntropyResult,
    FLOPsMetric,
    FLOPsResult,
    # ── Per-metric measurement classes ───────────────────────────────────────
    LatencyMetric,
    # ── Per-metric result types ───────────────────────────────────────────────
    LatencyResult,
    # ── Backward-compatible names ─────────────────────────────────────────────
    LatencyTracker,
    PerplexityMetric,
    PerplexityResult,
    ProfilingResult,
    ThroughputMetric,
    ThroughputResult,
    profile_fn,
)

__all__ = [
    # counter
    "FLOPsBreakdown",
    "count_flops",
    # result types
    "LatencyResult",
    "ThroughputResult",
    "EnergyResult",
    "FLOPsResult",
    "PerplexityResult",
    "EntropyResult",
    # metric classes
    "LatencyMetric",
    "ThroughputMetric",
    "EnergyMetric",
    "FLOPsMetric",
    "PerplexityMetric",
    "EntropyMetric",
    # multi-run profiler
    "RunProfile",
    "MultiRunReport",
    "RunsProfiler",
    # backward compat
    "LatencyTracker",
    "ProfilingResult",
    "profile_fn",
    # 3D interactive plots (Plotly)
    "plot_3d_surface",
    "plot_3d_compare",
    "plot_3d_from_df",
    # 2D benchmark figures (matplotlib)
    "plot_scale",
    "plot_batch",
    "plot_steps",
    "plot_dtype",
    "plot_composite",
    # scalar comparison
    "plot_bar_compare",
]
