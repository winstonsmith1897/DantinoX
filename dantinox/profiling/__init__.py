from dantinox.profiling.counter import FLOPsBreakdown, count_flops
from dantinox.profiling.tracker import (
    # ── Per-metric result types ───────────────────────────────────────────────
    LatencyResult,
    ThroughputResult,
    EnergyResult,
    FLOPsResult,
    PerplexityResult,
    EntropyResult,
    # ── Per-metric measurement classes ───────────────────────────────────────
    LatencyMetric,
    ThroughputMetric,
    EnergyMetric,
    FLOPsMetric,
    PerplexityMetric,
    EntropyMetric,
    # ── Backward-compatible names ─────────────────────────────────────────────
    LatencyTracker,
    ProfilingResult,
    profile_fn,
)
from dantinox.profiling.runner import (
    RunProfile,
    MultiRunReport,
    RunsProfiler,
)
from dantinox.profiling.plots import (
    # 3D interactive (Plotly)
    plot_3d_surface,
    plot_3d_compare,
    plot_3d_from_df,
    # 2D matplotlib
    plot_scale,
    plot_batch,
    plot_steps,
    plot_dtype,
    plot_composite,
    # scalar bar chart
    plot_bar_compare,
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
