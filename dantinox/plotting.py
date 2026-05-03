"""
Plot generation for DantinoX benchmark results.

Delegates to the four plot modules bundled in ``dantinox/plots/``, producing
all 16 figures from a benchmark CSV.  The public API is the :class:`Plotter`
class; the four script modules can also be called directly for advanced use.

Usage::

    from dantinox import Plotter

    Plotter("benchmark_results.csv").run()
    Plotter("benchmark_results.csv").run(groups=["perf", "3d"])
"""

from __future__ import annotations

import importlib
import logging
import os
import types

from dantinox.exceptions import PlotError

log = logging.getLogger(__name__)

# group name → (module, list of figure functions)
_PLOT_GROUPS: dict[str, tuple[str, list[str]]] = {
    "perf": (
        "dantinox.plots.plot_perf",
        ["fig1_cache_breakdown", "fig2_seqlen_throughput",
         "fig3_flops_vs_cache", "fig4_batch_throughput", "fig5_prefill"],
    ),
    "insights": (
        "dantinox.plots.plot_insights",
        ["fig1_pareto", "fig2_serving", "fig3_mla_dial"],
    ),
    "3d": (
        "dantinox.plots.plot_3d",
        ["fig1_cache_surface", "fig2_quality_cube",
         "fig3_efficiency_cube", "fig4_serving_surface"],
    ),
    "3d_dkv": (
        "dantinox.plots.plot_3d_dkv",
        ["fig5_dkv_cache_seqlen", "fig6_kv_decoupling",
         "fig7_mla_quality", "fig8_dkv_numblocks"],
    ),
}

ALL_GROUPS: list[str] = list(_PLOT_GROUPS.keys())


def _run_group(
    group: str,
    in_csv: str,
    out_dir: str,
    batch_csv: str | None,
) -> list[str]:
    module_name, fig_fns = _PLOT_GROUPS[group]
    try:
        mod: types.ModuleType = importlib.import_module(module_name)
    except ImportError as exc:
        raise PlotError(f"Cannot import {module_name}: {exc}") from exc

    # Temporarily patch module-level path constants so the scripts write
    # to the caller's out_dir and read from the caller's CSV.
    orig_in  = getattr(mod, "IN_CSV",  None)
    orig_out = getattr(mod, "OUT_DIR", None)
    orig_batch = getattr(mod, "BATCH_CSV", None)
    mod.IN_CSV  = in_csv   # type: ignore[attr-defined]
    mod.OUT_DIR = out_dir  # type: ignore[attr-defined]
    if hasattr(mod, "BATCH_CSV") and batch_csv:
        mod.BATCH_CSV = batch_csv  # type: ignore[attr-defined]

    saved: list[str] = []
    try:
        df = mod.load()
        for fn_name in fig_fns:
            fn = getattr(mod, fn_name, None)
            if fn is None:
                log.warning("  %s not found in %s — skipped", fn_name, module_name)
                continue
            if fn_name == "fig4_batch_throughput":
                bdf = mod.load_batch() if hasattr(mod, "load_batch") else None
                if bdf is not None and not bdf.empty:
                    fn(bdf)
                else:
                    getattr(mod, "fig4_missing", lambda: None)()
            else:
                fn(df)
            saved.append(fn_name)
            log.debug("  %s — done", fn_name)
    finally:
        if orig_in is not None:
            mod.IN_CSV = orig_in  # type: ignore[attr-defined]
        if orig_out is not None:
            mod.OUT_DIR = orig_out  # type: ignore[attr-defined]
        if orig_batch is not None:
            mod.BATCH_CSV = orig_batch  # type: ignore[attr-defined]

    return saved


class Plotter:
    """
    Generates all DantinoX benchmark plots from a CSV produced by
    :meth:`~dantinox.BenchmarkRunner.run`.

    Runs the four bundled plot modules (``perf``, ``insights``, ``3d``,
    ``3d_dkv``) and writes 16 PNG files to *out_dir*.

    Parameters
    ----------
    in_csv : str
        Path to ``benchmark_results.csv``.
    out_dir : str
        Directory where PNGs are written (created if absent).
    batch_csv : str, optional
        Path to ``batch_sweep_results.csv`` for the batch-throughput plot.
        If omitted, that figure is replaced with a placeholder.

    Raises
    ------
    PlotError
        If the CSV is missing or a group name is invalid.

    Examples
    --------
    >>> from dantinox import BenchmarkRunner, Plotter
    >>> BenchmarkRunner("runs").run(out_csv="benchmark_results.csv")
    >>> Plotter("benchmark_results.csv").run()
    """

    def __init__(
        self,
        in_csv: str = "benchmark_results.csv",
        out_dir: str = "plots",
        *,
        batch_csv: str | None = None,
    ) -> None:
        self.in_csv    = in_csv
        self.out_dir   = out_dir
        self.batch_csv = batch_csv

    def __repr__(self) -> str:
        return f"Plotter(in_csv={self.in_csv!r}, out_dir={self.out_dir!r})"

    def run(self, groups: list[str] | None = None) -> dict[str, list[str]]:
        """
        Generate plots and save them as PNGs.

        Parameters
        ----------
        groups : list[str], optional
            Subset of ``["perf", "insights", "3d", "3d_dkv"]``.
            Generates all four if omitted.

        Returns
        -------
        dict[str, list[str]]
            Mapping of group name → list of figure function names that ran.

        Raises
        ------
        PlotError
            If the benchmark CSV is not found or a group name is invalid.
        """
        if not os.path.exists(self.in_csv):
            raise PlotError(
                f"Benchmark CSV not found: {self.in_csv}\n"
                "Run BenchmarkRunner.run(out_csv='benchmark_results.csv') first."
            )

        os.makedirs(self.out_dir, exist_ok=True)
        selected = list(groups) if groups else ALL_GROUPS
        unknown  = [g for g in selected if g not in _PLOT_GROUPS]
        if unknown:
            raise PlotError(
                f"Unknown plot group(s): {unknown}. Valid groups: {ALL_GROUPS}"
            )

        results: dict[str, list[str]] = {}
        for group in selected:
            log.info("[%s] generating plots…", group)
            try:
                done = _run_group(group, self.in_csv, self.out_dir, self.batch_csv)
                results[group] = done
                log.info("  %d figures written to %s/", len(done), self.out_dir)
            except PlotError:
                raise
            except Exception as exc:
                log.error("  group '%s' failed: %s", group, exc)
                results[group] = []

        return results
