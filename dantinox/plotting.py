"""
Automated plot generation for DantinoX benchmark results.

All plot functions are collected from the standalone scripts
(plot_insights, plot_perf, plot_3d, plot_3d_dkv) and exposed through
the :class:`Plotter` class, which can run them all or a named subset.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from collections.abc import Sequence

from dantinox.exceptions import PlotError

log = logging.getLogger(__name__)

# Groups map a short name → (module_path, list_of_figure_functions)
_PLOT_GROUPS: dict[str, tuple[str, list[str]]] = {
    "insights": (
        "plot_insights",
        ["fig1_pareto", "fig2_serving", "fig3_mla_dial"],
    ),
    "perf": (
        "plot_perf",
        ["fig1_cache_breakdown", "fig2_seqlen_throughput",
         "fig3_flops_vs_cache", "fig4_batch_throughput", "fig5_prefill"],
    ),
    "3d": (
        "plot_3d",
        ["fig1_cache_surface", "fig2_quality_cube",
         "fig3_efficiency_cube", "fig4_serving_surface"],
    ),
    "3d_dkv": (
        "plot_3d_dkv",
        ["fig5_dkv_cache_seqlen", "fig6_kv_decoupling",
         "fig7_mla_quality", "fig8_dkv_numblocks"],
    ),
}

ALL_GROUPS: list[str] = list(_PLOT_GROUPS.keys())


def _import_plot_module(module_name: str, repo_root: str) -> types.ModuleType:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module(module_name)


def _run_group(
    group: str,
    in_csv: str,
    out_dir: str,
    repo_root: str,
    batch_csv: str | None = None,
) -> list[str]:
    module_name, fig_fns = _PLOT_GROUPS[group]
    try:
        mod = _import_plot_module(module_name, repo_root)
    except ImportError as exc:
        raise PlotError(f"Cannot import {module_name}: {exc}") from exc

    orig_in  = getattr(mod, "IN_CSV",  None)
    orig_out = getattr(mod, "OUT_DIR", None)
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
                log.warning("Function %s not found in %s — skipped", fn_name, module_name)
                continue
            if fn_name == "fig4_batch_throughput" and group == "perf":
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
            mod.IN_CSV = orig_in   # type: ignore[attr-defined]
        if orig_out is not None:
            mod.OUT_DIR = orig_out  # type: ignore[attr-defined]

    return saved


class Plotter:
    """
    Generates all DantinoX benchmark plots from a CSV of benchmark results.

    Parameters
    ----------
    in_csv : str
        Path to the ``benchmark_results.csv`` produced by
        :meth:`BenchmarkRunner.run`.
    out_dir : str
        Directory where PNGs will be written (default ``"plots"``).
    batch_csv : str, optional
        Path to ``batch_sweep_results.csv`` for the batch throughput plot.

    Raises
    ------
    PlotError
        If the CSV is missing or a plot group fails to import.

    Examples
    --------
    >>> from dantinox import BenchmarkRunner
    >>> from dantinox.plotting import Plotter
    >>>
    >>> df = BenchmarkRunner("runs").run(out_csv="benchmark_results.csv")
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
        self._repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __repr__(self) -> str:
        return f"Plotter(in_csv={self.in_csv!r}, out_dir={self.out_dir!r})"

    def run(
        self,
        groups: Sequence[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Generate plots.

        Parameters
        ----------
        groups : list[str], optional
            Subset of groups to generate. Available: ``insights``, ``perf``,
            ``3d``, ``3d_dkv``. Generates all if omitted.

        Returns
        -------
        dict[str, list[str]]
            Mapping of group name → list of figure functions that ran.

        Raises
        ------
        PlotError
            If the benchmark CSV is not found or a group cannot be imported.
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
                done = _run_group(
                    group, self.in_csv, self.out_dir,
                    self._repo_root, self.batch_csv,
                )
                results[group] = done
                log.info("  %d figures written to %s/", len(done), self.out_dir)
            except PlotError:
                raise
            except Exception as exc:
                log.error("  Group '%s' failed: %s", group, exc)
                results[group] = []

        return results
