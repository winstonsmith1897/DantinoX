"""
Automated plot generation for DantinoX benchmark results.

All plot functions are collected from the standalone scripts
(plot_insights, plot_perf, plot_3d, plot_3d_dkv) and exposed through
the ``Plotter`` class, which can run them all or a named subset.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Optional, Sequence


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

ALL_GROUPS = list(_PLOT_GROUPS.keys())


def _import_plot_module(module_name: str, repo_root: str):
    """Import a root-level plot_*.py script as a module."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module(module_name)


def _run_group(
    group: str,
    in_csv: str,
    out_dir: str,
    repo_root: str,
    batch_csv: Optional[str] = None,
) -> list[str]:
    """Run all figure functions in a group, return list of saved filenames."""
    module_name, fig_fns = _PLOT_GROUPS[group]
    mod = _import_plot_module(module_name, repo_root)

    # Temporarily patch the module-level paths so _save() writes to out_dir
    orig_in  = getattr(mod, "IN_CSV",  None)
    orig_out = getattr(mod, "OUT_DIR", None)
    mod.IN_CSV  = in_csv
    mod.OUT_DIR = out_dir
    if hasattr(mod, "BATCH_CSV") and batch_csv:
        mod.BATCH_CSV = batch_csv

    saved = []
    try:
        df = mod.load()
        for fn_name in fig_fns:
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue
            if fn_name == "fig4_batch_throughput" and group == "perf":
                # This figure takes a batch DataFrame, not the main one
                bdf = mod.load_batch() if hasattr(mod, "load_batch") else None
                if bdf is not None and not bdf.empty:
                    fn(bdf)
                else:
                    getattr(mod, "fig4_missing", lambda: None)()
            else:
                fn(df)
            saved.append(fn_name)
    finally:
        if orig_in  is not None: mod.IN_CSV  = orig_in
        if orig_out is not None: mod.OUT_DIR = orig_out

    return saved


class Plotter:
    """
    Generates all DantinoX benchmark plots from a CSV of benchmark results.

    Parameters
    ----------
    in_csv : str
        Path to the ``benchmark_results.csv`` produced by
        ``BenchmarkRunner.run(out_csv=...)`` or ``benchmark.py``.
    out_dir : str
        Directory where PNGs will be written (default ``"plots"``).
    batch_csv : str, optional
        Path to ``batch_sweep_results.csv`` for the batch throughput plot.

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
        batch_csv: Optional[str] = None,
    ) -> None:
        self.in_csv    = in_csv
        self.out_dir   = out_dir
        self.batch_csv = batch_csv
        self._repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(
        self,
        groups: Optional[Sequence[str]] = None,
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
            Mapping of group name → list of figure function names that ran.
        """
        if not os.path.exists(self.in_csv):
            raise FileNotFoundError(
                f"Benchmark CSV not found: {self.in_csv}\n"
                "Run BenchmarkRunner.run(out_csv='benchmark_results.csv') first."
            )

        os.makedirs(self.out_dir, exist_ok=True)
        selected = list(groups) if groups else ALL_GROUPS
        unknown  = [g for g in selected if g not in _PLOT_GROUPS]
        if unknown:
            raise ValueError(f"Unknown plot group(s): {unknown}. Valid: {ALL_GROUPS}")

        results: dict[str, list[str]] = {}
        for group in selected:
            print(f"\n[{group}] generating plots…")
            try:
                done = _run_group(
                    group, self.in_csv, self.out_dir,
                    self._repo_root, self.batch_csv,
                )
                results[group] = done
                print(f"  {len(done)} figures saved to {self.out_dir}/")
            except Exception as exc:
                print(f"  ERROR in group '{group}': {exc}")
                results[group] = []

        return results
