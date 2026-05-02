"""
Plot generation for DantinoX benchmark results.

All chart logic lives directly in :class:`Plotter` — no external scripts
are needed. Works from both the installed wheel and an editable checkout.

Usage::

    from dantinox import Plotter

    Plotter("benchmark_results.csv").run()
    Plotter("benchmark_results.csv").run(groups=["perf"])
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dantinox.exceptions import PlotError

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

GROUPS: list[str] = ["perf", "3d", "insights"]


class Plotter:
    """
    Generates DantinoX benchmark plots from a CSV produced by
    :meth:`~dantinox.BenchmarkRunner.run`.

    Parameters
    ----------
    in_csv : str
        Path to ``benchmark_results.csv``.
    out_dir : str
        Directory where PNGs are written (created if absent).

    Raises
    ------
    PlotError
        If the CSV is missing, a group name is invalid, or matplotlib
        is not installed.

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
    ) -> None:
        self.in_csv  = in_csv
        self.out_dir = out_dir

    def __repr__(self) -> str:
        return f"Plotter(in_csv={self.in_csv!r}, out_dir={self.out_dir!r})"

    # ── public ──────────────────────────────────────────────────────────────

    def run(self, groups: list[str] | None = None) -> dict[str, str]:
        """
        Generate plots and save them as PNGs.

        Parameters
        ----------
        groups : list[str], optional
            Subset of ``["perf", "3d", "insights"]``.  Generates all
            if omitted.

        Returns
        -------
        dict[str, str]
            Mapping of group name → absolute path of the saved PNG.
        """
        if not os.path.exists(self.in_csv):
            raise PlotError(
                f"Benchmark CSV not found: {self.in_csv}\n"
                "Run BenchmarkRunner.run(out_csv='benchmark_results.csv') first."
            )

        try:
            import pandas as pd
        except ImportError as exc:
            raise PlotError(
                "pandas is required for plotting. "
                "Install it with: pip install 'dantinox[benchmark]'"
            ) from exc

        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend; safe in notebooks too
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise PlotError(
                "matplotlib is required for plotting. "
                "Install it with: pip install 'dantinox[benchmark]'"
            ) from exc

        df = pd.read_csv(self.in_csv)
        os.makedirs(self.out_dir, exist_ok=True)

        selected = list(groups) if groups else GROUPS
        unknown  = [g for g in selected if g not in GROUPS]
        if unknown:
            raise PlotError(f"Unknown group(s): {unknown}. Valid groups: {GROUPS}")

        saved: dict[str, str] = {}
        for group in selected:
            try:
                path = getattr(self, f"_plot_{group}")(df, plt)
                if path:
                    saved[group] = path
                    log.info("[%s] saved → %s", group, path)
            except Exception as exc:
                log.warning("[%s] skipped: %s", group, exc)

        return saved

    # ── private: one method per group ───────────────────────────────────────

    def _plot_perf(self, df: "pd.DataFrame", plt) -> str:
        """3-panel overview: throughput vs seq-len, throughput vs batch, prefill latency."""
        seq_cols   = [c for c in df.columns if c.startswith("tps_") and not c.startswith("tps_bs")]
        batch_cols = [c for c in df.columns if c.startswith("tps_bs")]
        seq_lens   = [int(c.replace("tps_", ""))   for c in seq_cols]
        batch_sizes = [int(c.replace("tps_bs", "")) for c in batch_cols]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        ax = axes[0]
        for _, row in df.iterrows():
            ax.plot(seq_lens, [row[c] for c in seq_cols], marker="o", label=row["run"])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Tokens / s")
        ax.set_title("Decode throughput vs seq-len (BS=1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for _, row in df.iterrows():
            ax.plot(batch_sizes, [row[c] for c in batch_cols], marker="s", label=row["run"])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Tokens / s")
        ax.set_title("Batch throughput vs batch size")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.barh(df["run"], df["prefill_ms"], color="steelblue")
        ax.set_xlabel("Prefill latency (ms)")
        ax.set_title("Prefill latency")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.out_dir, "benchmark_overview.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(path)

    def _plot_3d(self, df: "pd.DataFrame", plt) -> str:
        """3-D surface: tokens/s × sequence-length × batch-size."""
        import numpy as np

        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError as exc:
            raise PlotError("mpl_toolkits.mplot3d not available") from exc

        seq_cols   = [c for c in df.columns if c.startswith("tps_") and not c.startswith("tps_bs")]
        batch_cols = [c for c in df.columns if c.startswith("tps_bs")]
        if not seq_cols or not batch_cols:
            raise PlotError("No tps_* columns found — run the benchmark first.")

        xs = [int(c.replace("tps_", ""))   for c in seq_cols]
        ys = [int(c.replace("tps_bs", "")) for c in batch_cols]
        X, Y = np.meshgrid(xs, ys)

        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection="3d")
        for _, row in df.iterrows():
            Z = np.array([[row.get(f"tps_{x}", np.nan) for x in xs] for _ in ys])
            ax.plot_surface(X, Y, Z, alpha=0.75, label=str(row["run"]))
        ax.set_xlabel("Seq-len")
        ax.set_ylabel("Batch size")
        ax.set_zlabel("Tokens / s")
        ax.set_title("Throughput surface")

        path = os.path.join(self.out_dir, "throughput_surface.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(path)

    def _plot_insights(self, df: "pd.DataFrame", plt) -> str | None:
        """Scatter: params vs val_loss, cache vs throughput (needs ≥2 runs)."""
        import numpy as np

        has_loss  = "val_loss" in df.columns and df["val_loss"].notna().any()
        has_cache = "theoretical_cache_mb" in df.columns
        seq_cols  = [c for c in df.columns if c.startswith("tps_") and not c.startswith("tps_bs")]
        has_tps   = bool(seq_cols)

        n_panels = sum([has_loss, has_cache and has_tps])
        if n_panels == 0:
            log.warning("[insights] not enough columns — skipped")
            return None

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        idx = 0
        if has_loss:
            ax = axes[idx]; idx += 1
            best_tps = df[seq_cols].max(axis=1)
            ax.scatter(df["val_loss"], best_tps, s=80, zorder=3)
            for _, row in df.iterrows():
                ax.annotate(
                    row["run"], (row["val_loss"], df.loc[_, seq_cols].max()),
                    fontsize=6, ha="right",
                )
            ax.set_xlabel("Val loss"); ax.set_ylabel("Peak tokens / s")
            ax.set_title("Quality vs throughput (lower-left is better)")
            ax.grid(True, alpha=0.3)

        if has_cache and has_tps:
            ax = axes[idx]
            best_tps = df[seq_cols].max(axis=1)
            ax.scatter(df["theoretical_cache_mb"], best_tps, s=80, color="darkorange", zorder=3)
            for _, row in df.iterrows():
                ax.annotate(
                    row["run"], (row["theoretical_cache_mb"], df.loc[_, seq_cols].max()),
                    fontsize=6, ha="right",
                )
            ax.set_xlabel("KV-cache size (MB)"); ax.set_ylabel("Peak tokens / s")
            ax.set_title("Cache size vs throughput")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.out_dir, "benchmark_insights.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(path)
