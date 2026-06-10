from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class RenderConfig:
    """Controls how a :class:`Chart` renders and saves its output.

    Attributes
    ----------
    backend  : Plotting engine — ``"matplotlib"`` (default) or ``"plotly"``.
    fmt      : Output file format — ``"png"`` | ``"pdf"`` | ``"svg"``.
    dpi      : Raster resolution (ignored for vector formats).
    width    : Figure width in inches.
    height   : Figure height in inches.
    style    : Named style preset — ``"publication"`` | ``"dark"`` | ``"minimal"``.
    """

    backend: str  = "matplotlib"
    fmt: str      = "png"
    dpi: int      = 150
    width: float  = 10.0
    height: float = 6.0
    style: str    = "publication"

    def __post_init__(self) -> None:
        if self.backend not in ("matplotlib", "plotly"):
            raise ValueError(
                f"backend must be 'matplotlib' or 'plotly'; got {self.backend!r}"
            )
        if self.fmt not in ("png", "pdf", "svg"):
            raise ValueError(
                f"fmt must be 'png', 'pdf', or 'svg'; got {self.fmt!r}"
            )
        if self.style not in ("publication", "dark", "minimal"):
            raise ValueError(
                f"style must be 'publication', 'dark', or 'minimal'; got {self.style!r}"
            )


class Chart(ABC):
    """Abstract base for all DantinoX visualization charts.

    Implementing a new chart requires only two overrides::

        class AccuracyChart(Chart):
            name     = "accuracy"
            accepts  = SuiteReport          # or pd.DataFrame, list[dict], …

            def _render_mpl(self, data, config, fig, ax):
                ax.plot(data.epochs, data.accuracy)
                ax.set_title("Accuracy over epochs")

    The chart is then automatically available through the
    :class:`~dantinox.visualization.visualizer.Visualizer` registry once
    imported::

        from dantinox.visualization import Visualizer
        Visualizer.register(AccuracyChart)
    """

    name: ClassVar[str]
    accepts: ClassVar[type]

    # ── Public interface ──────────────────────────────────────────────────────

    def render(
        self,
        data: Any,
        config: RenderConfig,
        out_path: Path,
    ) -> Path:
        """Render the chart and write it to *out_path*.

        Dispatches to the appropriate backend method based on
        ``config.backend``.  Returns the resolved output path.
        """
        if config.backend == "plotly":
            return self._dispatch_plotly(data, config, out_path)
        return self._dispatch_mpl(data, config, out_path)

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _dispatch_mpl(
        self,
        data: Any,
        config: RenderConfig,
        out_path: Path,
    ) -> Path:
        import matplotlib.pyplot as plt
        from dantinox.visualization.style import apply_style

        apply_style(config.style)
        fig, ax = plt.subplots(figsize=(config.width, config.height))
        self._render_mpl(data, config, fig, ax)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _dispatch_plotly(
        self,
        data: Any,
        config: RenderConfig,
        out_path: Path,
    ) -> Path:
        # Subclasses override _render_plotly; default falls back to matplotlib.
        if not self._has_plotly_impl():
            return self._dispatch_mpl(data, config, out_path)
        fig = self._render_plotly(data, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path))
        return out_path

    # ── Subclass hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _render_mpl(
        self,
        data: Any,
        config: RenderConfig,
        fig: Any,
        ax: Any,
    ) -> None:
        """Draw onto *ax* using matplotlib.  Subclasses must implement this."""
        ...

    def _render_plotly(self, data: Any, config: RenderConfig) -> Any:
        """Return a plotly Figure.  Optional — falls back to matplotlib."""
        raise NotImplementedError

    def _has_plotly_impl(self) -> bool:
        return type(self)._render_plotly is not Chart._render_plotly

    # ── Multi-axes charts ─────────────────────────────────────────────────────

    def _dispatch_mpl_multi(
        self,
        data: Any,
        config: RenderConfig,
        out_path: Path,
        nrows: int = 1,
        ncols: int = 1,
        **subplot_kwargs: Any,
    ) -> Path:
        """Variant of _dispatch_mpl that creates a figure with multiple axes.

        Calls ``_render_mpl_multi(data, config, fig, axes)`` instead.
        """
        import matplotlib.pyplot as plt
        from dantinox.visualization.style import apply_style

        apply_style(config.style)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(config.width, config.height),
            **subplot_kwargs,
        )
        self._render_mpl_multi(data, config, fig, axes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_mpl_multi(self, data: Any, config: RenderConfig, fig: Any, axes: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _render_mpl or _render_mpl_multi"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
