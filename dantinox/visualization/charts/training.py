from __future__ import annotations

from typing import Any

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import get_palette


class TrainingCurveChart(Chart):
    """Plot training (and optional validation) loss over epochs or steps.

    Accepts
    -------
    ``pandas.DataFrame``
        Must contain a numeric ``loss`` column.  Optional ``val_loss`` and
        ``epoch`` / ``step`` columns are used automatically when present.
    ``str``
        Path to a CSV with the same schema.

    Example::

        from dantinox.visualization import Visualizer
        Visualizer().render(report, charts=["training_curve"], out_dir="plots")
    """

    name    = "training_curve"
    accepts = object  # DataFrame or CSV path

    def render(self, data: Any, config: RenderConfig, out_path: Any) -> Any:
        return self._dispatch_mpl(data, config, out_path)

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        import pandas as pd
        from dantinox.visualization.style import get_palette

        df     = _load_df(data)
        colors = get_palette(config.style)
        x_col  = "step" if "step" in df.columns else "epoch" if "epoch" in df.columns else None
        x      = df[x_col] if x_col else range(len(df))

        ax.plot(x, df["loss"], color=colors[0], label="Train loss", linewidth=1.8)

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            ax.plot(
                x, df["val_loss"],
                color=colors[1], linestyle="--",
                label="Val loss", linewidth=1.8,
            )

        ax.set_xlabel(x_col.capitalize() if x_col else "Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training curve")
        ax.legend()


def _load_df(data: Any):
    import pandas as pd
    if isinstance(data, str):
        return pd.read_csv(data)
    return data
