from __future__ import annotations

from typing import Any

# ── Named colour palettes ─────────────────────────────────────────────────────

PALETTES: dict[str, list[str]] = {
    "publication": ["#3A86FF", "#FF6B35", "#2DC653", "#8338EC", "#FB5607", "#FFBE0B"],
    "dark":        ["#00B4D8", "#EF233C", "#80ED99", "#F72585", "#4CC9F0", "#F9C74F"],
    "minimal":     ["#222222", "#555555", "#888888", "#BBBBBB", "#DDDDDD", "#F0F0F0"],
}

# Model-type colours — consistent across all charts
TYPE_COLORS: dict[str, str] = {
    "MLA": "#3A86FF",
    "GQA": "#FF6B35",
    "MHA": "#2DC653",
    "AR":  "#8338EC",
}


# ── Style presets ─────────────────────────────────────────────────────────────

_PUBLICATION_RC: dict[str, Any] = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
}

_DARK_RC: dict[str, Any] = {
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444466",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "axes.grid":         True,
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "lines.linewidth":   2.0,
}

_MINIMAL_RC: dict[str, Any] = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "black",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "lines.linewidth":   1.5,
}

_RC_MAP: dict[str, dict[str, Any]] = {
    "publication": _PUBLICATION_RC,
    "dark":        _DARK_RC,
    "minimal":     _MINIMAL_RC,
}


def apply_style(name: str = "publication") -> None:
    """Apply the named matplotlib rcParam preset globally."""
    import matplotlib as mpl
    rc = _RC_MAP.get(name, _PUBLICATION_RC)
    mpl.rcParams.update(rc)


def get_palette(style: str = "publication") -> list[str]:
    """Return the colour palette for the given style preset."""
    return PALETTES.get(style, PALETTES["publication"])
