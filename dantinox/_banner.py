"""DantinoX ASCII-art banner — printed at training start and CLI entry."""
from __future__ import annotations

import os
import sys

_DIM = "\033[2m"
_R   = "\033[0m"
_W   = "\033[1;97m"

# ── Block-letter glyphs (5 rows × 5 cols each) ────────────────────────────────
_G = {
    'D': ["████ ", "█   █", "█   █", "█   █", "████ "],
    'a': ["     ", " ███ ", "█   █", "█  ██", " ████"],
    'n': ["     ", "████ ", "█   █", "█   █", "█   █"],
    't': ["  █  ", "█████", "  █  ", "  █  ", "  ██ "],
    'i': [" █   ", "     ", " █   ", " █   ", " ███ "],
    'o': ["     ", " ███ ", "█   █", "█   █", " ███ "],
    'X': ["█   █", " █ █ ", "  █  ", " █ █ ", "█   █"],
}

# One 256-color stop per letter: magenta → lavender → violet → blue → azure → cyan
_GRAD = {
    'D': 201,   # #ff00ff  magenta
    'a': 171,   # #d75fff  purple-pink
    'n': 141,   # #af87ff  lavender
    't':  99,   # #875fff  blue-purple
    'i':  69,   # #5f87ff  cornflower blue
    'o':  39,   # #00afff  azure
    'X':  51,   # #00ffff  cyan
}
# 'n' appears twice; both use the same color key so the gradient is symmetric
_TEXT = "DantinoX"

_printed = False


def _render() -> list[str]:
    """Return the 5 rows of the DANTINOX block-letter logo."""
    lines = []
    for row in range(5):
        parts = []
        seen: dict[str, int] = {}          # track repeated letters
        for ch in _TEXT:
            col = _GRAD.get(ch, 255)
            parts.append(f"\033[38;5;{col}m{_G[ch][row]}\033[0m")
        lines.append(" ".join(parts))
    return lines


def print_banner(version: str = "") -> None:
    """Print the DantinoX logo to stderr.

    No-ops when stderr is not a TTY, ``NO_COLOR`` is set, ``TERM=dumb``,
    or the banner has already been shown once in this process.
    """
    global _printed
    if _printed:
        return
    if not sys.stderr.isatty():
        return
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return

    _printed = True
    logo  = _render()
    pad   = "  "
    tag   = f"{_DIM}JAX/Flax transformer library" + (f"  v{version}" if version else "") + _R
    lines = ["", *[pad + ln for ln in logo], "", pad + tag, ""]
    print("\n".join(lines), file=sys.stderr)
