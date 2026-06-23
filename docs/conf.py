"""Sphinx configuration for DantinoX documentation.

Theme: sphinx_rtd_theme  (classic Read the Docs layout)
Markdown: myst_parser    (all .md files processed by MyST)
"""

from __future__ import annotations

import os
import sys

# Make the source package importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

# ── Project metadata ──────────────────────────────────────────────────────────
project   = "DantinoX"
author    = "Marco Simoni"
copyright = "2026, Marco Simoni"
release   = "0.4.0"
version   = "0.4"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "myst_parser",          # Markdown (.md) source files via MyST
    "sphinx.ext.autodoc",   # API docs extracted from docstrings
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinx.ext.viewcode",  # [source] links on API pages
    "sphinx.ext.intersphinx",  # Links to JAX / Flax / Python docs
    "sphinx_rtd_theme",     # Classic Read the Docs HTML theme
    "sphinx_copybutton",    # Copy button on code blocks
]

# ── Source file types ─────────────────────────────────────────────────────────
# Both .rst (traditional Sphinx) and .md (MyST-Parser) are accepted.
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

master_doc = "index"

# ── Markdown support (MyST-Parser) ───────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",      # ::: directive blocks in Markdown
    "deflist",          # Definition lists (term\n: definition)
    "fieldlist",        # Field lists (:key: value)
    "html_admonition",  # Raw <div class="admonition"> HTML blocks
    "html_image",       # Raw <img> tags
    "attrs_inline",     # Inline attribute syntax {.class #id}
    "tasklist",         # GitHub-style - [ ] / - [x] task lists
    "dollarmath",       # Inline $…$ and display $$…$$ maths
]

myst_heading_anchors = 3   # auto-generate anchors for h1–h3

# ── General ───────────────────────────────────────────────────────────────────
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "coverage",      # HTML coverage report — not part of the Sphinx source tree
    "index.md",      # MkDocs landing page; replaced by index.rst for Sphinx
    # Flat legacy overview pages superseded by structured subdirectories;
    # none of these appear in the MkDocs nav either.
    "api.md",
    "generation.md",
    "training.md",
]

templates_path = ["_templates"]

# ── autodoc ───────────────────────────────────────────────────────────────────
autoclass_content = "both"          # merge class + __init__ docstrings
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "private-members":  False,
    "show-inheritance": True,
    "member-order":     "bysource",
}

# ── napoleon (Google-style docstrings) ───────────────────────────────────────
napoleon_google_docstring       = True
napoleon_numpy_docstring        = False
napoleon_include_init_with_doc  = True
napoleon_use_ivar               = True

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3",              None),
    "jax":    ("https://jax.readthedocs.io/en/latest/",  None),
    "flax":   ("https://flax.readthedocs.io/en/latest/", None),
    "numpy":  ("https://numpy.org/doc/stable/",          None),
}

# ── HTML output — sphinx_rtd_theme ────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    # Sidebar navigation
    "logo_only":                    False,
    "collapse_navigation":          True,
    "sticky_navigation":            True,
    "navigation_depth":             4,
    "includehidden":                True,
    "titles_only":                  False,
    # Prev / next arrows at the bottom of each page
    "prev_next_buttons_location":   "bottom",
    # Show external-link icon next to off-site hrefs
    "style_external_links":         True,
    # Top navigation-bar background (RTD classic blue)
    "style_nav_header_background":  "#2980B9",
}

html_logo        = "images/dantinox-transparent.png"
html_favicon     = "images/dantinox.png"
html_static_path = ["_static"]

# Inject a custom stylesheet after the theme CSS so overrides work cleanly.
html_css_files = ["custom.css"]

html_context = {
    "display_github": True,
    "github_user":    "winstonsmith1897",
    "github_repo":    "DantinoX",
    "github_version": "main",
    "conf_py_path":   "/docs/",
}

html_show_sourcelink = True
html_show_sphinx     = True
