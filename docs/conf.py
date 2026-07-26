"""Sphinx configuration for the stimela-ninja documentation.

Autodoc imports the ``shinobi`` package, so the build environment must have it
installed (``uv sync --group docs`` locally; Read the Docs installs it via
``.readthedocs.yaml``). The package lives under ``src/``, added to sys.path
below so an editable/uninstalled checkout also builds.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("../src"))

from shinobi import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "stimela-ninja"
author = "Sphesihle Makhathini"
copyright = f"{datetime.now(tz=timezone.utc).year}, {author}"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
# The design_*.md files are internal design scratch, not user-facing docs --
# they record how a feature was argued into existence, and the user-facing
# half lives under concepts/ once it ships. Excluded rather than left out of
# a toctree, which is what "document isn't included in any toctree" means
# under `sphinx-build -W`.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "design_sandbox.md",
    "design_cache_tiers.md",
]

# Treat warnings as build-relevant but don't fail the build on missing
# autodoc targets during early scaffolding.
nitpicky = False

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
# pydantic BaseModels carry a lot of inherited machinery; don't document it.
autodoc_inherit_docstrings = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render Google-style "Attributes:" sections as an :ivar: field list on the
# class docstring instead of standalone `.. attribute::` directives -- the
# latter collides with autodoc's own scan of annotated class attributes
# (e.g. Backend.name), producing "duplicate object description" warnings.
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"stimela-ninja {release}"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/shinobi-dosho/stimela-ninja/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
