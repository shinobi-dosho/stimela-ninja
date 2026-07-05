"""Sphinx configuration for the stimela-ninja documentation.

Autodoc imports the ``shinobi`` package, so the build environment must have it
installed (``uv sync --group docs`` locally; Read the Docs installs it via
``.readthedocs.yaml``). The package lives under ``src/``, added to sys.path
below so an editable/uninstalled checkout also builds.
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

from shinobi import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "stimela-ninja"
author = "Sphesihle Makhathini"
copyright = f"{date.today().year}, {author}"

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
# design_sandbox.md is internal design scratch, not user-facing docs.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "design_sandbox.md"]

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
    "source_repository": "https://github.com/SpheMakh/stimela-ninja/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
