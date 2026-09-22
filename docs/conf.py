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
    "design_remote_venv.md",
]

# Every cross-reference in this build resolves, so missing ones are reported
# rather than passed over: `sphinx-build -n` is clean, and a new dangling
# reference shows up as a warning instead of silently rendering as plain text.
nitpicky = True

# Two kinds of unresolvable reference are left, and each needs the opposite
# treatment.
#
# First, third-party names autodoc renders *unqualified*. Every module here
# uses `from __future__ import annotations`, so an annotation is rendered
# exactly as the source spells it -- a bare `BaseModel`, which no inventory
# lists under that name. `_QUALIFY_XREFS` maps those onto the qualified name
# intersphinx does know; see `_qualify_bare_xref` at the end of this file for
# why that is a `missing-reference` hook rather than `autodoc_type_aliases`
# (which re-evaluates its values, and here yields `TypeAliasForwardRef`).
_QUALIFY_XREFS = {
    "BaseModel": "pydantic.BaseModel",
    "CliSettingsSource": "pydantic_settings.CliSettingsSource",
    "Path": "pathlib.Path",
    "PydanticBaseSettingsSource": "pydantic_settings.PydanticBaseSettingsSource",
}

# Second, names with nothing to link to anywhere -- which is why these are
# ignored rather than documented or qualified.
nitpick_ignore = [
    # pydantic-settings' internal type aliases. They appear in the settings
    # source signatures shinobi.config overrides, but are absent from every
    # published inventory.
    ("py:class", "DotenvType"),
    ("py:class", "EnvPrefixTarget"),
    ("py:class", "PathType"),
    # The wiring proxies behind `Recipe.inputs`/`Recipe.outputs`. They show up
    # in return annotations but are deliberately private -- documenting them
    # would advertise an API that isn't one.
    ("py:class", "shinobi.steps.schema._InputsProxy"),
    ("py:class", "shinobi.steps.schema._LoopOutputsProxy"),
    ("py:class", "shinobi.steps.schema._OutputsProxy"),
]

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

# -- Cross-reference resolution ----------------------------------------------


def _qualify_bare_xref(app, env, node, contnode):
    """Rewrite an unqualified third-party reference to its qualified name.

    Returning None hands the mutated node to the next ``missing-reference``
    handler, which is intersphinx's -- so this only has to supply the name,
    not resolve it. Connected below intersphinx's default priority so it runs
    first. See ``_QUALIFY_XREFS`` above for what this covers and why.
    """
    qualified = _QUALIFY_XREFS.get(node.get("reftarget"))
    if qualified is not None:
        node["reftarget"] = qualified
    return None


def setup(app):
    app.connect("missing-reference", _qualify_bare_xref, priority=400)
