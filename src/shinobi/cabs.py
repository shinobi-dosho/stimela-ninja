"""Resolve a `Cab`/`StepRef` by name across installed cab-provider packages.

shinobi ships no cabs itself. A cab-provider package (e.g. `dosho`, the
native shinobi cab repository) registers itself under the `shinobi.cabs`
packaging entry-point group in its own `pyproject.toml`:

    [project.entry-points."shinobi.cabs"]
    dosho = "dosho.registry"

The entry point's target is a module (or any object) exposing two
functions: `get(name: str) -> Cab | StepRef` (raising `KeyError` if `name`
isn't one of its cabs) and `list_cabs() -> list[str]`. A provider entry can
be either shape -- a `Cab` for real "binary"-flavour tools, or a `StepRef`
(what `@shinobi.pystep` produces) for Python-package tools that have no
standalone executable (e.g. CASA tasks, run via `ctx.import_func` inside a
container rather than argv-built and shelled out to) -- `Recipe.add_step`
already accepts either identically, so this resolver doesn't need to care
which one it got. This module only resolves *names* to providers -- it
never parses/builds a cab itself, and never imports a provider module
until a caller actually asks for one (so `ninja cabs list` doesn't pay the
cost of every installed provider unless something calls `list_cabs`).
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING

from shinobi.exceptions import CabLoadError

if TYPE_CHECKING:
    from shinobi.steps.schema import Cab, StepRef

_GROUP = "shinobi.cabs"


def _provider_entry_points() -> list[EntryPoint]:
    return sorted(entry_points(group=_GROUP), key=lambda ep: ep.name)


def get(name: str) -> "Cab | StepRef":
    """Resolve a cab by name, trying every installed `shinobi.cabs`
    provider in name order. The first provider whose own `get(name)`
    doesn't raise `KeyError` wins.
    """
    providers = _provider_entry_points()
    for ep in providers:
        module = ep.load()
        try:
            return module.get(name)
        except KeyError:
            continue
    installed = ", ".join(ep.name for ep in providers) or "none installed"
    raise CabLoadError(f"no such cab {name!r} in any shinobi.cabs provider ({installed})")


def list_cabs() -> dict[str, list[str]]:
    """`{provider_name: [cab_name, ...]}` across every installed provider."""
    return {ep.name: sorted(ep.load().list_cabs()) for ep in _provider_entry_points()}
