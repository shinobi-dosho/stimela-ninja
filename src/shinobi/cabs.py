"""Resolve a `Cab`/`StepRef` by name across installed cab-provider packages.

shinobi ships no cabs itself. A cab-provider package (e.g. `dosho`, the
native shinobi cab repository) registers itself under the `shinobi.cabs`
packaging entry-point group in its own `pyproject.toml`:

    [project.entry-points."shinobi.cabs"]
    dosho = "dosho.registry"

The entry point's target is a module (or any object) exposing
`list_cabs() -> list[str]` plus at least one resolver:

* `get_document(name) -> (dialect, text)` -- hand back the cab's *definition*
  and let shinobi build it. Preferred, and tried first. A provider that only
  ever returns documents needs no dependency on shinobi at all: it reads a
  file and returns bytes, which is what lets a cab repository ship as data
  rather than as Python (see dosho's `docs/design_data_registry.md`).
* `get(name) -> Cab | StepRef` -- hand back a live object, which requires the
  provider to import shinobi and construct it. Still fully supported; it is
  the only shape that can express a pystep, whose body is real code.

Either raises `KeyError` when `name` isn't one of its cabs, which is how the
search falls through to the next provider. A provider may expose both: the
document path is tried first, so a mixed provider can serve most cabs as data
and the rest as objects. A provider entry can
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


# dialect tag -> the loader entry that turns one document into cabs. Both
# existing loaders already expose `loads(text)`, so this is a dispatch table
# and not a new parsing layer.
_DIALECTS = ("cultcargo", "stimela-classic")


def build_document(dialect: str, text: str, name: str | None = None) -> "Cab | StepRef":
    """Build a cab from its definition text, in the named dialect.

    This is the half of the provider protocol shinobi owns: a provider that
    returns documents does not parse them, so it needs neither a parser nor
    shinobi itself.

    Args:
        dialect: One of `_DIALECTS`.
        text: The definition's raw text.
        name: Which cab to take, for dialects whose document can hold several
            (cult-cargo). Optional when the document defines exactly one.

    Raises:
        CabLoadError: On an unknown dialect, or when `name` does not name a cab
            the document defines.
    """
    if dialect == "cultcargo":
        from shinobi.loaders import cultcargo

        cabs = cultcargo.loads(text)
        # A requested name must be one the document defines. Falling back to
        # "it only defines one, so that must be it" would turn a provider
        # returning the wrong document into a silently wrong cab, under the
        # name the caller asked for. Providers alias freely -- dosho registers
        # its `simms_classic` attribute as `simms` -- but they alias to the
        # cab's *own* name, so a mismatch here is a bug, not a convention.
        if name is not None:
            if name in cabs:
                return cabs[name]
            raise CabLoadError(f"cult-cargo document defines {sorted(cabs)!r}, not {name!r}")
        if len(cabs) == 1:
            return next(iter(cabs.values()))
        raise CabLoadError(f"cult-cargo document defines {sorted(cabs)!r}; no name was given to choose between them")
    if dialect == "stimela-classic":
        from shinobi.loaders import stimela_classic

        return stimela_classic.loads(text)
    raise CabLoadError(f"unknown cab dialect {dialect!r} (known: {', '.join(_DIALECTS)})")


def get(name: str) -> "Cab | StepRef":
    """Resolve a cab by name, trying every installed `shinobi.cabs`
    provider in name order. Within a provider, `get_document` is tried
    before `get`; the first resolver that doesn't raise `KeyError` wins.
    """
    providers = _provider_entry_points()
    for ep in providers:
        module = ep.load()
        for resolver in ("get_document", "get"):
            fn = getattr(module, resolver, None)
            if fn is None:
                continue
            try:
                found = fn(name)
            except KeyError:
                continue
            return build_document(*found, name=name) if resolver == "get_document" else found
    installed = ", ".join(ep.name for ep in providers) or "none installed"
    raise CabLoadError(f"no such cab {name!r} in any shinobi.cabs provider ({installed})")


def list_cabs() -> dict[str, list[str]]:
    """`{provider_name: [cab_name, ...]}` across every installed provider."""
    return {ep.name: sorted(ep.load().list_cabs()) for ep in _provider_entry_points()}
