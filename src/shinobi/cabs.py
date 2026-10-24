"""Resolve providers and adapt neutral definitions to executable steps.

Providers register under ``shinobi.cabs`` and expose ``list_cabs`` plus
``get_document(name) -> (dialect, text)`` and/or ``get(name)``. Documents are
tried first; optional ``loader_options()`` supplies image mappings and explicit
filesystem roots. Dosho owns static parsing and name selection.

``get`` may return a neutral ``CabSpec``, an original ``@dosho.pystep`` function,
or an existing ``Scope``/``StepRef``. ``adapt_definition`` is the common boundary
used by providers, recipe binding and CLI targets. It binds trusted strict
types, checks runtime contracts and wraps only explicitly marked functions.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any

from shinobi.exceptions import CabLoadError

if TYPE_CHECKING:
    from shinobi.steps.schema import Cab, StepRef

_GROUP = "shinobi.cabs"


def _provider_entry_points() -> list[EntryPoint]:
    return sorted(entry_points(group=_GROUP), key=lambda ep: ep.name)


# dialect tag -> the loader entry that turns one document into cabs. Both
# existing loaders already expose `loads(text)`, so this is a dispatch table
# and not a new parsing layer. The tags name *dialects*, not the projects that
# publish files in them -- `yaml_cab` is scabha's cab dialect, of which
# cult-cargo is the largest published library.
_DIALECTS = ("yaml_cab", "stimela-classic", "dosho-cab-v1")


def build_document(dialect: str, text: str, name: str | None = None, **options: Any) -> "Cab | StepRef":
    """Compatibility entry: let Dosho select a definition, then adapt it.

    ``options`` are passed to Dosho's dialect loader. A requested name must
    match a definition in the document, including classic JSON definitions.
    """
    try:
        from dosho.validation import build_document as build_definition
    except ImportError as exc:
        raise ImportError("This Shinobi source requires matching Dosho sources with neutral definition support") from exc
    from dosho.validation.exceptions import CabLoadError as DoshoLoadError

    try:
        return adapt_definition(build_definition(dialect, text, name=name, **options))
    except DoshoLoadError as exc:
        raise CabLoadError(str(exc)) from exc


def adapt_definition(definition):
    """Bind a neutral definition, or preserve an existing executable scope.

    Only functions explicitly marked by Dosho are accepted. Plain callables
    remain ordinary Python and are never guessed to be workflow steps.
    """
    from dosho.validation import CabSpec
    from shinobi.steps.schema import Scope, StepRef

    if isinstance(definition, (Scope, StepRef)):
        return definition
    if isinstance(definition, CabSpec):
        from shinobi.loaders.yaml_cab import _compile

        return _compile(definition)
    try:
        from dosho.validation.pyfunc import get_pystep_definition
    except ImportError as exc:
        raise ImportError("This Shinobi source requires matching Dosho sources with neutral function schema support") from exc
    declaration = get_pystep_definition(definition)
    if declaration is not None:
        from shinobi.steps.pyfunc import pystep

        return pystep(**declaration.options)(definition)
    raise TypeError(f"expected a Scope, StepRef, CabSpec or @dosho.pystep function, got {type(definition).__name__}")


def _loader_options(module: Any) -> dict[str, Any]:
    """A provider's `loader_options()`, or nothing if it has none.

    A document is not always self-contained. dosho's name their image
    symbolically (`image: WSCLEAN`) so a deployment's overrides still decide
    the reference at load time; another provider might use package-scoped
    includes. Neither is resolvable from the text alone, and shinobi has no
    manifest and no package map of its own -- the provider does.

    Optional, so a provider whose documents *are* self-contained implements
    nothing.
    """
    fn = getattr(module, "loader_options", None)
    return dict(fn()) if fn is not None else {}


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
            if resolver != "get_document":
                return adapt_definition(found)
            return build_document(*found, name=name, **_loader_options(module))
    installed = ", ".join(ep.name for ep in providers) or "none installed"
    raise CabLoadError(f"no such cab {name!r} in any shinobi.cabs provider ({installed})")


def list_cabs() -> dict[str, list[str]]:
    """`{provider_name: [cab_name, ...]}` across every installed provider."""
    return {ep.name: sorted(ep.load().list_cabs()) for ep in _provider_entry_points()}
