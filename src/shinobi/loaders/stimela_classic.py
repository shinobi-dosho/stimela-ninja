"""Compatibility loader for Stimela classic JSON cab definitions."""

from __future__ import annotations
from pathlib import Path
from dosho.validation import classic
from dosho.validation.exceptions import CabLoadError as DoshoLoadError
from shinobi.exceptions import CabLoadError
from shinobi.loaders._modelgen import TYPE_BINDINGS
from shinobi.loaders.yaml_cab import adapt_cab


def loads(text: str):
    try:
        return adapt_cab(classic.loads(text, type_bindings=TYPE_BINDINGS))
    except DoshoLoadError as exc:
        raise CabLoadError(str(exc)) from exc


def load_file(path: str | Path):
    return loads(Path(path).read_text())
