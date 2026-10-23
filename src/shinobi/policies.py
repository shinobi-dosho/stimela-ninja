"""Compatibility adapter: normalize framework products, then format in Dosho."""

from __future__ import annotations
from dosho.validation.policies import build_argv as _build_argv, EXECUTABLE_FLAVOURS
from dosho.validation.exceptions import UnsupportedFlavourError as DoshoFlavourError
from shinobi.exceptions import UnsupportedFlavourError
from shinobi.products import product_argv_value


def build_argv(cab, resolved):
    implicit = {name: meta.model_copy(update={"implicit": product_argv_value(meta.implicit)}) for name, meta in cab.field_meta.items()}
    cab = cab.model_copy(update={"field_meta": implicit})
    try:
        return _build_argv(cab, {name: product_argv_value(value) for name, value in resolved.items()})
    except DoshoFlavourError as exc:
        raise UnsupportedFlavourError(str(exc)) from exc


__all__ = ["build_argv", "EXECUTABLE_FLAVOURS"]
