"""Shared helpers for turning a loader's flat parameter specs into a
pydantic model class (the `inputs_model`/`outputs_model` a `Cab` needs).

Cab dtypes are strings (cult-cargo/stimela-classic convention): scalar
names (`str`/`int`/`float`/`bool`), file-like names (`File`/`MS`/
`Directory`/`URI`, all mapped to `pathlib.Path` so `path_fields` picks
them up for bind-mounting), and `list:<inner>` for lists.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, create_model


def sanitize(name: str) -> str:
    """Turn a cab parameter name into a valid Python identifier (pydantic
    field names must be identifiers). Non-identifier characters -- hyphens,
    dots, etc. common in cult-cargo/classic param names -- become
    underscores; a leading digit or a keyword is prefixed. The loader keeps
    the original name as a ``nom_de_guerre`` so the built argv still uses it.
    """
    cleaned = re.sub(r"\W", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def sanitize_unique(name: str, seen: dict[str, str]) -> str:
    """Like `sanitize`, but raises if two distinct raw names collide on the
    same sanitized identifier. `seen` maps a sanitized field name to the
    first raw name that produced it -- share one `seen` dict across a
    single cab's parameter list.
    """
    field = sanitize(name)
    if field in seen and seen[field] != name:
        raise ValueError(
            f"parameter names {seen[field]!r} and {name!r} both sanitize to "
            f"{field!r} -- rename one to avoid a silent collision"
        )
    seen[field] = name
    return field


_SCALAR_TYPES: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "bool": bool,
    "boolean": bool,
}

_FILE_LIKE = {"file", "ms", "directory", "dir", "uri", "url"}


def is_file_dtype(dtype: str) -> bool:
    """Whether a cab dtype string (e.g. from `ParamMeta.dtype`) is
    file-like. The single source of truth for that classification, shared
    with `dtype_to_type` below and with backends that need to recognise a
    dynamically pattern-matched param (no declared field/type annotation
    for `path_fields` to inspect) as needing a bind mount.
    """
    return str(dtype).strip().lower() in _FILE_LIKE


def dtype_to_type(dtype: str) -> Any:
    """Map a cab dtype string to a Python type. File-like dtypes become
    `pathlib.Path`; `list:<inner>` becomes `list[<inner>]`; anything
    unrecognised falls back to `str`.
    """
    dtype = str(dtype).strip()
    lower = dtype.lower()
    if lower.startswith("list:"):
        return list[dtype_to_type(dtype[5:])]
    if is_file_dtype(dtype):
        return Path
    return _SCALAR_TYPES.get(lower, str)


def build_model(
    name: str,
    fields: dict[str, tuple[str, bool, Any]],
    *,
    allow_extra: bool = False,
) -> type:
    """Create a pydantic model class named `name`.

    `fields` maps a field name to `(dtype, required, default)`. A required
    field with no default is `...`; everything else is Optional with its
    default (or None), so callers can omit it.
    """
    definitions: dict[str, tuple[Any, Any]] = {}
    for field_name, (dtype, required, default) in fields.items():
        py_type = dtype_to_type(dtype)
        if required and default is None:
            definitions[field_name] = (py_type, ...)
        else:
            definitions[field_name] = (py_type | None, default)

    config = ConfigDict(extra="allow") if allow_extra else None
    return create_model(name, __config__=config, **definitions)
