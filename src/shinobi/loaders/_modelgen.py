"""Shared helpers for turning a loader's flat parameter specs into a
pydantic model class (the `inputs_model`/`outputs_model` a `Cab` needs),
plus the generic YAML-composition primitives (`_include`/`_use` deep-merge)
every loader dialect builds its own resolution order on top of.

Cab dtypes are strings (cult-cargo/stimela-classic convention): scalar
names (`str`/`int`/`float`/`bool`), file-like names (`File`/`MS`/
`Directory`/`URI`, all mapped to `pathlib.Path` so `path_fields` picks
them up for bind-mounting), `list:<inner>` (cult-cargo/classic colon
syntax) and `List[<inner>]` (newer bracket syntax seen in both newer
cult-cargo cabs and caracal2's scabha-dialect config schemas) for lists.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Any, Callable

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


_BRACKET_LIST_RE = re.compile(r"^list\[(?P<inner>.+)\]$", re.IGNORECASE)


def dtype_to_type(dtype: str) -> Any:
    """Map a cab dtype string to a Python type. File-like dtypes become
    `pathlib.Path`; `list:<inner>` or `List[<inner>]` becomes
    `list[<inner>]`; anything unrecognised falls back to `str`.
    """
    dtype = str(dtype).strip()
    lower = dtype.lower()
    if lower.startswith("list:"):
        return list[dtype_to_type(dtype[5:])]
    if m := _BRACKET_LIST_RE.match(dtype):
        return list[dtype_to_type(m.group("inner"))]
    if is_file_dtype(dtype):
        return Path
    return _SCALAR_TYPES.get(lower, str)


def required_field_spec(py_type: Any, required: bool, default: Any) -> tuple[Any, Any]:
    """`(annotation, default)` for a `pydantic.create_model`/`Field` slot: a
    required field with no default is `(py_type, ...)`; everything else is
    Optional with its default (or None), so callers can omit it. The single
    source of truth for this rule -- shared by `build_model` below and by
    any loader (e.g. `worker_schema._leaf_field`) building one field at a
    time instead of a whole model in one call.
    """
    if required and default is None:
        return (py_type, ...)
    return (py_type | None, default)


def build_model(
    name: str,
    fields: dict[str, tuple[str, bool, Any]],
    *,
    allow_extra: bool = False,
) -> type:
    """Create a pydantic model class named `name`.

    `fields` maps a field name to `(dtype, required, default)`. See
    `required_field_spec` for the required/default rule applied to each.
    """
    definitions: dict[str, tuple[Any, Any]] = {
        field_name: required_field_spec(dtype_to_type(dtype), required, default)
        for field_name, (dtype, required, default) in fields.items()
    }
    config = ConfigDict(extra="allow") if allow_extra else None
    return create_model(name, __config__=config, **definitions)


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge two YAML-parsed values: dict keys are merged
    key-by-key (recursing into nested dicts), with `override`'s value
    winning on any key present in both; any other type just takes
    `override` outright. Shared by every loader's `_include`/`_use`
    composition (cult-cargo's "including/using file's own keys win"
    convention).
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


def get_path(root: dict[str, Any], dotted: str, *, error: type[Exception]) -> Any:
    """Look up a dotted path (`a.b.c`) in a nested dict, raising `error`
    with a clear message if any segment is missing.
    """
    node: Any = root
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise error(f"path '{dotted}' not found (stuck at '{part}')")
        node = node[part]
    return node


def resolve_directive(node: Any, key: str, entry_to_dict: Callable[[Any], Any]) -> Any:
    """Walk a nested dict/list structure. Wherever a dict has `key`, pop it
    (a single entry, or a list of entries), turn each entry into a fully
    resolved dict via `entry_to_dict`, `deep_merge` them together in order,
    then `deep_merge` the result under the dict's own remaining keys (which
    win on any conflict). This is the shared "resolve a composition
    directive" shape behind both `_include` and `_use` in every loader
    dialect -- `entry_to_dict` is responsible for whatever an entry means
    (a dotted lookup, a file path, ...) and for any further recursion its
    own result needs; this walker only handles the tree traversal and merge.
    """
    if isinstance(node, list):
        return [resolve_directive(item, key, entry_to_dict) for item in node]
    if not isinstance(node, dict):
        return node

    node = {k: resolve_directive(v, key, entry_to_dict) for k, v in node.items()}
    if key in node:
        spec = node.pop(key)
        entries = spec if isinstance(spec, list) else [spec]
        merged: dict[str, Any] = {}
        for entry in entries:
            merged = deep_merge(merged, entry_to_dict(entry))
        node = deep_merge(merged, node)
    return node
