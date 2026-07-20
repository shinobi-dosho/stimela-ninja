"""Shared helpers for turning a loader's flat parameter specs into a
pydantic model class (the `inputs_model`/`outputs_model` a `Cab` needs),
plus the generic YAML-composition primitives (`_include`/`_use` deep-merge)
every loader dialect builds its own resolution order on top of.

Cab dtypes are strings (cult-cargo/stimela-classic convention): scalar
names (`str`/`int`/`float`/`bool`), file-like names (`File`/`MS`/
`Directory`/`URI`, all mapped to `pathlib.Path` so `path_fields` picks
them up for bind-mounting), `list:<inner>` (cult-cargo/classic colon
syntax) and `List[<inner>]` (newer bracket syntax seen in both newer
cult-cargo cabs and caracal2's scabha-dialect config schemas) for lists,
and `Tuple[<a>, <b>, ...]`/`Union[<a>, <b>, ...]` (both bracket syntax,
nestable inside each other and inside `List[...]`) for tuples and unions.
"""

from __future__ import annotations

import functools
import keyword
import operator
import re
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import ConfigDict, Field, create_model


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
        raise ValueError(f"parameter names {seen[field]!r} and {name!r} both sanitize to {field!r} -- rename one to avoid a silent collision")
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
_TUPLE_RE = re.compile(r"^tuple\[(?P<inner>.+)\]$", re.IGNORECASE)
_UNION_RE = re.compile(r"^union\[(?P<inner>.+)\]$", re.IGNORECASE)


def _split_top_level(spec: str) -> list[str]:
    """Split a bracket-inner spec (`"int, int"`, `"str, List[int]"`) on
    top-level commas only -- commas nested inside a `[...]` (e.g. inside a
    `List[...]`/`Tuple[...]`/`Union[...]` argument) don't split.
    """
    parts = []
    depth = 0
    current: list[str] = []
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def dtype_to_type(dtype: str) -> Any:
    """Map a cab dtype string to a Python type. File-like dtypes become
    `pathlib.Path`; `list:<inner>` or `List[<inner>]` becomes
    `list[<inner>]`; `Tuple[<a>, <b>, ...]` becomes `tuple[<a>, <b>, ...]`;
    `Union[<a>, <b>, ...]` becomes `<a> | <b> | ...`; anything unrecognised
    falls back to `str`.
    """
    dtype = str(dtype).strip()
    lower = dtype.lower()
    if lower.startswith("list:"):
        return list[dtype_to_type(dtype[5:])]
    if m := _BRACKET_LIST_RE.match(dtype):
        return list[dtype_to_type(m.group("inner"))]
    if m := _TUPLE_RE.match(dtype):
        items = tuple(dtype_to_type(p) for p in _split_top_level(m.group("inner")))
        return tuple[items] if items else tuple
    if m := _UNION_RE.match(dtype):
        items = [dtype_to_type(p) for p in _split_top_level(m.group("inner"))]
        if items:
            return functools.reduce(operator.or_, items)
        return str
    if is_file_dtype(dtype):
        return Path
    return _SCALAR_TYPES.get(lower, str)


def validate_choices(choices: Any, *, error: type[Exception]) -> list[Any] | None:
    """Normalise a raw `choices:` value to a plain list, or `None` if it
    wasn't given -- the one place both `cultcargo._collect` and
    `worker_schema._leaf_field` check the "must be a list" shape, so the
    error message and the shape rule can't drift between the two scabha-
    dialect loaders.
    """
    if not choices:
        return None
    if not isinstance(choices, (list, tuple)):
        raise error(f"'choices' must be a list, got {choices!r}")
    return list(choices)


def narrow_choices(py_type: Any, choices: list[Any] | None) -> Any:
    """Narrow `py_type` to `typing.Literal[*choices]` when `choices` is a
    non-empty list -- shinobi's one enum-like schema mechanism (shared by
    `build_model` below and `worker_schema._leaf_field`), so an out-of-set
    value fails real pydantic validation instead of only being documented
    in a field's `info` text.
    """
    if not choices:
        return py_type
    return Literal[tuple(choices)]


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
    choices: dict[str, list[Any]] | None = None,
    extras: dict[str, dict[str, Any]] | None = None,
) -> type:
    """Create a pydantic model class named `name`.

    `fields` maps a field name to `(dtype, required, default)`. See
    `required_field_spec` for the required/default rule applied to each.
    `choices` maps a field name to its allowed values (see
    `narrow_choices`) -- omitted or absent for a field means its plain
    `dtype`-derived type applies unchanged. `extras` maps a field name to a
    `json_schema_extra` dict carried onto that field (e.g. `abbreviation`
    for the CLI); a field absent from `extras` gets none. Mirrors what
    `worker_schema._leaf_field` builds per field, so both scabha-dialect
    loaders attach field-level hints the same way.
    """
    choices = choices or {}
    extras = extras or {}

    def _spec(field_name: str, dtype: str, required: bool, default: Any) -> tuple[Any, Any]:
        annotation, field_default = required_field_spec(narrow_choices(dtype_to_type(dtype), choices.get(field_name)), required, default)
        extra = extras.get(field_name)
        if extra:
            return (annotation, Field(field_default, json_schema_extra=extra))
        return (annotation, field_default)

    definitions: dict[str, tuple[Any, Any]] = {field_name: _spec(field_name, dtype, required, default) for field_name, (dtype, required, default) in fields.items()}
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


def resolve_use(node: Any, root: dict[str, Any], *, error: type[Exception]) -> Any:
    """Resolve every `_use: dotted.path` directive in `node` (a `_use`
    target that itself has a `_use` resolves too, recursively) by dotted
    lookup against `root` (the fully `_include`-resolved document) --
    `cultcargo`'s and `worker_schema`'s dialects agree on this directive
    (deep-merge the target in, with the dict's own sibling keys winning),
    differing only in which exception type reports a bad dotted path.
    """

    def entry_to_dict(dotted: str) -> Any:
        """Resolve the `_use` target at `dotted`, recursing into its own `_use`.

        Args:
            dotted: Dotted path into `root` naming the `_use` target.

        Returns:
            The target node with its own `_use` directives resolved.
        """
        return resolve_directive(get_path(root, dotted, error=error), "_use", entry_to_dict)

    return resolve_directive(node, "_use", entry_to_dict)


# Leaf-descriptor keys both scabha-dialect loaders (`cultcargo`, a *cab*
# schema, and `worker_schema`, a *config* schema) agree mean "this is a
# leaf parameter, not a nested group" -- a loader that recognises extra
# keys of its own (e.g. cultcargo's `nom_de_guerre`/`mkdir`) extends this
# set rather than re-listing the shared ones.
COMMON_LEAF_KEYS = {
    "info",
    "dtype",
    "default",
    "required",
    "implicit",
    "choices",
    "abbreviation",
    "policies",
    "writable",
    "must_exist",
    "path_policies",
}


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
