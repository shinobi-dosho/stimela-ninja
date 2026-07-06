"""Turn an arbitrary pydantic model's fields into `click.Option`s at
runtime.

Not tied to `Cab`/`Recipe`/`Scope`: `build_options` only needs
`model.model_fields`, so it works for any pydantic `BaseModel` -- e.g.
`ninja run <target>` uses it for a Cab/Recipe/StepRef's `inputs_model`
(see `shinobi.cli`), and a downstream project's own CLI can reuse it the
same way for an unrelated config schema's `inputs_model` (e.g.
`shinobi.loaders.worker_schema.ConfigSchema`) instead of writing a second
click-option-builder.

A nested `BaseModel` field (a config *group*, as
`shinobi.loaders.worker_schema` produces for e.g. `obsinfo.plotelev.enable`
-- never seen in a cult-cargo cab's flat `inputs_model`, so this is a pure
extension, not a behaviour change for existing callers) is recursed into
and flattened to a single dotted-by-underscore option
(`--obsinfo-plotelev-enable`). `unflatten_kwargs` is the inverse: turn
`build_options`'s flat kwargs back into the nested dict
`model(**nested)` expects.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from shinobi.steps.schema import _unwrap_annotation


def is_list(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(is_list(arg) for arg in get_args(annotation))
    return origin in (list, tuple)


def _submodel(annotation) -> type[BaseModel] | None:
    """The `BaseModel` subclass an annotation names -- itself, or (for
    symmetry with leaf fields, though `worker_schema` never wraps a group
    field this way) inside an `Optional`/`Union` -- or `None` if it isn't
    one.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _is_path_annotation(annotation) -> bool:
    return any(isinstance(leaf, type) and issubclass(leaf, Path) for leaf in _unwrap_annotation(annotation))


def click_type(annotation, is_path: bool):
    if is_path:
        return click.Path()
    for leaf in _unwrap_annotation(annotation):
        if leaf in (int, float, bool, str):
            return {int: click.INT, float: click.FLOAT, bool: click.BOOL, str: click.STRING}[leaf]
    return click.STRING


def option_flag(field_name: str) -> str:
    # ONLY a straight "_" -> "-" replace: click derives the callback kwarg
    # name from this flag string, and it must round-trip back to the exact
    # flat name used here and in unflatten_kwargs.
    return "--" + field_name.replace("_", "-")


def bool_option_flag(field_name: str) -> str:
    """`--flag/--no-flag` form, so a boolean field defaulting `True` can
    still be explicitly set `False` from the CLI -- a bare `is_flag=True`
    option (the plain `option_flag` form) can only ever turn a flag *on*,
    never override a `True` default back to `False`. Click infers the same
    callback kwarg name from the primary (`--flag`) branch, so this still
    round-trips to `field_name` exactly like `option_flag`.
    """
    flag = field_name.replace("_", "-")
    return f"--{flag}/--no-{flag}"


def iter_leaf_fields(
    model: type[BaseModel], *, _prefix: str = "", _path: tuple[str, ...] = ()
) -> list[tuple[str, tuple[str, ...], FieldInfo]]:
    """`(flat_name, path, field)` for every leaf (non-`BaseModel`) field in
    `model`, recursing into nested `BaseModel` fields and flattening names
    with `_` -- e.g. `obsinfo.plotelev.enable` yields flat_name
    `"obsinfo_plotelev_enable"`, path `("obsinfo", "plotelev", "enable")`.
    A model with no nested `BaseModel` fields (every cult-cargo cab's
    `inputs_model`) yields exactly what a flat single-level walk would.
    """
    result: list[tuple[str, tuple[str, ...], FieldInfo]] = []
    for name, field in model.model_fields.items():
        sub = _submodel(field.annotation)
        if sub is not None:
            result.extend(iter_leaf_fields(sub, _prefix=f"{_prefix}{name}_", _path=(*_path, name)))
        else:
            result.append((f"{_prefix}{name}", (*_path, name), field))
    return result


def build_options(model: type[BaseModel]) -> list[click.Option]:
    options = []
    for flat_name, _path, field in iter_leaf_fields(model):
        required = field.is_required()
        default = None if field.default is PydanticUndefined else field.default
        kwargs: dict = {"required": required, "help": field.description}
        leaves = _unwrap_annotation(field.annotation)
        field_is_list = is_list(field.annotation)
        if bool in leaves and not field_is_list:
            kwargs.update(is_flag=True, default=bool(default))
            flag = bool_option_flag(flat_name)
        else:
            if default is not None:
                kwargs["default"] = default
            kwargs["type"] = click_type(field.annotation, _is_path_annotation(field.annotation))
            if field_is_list:
                kwargs["multiple"] = True
            flag = option_flag(flat_name)
        options.append(click.Option([flag], **kwargs))
    return options


def unflatten_kwargs(model: type[BaseModel], flat_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The inverse of `build_options`' flattening: turn flat
    `--parent-child`-style kwargs back into the nested dict
    `model(**nested)` expects (pydantic coerces a plain nested dict into
    its submodel automatically). A key absent or `None` in `flat_kwargs`
    (the user didn't pass that option) is omitted entirely, so the
    model's/submodel's own default applies instead of an explicit `None`.
    """
    nested: dict[str, Any] = {}
    for flat_name, path, _field in iter_leaf_fields(model):
        if flat_kwargs.get(flat_name) is None:
            continue
        node = nested
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = flat_kwargs[flat_name]
    return nested
