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
from typing import Any, Literal, Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from shinobi.steps.schema import _unwrap_annotation


def is_list(annotation) -> bool:
    """Check whether a type annotation names a list/tuple type.

    Args:
        annotation: A type annotation, possibly wrapped in `Optional`/`Union`.

    Returns:
        True if the annotation (or any of its `Union` arms) is `list` or
        `tuple`.
    """
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


def _literal_choices(annotation) -> tuple[Any, ...] | None:
    """The allowed values of a `typing.Literal` named by `annotation`
    (unwrapping `Optional`/`Union`), or `None` if it names no `Literal`. A
    cab's `choices:` key narrows the field's annotation to
    `Literal[*choices]` (see `loaders._modelgen.narrow_choices`), so this is
    how `click_type` recovers those values to build a `click.Choice`.
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin is Union or origin is types.UnionType:
        for arg in get_args(annotation):
            found = _literal_choices(arg)
            if found is not None:
                return found
    return None


def click_type(annotation, is_path: bool):
    """Pick the `click` parameter type for a field's annotation.

    Args:
        annotation: The field's type annotation.
        is_path: Whether the annotation is path-like (as determined by
            `_is_path_annotation`); takes priority over the leaf type.

    Returns:
        A `click.Path()` if `is_path`; a `click.Choice` if the annotation
        is a `typing.Literal` (a cab's `choices:` -- so an out-of-set value
        is rejected by click itself, with the allowed values listed in
        `--help` and the error); otherwise the `click` type matching the
        annotation's leaf type (`click.STRING` as fallback). Choice values
        are stringified: every real cab `choices:` list is strings, and the
        model's own `Literal` still validates the coerced value.
    """
    if is_path:
        return click.Path()
    choices = _literal_choices(annotation)
    if choices is not None:
        return click.Choice([str(c) for c in choices])
    for leaf in _unwrap_annotation(annotation):
        if leaf in (int, float, bool, str):
            return {int: click.INT, float: click.FLOAT, bool: click.BOOL, str: click.STRING}[leaf]
    return click.STRING


def option_flag(field_name: str) -> str:
    """Build a `--flag-name` click option string from a flat field name.

    Args:
        field_name: Flat, underscore-joined field name (e.g. `"foo_bar"`).

    Returns:
        The corresponding `--foo-bar` flag string.
    """
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
    """Turn a pydantic model's (possibly nested) fields into click options.

    Args:
        model: A pydantic `BaseModel` subclass; nested `BaseModel` fields are
            flattened via `iter_leaf_fields`.

    Returns:
        A list of `click.Option` instances, one per leaf field. Boolean
        fields become `--flag/--no-flag` options; list/tuple fields become
        `multiple=True` options; a field carrying an `abbreviation` on its
        `json_schema_extra` (a cab's `abbreviation:` key, threaded by the
        loaders) also gets a `-<abbrev>` short alias. click always derives
        the callback kwarg name from the long flag, so the short alias
        never affects the round-trip to `flat_name`.
    """
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
        options.append(click.Option([flag, *_abbreviation_opts(field)], **kwargs))
    return options


def _abbreviation_opts(field: FieldInfo) -> list[str]:
    """`["-<abbrev>"]` if `field` carries an `abbreviation` on its
    `json_schema_extra` (a cab's `abbreviation:` key), else `[]`. A
    secondary short-option alias for the field's long flag; multi-character
    single-dash names (`-as`, `-sublist`) are fine -- click matches the
    whole token, only rejecting glued forms like `-asVALUE`.
    """
    extra = field.json_schema_extra
    if isinstance(extra, dict) and extra.get("abbreviation"):
        return [f"-{extra['abbreviation']}"]
    return []


def unflatten_kwargs(model: type[BaseModel], flat_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The inverse of `build_options`' flattening: turn flat
    `--parent-child`-style kwargs back into the nested dict
    `model(**nested)` expects (pydantic coerces a plain nested dict into
    its submodel automatically). A key absent, `None`, or an empty tuple
    in `flat_kwargs` (the user didn't pass that option) is omitted
    entirely, so the model's/submodel's own default applies instead of an
    explicit `None`.
    """
    nested: dict[str, Any] = {}
    for flat_name, path, _field in iter_leaf_fields(model):
        value = flat_kwargs.get(flat_name)
        # click renders every list/tuple field as a `multiple=True` option,
        # which defaults to `()` when unset; treat that empty tuple like an
        # absent option so an optional non-list field (e.g. `Tuple[int, int]`
        # or `Union[str, Tuple[str, float]]`) falls back to its own default
        # instead of being handed an invalid `()`.
        if value is None or value == ():
            continue
        node = nested
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = flat_kwargs[flat_name]
    return nested
