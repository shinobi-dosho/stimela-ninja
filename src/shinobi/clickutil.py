"""Turn an arbitrary pydantic model's fields into `click.Option`s at
runtime.

Not tied to `Cab`/`Recipe`/`Scope`: `build_options` only needs
`model.model_fields` and `path_fields(model)`, so it works for any pydantic
`BaseModel` -- e.g. `ninja run <target>` uses it for a Cab/Recipe/StepRef's
`inputs_model` (see `shinobi.cli`), and a downstream project's own CLI can
reuse it the same way for an unrelated config schema's `inputs_model`
(e.g. `shinobi.loaders.worker_schema.ConfigSchema`) instead of writing a
second click-option-builder.
"""

from __future__ import annotations

import types
from typing import Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from shinobi.steps.schema import _unwrap_annotation, path_fields


def is_list(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(is_list(arg) for arg in get_args(annotation))
    return origin in (list, tuple)


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
    # model field name used to dispatch.
    return "--" + field_name.replace("_", "-")


def build_options(model: type[BaseModel]) -> list[click.Option]:
    paths = path_fields(model)
    options = []
    for name, field in model.model_fields.items():
        required = field.is_required()
        default = None if field.default is PydanticUndefined else field.default
        kwargs: dict = {"required": required, "help": field.description}
        leaves = _unwrap_annotation(field.annotation)
        field_is_list = is_list(field.annotation)
        if bool in leaves and not field_is_list:
            kwargs.update(is_flag=True, default=bool(default))
        else:
            if default is not None:
                kwargs["default"] = default
            kwargs["type"] = click_type(field.annotation, name in paths)
            if field_is_list:
                kwargs["multiple"] = True
        options.append(click.Option([option_flag(name)], **kwargs))
    return options
