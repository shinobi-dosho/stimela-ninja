"""Cab schema: the typed, declarative description of an atomic task.

A cab does not know how it will be executed (native process, container,
Slurm job, ...) -- that's the backend's job -- and it does not know what
recipe it's being called from -- recipes are just Python. All a cab
describes is: what are its parameters, and how do those parameters turn
into a command line.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParamSchema(BaseModel):
    """Schema for a single cab input or output."""

    dtype: str = "str"
    required: bool = False
    default: Any = None
    info: str | None = None
    # a value that is always set by the cab itself, not user-supplied
    implicit: Any = None
    # the name the underlying tool actually expects, if different from
    # the schema's (Python-friendly) key -- e.g. schema key "ms" mapping
    # to the tool's "--vis" flag
    nom_de_guerre: str | None = None


class Policies(BaseModel):
    """How a cab's parameters are turned into command-line arguments."""

    prefix: str = "--"
    # character substitutions applied to parameter names, e.g. {"_": "-"}
    replace: dict[str, str] = Field(default_factory=dict)
    # how list-valued parameters are joined into a single argument value
    list_sep: str = ","
    # if True, repeat the flag once per list item instead of joining
    repeat_list: bool = False

    def arg_name(self, name: str) -> str:
        for old, new in self.replace.items():
            name = name.replace(old, new)
        return f"{self.prefix}{name}"


class CabDef(BaseModel):
    """A cab definition: an atomic, backend-agnostic task."""

    name: str
    command: str
    info: str | None = None
    image: str | None = None
    flavour: str = "binary"
    policies: Policies = Field(default_factory=Policies)
    inputs: dict[str, ParamSchema] = Field(default_factory=dict)
    outputs: dict[str, ParamSchema] = Field(default_factory=dict)
    # regex -> list of wrangler action strings, e.g.
    # {"Flagged: (?P<percentage>[\\d.]+)%": ["PARSE_OUTPUT:percentage:float"]}
    wranglers: dict[str, list[str]] = Field(default_factory=dict)

    def param_name(self, schema_name: str, schema: ParamSchema) -> str:
        return schema.nom_de_guerre or schema_name


class RecipeInfo(BaseModel):
    """Metadata for a @recipe-decorated function: name, docstring, and its
    derived input schema. Exists purely so tooling (e.g. the `ninja run`
    CLI) can build --options from a recipe's signature without calling it.

    Unlike CabDef, there's no command/image/policies/outputs/wranglers --
    a recipe's body is its own orchestration logic and manages whatever
    backend/execution it needs itself; it has no single command of its own.
    """

    name: str
    info: str | None = None
    inputs: dict[str, ParamSchema] = Field(default_factory=dict)


_FILE_LIKE_MARKERS = ("file", "ms")


def is_file_like_dtype(dtype: str) -> bool:
    """True if a dtype string (e.g. "File", "MS", "list:File") looks like
    it refers to a filesystem path. Shared by the container/kubernetes
    backends (to decide what to bind-mount) and the CLI (to map a param
    to a click.Path() option).
    """
    dtype_lower = dtype.lower()
    return any(marker in dtype_lower for marker in _FILE_LIKE_MARKERS)
