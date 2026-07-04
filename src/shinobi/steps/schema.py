"""Definition-layer schema for the step model.

`Scope` is the base definition (schema, metadata, backend config). `Cab`
and `Recipe` extend it -- an atomic command and a composite of wired
sub-steps respectively. `StepRef` is the binding layer: a named reference
to a Scope plus an optional orchestration function, wiring, and per-step
constants; it is what both `@shinobi.step` and `@recipe.step` return.

There is no global function registry and no separate `Step` class -- the
orchestration function travels on the StepRef itself (see the design
plan's D1/D5). Dispatch never mutates a Scope; `Recipe` is the one
subclass that is deliberately mutable, via its builder methods, before
first execution.
"""

from __future__ import annotations

import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


class Mutability(str, Enum):
    """Whether a step's input may be changed in place by the step's own
    orchestration function without that change propagating back to the
    caller's object.
    """

    IMMUTABLE = "immutable"  # default: deep-copied before the step body runs
    MUTABLE = "mutable"  # opt-in: passed by reference, in-place changes persist


class ParamMeta(BaseModel):
    """Per-field metadata a plain pydantic model can't express: the name
    the underlying tool actually expects (`nom_de_guerre`), a value always
    supplied by the cab itself rather than the caller (`implicit`),
    human-facing help (`info`), and the cab dtype string (`dtype`, e.g.
    "File"/"MS") for a `ParamPattern` attr -- since a dynamically-named
    param has no declared field/type annotation for `path_fields` to
    inspect, this is how backends know to bind-mount its directory.
    """

    nom_de_guerre: str | None = None
    implicit: Any = None
    info: str | None = None
    dtype: str | None = None


class Policies(BaseModel):
    """How a cab's parameters are turned into command-line arguments."""

    prefix: str = "--"
    replace: dict[str, str] = Field(default_factory=dict)
    list_sep: str = ","
    repeat_list: bool = False

    def arg_name(self, name: str) -> str:
        for old, new in self.replace.items():
            name = name.replace(old, new)
        return f"{self.prefix}{name}"


class ParamPattern(BaseModel):
    """A family of inputs whose names are ``<prefix><separator><attr>``,
    where `prefix` is any string (not enumerable at cab-authoring time)
    and `attr` must be one of `attrs`. See AGENTS.md for the motivating
    tools (QuartiCal's ``solver.terms=[K,G]``, cubical's ``g-time-int``).
    """

    separator: str = "."
    attrs: dict[str, ParamMeta] = Field(default_factory=dict)

    def matches(self, name: str) -> ParamMeta | None:
        # Checked against each declared attr as a candidate suffix, not
        # split on the *last* separator: an attr itself can contain the
        # separator character (e.g. "time-int" with separator "-"), which
        # a blind rpartition() would split inside the attr. Prefers the
        # longest matching attr, in case one is a suffix of another.
        best_attr: str | None = None
        for attr in self.attrs:
            suffix = f"{self.separator}{attr}"
            if (
                name.endswith(suffix)
                and len(name) > len(suffix)
                and (best_attr is None or len(attr) > len(best_attr))
            ):
                best_attr = attr
        return self.attrs.get(best_attr) if best_attr is not None else None


def _unwrap_annotation(annotation: Any) -> list[Any]:
    """Flatten an annotation into its concrete leaf types, unwrapping
    Optional/Union and list/tuple containers -- used by `path_fields`.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        leaves: list[Any] = []
        for arg in get_args(annotation):
            leaves.extend(_unwrap_annotation(arg))
        return leaves
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        return _unwrap_annotation(args[0]) if args else [annotation]
    return [annotation]


def path_fields(model: type[BaseModel]) -> set[str]:
    """Names of every field of `model` whose (Optional/list-unwrapped) type
    is a filesystem path (``pathlib.Path``). File-like cab dtypes
    (File/MS/Directory/URI) map to Path, so this drives both container
    bind-mounting and the CLI's ``click.Path()`` mapping.
    """
    result: set[str] = set()
    for name, field in model.model_fields.items():
        for leaf in _unwrap_annotation(field.annotation):
            if isinstance(leaf, type) and issubclass(leaf, Path):
                result.add(name)
                break
    return result


class Scope(BaseModel):
    """Definition: schema, metadata, backend config. Never carries
    inputs/outputs/func fields -- those live in ExecContext/StepRef.
    """

    name: str
    info: str | None = None
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]
    backend: str | None = None
    input_mutability: dict[str, Mutability] = Field(default_factory=dict)

    def __call__(self, *, backend: str | None = None, **kwargs: Any):
        """Bare execution -- no orchestration function."""
        from shinobi.steps.dispatch import _dispatch

        return _dispatch(self, None, backend=backend, **kwargs)

    def mutability_of(self, field: str) -> Mutability:
        return self.input_mutability.get(field, Mutability.IMMUTABLE)

    def with_backend(self, backend: str | None) -> "Scope":
        """A copy bound to `backend`, or `self` unchanged if `backend` is
        None. Shared by `@shinobi.step` and `Recipe.step`, which both bind
        a per-step backend override onto a Scope before wrapping it in a
        StepRef.
        """
        return self.model_copy(update={"backend": backend}) if backend else self


class Cab(Scope):
    """An atomic step backed by a single command."""

    command: str
    flavour: str = "binary"
    image: str | None = None
    policies: Policies = Field(default_factory=Policies)
    field_meta: dict[str, ParamMeta] = Field(default_factory=dict)
    input_patterns: list[ParamPattern] = Field(default_factory=list)
    # regex -> list of wrangler action strings
    wranglers: dict[str, list[str]] = Field(default_factory=dict)

    def param_name(self, field: str) -> str:
        meta = self.field_meta.get(field)
        return meta.nom_de_guerre if meta and meta.nom_de_guerre else field

    def match_pattern(self, name: str) -> ParamMeta | None:
        for pattern in self.input_patterns:
            meta = pattern.matches(name)
            if meta is not None:
                return meta
        return None


class InputRef(BaseModel):
    """Wiring source: this sub-step's input comes from the enclosing
    Recipe's own input field `field`.
    """

    field: str


class OutputRef(BaseModel):
    """Wiring source: this input (or, in `Recipe.output_wiring`, the
    recipe's own output) comes from step `step`'s output field `field`.
    """

    step: str
    field: str


class StepRef(BaseModel):
    """A named, executable binding of a Scope: orchestration function,
    wiring (meaningful only inside a Recipe), and per-step constants.
    Returned by `@shinobi.step` (free-standing) and `@recipe.step`
    (appended to `recipe.steps`). `arbitrary_types_allowed` is needed
    only for `func`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    step: "Cab | Recipe"
    func: Callable | None = None
    wiring: dict[str, "InputRef | OutputRef"] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, *, backend: str | None = None, **kwargs: Any):
        """Standalone execution. `params` are merged under caller kwargs;
        wiring is ignored (it can only be resolved inside a running
        Recipe), so any wired-only fields must be supplied as kwargs --
        input validation catches omissions.
        """
        from shinobi.steps.dispatch import _dispatch

        return _dispatch(self.step, self.func, backend=backend, **{**self.params, **kwargs})


class _InputsProxy:
    """`recipe.inputs.ms` or `recipe.inputs("ms")` -> InputRef(field="ms")."""

    def __init__(self, recipe: "Recipe"):
        self._recipe = recipe

    def __call__(self, field: str) -> InputRef:
        return self.__getattr__(field)

    def __getattr__(self, field: str) -> InputRef:
        if field not in self._recipe.inputs_model.model_fields:
            raise AttributeError(
                f"'{field}' is not a field of {self._recipe.inputs_model.__name__}"
            )
        return InputRef(field=field)


class _StepOutputsProxy:
    """Second level of `recipe.outputs.<step>.<field>` -- validates the
    field against the sub-step's `outputs_model`.
    """

    def __init__(self, step: str, outputs_model: type[BaseModel]):
        self._step = step
        self._outputs_model = outputs_model

    def __getattr__(self, field: str) -> OutputRef:
        if field not in self._outputs_model.model_fields:
            raise AttributeError(
                f"'{field}' is not an output of step '{self._step}' "
                f"({self._outputs_model.__name__})"
            )
        return OutputRef(step=self._step, field=field)


class _OutputsProxy:
    """`recipe.outputs.clean.output_ms` or `recipe.outputs("clean",
    "output_ms")` -> OutputRef(step="clean", field="output_ms").
    """

    def __init__(self, recipe: "Recipe"):
        self._recipe = recipe

    def __call__(self, step: str, field: str) -> OutputRef:
        return getattr(self.__getattr__(step), field)

    def __getattr__(self, step: str) -> _StepOutputsProxy:
        for ref in self._recipe.steps:
            if ref.name == step:
                return _StepOutputsProxy(step, ref.step.outputs_model)
        raise AttributeError(f"No step named '{step}' in recipe '{self._recipe.name}'")


class Recipe(Scope):
    """A composite step: declared sub-steps with explicit wiring.

    The one deliberately mutable Scope: builder methods (`add_step`,
    `step`, `set_output`) extend `steps`/`output_wiring` before first run.
    """

    steps: list[StepRef] = Field(default_factory=list)
    output_wiring: dict[str, OutputRef] = Field(default_factory=dict)

    @property
    def inputs(self) -> _InputsProxy:
        """Wiring proxy (definition layer) -- NOT runtime values."""
        return _InputsProxy(self)

    @property
    def outputs(self) -> _OutputsProxy:
        """Wiring proxy (definition layer) -- NOT runtime values."""
        return _OutputsProxy(self)

    @staticmethod
    def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        wiring = {k: v for k, v in kwargs.items() if isinstance(v, (InputRef, OutputRef))}
        params = {k: v for k, v in kwargs.items() if k not in wiring}
        return wiring, params

    def add_step(self, name: str, scope: "Cab | Recipe", **kwargs: Any) -> "Recipe":
        wiring, params = self._split_kwargs(kwargs)
        self.steps.append(StepRef(name=name, step=scope, wiring=wiring, params=params))
        return self

    def step(self, *, scope: "Cab | Recipe", backend: str | None = None, **kwargs: Any):
        def decorator(func: Callable) -> StepRef:
            bound = scope.with_backend(backend)
            wiring, params = self._split_kwargs(kwargs)
            ref = StepRef(name=func.__name__, step=bound, func=func, wiring=wiring, params=params)
            self.steps.append(ref)
            return ref

        return decorator

    def set_output(self, field: str, ref: OutputRef) -> "Recipe":
        self.output_wiring[field] = ref
        return self


StepRef.model_rebuild()
Recipe.model_rebuild()
