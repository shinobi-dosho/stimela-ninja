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

import re
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, model_validator
from pydantic_core import PydanticUndefined

from shinobi.resources import Resources


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
    human-facing help (`info`), the cab dtype string (`dtype`, e.g.
    "File"/"MS") for a `ParamPattern` attr -- since a dynamically-named
    param has no declared field/type annotation for `path_fields` to
    inspect, this is how backends know to bind-mount its directory --
    `positional`: emitted as a bare value (no `--flag`), in
    field-declaration order, after every flagged/pattern-matched arg.
    `positional_head`: the same, but emitted *before* every flagged/
    pattern-matched arg instead of after -- real cult-cargo's own
    `cubical.yml` names this exact policy (`parset: {policies:
    {positional_head: true}}`) for a tool whose own CLI only recognises a
    leading bare token as a parset file to seed defaults from (CubiCal's
    `main.py`: `if len(sys.argv) > 1 and not sys.argv[1][0].startswith('-'):
    custom_parset_file = sys.argv[1]`); killMS's `kMS.py` has the identical
    `sys.argv[1]`-only check. A plain `positional` field there would always
    land after every `--flag`, which these two tools' own argv[1]-anchored
    parset detection can't see -- either raising ("Unexpected number of
    arguments", CubiCal) or silently not reading the parset at all
    (killMS, which never validates leftover-arg count). Setting both
    `positional` and `positional_head` on the same field is nonsensical;
    `positional_head` wins if both are set. Head positionals, like tail
    ones, are emitted in field-declaration order. -- and
    `repeat_as_tokens`: a list/tuple value is emitted as separate bare argv
    tokens (after the one flag occurrence, or as separate positional
    tokens) instead of joined into one comma-separated token -- real
    cult-cargo cabs express this as a per-field `policies: {repeat: list}`
    (see e.g. wsclean's `-size <w> <h>`/`-weight briggs <n>`, which need
    two separate argv tokens, not `"4096,4096"` as one).

    `choices`: the field's allowed values (cult-cargo/classic's `choices`
    key). A loader that sets this also narrows the field's real annotation
    on `inputs_model`/`outputs_model` to `typing.Literal[*choices]` (see
    `loaders._modelgen.narrow_choices`), so an out-of-set value fails
    pydantic validation the same way a wrong `dtype` would -- not merely
    documented in `info`. Kept here too (rather than only inferred from the
    model's own annotation) so a `ParamPattern` attr -- which has no
    declared model field for a dynamically-matched name -- can still carry
    it.

    `abbreviation`: a short single-dash CLI alias for the field's generated
    `--long-flag` (cult-cargo/classic's `abbreviation` key, e.g. simms'
    `ascii-sky` -> `-as`). Purely a `ninja run` convenience -- carried onto
    the field's `json_schema_extra` by the loaders so `clickutil.build_options`
    can emit the alias; it never affects the argv the tool actually receives
    (that still uses `nom_de_guerre`).

    On an *output* field, a string `implicit` containing `{name}`
    placeholders is resolved by `steps.dispatch._fill_outputs` as a
    `str.format` template against the step's prepared (validated) input
    values -- e.g. `implicit="{prefix}-MFS-image.fits"` derives a tool's
    output path from its own `prefix` input, without shinobi ever
    importing/executing the tool's own schema-generation code. A plain
    string with no `{...}` is used as a literal constant, same as on an
    input field.
    """

    nom_de_guerre: str | None = None
    implicit: Any = None
    info: str | None = None
    positional: bool = False
    positional_head: bool = False
    repeat_as_tokens: bool = False
    dtype: str | None = None
    choices: list[Any] | None = None
    abbreviation: str | None = None


class Policies(BaseModel):
    """How a cab's parameters are turned into command-line arguments.

    `key_value`/`repeat` mirror real cult-cargo cab-level policy keys
    verbatim (e.g. QuartiCal's `policies: {key_value: true, repeat: '[]',
    prefix: ''}`): `key_value=True` means a hydra-style single
    `name=value` argv token instead of two tokens (`--name`, `value`);
    `repeat="[]"` means a list value formats as one bracketed-literal
    token (`solver.terms=[K,G]`) instead of `list_sep`-joining. Distinct
    from a per-field `ParamMeta.repeat_as_tokens` (real per-field
    `policies: {repeat: list}`, e.g. wsclean's bare `-size 4096 4096`),
    which is a field-level override and takes precedence when set.

    `explicit_true`/`explicit_false` also mirror real cult-cargo cab-level
    policy keys verbatim (e.g. CubiCal's `policies: {explicit_true: true,
    explicit_false: false}`): by default a `True` boolean value emits as a
    bare flag (`--flag`, argparse `store_true`-style) and `False` is
    omitted entirely. Some real CLIs (CubiCal's own optparse-derived
    parser among them) instead expect every boolean option to always take
    an explicit value token -- passing a bare flag with no value corrupts
    parsing of everything after it, since the parser consumes the next
    token as that flag's value. `explicit_true=True` emits `--flag true`
    (two tokens, `"true"`/`"false"` lowercase) instead of a bare flag when
    the value is `True`; `explicit_false=True` does the same instead of
    omitting the flag when the value is `False`. Each direction is
    independent (CubiCal only needs `explicit_true`, never
    `explicit_false`), and this applies uniformly to declared fields and
    `ParamPattern`-matched dynamic ones (e.g. CubiCal's own
    per-Jones-term `g-solvable`).
    """

    prefix: str = "--"
    replace: dict[str, str] = Field(default_factory=dict)
    list_sep: str = ","
    repeat_list: bool = False
    key_value: bool = False
    repeat: str | None = None
    explicit_true: bool = False
    explicit_false: bool = False

    def arg_name(self, name: str) -> str:
        """Build the CLI flag name for a parameter name.

        Args:
            name: The parameter's declared/matched name.

        Returns:
            `name` with each `replace` substitution applied, prefixed by
            `prefix` (e.g. `"--"`).
        """
        for old, new in self.replace.items():
            name = name.replace(old, new)
        return f"{self.prefix}{name}"


class ParamSegment(BaseModel):
    """One level of a dotted/dashed dynamic-parameter name. A "shape"
    segment carries only `regex` -- soft validation, no metadata, for a
    level whose actual values can't be enumerated at cab-authoring time
    (e.g. a solver term name like QuartiCal's `K`/`G`). The "meta" segment
    -- always the last one in a `ParamPattern` -- carries `attrs`: the
    known, enumerable part, each value with its own ParamMeta.
    """

    regex: str | None = None
    attrs: dict[str, ParamMeta] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ParamSegment":
        if (self.regex is None) == (self.attrs is None):
            raise ValueError("ParamSegment needs exactly one of `regex` or `attrs`")
        return self


class ParamPattern(BaseModel):
    """A family of inputs whose names are `<segment><separator><segment>...`,
    e.g. QuartiCal's `K.type`/`G.time_interval` or cubical's `g1-solvable`/
    `g-time-int`. Matched as one anchored regex assembled from `segments`:
    exactly one segment is `attrs` (the known, enumerable part, each value
    with its own ParamMeta -- dtype/nom_de_guerre/info); every other segment
    is a `regex` (soft shape-validation of a level that can't be enumerated
    ahead of time). See AGENTS.md for the motivating tools.

    The `attrs` segment is usually last (cubical/QuartiCal: an
    unenumerable term name followed by a known attribute, `g1.solvable`),
    but doesn't have to be -- wsclean's dynamic output names are the
    opposite shape, a known/enumerable image type followed by an
    open-ended qualifier tail (`dirty.per-band`,
    `restored.i.per-interval.mfs`), so `attrs` there is the *first*
    segment. Only one segment may carry `attrs`; the rest must all be
    `regex`.

    A segment regex that should behave as an unconstrained "match anything"
    level (the old design's `prefix`) should be written lazily (`.+?`, not
    `.+`): with more than one registered attr, an eager `.+` prefers the
    *shortest* attr that completes an overall match, which is wrong when
    one attr is itself a suffix of another (e.g. "int" vs "time-int" with
    separator "-") -- `.+?` tries the shortest prefix first, which is
    exactly "prefer the longest/most specific attr".
    """

    separator: str = "."
    segments: list[ParamSegment]

    _compiled: re.Pattern = PrivateAttr()
    _attrs_index: int = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> "ParamPattern":
        attrs_indices = [i for i, seg in enumerate(self.segments) if seg.attrs is not None]
        if len(attrs_indices) != 1:
            raise ValueError("a ParamPattern must have exactly one segment that carries `attrs`")
        object.__setattr__(self, "_attrs_index", attrs_indices[0])
        parts: list[str] = []
        for i, seg in enumerate(self.segments):
            group = f"seg{i}"
            if seg.attrs is not None:
                # Longest-first: makes a longer attr win over a shorter one
                # that's also a valid alternative at the same split point.
                alt = "|".join(re.escape(a) for a in sorted(seg.attrs, key=len, reverse=True))
                parts.append(f"(?P<{group}>{alt})")
            else:
                parts.append(f"(?P<{group}>{seg.regex})")
        pattern = re.escape(self.separator).join(parts)
        object.__setattr__(self, "_compiled", re.compile(f"^{pattern}$"))
        return self

    def matches(self, name: str) -> ParamMeta | None:
        """Check whether `name` matches this pattern and look up its metadata.

        Args:
            name: A dynamic parameter name to test, e.g. `"g1.solvable"`.

        Returns:
            The `ParamMeta` for the matched `attrs` segment value, or
            `None` if `name` doesn't match the compiled pattern.
        """
        m = self._compiled.match(name)
        if not m:
            return None
        return self.segments[self._attrs_index].attrs[m.group(f"seg{self._attrs_index}")]


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


def readonly_path_fields(model: type[BaseModel]) -> set[str]:
    """Names of the `path_fields` explicitly marked ``writable: false`` in
    their schema (carried onto the field's ``json_schema_extra`` by
    ``loaders.worker_schema``). The container backend bind-mounts the
    directories these contribute read-only (``backends.container.bind_dirs``).
    A path field with no ``writable`` marker (the default, including every
    Python-typed pystep input) is treated as writable.
    """
    result: set[str] = set()
    for name in path_fields(model):
        extra = model.model_fields[name].json_schema_extra
        if isinstance(extra, dict) and extra.get("writable") is False:
            result.add(name)
    return result


class Scope(BaseModel):
    """Definition: schema, metadata, backend config. Never carries
    inputs/outputs/func fields -- those live in ExecContext/StepRef.

    `Cab`/`Recipe` are the two execution-aware subclasses `ExecContext.run()`
    knows how to run. A bare `Scope` is also valid -- it's the manual
    building block for a plain-Python-function step whose own function
    returns its `StepResult` directly rather than calling `ctx.run()`; see
    `StepRef`'s docstring and `steps/pyfunc.py`'s `@shinobi.pystep` (which
    automates this pattern from a function's own signature).

    `image` is optional: when set on a bare `Scope` (typically via
    `@shinobi.pystep(image=...)`), the step's Python function can be
    executed inside a container instead of in-process. `Cab` inherits
    this field for the same purpose (container backends need it to wrap
    argv in a runtime invocation).
    """

    name: str
    info: str | None = None
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]
    backend: str | None = None
    image: str | None = None
    # Names the virtualenv the `venv` backend runs this step in: either a
    # filesystem path or a key into `backend.venv.envs` (config). Only the
    # `venv` backend consults it; other backends ignore it. Typed `str` for
    # now, but shaped to accept a spec dict later (auto-provisioning) without
    # a schema break. A venv path is machine-specific, so it belongs here or
    # in config -- never in a shared cab repo.
    venv: str | None = None
    input_mutability: dict[str, Mutability] = Field(default_factory=dict)
    # Step-level skip-if-unchanged caching (shinobi.cache), same precedence
    # shape as `backend`: explicit call-time `cache=`/`cache_dir=` kwarg >
    # this Scope's own value > the enclosing recipe's > `AppConfig.cache`'s
    # default (itself disabled by default).
    cache: bool | None = None
    cache_dir: str | None = None
    # Per-step sandbox execution (shinobi.sandbox), same precedence shape as
    # `cache`: explicit call-time `sandbox=` kwarg > this Scope's own value >
    # the enclosing recipe's > `AppConfig.sandbox.enabled`.
    sandbox: bool | None = None
    # Extra files to rescue from the sandbox besides declared path-typed
    # outputs: glob patterns relative to the step's cwd, `str.format`-resolved
    # against the step's own prepared inputs (e.g. `"{prefix}-*.fits"` for a
    # tool whose dynamically-named output family can't be enumerated as
    # literal output fields). Only consulted when the step runs sandboxed.
    harvest: list[str] = Field(default_factory=list)
    # What this step costs to run, for the scheduler's admission control
    # (`shinobi.resources`). Undeclared (the default) means "free": the step
    # is admitted on `max_workers` alone, exactly as before this field
    # existed. Declared on the *leaf* that does the work -- a `Recipe`
    # inherits the field but must not use it (`build_graph` rejects that),
    # since a recipe is not a unit of execution and reserving for both it
    # and its leaves would double-count.
    resources: Resources | None = None

    @field_serializer("inputs_model", "outputs_model")
    def _serialize_param_model(self, model: type[BaseModel]) -> dict[str, Any]:
        """`inputs_model`/`outputs_model` are pydantic model *classes*, not
        instances -- not JSON-serializable by default (used by `ninja cab`/
        `ninja cabs show`'s `model_dump_json()`). Dump each field's
        annotation/required-ness/default as a plain dict instead of the
        class object itself.
        """
        return {
            name: {
                "type": str(field.annotation),
                "required": field.is_required(),
                "default": None if field.is_required() else field.default,
            }
            for name, field in model.model_fields.items()
        }

    def __call__(self, *, backend: str | None = None, cache: bool | None = None, cache_dir: str | None = None, sandbox: bool | None = None, **kwargs: Any):
        """Bare execution -- no orchestration function."""
        from shinobi.steps.dispatch import _dispatch

        return _dispatch(self, None, backend=backend, cache=cache, cache_dir=cache_dir, sandbox=sandbox, **kwargs)

    def mutability_of(self, field: str) -> Mutability:
        """Look up the declared mutability of an input field.

        Args:
            field: Name of the input field.

        Returns:
            The field's `Mutability`, defaulting to `Mutability.IMMUTABLE`
            if not explicitly declared.
        """
        return self.input_mutability.get(field, Mutability.IMMUTABLE)

    def with_backend(self, backend: str | None) -> "Scope":
        """A copy bound to `backend`, or `self` unchanged if `backend` is
        None. Shared by `@shinobi.step` and `Recipe.step`, which both bind
        a per-step backend override onto a Scope before wrapping it in a
        StepRef.
        """
        return self.model_copy(update={"backend": backend}) if backend else self

    def with_resources(self, resources: Resources | None) -> Scope:
        """A copy bound to `resources`, or `self` unchanged if None. Same
        shape as `with_backend`, and used the same way: `Recipe.add_step`,
        `Recipe.step` and `@shinobi.step` bind a per-step footprint onto a
        Scope before wrapping it in a StepRef, so one shared Cab can be
        declared cheap in one recipe position and expensive in another
        without either mutating the Cab itself.
        """
        return self.model_copy(update={"resources": resources}) if resources else self


class Cab(Scope):
    """An atomic step backed by a single command."""

    command: str
    flavour: str = "binary"
    policies: Policies = Field(default_factory=Policies)
    field_meta: dict[str, ParamMeta] = Field(default_factory=dict)
    input_patterns: list[ParamPattern] = Field(default_factory=list)
    # Output-side analog of `input_patterns`: validation only -- lets
    # `recipe.outputs(step, name)` accept a dynamically-named output (e.g.
    # wsclean's `dirty.per-band`) without it being a literal `outputs_model`
    # field. Does not resolve the output to a real value/path; a cab's
    # `outputs_model` still only ever gets populated for its *declared*
    # fields (see `_fill_outputs` in `steps/dispatch.py`).
    output_patterns: list[ParamPattern] = Field(default_factory=list)
    # regex -> list of wrangler action strings
    wranglers: dict[str, list[str]] = Field(default_factory=dict)

    def param_name(self, field: str) -> str:
        """Resolve the tool-facing name for a declared input field.

        Args:
            field: The cab's own (shinobi-side) field name.

        Returns:
            The field's `nom_de_guerre` if declared in `field_meta`,
            otherwise `field` unchanged.
        """
        meta = self.field_meta.get(field)
        return meta.nom_de_guerre if meta and meta.nom_de_guerre else field

    def match_pattern(self, name: str) -> ParamMeta | None:
        """Check `name` against this cab's dynamic input patterns.

        Args:
            name: An input name not declared as a literal field.

        Returns:
            The matched `ParamMeta`, or `None` if no `input_patterns`
            entry matches.
        """
        for pattern in self.input_patterns:
            meta = pattern.matches(name)
            if meta is not None:
                return meta
        return None

    def match_output_pattern(self, name: str) -> ParamMeta | None:
        """Check `name` against this cab's dynamic output patterns.

        Args:
            name: An output name not declared as a literal field.

        Returns:
            The matched `ParamMeta`, or `None` if no `output_patterns`
            entry matches.
        """
        for pattern in self.output_patterns:
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


class ScatterSpec(BaseModel):
    """Declaration that a step should fan out over one or more list-typed
    input fields. Each listed field must be a list at runtime and all
    listed fields must have the same length. The step is executed once per
    index, with slice `i` receiving element `i` of every scattered field.

    The step's own `inputs_model`/`outputs_model` describe one slice. A
    downstream step sees the scattered step's outputs gathered into lists
    (one element per slice), so it can scatter over them in turn or consume
    the whole list as a gathered result.
    """

    fields: list[str]

    @model_validator(mode="after")
    def _non_empty(self) -> "ScatterSpec":
        if not self.fields:
            raise ValueError("ScatterSpec must declare at least one field to fan out over")
        return self


class LoopIteration(BaseModel):
    """Marks a step as belonging to iteration `index` of the unrolled loop
    `loop` (see `Recipe.add_loop`). Carried by every step the unrolling
    produces; it is bookkeeping for the *skip* decision only -- the
    dependency edges that make the unrolled chain a real DAG are ordinary
    `wiring` plus `after`, never this.

    `prev_step` is the same body step one iteration earlier (so
    `selfcal.4.image` passes through `selfcal.3.image`), and
    `sentinel_step`/`sentinel_field` name the previous iteration's output
    whose existence on disk means "the loop has already converged, do no
    work". `prev_step` and `sentinel_step` are both None for the first
    iteration, which can never skip -- there is nothing before it to pass
    through or to have converged. `sentinel_field` is always set: which
    output carries the signal is a property of the loop, not of one
    iteration.
    """

    loop: str
    index: int
    prev_step: str | None = None
    sentinel_step: str | None = None
    sentinel_field: str


class StepRef(BaseModel):
    """A named, executable binding of a Scope: orchestration function,
    wiring (meaningful only inside a Recipe), and per-step constants.
    Returned by `@shinobi.step` (free-standing) and `@recipe.step`
    (appended to `recipe.steps`). `arbitrary_types_allowed` is needed
    only for `func`.

    `step` is typed as the general `Scope` (not `Cab | Recipe`) so it can
    also hold a bare `Scope` -- the manual, no-magic way to write a
    plain-Python-function step: build `Scope(name=, inputs_model=,
    outputs_model=)` yourself, write a function that always returns its
    own `StepResult` (never calls `ctx.run()`, which only knows how to
    execute a `Cab` or `Recipe`), and wrap it in a `StepRef` directly.
    `@shinobi.pystep` (`steps/pyfunc.py`) automates exactly this pattern
    by deriving the Scope's schema from the function's own signature.
    Passing a `Cab`/`Recipe` instance here is unaffected -- pydantic's
    default `revalidate_instances="never"` keeps an already-constructed
    instance's real subtype, it does not downcast to bare `Scope`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    step: Scope
    func: Callable | None = None
    wiring: dict[str, "InputRef | OutputRef | list[InputRef | OutputRef]"] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    scatter: ScatterSpec | None = None
    # Ordering-only dependency edges: step names this one must run after,
    # with no data flowing. `build_graph` folds these into `deps` exactly
    # like a wiring edge, so the scheduler and the offload compiler both
    # honour them. This exists because a declared graph sometimes has to
    # express "not before", not just "reads from" -- `Recipe.add_loop` needs
    # every iteration to run strictly after the previous one's convergence
    # was decided, and no data flows along that edge. It is deliberately
    # NOT an escape hatch for expressing arbitrary sequencing: prefer real
    # wiring wherever a value actually flows, so the graph keeps saying what
    # depends on what and why.
    after: list[str] = Field(default_factory=list)
    loop: LoopIteration | None = None

    @field_serializer("func")
    def _serialize_func(self, func: Callable | None) -> str | None:
        """`func` is a live Python callable (e.g. a `@shinobi.pystep`'s
        adapter), not JSON-serializable by default -- used by `ninja cabs
        show` on a pystep-backed provider entry. Dump its `__name__`
        instead of the callable itself, same reasoning as `Scope`'s own
        `inputs_model`/`outputs_model` field_serializer.
        """
        return getattr(func, "__name__", None) if func is not None else None

    def __call__(
        self,
        *,
        backend: str | None = None,
        cache: bool | None = None,
        cache_dir: str | None = None,
        provenance: bool | None = None,
        sandbox: bool | None = None,
        **kwargs: Any,
    ):
        """Standalone execution. `params` are merged under caller kwargs;
        wiring is ignored (it can only be resolved inside a running
        Recipe), so any wired-only fields must be supplied as kwargs --
        input validation catches omissions. `scatter` is likewise ignored
        outside a Recipe; to run one slice, pass its scalar inputs directly.
        `provenance` opts this run into image pinning + manifest emission,
        overriding the config default; `sandbox` likewise opts into (or out
        of) sandboxed execution.
        """
        from shinobi.steps.dispatch import _dispatch

        return _dispatch(
            self.step,
            self.func,
            backend=backend,
            cache=cache,
            cache_dir=cache_dir,
            provenance=provenance,
            sandbox=sandbox,
            **{**self.params, **kwargs},
        )


class _InputsProxy:
    """`recipe.inputs.ms` or `recipe.inputs("ms")` -> InputRef(field="ms")."""

    def __init__(self, recipe: "Recipe"):
        """Bind the proxy to `recipe`.

        Args:
            recipe: The recipe whose `inputs_model` fields this proxy
                exposes as `InputRef`s.
        """
        self._recipe = recipe

    def __call__(self, field: str) -> InputRef:
        """Same as attribute access, for a dynamic/non-identifier field name.

        Args:
            field: Name of a field on `recipe.inputs_model`.

        Returns:
            An `InputRef` for `field`.
        """
        return self.__getattr__(field)

    def __getattr__(self, field: str) -> InputRef:
        """Resolve `recipe.inputs.<field>` to an `InputRef`.

        Args:
            field: Name of a field on `recipe.inputs_model`.

        Returns:
            An `InputRef` for `field`.

        Raises:
            AttributeError: If `field` isn't a field of `recipe.inputs_model`.
        """
        if field not in self._recipe.inputs_model.model_fields:
            raise AttributeError(f"'{field}' is not a field of {self._recipe.inputs_model.__name__}")
        return InputRef(field=field)


class _StepOutputsProxy:
    """Second level of `recipe.outputs.<step>.<field>` -- validates the
    field against the sub-step's `outputs_model`, falling back to the
    step's `output_patterns` (if it's a `Cab`) for a dynamically-named
    output not literally declared in `outputs_model`.
    """

    def __init__(self, step: str, outputs_model: type[BaseModel], cab: "Cab | None" = None):
        """Bind the proxy to one recipe step's outputs.

        Args:
            step: Name of the producing step.
            outputs_model: The step's `outputs_model`.
            cab: The step's `Cab` instance, if it is one -- used to fall
                back to `output_patterns` for dynamically-named outputs.
        """
        self._step = step
        self._outputs_model = outputs_model
        self._cab = cab

    def __getattr__(self, field: str) -> OutputRef:
        """Resolve `recipe.outputs.<step>.<field>` to an `OutputRef`.

        Args:
            field: Name of an output field, declared or pattern-matched.

        Returns:
            An `OutputRef` for `(step, field)`.

        Raises:
            AttributeError: If `field` is neither a declared output field
                nor matched by the cab's `output_patterns`.
        """
        if field in self._outputs_model.model_fields:
            return OutputRef(step=self._step, field=field)
        if self._cab is not None and self._cab.match_output_pattern(field) is not None:
            return OutputRef(step=self._step, field=field)
        raise AttributeError(f"'{field}' is not an output of step '{self._step}' ({self._outputs_model.__name__})")


class _OutputsProxy:
    """`recipe.outputs.clean.output_ms` or `recipe.outputs("clean",
    "output_ms")` -> OutputRef(step="clean", field="output_ms").
    """

    def __init__(self, recipe: "Recipe"):
        """Bind the proxy to `recipe`.

        Args:
            recipe: The recipe whose steps' outputs this proxy exposes.
        """
        self._recipe = recipe

    def __call__(self, step: str, field: str) -> OutputRef:
        """Same as `recipe.outputs.<step>.<field>`, for dynamic names.

        Args:
            step: Name of the producing step.
            field: Name of the output field.

        Returns:
            An `OutputRef` for `(step, field)`.
        """
        return getattr(self.__getattr__(step), field)

    def __getattr__(self, step: str) -> _StepOutputsProxy:
        """Resolve `recipe.outputs.<step>` to a `_StepOutputsProxy`.

        Args:
            step: Name of a step declared in `recipe.steps`.

        Returns:
            A `_StepOutputsProxy` for that step's outputs.

        Raises:
            AttributeError: If no step named `step` exists in the recipe.
        """
        for ref in self._recipe.steps:
            if ref.name == step:
                cab = ref.step if isinstance(ref.step, Cab) else None
                return _StepOutputsProxy(step, ref.step.outputs_model, cab)
        raise AttributeError(f"No step named '{step}' in recipe '{self._recipe.name}'")


class _LoopOutputsProxy:
    """`loop.outputs.<field>` -> an `OutputRef` into the step of the *final*
    iteration that produces that body output, so a step after the loop wires
    from the loop without ever naming `max_iter`.
    """

    def __init__(self, loop: "LoopRef"):
        """Bind the proxy to `loop`.

        Args:
            loop: The `LoopRef` whose final-iteration outputs this exposes.
        """
        self._loop = loop

    def __call__(self, field: str) -> OutputRef:
        """Same as attribute access, for a dynamic/non-identifier name.

        Args:
            field: Name of an output field of the loop body.

        Returns:
            An `OutputRef` into the final iteration's producing step.
        """
        return self.__getattr__(field)

    def __getattr__(self, field: str) -> OutputRef:
        """Resolve `loop.outputs.<field>` to an `OutputRef`.

        Args:
            field: Name of an output field of the loop body.

        Returns:
            An `OutputRef` into the final iteration's producing step.

        Raises:
            AttributeError: If the body declares no such output.
        """
        try:
            return self._loop._final[field]
        except KeyError:
            raise AttributeError(f"'{field}' is not an output of loop '{self._loop.name}' (have: {sorted(self._loop._final)})") from None


class LoopRef:
    """What `Recipe.add_loop` returns: the loop's step names, and an
    `outputs` proxy resolving to the final iteration's producers.

    Not a `StepRef` -- a loop is not a step. It has already been unrolled
    into the recipe's `steps` by the time this is returned; this is just a
    handle for wiring whatever comes next.
    """

    def __init__(self, name: str, steps: list[str], final: dict[str, OutputRef]):
        """Bind the handle to an already-unrolled loop.

        Args:
            name: The loop's name.
            steps: Every step name the unrolling produced, in order.
            final: Body output field -> `OutputRef` into the final
                iteration's producing step.
        """
        self.name = name
        self.steps = steps
        self._final = final

    @property
    def outputs(self) -> _LoopOutputsProxy:
        """Wiring proxy (definition layer) -- NOT runtime values."""
        return _LoopOutputsProxy(self)


class Recipe(Scope):
    """A composite step: declared sub-steps with explicit wiring.

    The one deliberately mutable Scope: builder methods (`add_step`,
    `step`, `set_output`) extend `steps`/`output_wiring` before first run.
    """

    steps: list[StepRef] = Field(default_factory=list)
    output_wiring: dict[str, OutputRef] = Field(default_factory=dict)
    # Per-recipe override for how many steps may run concurrently; falls
    # back to AppConfig.execution.max_workers (default 1) when None. Same
    # precedence shape as `backend`. WARNING: with max_workers > 1, two
    # independent steps that wire in the *same* MUTABLE input run
    # concurrently against that shared object -- a data race shinobi cannot
    # detect. IMMUTABLE inputs (the default) are deep-copied per step and
    # are safe.
    max_workers: int | None = None

    @property
    def inputs(self) -> _InputsProxy:
        """Wiring proxy (definition layer) -- NOT runtime values."""
        return _InputsProxy(self)

    @property
    def outputs(self) -> _OutputsProxy:
        """Wiring proxy (definition layer) -- NOT runtime values."""
        return _OutputsProxy(self)

    @staticmethod
    def _is_wiring_value(v: Any) -> bool:
        """A single `InputRef`/`OutputRef`, or a non-empty list of them
        (e.g. `applycal`'s `gaintable=[recipe.outputs.k.caltable,
        recipe.outputs.g.caltable]` -- accumulating a variable number of
        upstream outputs into one list-typed input). A list is wiring only
        if *every* element is a ref -- a list mixing refs and literal
        values isn't supported, and is treated as a literal param instead
        (so it fails loudly in the callee's own validation, rather than
        silently dropping half its dependency edges).
        """
        if isinstance(v, (InputRef, OutputRef)):
            return True
        return isinstance(v, list) and bool(v) and all(isinstance(x, (InputRef, OutputRef)) for x in v)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        wiring = {k: v for k, v in kwargs.items() if cls._is_wiring_value(v)}
        params = {k: v for k, v in kwargs.items() if k not in wiring}
        return wiring, params

    def add_step(
        self,
        name: str,
        scope: "Scope | StepRef",
        *,
        scatter: list[str] | ScatterSpec | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> "Recipe":
        """Add a step. `scope` is usually a bare `Scope`/`Cab`/`Recipe`, but
        can also be an already-built `StepRef` (e.g. from `@shinobi.pystep`
        or `@shinobi.step`) -- its `func` is carried over so the step keeps
        its orchestration function, not just its schema.

        Args:
            scatter: Fields to fan out over. Each must be a field of the
                step's `inputs_model`; at runtime the corresponding value
                must be a list, and every scattered field must have the same
                length.
            resources: What this step costs to run, for the scheduler's
                admission control. An explicit keyword rather than a kwarg,
                since `**kwargs` here is split into wiring and step params
                and would otherwise swallow it as a constant input value.
        """
        scatter_spec = ScatterSpec(fields=scatter) if isinstance(scatter, list) else scatter
        wiring, params = self._split_kwargs(kwargs)
        if isinstance(scope, StepRef):
            ref = scope.model_copy(
                update={
                    "name": name,
                    "wiring": {**scope.wiring, **wiring},
                    "params": {**scope.params, **params},
                    "scatter": scatter_spec if scatter_spec is not None else scope.scatter,
                    # The footprint lives on the Scope, not the StepRef, so
                    # binding it here has to reach through to `scope.step` --
                    # copying only the StepRef would silently drop it.
                    "step": scope.step.with_resources(resources),
                }
            )
        else:
            ref = StepRef(name=name, step=scope.with_resources(resources), wiring=wiring, params=params, scatter=scatter_spec)
        self.steps.append(ref)
        return self

    def add_loop(
        self,
        name: str,
        body: "Scope | StepRef",
        *,
        max_iter: int,
        until: str,
        carry: dict[str, str],
        index_input: str | None = None,
        **kwargs: Any,
    ) -> LoopRef:
        """Unroll `body` `max_iter` times into this recipe, as a declared chain.

        The loop is *bounded unrolling with short-circuit pass-through*: the
        body's steps are flattened into this recipe once per iteration and
        chained by real `carry` wiring, so the whole thing is an ordinary
        statically-inspectable DAG -- `--dryrun` renders every iteration and
        the offload compiler emits a plain dependency chain. The only runtime
        decision is whether an already-declared step does any work: once an
        iteration writes the `until` sentinel file, every later step passes the
        corresponding previous iteration's outputs straight through.

        A `Recipe` body is flattened (`selfcal.3.image`); any other Scope is one
        step per iteration (`selfcal.3`).

        Args:
            name: The loop's name; every step it creates is prefixed with it.
            body: The loop body. Its outputs must be re-consumable as its
                inputs (see `carry`) -- a loop body is a fixed point.
            max_iter: How many times to unroll. The upper bound on iterations,
                not the exact count.
            until: Name of a **path-typed** output of `body`. Once that file
                exists, the loop has converged. A path (rather than a bool) is
                what lets the identical predicate serve both local execution
                and an offloaded sbatch script.
            carry: Body output field -> body input field, the loop-carried
                dependency. Required and explicit: these pairs *are* the edges
                between iterations, and inferring them from matching names
                would make the graph's shape depend on name coincidence.
            index_input: Name of a body input to bind to the 1-based iteration
                number. Without it every iteration resolves to identical
                values, so a body cannot name its outputs per cycle (a
                per-cycle image prefix, say) -- which real self-calibration
                does. Steps consuming it must tolerate a changing value: it is
                the one input that deliberately differs between iterations.
            **kwargs: Iteration 1's inputs, split into wiring and constants
                exactly as `add_step` does. Constants apply to every iteration.

        Returns:
            A `LoopRef` whose `outputs` proxy resolves to the final iteration.
        """
        if max_iter < 1:
            raise ValueError(f"loop '{name}' declares max_iter={max_iter}; it must be at least 1")

        # A StepRef body (e.g. from @shinobi.pystep) contributes its func and
        # per-step constants, same as add_step carries them over.
        body_ref = body if isinstance(body, StepRef) else None
        body = body_ref.step if body_ref is not None else body
        is_recipe = isinstance(body, Recipe)
        if is_recipe:
            for sub in body.steps:
                if sub.scatter is not None:
                    raise ValueError(f"loop '{name}': body step '{sub.name}' declares scatter, which is not supported inside a loop in this version")
                if isinstance(sub.step, Recipe):
                    raise ValueError(f"loop '{name}': body step '{sub.name}' is a nested Recipe, which is not supported inside a loop in this version")

        def iteration_step(k: int, sub_name: str) -> str:
            """The flattened name of body step `sub_name` in iteration `k`."""
            return f"{name}.{k}.{sub_name}" if is_recipe else f"{name}.{k}"

        def producer(k: int, out_field: str) -> OutputRef:
            """Which flattened step of iteration `k` produces body output
            `out_field`, resolved through the body's own `output_wiring`.
            """
            if not is_recipe:
                return OutputRef(step=iteration_step(k, ""), field=out_field)
            src = body.output_wiring.get(out_field)
            if src is None:
                raise ValueError(
                    f"loop '{name}': body output '{out_field}' is not wired to any of the body's steps (Recipe.set_output), so the loop cannot tell which step produces it"
                )
            return OutputRef(step=iteration_step(k, src.step), field=src.field)

        if until not in body.outputs_model.model_fields:
            raise ValueError(f"loop '{name}': until='{until}' is not an output of {body.outputs_model.__name__}")
        if until not in path_fields(body.outputs_model):
            raise ValueError(
                f"loop '{name}': until='{until}' must be a path-typed output (its existence on disk is the convergence signal), but {body.outputs_model.__name__}.{until} is not a path"
            )
        for out_field, in_field in carry.items():
            if out_field not in body.outputs_model.model_fields:
                raise ValueError(f"loop '{name}': carry key '{out_field}' is not an output of {body.outputs_model.__name__}")
            if in_field not in body.inputs_model.model_fields:
                raise ValueError(f"loop '{name}': carry value '{in_field}' is not an input of {body.inputs_model.__name__}")
        if index_input is not None and index_input not in body.inputs_model.model_fields:
            raise ValueError(f"loop '{name}': index_input='{index_input}' is not an input of {body.inputs_model.__name__}")
        producer(1, until)  # fail now if the sentinel has no declared producer

        initial, loop_params = self._split_kwargs(kwargs)
        existing = {ref.name for ref in self.steps}

        def bound_inputs(k: int) -> dict[str, InputRef | OutputRef | list[Any]]:
            """What each of the body's own input fields is wired from, in
            iteration `k`: the caller's wiring on the first pass, the previous
            iteration's outputs thereafter.
            """
            if k == 1:
                return dict(initial)
            return {**initial, **{in_field: producer(k - 1, out_field) for out_field, in_field in carry.items()}}

        created: list[str] = []
        for k in range(1, max_iter + 1):
            bound = bound_inputs(k)
            sentinel_step = producer(k - 1, until).step if k > 1 else None
            sentinel_field = producer(1, until).field
            body_steps = body.steps if is_recipe else [body_ref or StepRef(name=name, step=body)]

            for sub in body_steps:
                step_name = iteration_step(k, sub.name)
                if step_name in existing:
                    raise ValueError(f"loop '{name}' would create a step named '{step_name}', which already exists in recipe '{self.name}'")
                existing.add(step_name)

                wiring: dict[str, Any] = {}
                params = dict(sub.params)
                intra_iteration = False

                def rebind(source: InputRef | OutputRef, field: str) -> Any:
                    """Rewrite one of the body's own wiring sources into the
                    flattened graph: a reference to another body step becomes a
                    reference to that step in *this* iteration; a reference to
                    the body's own inputs becomes whatever the loop bound them
                    to for this iteration.
                    """
                    nonlocal intra_iteration
                    if isinstance(source, OutputRef):
                        intra_iteration = True
                        return OutputRef(step=iteration_step(k, source.step), field=source.field)
                    if source.field == index_input:
                        return k
                    bound_source = bound.get(source.field)
                    if bound_source is not None:
                        return bound_source
                    if source.field in loop_params:
                        return loop_params[source.field]
                    model_field = body.inputs_model.model_fields.get(source.field)
                    default = model_field.default if model_field is not None else PydanticUndefined
                    if default is PydanticUndefined:
                        raise ValueError(
                            f"loop '{name}': body step '{sub.name}' wires input '{field}' from the body input '{source.field}', but the loop was given no value for '{source.field}' and it has no default"
                        )
                    return default

                for field, source in sub.wiring.items():
                    if isinstance(source, list):
                        rebound = [rebind(one, field) for one in source]
                    else:
                        rebound = rebind(source, field)
                    if self._is_wiring_value(rebound):
                        wiring[field] = rebound
                    else:
                        params[field] = rebound

                # A non-Recipe body has no internal wiring to rewrite, so bind
                # its inputs from the loop directly.
                if not is_recipe:
                    for field, source in bound.items():
                        if field == index_input:
                            continue
                        wiring[field] = source
                    if index_input is not None:
                        params[index_input] = k
                    for field, value in loop_params.items():
                        params.setdefault(field, value)

                self.steps.append(
                    StepRef(
                        name=step_name,
                        step=sub.step,
                        func=sub.func,
                        wiring=wiring,
                        params=params,
                        # An *entry* step of the iteration reads nothing from
                        # its own iteration, so nothing otherwise orders it
                        # after the previous one. Without this edge it could be
                        # scheduled before the previous iteration had decided
                        # convergence -- and under offload it would carry no
                        # `afterok` link at all. No data flows here, only order.
                        after=[sentinel_step] if (sentinel_step is not None and not intra_iteration) else [],
                        loop=LoopIteration(
                            loop=name,
                            index=k,
                            prev_step=iteration_step(k - 1, sub.name) if k > 1 else None,
                            sentinel_step=sentinel_step,
                            sentinel_field=sentinel_field,
                        ),
                    )
                )
                created.append(step_name)

        final = {field: producer(max_iter, field) for field in body.outputs_model.model_fields if not is_recipe or field in body.output_wiring}
        return LoopRef(name=name, steps=created, final=final)

    def step(
        self,
        scope: Scope,
        *,
        backend: str | None = None,
        scatter: list[str] | ScatterSpec | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ):
        """Decorate a function as a new step appended to this recipe.

        Args:
            scope: The Cab, Recipe, or bare Scope to bind as this step.
            backend: Backend override for this step.
            scatter: Fields to fan out over (see `add_step`).
            resources: What this step costs to run (see `add_step`).
            **kwargs: Split into wiring (`InputRef`/`OutputRef` values) and
                per-step constant params via `_split_kwargs`.

        Returns:
            A decorator that binds the given function, appends the
            resulting `StepRef` to `self.steps`, and returns it.
        """

        scatter_spec = ScatterSpec(fields=scatter) if isinstance(scatter, list) else scatter

        def decorator(func: Callable) -> StepRef:
            """Bind `func` as this step's orchestration function.

            Args:
                func: The orchestration function.

            Returns:
                The new `StepRef`, already appended to `self.steps`.
            """
            bound = scope.with_backend(backend).with_resources(resources)
            wiring, params = self._split_kwargs(kwargs)
            ref = StepRef(
                name=func.__name__,
                step=bound,
                func=func,
                wiring=wiring,
                params=params,
                scatter=scatter_spec,
            )
            self.steps.append(ref)
            return ref

        return decorator

    def set_output(self, field: str, ref: OutputRef) -> "Recipe":
        """Wire a recipe output field to an upstream step's output.

        Args:
            field: Name of the recipe's own output field.
            ref: The step output that should populate `field`.

        Returns:
            `self`, for chaining.
        """
        self.output_wiring[field] = ref
        return self


StepRef.model_rebuild()
Recipe.model_rebuild()
