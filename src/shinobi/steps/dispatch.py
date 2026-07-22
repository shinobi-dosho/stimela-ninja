"""Dispatch for the step model.

`_dispatch` is the single entry point (used by `Scope.__call__`,
`StepRef.__call__`, and `_run_recipe` for sub-steps). It builds an
`ExecContext`, runs the orchestration function (if any), and enforces the
strict return contract. `ExecContext.run` merges overrides, validates and
mutability-processes the inputs, resolves the backend, and executes via
`_run_cab`/`_run_recipe`.

Backend resolution priority: explicit `backend` arg > the scope's own
`backend` > the enclosing recipe's backend > `AppConfig.load().backend.default`.
"""

from __future__ import annotations

import builtins
import copy
import heapq
import importlib
import logging
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, create_model
from shinobi.cache import combine_keys, compute_cache_key, get_cache_manifest
from shinobi.config import AppConfig
from shinobi.exceptions import CabRunError, ParameterError, ShinobiError, StepError
from shinobi.graph import build_graph
from shinobi.policies import build_argv
from shinobi.results import StepResult
from shinobi.sandbox import (
    absolutize_path_inputs,
    create_sandbox,
    discard_sandbox,
    harvest_outputs,
    prepare_output_parents,
    prune_unused_parents,
    relativize_path_outputs,
)
from shinobi.steps.loops import passthrough_result, should_skip
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, Recipe, Scope, StepRef
from shinobi.wranglers import apply_wranglers

# Run-log records (step lifecycle + captured tool output) go through the
# `shinobi` logger hierarchy; nothing prints unless a handler is attached
# (the CLI attaches a file handler via shinobi.logsetup when
# AppConfig.log.file is set).
logger = logging.getLogger("shinobi.run")

# Instance-override registry: lets tests register a specific backend
# instance (e.g. a RecordingBackend) under a name. Anything not overridden
# here is resolved through shinobi.backends.get_backend (the real,
# class-based registry).
_STEP_BACKENDS: dict[str, Any] = {}


def register_step_backend(name: str, backend: Any) -> None:
    """Register a backend *instance* under `name`, overriding the real
    class-based registry. Mainly for tests.
    """
    _STEP_BACKENDS[name] = backend


def get_step_backend(name: str) -> Any:
    """Resolve a backend instance by name, checking test overrides first.

    Args:
        name: Backend name, e.g. `"native"`, `"slurm"`, or a name registered
            via `register_step_backend`.

    Returns:
        The backend instance registered under `name` in `_STEP_BACKENDS`,
        else a fresh instance from `shinobi.backends.get_backend`.
    """
    if name in _STEP_BACKENDS:
        return _STEP_BACKENDS[name]
    from shinobi.backends import get_backend

    return get_backend(name)


def _prepare_inputs(scope: Scope, kwargs: dict[str, Any], *, validated: Any = None) -> dict[str, Any]:
    """Validate kwargs through inputs_model, then deep-copy every field
    not explicitly marked MUTABLE -- the actual enforcement mechanism.
    Re-validating an already-validated instance of the exact type through
    pydantic does NOT itself copy it (revalidate_instances="never" by
    default), so the deepcopy step below is load-bearing, not redundant.

    Constructing `scope.inputs_model(**kwargs)` is used for validation
    (missing/wrong-type fields raise here, before anything runs), but its
    *values* can't be used for MUTABLE fields even to skip a copy: pydantic
    already reconstructs container fields (e.g. list) during validation,
    so `validated.some_list is kwargs["some_list"]` is False even though
    nothing was meant to copy it. MUTABLE fields therefore read the
    caller's original object straight out of `kwargs`, bypassing the
    validated instance entirely -- true pass-by-reference, not "pydantic's
    copy, but we chose not to make a second one."

    `validated` lets a caller that already validated this exact `kwargs`
    (e.g. `ExecContext.__init__`, when `run()` is called with no overrides)
    pass that instance through instead of paying a second full pydantic
    validation pass for no new information.
    """
    if validated is None:
        try:
            validated = scope.inputs_model(**kwargs)
        except ValidationError as exc:
            raise ParameterError(f"{scope.name}: parameter validation failed:\n{exc}") from exc
    prepared: dict[str, Any] = {}
    for name in type(validated).model_fields:
        if scope.mutability_of(name) is Mutability.MUTABLE:
            value = kwargs[name] if name in kwargs else getattr(validated, name)
        else:
            value = copy.deepcopy(getattr(validated, name))
        prepared[name] = value
    # dynamically-named (pattern-matched) params land in model_extra when
    # the inputs_model allows extras; carry them through (immutable).
    extras = validated.model_extra or {}
    if extras:
        warnings.warn(
            f"'{scope.name}': parameter(s) {sorted(extras)} matched a dynamic "
            "parameter pattern and are passed through to the tool as-is -- "
            "shinobi has no declared field for them, so it cannot type/range-"
            "check them the way it does for the cab's declared parameters.",
            stacklevel=2,
        )
    for name, value in extras.items():
        prepared[name] = copy.deepcopy(value)
    return prepared


class ExecContext:
    """Live execution state, created by `_dispatch`. `inputs` is a
    validated snapshot for inspection; the raw caller kwargs are kept
    separately because MUTABLE fields must reach the backend as the
    caller's original objects.
    """

    def __init__(
        self,
        scope: Scope,
        raw_inputs: dict[str, Any],
        *,
        backend_override: str | None = None,
        recipe_backend: str | None = None,
        config: AppConfig | None = None,
        cache_enabled: bool = False,
        cache_dir: str = "",
        cache_path: str = "",
        stream: bool = True,
        pin: bool = False,
        sandbox_root: str | None = None,
        input_keys: dict[str, Any] | None = None,
    ):
        """Initialize execution state for one dispatched step.

        Args:
            scope: The Cab or Recipe being executed.
            raw_inputs: The caller's raw kwargs, validated against
                `scope.inputs_model`.
            backend_override: Explicit backend name for this call, highest
                priority in `resolve_backend_name`.
            recipe_backend: Backend name inherited from the enclosing
                recipe, used if nothing more specific is set.
            config: App configuration to fall back on; loaded fresh if not
                given.
            cache_enabled: Whether step-level result caching is active.
            cache_dir: Directory the cache manifest lives in.
            cache_path: Dotted step path used as this run's cache/log label.
            stream: Whether to stream the step's stdout/stderr live.
            sandbox_root: Scratch root for sandboxed execution
                (`shinobi.sandbox`), or `None` when this step runs
                unsandboxed -- the root doubles as the enabled flag.
            input_keys: Per-input-field cache keys of the steps that
                produced this step's wired inputs (see `shinobi.cache`).
                Only an enclosing recipe can know these; a top-level call
                has none.
        """
        self.scope = scope
        self._raw = raw_inputs
        try:
            self.inputs = scope.inputs_model(**raw_inputs)
        except ValidationError as exc:
            raise ParameterError(f"{scope.name}: parameter validation failed:\n{exc}") from exc
        self.outputs = None
        self._backend_override = backend_override
        self._recipe_backend = recipe_backend
        self._config = config
        self._cache_enabled = cache_enabled
        self._cache_dir = cache_dir
        self._cache_path = cache_path
        self._stream = stream
        # Provenance on -> digest-pin container images (pin-then-run). Read by
        # the pystep container path and threaded to backends via ctx.run().
        self._pin = pin
        # Sandbox scratch root, or None when unsandboxed. Read by the pystep
        # container path and threaded to _run_cab/_run_recipe via ctx.run().
        self._sandbox_root = sandbox_root
        # Upstream provenance for this step's wired inputs. A Recipe scope is
        # never cached itself, so it doesn't consume these -- it forwards them
        # to _run_recipe, which resolves each sub-step's InputRef wiring
        # against them (see shinobi.cache).
        self._input_keys = input_keys

    def prepare_inputs(self) -> dict[str, Any]:
        """Validated + mutability-processed inputs, with no overrides applied
        -- for a plain-function step's own function to call the underlying
        function with (see `steps/pyfunc.py`'s adapter, and the manual
        bare-`Scope` pattern documented on `Scope`/`StepRef`). Reuses the
        already-validated `self.inputs` snapshot rather than re-validating.
        """
        return _prepare_inputs(self.scope, self._raw, validated=self.inputs)

    def resolve_backend_name(self, override: str | None = None) -> str:
        """Resolve the effective backend name using the standard priority
        chain. Exposed so orchestration functions (e.g. the pystep
        adapter) can inspect which backend is active without duplicating
        the precedence logic.
        """
        return override or self._backend_override or self.scope.backend or self._recipe_backend or (self._config or AppConfig.load()).backend.default

    def import_func(self, func: str, module: str | None = None) -> Callable:
        """Import and return a callable by name.

        If `module` is None, looks up `func` in builtins (e.g. ``print``,
        ``len``). Otherwise imports `module` and returns `getattr(module, func)`.

        Useful for pysteps that invoke container-only functions (e.g. CASA
        tasks) without triggering linter warnings about missing imports on
        the host.
        """
        if module is None:
            return getattr(builtins, func)
        mod = importlib.import_module(module)
        return getattr(mod, func)

    def run(self, *, backend: str | None = None, **overrides: Any) -> StepResult:
        """Run the underlying Cab or Recipe with optional input overrides.

        Args:
            backend: Backend name to use for this run, taking priority
                over the scope's own/recipe-inherited/config default.
            **overrides: Input values to override on top of the raw inputs
                this context was created with.

        Returns:
            The step's `StepResult`. Also stored on `self.outputs`.

        Raises:
            TypeError: If `self.scope` is neither a `Cab` nor a `Recipe`
                (a plain-function step must return its result directly
                instead of calling `ctx.run()`).
        """
        raw = {**self._raw, **overrides}
        # No overrides -> `raw` is exactly what `self.inputs` already
        # validated in __init__; reuse it instead of re-validating.
        validated = self.inputs if not overrides else None
        prepared = _prepare_inputs(self.scope, raw, validated=validated)
        backend_name = self.resolve_backend_name(backend)
        if isinstance(self.scope, Cab):
            result = _run_cab(
                self.scope,
                prepared,
                backend_name,
                label=self._cache_path,
                stream=self._stream,
                pin=self._pin,
                sandbox_root=self._sandbox_root,
            )
        elif isinstance(self.scope, Recipe):
            result = _run_recipe(
                self.scope,
                prepared,
                backend_name,
                self._config,
                self._cache_enabled,
                self._cache_dir,
                self._cache_path,
                self._stream,
                provenance=self._pin,
                sandbox=self._sandbox_root is not None,
                input_keys=self._input_keys,
            )
        else:
            raise TypeError(
                f"{type(self.scope).__name__} scope has no ctx.run() support -- a "
                "plain-function step's own function must return its StepResult "
                "directly instead of calling ctx.run() (see Scope's docstring)"
            )
        self.outputs = result.outputs
        return result


def _emit_run_manifest(
    result: StepResult,
    ctx: "ExecContext",
    config: AppConfig,
    backend: str | None,
    target: str | None = None,
) -> None:
    """Write the run manifest for a completed top-level run. Callers gate on
    the resolved provenance flag; this stays best-effort -- a provenance
    failure warns but never fails the run.
    """
    try:
        from shinobi.provenance import build_manifest, run_manifest_path

        manifest = build_manifest(result, backend=ctx.resolve_backend_name(backend), target=target)
        manifest.write(run_manifest_path(config, result.name))
    except Exception as exc:  # noqa: BLE001 -- provenance must not break a run
        warnings.warn(f"failed to write run manifest for {result.name!r}: {exc}", stacklevel=2)


def _dispatch(
    scope: Scope,
    func: Callable | None,
    *,
    backend: str | None = None,
    cache: bool | None = None,
    cache_dir: str | None = None,
    stream: bool | None = None,
    provenance: bool | None = None,
    sandbox: bool | None = None,
    _recipe_backend: str | None = None,
    _recipe_cache: bool | None = None,
    _recipe_cache_dir: str | None = None,
    _recipe_stream: bool | None = None,
    _recipe_provenance: bool | None = None,
    _recipe_sandbox: bool | None = None,
    _cache_path: str | None = None,
    _config: AppConfig | None = None,
    _provenance_target: str | None = None,
    _input_keys: dict[str, Any] | None = None,
    **kwargs: Any,
) -> StepResult:
    config = _config or AppConfig.load()
    cache_enabled = cache if cache is not None else scope.cache if scope.cache is not None else _recipe_cache if _recipe_cache is not None else config.cache.enabled
    cache_dir_value = cache_dir or scope.cache_dir or _recipe_cache_dir or config.cache.dir
    cache_path = _cache_path or scope.name
    stream_enabled = stream if stream is not None else _recipe_stream if _recipe_stream is not None else config.log.stream
    # Provenance (image pinning + manifest emission) is one opt-in switch,
    # resolved highest-priority-first like cache: explicit arg (CLI
    # --provenance) > inherited-from-recipe > config default.
    provenance_enabled = provenance if provenance is not None else _recipe_provenance if _recipe_provenance is not None else config.provenance.enabled
    # Sandbox resolves like cache: explicit arg > the scope's own value >
    # inherited-from-recipe > config default. The resolved switch travels as
    # the scratch root itself (None = disabled).
    sandbox_enabled = sandbox if sandbox is not None else scope.sandbox if scope.sandbox is not None else _recipe_sandbox if _recipe_sandbox is not None else config.sandbox.enabled
    # A Recipe-shaped scope is never itself cached -- its own sub-steps
    # each get their own cache check via their own recursive _dispatch
    # call (see shinobi.cache's module docstring for why).
    cacheable = cache_enabled and not isinstance(scope, Recipe)

    ctx = ExecContext(
        scope,
        kwargs,
        backend_override=backend,
        recipe_backend=_recipe_backend,
        config=_config,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir_value,
        cache_path=cache_path,
        stream=stream_enabled,
        pin=provenance_enabled,
        sandbox_root=config.sandbox.dir if sandbox_enabled else None,
        input_keys=_input_keys,
    )

    manifest = None
    cache_key = None
    if cacheable:
        manifest = get_cache_manifest(cache_dir_value)
        prepared_for_key = ctx.prepare_inputs()
        cache_key = compute_cache_key(scope, func, prepared_for_key, _input_keys)
        hit = manifest.check(cache_path, cache_key, scope, prepared_for_key)
        if hit is not None:
            # A hit stands in for the run that first produced this key, so it
            # must advertise the same provenance -- otherwise dependents would
            # see the producer's key vanish and their own keys would flip
            # between runs purely on whether the producer hit or ran.
            hit.cache_key = cache_key
            logger.info("step %s: cache hit -- skipping run", cache_path)
            if _cache_path is None and provenance_enabled:
                _emit_run_manifest(hit, ctx, config, backend, target=_provenance_target)
            return hit

    logger.info(
        "step %s: starting%s",
        cache_path,
        " (sandboxed)" if sandbox_enabled else "",
    )
    try:
        if func is None:
            result = ctx.run()
        else:
            result = func(ctx)
            if result is None:
                result = ctx.run()
            elif not isinstance(result, StepResult):
                raise TypeError(f"step function {getattr(func, '__name__', func)!r} must return StepResult or None, got {type(result).__name__}")
    except Exception:
        logger.exception("step %s: raised", cache_path)
        raise

    # A recipe's stdout/stderr aggregate its sub-steps', and each sub-step
    # already logged its own via its recursive _dispatch -- re-logging the
    # aggregate would duplicate every line.
    if result.kind != "recipe":
        for line in result.stdout.splitlines():
            logger.info("[%s] %s", cache_path, line)
        for line in result.stderr.splitlines():
            logger.info("[%s] %s", cache_path, line)
    if result.success:
        logger.info("step %s: finished (returncode 0)", cache_path)
    else:
        logger.error("step %s: failed (returncode %s)", cache_path, result.returncode)

    if cacheable:
        result.cache_key = cache_key
    if cacheable and result.success:
        manifest.record(cache_path, cache_key, result)
    if _cache_path is None and provenance_enabled:
        _emit_run_manifest(result, ctx, config, backend, target=_provenance_target)
    return result


def _fill_outputs(cab: Cab, prepared: dict[str, Any], run, wrangled: dict[str, Any]):
    """Fill the cab's outputs_model by priority: wrangler value >
    same-named final input > reserved run field (returncode/stdout/stderr)
    > `ParamMeta.implicit` template/constant > field default.

    An `implicit` string containing `{...}` placeholders is resolved as a
    `str.format` template against `prepared` (the step's own validated
    input values) -- e.g. wsclean's `implicit="{prefix}-MFS-image.fits"`
    derives its output path from the `prefix` input. A plain string with
    no placeholders is used as-is, same as an input field's `implicit`.
    """
    reserved = {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    values: dict[str, Any] = {}
    for name in cab.outputs_model.model_fields:
        if name in wrangled:
            values[name] = wrangled[name]
        elif name in prepared:
            values[name] = prepared[name]
        elif name in reserved:
            values[name] = reserved[name]
        else:
            meta = cab.field_meta.get(name)
            if meta is not None and isinstance(meta.implicit, str):
                values[name] = _resolve_implicit_template(cab, name, meta.implicit, prepared)
    try:
        return cab.outputs_model(**values)
    except ValidationError as exc:
        raise ParameterError(f"{cab.name}: output validation failed:\n{exc}") from exc


def _resolve_implicit_template(cab: Cab, field: str, template: str, prepared: dict[str, Any]) -> str:
    try:
        return template.format(**prepared)
    except KeyError as exc:
        raise ParameterError(f"cab {cab.name!r} output {field!r} implicit template {template!r} references unknown input {exc}") from exc


def _run_cab(
    cab: Cab,
    prepared: dict[str, Any],
    backend_name: str,
    *,
    label: str = "",
    stream: bool = True,
    pin: bool = False,
    sandbox_root: str | None = None,
) -> StepResult:
    # Sandboxed run (shinobi.sandbox): the tool's cwd is a private scratch
    # dir; path-typed inputs are anchored back at the workspace so the tool
    # still reads/mutates the caller's real files. argv and the backend's
    # bind mounts are built from the anchored values, but output filling
    # below keeps the caller's original (workspace-relative) values, so
    # declared output paths stay stable whether or not a sandbox was used.
    sandbox_dir = None
    run_inputs = prepared
    if sandbox_root is not None:
        workspace = Path.cwd()
        sandbox_dir = create_sandbox(sandbox_root, label or cab.name)
        precreated = prepare_output_parents(cab, prepared, sandbox_dir)
        run_inputs = absolutize_path_inputs(cab, prepared, workspace)
    argv = build_argv(cab, run_inputs)
    backend = get_step_backend(backend_name)
    import shlex

    logger.debug("step %s: backend=%s argv: %s", label or cab.name, backend_name, shlex.join(argv))
    # The backend gets the prepared dict (not a rebuilt model) so MUTABLE
    # fields reach it as the caller's own objects by reference -- rebuilding
    # a pydantic model here would deep-copy every container and break that.
    run = backend.run(
        cab,
        argv,
        run_inputs,
        label=label or cab.name,
        stream=stream,
        pin=pin,
        cwd=str(sandbox_dir) if sandbox_dir is not None else None,
    )
    lines = run.stdout.splitlines() + run.stderr.splitlines()
    wrangled = apply_wranglers(cab.wranglers, lines)
    outputs = _fill_outputs(cab, prepared, run, wrangled)
    if sandbox_dir is not None:
        outputs = relativize_path_outputs(cab, outputs, workspace)
        if run.returncode == 0:
            prune_unused_parents(precreated)
            harvest_outputs(cab, outputs, prepared, sandbox_dir, workspace)
            discard_sandbox(sandbox_dir)
        else:
            warnings.warn(
                f"step '{label or cab.name}' failed (returncode {run.returncode}); its sandbox is kept for post-mortem at {sandbox_dir}",
                stacklevel=2,
            )
    return StepResult(
        name=cab.name,
        returncode=run.returncode,
        outputs=outputs,
        inputs=cab.inputs_model(**prepared),
        stdout=run.stdout,
        stderr=run.stderr,
        kind="cab",
        backend=backend_name,
        image=cab.image,
        image_digest=run.image_digest,
        containerized=run.containerized,
        sandboxed=sandbox_dir is not None,
    )


def _resolve_wiring(ref, prepared: dict[str, Any], results: dict[str, StepResult]) -> dict[str, Any]:
    """A sub-step's effective kwargs: its per-step `params`, with wiring
    (recipe inputs via `InputRef`, upstream outputs via `OutputRef`) merged
    on top. Every `OutputRef.step` here is guaranteed to be in `results`
    already -- the scheduler only makes a step ready once all its upstream
    dependencies have completed.
    """

    def resolve_one(field: str, source: InputRef | OutputRef) -> Any:
        """Resolve a single wiring source to its concrete value.

        Args:
            source: Either an `InputRef` (a recipe input) or an `OutputRef`
                (an already-completed upstream step's output).

        Returns:
            The resolved value.
        """
        if isinstance(source, InputRef):
            return prepared[source.field]
        try:
            return getattr(results[source.step].outputs, source.field)
        except AttributeError as exc:
            raise StepError(f"step '{ref.name}' cannot resolve wiring for input '{field}': step '{source.step}' has no output '{source.field}'") from exc

    wired: dict[str, Any] = {}
    for field, source in ref.wiring.items():
        if isinstance(source, list):
            wired[field] = [resolve_one(field, s) for s in source]
        else:
            wired[field] = resolve_one(field, source)
    return {**ref.params, **wired}  # wiring overrides params


def _resolve_input_keys(ref, inbound_keys: dict[str, Any], results: dict[str, StepResult]) -> dict[str, Any]:
    """The provenance half of `_resolve_wiring`: for each input this
    sub-step wires, the cache key of whatever produced it (see
    `shinobi.cache`).

    An `OutputRef` resolves against the producing step's result; an
    `InputRef` reaches past the recipe boundary to `inbound_keys` -- the
    provenance the enclosing recipe was itself handed -- so a nested recipe
    doesn't sever the chain. Fields whose producer has no key (caching
    disabled, or an uncacheable producer) are omitted rather than recorded
    as `None`, so enabling caching part-way up a pipeline doesn't rewrite
    the keys of steps that had no provenance to begin with.

    `ref.params` are deliberately absent: a constant declared on the step is
    already hashed by value as an ordinary param, and nothing produced it.
    """

    def key_of(source: InputRef | OutputRef) -> Any:
        if isinstance(source, InputRef):
            return inbound_keys.get(source.field)
        producer = results.get(source.step)
        return producer.provenance_key(source.field) if producer is not None else None

    keys: dict[str, Any] = {}
    for field, source in ref.wiring.items():
        if isinstance(source, list):
            resolved = [key_of(one) for one in source]
            # `any`, not `all`, on purpose. Presence here also suppresses the
            # content hash (see `compute_cache_key`), and for a field that is
            # also a declared output of this step there is no content hash to
            # fall back to -- it is excluded either way -- so requiring every
            # element to have a key would leave such a field keyed on nothing
            # at all. Partial provenance still invalidates on the elements it
            # does cover.
            if any(key is not None for key in resolved):
                keys[field] = resolved
        else:
            resolved_one = key_of(source)
            if resolved_one is not None:
                keys[field] = resolved_one
    return keys


class ScatterError(ValueError):
    """A scattered step received inconsistent inputs: a scatter field is not
    a list, or two scatter fields declared for the same step have different
    lengths.
    """


def _build_scatter_slices(ref: StepRef, sub_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-slice kwargs for a scattered step.

    Every scatter field must be a list and all must have the same length.
    Each returned slice is a copy of `sub_kwargs` with the scattered fields
    replaced by their element at that index.
    """
    spec = ref.scatter
    assert spec is not None
    fields = spec.fields
    lengths: set[int] = set()
    for field in fields:
        if field not in sub_kwargs:
            raise ScatterError(f"step '{ref.name}' scatters over '{field}' but no value was supplied for it (wiring or params must provide '{field}')")
        value = sub_kwargs[field]
        if not isinstance(value, list):
            raise ScatterError(f"step '{ref.name}' scatters over '{field}' but the resolved value is {type(value).__name__}, not a list")
        lengths.add(len(value))
    if len(lengths) != 1:
        lengths_str = ", ".join(sorted(str(length) for length in lengths))
        raise ScatterError(f"step '{ref.name}' scatter fields {fields} have different lengths: {lengths_str}")
    n = lengths.pop()
    slices: list[dict[str, Any]] = []
    for i in range(n):
        slice_kwargs = dict(sub_kwargs)
        for field in fields:
            slice_kwargs[field] = sub_kwargs[field][i]
        slices.append(slice_kwargs)
    return slices


def _scatter_inputs_model(scope: Scope, scatter_fields: set[str]) -> type[BaseModel]:
    """Model for the aggregated inputs of a scattered step.

    Scattered fields become lists of their element type; non-scattered
    fields keep their original type and declared default.
    """
    original = scope.inputs_model
    fields: dict[str, tuple[Any, Any]] = {}
    for name, field in original.model_fields.items():
        if name in scatter_fields:
            inner = field.annotation if field.annotation is not None else Any
            fields[name] = (list[inner], ...)
        elif field.is_required():
            fields[name] = (field.annotation, ...)
        elif field.default_factory is not None:
            fields[name] = (field.annotation, Field(default_factory=field.default_factory))
        else:
            fields[name] = (field.annotation, field.default)
    return create_model(f"{original.__name__}ScatterInputs", **fields)


def _scatter_outputs_model(scope: Scope) -> type[BaseModel]:
    """Model for the gathered outputs of a scattered step.

    Every output field becomes a list of its original type, with one element
    per slice.
    """
    original = scope.outputs_model
    fields: dict[str, tuple[Any, Any]] = {}
    for name, field in original.model_fields.items():
        inner = field.annotation if field.annotation is not None else Any
        fields[name] = (list[inner], ...)
    return create_model(f"{original.__name__}ScatterOutputs", **fields)


def _scatter_kind(scope: Scope) -> str:
    """The `StepResult.kind` value for an empty/zero-length scatter step."""
    if isinstance(scope, Recipe):
        return "recipe"
    if isinstance(scope, Cab):
        return "cab"
    return "pyfunc"


def _aggregate_scatter_results(
    scope: Scope,
    scatter_fields: list[str],
    sub_kwargs: dict[str, Any],
    slices: list[StepResult],
) -> StepResult:
    """Gather per-slice results into a single StepResult.

    Outputs are gathered into lists (one element per slice). Inputs are the
    list-valued scatter fields plus the shared scalar fields. If any slice
    failed, outputs are empty lists and the returncode is the first failure
    by slice index.
    """
    scatter_set = set(scatter_fields)
    InputsModel = _scatter_inputs_model(scope, scatter_set)
    OutputsModel = _scatter_outputs_model(scope)

    if any(s.returncode != 0 for s in slices):
        outputs_data = {name: [] for name in scope.outputs_model.model_fields}
        returncode = next(s.returncode for s in slices if s.returncode != 0)
    else:
        outputs_data = {name: [getattr(s.outputs, name) for s in slices] for name in scope.outputs_model.model_fields}
        returncode = 0

    inputs_data = {name: value for name, value in sub_kwargs.items() if name in scope.inputs_model.model_fields}

    outputs = OutputsModel(**outputs_data)
    inputs = InputsModel(**inputs_data)

    stdout = "\n".join(s.stdout for s in slices if s.stdout)
    stderr = "\n".join(s.stderr for s in slices if s.stderr)
    return StepResult(
        name=scope.name,
        returncode=returncode,
        outputs=outputs,
        inputs=inputs,
        stdout=stdout,
        stderr=stderr,
        kind=slices[0].kind if slices else _scatter_kind(scope),
        backend=slices[0].backend if slices else None,
        image=scope.image,
        image_digest=slices[0].image_digest if slices else None,
        containerized=any(s.containerized for s in slices),
        sandboxed=any(s.sandboxed for s in slices),
        # Every slice is keyed independently, but downstream wires the
        # *gathered* result -- so its provenance is all the slices' keys
        # together, and a change in any one of them invalidates dependents.
        cache_key=combine_keys([s.cache_key for s in slices]),
    )


def _run_recipe(
    recipe: Recipe,
    prepared: dict[str, Any],
    backend_name: str,
    config: AppConfig | None,
    cache_enabled: bool = False,
    cache_dir: str = "",
    cache_path: str = "",
    stream: bool = True,
    provenance: bool = False,
    sandbox: bool = False,
    input_keys: dict[str, Any] | None = None,
) -> StepResult:
    """Topological wavefront scheduler over the recipe's declared DAG.

    Steps run on a `ThreadPoolExecutor` (threads park on the blocking
    `Backend.run`); a step becomes ready only once every step it wires an
    output from has completed. A step declared with `scatter` expands into
    N parallel slices at runtime; the step only completes when every slice
    has finished, and its outputs are gathered into lists (one element per
    slice).

    The ready set is drained lowest-declaration-index first, so
    `max_workers=1` reproduces exact sequential declaration order. On the
    first failure (non-zero returncode) or worker exception, no further
    steps are submitted, but already-running steps drain (a launched job
    can't be honestly cancelled). All aggregation -- stdout, stderr,
    outputs, the winning returncode -- is done in declaration order regardless
    of completion order, so results are deterministic.

    This is also where cache provenance is threaded (see `shinobi.cache`):
    the recipe knows which step produced each of its sub-steps' wired
    inputs, so it resolves that into per-field upstream cache keys
    (`_resolve_input_keys`) and hands them to each sub-step's `_dispatch`.
    `input_keys` is the same thing arriving from *outside*, for a recipe
    nested in another recipe.
    """
    config = config or AppConfig.load()  # resolve once; workers never call load()
    graph = build_graph(recipe)
    max_workers = recipe.max_workers or config.execution.max_workers

    results: dict[str, StepResult] = {}
    indeg = [len(graph.deps[i]) for i in range(len(graph.names))]
    ready: list[int] = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(ready)
    failures: list[tuple[int, StepResult]] = []
    errors: list[tuple[int, BaseException]] = []
    stop = False

    # State for in-flight scattered steps: the original list-valued sub_kwargs
    # (needed to build the aggregated inputs model) and per-slice results.
    scatter_sub_kwargs: dict[int, dict[str, Any]] = {}
    slice_results: dict[int, list[StepResult | None]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future, tuple[int, int | None]] = {}

        def _release_dependents(i: int) -> None:
            for dependent in graph.dependents[i]:
                indeg[dependent] -= 1
                if indeg[dependent] == 0:
                    heapq.heappush(ready, dependent)

        def _step_completed(i: int, res: StepResult) -> None:
            nonlocal stop
            results[recipe.steps[i].name] = res
            if res.returncode != 0:
                failures.append((i, res))
                stop = True
                return
            _release_dependents(i)

        def _submit_step(i: int) -> None:
            ref = recipe.steps[i]
            sub_kwargs = _resolve_wiring(ref, prepared, results)
            sub_input_keys = _resolve_input_keys(ref, input_keys or {}, results)
            # An unrolled loop iteration whose predecessor already converged
            # does no work: it hands the same body step's previous outputs
            # on, completing immediately without occupying a worker. The
            # sentinel producer is guaranteed complete -- add_loop gives
            # every iteration a dependency edge to it.
            if should_skip(ref, results):
                prev = results[ref.loop.prev_step]
                logger.info("step %s%s: skipped (loop '%s' converged)", f"{cache_path}." if cache_path else "", ref.name, ref.loop.loop)
                _step_completed(i, passthrough_result(ref, prev, ref.step.inputs_model(**_prepare_inputs(ref.step, sub_kwargs))))
                return
            if ref.scatter is not None:
                slices = _build_scatter_slices(ref, sub_kwargs)
                if not slices:
                    # Zero-length scatter: produce an empty aggregated result
                    # and immediately release dependents.
                    _step_completed(
                        i,
                        _aggregate_scatter_results(ref.step, ref.scatter.fields, sub_kwargs, []),
                    )
                    return
                scatter_sub_kwargs[i] = sub_kwargs
                slice_results[i] = [None] * len(slices)
                for slice_idx, slice_kwargs in enumerate(slices):
                    fut = pool.submit(
                        _dispatch,
                        ref.step,
                        ref.func,
                        _recipe_backend=backend_name,
                        _recipe_cache=cache_enabled,
                        _recipe_cache_dir=cache_dir,
                        _recipe_stream=stream,
                        _recipe_provenance=provenance,
                        _recipe_sandbox=sandbox,
                        _cache_path=f"{cache_path}.{ref.name}[{slice_idx}]",
                        _config=config,
                        _input_keys=sub_input_keys,
                        **slice_kwargs,
                    )
                    futures[fut] = (i, slice_idx)
            else:
                fut = pool.submit(
                    _dispatch,
                    ref.step,
                    ref.func,
                    _recipe_backend=backend_name,
                    _recipe_cache=cache_enabled,
                    _recipe_cache_dir=cache_dir,
                    _recipe_stream=stream,
                    _recipe_provenance=provenance,
                    _recipe_sandbox=sandbox,
                    _cache_path=f"{cache_path}.{ref.name}",
                    _config=config,
                    _input_keys=sub_input_keys,
                    **sub_kwargs,
                )
                futures[fut] = (i, None)

        def submit_ready() -> None:
            """Submit as many ready steps as there is worker capacity for.

            Pops from `ready` (a min-heap of step indices, so lowest
            declaration-index steps are drained first) until `ready` is
            empty, `stop` is set, or `futures` is at `max_workers` capacity.
            """
            while ready and not stop and len(futures) < max_workers:
                i = heapq.heappop(ready)
                _submit_step(i)

        submit_ready()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                i, slice_idx = futures.pop(fut)
                try:
                    res = fut.result()
                except BaseException as exc:  # noqa: BLE001 -- re-raised below
                    errors.append((i, exc))
                    stop = True
                    continue
                if slice_idx is None:
                    _step_completed(i, res)
                else:
                    slice_results[i][slice_idx] = res
                    completed = slice_results[i]
                    if all(r is not None for r in completed):
                        ref = recipe.steps[i]
                        aggregated = _aggregate_scatter_results(
                            ref.step,
                            ref.scatter.fields,
                            scatter_sub_kwargs[i],
                            [r for r in completed if r is not None],
                        )
                        _step_completed(i, aggregated)
                        del slice_results[i]
                        del scatter_sub_kwargs[i]
            submit_ready()

    # A worker exception (e.g. a bad override's ValidationError) propagates
    # out of the recipe, first-by-declaration if several occurred. Add the
    # recipe step context to the message so the caller can tell *which* step
    # failed, without losing the original exception type for Shinobi errors.
    if errors:
        i, exc = min(errors, key=lambda e: e[0])
        ref_name = recipe.steps[i].name
        msg = f"step '{ref_name}' in recipe '{recipe.name}' failed: {exc}"
        if isinstance(exc, ShinobiError):
            raise type(exc)(msg) from exc
        raise StepError(msg) from exc

    # If any step failed, surface its error before we try to collect declared
    # outputs -- a failed step's outputs model may be hollow and missing the
    # field, which would mask the real failure behind an AttributeError.
    if failures:
        i, failed = min(failures, key=lambda f: f[0])
        ref_name = recipe.steps[i].name
        raise CabRunError(f"step '{ref_name}' in recipe '{recipe.name}' failed (returncode {failed.returncode})")

    ordered = [ref.name for ref in recipe.steps if ref.name in results]
    outputs = {field: getattr(results[out_ref.step].outputs, out_ref.field) for field, out_ref in recipe.output_wiring.items() if out_ref.step in results}
    # Per-output provenance for whoever consumes this recipe's outputs: each
    # declared output is keyed by the sub-step that actually produced it, not
    # by the recipe as a whole -- see StepResult.provenance_key.
    output_keys = {
        field: key for field, out_ref in recipe.output_wiring.items() if out_ref.step in results and (key := results[out_ref.step].provenance_key(out_ref.field)) is not None
    }
    return StepResult(
        name=recipe.name,
        returncode=0,
        outputs=recipe.outputs_model(**outputs),
        inputs=recipe.inputs_model(**prepared),
        stdout="\n".join(s for name in ordered if (s := results[name].stdout)),
        stderr="\n".join(s for name in ordered if (s := results[name].stderr)),
        kind="recipe",
        sub_results={name: results[name] for name in ordered},
        output_keys=output_keys,
    )
