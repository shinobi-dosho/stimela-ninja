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
import threading
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field, ValidationError, create_model
from shinobi.backends._stream import display_label, terminate_all
from shinobi.cache import (
    ExecutionIdentity,
    ProvenanceKey,
    as_provenance_key,
    combine_keys,
    compute_cache_key,
    get_cache_manifest,
    invalidate_path_hashes,
    resolve_input_keys,
    set_content_sample,
)
from shinobi.snapshots import SnapshotGuard, announce_run, eligible_fields, get_journal, mutation_paths, new_run_id, reconcile
from shinobi.config import AppConfig
from shinobi.exceptions import CabRunError, DatasetLifecycleUnavailableError, DatasetLifecycleViolationError, ParameterError, ShinobiError, StepError
from shinobi.graph import build_graph
from shinobi.policies import build_argv
from shinobi.resources import Budget, Resources
from shinobi.results import BackendRun, StepResult, explain_returncode
from shinobi.sandbox import (
    absolutize_path_inputs,
    clear_stale_outputs,
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

# Shared "this step declares nothing" footprint. Reserving it is a no-op, so
# an undeclared step never touches the budget's lock.
_NO_RESOURCES = Resources()

# Slice index standing for "this unit is a whole step, not a scatter slice",
# used only inside the scheduler's `pending` heap. Heap entries must be
# order-comparable and `None` cannot be ordered against an int, so the
# sentinel is a number that sorts before every real slice index.
_NOT_A_SLICE = -1

# How far past a budget-blocked head the scheduler will look for work that
# fits alongside it. Purely a **performance** bound, not a correctness one:
# it stops a wide fan-out turning every refusal into a scan of the whole
# ready set. Nothing is lost by the cut-off, because the head is retried on
# every pass and the ready set is re-scanned from the front each time, so a
# candidate beyond the window is simply reconsidered on a later pass.
_BACKFILL_LOOKAHEAD = 8


def _scope_dataset_declarations(scope: Scope, prefix: str = "") -> list[str]:
    """Describe strict dataset fields in a scope tree without inspecting I/O."""

    from shinobi.datasets import dataset_declarations

    declarations = []
    for side, model in (("input", scope.inputs_model), ("output", scope.outputs_model)):
        declarations.extend(f"{prefix}{scope.name} {side} '{name}' ({declaration.profile})" for name, declaration in dataset_declarations(model).items())
    declarations.extend(f"{prefix}{scope.name} access '{access.field}' ({access.mode.value} {access.table.value})" for access in scope.dataset_accesses)
    if isinstance(scope, Recipe):
        for ref in scope.steps:
            declarations.extend(_scope_dataset_declarations(ref.step, f"{prefix}{ref.name}/"))
    return declarations


def _refuse_unenforced_datasets(scope: Scope) -> None:
    declarations = _scope_dataset_declarations(scope)
    if declarations:
        raise DatasetLifecycleUnavailableError(
            "strict CASA/MSv2 dataset annotations are declarative in this release and cannot execute until "
            "validation, staging and recovery enforce the same lifecycle contract; use Path or the legacy loader "
            f"dtype 'MS' for current execution. Declared field(s): {', '.join(declarations)}"
        )


def _leaf_dataset_writes(scope: Scope) -> set[str]:
    """Dataset fields a leaf writes or creates, from declarations alone.

    Declared ``write``/``create`` accesses, plus the inferred shapes that
    access resolution would also treat as writes: an annotated output that
    is not an input (a create), and an annotated field in the shared
    mutated/write-path sets.
    """

    from shinobi.dataset_access import DatasetMode
    from shinobi.datasets import dataset_declarations
    from shinobi.steps.schema import mutated_path_fields, write_path_fields

    inputs = dataset_declarations(scope.inputs_model)
    outputs = dataset_declarations(scope.outputs_model)
    declared = {access.field for access in scope.dataset_accesses if access.mode is not DatasetMode.READ}
    inferred = (set(outputs) - set(inputs)) | ((mutated_path_fields(scope) | write_path_fields(scope)) & (set(inputs) | set(outputs)))
    explicit_reads = {access.field for access in scope.dataset_accesses if access.mode is DatasetMode.READ}
    return declared | (inferred - explicit_reads)


def _scope_tree_writes_datasets(scope: Scope) -> bool:
    if isinstance(scope, Recipe):
        return any(_scope_tree_writes_datasets(ref.step) for ref in scope.steps)
    return bool(_leaf_dataset_writes(scope))


def _strict_mutation_cache_issues(scope: Scope, cache: bool | None, recipe_cache: bool | None, cache_dir: str | None, recipe_cache_dir: str | None, config: AppConfig) -> list[str]:
    """Why a strict mutation workflow cannot name or recover its dataset states.

    Exact recovery names every predecessor and successor by the writing
    step's cache key (`shinobi.snapshots`). Caching is therefore turned on
    automatically for a strict writer -- whatever the configured default --
    and only an *explicit* ``cache=False`` refuses: the call argument
    (``ninja run --no-cache``), the writer's own ``Scope.cache`` or an
    enclosing recipe's, with the same precedence as `_dispatch`. Snapshots
    must not be ``off``, which is an explicit escape hatch rather than a
    default. Every writer must also share the workflow's cache directory:
    that is the one journal the workflow reconciles under its claim, and a
    writer journalling elsewhere would leave an interruption nobody
    recovers.
    """

    if config.cache.snapshots.mode == "off":
        return ["cache.snapshots.mode is 'off'; exact MSv2 mutation recovery needs 'auto' or 'copy'"]
    # None = nobody said; a strict writer then caches automatically.
    root = cache if cache is not None else scope.cache if scope.cache is not None else recipe_cache
    root_dir = cache_dir or scope.cache_dir or recipe_cache_dir or config.cache.dir
    issues: list[str] = []

    def visit(current: Scope, inherited: bool | None, prefix: str) -> None:
        if isinstance(current, Recipe):
            for ref in current.steps:
                child = ref.step
                effective = child.cache if child.cache is not None else inherited
                visit(child, effective, f"{prefix}{ref.name}")
            return
        if not _leaf_dataset_writes(current):
            return
        label = prefix or current.name
        if inherited is False:
            issues.append(f"writing step {label!r} has caching explicitly disabled; exact MSv2 mutation recovery names dataset states by cache key (drop --no-cache/cache=False)")
        # A child's own directory beats the recipe's resolved one in
        # `_dispatch`; only the root scope sees the explicit argument.
        if current is not scope and current.cache_dir is not None and Path(current.cache_dir).resolve() != Path(root_dir).resolve():
            issues.append(
                f"writing step {label!r} uses cache_dir {current.cache_dir!r}, not the workflow's {root_dir!r}; its journal would be outside the recovery the workflow performs"
            )

    visit(scope, root, "")
    return issues


def _dataset_execution_backends(scope: Scope, func: Callable | None, inherited: str) -> tuple[str, ...]:
    """Validate the deliberately small contained native execution shape.

    Writes and creates are admitted here; whether the workflow can promise
    exact recovery for them is decided by the mutation lifecycle.
    """

    from shinobi.dataset_access import scope_has_dataset_contract, scope_tree_has_dataset_contract
    from shinobi.datasets import DatasetKind, dataset_declarations
    from shinobi.steps.pyfunc import PystepCallable

    backends: set[str] = set()

    def visit(current: Scope, current_func: Callable | None, backend_name: str, *, nested: bool = False, scattered: bool = False) -> None:
        has_contract = scope_has_dataset_contract(current)
        if isinstance(current, Recipe):
            # Recipe boundary annotations describe values crossing the
            # boundary; access contracts still belong to the atomic leaves.
            # ``Recipe`` validation already rejects recipe-level accesses.
            if current_func is not None and scope_tree_has_dataset_contract(current):
                raise DatasetLifecycleUnavailableError("contained MSv2 execution refused: recipe orchestration functions are not an ordinary dataset route")
            for ref in current.steps:
                child_has_contract = scope_tree_has_dataset_contract(ref.step)
                if isinstance(ref.step, Recipe) and child_has_contract:
                    raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: step {ref.name!r} is a nested dataset recipe; flatten it")
                if ref.scatter is not None and child_has_contract:
                    raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: step {ref.name!r} scatters a dataset contract")
                visit(ref.step, ref.func, ref.step.backend or backend_name, nested=True, scattered=ref.scatter is not None)
            return
        # One outer lifecycle claim covers the whole flat recipe, not merely
        # the annotated leaves.  Letting an unannotated sibling select a
        # container, venv or scheduler would put an unobserved route inside
        # that claim and could hand it the MS through an ordinary Path.
        if backend_name != "native":
            raise DatasetLifecycleUnavailableError(
                f"contained MSv2 execution refused: backend {backend_name!r} is not the supported native local route for leaf {current.name!r}; every leaf under the lifecycle must use native"
            )
        if not has_contract:
            return
        if nested and isinstance(current, Recipe):
            raise DatasetLifecycleUnavailableError("contained MSv2 execution refused: nested dataset recipes are not supported")
        if scattered:
            raise DatasetLifecycleUnavailableError("contained MSv2 execution refused: scattered dataset access is not supported")
        if isinstance(current, Cab):
            if current_func is not None:
                raise DatasetLifecycleUnavailableError("contained MSv2 execution refused: cab orchestration functions are not an ordinary dataset route")
            if current.flavour != "binary":
                raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: cab flavour {current.flavour!r} is not the supported binary route")
        elif not isinstance(current_func, PystepCallable):
            raise DatasetLifecycleUnavailableError(
                "strict dataset annotations remain declarative for manual Scope execution; use a Cab or @pystep for the contained native lifecycle"
            )
        declarations = {
            **dataset_declarations(current.inputs_model),
            **dataset_declarations(current.outputs_model),
        }
        if any(token in name for name in declarations for token in (".", "[]", ".*")):
            names = ", ".join(sorted(declarations))
            raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: nested dataset fields are not direct: {names}")
        unsupported = sorted(name for name, declaration in declarations.items() if declaration.kind is not DatasetKind.MEASUREMENT_SET_V2)
        if unsupported:
            raise DatasetLifecycleUnavailableError(
                "strict dataset contract cannot execute until validation, staging and recovery support this route; "
                "the contained lifecycle accepts only MeasurementSetV2 fields, not " + ", ".join(unsupported)
            )
        backends.add(backend_name)

    visit(scope, func, inherited)
    return tuple(sorted(backends))


def _any_resources(recipe: Recipe) -> bool:
    """Whether anything in `recipe`, at any nesting depth, declares a
    resource footprint -- i.e. whether admission control is worth building.

    Recurses into nested recipes because that is where the footprints
    actually live: a caracal-shaped pipeline declares nothing at the top
    level and everything two levels down.
    """
    for ref in recipe.steps:
        scope = ref.step
        if isinstance(scope, Recipe):
            if _any_resources(scope):
                return True
        elif scope.resources is not None:
            return True
    return False


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


def _local_execution_identity(scope: Scope, ctx: "ExecContext", *, pinned: bool) -> ExecutionIdentity | None:
    """Resolve cache-relevant software facts for an in-process dispatch.

    ``None`` means the selected environment could not be fingerprinted, in
    which case the caller executes without reading or writing a reusable
    cache entry.  A cache optimization must never turn an unknown mutable
    environment into a hit.
    """
    backend = ctx.resolve_backend_name()
    image_digest = None
    if pinned and scope.image:
        from shinobi.backends.container import CONTAINER_RUNTIMES, _pin_image

        if backend in CONTAINER_RUNTIMES:
            _pinned_ref, image_digest = _pin_image(backend, scope.image)
            if image_digest is None:
                return None

    resolved_venv = None
    resolved_venv_digest = None
    if backend == "venv":
        from shinobi.backends.venv import resolve_venv, venv_digest

        resolved = resolve_venv(scope.venv, ctx._config)
        if resolved is not None:
            resolved_venv = str(resolved)
            resolved_venv_digest = venv_digest(resolved)
            if resolved_venv_digest is None:
                return None
    return ExecutionIdentity(
        image_digest=image_digest,
        venv=resolved_venv,
        venv_digest=resolved_venv_digest,
    )


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
        validated = _validate_inputs(scope, kwargs)
    prepared: dict[str, Any] = {}
    for name in type(validated).model_fields:
        if scope.mutability_of(name) is Mutability.MUTABLE:
            value = kwargs[name] if name in kwargs else getattr(validated, name)
        else:
            value = copy.deepcopy(getattr(validated, name))
        prepared[name] = value
    # Dynamically-named params and typos alike land in model_extra when the
    # inputs model allows extras. Carry both through (immutable), but do not
    # describe every extra as pattern-matched: build_argv emits only names a
    # Cab's input_patterns actually admit.
    extras = validated.model_extra or {}
    if extras:
        if isinstance(scope, Cab):
            matched = sorted(name for name in extras if scope.match_pattern(name) is not None)
            unmatched = sorted(set(extras) - set(matched))
        else:
            matched = []
            unmatched = sorted(extras)
        if matched:
            warnings.warn(
                f"'{scope.name}': parameter(s) {matched} matched a dynamic "
                "parameter pattern and are passed through to the tool as-is -- "
                "shinobi has no declared field for them, so it cannot type/range-"
                "check them the way it does for the cab's declared parameters.",
                stacklevel=2,
            )
        if unmatched:
            if isinstance(scope, Cab):
                detail = "they are retained in the prepared inputs but are not passed to the tool command line."
            else:
                detail = "they are retained in the prepared inputs but are not treated as dynamic tool parameters."
            warnings.warn(
                f"'{scope.name}': extra parameter(s) {unmatched} were accepted by the input model but did not match a dynamic parameter pattern; " + detail,
                stacklevel=2,
            )
    for name, value in extras.items():
        prepared[name] = copy.deepcopy(value)
    return prepared


def _validate_inputs(scope: Scope, kwargs: dict[str, Any]) -> BaseModel:
    """Validate one raw input mapping with dispatch's public error contract."""
    try:
        return scope.inputs_model(**kwargs)
    except ValidationError as exc:
        raise ParameterError(f"{scope.name}: parameter validation failed:\n{exc}") from exc


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
        clear_outputs: bool = True,
        input_keys: dict[str, Any] | None = None,
        budget: Budget | None = None,
        run_id: str = "",
        validated_inputs: BaseModel | None = None,
        leaf_inputs: dict[int, tuple[BaseModel, bool]] | None = None,
        dataset_lifecycle: Any | None = None,
        publication_gate: Callable[[Callable[[], None]], None] | None = None,
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
            clear_outputs: Whether a re-run deletes the previous run's
                product from each declared output path the tool writes
                directly (`sandbox.clear_stale_outputs`).
            input_keys: Per-input-field cache keys of the steps that
                produced this step's wired inputs (see `shinobi.cache`).
                Only an enclosing recipe can know these; a top-level call
                has none.
            budget: Admission control shared with the enclosing run, if any
                (see `shinobi.resources`). Only a `Recipe` scope uses it, by
                forwarding it to its own `_run_recipe` -- which is the whole
                point: every nested recipe's scheduler admits against the
                *same* budget, so branches cannot each independently decide
                they own the machine.
            validated_inputs: An already-validated instance for this exact
                raw input mapping. Internal callers use it to share one
                default/default-factory evaluation with ownership discovery.
            leaf_inputs: Claim-time validated inputs for a recipe's steps,
                keyed by StepRef identity (see `scope_path_accesses`), plus
                whether each model is fully knowable and can be reused as-is.
        """
        self.scope = scope
        self._raw = raw_inputs
        self.inputs = validated_inputs if validated_inputs is not None else _validate_inputs(scope, raw_inputs)
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
        # Whether a re-run clears the previous product from the declared
        # output paths the tool writes directly (sandbox.clear_stale_outputs).
        # Read by the pystep paths and threaded to _run_cab via ctx.run().
        self._clear_outputs = clear_outputs
        # Upstream provenance for this step's wired inputs. A Recipe scope is
        # never cached itself, so it doesn't consume these -- it forwards them
        # to _run_recipe, which resolves each sub-step's InputRef wiring
        # against them (see shinobi.cache).
        self._input_keys = input_keys
        self._budget = budget
        # Step-level validated inputs resolved during ownership discovery.
        # A Recipe forwards them to `_run_recipe`; fully knowable steps reuse
        # the exact model, while runtime-dependent ones reuse only defaults.
        self._leaf_inputs = leaf_inputs
        # Non-None only for leaves executing under the top-level contained
        # MSv2 read lifecycle and its already-acquired shared claim.
        self._dataset_lifecycle = dataset_lifecycle
        # The strict dataset wrapper queues every leaf's cache/snapshot and
        # run-manifest publication here until its outer postcondition passes.
        self._publication_gate = publication_gate
        # Identifies the whole top-level dispatch, threaded down like
        # `_config` so every step of one run agrees on it. Names trash
        # directories and stamps manifest entries, both of which are read
        # across runs -- which is why it is a uuid and not a pid.
        self._run_id = run_id

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

    def import_callable(self, name: str, module: str | None = None) -> Callable:
        """Import and return a callable by name.

        If `module` is None, looks up `name` in builtins (e.g. ``print``,
        ``len``). Otherwise imports `module` -- which may be a dotted path of
        any depth (``"astropy.io.fits"``) -- and returns
        `getattr(module, name)`.

        Useful for pysteps that invoke container-only callables (e.g. CASA
        tasks) without triggering linter warnings about missing imports on
        the host.

        **Callable, not function**, and the distinction is not pedantry: what
        a pystep asks for here is a *class* about as often as it is a
        function (``casacore.tables.table``, ``astropy.wcs.WCS``,
        ``astropy.coordinates.SkyCoord``), and numpy hands back neither --
        ``numpy.log10`` is a ``ufunc`` and ``numpy.linspace`` an
        ``_ArrayFunctionDispatcher``. The one property all of them share is
        the one the caller is about to rely on, so that is the one checked.

        **The check is what makes the split point safe.** This always returns
        an *attribute of* a module, never a module: name the full dotted path
        in `module` and the attribute in `name`. Splitting it at the wrong
        point asks for a submodule as though it were an attribute of its
        parent, and whether that raises depends on the package --
        ``("ndimage", "scipy")`` quietly returns the *module* on any scipy new
        enough to expose its subpackages lazily, while ``("tables",
        "casacore")`` and ``("fits", "astropy.io")`` raise `AttributeError`.
        Version-dependent luck is a bad way to learn the path was wrong, so a
        non-callable result is a `TypeError` here rather than a mystery at the
        first call site. Use `import_module` when the step wants the module.
        """
        obj = getattr(builtins, name) if module is None else getattr(importlib.import_module(module), name)
        if not callable(obj):
            where = f"{module}.{name}" if module else f"builtins.{name}"
            hint = (
                f' -- it is a module, so the dotted path is split one segment too early: ask for it with ctx.import_module("{where}"),'
                " or name the attribute you actually want and put the rest in the module argument"
                if isinstance(obj, ModuleType)
                else ""
            )
            raise TypeError(f"import_callable({name!r}, {module!r}) returned {type(obj).__name__}, which is not callable{hint}")
        return obj

    def import_func(self, func: str, module: str | None = None) -> Callable:
        """Deprecated alias for `import_callable`.

        Kept because it is what every existing pystep calls, and renaming a
        method a container shim lifts by source is not a change worth
        breaking a caller over. The name was wrong -- half of what real
        pysteps ask for is a class, not a function -- but the behaviour is
        `import_callable`'s, including its callability check, so there is
        exactly one implementation to reason about.
        """
        return self.import_callable(func, module)

    def import_module(self, module: str) -> ModuleType:
        """Import and return a module by its full dotted path.

        The companion to `import_callable` for the case where a pystep wants
        the module itself rather than one of its attributes -- ``np =
        ctx.import_module("numpy")``, ``fits =
        ctx.import_module("astropy.io.fits")``. `import_callable` cannot
        express this: its one-argument form is a builtins lookup, and its
        two-argument form ends in a `getattr`, which does not reach a
        submodule (`import_module` does not bind one onto its parent
        package) -- and now rejects the module it would occasionally have
        returned by accident.

        Kept as a separate name rather than folded into `import_callable`'s
        one-argument form so the return type stays predictable per method
        instead of varying with whatever the name happens to resolve to.
        """
        return importlib.import_module(module)

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
                label=display_label(self._cache_path),
                stream=self._stream,
                pin=self._pin,
                sandbox_root=self._sandbox_root,
                clear_outputs=self._clear_outputs,
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
                budget=self._budget,
                run_id=self._run_id,
                leaf_inputs=self._leaf_inputs,
                validated_inputs=validated,
                dataset_lifecycle=self._dataset_lifecycle,
                publication_gate=self._publication_gate,
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


_snapshot_warned: set[tuple[str, str]] = set()


def _snapshot_guard(
    scope: Scope,
    ctx: "ExecContext",
    cache_dir: str,
    cache_path: str,
    cache_key: str | None,
    run_id: str,
    input_keys: dict[str, Any] | None,
    wired_fields: set[str] | None,
    slice_index: int | None,
    config: AppConfig,
    success_record: Path | None,
    success_step_path: str | None,
) -> SnapshotGuard | None:
    """A `SnapshotGuard` for this step, or `None` if it mutates nothing.

    A step whose mutated fields are *all* excluded still gets a guard, and
    that is deliberate. It cannot be snapshotted, but it does write, and
    something has to record that -- otherwise the write is invisible to the
    journal and a later restore rolls the path back over it, discarding work
    that nothing will regenerate because the step that made it cache-hits.

    Every field it declines is warned about exactly once per process --
    these are properties of a recipe's shape, not of a run, so repeating
    them per step per run would bury the ones that matter.
    """
    prepared = ctx.prepare_inputs()
    protected, excluded = eligible_fields(scope, prepared, input_keys, wired_fields, slice_index is not None)
    for exclusion in excluded:
        if (cache_path, exclusion.field) not in _snapshot_warned:
            _snapshot_warned.add((cache_path, exclusion.field))
            logger.warning(
                "step %s: no snapshot protection for mutated input '%s' -- %s. This step runs against live disk, exactly as it would with snapshots off.",
                cache_path,
                exclusion.field,
                exclusion.reason,
            )
    # Paths written through an excluded field: unprotectable, but not
    # unrecordable (see `SnapshotGuard._taint_excluded`).
    all_mutations = mutation_paths(scope, prepared)
    tainting = {exclusion.field: all_mutations[exclusion.field] for exclusion in excluded if exclusion.field in all_mutations}
    if not protected and not tainting:
        return None
    return SnapshotGuard(
        journal=get_journal(cache_dir),
        step_path=cache_path,
        cache_key=cache_key,
        run_id=run_id,
        fields=protected,
        input_keys=input_keys,
        wired_fields=wired_fields,
        force_copy=config.cache.snapshots.mode == "copy",
        tainting=tainting,
        success_record=success_record,
        success_step_path=success_step_path,
    )


def _strict_snapshot_guard(
    scope: Scope,
    ctx: "ExecContext",
    leaf: Any,
    cache_dir: str,
    cache_path: str,
    cache_key: str,
    run_id: str,
    input_keys: dict[str, Any] | None,
    wired_fields: set[str] | None,
    boundary_fields: frozenset[str],
    slice_index: int | None,
    config: AppConfig,
) -> SnapshotGuard:
    """Tier 1 under a strict MSv2 policy for one writing leaf.

    Uses the same eligibility rules as `_snapshot_guard`, but a strict field
    Tier 1 would decline is refused rather than run unprotected. A field
    wired from the top-level recipe's own input is a boundary path, exactly
    as if the dataset had been passed to the step directly.
    """

    from shinobi.dataset_lifecycle import LeafMutationError

    wired = set(wired_fields or ()) - set(boundary_fields) if wired_fields is not None else None
    _protected, excluded = eligible_fields(scope, ctx.prepare_inputs(), input_keys, wired, slice_index is not None)
    blocked = [exclusion for exclusion in excluded if exclusion.field in leaf.fields]
    if blocked:
        raise LeafMutationError(
            f"strict MSv2 mutation of {cache_path!r} refused before launch: " + "; ".join(f"'{exclusion.field}' cannot be protected: {exclusion.reason}" for exclusion in blocked)
        )
    return SnapshotGuard(
        journal=get_journal(cache_dir),
        step_path=cache_path,
        cache_key=cache_key,
        run_id=run_id,
        fields=dict(leaf.fields),
        input_keys=input_keys,
        wired_fields=wired,
        force_copy=config.cache.snapshots.mode == "copy",
        success_record=leaf.lifecycle.store.path,
        success_step_path=cache_path,
        strict=leaf.policy(),
        success_kind="dataset-lifecycle",
    )


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
    _wired_fields: set[str] | None = None,
    _budget: Budget | None = None,
    _run_id: str | None = None,
    _slice_index: int | None = None,
    _execution_identity: ExecutionIdentity | None = None,
    _snapshot_success_record: Path | None = None,
    _snapshot_success_step_path: str | None = None,
    _result_commit: Callable[[StepResult, Callable[[], None]], None] | None = None,
    _publication_gate: Callable[[Callable[[], None]], None] | None = None,
    _workspace_claimed: bool = False,
    _dataset_lifecycle: Any | None = None,
    _validated_inputs: BaseModel | None = None,
    _leaf_inputs: dict[int, tuple[BaseModel, bool]] | None = None,
    _boundary_fields: frozenset[str] = frozenset(),
    overwrite_steps: Sequence[str] = (),
    **kwargs: Any,
) -> StepResult:
    config = _config or AppConfig.load()
    run_id = _run_id or new_run_id()
    dataset_declarations = _scope_dataset_declarations(scope)
    if overwrite_steps and (not dataset_declarations or _dataset_lifecycle is not None or _cache_path is not None):
        raise DatasetLifecycleUnavailableError(
            "overwrite applies only to a top-level workflow whose steps create a strict MeasurementSetV2; declare the created field as MeasurementSetV2"
        )
    if dataset_declarations and _dataset_lifecycle is None:
        # Strict execution enters only through this top-level wrapper.  A
        # nested invocation without its parent's lifecycle token would be an
        # unclaimed reader and remains refused.
        if _cache_path is not None or _workspace_claimed:
            _refuse_unenforced_datasets(scope)
        from shinobi.dataset_lifecycle import (
            DatasetLifecycle,
            DatasetLifecyclePhase,
            claim_covers_accesses,
            claim_covers_snapshot,
            observe_roots,
            overwrite_created_datasets,
            pending_dataset_recovery,
            resolve_lifecycle_snapshot,
        )
        from shinobi.ownership import (
            WorkspaceOwnershipError,
            acquire_workspace,
            contained_access_issues,
            ownership_workspace,
            scope_path_accesses,
        )
        from shinobi.steps.schema import paths_overlap

        launch_workspace = Path.cwd().resolve()
        root_backend = backend or scope.backend or _recipe_backend or config.backend.default
        # One skeleton, two capabilities: a workflow in which any leaf writes
        # or creates a strict dataset runs the mutation lifecycle (exclusive
        # claim, recovery, per-leaf postconditions); otherwise the read one.
        mutation = _scope_tree_writes_datasets(scope)
        noun = "mutation" if mutation else "read"
        lifecycle = DatasetLifecycle.start(
            workspace=launch_workspace,
            attempt_id=run_id,
            scope=scope.name,
            backends=(root_backend,),
            mutation=mutation,
        )
        lease = None
        executing = False
        primary_error: BaseException | None = None
        validated_inputs = None
        leaf_inputs = None
        planned = None
        pending_publications: list[Callable[[], None]] = []
        publication_lock = threading.Lock()

        def hold_publication(action: Callable[[], None]) -> None:
            with publication_lock:
                pending_publications.append(action)

        def publish_validated_result() -> None:
            with publication_lock:
                actions = tuple(pending_publications)
                pending_publications.clear()
            for action in actions:
                action()

        def transition_preserving(
            phase: DatasetLifecyclePhase,
            reason: str,
            primary: BaseException,
            **changes: Any,
        ) -> None:
            try:
                lifecycle.transition(phase, reason, **changes)
            except BaseException as transition_exc:
                primary.add_note(
                    f"dataset lifecycle record update also failed: {type(transition_exc).__name__}: {transition_exc}; inspect the pending attempt record before recovery"
                )
                logger.exception("contained MSv2 %s failed and its lifecycle record update also failed", noun)

        def resolve_snapshot():
            return resolve_lifecycle_snapshot(
                scope,
                validated_inputs,
                workspace=launch_workspace,
                validated_steps=leaf_inputs,
                mutation=mutation,
            )

        try:
            backends = _dataset_execution_backends(scope, func, root_backend)
            validated_inputs = _validated_inputs if _validated_inputs is not None else _validate_inputs(scope, kwargs)
            effective_cache_dir = cache_dir or scope.cache_dir or _recipe_cache_dir or config.cache.dir
            if mutation:
                cache_issues = _strict_mutation_cache_issues(scope, cache, _recipe_cache, cache_dir, _recipe_cache_dir, config)
                if cache_issues:
                    raise DatasetLifecycleUnavailableError("contained MSv2 mutation refused: " + "; ".join(cache_issues))
            if overwrite_steps:
                if not mutation:
                    raise DatasetLifecycleUnavailableError("overwrite applies only to steps that create a strict MeasurementSetV2; this workflow only reads")
                # Before planning, which refuses an existing CREATE target:
                # this is the one explicit way past that refusal.
                overwrites = overwrite_created_datasets(
                    scope,
                    validated_inputs,
                    tuple(overwrite_steps),
                    workspace=launch_workspace,
                    cache_dir=effective_cache_dir,
                    run_id=run_id,
                )
                for record in overwrites:
                    if record.existed:
                        logger.warning("overwrite: deleted %s (step %s, field %s)", record.path, record.step, record.field)
                if overwrites:
                    logger.warning("overwrite: invalidated cached results of %s", ", ".join(overwrites[0].invalidated))
                lifecycle.amend(overwrites=overwrites)
            accesses, leaf_inputs = scope_path_accesses(scope, validated_inputs, workspace=launch_workspace)
            planned = resolve_snapshot()
            lifecycle.transition(
                DatasetLifecyclePhase.PLANNED,
                (
                    f"resolved {len(planned.observations) + len(planned.absent_roots)} contained ordinary MSv2 root(s) for mutation"
                    if mutation
                    else "resolved one contained ordinary MSv2 read closure"
                ),
                backends=backends,
                capability_supported=True,
                planned_accesses=planned.accesses,
                pre_observations=planned.observations,
                **({"absent_roots": planned.absent_roots} if mutation else {}),
            )
            resources = {resource for access in planned.accesses for resource in access.resources}
            access_issues = contained_access_issues(
                scope,
                validated_inputs,
                workspace=launch_workspace,
                dataset_resources=resources,
                validated_steps=leaf_inputs,
            )
            if access_issues:
                raise DatasetLifecycleUnavailableError(f"contained MSv2 {noun} refused: " + "; ".join(access_issues))
            # A dataset's own declared write names exactly one of its closure
            # resources; any other write reaching into a closure is generic.
            overlapping_writes = sorted(
                {path for path, writes in accesses if writes and not (mutation and path in resources) and any(paths_overlap(path, resource) for resource in resources)},
                key=str,
            )
            if overlapping_writes:
                raise DatasetLifecycleUnavailableError(
                    f"contained MSv2 {noun} refused: a generic write overlaps the {'dataset' if mutation else 'read-only'} closure: " + ", ".join(map(str, overlapping_writes))
                )
            writable_paths = [path for path, writes in accesses if writes]
            claim_workspace = ownership_workspace(launch_workspace, writable_paths) if writable_paths else launch_workspace
            lease = acquire_workspace(
                claim_workspace,
                run_id,
                kind="local",
                accesses=accesses,
            )
            lifecycle.transition(
                DatasetLifecyclePhase.CLAIMED,
                "acquired ownership covering every closure resource at its declared strength" if mutation else "acquired ownership covering every read-only closure resource",
                claim=lease.owner,
            )
            revalidated = resolve_snapshot()
            covered = not claim_covers_accesses(lease.owner, revalidated.accesses) if mutation else claim_covers_snapshot(lease.owner, revalidated)
            if revalidated != planned or not covered:
                raise DatasetLifecycleUnavailableError(f"contained MSv2 {noun} refused before backend execution: the access plan or observation changed after claim")
            if mutation:
                # Crash recovery for the datasets this workflow writes, under
                # its exclusive claim -- which is what makes it safe without
                # the best-effort "am I alone" probe: no cooperating writer
                # can hold these paths now. Roots it only reads stay under a
                # shared claim and are left to the check below.
                written_roots = {access.root for access in revalidated.accesses if access.writes and access.root is not None}
                notes = reconcile(effective_cache_dir, get_cache_manifest(effective_cache_dir), paths=written_roots) if written_roots else []
                if notes:
                    for note in notes:
                        logger.warning("dataset recovery: %s", note)
                    lifecycle.amend(recovery=(*lifecycle.record.recovery, *notes))
            # An interrupted writer may have left a journal while snapshots were
            # enabled previously.  The current snapshot policy cannot make that
            # pending mutation safe to recover under a shared read claim.
            pending = pending_dataset_recovery(effective_cache_dir, revalidated)
            if pending:
                raise DatasetLifecycleUnavailableError(
                    f"contained MSv2 {noun} refused before backend execution: pending mutation recovery requires an exclusive writer claim: " + ", ".join(map(str, pending))
                )
            if mutation and notes:
                # Recovery may have rolled a dataset back; the baseline the
                # workflow runs against is the recovered one.
                revalidated = resolve_snapshot()
                lifecycle.amend(pre_observations=revalidated.observations)
                planned = revalidated
            lifecycle.transition(
                DatasetLifecyclePhase.REVALIDATED,
                "post-claim access plan and observation match the claimed baseline",
            )
            lifecycle.transition(
                DatasetLifecyclePhase.EXECUTING,
                f"entering native local dispatch under the {'exclusive' if mutation else 'shared read'} claim",
            )
            planned_roots = tuple(sorted({access.root for access in planned.accesses if access.root is not None}, key=str))
            executing = True
            try:
                result = _dispatch(
                    scope,
                    func,
                    backend=backend,
                    cache=cache,
                    cache_dir=cache_dir,
                    stream=stream,
                    provenance=provenance,
                    sandbox=sandbox,
                    _recipe_backend=_recipe_backend,
                    _recipe_cache=_recipe_cache,
                    _recipe_cache_dir=_recipe_cache_dir,
                    _recipe_stream=_recipe_stream,
                    _recipe_provenance=_recipe_provenance,
                    _recipe_sandbox=_recipe_sandbox,
                    _cache_path=_cache_path,
                    _config=config,
                    _provenance_target=_provenance_target,
                    _input_keys=_input_keys,
                    _wired_fields=_wired_fields,
                    _budget=_budget,
                    _run_id=run_id,
                    _slice_index=_slice_index,
                    _execution_identity=_execution_identity,
                    _snapshot_success_record=_snapshot_success_record,
                    _snapshot_success_step_path=_snapshot_success_step_path,
                    _result_commit=_result_commit,
                    # A mutation workflow validates and publishes per leaf:
                    # each writer's successor must be committed before the
                    # next leaf consumes it, so nothing can wait for the end.
                    _publication_gate=None if mutation else hold_publication,
                    _workspace_claimed=True,
                    _dataset_lifecycle=lifecycle,
                    _validated_inputs=validated_inputs,
                    _leaf_inputs=leaf_inputs,
                    **kwargs,
                )
            except BaseException as exc:
                if mutation:
                    # Each leaf has already validated, rolled back or marked
                    # its own datasets; the workflow record states the end.
                    try:
                        final, _absent = observe_roots(planned_roots, launch_workspace)
                    except BaseException as observe_exc:
                        exc.add_note(f"final dataset observation also failed: {type(observe_exc).__name__}: {observe_exc}")
                        final = ()
                    transition_preserving(
                        DatasetLifecyclePhase.FAILED,
                        f"execution raised {type(exc).__name__}: {exc}",
                        exc,
                        post_observations=final,
                    )
                    raise
                try:
                    post = resolve_lifecycle_snapshot(
                        scope,
                        validated_inputs,
                        workspace=launch_workspace,
                        validated_steps=leaf_inputs,
                    )
                except BaseException as observe_exc:
                    violation = DatasetLifecycleViolationError("strict MSv2 reader failed and its read-only postcondition could not be established")
                    transition_preserving(
                        DatasetLifecyclePhase.FAILED,
                        f"execution raised {type(exc).__name__}; post-execution observation failed: {type(observe_exc).__name__}: {observe_exc}",
                        violation,
                    )
                    raise violation from exc
                if post != planned:
                    violation = DatasetLifecycleViolationError("strict MSv2 reader changed its dataset before failing")
                    transition_preserving(
                        DatasetLifecyclePhase.FAILED,
                        f"execution raised {type(exc).__name__} and the MSv2 changed",
                        violation,
                        post_observations=post.observations,
                    )
                    raise violation from exc
                transition_preserving(
                    DatasetLifecyclePhase.FAILED,
                    f"execution raised {type(exc).__name__}: {exc}",
                    exc,
                    post_observations=post.observations,
                )
                raise
            if mutation:
                try:
                    final, _absent = observe_roots(planned_roots, launch_workspace)
                except BaseException as exc:
                    violation = DatasetLifecycleViolationError("strict MSv2 workflow completed but its final dataset observation could not be established")
                    transition_preserving(DatasetLifecyclePhase.FAILED, f"final observation failed: {type(exc).__name__}: {exc}", violation)
                    raise violation from exc
                # Written roots were validated leaf by leaf; a root nothing
                # declared a write to must be exactly as planned.
                written = {access.root for access in planned.accesses if access.writes}
                baseline = {observation.root: observation for observation in planned.observations}
                changed = sorted((observation.root for observation in final if observation.root not in written and baseline.get(observation.root) != observation), key=str)
                if changed:
                    violation = DatasetLifecycleViolationError("strict MSv2 workflow changed a dataset it declared only as read: " + ", ".join(map(str, changed)))
                    transition_preserving(DatasetLifecyclePhase.FAILED, str(violation), violation, post_observations=final)
                    raise violation
                lifecycle.transition(
                    DatasetLifecyclePhase.VALIDATED,
                    "every leaf's declared postconditions held and read-only roots are unchanged",
                    post_observations=final,
                )
                if result.success:
                    lifecycle.transition(DatasetLifecyclePhase.COMMITTED, "native contained MSv2 mutation workflow committed")
                else:
                    lifecycle.transition(DatasetLifecyclePhase.FAILED, f"native workflow returned non-zero status {result.returncode}")
                return result
            try:
                post = resolve_lifecycle_snapshot(
                    scope,
                    validated_inputs,
                    workspace=launch_workspace,
                    validated_steps=leaf_inputs,
                )
            except BaseException as exc:
                violation = DatasetLifecycleViolationError("strict MSv2 reader completed but its read-only postcondition could not be established")
                transition_preserving(
                    DatasetLifecyclePhase.FAILED,
                    f"post-execution observation failed: {type(exc).__name__}: {exc}",
                    violation,
                )
                raise violation from exc
            if post != planned:
                violation = DatasetLifecycleViolationError("strict MSv2 reader changed its dataset")
                transition_preserving(
                    DatasetLifecyclePhase.FAILED,
                    "read-only postcondition failed: the MSv2 access plan or observation changed during execution",
                    violation,
                    post_observations=post.observations,
                )
                raise violation
            lifecycle.transition(
                DatasetLifecyclePhase.VALIDATED,
                "read-only postcondition matches the claimed baseline",
                post_observations=post.observations,
            )
            publish_validated_result()
            if result.success:
                lifecycle.transition(DatasetLifecyclePhase.COMMITTED, "native contained MSv2 read committed")
            else:
                lifecycle.transition(
                    DatasetLifecyclePhase.FAILED,
                    f"native reader returned non-zero status {result.returncode}",
                )
            return result
        except BaseException as exc:
            primary_error = exc
            if lifecycle.record.phase not in {
                DatasetLifecyclePhase.COMMITTED,
                DatasetLifecyclePhase.REFUSED,
                DatasetLifecyclePhase.FAILED,
            }:
                phase = DatasetLifecyclePhase.FAILED if executing else DatasetLifecyclePhase.REFUSED
                transition_preserving(phase, f"{type(exc).__name__}: {exc}", exc)
            if isinstance(exc, WorkspaceOwnershipError):
                wrapped = DatasetLifecycleUnavailableError(f"contained MSv2 {noun} claim refused: {exc}")
                primary_error = wrapped
                raise wrapped from exc
            raise
        finally:
            if lease is not None:
                try:
                    lease.release()
                except BaseException as cleanup_exc:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        f"workspace lease cleanup also failed: {type(cleanup_exc).__name__}: {cleanup_exc}; the exact claim remains for inspection/reconciliation"
                    )
                    logger.exception("contained MSv2 %s failed and its workspace lease cleanup also failed", noun)
    if dataset_declarations and _dataset_lifecycle is None:
        _refuse_unenforced_datasets(scope)
    if _cache_path is None and not _workspace_claimed:
        from shinobi.ownership import acquire_workspace, ownership_workspace, scope_declares_writes, scope_path_accesses, scope_requires_ownership

        if scope_requires_ownership(scope):
            # Ownership must see exactly the values execution will see,
            # including defaults and default factories. Validate once and
            # thread that instance into ExecContext: validating separately
            # could claim one factory-produced path and execute with another.
            validated_inputs = _validated_inputs if _validated_inputs is not None else _validate_inputs(scope, kwargs)
            launch_workspace = Path.cwd()
            accesses, leaf_inputs = scope_path_accesses(scope, validated_inputs, workspace=launch_workspace)
            writes = [path for path, writes in accesses if writes]
            if writes:
                workspace = ownership_workspace(launch_workspace, writes)
            elif scope_declares_writes(scope):
                workspace = launch_workspace.resolve()
                accesses.append((workspace, True))
            else:
                workspace = launch_workspace.resolve()
            lease = acquire_workspace(
                workspace,
                run_id,
                kind="local",
                accesses=accesses,
            )
            try:
                return _dispatch(
                    scope,
                    func,
                    backend=backend,
                    cache=cache,
                    cache_dir=cache_dir,
                    stream=stream,
                    provenance=provenance,
                    sandbox=sandbox,
                    _recipe_backend=_recipe_backend,
                    _recipe_cache=_recipe_cache,
                    _recipe_cache_dir=_recipe_cache_dir,
                    _recipe_stream=_recipe_stream,
                    _recipe_provenance=_recipe_provenance,
                    _recipe_sandbox=_recipe_sandbox,
                    _cache_path=_cache_path,
                    _config=config,
                    _provenance_target=_provenance_target,
                    _input_keys=_input_keys,
                    _wired_fields=_wired_fields,
                    _budget=_budget,
                    _run_id=run_id,
                    _slice_index=_slice_index,
                    _execution_identity=_execution_identity,
                    _snapshot_success_record=_snapshot_success_record,
                    _snapshot_success_step_path=_snapshot_success_step_path,
                    _result_commit=_result_commit,
                    _publication_gate=_publication_gate,
                    _workspace_claimed=True,
                    _dataset_lifecycle=_dataset_lifecycle,
                    _validated_inputs=validated_inputs,
                    _leaf_inputs=leaf_inputs,
                    **kwargs,
                )
            finally:
                lease.release()
    if _cache_path is None:
        # Top-level entry: the start of a run, and the memoized boundary-path
        # hashes and execution-environment identities must not outlive one.
        # Anything could have happened to the workspace, a mutable image tag,
        # or a venv between two runs sharing a process, and none of it went
        # through the post-execution invalidation below.
        invalidate_path_hashes()
        from shinobi.backends.container import clear_image_pin_cache
        from shinobi.backends.venv import venv_digest

        clear_image_pin_cache()
        venv_digest.cache_clear()
        # Whether boundary fingerprints carry a content sample is a property
        # of the workspace, not of a step, so it is set once per run rather
        # than passed down (see `cache.set_content_sample`).
        set_content_sample(config.cache.content_sample)
    cache_enabled = cache if cache is not None else scope.cache if scope.cache is not None else _recipe_cache if _recipe_cache is not None else config.cache.enabled
    # A strict MSv2 writer caches automatically: its dataset states are
    # named by cache key. The workflow has already refused any *explicit*
    # disable on its path (`_strict_mutation_cache_issues`), so the only
    # value overridden here is an inherited or configured default.
    strict_cache_auto = (
        _dataset_lifecycle is not None
        and getattr(_dataset_lifecycle, "mutation", False)
        and not isinstance(scope, Recipe)
        and cache is None
        and scope.cache is None
        and not cache_enabled
        and bool(_leaf_dataset_writes(scope))
    )
    if strict_cache_auto:
        cache_enabled = True
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
    # Mutation-chain snapshots (shinobi.snapshots) ride on caching being on
    # *somewhere* in the effective chain, not on this scope being cacheable.
    # An uncached mutating step inside an otherwise-cached recipe still has
    # to dirty the journal, or a cached consumer downstream would restore
    # over work it cannot see -- see the never-worse rules.
    # Two different questions, and conflating them silently disabled crash
    # recovery for every real pipeline. Whether Tier 1 is *active for this
    # run* has nothing to do with the shape of the scope being dispatched;
    # whether *this scope* gets a snapshot guard does, because a Recipe is
    # never itself cached and mutates nothing of its own -- its leaves do.
    # A top-level target is almost always a Recipe, so gating reconciliation
    # on the leaf-level answer meant it never ran outside the tests that
    # called it directly.
    snapshots_active = config.cache.snapshots.mode != "off" and (cache_enabled or bool(_recipe_cache) or config.cache.enabled)
    snapshots_enabled = snapshots_active and not isinstance(scope, Recipe)
    if _cache_path is None and snapshots_active:
        # Crash recovery, before any step of this run looks at the disk --
        # but only once we can show no other shinobi is live on this cache
        # directory. Reconciliation reads "marker set, no manifest entry" as
        # a corpse and rolls the workspace back; against a run that is merely
        # still going, that deletes its work mid-write. When we cannot
        # establish we are alone we skip, which is safe: the marker stays
        # set, so the next restore still forces a rollback (branch 1), and
        # only the tidying waits.
        reconcile_paths = None
        if _dataset_lifecycle is not None:
            claim = _dataset_lifecycle.record.claim
            reconcile_paths = {Path(access.path) for access in claim.accesses if access.writes} if claim is not None else set()
        presence = announce_run(cache_dir_value, run_id)
        alone = presence.alone()
        if alone and (reconcile_paths is None or reconcile_paths):
            for note in reconcile(cache_dir_value, get_cache_manifest(cache_dir_value), paths=reconcile_paths):
                logger.warning("cache: %s", note)
        elif not alone:
            logger.warning(
                "cache: not reconciling %s -- another shinobi process appears to be using it (or this filesystem does not support locking). Interrupted steps still recover on their own; run 'ninja cache check' when the other run has finished.",
                cache_dir_value,
            )

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
        clear_outputs=config.execution.clear_stale_outputs,
        input_keys=_input_keys,
        budget=_budget,
        run_id=run_id,
        validated_inputs=_validated_inputs,
        leaf_inputs=_leaf_inputs,
        dataset_lifecycle=_dataset_lifecycle,
        publication_gate=_publication_gate,
    )

    manifest = None
    cache_key = None

    def publish(action: Callable[[], None]) -> None:
        if _publication_gate is None:
            action()
        else:
            _publication_gate(action)

    # A leaf carrying a strict dataset contract inside a mutation lifecycle:
    # it records its own cache decision, observations and outcomes.
    strict_leaf = None
    if _dataset_lifecycle is not None and getattr(_dataset_lifecycle, "mutation", False) and not isinstance(scope, Recipe):
        from shinobi.dataset_access import scope_has_dataset_contract

        if scope_has_dataset_contract(scope):
            from shinobi.dataset_lifecycle import StrictLeaf

            strict_leaf = StrictLeaf(_dataset_lifecycle, cache_path, scope, ctx.prepare_inputs())

    if cacheable:
        execution_identity = _execution_identity or _local_execution_identity(scope, ctx, pinned=provenance_enabled)
        if execution_identity is None:
            logger.warning("step %s: cache disabled -- selected execution environment could not be fingerprinted", cache_path)
            cacheable = False
    if strict_leaf is not None and strict_leaf.writes and not (cacheable and snapshots_enabled):
        reason = "exact MSv2 mutation recovery names states by cache key and needs Tier 1 snapshots, but " + (
            "this step is not cacheable (disabled, or its execution environment could not be fingerprinted)" if not cacheable else "snapshots are off"
        )
        strict_leaf.fail(None, reason, refused=True)
        raise DatasetLifecycleUnavailableError(f"strict MSv2 mutation of {cache_path!r} refused before launch: {reason}")
    if cacheable:
        manifest = get_cache_manifest(cache_dir_value)
        prepared_for_key = ctx.prepare_inputs()
        cache_key = compute_cache_key(scope, func, prepared_for_key, _input_keys, execution_identity)
        hit = manifest.check(cache_path, cache_key, scope, prepared_for_key)
        if hit is not None and strict_leaf is not None and not strict_leaf.decide_reuse(cache_key, _input_keys, get_journal(cache_dir_value)):
            # The key and outputs match, but the dataset has moved on to a
            # state that does not contain this step's work (see
            # `snapshots.strict_reuse_issue`). Re-run it: the guard restores
            # its exact predecessor first.
            logger.warning("step %s: cache hit rejected for strict MSv2 reuse -- %s", cache_path, strict_leaf.cache.reason)
            hit = None
        if hit is not None:
            # A hit stands in for the run that first produced this key, so it
            # must advertise the same provenance -- otherwise dependents would
            # see the producer's key vanish and their own keys would flip
            # between runs purely on whether the producer hit or ran.
            hit.cache_key = cache_key
            if hit.venv_digest is None and execution_identity.venv_digest is not None:
                # The key embeds this fingerprint, so a match proves it even
                # when the recording run did not pin (and so stored none).
                # Venv steps stay unpinned in the manifest either way.
                hit.venv_digest = execution_identity.venv_digest
            logger.info("step %s: cache hit -- skipping run", cache_path)

            def publish_hit() -> None:
                if _result_commit is not None:
                    _result_commit(hit, lambda: None)
                if _cache_path is None and provenance_enabled:
                    _emit_run_manifest(hit, ctx, config, backend, target=_provenance_target)

            publish(publish_hit)
            return hit

    logger.info(
        "step %s: starting%s",
        cache_path,
        " (sandboxed)" if sandbox_enabled else "",
    )
    # Tier 1's pre-run hooks: roll the mutated paths back to the states this
    # step's DAG position calls for, then mark them in flight, then snapshot
    # what is being consumed. All of it after the cache key is computed --
    # which is safe because a mutated path contributes only its path string
    # to the key, so no restore can move it (see `snapshots.before_run`).
    if strict_leaf is not None:
        strict_leaf.decide_run(cache_key, _input_keys, cacheable=cacheable, automatic=strict_cache_auto)
    if strict_leaf is not None and strict_leaf.writes:
        assert cache_key is not None
        try:
            guard = _strict_snapshot_guard(
                scope,
                ctx,
                strict_leaf,
                cache_dir_value,
                cache_path,
                cache_key,
                run_id,
                _input_keys,
                _wired_fields,
                _boundary_fields,
                _slice_index,
                config,
            )
            guard.before_run()
        except BaseException as exc:
            strict_leaf.fail(None, f"refused before launch: {exc}", refused=True)
            raise
    else:
        guard = (
            _snapshot_guard(
                scope,
                ctx,
                cache_dir_value,
                cache_path,
                cache_key,
                run_id,
                _input_keys,
                _wired_fields,
                _slice_index,
                config,
                _snapshot_success_record,
                _snapshot_success_step_path,
            )
            if snapshots_enabled
            else None
        )
        if guard is not None and _publication_gate is not None:
            raise DatasetLifecycleUnavailableError("contained MSv2 read refused: a lifecycle leaf requires mutation recovery, which only the mutation lifecycle provides")
        if guard is not None:
            guard.before_run()
    if strict_leaf is not None:
        try:
            strict_leaf.observe_before(guard)
        except BaseException as exc:
            if guard is not None:
                guard.after_failure()
            strict_leaf.fail(guard, f"predecessor observation failed: {type(exc).__name__}: {exc}", refused=True)
            raise
    try:
        if func is None:
            result = ctx.run()
        else:
            result = func(ctx)
            if result is None:
                result = ctx.run()
            elif not isinstance(result, StepResult):
                raise TypeError(f"step function {getattr(func, '__name__', func)!r} must return StepResult or None, got {type(result).__name__}")
    except BaseException as step_exc:
        # BaseException, not Exception: an interrupt is now an orderly unwind
        # (the child is stopped first), so the step's workspace needs the same
        # rollback any other failure gets. Catching only Exception left an
        # interrupted step relying on the marker-plus-reconcile path alone.
        logger.exception("step %s: raised", cache_path)
        if guard is not None:
            # The workspace goes back to exactly what it was before this step
            # touched it. The marker stays set on purpose -- the next run then
            # forces a rollback before re-executing.
            guard.after_failure()
        if strict_leaf is not None:
            strict_leaf.fail(guard, f"step raised {type(step_exc).__name__}: {step_exc}")
        raise
    finally:
        # This step may have written to the workspace -- including to an
        # unwired boundary path a *later* step reads, whose memoized content
        # hash would otherwise be pre-mutation and produce a false cache hit.
        # In the `finally` because a step that raised part-way through has
        # still had the chance to write. See `cache._hash_path`.
        invalidate_path_hashes()

    if strict_leaf is not None and result.success:
        # Exit status zero is not success for a strict step: its declared
        # postconditions decide, before anything names or publishes the
        # successor. A violating writer is rolled back to its predecessor.
        try:
            strict_leaf.validate(guard)
        except BaseException as exc:
            if guard is not None:
                guard.after_failure()
            strict_leaf.fail(guard, f"{type(exc).__name__}: {exc}")
            raise

    # A recipe's stdout/stderr aggregate its sub-steps', and each sub-step
    # already logged its own via its recursive _dispatch -- re-logging the
    # aggregate would duplicate every line.
    if result.kind != "recipe":
        # `display_label`, not `cache_path`: this prefix is repeated once per
        # line of the step's output, and the root scope name in it is constant
        # for the whole run. The lifecycle records below keep the full path --
        # they are one line per step, and there the identity is the point.
        shown = display_label(cache_path)
        for line in result.stdout.splitlines():
            logger.info("[%s] %s", shown, line)
        for line in result.stderr.splitlines():
            logger.info("[%s] %s", shown, line)
    if result.success:
        logger.info("step %s: finished (returncode 0)", cache_path)
    else:
        logger.error("step %s: failed (returncode %s)", cache_path, explain_returncode(result.returncode))

    if cacheable:
        result.cache_key = cache_key

    def publish_result() -> None:
        if result.success:
            # The five-stage commit lives in the guard so its ordering
            # constraints are enforced in one place -- above all that the tip
            # snapshot (S1) precedes the explicit success oracle (S3), or a
            # committed result could name a state with nothing snapshotted.
            def _record() -> None:
                if strict_leaf is not None:
                    # A strict leaf's committed entry is its marker's success
                    # oracle, so it precedes the reusable cache index.
                    strict_leaf.commit()
                if cacheable:
                    manifest.record(cache_path, cache_key, result, run_id=run_id)

            def _commit() -> None:
                if _result_commit is None:
                    _record()
                else:
                    _result_commit(result, _record)

            if guard is not None:
                try:
                    guard.after_success(_commit)
                except BaseException as exc:
                    # Before S3 the guard has rolled back; after it, the
                    # committed oracle stands and only tidying failed.
                    recorded = strict_leaf.lifecycle.leaf(cache_path) if strict_leaf is not None else None
                    if strict_leaf is not None and (recorded is None or recorded.outcome != "committed"):
                        strict_leaf.fail(guard, f"commit failed: {type(exc).__name__}: {exc}")
                    raise
            else:
                _commit()
        else:
            if guard is not None:
                guard.after_failure()
            if strict_leaf is not None:
                strict_leaf.fail(guard, f"step returned non-zero status {result.returncode}")
            if _result_commit is not None:
                _result_commit(result, lambda: None)
        if _cache_path is None and provenance_enabled:
            _emit_run_manifest(result, ctx, config, backend, target=_provenance_target)

    publish(publish_result)
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
    from shinobi.steps.schema import StaticOutputResolutionError, static_output_values

    reserved = {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    # Higher-priority runtime sources must also suppress evaluation of a
    # lower-priority implicit template.  A wrangler (or reserved backend
    # result without a same-named input) is sufficient even when that unused
    # template cannot be formatted from the prepared inputs.
    skip_static = set(wrangled) | {name for name in reserved if name not in prepared}
    try:
        static = static_output_values(cab, prepared, skip_outputs=skip_static, strict_templates=True)
    except StaticOutputResolutionError as exc:
        raise ParameterError(str(exc)) from exc
    values: dict[str, Any] = {}
    for name in cab.outputs_model.model_fields:
        if name in wrangled:
            values[name] = wrangled[name]
        elif name in prepared:
            values[name] = static[name]
        elif name in reserved:
            values[name] = reserved[name]
        elif name in static:
            values[name] = static[name]
    try:
        return cab.outputs_model(**values)
    except ValidationError as exc:
        raise ParameterError(f"{cab.name}: output validation failed:\n{exc}") from exc


def _report_elision(run: BackendRun, label: str, *, wrangled: bool) -> None:
    """Say so when a step's captured output was capped.

    Two different events, deliberately at two different levels. Ordinary
    elision is a *note*: the run is fine, the dropped lines are progress
    chatter, and the text says where they were -- logging it at WARNING
    would train operators to ignore the word on a chatty pipeline. It does
    not claim the run log escaped: that log is written from this same
    captured text (see `_dispatch`), so it carries the elision marker too,
    and only the live echo saw every line. A
    dropped *wrangler-matching* line is a warning, because the buffer
    retains matches wherever they occur, so reaching this means the
    retained-match ceiling was hit and an output value may be missing.
    """
    dropped = run.stdout_dropped + run.stderr_dropped
    if not dropped:
        return
    logger.info(
        "step %s: %s line%s of captured output elided (log.capture_head_lines/capture_tail_lines); the live stream was not capped, but the run log is written from this same captured text and carries the elision",
        label,
        dropped,
        "s" if dropped != 1 else "",
    )
    if run.wrangler_lines_dropped and wrangled:
        warnings.warn(
            f"step '{label}': output was capped past the point where its wranglers could all be applied, so a declared output may be unset or stale; raise log.capture_head_lines/capture_tail_lines to re-run with the full capture",
            stacklevel=2,
        )


def _run_cab(
    cab: Cab,
    prepared: dict[str, Any],
    backend_name: str,
    *,
    label: str = "",
    stream: bool = True,
    pin: bool = False,
    sandbox_root: str | None = None,
    clear_outputs: bool = True,
) -> StepResult:
    # Sandboxed run (shinobi.sandbox): the tool's cwd is a private scratch
    # dir; path-typed inputs are anchored back at the workspace so the tool
    # still reads/mutates the caller's real files. argv and the backend's
    # bind mounts are built from the anchored values, but output filling
    # below keeps the caller's original (workspace-relative) values, so
    # declared output paths stay stable whether or not a sandbox was used.
    sandbox_dir = None
    run_inputs = prepared
    workspace = Path.cwd()
    if sandbox_root is not None:
        sandbox_dir = create_sandbox(sandbox_root, label or cab.name)
        precreated = prepare_output_parents(cab, prepared, sandbox_dir)
        run_inputs = absolutize_path_inputs(cab, prepared, workspace)
    # Against `run_inputs`, so the destinations are the ones the tool is
    # really given, and before the run, because a tool that refuses to
    # overwrite has already failed by the time anything harvests.
    if clear_outputs:
        clear_stale_outputs(cab, run_inputs, workspace, sandboxed=sandbox_dir is not None)
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
    _report_elision(run, label or cab.name, wrangled=bool(cab.wranglers))
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
                f"step '{label or cab.name}' failed (returncode {explain_returncode(run.returncode)}); its sandbox is kept for post-mortem at {sandbox_dir}",
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
        sandbox_path=str(sandbox_dir) if sandbox_dir is not None and sandbox_dir.exists() else None,
        backend=backend_name,
        image=cab.image,
        image_digest=run.image_digest,
        containerized=run.containerized,
        venv=run.venv,
        venv_digest=run.venv_digest,
        sandboxed=sandbox_dir is not None,
        resources=cab.resources,
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
    by slice index. Provenance is gathered too, per output field -- see
    `output_keys` below.
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

    # Per-output provenance for whoever consumes the gathered result, built
    # the same way as the gathered outputs themselves: one field at a time,
    # across the slices.
    #
    # Doing it per field rather than leaving `cache_key` to answer for the
    # whole step is what keeps a scattered *Recipe* keyed at all. A leaf
    # slice resolves every field to its own `cache_key`, so for a scattered
    # leaf this reproduces exactly what `cache_key` already said (same list,
    # same hash -- no existing key moves). A recipe slice does not have a
    # `cache_key`: it carries `output_keys`, one per declared output. So
    # `combine_keys([s.cache_key ...])` over recipe slices combines a list of
    # `None`s and yields `None`, and the gathered result hands its consumers
    # no provenance at all -- their `__upstream__` loses the field, and a
    # path input that is *also* mutated in place then drops out of the key
    # entirely (`compute_cache_key`). The consumer of a scattered recipe
    # therefore cache-hit forever, however the producer changed, and an
    # in-place mutation it was supposed to apply was silently never applied.
    # **Deliberately `producer_field=None`.** A `ProvenanceKey`'s field names
    # a state for Tier 1 (`state_name(key, field)`), and this key names no
    # state: it is a synthetic hash standing for N slices, handed *identically*
    # to every slice of a downstream scattered consumer (they share one
    # `sub_input_keys`). Stamping a field on it would make every slice compute
    # the same `state_name`, and `Journal.snapshot_dir` is a flat namespace
    # whose `_take` early-returns on an existing name -- so one slice's tree
    # would be snapshotted under a name every *other* slice's chain then
    # records as consumed, and `_restore` checks only that the name exists,
    # never that it belongs to this chain. Two targets, one state: the second
    # gets the first's data put back over it.
    #
    # With no field, `_required_state`'s `producer_field is not None` guard
    # falls through to the chain's own `consumed`/`head`, which is per path and
    # therefore per target. `eligible_fields`/`_single_key` still see a key, so
    # nothing is spuriously excluded, and the key still hashes into
    # `__upstream__` as its string -- the cache fix is untouched.
    #
    # This closes the same hole for a scattered *leaf*, which had it already:
    # `_resolve_input_keys` wrapped its plain `cache_key` with the consuming
    # field's name.
    output_keys: dict[str, Any] = {}
    for field in scope.outputs_model.model_fields:
        combined = combine_keys([s.provenance_key(field) for s in slices])
        if combined is not None:
            output_keys[field] = ProvenanceKey(combined, None)

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
        venv=slices[0].venv if slices else None,
        venv_digest=slices[0].venv_digest if slices else None,
        sandboxed=any(s.sandboxed for s in slices),
        resources=scope.resources,
        # Every slice is keyed independently, but downstream wires the
        # *gathered* result -- so its provenance is all the slices' keys
        # together, and a change in any one of them invalidates dependents.
        cache_key=combine_keys([s.cache_key for s in slices]),
        # `or None` so an aggregate with nothing to say falls back to
        # `cache_key` exactly as it did before there was a per-field answer
        # -- `provenance_key` stops consulting `cache_key` the moment
        # `output_keys` is set, empty or not.
        output_keys=output_keys or None,
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
    budget: Budget | None = None,
    run_id: str = "",
    leaf_inputs: dict[int, tuple[BaseModel, bool]] | None = None,
    validated_inputs: BaseModel | None = None,
    dataset_lifecycle: Any | None = None,
    publication_gate: Callable[[Callable[[], None]], None] | None = None,
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

    Admission has a second gate besides `max_workers`: if any step declares
    a resource footprint (`Scope.resources`), work is also admitted against
    a `Budget` (see `shinobi.resources`). The budget is created once at the
    outermost recipe that needs one and shared by every nested recipe's
    scheduler -- a per-invocation budget would be useless for the pipelines
    that motivated this, which nest each parallel branch as its own Recipe.
    A recipe where nothing declares anything never builds one and schedules
    exactly as it always did.
    """
    config = config or AppConfig.load()  # resolve once; workers never call load()
    # Access planning is a pre-dispatch operation over the same declared
    # graph. Strict MSv2 execution is still refused by `_dispatch`; this
    # shared path is ready for that lifecycle boundary without teaching the
    # scheduler a second conflict model.
    from shinobi.dataset_access import plan_recipe_accesses

    graph = (
        plan_recipe_accesses(recipe, prepared, workspace=Path.cwd(), validated_steps=leaf_inputs, validated_inputs=validated_inputs).graph
        if leaf_inputs is not None
        else build_graph(recipe)
    )
    max_workers = recipe.max_workers or config.execution.max_workers

    # Built lazily, and only by the outermost recipe that needs one: nothing
    # is detected, logged or locked for a recipe that declares no footprints.
    if budget is None and _any_resources(recipe):
        totals, source = config.execution.resources.resolve()
        budget = Budget(totals)
        logger.info("recipe %s: admitting steps against a budget of %s (%s)", recipe.name, totals.describe(), source)

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

    # Admission units awaiting a worker slot and (if declared) budget. A unit
    # is one step, or one slice of a scattered step -- slices are admitted
    # individually rather than all submitted at once, so a wide scatter can no
    # longer put more work in flight than the scheduler accounted for.
    #
    # Declaration order across *steps* is carried by this heap itself: the
    # head is always the lowest-index unit, whichever steps happen to be
    # queued. That matters because backfill (`_backfill_behind`) expands a
    # second step's units while the head is parked, so the ordinary refill
    # discipline below -- only expand the next step once this queue has
    # drained -- is an expansion *policy*, not the thing keeping order true.
    # Being a heap rather than a plain queue is what makes that safe.
    # `None` (not a slice) is stored as -1, so heap comparison never has to
    # order None against an int.
    pending: list[tuple[int, int]] = []
    unit_payload: dict[tuple[int, int | None], tuple[dict[str, Any], dict[str, Any], BaseModel | None]] = {}
    futures: dict[Future, tuple[int, int | None]] = {}
    # What each in-flight future reserved, and whether that reservation came
    # from `Budget.try_backfill`, so it can be released on every exit path --
    # including ones that never reach the reap loop. The flag has to travel
    # with the reservation: the budget's cumulative backfill total is only
    # honest if every grant is matched by a release that says so.
    held: dict[Future, tuple[Resources, bool]] = {}

    def _return_reservations() -> None:
        """Give back every reservation this recipe still holds.

        Not just belt-and-braces over the per-future release: `_expand` runs
        on *this* thread and can raise (unresolvable wiring, a mismatched
        scatter), unwinding straight past the reap loop. The pool's own
        `__exit__` then joins the in-flight work, but nobody reaps it -- so
        without this its reservations would be leaked permanently, and
        because the budget is shared that would throttle every sibling
        recipe for the rest of the run. Also drops this scheduler's place in
        the budget queue: a ticket left behind by a scheduler that has
        stopped waiting would block every later arrival forever.
        """
        if budget is None:
            return
        for demand, backfilled in held.values():
            budget.release(demand, backfilled=backfilled)
        held.clear()
        budget.abandon()

    def _teardown_on_interrupt(exc_type, exc, tb) -> bool:
        """On Ctrl-C, stop the work before anyone waits for it to finish.

        An interrupt is delivered to the *main* thread only. Every worker is
        parked inside its own `run_streaming`, blocked on a child that knows
        nothing about it -- so the pool's `__exit__`, which joins those
        threads, would wait on containers nobody has told to stop. That is
        not a leak that shows up later: it hangs ninja at the exact moment
        the user asked it to stop, with the whole run still burning CPU.

        Registered *after* the pool so it runs *before* `pool.__exit__` in
        the unwind (callbacks are LIFO), which is the whole point -- killing
        the children is what lets the join it precedes actually complete.
        """
        if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            terminate_all(reason=f"recipe '{recipe.name}' interrupted")
        return False  # never suppress; this is a hook, not a handler

    # `ExitStack`, not a plain `with`: callbacks run in reverse order of
    # registration, so `_return_reservations` (registered first) runs *after*
    # the pool's `__exit__` has joined every in-flight future -- reservations
    # come back once the work they were taken for has genuinely stopped.
    with ExitStack() as stack:
        stack.callback(_return_reservations)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=max_workers))
        stack.push(_teardown_on_interrupt)

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

        def _expand(i: int) -> None:
            """Resolve step `i`'s inputs and queue its admission unit(s).

            A step that does no work -- a converged loop iteration, a
            zero-length scatter -- completes right here and queues nothing,
            so it never occupies a worker or any budget.
            """
            ref = recipe.steps[i]
            sub_kwargs = _resolve_wiring(ref, prepared, results)
            # Reuse the leaf's default/default-factory evaluation that
            # ownership discovery already performed, so a non-deterministic
            # factory cannot claim one path and execute with another. Fields
            # that wiring just resolved are never touched, so upstream
            # outputs always win over a claim-time default.
            validated_leaf = None
            reusable_leaf = False
            if leaf_inputs is not None and ref.scatter is None:
                snapshot = leaf_inputs.get(id(ref))
                if snapshot is not None:
                    validated_leaf, reusable_leaf = snapshot
                    for name in type(validated_leaf).model_fields:
                        if name not in sub_kwargs:
                            sub_kwargs[name] = getattr(validated_leaf, name)
            sub_input_keys = resolve_input_keys(ref, input_keys or {}, results)
            # An unrolled loop iteration whose predecessor already converged
            # does no work: it hands the same body step's previous outputs
            # on, completing immediately without occupying a worker. The
            # sentinel producer is guaranteed complete -- add_loop gives
            # every iteration a dependency edge to it.
            if should_skip(ref, results):
                prev = results[ref.loop.prev_step]
                logger.info("step %s%s: skipped (loop '%s' converged)", f"{cache_path}." if cache_path else "", ref.name, ref.loop.loop)
                effective = validated_leaf if reusable_leaf else ref.step.inputs_model(**_prepare_inputs(ref.step, sub_kwargs))
                _step_completed(i, passthrough_result(ref, prev, effective))
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
                    unit_payload[(i, slice_idx)] = (slice_kwargs, sub_input_keys, None)
                    heapq.heappush(pending, (i, slice_idx))
                return
            unit_payload[(i, None)] = (sub_kwargs, sub_input_keys, validated_leaf if reusable_leaf else None)
            heapq.heappush(pending, (i, _NOT_A_SLICE))

        def _demand_of(i: int) -> Resources:
            """What step `i` reserves while it runs.

            A nested `Recipe` reserves nothing: it is not a unit of
            execution, its leaves are, and they schedule against this same
            shared budget. Reserving for both would double-count -- and
            deadlock, since the parent would hold that reservation while the
            nested scheduler waited for budget only the parent could release.
            (`build_graph` rejects the declaration outright; this is the
            invariant the scheduler itself relies on.)
            """
            scope = recipe.steps[i].step
            if scope.resources is None or isinstance(scope, Recipe):
                return _NO_RESOURCES
            return scope.resources

        def _submit_unit(i: int, slice_idx: int | None, demand: Resources, backfilled: bool = False) -> None:
            """Submit one admission unit that has already been granted."""
            ref = recipe.steps[i]
            unit_kwargs, sub_input_keys, validated_inputs = unit_payload.pop((i, slice_idx))
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
                _cache_path=f"{cache_path}.{ref.name}" if slice_idx is None else f"{cache_path}.{ref.name}[{slice_idx}]",
                _config=config,
                _budget=budget,
                _input_keys=sub_input_keys,
                # The *declared* wiring, not just the keyed subset. Tier 1
                # has to tell "unwired boundary path" from "wired to a
                # producer that had no cache key" -- `_input_keys` omits
                # both, and they need opposite treatment (the first is
                # named from the journal, the second cannot be named at
                # all). See `snapshots.eligible_fields`.
                _wired_fields=set(ref.wiring),
                # Fields wired straight from a top-level recipe's own inputs.
                # They carry no producer key because they have no producer:
                # they are the workflow boundary, not a keyless upstream step.
                # Only the strict dataset guard distinguishes the two (see
                # `_strict_snapshot_guard`); ordinary Tier 1 is unchanged.
                _boundary_fields=frozenset(field for field, source in ref.wiring.items() if isinstance(source, InputRef)) if input_keys is None else frozenset(),
                _run_id=run_id,
                _slice_index=slice_idx,
                _leaf_inputs=leaf_inputs,
                _validated_inputs=validated_inputs,
                _dataset_lifecycle=dataset_lifecycle,
                _publication_gate=publication_gate,
                **unit_kwargs,
            )
            futures[fut] = (i, slice_idx)
            held[fut] = (demand, backfilled)

        def _backfill_behind(head_index: int, head_demand: Resources) -> bool:
            """Admit one unit past a head the budget just refused, or expand
            a `ready` step so the next pass can.

            Blocking on the head is what stops a large step being starved,
            but it also idles a machine that has room for work the head will
            not use. `Budget.try_backfill` admits only units the head can
            still start alongside once the work reserved *now* drains, so
            the head waits for exactly what it was already waiting for.

            Returns:
                Whether anything was done -- if so the caller should retry
                admission rather than park.
            """
            # `sorted`, not `enumerate`: a heap orders only its root, so
            # walking the backing list would pick an arbitrary fitting unit
            # rather than the lowest-index one. Backfill relaxes *whether*
            # queued work waits for the head, never the order it is chosen
            # in -- everything else here drains by declaration index and this
            # must too, or which step overtakes a parked head depends on heap
            # layout. `pending` is small (one step's units, plus whatever
            # earlier backfills expanded), so sorting it is cheap.
            for entry in sorted(pending)[1:]:
                i, slot = entry
                if budget.try_backfill(_demand_of(i), head_demand, recipe.steps[i].name):
                    pending.remove(entry)  # entries are unique (step, slice) pairs
                    heapq.heapify(pending)
                    _submit_unit(i, None if slot == _NOT_A_SLICE else slot, _demand_of(i), backfilled=True)
                    return True

            # Nothing queued fits; look a little further into `ready`. Only
            # the chosen step is expanded, so wiring is not resolved for
            # steps that may never run. `ready` is a heap: pop candidates and
            # push back the ones not taken, because slicing it would not give
            # the lowest indices and mutating it in place would corrupt every
            # later push/pop -- which is what the ordering guarantee rests on.
            examined: list[int] = []
            chosen: int | None = None
            while ready and len(examined) < _BACKFILL_LOOKAHEAD:
                candidate = heapq.heappop(ready)
                # A lower-index step is not a backfill at all: declaration
                # order says it outranks the current head, so expand it
                # unconditionally and let it become the next head. Filtering
                # it through the backfill rule would strand it here behind a
                # step it should precede. (`build_graph` puts no index
                # ordering on `after`/`OutputRef`, so this really happens.)
                if candidate < head_index or budget.can_backfill(_demand_of(candidate), head_demand):
                    chosen = candidate
                    break
                examined.append(candidate)
            for candidate in examined:
                heapq.heappush(ready, candidate)
            if chosen is None:
                return False
            _expand(chosen)
            return True

        def submit_ready() -> tuple[bool, int]:
            """Admit as much queued work as capacity and budget allow.

            Units are drained lowest-declaration-index first, and admission
            *blocks* on the head unit rather than skipping past it: a step
            that does not fit holds the queue, so a large step can never be
            starved by a stream of smaller ones behind it. `_backfill_behind`
            then recovers the capacity that blocking would otherwise idle,
            without moving the head's start time.

            Returns:
                `(parked, generation)`. `parked` is True only when the head
                unit was refused by the *budget* (not by worker capacity),
                in which case the caller may wait on `generation` -- passing
                that value back to `Budget.wait_for_change` is what makes
                refuse-then-wait race-free.
            """
            while not stop:
                # Refill from `ready`. A `while`, not a single shot: a step
                # that does no work completes inside `_expand` and pushes its
                # dependents onto `ready`, which one refill pass would leave
                # stranded -- and an empty `pending` with a non-empty `ready`
                # is exactly the state that would idle forever below.
                while not pending and ready:
                    _expand(heapq.heappop(ready))
                if not pending or len(futures) >= max_workers:
                    return False, 0
                i, slot = pending[0]
                slice_idx = None if slot == _NOT_A_SLICE else slot
                demand = _demand_of(i)
                if budget is None:
                    granted, generation = True, 0
                else:
                    granted, generation = budget.try_acquire(demand, recipe.steps[i].name)
                if not granted:
                    # Backfill only where concurrency is already allowed. At
                    # `max_workers=1` a sibling recipe holding the shared
                    # budget can leave this scheduler with nothing in flight
                    # and a refused head -- and admitting a later unit there
                    # would break the documented guarantee that
                    # `max_workers=1` reproduces exact declaration order.
                    if max_workers > 1 and _backfill_behind(i, demand):
                        continue
                    return True, generation
                heapq.heappop(pending)
                _submit_unit(i, slice_idx, demand)
            return False, 0

        parked, generation = submit_ready()
        # `ready`/`pending` are deliberately gated on `not stop`: once a
        # failure or worker exception has stopped submission, un-submitted
        # work is abandoned and only the in-flight steps are drained. Without
        # that gate the queue would never empty and this would hang.
        while futures or (not stop and (pending or ready)):
            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    i, slice_idx = futures.pop(fut)
                    demand, was_backfill = held.pop(fut, (_NO_RESOURCES, False))
                    try:
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
                    finally:
                        # Every reap path returns the reservation, including
                        # the worker-exception one that `continue`s out.
                        if budget is not None:
                            budget.release(demand, backfilled=was_backfill)
            elif parked:
                # Nothing of ours is running and the head unit doesn't fit:
                # only another recipe sharing this budget can unblock us.
                budget.wait_for_change(generation)
            parked, generation = submit_ready()

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
        raise CabRunError(f"step '{ref_name}' in recipe '{recipe.name}' failed (returncode {explain_returncode(failed.returncode)})")

    ordered = [ref.name for ref in recipe.steps if ref.name in results]
    outputs = {field: getattr(results[out_ref.step].outputs, out_ref.field) for field, out_ref in recipe.output_wiring.items() if out_ref.step in results}
    # Per-output provenance for whoever consumes this recipe's outputs: each
    # declared output is keyed by the sub-step that actually produced it, not
    # by the recipe as a whole -- see StepResult.provenance_key. `out_ref.field`
    # names the producing sub-step's own output field, so it is the right
    # `ProvenanceKey` field here; a sub-step that is itself a recipe already
    # carries one naming its leaf, and `as_provenance_key` leaves that alone
    # rather than renaming the state to this recipe's export name.
    output_keys = {
        field: as_provenance_key(key, out_ref.field)
        for field, out_ref in recipe.output_wiring.items()
        if out_ref.step in results and (key := results[out_ref.step].provenance_key(out_ref.field)) is not None
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
