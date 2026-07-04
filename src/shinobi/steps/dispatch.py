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

import copy
import warnings
from typing import Any, Callable

from shinobi.config import AppConfig
from shinobi.policies import build_argv
from shinobi.results import StepResult
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, Recipe, Scope
from shinobi.wranglers import apply_wranglers

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
    if name in _STEP_BACKENDS:
        return _STEP_BACKENDS[name]
    from shinobi.backends import get_backend

    return get_backend(name)


def _prepare_inputs(
    scope: Scope, kwargs: dict[str, Any], *, validated: Any = None
) -> dict[str, Any]:
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
        validated = scope.inputs_model(**kwargs)
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
    ):
        self.scope = scope
        self._raw = raw_inputs
        self.inputs = scope.inputs_model(**raw_inputs)
        self.outputs = None
        self._backend_override = backend_override
        self._recipe_backend = recipe_backend
        self._config = config

    def run(self, *, backend: str | None = None, **overrides: Any) -> StepResult:
        raw = {**self._raw, **overrides}
        # No overrides -> `raw` is exactly what `self.inputs` already
        # validated in __init__; reuse it instead of re-validating.
        validated = self.inputs if not overrides else None
        prepared = _prepare_inputs(self.scope, raw, validated=validated)
        backend_name = (
            backend
            or self._backend_override
            or self.scope.backend
            or self._recipe_backend
            or (self._config or AppConfig.load()).backend.default
        )
        if isinstance(self.scope, Cab):
            result = _run_cab(self.scope, prepared, backend_name)
        else:
            result = _run_recipe(self.scope, prepared, backend_name, self._config)
        self.outputs = result.outputs
        return result


def _dispatch(
    scope: Scope,
    func: Callable | None,
    *,
    backend: str | None = None,
    _recipe_backend: str | None = None,
    _config: AppConfig | None = None,
    **kwargs: Any,
) -> StepResult:
    ctx = ExecContext(
        scope,
        kwargs,
        backend_override=backend,
        recipe_backend=_recipe_backend,
        config=_config,
    )
    if func is None:
        return ctx.run()
    result = func(ctx)
    if result is None:
        return ctx.run()
    if not isinstance(result, StepResult):
        raise TypeError(
            f"step function {getattr(func, '__name__', func)!r} must return "
            f"StepResult or None, got {type(result).__name__}"
        )
    return result


def _fill_outputs(cab: Cab, prepared: dict[str, Any], run, wrangled: dict[str, Any]):
    """Fill the cab's outputs_model by priority: wrangler value >
    same-named final input > reserved run field (returncode/stdout/stderr)
    > field default.
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
    return cab.outputs_model(**values)


def _run_cab(cab: Cab, prepared: dict[str, Any], backend_name: str) -> StepResult:
    argv = build_argv(cab, prepared)
    backend = get_step_backend(backend_name)
    # The backend gets the prepared dict (not a rebuilt model) so MUTABLE
    # fields reach it as the caller's own objects by reference -- rebuilding
    # a pydantic model here would deep-copy every container and break that.
    run = backend.run(cab, argv, prepared)
    lines = run.stdout.splitlines() + run.stderr.splitlines()
    wrangled = apply_wranglers(cab.wranglers, lines)
    outputs = _fill_outputs(cab, prepared, run, wrangled)
    return StepResult(
        name=cab.name,
        returncode=run.returncode,
        outputs=outputs,
        inputs=cab.inputs_model(**prepared),
        stdout=run.stdout,
        stderr=run.stderr,
    )


def _run_recipe(
    recipe: Recipe, prepared: dict[str, Any], backend_name: str, config: AppConfig | None
) -> StepResult:
    results: dict[str, StepResult] = {}
    stdouts: list[str] = []
    stderrs: list[str] = []
    returncode = 0

    for ref in recipe.steps:
        wired: dict[str, Any] = {}
        for field, source in ref.wiring.items():
            if isinstance(source, InputRef):
                wired[field] = prepared[source.field]
            elif isinstance(source, OutputRef):
                wired[field] = getattr(results[source.step].outputs, source.field)
        sub_kwargs = {**ref.params, **wired}  # wiring overrides params
        res = _dispatch(
            ref.step, ref.func, _recipe_backend=backend_name, _config=config, **sub_kwargs
        )
        results[ref.name] = res
        stdouts.append(res.stdout)
        stderrs.append(res.stderr)
        if res.returncode != 0:
            returncode = res.returncode
            break

    outputs = {
        field: getattr(results[out_ref.step].outputs, out_ref.field)
        for field, out_ref in recipe.output_wiring.items()
        if out_ref.step in results
    }
    return StepResult(
        name=recipe.name,
        returncode=returncode,
        outputs=recipe.outputs_model(**outputs),
        inputs=recipe.inputs_model(**prepared),
        stdout="\n".join(s for s in stdouts if s),
        stderr="\n".join(s for s in stderrs if s),
    )
