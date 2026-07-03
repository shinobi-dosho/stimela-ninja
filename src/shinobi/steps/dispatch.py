"""Dispatch for the new CabDef/RecipeDef step model: resolves inputs
(with mutability enforcement), runs the step's own orchestration function,
and recurses into sub-steps for a RecipeDef.

Backend resolution priority: a CabDef's own `backend` wins if set; else
the enclosing RecipeDef's `backend`; else `AppConfig.load().backend.default`
-- reusing shinobi.config's existing, already-layered (CLI override > env
var > config file > built-in default) mechanism rather than inventing a
second one (e.g. a contextvar) that would do the same job.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from pydantic import BaseModel

from shinobi.config import AppConfig
from shinobi.steps.backend import NativeStepBackend, StepBackend
from shinobi.steps.schema import CabDef, InputRef, Mutability, OutputRef, RecipeDef

_STEP_BACKENDS: dict[str, StepBackend] = {"native": NativeStepBackend()}


def register_step_backend(name: str, backend: StepBackend) -> None:
    """Register a StepBackend instance under `name`, so CabDef.backend/
    RecipeDef.backend/run_step(backend=...) can select it. Mainly useful
    for tests (e.g. registering a RecordingStepBackend); real backends
    are registered once at import time, as "native" is above.
    """
    _STEP_BACKENDS[name] = backend


def get_step_backend(name: str) -> StepBackend:
    try:
        return _STEP_BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown step backend '{name}' (available: {sorted(_STEP_BACKENDS)})") from None


def _prepare_inputs(defn: CabDef | RecipeDef, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate kwargs through inputs_model, then deep-copy every field
    not explicitly marked MUTABLE -- the actual enforcement mechanism.
    Re-validating an already-validated instance of the exact type through
    pydantic does NOT itself copy it (revalidate_instances="never" by
    default), so the deepcopy step below is load-bearing, not redundant.

    Constructing `defn.inputs_model(**kwargs)` is used for validation
    (missing/wrong-type fields raise here, before anything runs), but its
    *values* can't be used for MUTABLE fields even to skip a copy: pydantic
    already reconstructs container fields (e.g. list) during validation,
    so `validated.some_list is kwargs["some_list"]` is False even though
    nothing was meant to copy it. MUTABLE fields therefore read the
    caller's original object straight out of `kwargs`, bypassing the
    validated instance entirely -- true pass-by-reference, not "pydantic's
    copy, but we chose not to make a second one."
    """
    validated = defn.inputs_model(**kwargs)
    prepared: dict[str, Any] = {}
    for name in type(validated).model_fields:
        if defn.mutability_of(name) is Mutability.MUTABLE:
            value = kwargs[name] if name in kwargs else getattr(validated, name)
        else:
            value = copy.deepcopy(getattr(validated, name))
        prepared[name] = value
    return prepared


def run_step(
    defn: CabDef | RecipeDef,
    func: Callable[..., dict[str, Any] | None] | None,
    *,
    backend: str | None = None,
    _recipe_backend: str | None = None,
    _config: AppConfig | None = None,
    **kwargs: Any,
) -> BaseModel:
    """Run a single CabDef or RecipeDef step.

    `backend` is an explicit override (priority 0, above even the
    CabDef's own declared backend) -- mirrors `ninja run --backend`
    already overriding things today. `_recipe_backend`/`_config` are
    threaded through recursive calls for RecipeDef sub-steps; not part
    of the public single-step-call surface.
    """
    prepared = _prepare_inputs(defn, kwargs)
    overrides = func(**prepared) if func is not None else None
    final_kwargs = {**prepared, **(overrides or {})}
    final = defn.inputs_model(**final_kwargs)  # re-validate the merged result

    backend_name = backend or defn.backend or _recipe_backend or (_config or AppConfig.load()).backend.default

    if isinstance(defn, CabDef):
        step_backend = get_step_backend(backend_name)
        raw_outputs = step_backend.run(defn, final)
        return defn.outputs_model(**raw_outputs)

    return _run_recipe(defn, final, backend_name, _config)


def _run_recipe(
    defn: RecipeDef, final: BaseModel, backend_name: str, config: AppConfig | None
) -> BaseModel:
    results: dict[str, BaseModel] = {}

    for ref in defn.steps:
        sub_kwargs: dict[str, Any] = {}
        for field, source in ref.wiring.items():
            if isinstance(source, InputRef):
                sub_kwargs[field] = getattr(final, source.field)
            elif isinstance(source, OutputRef):
                sub_kwargs[field] = getattr(results[source.step], source.field)

        from shinobi.steps.decorator import Step  # avoid a circular import at module load

        if isinstance(ref.step, Step):
            sub_defn, sub_func = ref.step.defn, ref.step.func
        else:
            sub_defn, sub_func = ref.step, None

        results[ref.name] = run_step(
            sub_defn, sub_func, _recipe_backend=backend_name, _config=config, **sub_kwargs
        )

    outputs = {
        field: getattr(results[out_ref.step], out_ref.field)
        for field, out_ref in defn.output_wiring.items()
    }
    return defn.outputs_model(**outputs)
