# Scope / Cab / Recipe design (consolidated)

This file consolidates two design documents from the recipe-v3 rewrite:
the original draft (`scope_design_plan.md`) and its refinement
(`shinobi_recipe_v3.md`). The refined version is what shipped and is
reproduced below in full; the original draft is not repeated verbatim
since the "Changes from the previous draft" section at the end already
records, point by point, what it got wrong and how the refinement fixed
it (most notably: the draft's `_FUNC_REGISTRY` was dropped in favor of
`func` living directly on `StepRef` — see D5).

## Overview

`Scope` is the definition (schema, metadata, backend config). `ExecContext` is the live execution state (inputs, outputs). `StepRef` is the binding layer: a named reference to a Scope plus an optional orchestration function, wiring, and per-step constants — it is what both `@shinobi.step` and `@recipe.step` return, and what a Recipe's `steps` list contains. `Cab` and `Recipe` extend Scope with command/argv machinery and sub-step wiring respectively. There is no global function registry and no separate `Step` wrapper class.

## Usage

```python
@shinobi.step(scope=cultcargo.wsclean, backend='native')
def make_image(ctx):
    return ctx.run(size=2048)   # kwargs to run() are input overrides

result = make_image(ms="obs.ms", size=1024)
# make_image is a StepRef; result is a StepResult
# result.inputs.size == 2048 (override applied at run())
# ctx.inputs.size == 1024 (the original call's inputs, never mutated)
```

## Class hierarchy

```python
class Scope(BaseModel):
    """Definition: schema, metadata, backend config. Never has
    inputs/outputs/func fields — those live in ExecContext/StepRef.
    Dispatch never mutates a Scope; Recipe is the one deliberately
    mutable subclass (builder methods extend it before first run).
    """
    name: str
    info: str | None = None
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]
    backend: str | None = None
    input_mutability: dict[str, Mutability] = Field(default_factory=dict)

    def __call__(self, **kwargs) -> StepResult:
        """Bare execution — no orchestration function."""
        return _dispatch(self, None, **kwargs)

    def mutability_of(self, field: str) -> Mutability:
        return self.input_mutability.get(field, Mutability.IMMUTABLE)


class ExecContext:
    """Live execution state, created by _dispatch. Plain Python class.
    `inputs` is a validated snapshot for inspection; the raw caller
    kwargs are kept separately because MUTABLE fields must reach the
    backend as the caller's original objects (see D8).
    """
    scope: Scope
    inputs: BaseModel               # validated snapshot (read-only by convention)
    outputs: BaseModel | None       # populated by run()

    def run(self, *, backend: str | None = None, **overrides) -> StepResult:
        """Execute with optional input overrides. Merges overrides over
        the raw inputs, validates + applies mutability policy, executes.
        Cab: build_argv → backend.run → wranglers → StepResult
        Recipe: resolve wiring → dispatch sub-steps → aggregate StepResult
        """
        ...


class Cab(Scope):
    """Atomic step backed by a command."""
    command: str
    flavour: str = "binary"
    image: str | None = None
    policies: Policies = Field(default_factory=Policies)
    field_meta: dict[str, ParamMeta] = Field(default_factory=dict)
    input_patterns: list[ParamPattern] = Field(default_factory=list)
    wranglers: dict[str, list[str]] = Field(default_factory=dict)


class Recipe(Scope):
    """Composite step: declared sub-steps with explicit wiring."""
    steps: list[StepRef] = Field(default_factory=list)
    output_wiring: dict[str, OutputRef] = Field(default_factory=dict)

    # Builder surface (see D9)
    inputs: _InputsProxy    # property — wiring proxy, NOT runtime values
    outputs: _OutputsProxy  # property — wiring proxy, NOT runtime values

    def add_step(self, name: str, scope: Scope, **kwargs) -> "Recipe": ...
    def step(self, *, scope: Scope, backend: str | None = None, **kwargs): ...
    def set_output(self, field: str, ref: OutputRef) -> "Recipe": ...


class StepRef(BaseModel):
    """A named, executable binding of a Scope: orchestration function,
    wiring (meaningful only inside a Recipe), and per-step constants.
    Returned by @shinobi.step (free-standing) and @recipe.step (appended
    to recipe.steps). arbitrary_types_allowed is needed only for `func`.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    step: Scope                    # Cab | Recipe (both are Scopes)
    func: Callable | None = None   # orchestration function
    wiring: dict[str, InputRef | OutputRef] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)  # per-step constants

    def __call__(self, *, backend: str | None = None, **kwargs) -> StepResult:
        """Standalone execution. params are merged under kwargs; wiring is
        ignored (it can only be resolved inside a running Recipe), so any
        wired-only fields must be supplied as kwargs — input validation
        catches omissions.
        """
        return _dispatch(self.step, self.func, backend=backend,
                         **{**self.params, **kwargs})
```

## Key design decisions

### D1 — Scope is definition; ExecContext is execution; StepRef is the binding

**Scope** (pydantic BaseModel) is the definition: schema, metadata, backend config. It never has `inputs`, `outputs`, or `func` fields. `cultcargo.wsclean` is a Scope — a clean data structure you can inspect, serialize, pass around. Dispatch treats every Scope as read-only; Recipe is the one subclass that is deliberately mutable, via its builder methods, before first execution.

**ExecContext** (plain Python class) is the live execution state: validated inputs, outputs (after run). Created inside `_dispatch`, passed to the orchestration function as its first positional argument.

**StepRef** (pydantic, `arbitrary_types_allowed` for `func` only) binds a Scope to an orchestration function plus wiring/params. It is the *single* carrier of `func` — there is no global registry (see D5). The same StepRef type serves free-standing decorated steps and recipe sub-steps; recipe execution and standalone calls read `(ref.step, ref.func, ref.params)` identically.

This separation:
- Makes the boundary explicit — no "inputs=None" confusion
- Keeps Scope a clean pydantic model (no `arbitrary_types_allowed`, `model_dump()` is faithful)
- Two decorated functions over the *same* Scope, or same-named functions in different recipes, cannot collide — each has its own StepRef

### D2 — `ctx.inputs` is a read-only snapshot; overrides go to `run(**kwargs)`

`ctx.inputs` is a validated BaseModel instance representing the inputs the step was called with. The function reads from it but does not mutate it. Per-step overrides are passed to `ctx.run(**overrides)` — merged over the raw inputs, re-validated, mutability-processed, and used for execution. `StepResult.inputs` records the *effective* (post-override) inputs; `ctx.inputs` keeps the original call inspectable.

### D3 — Function returns StepResult (strict)

The orchestration function must return either the `StepResult` from `ctx.run()` or `None`:
- `None` → auto-run: `_dispatch` calls `ctx.run()` with no overrides (the common near-empty-body case)
- `StepResult` → passed through
- anything else → `TypeError` at dispatch time (silently forwarding a wrong return type hides bugs)

StepResult carries `name`, `returncode`, `stdout`, `stderr`, `outputs: BaseModel`, `inputs: BaseModel`, `.success`. For a Recipe these aggregate from sub-steps.

### D4 — Multiple `ctx.run()` calls

`ctx.run()` may be called more than once in a function body (e.g. iterative selfcal loops). Each call executes independently with its own override set; `ctx.outputs` reflects the most recent run. The function decides which StepResult to return.

### D5 — `@shinobi.step` returns a StepRef; no function registry

```python
def step(*, scope: Scope, backend: str | None = None, name: str | None = None, **params):
    def decorator(func):
        bound_scope = scope.model_copy(update={"backend": backend}) if backend else scope
        return StepRef(name=name or func.__name__, step=bound_scope,
                       func=func, params=params)
    return decorator
```

The original Scope is never mutated (`model_copy` under a new binding). `make_image` is a StepRef: calling it dispatches with `func` attached, `ctx` passed as the first positional argument. `func` travels on the StepRef itself, so there is no module-level `_FUNC_REGISTRY`, no name-collision hazard, and `Scope.__call__` stays trivially "bare execution" — calling `cultcargo.wsclean(...)` never runs anyone's orchestration function.

### D6 — `_dispatch` is the single entry point

```python
def _dispatch(scope, func, *, backend=None,
              _recipe_backend=None, _config=None, **kwargs) -> StepResult:
    ctx = ExecContext(scope, raw_inputs=kwargs, backend_override=backend,
                      recipe_backend=_recipe_backend, config=_config)
    if func is None:
        return ctx.run()
    result = func(ctx)
    if result is None:
        return ctx.run()
    if not isinstance(result, StepResult):
        raise TypeError(f"step function {func.__name__!r} must return "
                        f"StepResult or None, got {type(result).__name__}")
    return result
```

Called by `Scope.__call__` (func=None), `StepRef.__call__`, and `_run_recipe` (sub-steps). `_recipe_backend`/`_config` are explicit keyword-only parameters threaded through recursion — never mixed into input kwargs, exactly as `run_step` does today. Backend resolution priority: explicit `backend` arg > `scope.backend` > enclosing recipe's backend > `AppConfig.load().backend.default`.

### D7 — Execution helpers

- `_run_cab(cab, prepared_inputs, backend_name)` — build_argv → backend.run → wranglers → StepResult
- `_run_recipe(recipe, prepared_inputs, backend_name, config)` — iterate steps, resolve wiring, recurse via `_dispatch`, aggregate StepResult

Both receive the *prepared* (validated + mutability-processed) inputs, not the ExecContext — they have no reason to see raw state.

### D8 — `ExecContext.run()`: merge, validate, enforce mutability, execute

```python
class ExecContext:
    def __init__(self, scope, raw_inputs, *, backend_override=None,
                 recipe_backend=None, config=None):
        self.scope = scope
        self._raw = raw_inputs                       # caller's original objects
        self.inputs = scope.inputs_model(**raw_inputs)  # validated snapshot
        self.outputs = None
        self._backend_override = backend_override
        self._recipe_backend = recipe_backend
        self._config = config

    def run(self, *, backend=None, **overrides) -> StepResult:
        raw = {**self._raw, **overrides}
        prepared = _prepare_inputs(self.scope, raw)   # see below
        backend_name = (backend or self._backend_override or self.scope.backend
                        or self._recipe_backend
                        or (self._config or AppConfig.load()).backend.default)
        if isinstance(self.scope, Cab):
            result = _run_cab(self.scope, prepared, backend_name)
        else:
            result = _run_recipe(self.scope, prepared, backend_name, self._config)
        self.outputs = result.outputs
        return result
```

`_prepare_inputs` keeps the current `steps/dispatch.py` semantics unchanged — they are load-bearing: validate through `inputs_model` (missing/wrong-type raises before anything runs), then **deep-copy every field not marked MUTABLE** (revalidation does not copy), while **MUTABLE fields bypass the validated instance and read the caller's original object from the raw dict** (pydantic reconstructs containers during validation, so the validated instance's values are useless for pass-by-reference). This is why ExecContext keeps `_raw` alongside the validated `inputs` snapshot: `model_dump()` round-trips would destroy both properties.

### D9 — Recipe construction: hybrid (declarative + builder + decorator)

All three APIs produce the same `steps: list[StepRef]` and `output_wiring` — dispatch doesn't care which was used.

**Declarative:**
```python
recipe = Recipe(
    name="selfcal",
    inputs_model=SelfcalInputs,
    outputs_model=SelfcalOutputs,
    steps=[
        StepRef(name="clean", step=wsclean, wiring={"ms": InputRef(field="ms")}),
        StepRef(name="cal", step=cubical, wiring={"ms": OutputRef(step="clean", field="output_ms")}),
    ],
    output_wiring={"final_ms": OutputRef(step="cal", field="output_ms")},
)
```

**Builder with wiring proxies:**
```python
recipe = Recipe(name="selfcal", inputs_model=SelfcalInputs, outputs_model=SelfcalOutputs)

recipe.add_step("clean", wsclean,
    ms=recipe.inputs.ms,      # attribute access → InputRef(field="ms")
    size=1024,
).add_step("cal", cubical,
    ms=recipe.outputs.clean.output_ms,  # → OutputRef(step="clean", field="output_ms")
    g_time_int=10,
).set_output("final_ms", recipe.outputs.cal.output_ms)
```

**Decorator (unified with `@shinobi.step`):**
```python
@recipe.step(scope=wsclean, ms=recipe.inputs.ms, size=1024)
def clean(ctx):
    return ctx.run()

@recipe.step(scope=cubical, ms=recipe.outputs.clean.output_ms, g_time_int=10)
def cal(ctx):
    return ctx.run()

recipe.set_output("final_ms", recipe.outputs.cal.output_ms)
```

**Builder methods:**
```python
class Recipe(Scope):
    @property
    def inputs(self) -> _InputsProxy: return _InputsProxy(self)

    @property
    def outputs(self) -> _OutputsProxy: return _OutputsProxy(self)

    @staticmethod
    def _split_kwargs(kwargs):
        wiring = {k: v for k, v in kwargs.items() if isinstance(v, (InputRef, OutputRef))}
        params = {k: v for k, v in kwargs.items() if k not in wiring}
        return wiring, params

    def add_step(self, name: str, scope: Scope, **kwargs) -> "Recipe":
        wiring, params = self._split_kwargs(kwargs)
        self.steps.append(StepRef(name=name, step=scope, wiring=wiring, params=params))
        return self

    def step(self, *, scope: Scope, backend: str | None = None, **kwargs):
        def decorator(func):
            bound = scope.model_copy(update={"backend": backend}) if backend else scope
            wiring, params = self._split_kwargs(kwargs)
            ref = StepRef(name=func.__name__, step=bound, func=func,
                          wiring=wiring, params=params)
            self.steps.append(ref)
            return ref
        return decorator

    def set_output(self, field: str, ref: OutputRef) -> "Recipe":
        self.output_wiring[field] = ref
        return self
```

**Wiring proxies** validate eagerly at construction time:

```python
class _InputsProxy:
    """recipe.inputs.ms or recipe.inputs("ms") → InputRef(field="ms")."""
    def __init__(self, recipe): self._recipe = recipe
    def __call__(self, field): return self.__getattr__(field)
    def __getattr__(self, field):
        if field not in self._recipe.inputs_model.model_fields:
            raise AttributeError(
                f"'{field}' is not a field of {self._recipe.inputs_model.__name__}")
        return InputRef(field=field)

class _OutputsProxy:
    """recipe.outputs.clean.output_ms or recipe.outputs("clean", "output_ms")
    → OutputRef(step="clean", field="output_ms")."""
    def __init__(self, recipe): self._recipe = recipe
    def __call__(self, step, field): return getattr(self.__getattr__(step), field)
    def __getattr__(self, step):
        for ref in self._recipe.steps:
            if ref.name == step:
                return _StepOutputsProxy(step, ref.step.outputs_model)
        raise AttributeError(f"No step named '{step}' in recipe '{self._recipe.name}'")

class _StepOutputsProxy:
    """Two-level access — validates the field against the sub-step's
    outputs_model (the StepRef's Scope carries it, so full validation
    is possible at both levels)."""
    def __init__(self, step, outputs_model):
        self._step, self._outputs_model = step, outputs_model
    def __getattr__(self, field):
        if field not in self._outputs_model.model_fields:
            raise AttributeError(
                f"'{field}' is not an output of step '{self._step}' "
                f"({self._outputs_model.__name__})")
        return OutputRef(step=self._step, field=field)
```

Consequence of eager validation: wiring can only reference steps that already exist, so builder/decorator construction is inherently ordered (topological by construction). Declarative construction writes refs directly and is validated at run time.

**Naming note:** `recipe.inputs` / `recipe.outputs` (wiring proxies on the definition) deliberately reuse the words `inputs` / `outputs` that mean *runtime values* on ExecContext/StepResult. Same words, different layer: on a Recipe you are wiring; on a ctx/result you are reading values. Documented in D11.

### D10 — Recipe execution resolves wiring and dispatches sub-steps

```python
def _run_recipe(recipe, prepared, backend_name, config):
    results: dict[str, StepResult] = {}
    for ref in recipe.steps:
        wired = {}
        for field, source in ref.wiring.items():
            if isinstance(source, InputRef):
                wired[field] = prepared[source.field]
            else:  # OutputRef
                wired[field] = getattr(results[source.step].outputs, source.field)
        sub_kwargs = {**ref.params, **wired}  # wiring overrides params
        results[ref.name] = _dispatch(ref.step, ref.func,
                                      _recipe_backend=backend_name,
                                      _config=config, **sub_kwargs)
    outputs = {field: getattr(results[out_ref.step].outputs, out_ref.field)
               for field, out_ref in recipe.output_wiring.items()}
    return StepResult(name=recipe.name,
                      outputs=recipe.outputs_model(**outputs),
                      inputs=recipe.inputs_model(**prepared), ...)
```

Sub-step orchestration functions run because `ref.func` is passed to `_dispatch` — the StepRef carries it (D1/D5); nothing is looked up by name anywhere.

### D11 — Naming

- `Scope` — base class (definition)
- `ExecContext` — live execution state (inputs, outputs)
- `Cab(Scope)` — atomic (was `CabDef`)
- `Recipe(Scope)` — composite (was `RecipeDef`)
- `StepRef` — named binding of Scope + func + wiring/params; the one carrier of orchestration functions
- `@shinobi.step` — standalone decorator → free-standing StepRef
- `@recipe.step` — recipe method → StepRef appended to `recipe.steps`

Both decorators return a StepRef, so the collision noted in earlier drafts dissolves: same name, same return type, one appends to a recipe as a side effect. `recipe.inputs`/`recipe.outputs` are wiring proxies (definition layer); `ctx.inputs`/`result.outputs` are runtime values (execution layer) — same words, different layer, called out in docs.

### D12 — What gets deleted

- `Step` class (`steps/decorator.py`) — StepRef absorbs the (scope, func) pairing; ExecContext takes the runtime role
- `run_step()` free function — becomes `_dispatch` + `ExecContext.run`
- `CabDef` / `RecipeDef` — renamed to `Cab` / `Recipe`
- Legacy top-level modules `shinobi/schema.py`, `shinobi/decorators.py`, `shinobi/recipe.py` (the old "no declared graph" model) — deleted; `shinobi.steps` is the model, re-exported from `shinobi/__init__.py`

## Migration impact

All paths under `src/shinobi/`:

- `steps/schema.py` — `CabDef` → `Cab`, `RecipeDef` → `Recipe`, add `Scope` base, `StepRef` gains `func`/`params` and `__call__`, add proxies, add Cab command/flavour/policies/field_meta/input_patterns/wranglers fields (from the old top-level schema)
- `steps/decorator.py` — `@step(scope=, backend=, **params)` returns StepRef; delete `Step` class
- `steps/dispatch.py` — `run_step()` → `_dispatch()`; add `ExecContext`; keep `_prepare_inputs` semantics verbatim; `_run_cab` gains argv/wrangler pipeline (from the old dispatch path)
- `results.py` — `StepResult` gains `inputs` field
- `dag.py` — `graph_nodes()` takes `Recipe` (was `RecipeDef`)
- `cli.py` — update type references
- `backends/` — `StepBackend.run()` takes `Cab` (was `CabDef`)
- `loaders/` — emit `Cab` (was `CabDef`)
- `schema.py`, `decorators.py`, `recipe.py` (top-level legacy modules) — deleted
- `__init__.py` — re-export `Scope`, `ExecContext`, `Cab`, `Recipe`, `StepRef`, `step`
- `tests/` — rename `CabDef`/`RecipeDef` references; delete legacy-model tests; add coverage for: standalone StepRef call with params, two decorated functions sharing one Scope, same-named step functions in two recipes, mutability preservation through `run(**overrides)`, non-StepResult return → TypeError
- `AGENTS.md` — update class names and execution model

### Backward compatibility

None. Clean rename/restructure as part of the full rewrite. No deprecation period.

## Changes from the previous draft

- **Dropped `_FUNC_REGISTRY` entirely** — it was keyed by function name but looked up by scope name (never matched), and couldn't support two functions over one Scope or same-named functions in two recipes. `func` now lives on StepRef; both decorators return StepRef.
- **Mutability enforcement restored** — `ExecContext` keeps raw inputs alongside the validated snapshot; `run()` goes through `_prepare_inputs` (deepcopy immutables, pass mutables by original reference), never `model_dump()`.
- **`ExecContext.run()` no longer discards its merged/validated inputs** — helpers receive the prepared inputs; `ctx.outputs` is set from the result.
- **`StepRef.__call__` merges `params`** under caller kwargs instead of dropping them; wiring is ignored standalone (documented).
- **Backend/config threading made explicit** — `_dispatch(..., _recipe_backend=, _config=)` keyword-only, matching today's `run_step`; ExecContext carries them for `run()`.
- **Auto-run rule tightened** — `None` → auto-run, `StepResult` → pass through, anything else → `TypeError`.
- **`_StepOutputsProxy` now validates fields** via the sub-step Scope's `outputs_model`.
- **Fixed the closure-reassignment bug** in the decorator sketch (`bound_scope = scope.model_copy(...)`).
- **"Immutable Scope" reframed** as "definition, never mutated by dispatch"; Recipe is explicitly the mutable-by-builder exception.
- **Migration scope clarified** — files are `src/shinobi/steps/*`; legacy top-level `schema.py`/`decorators.py`/`recipe.py` are deleted in this change.

Note: this refined version is what actually shipped in the recipe-v3 rewrite (2026-07-04), with two further corrections found by post-implementation code review and folded into the real code (not reflected in the sketches above, which are historical): `_run_recipe` stops dispatching sub-steps on the first failure instead of continuing past it, and `ParamMeta` gained a `dtype` field so dynamically pattern-matched (`ParamPattern`) inputs can still be recognised as file-like for container bind-mounting. See `.claude/HANDOVER.md` for the full review record.
