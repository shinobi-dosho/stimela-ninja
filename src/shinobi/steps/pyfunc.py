"""`@shinobi.pystep`: turn a plain, type-hinted Python function into a step
without hand-writing pydantic `inputs_model`/`outputs_model` classes.

`inputs_model` is derived from the function's own parameters (via
`inspect.signature` + `typing.get_type_hints` -- not `param.annotation`
directly, since every module in this codebase uses
`from __future__ import annotations`, making raw annotations lazy strings).
`outputs_model` is derived from its return-type annotation: a `BaseModel`
subclass is used directly (the function must return an instance of it); no
annotation or `-> None` means no outputs, and the function must return
`None`. Any other return annotation is rejected at decoration time -- there
is no auto-wrapping of a bare scalar/dict return into an invented field
name, since that would be exactly the kind of implicit magic this project
avoids elsewhere.

This builds a bare `Scope` (not a `Cab`, not a `Recipe`) and wraps the
function in an adapter that returns its own `StepResult` directly, never
calling `ctx.run()` -- see `Scope`/`StepRef`'s docstrings in `schema.py` for
why a bare `Scope` is a real, supported shape, not a special case bolted on
here. `@shinobi.step` (`decorator.py`), by contrast, never introspects the
decorated function's signature at all -- `scope.inputs_model` is the schema
authority there. Use `@shinobi.pystep` when you have a plain function and no
external tool; use `@shinobi.step` when you have an existing `Cab`/`Recipe`.

Caveat: `typing.get_type_hints` resolves annotations against the function's
own module globals, so any `BaseModel` used in the signature or return type
must be defined at module level, not inside another function.

v1 always deep-copies every input before calling the function (the `Scope`
default, `Mutability.IMMUTABLE` for every field) -- there is no per-parameter
mutability override yet; add one if a real need surfaces.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

from shinobi.results import StepResult
from shinobi.steps.schema import Scope, StepRef

if TYPE_CHECKING:
    from shinobi.steps.dispatch import ExecContext

_UNSUPPORTED_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
    inspect.Parameter.POSITIONAL_ONLY,
)


def _pascal(func_name: str) -> str:
    return "".join(word.capitalize() for word in func_name.split("_") if word)


def _inputs_model_from_signature(func: Callable) -> type[BaseModel]:
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    fields: dict[str, tuple[Any, Any]] = {}
    for pname, param in sig.parameters.items():
        if param.kind in _UNSUPPORTED_KINDS:
            raise TypeError(
                f"pystep {func.__name__!r}: parameter {pname!r} is "
                f"{param.kind.description} -- only plain positional-or-keyword "
                "parameters (with a real type hint) are supported"
            )
        if pname not in hints:
            raise TypeError(
                f"pystep {func.__name__!r}: parameter {pname!r} has no type "
                "hint -- every parameter needs one so its inputs_model can be "
                "derived from the signature"
            )
        required = param.default is inspect.Parameter.empty
        fields[pname] = (hints[pname], ... if required else param.default)
    return create_model(f"{_pascal(func.__name__)}Inputs", **fields)


def _outputs_model_from_return(func: Callable) -> tuple[type[BaseModel], bool]:
    hints = get_type_hints(func)
    ret = hints.get("return")
    if ret is None or ret is type(None):
        return create_model(f"{_pascal(func.__name__)}Outputs"), True
    if isinstance(ret, type) and issubclass(ret, BaseModel):
        return ret, False
    raise TypeError(
        f"pystep {func.__name__!r}: return type {ret!r} isn't a BaseModel "
        "subclass (or None) -- declare a BaseModel and return an instance "
        "of it, rather than a bare scalar/dict/list, so outputs stay "
        "explicitly named and typed"
    )


def _make_adapter(
    func: Callable, outputs_model: type[BaseModel], is_empty: bool
) -> Callable[[ExecContext], StepResult]:
    def _adapter(ctx: ExecContext) -> StepResult:
        prepared = ctx.prepare_inputs()
        ret = func(**prepared)
        if is_empty:
            if ret is not None:
                raise TypeError(
                    f"pystep {func.__name__!r} has no declared outputs (no return "
                    f"annotation, or -> None) but returned {type(ret).__name__!r} "
                    "instead of None"
                )
            outputs: BaseModel = outputs_model()
        else:
            if not isinstance(ret, outputs_model):
                raise TypeError(
                    f"pystep {func.__name__!r} must return {outputs_model.__name__!r}, "
                    f"got {type(ret).__name__!r}"
                )
            outputs = ret
        return StepResult(
            name=ctx.scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout="",
            stderr="",
        )

    return _adapter


def pystep(
    *, name: str | None = None, info: str | None = None, **params: Any
) -> Callable[[Callable], StepRef]:
    """Decorate (or directly call on an existing function, matching
    `@shinobi.step`'s precedent: `pystep()(existing_func)`) a plain,
    type-hinted function to turn it into a `StepRef`. See the module
    docstring for the schema-derivation and outputs rules.

    No `backend` kwarg: a bare-`Scope` step never dispatches through a
    backend (`ctx.run()` isn't called), so it would be dead, misleading API
    surface. `**params` are per-call constants, same as `@shinobi.step`.
    """

    def decorator(func: Callable) -> StepRef:
        inputs_model = _inputs_model_from_signature(func)
        outputs_model, is_empty = _outputs_model_from_return(func)
        adapter = _make_adapter(func, outputs_model, is_empty)
        step_name = name or func.__name__
        scope = Scope(
            name=step_name,
            info=info if info is not None else inspect.getdoc(func),
            inputs_model=inputs_model,
            outputs_model=outputs_model,
        )
        return StepRef(name=step_name, step=scope, func=adapter, params=params)

    return decorator
