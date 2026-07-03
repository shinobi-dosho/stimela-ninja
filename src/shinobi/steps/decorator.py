"""`@shinobi.step`: wrap an existing CabDef/RecipeDef around a function.

Unlike the old `@cab` (shinobi.decorators), there is no signature-inference
fallback -- `defn.inputs_model`/`outputs_model` are the schema, full stop;
the decorated function's own signature is never introspected. The function
receives (mutability-processed) inputs as ordinary keyword arguments, and
its return value carries any transform: `None` means "use inputs
unchanged" (the common case -- a near-empty step body, mirroring the old
`@cab`'s near-empty stub functions); a returned `dict[str, Any]` is merged
over the resolved inputs as overrides before the command/sub-steps run.

(Python gives no way to observe a callee's local-variable reassignments
after it returns without frame/exec tricks -- which this project's own
"never eval()/exec()" stance rules out anyway -- so ordinary
arguments-in/return-value-out is the mechanism, not literal namespace
injection.)
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from shinobi.steps.dispatch import run_step
from shinobi.steps.schema import CabDef, RecipeDef


class Step:
    """The callable produced by @step: a thin, named wrapper around a
    CabDef/RecipeDef plus the orchestration function that runs between
    resolving inputs and dispatching to the command/sub-steps.
    """

    def __init__(self, defn: CabDef | RecipeDef, func: Callable[..., dict[str, Any] | None]):
        self.defn = defn
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, **kwargs: Any) -> Any:
        return run_step(self.defn, self.func, **kwargs)


def step(defn: CabDef | RecipeDef) -> Callable[[Callable[..., dict[str, Any] | None]], Step]:
    """Decorate a function with an existing CabDef/RecipeDef. See module
    docstring.
    """

    def decorator(func: Callable[..., dict[str, Any] | None]) -> Step:
        return Step(defn, func)

    return decorator
