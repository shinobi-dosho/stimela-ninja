"""`@shinobi.step`: bind an orchestration function to a Scope.

Returns a `StepRef` (the same type `@recipe.step` produces) -- the single
carrier of the orchestration function. There is no global function
registry: `func` travels on the StepRef itself, so two functions over one
Scope, or same-named functions in different recipes, never collide.

The decorated name is callable and dispatches with `ctx` passed as the
first positional argument; the function returns either the `StepResult`
from `ctx.run()` or `None` (auto-run). The function's own signature is
never introspected -- `scope.inputs_model` is the schema authority.
"""

from __future__ import annotations

from typing import Any, Callable

from shinobi.steps.schema import Scope, ScatterSpec, StepRef


def step(
    scope: Scope,
    *,
    backend: str | None = None,
    name: str | None = None,
    scatter: list[str] | ScatterSpec | None = None,
    **params: Any,
) -> Callable[[Callable], StepRef]:
    """Decorate a function with an existing Scope (Cab or Recipe). See the
    module docstring.
    """

    def decorator(func: Callable) -> StepRef:
        """Bind `func` as the orchestration function for `scope`.

        Args:
            func: The orchestration function to bind. Its own signature is
                never introspected.

        Returns:
            A `StepRef` carrying `scope`, `func`, the step's params, and
            its scatter spec.
        """
        bound_scope = scope.with_backend(backend)
        scatter_spec = ScatterSpec(fields=scatter) if isinstance(scatter, list) else scatter
        return StepRef(
            name=name or func.__name__,
            step=bound_scope,
            func=func,
            params=params,
            scatter=scatter_spec,
        )

    return decorator
