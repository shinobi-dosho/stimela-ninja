"""A no-op backend for `ninja run --dryrun`: records what WOULD run
instead of running it. See shinobi.dag for how the resulting trace
becomes a dependency graph, and shinobi.cli for how every registered
backend class gets its run() swapped out for this one during a dry run
of a @recipe target (a recipe constructs its own backend internally, so
intercepting it means patching the backend classes themselves, not just
shinobi.backends.get_backend -- see the module docstring in cli.py).

Not registered in the normal backend registry: this is only ever
constructed directly by the --dryrun code path, never selectable via
--backend/config, since it doesn't actually run anything.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from shinobi.backends import Backend
from shinobi.dag import TraceStep, find_dependencies, placeholder
from shinobi.results import Result
from shinobi.schema import CabDef


class TraceBackend(Backend):
    name = "trace"

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        call_id = len(self.steps)
        depends_on = find_dependencies(params)
        if not depends_on and self.steps:
            depends_on = {self.steps[-1].id}
        self.steps.append(TraceStep(id=call_id, name=cab.name, depends_on=depends_on))

        outputs = {name: placeholder(call_id, name) for name in cab.outputs}
        return Result(cab_name=cab.name, returncode=0, stdout="", stderr="", outputs=outputs)


@contextmanager
def patch_all_backends(tracer: TraceBackend):
    """Monkeypatch every registered backend *class*'s run() to redirect to
    `tracer`, for the duration of the context.

    This has to patch the classes themselves, not
    shinobi.backends.get_backend or shinobi.recipe.call: a recipe
    typically does ``from shinobi.backends import get_backend`` at its
    own module's top level, and patching a module-level function
    wouldn't affect a name already bound that way in the recipe's own
    namespace. A class's `run` method, though, is looked up dynamically
    via the instance's type at call time -- patching it here reaches
    every instance regardless of how the recipe imported anything.
    """
    from shinobi.backends import registered_backend_classes

    def _traced_run(self, cab, argv, params, _tracer=tracer):
        return _tracer.run(cab, argv, params)

    originals = {cls: cls.run for cls in registered_backend_classes()}
    for cls in originals:
        cls.run = _traced_run
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.run = original
