"""The new step-based architecture: `@shinobi.step` wraps an existing
CabDef/RecipeDef around a function. See schema.py/decorator.py/dispatch.py
docstrings, and the plan/AGENTS.md for how this relates to (and, per a
confirmed design decision, eventually replaces) the older
`shinobi.decorators`/`shinobi.recipe` model -- not yet migrated, see those
modules' own docstrings for what still runs on the old model today.
"""

from __future__ import annotations

from shinobi.steps.backend import NativeStepBackend, RecordingStepBackend, StepBackend
from shinobi.steps.decorator import Step, step
from shinobi.steps.dispatch import get_step_backend, register_step_backend, run_step
from shinobi.steps.schema import CabDef, InputRef, Mutability, OutputRef, RecipeDef, StepRef

# StepRef.step/wiring reference "Step" by forward-ref string (schema.py
# can't import decorator.py directly -- decorator.py imports dispatch.py
# imports schema.py, so schema.py -> decorator.py would be circular).
# Resolve it here, once, now that Step exists.
StepRef.model_rebuild()

__all__ = [
    "CabDef",
    "InputRef",
    "Mutability",
    "NativeStepBackend",
    "OutputRef",
    "RecipeDef",
    "RecordingStepBackend",
    "Step",
    "StepBackend",
    "StepRef",
    "get_step_backend",
    "register_step_backend",
    "run_step",
    "step",
]
