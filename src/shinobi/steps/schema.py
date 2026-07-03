"""Schema types for the new step-based architecture: `CabDef`/`RecipeDef`
are pydantic-model-backed descriptions of a step's inputs/outputs, and a
`RecipeDef` declares its sub-steps' wiring explicitly (which output field
of step A feeds which input field of step B).

This is a deliberate reversal of the "recipes are plain Python, no
declared graph" position in the old `shinobi.schema`/`shinobi.decorators`
module (still present, still fully functional, not touched by this
prototype) -- see AGENTS.md and the plan that introduced this module for
why. `shinobi.steps` is additive: nothing in `shinobi.schema`,
`shinobi.decorators`, `shinobi.recipe`, `shinobi.cli`, or `shinobi.backends`
is changed or removed by adding this.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # only for type checkers -- decorator.py imports dispatch.py imports
    # this module, so an unconditional import here would be circular at
    # runtime. StepRef.model_rebuild() (steps/__init__.py) resolves the
    # "Step" forward-ref string once Step actually exists.
    from shinobi.steps.decorator import Step


class Mutability(str, Enum):
    """Whether a step's input may be changed by the step's own
    orchestration function without that change propagating back to the
    caller's object.
    """

    IMMUTABLE = "immutable"  # default: deep-copied before the step body runs
    MUTABLE = "mutable"  # opt-in: passed by reference, in-place changes persist


class CabDef(BaseModel):
    """An atomic step backed by a single command.

    `inputs_model`/`outputs_model` are plain pydantic model *classes* (not
    instances) describing the step's parameters -- reusable independently
    of shinobi, and the actual schema authority: `@shinobi.step` never
    derives one from a function signature the way the old `@cab` does.
    """

    name: str
    command: str
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]
    input_mutability: dict[str, Mutability] = Field(default_factory=dict)
    backend: str | None = None  # backend-resolution priority 1, see steps.dispatch
    info: str | None = None

    def mutability_of(self, field: str) -> Mutability:
        return self.input_mutability.get(field, Mutability.IMMUTABLE)


class InputRef(BaseModel):
    """Wiring source: this sub-step's input comes from the enclosing
    RecipeDef's own input field `field`.
    """

    field: str


class OutputRef(BaseModel):
    """Wiring source: this input (or the enclosing RecipeDef's own output,
    in `RecipeDef.output_wiring`) comes from step `step`'s output field
    `field`.
    """

    step: str
    field: str


class StepRef(BaseModel):
    """One named sub-step inside a RecipeDef, plus how its own inputs are
    wired. `step` may be a bare CabDef/RecipeDef, or an already-`@step`-
    decorated `Step` (its own orchestration function included, reused as
    a sub-step) -- `arbitrary_types_allowed` is needed here only for the
    `Step` case; CabDef/RecipeDef still validate structurally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    step: "CabDef | RecipeDef | Step"
    wiring: dict[str, "InputRef | OutputRef"] = Field(default_factory=dict)


class RecipeDef(BaseModel):
    """A composite step: named sub-steps plus the wiring between them."""

    name: str
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]
    input_mutability: dict[str, Mutability] = Field(default_factory=dict)
    backend: str | None = None  # backend-resolution priority 2, see steps.dispatch
    info: str | None = None
    steps: list[StepRef] = Field(default_factory=list)
    # recipe's own output field name -> which sub-step's output feeds it
    output_wiring: dict[str, OutputRef] = Field(default_factory=dict)

    def mutability_of(self, field: str) -> Mutability:
        return self.input_mutability.get(field, Mutability.IMMUTABLE)


# StepRef.step/wiring reference "Step" (defined in steps.decorator) by
# forward-ref string; steps/__init__.py resolves this once, after Step
# exists, via StepRef.model_rebuild().
