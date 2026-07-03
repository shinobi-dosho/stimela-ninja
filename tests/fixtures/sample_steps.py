"""Shared CabDef/RecipeDef/pydantic model fixtures for tests/test_steps_*.py,
mirroring tests/fixtures/sample_targets.py's role for the old-model
tests/test_cli.py.
"""

from __future__ import annotations

from pydantic import BaseModel

from shinobi.steps import CabDef, InputRef, Mutability, OutputRef, RecipeDef, StepRef, step


class CommandOutputs(BaseModel):
    """The real output shape NativeStepBackend.run() always returns."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class EchoInputs(BaseModel):
    text: str = "hi"


echo_cab = CabDef(name="echo", command="/bin/echo", inputs_model=EchoInputs, outputs_model=CommandOutputs)


class NameInputs(BaseModel):
    name: str = "out.txt"


class PathOutputs(BaseModel):
    # a RecordingStepBackend (used by the dispatch/decorator tests that
    # don't execute anything for real) returns {}, so every field here
    # needs a default.
    path: str | None = None


make_value_cab = CabDef(
    name="make_value", command="/bin/echo", inputs_model=NameInputs, outputs_model=PathOutputs
)


class UseValueInputs(BaseModel):
    path: str | None = None


class OkOutputs(BaseModel):
    ok: bool = True


use_value_cab = CabDef(
    name="use_value", command="/bin/echo", inputs_model=UseValueInputs, outputs_model=OkOutputs
)


class ChainedInputs(BaseModel):
    name: str = "out.txt"


chained_recipe = RecipeDef(
    name="chained",
    inputs_model=ChainedInputs,
    outputs_model=OkOutputs,
    steps=[
        StepRef(name="make", step=make_value_cab, wiring={"name": InputRef(field="name")}),
        StepRef(name="use", step=use_value_cab, wiring={"path": OutputRef(step="make", field="path")}),
    ],
    output_wiring={"ok": OutputRef(step="use", field="ok")},
)


# -- mutability fixtures --


class ListInputs(BaseModel):
    items: list[int] = []


immutable_list_cab = CabDef(
    name="immutable_list", command="/bin/true", inputs_model=ListInputs, outputs_model=PathOutputs
)

mutable_list_cab = CabDef(
    name="mutable_list",
    command="/bin/true",
    inputs_model=ListInputs,
    outputs_model=PathOutputs,
    input_mutability={"items": Mutability.MUTABLE},
)


@step(immutable_list_cab)
def append_to_immutable(items: list[int]):
    items.append(99)
    return None


@step(mutable_list_cab)
def append_to_mutable(items: list[int]):
    items.append(99)
    return None
