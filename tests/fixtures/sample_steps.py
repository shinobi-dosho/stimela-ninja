"""Shared Cab/Recipe/pydantic model fixtures for tests/test_steps_*.py."""

from __future__ import annotations

from pydantic import BaseModel

from shinobi.steps import Cab, InputRef, Mutability, OutputRef, Recipe, StepRef


class CommandOutputs(BaseModel):
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class EchoInputs(BaseModel):
    text: str = "hi"


echo_cab = Cab(name="echo", command="/bin/echo", inputs_model=EchoInputs, outputs_model=CommandOutputs)


class NameInputs(BaseModel):
    name: str = "out.txt"


class PathOutputs(BaseModel):
    # a RecordingBackend (used by tests that don't execute anything) returns
    # empty stdout/stderr, so every field here needs a default.
    path: str | None = None


make_value_cab = Cab(name="make_value", command="/bin/echo", inputs_model=NameInputs, outputs_model=PathOutputs)


class UseValueInputs(BaseModel):
    path: str | None = None


class OkOutputs(BaseModel):
    ok: bool = True


use_value_cab = Cab(name="use_value", command="/bin/echo", inputs_model=UseValueInputs, outputs_model=OkOutputs)


chained_recipe = Recipe(
    name="chained",
    inputs_model=NameInputs,
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


immutable_list_cab = Cab(name="immutable_list", command="/bin/true", inputs_model=ListInputs, outputs_model=PathOutputs)

mutable_list_cab = Cab(
    name="mutable_list",
    command="/bin/true",
    inputs_model=ListInputs,
    outputs_model=PathOutputs,
    input_mutability={"items": Mutability.MUTABLE},
)
