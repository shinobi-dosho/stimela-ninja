"""Tiny Cab / Recipe / StepRef targets used by tests/test_cli.py, and as a
concrete example of the 'path/to/file.py:name' target syntax `ninja run`
expects.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from shinobi.steps import Cab, InputRef, OutputRef, Recipe, StepRef, step


class GreetInputs(BaseModel):
    text: str = "hi"


class CommandOutputs(BaseModel):
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


greet = Cab(name="greet", command="/bin/echo", info="Echo TEXT back.", inputs_model=GreetInputs,
            outputs_model=CommandOutputs)


class GreetImageInputs(BaseModel):
    restored_image: str  # required, and underscored to check CLI flag round-tripping


greet_image = Cab(
    name="greet_image",
    command="/bin/echo",
    info="A cab with an underscored param, to check CLI flag round-tripping.",
    inputs_model=GreetImageInputs,
    outputs_model=CommandOutputs,
)


class NoInputs(BaseModel):
    pass


fail = Cab(name="fail", command="/bin/false", info="Always fails.", inputs_model=NoInputs,
           outputs_model=CommandOutputs)


@step(scope=greet)
def greet_step(ctx):
    """A decorated step target for the CLI."""
    return ctx.run()


# -- recipe target for --dryrun graph rendering --


class MakeInputs(BaseModel):
    name: str = "out.txt"


class PathOutputs(BaseModel):
    path: str | None = None


class UseInputs(BaseModel):
    path: str | None = None


class OkOutputs(BaseModel):
    ok: bool = True


make_file = Cab(name="make_file", command="/bin/echo", inputs_model=MakeInputs, outputs_model=PathOutputs)
use_file = Cab(name="use_file", command="/bin/echo", inputs_model=UseInputs, outputs_model=OkOutputs)


chained = Recipe(
    name="chained",
    inputs_model=MakeInputs,
    outputs_model=OkOutputs,
    steps=[
        StepRef(name="make_file", step=make_file, wiring={"name": InputRef(field="name")}),
        StepRef(name="use_file", step=use_file, wiring={"path": OutputRef(step="make_file", field="path")}),
    ],
    output_wiring={"ok": OutputRef(step="use_file", field="ok")},
)


# -- offloadable recipe target for `ninja compile` (Path-typed data flow) --


class MSRecipeInputs(BaseModel):
    ms: Path = Path("data.ms")


class MSMakeInputs(BaseModel):
    ms: Path  # the tool writes here; passthrough to its output


class MSOutputs(BaseModel):
    ms: Path | None = None


class MSUseInputs(BaseModel):
    ms: Path | None = None


ms_make = Cab(name="ms_make", command="mk", inputs_model=MSMakeInputs, outputs_model=MSOutputs)
ms_use = Cab(name="ms_use", command="use", inputs_model=MSUseInputs, outputs_model=OkOutputs)


path_pipe = Recipe(
    name="path_pipe",
    inputs_model=MSRecipeInputs,
    outputs_model=OkOutputs,
    steps=[
        StepRef(name="ms_make", step=ms_make, wiring={"ms": InputRef(field="ms")}),
        StepRef(name="ms_use", step=ms_use, wiring={"ms": OutputRef(step="ms_make", field="ms")}),
    ],
    output_wiring={"ok": OutputRef(step="ms_use", field="ok")},
)
