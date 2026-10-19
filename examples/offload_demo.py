"""Two path-wired steps that can compile to a Slurm workflow.

Compile without submitting with ``ninja compile examples/offload_demo.py:pipe --target /scratch/made.ms --container-runtime none``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from shinobi.steps import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


class PipeInputs(BaseModel):
    target: Path = Path("made.ms")


class TouchInputs(BaseModel):
    out: Path


class PathOutputs(BaseModel):
    # The output path is known before the command runs.
    out: Path | None = None


class CatInputs(BaseModel):
    f: Path | None = None


class OkOutputs(BaseModel):
    ok: bool = True


make = Cab(
    name="make",
    command="/bin/touch",
    inputs_model=TouchInputs,
    outputs_model=PathOutputs,
    field_meta={"out": ParamMeta(positional=True)},
)

use = Cab(
    name="use",
    command="/bin/cat",
    inputs_model=CatInputs,
    outputs_model=OkOutputs,
    field_meta={"f": ParamMeta(positional=True)},
)


pipe = Recipe(
    name="pipe",
    inputs_model=PipeInputs,
    outputs_model=OkOutputs,
    steps=[
        StepRef(name="make", step=make, wiring={"out": InputRef(field="target")}),
        StepRef(name="use", step=use, wiring={"f": OutputRef(step="make", field="out")}),
    ],
    output_wiring={"ok": OutputRef(step="use", field="ok")},
)
