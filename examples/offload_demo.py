"""A minimal, offloadable recipe -- the smallest thing that shows
compile-and-offload end to end.

Two steps, wired by a filesystem path:
  make  -- `touch` a file at `target`
  use   -- `cat` that same file (its input is wired from make's output)

Because the only thing crossing between steps is a *path* (no orchestration
functions, no MUTABLE inputs, no wrangler-derived values), this recipe is
offload-eligible: `check_offloadable` accepts it, and `compile_slurm` turns
it into two `sbatch` scripts linked by `--dependency=afterok`.

Try it:

    # see the compiled Slurm workflow (no cluster needed, nothing submitted)
    ninja compile examples/offload_demo.py:pipe --target /scratch/made.ms \\
        --container-runtime none

    # or, run it locally instead (the same recipe, driven in-process)
    ninja run examples/offload_demo.py:pipe --target /tmp/made.ms

    # and to actually hand it to a real Slurm cluster and detach:
    ninja compile examples/offload_demo.py:pipe --target /scratch/made.ms \\
        --container-runtime none --submit
    ninja status /scratch/.shinobi/pipe/handle.json
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from shinobi.steps import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


class PipeInputs(BaseModel):
    target: Path = Path("made.ms")  # where `make` writes, and `use` reads


class TouchInputs(BaseModel):
    out: Path  # the file to create (a positional arg to `touch`)


class PathOutputs(BaseModel):
    # `out` is a passthrough of the `out` input -- so its value is knowable
    # statically at compile time (no need to run the step to learn the path).
    out: Path | None = None


class CatInputs(BaseModel):
    f: Path | None = None  # the file to read (a positional arg to `cat`)


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
        # make.out <- the recipe's own `target` input
        StepRef(name="make", step=make, wiring={"out": InputRef(field="target")}),
        # use.f <- make.out  (this OutputRef is the inter-step dependency edge)
        StepRef(name="use", step=use, wiring={"f": OutputRef(step="make", field="out")}),
    ],
    output_wiring={"ok": OutputRef(step="use", field="ok")},
)
