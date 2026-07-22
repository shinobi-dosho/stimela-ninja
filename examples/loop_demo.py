"""Declared loops -- a runnable `Recipe.add_loop` demo using only `sh`.

The shape is self-calibration's: repeat a cycle until the result is good
enough, but expressed so the graph is still declared and inspectable before
anything runs. `add_loop` unrolls the body `max_iter` times into the parent
recipe, so `--dryrun` draws every iteration -- and once a cycle writes the
sentinel file, the remaining ones pass its outputs through without running.

Each cycle appends a line to a work file and then "assesses" it: the assessor
writes the sentinel once the file has reached `target` lines. That stands in
for a real fidelity check (caracal's `aimfast` JSON) while keeping the example
runnable anywhere.

Run it::

    ninja run examples/loop_demo.py:pipeline --dryrun   # all 5 cycles, declared
    ninja run examples/loop_demo.py:pipeline

Three cycles run and two are skipped, since the work file reaches 3 lines on
the third pass. The `prepare` step clears any sentinel from a previous run, so
it is repeatable.
"""

from pathlib import Path

from pydantic import BaseModel

from shinobi import Cab, Recipe
from shinobi.steps.schema import ParamMeta

MAX_CYCLES = 5
TARGET_LINES = 3


class PrepareIn(BaseModel):
    script: str = ""
    work: Path
    flag: Path


class PrepareOut(BaseModel):
    work: Path | None = None


class RefineIn(BaseModel):
    script: str = ""
    work: Path
    cycle: int = 1


class RefineOut(BaseModel):
    work: Path | None = None


class AssessIn(BaseModel):
    script: str = ""
    work: Path
    flag: Path
    target: int = TARGET_LINES


class AssessOut(BaseModel):
    flag: Path | None = None


class CycleIn(BaseModel):
    work: Path
    flag: Path
    cycle: int = 1


class CycleOut(BaseModel):
    work: Path | None = None
    converged: Path | None = None


class PipelineIn(BaseModel):
    work: Path = Path("loop-demo-work.txt")
    flag: Path = Path("loop-demo-converged.flag")


class PipelineOut(BaseModel):
    work: Path | None = None


def _sh(name: str, script: str, inputs, outputs) -> Cab:
    """A cab that runs `sh -c <script> <args...>`.

    The script is a `positional_head` implicit, so it lands immediately after
    `-c`; the remaining fields follow as positionals, which `sh` exposes as
    `$0`, `$1`, ... in declaration order.
    """
    return Cab(
        name=name,
        command="sh -c",
        inputs_model=inputs,
        outputs_model=outputs,
        backend="native",
        field_meta={
            "script": ParamMeta(implicit=script, positional_head=True),
            **{f: ParamMeta(positional=True) for f in inputs.model_fields if f != "script"},
        },
    )


# Start from a clean slate so the example is repeatable: truncate the work
# file and remove any sentinel left by an earlier run.
prepare = _sh("prepare", 'rm -f "$1"; : > "$0"', PrepareIn, PrepareOut)

# `work` is both an input and an output, so its path passes through unchanged
# -- exactly the "fixed point" a loop body must be for `carry` to work.
refine = _sh("refine", 'printf "refined on cycle %s\\n" "$1" >> "$0"', RefineIn, RefineOut)

# Writes the sentinel only once the work file is long enough. Its *existence*
# is the convergence signal -- which is why `until` names a path rather than a
# bool: the identical test works in-process and as a shell guard in an
# offloaded sbatch script. Always exits 0; "not converged yet" is not a failure.
assess = _sh("assess", 'if [ "$(wc -l < "$0")" -ge "$2" ]; then touch "$1"; fi', AssessIn, AssessOut)

cycle = Recipe(name="cycle", inputs_model=CycleIn, outputs_model=CycleOut)
cycle.add_step("refine", refine, work=cycle.inputs.work, cycle=cycle.inputs.cycle)
cycle.add_step("assess", assess, work=cycle.outputs.refine.work, flag=cycle.inputs.flag)
cycle.set_output("work", cycle.outputs.refine.work)
cycle.set_output("converged", cycle.outputs.assess.flag)

pipeline = Recipe(name="pipeline", inputs_model=PipelineIn, outputs_model=PipelineOut)
pipeline.add_step("prepare", prepare, work=pipeline.inputs.work, flag=pipeline.inputs.flag)
loop = pipeline.add_loop(
    "refine_until_good",
    cycle,
    max_iter=MAX_CYCLES,
    until="converged",
    # This cycle's work file becomes the next cycle's input: the loop-carried
    # dependency, and the real graph edge between iterations.
    carry={"work": "work"},
    index_input="cycle",
    work=pipeline.outputs.prepare.work,
    flag=pipeline.inputs.flag,
)
pipeline.set_output("work", loop.outputs.work)
