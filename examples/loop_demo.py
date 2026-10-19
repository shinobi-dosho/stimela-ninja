"""Show a bounded ``Recipe.add_loop`` using shell commands.

Each cycle appends a line; a sentinel ends work after three cycles. Inspect
all five declared cycles with ``ninja run examples/loop_demo.py:pipeline --dryrun``.
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
    """Run ``sh -c`` with the script first and inputs as positional args."""
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


# Clear the work file and sentinel before each run.
prepare = _sh("prepare", 'rm -f "$1"; : > "$0"', PrepareIn, PrepareOut)

# Pass the work path through each cycle.
refine = _sh("refine", 'printf "refined on cycle %s\\n" "$1" >> "$0"', RefineIn, RefineOut)

# The sentinel path signals convergence.
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
    # Carry the work path into the next cycle.
    carry={"work": "work"},
    index_input="cycle",
    work=pipeline.outputs.prepare.work,
    flag=pipeline.inputs.flag,
)
pipeline.set_output("work", loop.outputs.work)
