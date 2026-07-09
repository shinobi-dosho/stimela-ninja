import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.graph import RecipeNotOffloadableError
from shinobi.offload import OffloadCompileError, compile_slurm, status_slurm
from shinobi.steps.schema import Cab, InputRef, OutputRef, Recipe, StepRef


class RecipeIn(BaseModel):
    ms: Path = Path("data.ms")


class MakeIn(BaseModel):
    ms: Path  # the tool writes here; it's also the step's output (passthrough)


class MSOut(BaseModel):
    ms: Path | None = None


class UseIn(BaseModel):
    ms: Path | None = None


class OkOut(BaseModel):
    ok: bool = True


def _linear_recipe(make=None, use=None):
    make = make or Cab(name="make", command="mk", inputs_model=MakeIn, outputs_model=MSOut)
    use = use or Cab(name="use", command="use", inputs_model=UseIn, outputs_model=OkOut)
    return Recipe(
        name="pipe",
        inputs_model=RecipeIn,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="use", step=use, wiring={"ms": OutputRef(step="make", field="ms")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )


def test_compiles_to_topologically_ordered_dependency_chain():
    wf = compile_slurm(_linear_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    assert wf.recipe == "pipe"
    assert [j.name for j in wf.jobs] == ["make", "use"]
    assert wf.jobs[0].depends_on == []
    assert wf.jobs[1].depends_on == ["make"]


def test_inter_step_path_flows_through_statically():
    wf = compile_slurm(_linear_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    # make writes /scratch/x.ms; use must receive that same path, resolved at
    # compile time from make's same-named-input passthrough output.
    assert "mk --ms /scratch/x.ms" in wf.jobs[0].script
    assert "use --ms /scratch/x.ms" in wf.jobs[1].script


def test_script_has_sbatch_directives_and_logs():
    wf = compile_slurm(_linear_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    script = wf.jobs[0].script
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=make" in script
    assert "#SBATCH --chdir=/work" in script
    assert "#SBATCH --output=/work/.shinobi/pipe/make.out" in script


def test_sbatch_opts_are_emitted():
    wf = compile_slurm(
        _linear_recipe(), {"ms": "/x.ms"}, workdir="/work", container_runtime=None, sbatch_opts={"partition": "gpu"}
    )
    assert "#SBATCH --partition=gpu" in wf.jobs[0].script


def test_container_image_is_wrapped_in_runtime():
    make = Cab(name="make", command="mk", inputs_model=MakeIn, outputs_model=MSOut, image="repo/img:1")
    wf = compile_slurm(_linear_recipe(make=make), {"ms": "/x.ms"}, workdir="/work", container_runtime="apptainer")
    assert "apptainer" in wf.jobs[0].script
    assert "repo/img:1" in wf.jobs[0].script


def test_non_offloadable_recipe_is_rejected():
    recipe = _linear_recipe()
    recipe.steps[1].func = lambda ctx: ctx.run()
    with pytest.raises(RecipeNotOffloadableError):
        compile_slurm(recipe, {"ms": "/x.ms"})


def test_unresolvable_inter_step_path_raises_compile_error():
    # make's output `ms` has no same-named input and no default -> can't be
    # statically resolved, so wiring it downstream is a hard compile error.
    class WhereIn(BaseModel):
        where: Path = Path("out.ms")

    make = Cab(name="make", command="mk", inputs_model=WhereIn, outputs_model=MSOut)
    recipe = Recipe(
        name="pipe",
        inputs_model=RecipeIn,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make, wiring={"where": InputRef(field="ms")}),
            StepRef(name="use", step=Cab(name="use", command="use", inputs_model=UseIn, outputs_model=OkOut),
                    wiring={"ms": OutputRef(step="make", field="ms")}),
        ],
    )
    with pytest.raises(OffloadCompileError, match="isn't statically known"):
        compile_slurm(recipe, {"ms": "/x.ms"}, container_runtime=None)


def test_unsafe_cab_name_is_rejected():
    make = Cab(name="make\ninjected", command="mk", inputs_model=MakeIn, outputs_model=MSOut)
    with pytest.raises(OffloadCompileError, match="unsafe"):
        compile_slurm(_linear_recipe(make=make), {"ms": "/x.ms"}, container_runtime=None)


def test_diamond_dependencies_are_captured():
    a = Cab(name="a", command="a", inputs_model=MakeIn, outputs_model=MSOut)
    mid = Cab(name="m", command="m", inputs_model=UseIn, outputs_model=MSOut)

    class TwoIn(BaseModel):
        left: Path | None = None
        right: Path | None = None

    join = Cab(name="j", command="j", inputs_model=TwoIn, outputs_model=OkOut)
    recipe = Recipe(
        name="diamond",
        inputs_model=RecipeIn,
        outputs_model=OkOut,
        steps=[
            StepRef(name="a", step=a, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="b", step=mid, wiring={"ms": OutputRef(step="a", field="ms")}),
            StepRef(name="c", step=mid, wiring={"ms": OutputRef(step="a", field="ms")}),
            StepRef(name="d", step=join,
                    wiring={"left": OutputRef(step="b", field="ms"), "right": OutputRef(step="c", field="ms")}),
        ],
    )
    wf = compile_slurm(recipe, {"ms": "/x.ms"}, container_runtime=None)
    by_name = {j.name: j for j in wf.jobs}
    assert by_name["b"].depends_on == ["a"]
    assert by_name["c"].depends_on == ["a"]
    assert sorted(by_name["d"].depends_on) == ["b", "c"]


def test_status_slurm_ignores_batch_and_extern_rows(monkeypatch):
    """Regression test: `status_slurm` used to just take `lines[0]` from
    `sacct` output, assuming the bare job id row always comes first before
    any `.batch`/`.extern` sub-step rows -- unlike the blocking backend's
    `_wait`, which matches on the job id field explicitly. Now both share
    `sacct_job_fields`, so `status_slurm` is robust to row order too.
    """

    def fake_run(argv, **kwargs):
        assert argv[:2] == ["sacct", "-j"]
        # deliberately out of order: sub-step rows before the bare job row
        out = "42.extern|RUNNING\n42.batch|RUNNING\n42|COMPLETED\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert status_slurm({"step": "42"}) == {"step": "COMPLETED"}


def test_status_slurm_reports_unknown_for_missing_job(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    assert status_slurm({"step": "42"}) == {"step": "UNKNOWN"}
