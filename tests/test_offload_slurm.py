import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.exceptions import BackendError
from shinobi.graph import RecipeNotOffloadableError
from shinobi.offload import OffloadCompileError, compile_slurm, status_slurm, submit_slurm
from shinobi.resources import Resources
from shinobi.steps.schema import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


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
    wf = compile_slurm(_linear_recipe(), {"ms": "/x.ms"}, workdir="/work", container_runtime=None, sbatch_opts={"partition": "gpu"})
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
            StepRef(name="use", step=Cab(name="use", command="use", inputs_model=UseIn, outputs_model=OkOut), wiring={"ms": OutputRef(step="make", field="ms")}),
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
            StepRef(name="d", step=join, wiring={"left": OutputRef(step="b", field="ms"), "right": OutputRef(step="c", field="ms")}),
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
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    assert status_slurm({"step": "42"}) == {"step": "UNKNOWN"}


def _find_script_dir(workdir: Path) -> Path | None:
    matches = list(workdir.glob("shinobi-slurm-*"))
    return matches[0] if matches else None


def test_submit_slurm_removes_script_dir_on_success(tmp_path, monkeypatch):
    wf = compile_slurm(_linear_recipe(), {"ms": "/x.ms"}, workdir=str(tmp_path), container_runtime=None)

    def fake_run(argv, **kwargs):
        # the script file must exist and be readable at submission time,
        # exactly like a real sbatch invocation would need
        assert Path(argv[-1]).exists()
        return subprocess.CompletedProcess(argv, 0, stdout="99\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    submit_slurm(wf, workdir=str(tmp_path))
    assert _find_script_dir(tmp_path) is None


def test_submit_slurm_removes_script_dir_on_failure(tmp_path, monkeypatch):
    wf = compile_slurm(_linear_recipe(), {"ms": "/x.ms"}, workdir=str(tmp_path), container_runtime=None)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom"),
    )
    with pytest.raises(BackendError):
        submit_slurm(wf, workdir=str(tmp_path))
    assert _find_script_dir(tmp_path) is None


# --- unrolled loops (Recipe.add_loop) ------------------------------------


class CycleIn(BaseModel):
    ms: Path
    flag: Path


class CycleOut(BaseModel):
    ms: Path | None = None
    converged: Path | None = None


class AssessIn(BaseModel):
    ms: Path
    flag: Path


class FlagOut(BaseModel):
    flag: Path | None = None


def _loop_recipe(max_iter: int = 3) -> Recipe:
    work = Cab(name="work", command="wk", inputs_model=MakeIn, outputs_model=MSOut)
    assess = Cab(name="assess", command="as", inputs_model=AssessIn, outputs_model=FlagOut)
    body = Recipe(name="cycle", inputs_model=CycleIn, outputs_model=CycleOut)
    body.add_step("work", work, ms=InputRef(field="ms"))
    body.add_step("assess", assess, ms=OutputRef(step="work", field="ms"), flag=InputRef(field="flag"))
    body.set_output("ms", OutputRef(step="work", field="ms"))
    body.set_output("converged", OutputRef(step="assess", field="flag"))

    recipe = Recipe(name="loop", inputs_model=RecipeIn, outputs_model=OkOut)
    recipe.add_loop(
        "sc",
        body,
        max_iter=max_iter,
        until="converged",
        carry={"ms": "ms"},
        ms=InputRef(field="ms"),
        flag=Path("/scratch/converged.flag"),
    )
    return recipe


def test_loop_compiles_to_a_dependency_chain():
    wf = compile_slurm(_loop_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    assert [j.name for j in wf.jobs] == [
        "sc.1.work",
        "sc.1.assess",
        "sc.2.work",
        "sc.2.assess",
        "sc.3.work",
        "sc.3.assess",
    ]
    # The ordering edge onto the previous iteration's sentinel producer must
    # survive into afterok, or a later iteration could start before the
    # convergence it depends on had been decided.
    assert wf.jobs[2].depends_on == ["sc.1.work", "sc.1.assess"]


def test_loop_iterations_after_the_first_carry_a_sentinel_guard():
    wf = compile_slurm(_loop_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    assert "if [ -e /scratch/converged.flag ]; then" not in wf.jobs[0].script  # iteration 1 never skips
    for job in wf.jobs[2:]:
        assert "if [ -e /scratch/converged.flag ]; then" in job.script
        assert "  exit 0" in job.script


def test_loop_jobs_are_named_per_step_not_per_cab():
    """One cab backs six steps here; per-cab naming would point every
    iteration's --output at the same file.
    """
    wf = compile_slurm(_loop_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    names = [j.name for j in wf.jobs]
    assert len(set(names)) == len(names)
    assert "#SBATCH --job-name=sc.2.work" in wf.jobs[2].script
    assert "#SBATCH --output=/work/.shinobi/loop/sc.2.work.out" in wf.jobs[2].script


def test_identity_carried_paths_need_no_link_on_skip():
    """A path carried unchanged resolves to the same string every iteration,
    so a skipped job has nothing to materialise -- the guard is a bare exit.
    """
    wf = compile_slurm(_loop_recipe(), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    assert "ln -sfn" not in wf.jobs[2].script


class ImageIn(BaseModel):
    ms: Path
    cycle: int = 1


class ImageOut(BaseModel):
    image: Path | None = None


def test_per_cycle_output_naming_is_rejected_not_silently_wrong():
    """A body naming outputs per cycle (`index_input` feeding an `implicit`
    template) cannot be resolved statically, so it is rejected with the
    existing clear error rather than compiled into a path that no job writes.
    This is why a skipped job never has to materialise anything.
    """
    work = Cab(name="work", command="wk", inputs_model=MakeIn, outputs_model=MSOut)
    image = Cab(
        name="image",
        command="im",
        inputs_model=ImageIn,
        outputs_model=ImageOut,
        field_meta={"image": ParamMeta(implicit="/scratch/img-cycle{cycle}.fits")},
    )
    assess = Cab(name="assess", command="as", inputs_model=AssessIn, outputs_model=FlagOut)

    class CycleImgIn(BaseModel):
        ms: Path
        flag: Path
        cycle: int = 1

    class CycleImgOut(BaseModel):
        ms: Path | None = None
        image: Path | None = None
        converged: Path | None = None

    body = Recipe(name="cycle", inputs_model=CycleImgIn, outputs_model=CycleImgOut)
    body.add_step("work", work, ms=InputRef(field="ms"))
    body.add_step("image", image, ms=OutputRef(step="work", field="ms"), cycle=InputRef(field="cycle"))
    body.add_step("assess", assess, ms=OutputRef(step="work", field="ms"), flag=InputRef(field="flag"))
    body.set_output("ms", OutputRef(step="work", field="ms"))
    body.set_output("image", OutputRef(step="image", field="image"))
    body.set_output("converged", OutputRef(step="assess", field="flag"))

    recipe = Recipe(name="loop", inputs_model=RecipeIn, outputs_model=OkOut)
    recipe.add_loop(
        "sc",
        body,
        max_iter=3,
        until="converged",
        carry={"ms": "ms"},
        index_input="cycle",
        ms=InputRef(field="ms"),
        flag=Path("/scratch/f"),
    )
    recipe.add_step("pub", Cab(name="pub", command="pb", inputs_model=ImageOut, outputs_model=OkOut), image=OutputRef(step="sc.3.image", field="image"))

    with pytest.raises(OffloadCompileError, match="isn't statically known"):
        compile_slurm(recipe, {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)


def test_index_input_varies_per_iteration_locally():
    """`index_input` binds the 1-based cycle number, so a body can name its
    outputs per cycle when running locally (where `_fill_outputs` resolves
    implicit templates against real inputs).
    """
    work = Cab(name="work", command="wk", inputs_model=MakeIn, outputs_model=MSOut)
    image = Cab(name="image", command="im", inputs_model=ImageIn, outputs_model=ImageOut)
    assess = Cab(name="assess", command="as", inputs_model=AssessIn, outputs_model=FlagOut)

    class CycleImgIn(BaseModel):
        ms: Path
        flag: Path
        cycle: int = 1

    class CycleImgOut(BaseModel):
        ms: Path | None = None
        converged: Path | None = None

    body = Recipe(name="cycle", inputs_model=CycleImgIn, outputs_model=CycleImgOut)
    body.add_step("work", work, ms=InputRef(field="ms"))
    body.add_step("image", image, ms=OutputRef(step="work", field="ms"), cycle=InputRef(field="cycle"))
    body.add_step("assess", assess, ms=OutputRef(step="work", field="ms"), flag=InputRef(field="flag"))
    body.set_output("ms", OutputRef(step="work", field="ms"))
    body.set_output("converged", OutputRef(step="assess", field="flag"))

    recipe = Recipe(name="loop", inputs_model=RecipeIn, outputs_model=OkOut)
    recipe.add_loop("sc", body, max_iter=3, until="converged", carry={"ms": "ms"}, index_input="cycle", ms=InputRef(field="ms"), flag=Path("/scratch/f"))

    by_name = {ref.name: ref for ref in recipe.steps}
    assert by_name["sc.1.image"].params["cycle"] == 1
    assert by_name["sc.3.image"].params["cycle"] == 3


# -- declared resource limits --


def test_compiled_jobs_carry_per_step_resource_directives():
    """`compile_slurm` passes one workflow-global `sbatch_opts` to every job,
    so per-step allocation has to come from each step's own declaration.
    """
    make = Cab(name="make", command="mk", inputs_model=MakeIn, outputs_model=MSOut, resources=Resources(cpus=8, memory="32GiB"))
    use = Cab(name="use", command="use", inputs_model=UseIn, outputs_model=OkOut)
    wf = compile_slurm(_linear_recipe(make=make, use=use), {"ms": "/scratch/x.ms"}, workdir="/work", container_runtime=None)
    scripts = {job.name: job.script for job in wf.jobs}
    assert "#SBATCH --cpus-per-task=8" in scripts["make"]
    assert f"#SBATCH --mem={32 * 1024}M" in scripts["make"]
    # the undeclared step gets no directives of its own
    assert "--cpus-per-task" not in scripts["use"]
    assert "--mem=" not in scripts["use"]


def test_workflow_sbatch_opts_win_over_a_step_declaration():
    make = Cab(name="make", command="mk", inputs_model=MakeIn, outputs_model=MSOut, resources=Resources(memory="32GiB"))
    wf = compile_slurm(
        _linear_recipe(make=make),
        {"ms": "/scratch/x.ms"},
        workdir="/work",
        container_runtime=None,
        sbatch_opts={"mem": "100G"},
    )
    script = next(job.script for job in wf.jobs if job.name == "make")
    assert "#SBATCH --mem=100G" in script
    assert f"--mem={32 * 1024}M" not in script
