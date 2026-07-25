"""Live integration test for the Slurm offload path against a REAL Slurm
controller (sbatch + slurmdbd/sacct), the first live Slurm coverage in this
project. Skipped unless a test cluster is up -- see tests/slurm_live/README.md
for the one-time setup (`docker compose up` + a couple of env vars).

Unlike the golden tests in test_offload_slurm.py (which only inspect the
compiled scripts), this actually submits a dependency-chained workflow and
polls `sacct`, so it exercises `submit_slurm`/`status_slurm` end to end --
including that `--dependency=afterok` really gates the second step, that the
job's stdout/error dir is created (a bug the live cluster caught that golden
tests couldn't), and that a file one step writes to the shared workdir is
seen by the next. Single-node; it does not prove multi-node scheduling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.offload import compile_slurm, status_slurm, submit_slurm
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef

CONTAINER = os.environ.get("SHINOBI_SLURM_CONTAINER", "shinobi-slurm")
WORKDIR = os.environ.get("SHINOBI_SLURM_WORKDIR")
SHIM_BIN = Path(__file__).parent / "slurm_live" / "bin"
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}


def _cluster_ready() -> bool:
    if not WORKDIR or not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "exec", CONTAINER, "sinfo"], capture_output=True).returncode == 0


requires_slurm_cluster = pytest.mark.skipif(
    not _cluster_ready(),
    reason="live Slurm test cluster not running (see tests/slurm_live/README.md)",
)


class _RecipeIn(BaseModel):
    target: Path


class _TouchIn(BaseModel):
    out: Path


class _PathOut(BaseModel):
    out: Path | None = None


class _CatIn(BaseModel):
    f: Path | None = None


class _OkOut(BaseModel):
    ok: bool = True


def _touch_then_cat_recipe() -> Recipe:
    # mk writes a file at a path; use reads that same path (wired from mk's
    # output). If afterok + the shared workdir both work, use succeeds.
    mk = Cab(
        name="mk",
        command="/bin/touch",
        inputs_model=_TouchIn,
        outputs_model=_PathOut,
        field_meta={"out": ParamMeta(positional=True)},
    )
    use = Cab(
        name="use",
        command="/bin/cat",
        inputs_model=_CatIn,
        outputs_model=_OkOut,
        field_meta={"f": ParamMeta(positional=True)},
    )
    return Recipe(
        name="livepipe",
        inputs_model=_RecipeIn,
        outputs_model=_OkOut,
        steps=[
            StepRef(name="mk", step=mk, wiring={"out": InputRef(field="target")}),
            StepRef(name="use", step=use, wiring={"f": OutputRef(step="mk", field="out")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )


@requires_slurm_cluster
def test_offloaded_dependency_chain_runs_on_a_real_slurm(monkeypatch):
    monkeypatch.setenv("PATH", f"{SHIM_BIN}{os.pathsep}{os.environ['PATH']}")
    target = f"{WORKDIR}/made-{os.getpid()}.ms"
    Path(target).unlink(missing_ok=True)

    workflow = compile_slurm(_touch_then_cat_recipe(), {"target": target}, workdir=WORKDIR, container_runtime=None)
    assert [j.name for j in workflow.jobs] == ["mk", "use"]
    assert workflow.jobs[1].depends_on == ["mk"]

    job_ids = submit_slurm(workflow, workdir=WORKDIR)
    assert set(job_ids) == {"mk", "use"}

    deadline = time.time() + 90
    states = status_slurm(job_ids)
    while not all(s in _TERMINAL for s in states.values()):
        assert time.time() < deadline, f"jobs did not finish in time: {states}"
        time.sleep(2)
        states = status_slurm(job_ids)

    assert states == {"mk": "COMPLETED", "use": "COMPLETED"}, states
    # the file mk created in the shared workdir is real (afterok + shared FS)
    assert Path(target).exists()


# ---------------------------------------------------------------------------
# In-place mutation ordering, on a real controller
# ---------------------------------------------------------------------------


class _MutateIn(BaseModel):
    ms: Path


def _append_chain_recipe(scripts: dict[str, str]) -> Recipe:
    """Three steps that each append one line to the *same* file, with no
    wiring between them -- the shape of a real flag -> gaincal -> applycal
    chain rewriting one MS in place.

    Nothing here declares a dependency; the only thing the three share is
    the path, which they all declare MUTABLE. If the compiler's mutation
    ordering is wrong Slurm runs them concurrently and the appends race; if
    it is right the file ends up with exactly three lines in declaration
    order. `scripts` maps step name to a tiny shell script (written to the
    shared workdir by the test) that appends that step's name to `$1`.
    """

    def _appender(name: str) -> Cab:
        return Cab(
            name=name,
            command=scripts[name],
            inputs_model=_MutateIn,
            outputs_model=_OkOut,
            input_mutability={"ms": Mutability.MUTABLE},
            field_meta={"ms": ParamMeta(positional=True)},
        )

    return Recipe(
        name="livemutate",
        inputs_model=_MutateIn,
        outputs_model=_OkOut,
        steps=[StepRef(name=n, step=_appender(n), wiring={"ms": InputRef(field="ms")}) for n in ("flag", "gaincal", "applycal")],
    )


@requires_slurm_cluster
def test_in_place_mutation_chain_is_serialized_on_a_real_slurm(monkeypatch):
    """The claim Thread 3 rests on: steps sharing a mutated path really do
    run in order on a cluster, purely from the derived `afterok` edges.

    A golden test can only show the compiler emitted the dependency; this
    shows Slurm honoured it. Each step sleeps briefly before appending, so
    an unordered submission would interleave rather than pass by luck.
    """
    monkeypatch.setenv("PATH", f"{SHIM_BIN}{os.pathsep}{os.environ['PATH']}")
    pid = os.getpid()
    target = f"{WORKDIR}/mutated-{pid}.txt"
    Path(target).unlink(missing_ok=True)

    scripts = {}
    for name in ("flag", "gaincal", "applycal"):
        path = Path(WORKDIR) / f"append-{name}-{pid}.sh"
        path.write_text(f'#!/bin/sh\nsleep 1\necho {name} >> "$1"\n')
        path.chmod(0o755)
        scripts[name] = str(path)

    try:
        workflow = compile_slurm(_append_chain_recipe(scripts), {"ms": target}, workdir=WORKDIR, container_runtime=None)
        # derived purely from the shared mutated path -- nothing is wired
        assert [j.depends_on for j in workflow.jobs] == [[], ["flag"], ["gaincal"]]

        job_ids = submit_slurm(workflow, workdir=WORKDIR)
        deadline = time.time() + 180
        states = status_slurm(job_ids)
        while not all(s in _TERMINAL for s in states.values()):
            assert time.time() < deadline, f"jobs did not finish in time: {states}"
            time.sleep(2)
            states = status_slurm(job_ids)

        assert set(states.values()) == {"COMPLETED"}, states
        assert Path(target).read_text().split() == ["flag", "gaincal", "applycal"]
    finally:
        for path in scripts.values():
            Path(path).unlink(missing_ok=True)
        Path(target).unlink(missing_ok=True)
