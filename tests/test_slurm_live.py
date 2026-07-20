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
from shinobi.steps.schema import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef

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
