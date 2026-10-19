"""Run/check the detached FITS lifecycle acceptance gate.

Run only after ``run_m2.py`` has proved the exact shared mount.  Every
scientific command executes in a Slurm allocation; the driver never creates
or mutates the FITS file itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from shinobi import Cab, FITSAccess, DatasetMode, FITSImage
from shinobi.config import AppConfig
from shinobi.offload.bundle import freeze_recipe
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import AttemptInvocation, ExecutionPlan, Finalization
from shinobi.offload.records import AttemptRecord
from shinobi.steps.schema import InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef


class Target(BaseModel):
    target: Path


class Empty(BaseModel):
    pass


class FITSIn(BaseModel):
    file: FITSImage


class ScriptTarget(BaseModel):
    script: str
    target: Path


class ScriptFITS(BaseModel):
    script: str
    file: FITSImage


class ReadFITS(BaseModel):
    script: str
    file: FITSImage
    report: Path


class ReadOutput(BaseModel):
    artifact: Path


CREATE = "from astropy.io import fits;import numpy as n,sys;h=fits.PrimaryHDU(n.full((2,2),1,dtype=n.int32));h.header['CTYPE1']='LINEAR';h.header['CTYPE2']='LINEAR';h.writeto(sys.argv[1])"


def write_script(value: int) -> str:
    # This fixture has four big-endian int32 pixels after its one header block.
    # The tool needs only Python; the worker performs the Astropy inspection.
    return (
        "from pathlib import Path;import struct,sys;"
        "p=Path(sys.argv[1]);q=p.with_suffix('.replacement');b=bytearray(p.read_bytes());"
        f"b[2880:2896]=struct.pack('>4i',*([{value}]*4));"
        "q.write_bytes(b);q.replace(p)"
    )


READ = "from astropy.io import fits;import sys;from pathlib import Path;v=','.join(str(int(x)) for x in fits.getdata(sys.argv[1]).flat);Path(sys.argv[2]).write_text(v)"


def _create_cab(python: Path) -> Cab:
    return Cab(
        name="create-fits",
        command=f"{python} -c",
        inputs_model=ScriptTarget,
        outputs_model=FITSIn,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "target": ParamMeta(positional=True),
            "file": ParamMeta(implicit="{target}"),
        },
        fits_accesses=[
            FITSAccess(
                field="file",
                mode=DatasetMode.CREATE,
            )
        ],
    )


def _write_cab(python: Path, image: Path | None = None) -> Cab:
    return Cab(
        name="write-fits",
        command="python3 -c" if image else f"{python} -c",
        image=str(image) if image else None,
        backend="apptainer" if image else "native",
        inputs_model=ScriptFITS,
        outputs_model=FITSIn,
        field_meta={"script": ParamMeta(positional_head=True), "file": ParamMeta(positional=True)},
        input_mutability={"file": Mutability.MUTABLE},
        fits_accesses=[FITSAccess(field="file", mode=DatasetMode.WRITE)],
    )


def _read_cab(python: Path) -> Cab:
    return Cab(
        name="read-fits",
        command=f"{python} -c",
        inputs_model=ReadFITS,
        outputs_model=ReadOutput,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "file": ParamMeta(positional=True),
            "report": ParamMeta(positional=True, write_path=True),
            "artifact": ParamMeta(implicit="{report}"),
        },
        fits_accesses=[FITSAccess(field="file", mode=DatasetMode.READ)],
    )


def create_write_read(workspace: Path, python: Path, image: Path | None = None) -> Recipe:
    return Recipe(
        name="physical-fits",
        inputs_model=Target,
        outputs_model=Empty,
        steps=[
            StepRef(name="create", step=_create_cab(python), params={"script": CREATE}, wiring={"target": InputRef(field="target")}),
            StepRef(name="write", step=_write_cab(python, image), params={"script": write_script(7)}, wiring={"file": OutputRef(step="create", field="file")}),
            StepRef(
                name="read",
                step=_read_cab(python),
                params={"script": READ, "report": workspace / "pixels.txt"},
                wiring={"file": OutputRef(step="write", field="file")},
            ),
        ],
        cache_dir=str(workspace / ".shinobi" / "cache"),
    )


def mutate(workspace: Path, python: Path, value: int, image: Path | None = None) -> Recipe:
    return Recipe(
        name=f"physical-fits-write-{value}",
        inputs_model=FITSIn,
        outputs_model=FITSIn,
        steps=[StepRef(name="write", step=_write_cab(python, image), params={"script": write_script(value)}, wiring={"file": InputRef(field="file")})],
        output_wiring={"file": OutputRef(step="write", field="file")},
        cache_dir=str(workspace / ".shinobi" / "cache"),
    )


def _config(workspace: Path) -> AppConfig:
    return AppConfig.model_validate({"cache": {"enabled": True, "dir": str(workspace / ".shinobi" / "cache")}})


def _submit(args, label: str, recipe: Recipe, inputs: dict, nodes: dict[str, str], fault: str | None = None):
    bundle = freeze_recipe(recipe, inputs, config=_config(args.root / "workspace"), workspace=args.root / "workspace")
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=args.root / "submissions",
        worker_python=args.worker_python,
        dataset_storage_qualification=args.storage_qualification,
        sbatch_opts={"partition": args.partition, **getattr(args, "sbatch_opts", {})},
        step_sbatch_opts={name: {"nodelist": node} for name, node in nodes.items()},
    )
    if fault is not None:
        hook = args.root / f"fault-{fault}"
        hook.mkdir()
        marker = hook / "armed"
        marker.write_text("1")
        (hook / "sitecustomize.py").write_text(
            "import os\nfrom pathlib import Path\n"
            f"p=Path({str(marker)!r})\n"
            "from shinobi.snapshots import faults\n"
            "def stop():\n"
            " try: p.unlink()\n"
            " except FileNotFoundError: return\n"
            " os._exit(86)\n"
            f"faults.hooks[{fault!r}]=stop\n"
        )
        plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
        job = next(item for item in workflow.jobs if item.name == "write")
        job.script = job.script.replace(f"PYTHONPATH={plan.worker.source}", f"PYTHONPATH={hook}:{plan.worker.source}", 1)
    handle = submit_worker_slurm(workflow)
    handles = args.root / "handles"
    handles.mkdir(exist_ok=True)
    (handles / f"{label}.json").write_text(json.dumps({"submission": str(handle.submission_dir), "jobs": handle.jobs, "finalizer": handle.finalizer_job, "nodes": nodes}, indent=2))
    return handle


def _wait(submission: Path, timeout: float = 300) -> Finalization:
    path = submission / "finalization.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.5)
    return Finalization.model_validate_json(path.read_text())


def submit_phase(args) -> int:
    """Submit from a short-lived process; no driver stays alive for the jobs."""
    workspace = args.root / "workspace"
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(args.root / "workspace-owners.json")
    file = workspace / "image.fits"
    if args.phase == "create-write-read":
        _submit(
            args, args.phase, create_write_read(workspace, args.fits_python, args.image), {"target": file}, {"create": args.nodes[0], "write": args.nodes[1], "read": args.nodes[2]}
        )
    elif args.phase == "requeue":
        recipe = mutate(workspace, args.fits_python, 15, args.image)
        ready = workspace / "requeue-ready"
        script = (
            "from pathlib import Path;import struct,sys,time\n"
            f"ready=Path({str(ready)!r})\n"
            "p=Path(sys.argv[1])\n"
            "if not ready.exists():\n"
            f" {write_script(42)}\n"
            " ready.write_text('partial')\n"
            " time.sleep(180)\n"
            "assert struct.unpack('>4i',p.read_bytes()[2880:2896]) == (11,11,11,11), 'restart did not restore predecessor'\n"
            f"{write_script(15)}\n"
        )
        recipe.steps[0].params["script"] = script
        _submit(args, args.phase, recipe, {"file": file}, {"write": args.nodes[1]})
    else:
        value = 11 if args.phase == "fault-result" else 9
        fault = {"fault-s2": "S2", "fault-result": "W_RESULT"}.get(args.phase)
        _submit(args, args.phase, mutate(workspace, args.fits_python, value, args.image), {"file": file}, {"write": args.nodes[1] if fault else args.nodes[2]}, fault=fault)
    return 0


def _launch(args, phase: str) -> Finalization:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "submit-phase",
            "--phase",
            phase,
            "--root",
            str(args.root),
            "--worker-python",
            str(args.worker_python),
            "--fits-python",
            str(args.fits_python),
            "--image",
            str(args.image),
            "--storage-qualification",
            str(args.storage_qualification),
            "--partition",
            args.partition,
            "--nodes",
            *args.nodes,
        ],
        check=True,
    )
    handle = json.loads((args.root / "handles" / f"{phase}.json").read_text())
    if phase == "requeue":
        ready = args.root / "workspace" / "requeue-ready"
        deadline = time.monotonic() + 120
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("writer did not reach the requeue barrier")
            time.sleep(0.25)
        subprocess.run(["scontrol", "requeue", handle["jobs"]["write"]], check=True)
    return _wait(Path(handle["submission"]))


def run(args) -> int:
    args.root.mkdir(parents=True, exist_ok=False)
    workspace = args.root / "workspace"
    workspace.mkdir()
    file = workspace / "image.fits"
    first = _launch(args, "create-write-read")
    assert first.complete and (workspace / "pixels.txt").read_text() == "7,7,7,7"
    predecessor = file.read_bytes()
    killed = _launch(args, "fault-s2")
    assert not killed.complete
    assert file.read_bytes() == predecessor, "rollback changed predecessor bytes"
    retried = _launch(args, "retry")
    assert retried.complete
    committed = _launch(args, "fault-result")
    assert committed.complete, "published immutable result must survive worker exit"
    assert _launch(args, "requeue").complete
    sys.stdout.write(json.dumps({"complete": True, "root": str(args.root)}) + "\n")
    return 0


def check(args) -> int:
    for label, complete in (("create-write-read", True), ("fault-s2", False), ("retry", True), ("fault-result", True), ("requeue", True)):
        handle = json.loads((args.root / "handles" / f"{label}.json").read_text())
        submission = Path(handle["submission"])
        finalized = Finalization.model_validate_json((submission / "finalization.json").read_text())
        assert finalized.complete is complete
        plan = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
        assert plan.fits_lifecycle is not None
        if label == "requeue":
            planned = plan.attempt("write")
            invocation = AttemptInvocation.model_validate_json((submission / "attempts" / str(planned.attempt_id) / "restart-00000001.json").read_text())
            assert invocation.restart_count == 1 and invocation.attempt_id == finalized.steps[0].attempt_id
            assert invocation.attempt_id != planned.attempt_id
        for step in finalized.steps:
            record = AttemptRecord.model_validate_json((submission / "attempts" / str(step.attempt_id) / "final.json").read_text())
            assert record.schema_version == 5 and record.fits_lifecycle is not None
            assert record.workflow_id == plan.workflow_id and record.bundle_digest == plan.bundle_digest
            assert record.attempt_id == step.attempt_id and record.step_path == step.step_path
            assert record.fits_lifecycle.outcome == ("committed" if complete else "failed")
            placement = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", handle["jobs"][step.step_path], "--format=NodeList"], text=True).strip()
            assert placement == handle["nodes"][step.step_path], placement
            if complete and step.step_path == "write":
                assert record.observation.containerized and record.observation.image_digest
    report = args.root / "workspace" / "final.txt"
    subprocess.run([str(args.fits_python), "-c", READ, str(args.root / "workspace" / "image.fits"), str(report)], check=True)
    assert report.read_text() == "15,15,15,15"
    sys.stdout.write(json.dumps({"complete": True}) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "check", "submit-phase"))
    parser.add_argument("--phase", choices=("create-write-read", "fault-s2", "retry", "fault-result", "requeue"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--fits-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--image", type=Path, default=Path("/data/images/python-3.12-alpine.sif"))
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--storage-qualification", type=Path, required=True)
    parser.add_argument("--nodes", nargs=3, default=("k1", "n1", "n2"))
    args = parser.parse_args()
    return globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    sys.exit(main())
