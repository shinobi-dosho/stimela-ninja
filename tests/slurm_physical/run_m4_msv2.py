"""Run/check the real-Casacore detached MSv2 lifecycle acceptance gate.

Run only after ``run_m2.py`` has proved the exact shared mount.  Every
scientific command executes in a Slurm allocation; the driver never creates
or mutates the Measurement Set itself.
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

from shinobi import Cab, DatasetAccess, DatasetColumns, DatasetMode, MeasurementSetV2
from shinobi.config import AppConfig
from shinobi.offload.bundle import freeze_recipe
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import ExecutionPlan, Finalization
from shinobi.steps.schema import InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef


class Target(BaseModel):
    target: Path


class Empty(BaseModel):
    pass


class MSIn(BaseModel):
    ms: MeasurementSetV2


class ScriptTarget(BaseModel):
    script: str
    target: Path


class ScriptMS(BaseModel):
    script: str
    ms: MeasurementSetV2


class ReadMS(BaseModel):
    script: str
    ms: MeasurementSetV2
    report: Path


class ReadOutput(BaseModel):
    artifact: Path


CREATE = (
    "import casacore.tables as t,numpy as n,sys;"
    "p=sys.argv[1];m=t.default_ms(p);"
    "m.addcols(t.maketabdesc([t.makearrcoldesc('DATA',0j,ndim=2)]));"
    "m.addrows(4);m.putcol('SCAN_NUMBER',n.full(4,1,dtype=n.int32));m.close()"
)


def write_script(scan: int) -> str:
    return f"import casacore.tables as t,numpy as n,sys;m=t.table(sys.argv[1],readonly=False,ack=False);m.putcol('SCAN_NUMBER',n.full(m.nrows(),{scan},dtype=n.int32));m.close()"


READ = (
    "import casacore.tables as t,sys;from pathlib import Path;"
    "m=t.table(sys.argv[1],ack=False);v=','.join(str(int(x)) for x in m.getcol('SCAN_NUMBER'));m.close();"
    "Path(sys.argv[2]).write_text(v)"
)


def _create_cab(python: Path) -> Cab:
    return Cab(
        name="create-ms",
        command=f"{python} -c",
        inputs_model=ScriptTarget,
        outputs_model=MSIn,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "target": ParamMeta(positional=True),
            "ms": ParamMeta(implicit="{target}"),
        },
        dataset_accesses=[
            DatasetAccess(
                field="ms",
                mode=DatasetMode.CREATE,
                columns=DatasetColumns(create=("DATA",)),
                allow_schema_change=True,
                allow_row_count_change=True,
            )
        ],
    )


def _write_cab(python: Path) -> Cab:
    return Cab(
        name="write-ms",
        command=f"{python} -c",
        inputs_model=ScriptMS,
        outputs_model=MSIn,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True)},
        input_mutability={"ms": Mutability.MUTABLE},
        dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.WRITE, columns=DatasetColumns(write=("SCAN_NUMBER",)))],
    )


def _read_cab(python: Path) -> Cab:
    return Cab(
        name="read-ms",
        command=f"{python} -c",
        inputs_model=ReadMS,
        outputs_model=ReadOutput,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "ms": ParamMeta(positional=True),
            "report": ParamMeta(positional=True, write_path=True),
            "artifact": ParamMeta(implicit="{report}"),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.READ, columns=DatasetColumns(read=("SCAN_NUMBER",)))],
    )


def create_write_read(workspace: Path, python: Path) -> Recipe:
    return Recipe(
        name="physical-msv2",
        inputs_model=Target,
        outputs_model=Empty,
        steps=[
            StepRef(name="create", step=_create_cab(python), params={"script": CREATE}, wiring={"target": InputRef(field="target")}),
            StepRef(name="write", step=_write_cab(python), params={"script": write_script(7)}, wiring={"ms": OutputRef(step="create", field="ms")}),
            StepRef(
                name="read",
                step=_read_cab(python),
                params={"script": READ, "report": workspace / "scans.txt"},
                wiring={"ms": OutputRef(step="write", field="ms")},
            ),
        ],
        cache_dir=str(workspace / ".shinobi" / "cache"),
    )


def mutate(workspace: Path, python: Path, scan: int) -> Recipe:
    return Recipe(
        name=f"physical-msv2-write-{scan}",
        inputs_model=MSIn,
        outputs_model=MSIn,
        steps=[StepRef(name="write", step=_write_cab(python), params={"script": write_script(scan)}, wiring={"ms": InputRef(field="ms")})],
        output_wiring={"ms": OutputRef(step="write", field="ms")},
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
        sbatch_opts={"partition": args.partition},
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
    (handles / f"{label}.json").write_text(json.dumps({"submission": str(handle.submission_dir)}, indent=2))
    return handle


def _wait(submission: Path, timeout: float = 300) -> Finalization:
    path = submission / "finalization.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.5)
    return Finalization.model_validate_json(path.read_text())


def run(args) -> int:
    args.root.mkdir(parents=True, exist_ok=False)
    workspace = args.root / "workspace"
    workspace.mkdir()
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(args.root / "workspace-owners.json")
    ms = workspace / "observation.ms"
    first = _wait(
        _submit(
            args,
            "create-write-read",
            create_write_read(workspace, args.casacore_python),
            {"target": ms},
            {"create": args.nodes[0], "write": args.nodes[1], "read": args.nodes[2]},
        ).submission_dir
    )
    assert first.complete and (workspace / "scans.txt").read_text() == "7,7,7,7"

    killed = _wait(
        _submit(
            args,
            "fault-s2",
            mutate(workspace, args.casacore_python, 9),
            {"ms": ms},
            {"write": args.nodes[1]},
            fault="S2",
        ).submission_dir
    )
    assert not killed.complete
    subprocess.run([str(args.casacore_python), "-c", READ, str(ms), str(workspace / "after-fault.txt")], check=True)
    assert (workspace / "after-fault.txt").read_text() == "7,7,7,7"

    retried = _wait(_submit(args, "retry", mutate(workspace, args.casacore_python, 9), {"ms": ms}, {"write": args.nodes[2]}).submission_dir)
    assert retried.complete
    sys.stdout.write(json.dumps({"complete": True, "root": str(args.root)}) + "\n")
    return 0


def check(args) -> int:
    for label, complete in (("create-write-read", True), ("fault-s2", False), ("retry", True)):
        submission = Path(json.loads((args.root / "handles" / f"{label}.json").read_text())["submission"])
        finalized = Finalization.model_validate_json((submission / "finalization.json").read_text())
        assert finalized.complete is complete
        plan = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
        assert plan.dataset_lifecycle is not None
        assert all((submission / "attempts" / str(step.attempt_id) / "final.json").exists() for step in finalized.steps)
    report = args.root / "workspace" / "final.txt"
    subprocess.run([str(args.casacore_python), "-c", READ, str(args.root / "workspace" / "observation.ms"), str(report)], check=True)
    assert report.read_text() == "9,9,9,9"
    sys.stdout.write(json.dumps({"complete": True}) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--casacore-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--storage-qualification", type=Path, required=True)
    parser.add_argument("--nodes", nargs=3, default=("k1", "n1", "n2"))
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
