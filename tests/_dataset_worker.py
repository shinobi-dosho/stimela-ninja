"""Shared synthetic dataset declarations and strict worker setup for tests."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

import shinobi.dataset_access as access_module
import shinobi.dataset_lifecycle as lifecycle_module
from shinobi import Cab, DatasetAccess, DatasetColumns, DatasetMode, MeasurementSetV2
from shinobi.config import AppConfig
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.datasets import MSV2_STRUCTURAL_V1, DatasetDescriptor, DatasetStatus
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.records import AttemptRecord
from shinobi.offload.slurm import prepare_worker_slurm
from shinobi.offload.worker import ExecutionPlan, SubmittedJob
from shinobi.ownership import acquire_workspace
from shinobi.steps.schema import InputRef, Mutability, ParamMeta, Recipe, StepRef
from tests._shared_storage import qualified_storage


class Empty(BaseModel):
    pass


class StrictMS(BaseModel):
    ms: MeasurementSetV2


class StrictToolIn(BaseModel):
    script: str
    ms: MeasurementSetV2


def _install_dataset(monkeypatch, workspace: Path, root: Path) -> None:
    requirements = (ClosureRequirement.TABLE_MEMBERS,)

    def closure(value, **_kwargs):
        path = Path(value).resolve()
        return DatasetClosure(
            requested_root=path,
            storage_namespace=workspace.resolve(),
            root=path,
            status=ClosureStatus.VALID,
            message="test closure",
            resources=(
                ClosureResource(
                    path=path,
                    namespace_path=path.relative_to(workspace.resolve()),
                    members=("MAIN",),
                    table_files=("table.dat",),
                    external_to_root=False,
                    storage_managers=("StandardStMan",),
                ),
            ),
            capabilities=ClosureCapabilities(
                copy_requirements=requirements,
                mount_requirements=requirements,
                stage_requirements=requirements,
                materialize_requirements=requirements,
                restore_requirements=requirements,
            ),
        )

    descriptor = DatasetDescriptor(
        path=root.resolve(),
        expected=MSV2_STRUCTURAL_V1,
        status=DatasetStatus.VALID,
        message="valid test MSv2",
        columns=("DATA",),
    )
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(workspace / "registry.json"))
    monkeypatch.setattr(access_module, "resolve_dataset_closure", closure)
    monkeypatch.setattr(lifecycle_module, "resolve_dataset_closure", closure)
    monkeypatch.setattr(lifecycle_module, "inspect_measurement_set_v2", lambda *_args, **_kwargs: descriptor)


def _recipe(workspace: Path, *, fail: bool = False, read: bool = False) -> Recipe:
    if read:
        script = "from pathlib import Path;import sys;Path(sys.argv[1],'table.dat').read_text()"
        access = DatasetAccess(field="ms", mode=DatasetMode.READ, columns=DatasetColumns(read=("DATA",)))
        mutability = {}
    else:
        suffix = "|partial" if fail else "|written"
        script = f"from pathlib import Path;import sys;p=Path(sys.argv[1],'table.dat');p.write_text(p.read_text()+{suffix!r});" + ("sys.exit(7)" if fail else "")
        access = DatasetAccess(field="ms", mode=DatasetMode.WRITE, columns=DatasetColumns(write=("DATA",)))
        mutability = {"ms": Mutability.MUTABLE}
    cab = Cab(
        name="tool",
        command=f"{sys.executable} -c",
        inputs_model=StrictToolIn,
        outputs_model=Empty if read else StrictMS,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True)},
        input_mutability=mutability,
        dataset_accesses=[access],
    )
    return Recipe(
        name="strict-worker",
        inputs_model=StrictMS,
        outputs_model=Empty if read else StrictMS,
        steps=[StepRef(name="tool", step=cab, params={"script": script}, wiring={"ms": InputRef(field="ms")})],
        output_wiring={},
        cache_dir=str(workspace / "cache"),
    )


def _prepared(workspace: Path, recipe: Recipe, root: Path, *, qualification_root: Path | None = None):
    config = AppConfig.load(config_file=workspace / "missing-config.yml")
    bundle = freeze_recipe(recipe, {"ms": root}, config=config, workspace=workspace)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=workspace / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(workspace, storage_root=qualification_root),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    assert workflow.execution_blocked_reason is None
    assert plan.dataset_lifecycle is not None
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    return workflow, plan, lease


def _record(workflow, plan) -> AttemptRecord:
    attempt = plan.attempts[0]
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    return AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )


def _register_jobs(workflow, plan, *step_paths: str) -> None:
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    (workflow.submission_dir / "jobs").mkdir(exist_ok=True)
    for index, step_path in enumerate(step_paths):
        write_new(
            workflow.submission_dir / "jobs" / f"{index:04d}.json",
            SubmittedJob(
                workflow_id=plan.workflow_id,
                bundle_digest=bundle.digest,
                step_path=step_path,
                attempt_id=plan.attempt(step_path).attempt_id,
                job_id=str(42 + index),
            ),
        )
