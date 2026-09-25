from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi.dataset_access as access_module
import shinobi.dataset_lifecycle as lifecycle_module
from shinobi import Cab, DatasetAccess, DatasetColumns, DatasetMode, MeasurementSetV2
from shinobi.config import AppConfig
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_V1
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.offload._codec import BundleError
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.records import AttemptRecord
from shinobi.offload.slurm import prepare_worker_slurm
from shinobi.offload.worker import ExecutionPlan, SubmittedJob, execute_step, finalize_submission
from shinobi.ownership import acquire_workspace, inspect_ownership, inspect_workspace
from shinobi.snapshots import chain_id, get_journal
from shinobi.steps.schema import InputRef, Mutability, ParamMeta, Recipe, StepRef
from tests._shared_storage import qualified_storage


class Empty(BaseModel):
    pass


class StrictMS(BaseModel):
    ms: MeasurementSetV2


class StrictToolIn(BaseModel):
    script: str
    ms: MeasurementSetV2


class CreateIn(BaseModel):
    script: str
    target: Path


class CreateRoot(BaseModel):
    target: Path


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


def _prepared(workspace: Path, recipe: Recipe, root: Path):
    bundle = freeze_recipe(recipe, {"ms": root}, config=AppConfig(), workspace=workspace)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=workspace / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(workspace),
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


def _create_recipe(workspace: Path) -> Recipe:
    script = "from pathlib import Path;import sys;p=Path(sys.argv[1]);p.mkdir();(p/'table.dat').write_text('created')"
    cab = Cab(
        name="create",
        command=f"{sys.executable} -c",
        inputs_model=CreateIn,
        outputs_model=StrictMS,
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
            )
        ],
    )
    return Recipe(
        name="create-worker",
        inputs_model=CreateRoot,
        outputs_model=StrictMS,
        steps=[StepRef(name="create", step=cab, params={"script": script}, wiring={"target": InputRef(field="target")})],
        cache_dir=str(workspace / "cache"),
    )


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


def test_read_worker_commits_compute_side_lifecycle(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path, read=True), root)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = _record(workflow, plan)
        assert record.committed
        assert record.dataset_lifecycle is not None
        assert record.dataset_lifecycle.outcome == "committed"
        assert (root / "table.dat").read_text() == "raw"
    finally:
        lease.release()


def test_write_worker_uses_immutable_attempt_as_success_oracle(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path), root)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = _record(workflow, plan)
        assert record.committed
        assert record.dataset_lifecycle is not None
        assert record.dataset_lifecycle.leaves[0].outcome == "committed"
        assert (root / "table.dat").read_text() == "raw|written"
        assert get_journal(str(tmp_path / "cache")).get(chain_id(root)).marker is None
    finally:
        lease.release()


def test_create_worker_validates_and_commits_new_dataset(monkeypatch, tmp_path):
    root = tmp_path / "created.ms"
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = _create_recipe(tmp_path)
    bundle = freeze_recipe(recipe, {"target": root}, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(tmp_path),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = _record(workflow, plan)
        assert record.committed
        assert record.dataset_lifecycle is not None
        assert record.dataset_lifecycle.leaves[0].mutations[0].mode is DatasetMode.CREATE
        assert (root / "table.dat").read_text() == "created"
    finally:
        lease.release()


def test_failed_writer_is_restored_before_failed_attempt_publication(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path, fail=True), root)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 7
        record = _record(workflow, plan)
        assert record.state == "failed"
        assert record.dataset_lifecycle is not None
        assert record.dataset_lifecycle.outcome == "failed"
        assert (root / "table.dat").read_text() == "raw"
        assert get_journal(str(tmp_path / "cache")).get(chain_id(root)).marker is None
    finally:
        lease.release()


def test_dataset_worker_requires_persisted_storage_qualification(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    bundle = freeze_recipe(_recipe(tmp_path, read=True), {"ms": root}, config=AppConfig(), workspace=tmp_path)

    with pytest.raises(DatasetLifecycleUnavailableError, match="no persisted site qualification"):
        prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))


def test_compute_worker_rejects_changed_storage_qualification(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path, read=True), root)
    qualification = Path(plan.dataset_lifecycle.qualification_path)
    qualification.write_text(qualification.read_text() + "\n")
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        record = _record(workflow, plan)
        assert "qualification differs from the immutable execution plan" in record.error
        assert (root / "table.dat").read_text() == "raw"
    finally:
        lease.release()


def test_stale_invocation_is_fenced_before_strict_commit(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path), root)
    import shinobi.offload.worker as worker_module

    original = worker_module._require_current_invocation
    checks = 0

    def supersede_at_publication(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks > 1:
            raise BundleError("test invocation was superseded before publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_module, "_require_current_invocation", supersede_at_publication)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        assert not (workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").exists()
        assert (workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "publication-error.json").is_file()
        assert (root / "table.dat").read_text() == "raw"
        assert get_journal(str(tmp_path / "cache")).get(chain_id(root)).marker is None
    finally:
        lease.release()


def test_read_worker_with_generic_snapshot_publishes_after_dataset_validation(monkeypatch, tmp_path):
    class ReadOnly(BaseModel):
        script: str
        ms: MeasurementSetV2

    class WriteReport(BaseModel):
        script: str
        target: Path

    class ReportOut(BaseModel):
        artifact: Path

    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    report = tmp_path / "report.txt"
    _install_dataset(monkeypatch, tmp_path, root)
    reader = Cab(
        name="read-only",
        command=f"{sys.executable} -c",
        inputs_model=ReadOnly,
        outputs_model=Empty,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "ms": ParamMeta(positional=True),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.READ, columns=DatasetColumns(read=("DATA",)))],
    )
    writer = Cab(
        name="write-report",
        command=f"{sys.executable} -c",
        inputs_model=WriteReport,
        outputs_model=ReportOut,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "target": ParamMeta(positional=True),
            "artifact": ParamMeta(implicit="{target}"),
        },
    )
    recipe = Recipe(
        name="read-report-worker",
        inputs_model=StrictMS,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="read",
                step=reader,
                params={"script": "from pathlib import Path;import sys;Path(sys.argv[1],'table.dat').read_text()"},
                wiring={"ms": InputRef(field="ms")},
            ),
            StepRef(
                name="report",
                step=writer,
                params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('reported')", "target": report},
            ),
        ],
        cache_dir=str(tmp_path / "cache"),
    )
    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    bundle = freeze_recipe(recipe, {"ms": root}, config=config, workspace=tmp_path)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(tmp_path),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    try:
        attempt = plan.attempt("report")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
        record = AttemptRecord.read(
            workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
            workflow_id=plan.workflow_id,
            attempt_id=attempt.attempt_id,
            step_path=attempt.step_path,
            bundle_digest=bundle.digest,
        )
        assert record.committed
        assert report.read_text() == "reported"
    finally:
        lease.release()


def test_read_worker_rejects_dataset_change_before_generic_snapshot_commit(monkeypatch, tmp_path):
    class ReadReport(BaseModel):
        script: str
        ms: MeasurementSetV2
        target: Path

    class ReportOut(BaseModel):
        artifact: Path

    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    report = tmp_path / "report.txt"
    _install_dataset(monkeypatch, tmp_path, root)
    cab = Cab(
        name="read-report",
        command=f"{sys.executable} -c",
        inputs_model=ReadReport,
        outputs_model=ReportOut,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "ms": ParamMeta(positional=True),
            "target": ParamMeta(positional=True),
            "artifact": ParamMeta(implicit="{target}"),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.READ, columns=DatasetColumns(read=("DATA",)))],
    )
    recipe = Recipe(
        name="read-report-worker",
        inputs_model=StrictMS,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="read",
                step=cab,
                params={
                    "script": "from pathlib import Path;import sys;Path(sys.argv[1],'table.dat').write_text('mutated');Path(sys.argv[2]).write_text('reported')",
                    "target": report,
                },
                wiring={"ms": InputRef(field="ms")},
            )
        ],
        cache_dir=str(tmp_path / "cache"),
    )
    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    bundle = freeze_recipe(recipe, {"ms": root}, config=config, workspace=tmp_path)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(tmp_path),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    try:
        attempt = plan.attempt("read")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        record = AttemptRecord.read(
            workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
            workflow_id=plan.workflow_id,
            attempt_id=attempt.attempt_id,
            step_path=attempt.step_path,
            bundle_digest=bundle.digest,
        )
        assert record.state == "failed"
        assert record.dataset_lifecycle is not None
        assert record.dataset_lifecycle.outcome == "failed"
        assert "changed a read-only dataset" in (record.error or "")
        assert report.read_text() == "reported"
        assert get_journal(str(tmp_path / "cache")).get(chain_id(report.resolve())) is None
    finally:
        lease.release()


def test_finalizer_releases_only_after_committed_dataset_evidence(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, _lease = _prepared(tmp_path, _recipe(tmp_path, read=True), root)
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=attempt.step_path,
            attempt_id=attempt.attempt_id,
            job_id="42",
        ),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "COMPLETED"))
    before = inspect_ownership(Path(plan.ownership_workspace), str(plan.workflow_id))
    assert before.liveness == "uncertain"
    assert "dataset recovery/release evidence" in before.detail

    finalized = finalize_submission(workflow.submission_dir)

    assert finalized.complete
    assert (workflow.submission_dir / "dataset-settlement.json").is_file()
    assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None


def test_finalizer_releases_failed_writer_only_after_verified_recovery(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, _lease = _prepared(tmp_path, _recipe(tmp_path, fail=True), root)
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 7
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=attempt.step_path,
            attempt_id=attempt.attempt_id,
            job_id="42",
        ),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "FAILED"))

    finalized = finalize_submission(workflow.submission_dir)

    assert not finalized.complete
    assert (workflow.submission_dir / "finalization.json").is_file()
    assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None
    assert (root / "table.dat").read_text() == "raw"


def test_finalizer_republishes_failure_after_missing_attempt_record(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, _lease = _prepared(tmp_path, _recipe(tmp_path, fail=True), root)
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 7
    final_path = workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json"
    final_path.unlink()
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=attempt.step_path,
            attempt_id=attempt.attempt_id,
            job_id="42",
        ),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "FAILED"))

    finalized = finalize_submission(workflow.submission_dir)

    assert not finalized.complete
    recovered = _record(workflow, plan)
    assert recovered.state == "failed"
    assert recovered.dataset_lifecycle is not None
    assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None


def test_finalizer_retains_claim_when_lifecycle_evidence_is_corrupt(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path, read=True), root)
    attempt = plan.attempts[0]
    lifecycle_path = workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "dataset-lifecycle.json"
    lifecycle_path.parent.mkdir(parents=True)
    lifecycle_path.write_text("not json")
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=attempt.step_path,
            attempt_id=attempt.attempt_id,
            job_id="42",
        ),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "FAILED"))
    try:
        finalized = finalize_submission(workflow.submission_dir)
        assert not finalized.complete
        assert not (workflow.submission_dir / "finalization.json").exists()
        assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is not None
    finally:
        lease.release()
