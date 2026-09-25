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
from shinobi.steps.schema import InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef
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


def test_local_create_admits_the_same_root_alias_as_the_worker(monkeypatch, tmp_path):
    # One containment rule for both routes: the generic target the worker
    # admits above must not be refused by strict local dispatch.
    root = tmp_path / "created.ms"
    _install_dataset(monkeypatch, tmp_path, root)
    monkeypatch.chdir(tmp_path)

    recipe = _create_recipe(tmp_path)
    recipe.output_wiring["ms"] = OutputRef(step="create", field="ms")

    result = recipe(target=root, backend="native")

    assert result.success
    assert (root / "table.dat").read_text() == "created"


class AliasIn(BaseModel):
    ms: MeasurementSetV2
    alias: Path


class CreateAliasIn(BaseModel):
    alias: Path


class CreateAliasOut(BaseModel):
    ms: MeasurementSetV2
    product: Path | None = None


def test_generic_alias_is_admitted_only_at_a_created_dataset_root(monkeypatch, tmp_path):
    from shinobi.ownership import contained_access_issues

    written = tmp_path / "data.ms"
    written.mkdir()
    created = tmp_path / "new.ms"
    _install_dataset(monkeypatch, tmp_path, written)

    def creator_issues(alias: Path, **meta) -> tuple[str, ...]:
        field_meta = {"ms": ParamMeta(implicit=str(created))}
        if meta:
            field_meta |= {"alias": ParamMeta(**meta), "product": ParamMeta(implicit="{alias}")}
        cab = Cab(
            name="creator",
            command="true",
            inputs_model=CreateAliasIn,
            outputs_model=CreateAliasOut,
            field_meta=field_meta,
            dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.CREATE, columns=DatasetColumns(create=("DATA",)), allow_schema_change=True)],
        )
        return contained_access_issues(cab, {"alias": alias}, workspace=tmp_path, dataset_resources={created.resolve()})

    writer = Cab(
        name="writer",
        command="true",
        inputs_model=AliasIn,
        outputs_model=StrictMS,
        dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.WRITE, columns=DatasetColumns(write=("DATA",)))],
    )

    def overlap(issues: tuple[str, ...]) -> bool:
        return any("overlaps the MSv2 closure" in issue for issue in issues)

    # Naming where a new dataset goes is the one admitted alias.
    assert not overlap(creator_issues(created))
    # Below the created root the alias can reach members nothing observes.
    assert overlap(creator_issues(created / "SUBTABLE"))
    # A write_path destination is cleared before launch: never the dataset.
    assert overlap(creator_issues(created, write_path=True))
    # An existing dataset the step writes is never re-reachable generically.
    assert overlap(contained_access_issues(writer, {"ms": written, "alias": written}, workspace=tmp_path, dataset_resources={written.resolve()}))


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


class TwoMS(BaseModel):
    first: MeasurementSetV2
    second: MeasurementSetV2


def _two_writers(tmp_path: Path, monkeypatch):
    roots = (tmp_path / "a.ms", tmp_path / "b.ms")
    for root in roots:
        root.mkdir()
        (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, roots[0])
    writer = _recipe(tmp_path).steps[0]
    recipe = Recipe(
        name="two-writers",
        inputs_model=TwoMS,
        outputs_model=Empty,
        steps=[
            writer.model_copy(update={"name": "A", "wiring": {"ms": InputRef(field="first")}}),
            writer.model_copy(update={"name": "B", "wiring": {"ms": InputRef(field="second")}}),
        ],
        cache_dir=str(tmp_path / "cache"),
    )
    bundle = freeze_recipe(recipe, {"first": roots[0], "second": roots[1]}, config=AppConfig(), workspace=tmp_path)
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
    return roots, workflow, plan, lease


def test_independent_writer_does_not_see_a_sibling_write_as_a_violation(monkeypatch, tmp_path):
    import shinobi.offload.worker as worker_module

    (first, second), workflow, plan, lease = _two_writers(tmp_path, monkeypatch)
    finish = worker_module._WorkerDatasetLifecycle.finish

    def sibling_commits_meanwhile(self, result):
        # B is unordered with A, so its write may land while A runs.
        (second / "table.dat").write_text("raw|written-by-B")
        return finish(self, result)

    monkeypatch.setattr(worker_module._WorkerDatasetLifecycle, "finish", sibling_commits_meanwhile)
    try:
        attempt = plan.attempt("A")
        assert execute_step(workflow.submission_dir, "A", attempt.attempt_id) == 0
        assert (first / "table.dat").read_text() == "raw|written"
    finally:
        lease.release()


def test_finalizer_leaves_a_never_started_dependant_cancelled(monkeypatch, tmp_path):
    _roots, workflow, plan, _lease = _two_writers(tmp_path, monkeypatch)
    assert execute_step(workflow.submission_dir, "A", plan.attempt("A").attempt_id) == 0
    _register_jobs(workflow, plan, "A", "B")
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"A": "COMPLETED", "B": "CANCELLED by 0"})

    finalized = finalize_submission(workflow.submission_dir)

    states = {step.step_path: step.state for step in finalized.steps}
    assert states == {"A": "succeeded", "B": "cancelled"}
    assert not (workflow.submission_dir / "attempts" / str(plan.attempt("B").attempt_id) / "final.json").exists()
    assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None


def test_refusal_before_observation_is_terminal_and_releasable(monkeypatch, tmp_path):
    import shinobi.offload.worker as worker_module

    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, _lease = _prepared(tmp_path, _recipe(tmp_path), root)
    monkeypatch.setattr(worker_module, "_planned_access_matches", lambda planned, actual: False)
    attempt = plan.attempts[0]

    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    record = _record(workflow, plan)
    assert record.state == "failed" and record.dataset_lifecycle is not None
    assert record.dataset_lifecycle.outcome == "refused" and record.dataset_lifecycle.pre_observations == ()

    _register_jobs(workflow, plan, attempt.step_path)
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "FAILED"))
    finalize_submission(workflow.submission_dir)

    assert (root / "table.dat").read_text() == "raw"
    assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None


def test_failed_record_link_after_validation_rolls_the_writer_back(monkeypatch, tmp_path):
    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path), root)
    write = AttemptRecord.write

    def no_space_for_success(self, *args, **kwargs):
        if self.state == "succeeded":
            raise OSError(28, "No space left on device")
        return write(self, *args, **kwargs)

    monkeypatch.setattr(AttemptRecord, "write", no_space_for_success)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        record = _record(workflow, plan)
        assert record.state == "failed"
        assert record.dataset_lifecycle is not None and record.dataset_lifecycle.outcome == "failed"
        assert (root / "table.dat").read_text() == "raw"
        assert get_journal(str(tmp_path / "cache")).get(chain_id(root)).marker is None
    finally:
        lease.release()


def test_read_only_change_after_precommit_is_caught_at_publication(monkeypatch, tmp_path):
    import shinobi.offload.worker as worker_module

    root = tmp_path / "data.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    workflow, plan, lease = _prepared(tmp_path, _recipe(tmp_path, read=True), root)
    finish = worker_module._WorkerDatasetLifecycle.finish

    def changed_meanwhile(self, result):
        (root / "table.dat").write_text("raw|changed after precommit")
        return finish(self, result)

    monkeypatch.setattr(worker_module._WorkerDatasetLifecycle, "finish", changed_meanwhile)
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        record = _record(workflow, plan)
        assert record.state == "failed" and record.dataset_lifecycle.outcome == "failed"
        assert "read-only dataset" in record.error
    finally:
        lease.release()
