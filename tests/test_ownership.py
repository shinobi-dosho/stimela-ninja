import sys
from pathlib import Path

import pytest
import shinobi
from click.testing import CliRunner
from pydantic import BaseModel, Field, model_validator

from shinobi.cli import main
from shinobi.config import AppConfig
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import ExecutionPlan, SubmittedJob
from shinobi.results import StepResult
from shinobi.ownership import (
    WorkspaceOwnershipError,
    WorkspaceAccess,
    WorkspaceOwner,
    WorkspaceRegistryStore,
    WorkspaceOwnershipStore,
    acquire_workspace,
    inspect_ownership,
    inspect_workspace,
    ownership_workspace,
    ownership_registry,
    reconcile_ownership,
    release_workspace,
    scope_declares_writes,
)
from shinobi.storage import SharedStorageError
from shinobi.steps.schema import Cab, InputRef, OutputRef, ParamMeta, Recipe, Scope, StepRef


class Empty(BaseModel):
    pass


class PathOut(BaseModel):
    out: Path


def test_second_workflow_is_refused_without_replacing_first(tmp_path):
    first = acquire_workspace(tmp_path, "first", kind="slurm", submission=tmp_path / "submission")
    with pytest.raises(WorkspaceOwnershipError, match="owned by slurm workflow first"):
        acquire_workspace(tmp_path, "second", kind="local")
    assert inspect_workspace(tmp_path) == first.owner


def test_independent_workspaces_can_be_owned_concurrently(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    acquire_workspace(left, "left-run", kind="slurm")
    acquire_workspace(right, "right-run", kind="slurm")
    assert inspect_workspace(left).workflow_id == "left-run"
    assert inspect_workspace(right).workflow_id == "right-run"


def test_workspace_aliases_share_one_owner(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    acquire_workspace(alias, "first", kind="slurm")
    with pytest.raises(WorkspaceOwnershipError):
        acquire_workspace(real, "second", kind="slurm")


def test_different_workdirs_and_aliases_choose_the_same_path_authority(tmp_path):
    shared = tmp_path / "shared.ms"
    shared.mkdir()
    alias = tmp_path / "alias.ms"
    alias.symlink_to(shared, target_is_directory=True)
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    left_root = ownership_workspace(left, [alias])
    right_root = ownership_workspace(right, [shared])
    assert left_root == right_root == tmp_path.resolve()
    acquire_workspace(left_root, "left-run", kind="slurm", accesses=[(alias, True)])
    with pytest.raises(WorkspaceOwnershipError, match="owned by slurm workflow left-run"):
        acquire_workspace(right_root, "right-run", kind="slurm", accesses=[(shared, True)])


def test_nested_workdirs_cannot_claim_the_same_canonical_path(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "run"
    dataset = inner / "obs.ms"
    dataset.mkdir(parents=True)
    column = dataset / "CORRECTED"
    column.write_text("data")
    outer_root = ownership_workspace(outer, [dataset])
    inner_root = ownership_workspace(inner, [column])
    assert outer_root == inner
    assert inner_root == dataset
    acquire_workspace(outer_root, "outer-run", kind="slurm", accesses=[(dataset, True)])

    with pytest.raises(WorkspaceOwnershipError, match="conflicts with slurm workflow outer-run"):
        acquire_workspace(inner_root, "inner-run", kind="slurm", accesses=[(column.resolve(), True)])
    assert inspect_workspace(inner_root) is None

    release_workspace(outer_root, "outer-run")
    acquire_workspace(inner_root, "inner-first", kind="slurm", accesses=[(column.resolve(), True)])
    with pytest.raises(WorkspaceOwnershipError, match="conflicts with slurm workflow inner-first"):
        acquire_workspace(outer_root, "outer-second", kind="slurm", accesses=[(dataset, True)])
    assert inspect_workspace(outer_root) is None


def test_cross_authority_read_write_overlap_is_refused(tmp_path):
    data = tmp_path / "data"
    scratch = tmp_path / "scratch"
    data.mkdir()
    scratch.mkdir()
    shared = data / "shared.ms"
    shared.mkdir()
    output = scratch / "product.fits"

    acquire_workspace(data, "data-writer", kind="slurm", accesses=[(shared, True)])
    with pytest.raises(WorkspaceOwnershipError, match="conflicts with slurm workflow data-writer"):
        acquire_workspace(scratch, "scratch-writer", kind="slurm", accesses=[(shared, False), (output, True)])
    assert inspect_workspace(scratch) is None


def test_offloaded_workdirs_sharing_a_dataset_compile_to_one_authority(tmp_path):
    class Ms(BaseModel):
        ms: Path

    cab = Cab(name="rewrite", command="true", inputs_model=Ms, outputs_model=Ms)
    recipe = Recipe(
        name="rewrite",
        inputs_model=Ms,
        outputs_model=Ms,
        steps=[StepRef(name="rewrite", step=cab, wiring={"ms": InputRef(field="ms")})],
        output_wiring={"ms": OutputRef(step="rewrite", field="ms")},
    )
    shared = tmp_path / "shared.ms"
    shared.mkdir()
    plans = []
    workflows = []
    for name in ("left", "right"):
        workdir = tmp_path / name
        workdir.mkdir()
        workflow = prepare_worker_slurm(
            freeze_recipe(recipe, {"ms": shared}, config=AppConfig(), workspace=workdir),
            submission_root=workdir / "submissions",
            worker_python=Path(sys.executable),
        )
        workflows.append(workflow)
        plans.append(ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text()))

    assert {plan.ownership_workspace for plan in plans} == {str(tmp_path.resolve())}
    acquire_workspace(
        tmp_path,
        str(plans[0].workflow_id),
        kind="slurm",
        submission=workflows[0].submission_dir,
        accesses=((Path(access.path), access.writes) for access in plans[0].accesses),
    )
    with pytest.raises(WorkspaceOwnershipError, match=f"workflow {plans[0].workflow_id}"):
        submit_worker_slurm(workflows[1])


def test_submission_claim_is_rolled_back_when_ownership_fails(tmp_path):
    class Ms(BaseModel):
        ms: Path

    cab = Cab(name="rewrite", command="true", inputs_model=Ms, outputs_model=Ms)
    recipe = Recipe(
        name="rewrite",
        inputs_model=Ms,
        outputs_model=Ms,
        steps=[StepRef(name="rewrite", step=cab, wiring={"ms": InputRef(field="ms")})],
        output_wiring={"ms": OutputRef(step="rewrite", field="ms")},
    )
    shared = tmp_path / "shared.ms"
    shared.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {"ms": shared}, config=AppConfig(), workspace=workdir),
        submission_root=workdir / "submissions",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    acquire_workspace(
        tmp_path,
        "other-owner",
        kind="slurm",
        submission=tmp_path / "other-submission",
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
    )
    with pytest.raises(WorkspaceOwnershipError, match="owned by slurm workflow other-owner"):
        submit_worker_slurm(workflow)
    assert not (workflow.submission_dir / "submission-claim.json").exists()


def test_local_owner_is_live_until_release(tmp_path):
    lease = acquire_workspace(tmp_path, "local-run", kind="local")
    assert inspect_ownership(tmp_path).liveness == "live"
    with pytest.raises(WorkspaceOwnershipError, match="ownership is live"):
        reconcile_ownership(tmp_path)
    lease.release()
    assert inspect_ownership(tmp_path).liveness == "free"


def test_registry_storage_failure_is_uncertain_not_an_inspection_crash(tmp_path, monkeypatch):
    lease = acquire_workspace(tmp_path, "local-run", kind="local")
    original = WorkspaceRegistryStore.owner

    def unavailable(self, workflow_id):
        raise SharedStorageError("registry unavailable")

    monkeypatch.setattr(WorkspaceRegistryStore, "owner", unavailable)
    inspection = inspect_ownership(tmp_path)
    assert inspection.liveness == "uncertain"
    assert "registry unavailable" in inspection.detail
    monkeypatch.setattr(WorkspaceRegistryStore, "owner", original)
    lease.release()


def test_root_owner_storage_failure_is_uncertain_in_api_and_cli(tmp_path, monkeypatch):
    def unavailable(self):
        raise SharedStorageError("owner unavailable")

    monkeypatch.setattr(WorkspaceOwnershipStore, "owner", unavailable)
    inspection = inspect_ownership(tmp_path)
    assert inspection.owner is None
    assert inspection.liveness == "uncertain"
    assert "owner unavailable" in inspection.detail
    with pytest.raises(WorkspaceOwnershipError, match="ownership is uncertain"):
        reconcile_ownership(tmp_path)

    runner = CliRunner()
    inspected = runner.invoke(main, ["workspace", "inspect", "--workdir", str(tmp_path)])
    assert inspected.exit_code == 0
    assert "workspace: uncertain" in inspected.output
    released = runner.invoke(
        main,
        ["workspace", "release", "--workdir", str(tmp_path), "--workflow-id", "unknown", "--force"],
    )
    assert released.exit_code == 1
    assert "ownership is uncertain" in released.output


def test_failed_registry_admission_removes_the_exact_root_claim_it_inserted(tmp_path, monkeypatch):
    def refused(self, owner):
        raise WorkspaceOwnershipError("registry refused")

    monkeypatch.setattr(WorkspaceRegistryStore, "acquire", refused)
    with pytest.raises(WorkspaceOwnershipError, match="registry refused"):
        acquire_workspace(tmp_path, "failed", kind="slurm")
    assert inspect_workspace(tmp_path) is None


def test_same_workflow_reacquire_with_different_metadata_keeps_original_claim(tmp_path):
    original = acquire_workspace(
        tmp_path,
        "same-id",
        kind="slurm",
        submission=tmp_path / "first",
        accesses=[(tmp_path / "first.ms", True)],
    )
    with pytest.raises(WorkspaceOwnershipError, match="conflicting metadata"):
        acquire_workspace(
            tmp_path,
            "same-id",
            kind="slurm",
            submission=tmp_path / "second",
            accesses=[(tmp_path / "second.ms", True)],
        )
    assert inspect_workspace(tmp_path) == original.owner


def test_explicit_reconcile_releases_a_demonstrably_dead_local_owner(tmp_path):
    store = WorkspaceOwnershipStore(tmp_path)
    owner = WorkspaceOwner(
        workflow_id="dead",
        kind="local",
        workspace=str(tmp_path.resolve()),
        acquired_at=0,
        registry=str(ownership_registry()),
    )
    store.acquire(owner)
    WorkspaceRegistryStore(ownership_registry()).acquire(owner)
    store.path.with_name("workspace-owner.dead.live").touch()
    assert inspect_ownership(tmp_path).liveness == "dead"
    assert reconcile_ownership(tmp_path).owner == owner
    assert inspect_workspace(tmp_path) is None


def test_slurm_reconcile_refuses_uncertain_and_live_then_releases_terminal(tmp_path, monkeypatch):
    class FixedPathOut(BaseModel):
        out: Path = Path("product")

    cab = Cab(name="write", command="true", inputs_model=Empty, outputs_model=FixedPathOut)
    recipe = Recipe(
        name="owned",
        inputs_model=Empty,
        outputs_model=FixedPathOut,
        steps=[StepRef(name=name, step=cab) for name in ("first", "second")],
        output_wiring={"out": OutputRef(step="first", field="out")},
    )
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
        submission_root=tmp_path / "submissions",
        worker_python=Path(sys.executable),
    )
    submission = workflow.submission_dir
    plan = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
    bundle = RecipeBundle.read(submission / "bundle.json")
    (submission / "jobs").mkdir()
    first = plan.attempts[0]
    write_new(
        submission / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=first.step_path,
            attempt_id=first.attempt_id,
            job_id="42",
        ),
    )
    acquire_workspace(
        tmp_path,
        str(plan.workflow_id),
        kind="slurm",
        submission=submission,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
    )

    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "UNKNOWN"))
    assert inspect_ownership(tmp_path).liveness == "uncertain"
    with pytest.raises(WorkspaceOwnershipError, match="ownership is uncertain"):
        reconcile_ownership(tmp_path)

    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "RUNNING"))
    assert inspect_ownership(tmp_path).liveness == "live"
    with pytest.raises(WorkspaceOwnershipError, match="ownership is live"):
        reconcile_ownership(tmp_path)

    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "CANCELLED"))
    assert inspect_ownership(tmp_path).liveness == "uncertain"
    with pytest.raises(WorkspaceOwnershipError, match="ownership is uncertain"):
        reconcile_ownership(tmp_path)

    second = plan.attempts[1]
    write_new(
        submission / "jobs" / "0001.json",
        SubmittedJob(
            workflow_id=plan.workflow_id,
            bundle_digest=bundle.digest,
            step_path=second.step_path,
            attempt_id=second.attempt_id,
            job_id="43",
        ),
    )
    released = reconcile_ownership(tmp_path)
    assert released.liveness == "dead"
    assert "every planned Slurm job" in released.detail
    assert inspect_workspace(tmp_path) is None


def test_explicit_release_requires_the_recorded_identity(tmp_path):
    acquire_workspace(tmp_path, "right-id", kind="slurm")
    with pytest.raises(WorkspaceOwnershipError, match="not wrong-id"):
        release_workspace(tmp_path, "wrong-id")
    assert inspect_workspace(tmp_path).workflow_id == "right-id"
    assert release_workspace(tmp_path, "right-id")


def test_scope_write_detection_includes_plain_pystep_outputs():
    scope = Scope(name="plain", inputs_model=Empty, outputs_model=PathOut)
    assert scope_declares_writes(scope)


def test_unresolved_output_with_known_reads_takes_a_conservative_write_claim(tmp_path, monkeypatch):
    class ReadIn(BaseModel):
        src: Path

    source = tmp_path / "input.txt"
    source.write_text("data")

    @shinobi.pystep()
    def produce(src: Path) -> PathOut:
        owner = inspect_workspace(tmp_path)
        assert owner is not None
        assert WorkspaceAccess(path=str(tmp_path.resolve()), writes=True) in owner.accesses
        return PathOut(out=tmp_path / f"{src.stem}.out")

    monkeypatch.chdir(tmp_path)
    assert produce(src=source).success


def test_immutable_pystep_rewriting_same_named_path_still_needs_ownership():
    class Ms(BaseModel):
        ms: Path

    scope = Scope(name="rewrite", inputs_model=Ms, outputs_model=Ms)
    assert scope.mutability_of("ms").value == "immutable"
    assert scope_declares_writes(scope)

    destination = Scope(
        name="create",
        inputs_model=Ms,
        outputs_model=Empty,
        field_meta={"ms": ParamMeta(write_path=True)},
    )
    assert scope_declares_writes(destination)


def test_workspace_cli_inspects_and_explicitly_releases(tmp_path):
    acquire_workspace(tmp_path, "detached", kind="slurm", submission=tmp_path / "missing")
    runner = CliRunner()
    inspected = runner.invoke(main, ["workspace", "inspect", "--workdir", str(tmp_path)])
    assert inspected.exit_code == 0
    assert "workspace: uncertain" in inspected.output
    assert "workflow: detached" in inspected.output

    mismatched = runner.invoke(
        main,
        ["workspace", "release", "--workdir", str(tmp_path), "--workflow-id", "newer", "--force"],
    )
    assert mismatched.exit_code == 1
    assert "owned by workflow detached, not newer" in mismatched.output

    released = runner.invoke(
        main,
        ["workspace", "release", "--workdir", str(tmp_path), "--workflow-id", "detached", "--force"],
    )
    assert released.exit_code == 0
    assert "workspace: released" in released.output


def test_local_dispatch_holds_and_releases_workspace_owner(tmp_path, monkeypatch):
    seen = []

    @shinobi.pystep()
    def write() -> PathOut:
        seen.append(inspect_ownership(tmp_path).liveness)
        output = tmp_path / "product"
        output.write_text("done")
        return PathOut(out=output)

    monkeypatch.chdir(tmp_path)
    assert write().success
    assert seen == ["live"]
    assert inspect_ownership(tmp_path).liveness == "free"


def test_local_dispatch_refuses_a_detached_owner_before_running(tmp_path, monkeypatch):
    acquire_workspace(tmp_path, "queued", kind="slurm", submission=tmp_path / "submission")
    called = False

    @shinobi.pystep()
    def write() -> PathOut:
        nonlocal called
        called = True
        return PathOut(out=tmp_path / "never")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkspaceOwnershipError, match="owned by slurm workflow queued"):
        write()
    assert called is False


def test_local_dispatch_owns_the_mutation_path_not_the_launch_directory(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    workdir.mkdir()
    dataset = tmp_path / "shared.ms"
    dataset.mkdir()
    called = False

    @shinobi.pystep(write_paths=["ms"])
    def rewrite(ms: Path) -> None:
        nonlocal called
        called = True
        assert inspect_workspace(tmp_path).kind == "local"

    monkeypatch.chdir(workdir)
    rewrite(ms=dataset)
    assert called is True
    assert inspect_workspace(tmp_path) is None


def test_local_leaf_claim_includes_defaulted_write_paths_and_reuses_validation(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    explicit = tmp_path / "explicit.ms"
    defaulted = tmp_path / "default.ms"
    calls = 0

    def default_target() -> Path:
        nonlocal calls
        calls += 1
        return defaulted

    class Inputs(BaseModel):
        explicit: Path
        defaulted: Path = Field(default_factory=default_target)

    scope = Scope(
        name="defaulted-writes",
        inputs_model=Inputs,
        outputs_model=Empty,
        field_meta={
            "explicit": ParamMeta(write_path=True),
            "defaulted": ParamMeta(write_path=True),
        },
    )

    def body(ctx):
        owner = inspect_workspace(tmp_path)
        assert owner is not None
        assert WorkspaceAccess(path=str(explicit.resolve()), writes=True) in owner.accesses
        assert WorkspaceAccess(path=str(defaulted.resolve()), writes=True) in owner.accesses
        return StepResult(name=scope.name, returncode=0, outputs=Empty(), inputs=ctx.inputs)

    monkeypatch.chdir(launch)
    result = StepRef(name=scope.name, step=scope, func=body)(explicit=explicit)
    assert result.success
    assert calls == 1


def test_nested_recipe_leaf_reuses_default_factory_from_claim(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    explicit = tmp_path / "explicit.ms"
    defaulted = tmp_path / "default.ms"
    calls = 0
    validations = 0

    def default_target() -> Path:
        nonlocal calls
        calls += 1
        return defaulted

    class LeafInputs(BaseModel):
        explicit: Path
        defaulted: Path = Field(default_factory=default_target)

        @model_validator(mode="after")
        def count_validation(self):
            nonlocal validations
            validations += 1
            return self

    class RecipeInputs(BaseModel):
        explicit: Path

    scope = Scope(
        name="leaf",
        inputs_model=LeafInputs,
        outputs_model=Empty,
        field_meta={
            "explicit": ParamMeta(write_path=True),
            "defaulted": ParamMeta(write_path=True),
        },
    )

    seen = {}

    def body(ctx):
        seen["owner"] = inspect_workspace(tmp_path)
        return StepResult(name=scope.name, returncode=0, outputs=Empty(), inputs=ctx.inputs)

    inner = Recipe(
        name="inner",
        inputs_model=RecipeInputs,
        outputs_model=Empty,
        steps=[StepRef(name="leaf", step=scope, func=body, wiring={"explicit": InputRef(field="explicit")})],
    )
    recipe = Recipe(
        name="nested-defaults",
        inputs_model=RecipeInputs,
        outputs_model=Empty,
        steps=[StepRef(name="inner", step=inner, wiring={"explicit": InputRef(field="explicit")})],
    )

    monkeypatch.chdir(launch)
    result = recipe(explicit=explicit)
    assert result.success
    assert calls == 1
    assert validations == 1
    assert WorkspaceAccess(path=str(explicit.resolve()), writes=True) in seen["owner"].accesses
    assert WorkspaceAccess(path=str(defaulted.resolve()), writes=True) in seen["owner"].accesses
