"""Shared direct-worker test setup, including simulated submission ownership."""

from __future__ import annotations

import sys
from pathlib import Path

from shinobi.config import AppConfig
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.slurm import prepare_worker_slurm
from shinobi.offload.worker import ExecutionPlan
from shinobi.offload.worker import execute_step as _execute_step_impl
from shinobi.ownership import acquire_workspace, inspect_workspace, release_workspace
from shinobi.steps.schema import Recipe


def _activate_worker(submission_dir: Path) -> None:
    """Give direct worker-unit invocations the ownership submit_slurm creates."""
    plan = ExecutionPlan.model_validate_json((submission_dir / "execution.json").read_text())
    marker = submission_dir / "ownership.json"
    if not plan.ownership_required or marker.exists():
        return
    acquire_worker_ownership(submission_dir, plan)


def acquire_worker_ownership(submission_dir: Path, plan: ExecutionPlan):
    """Publish the ownership claim in the frozen plan's registry."""
    assert plan.ownership_workspace is not None
    assert plan.ownership_registry is not None
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(submission_dir / "ownership.json", lease.owner)
    return lease


def prepare_qualified(workspace: Path, recipe: Recipe, inputs, *, config: AppConfig, qualification_root: Path | None = None):
    """Prepare a storage-qualified worker with explicit ownership."""
    from tests._shared_storage import qualified_storage

    bundle = freeze_recipe(recipe, inputs, config=config, workspace=workspace)
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=workspace / "runs",
        worker_python=Path(sys.executable),
        dataset_storage_qualification=qualified_storage(workspace, storage_root=qualification_root),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    assert workflow.execution_blocked_reason is None
    lease = acquire_worker_ownership(workflow.submission_dir, plan)
    return workflow, plan, lease


def execute_step(submission_dir: Path, step_path, attempt_id):
    _activate_worker(submission_dir)
    result = _execute_step_impl(submission_dir, step_path, attempt_id)
    plan = ExecutionPlan.model_validate_json((submission_dir / "execution.json").read_text())
    settled = all((submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").exists() for attempt in plan.attempts)
    workspace = Path(plan.ownership_workspace or RecipeBundle.read(submission_dir / "bundle.json").workspace)
    if plan.ownership_required and settled:
        owner = inspect_workspace(workspace)
        if owner is not None and owner.workflow_id == str(plan.workflow_id):
            release_workspace(workspace, owner.workflow_id)
    return result


def prepare_cached(tmp_path: Path, recipe: Recipe, config: AppConfig | None = None):
    config = config or AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    bundle = freeze_recipe(recipe, {}, config=config, workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    staged = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    return workflow, staged, plan
