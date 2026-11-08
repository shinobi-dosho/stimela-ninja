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
    bundle = RecipeBundle.read(submission_dir / "bundle.json")
    lease = acquire_workspace(
        Path(plan.ownership_workspace or bundle.workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=submission_dir,
        accesses=((Path(access.path), access.writes) for access in plan.accesses),
    )
    write_new(marker, lease.owner)


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
