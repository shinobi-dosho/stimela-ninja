"""Short-lived compute worker for frozen shared-storage submissions.

Each invocation executes exactly one declared step.  It never reconstructs a
recipe from mutable user source: schemas and wiring come from ``bundle.json``
and pystep code is imported only from the source snapshot staged beside it.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid5

from shinobi import __version__
from shinobi.config import AppConfig
from shinobi.cache import ExecutionIdentity, get_cache_manifest, resolve_input_keys
from shinobi.offload._codec import BundleError, WireModel, unpack
from shinobi.offload.bundle import RecipeBundle, Submission, write_new
from shinobi.offload.code import source_tree_digest
from shinobi.offload.records import AttemptRecord
from shinobi.ownership import WorkspaceAccess
from shinobi.provenance import RunManifest, build_manifest
from shinobi.results import StepResult
from shinobi.snapshots import faults, mutation_paths, reconcile
from shinobi.steps.dispatch import _dispatch, _prepare_inputs
from shinobi.steps.loops import passthrough_result, should_skip
from shinobi.steps.pyfunc import _make_adapter
from shinobi.steps.schema import InputRef, OutputRef, Scope, declared_output_dirs

logger = logging.getLogger(__name__)


class PlannedAttempt(WireModel):
    """Immutable invocation-zero identity and namespace for later retries."""

    step_path: str
    attempt_id: UUID


class AttemptInvocation(WireModel):
    """The distinct identity selected for one scheduler invocation."""

    schema_version: Literal[1] = 1
    planned_attempt_id: UUID
    attempt_id: UUID
    restart_count: int


class WorkerEnvironment(WireModel):
    python: str
    source: str
    source_digest: str
    shinobi_version: str = __version__
    python_version: str
    platform: str
    distributions_digest: str | None = None


class ExecutionPlan(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    worker: WorkerEnvironment
    attempts: tuple[PlannedAttempt, ...]
    ownership_required: bool = False
    ownership_workspace: str | None = None
    ownership_registry: str | None = None
    accesses: tuple[WorkspaceAccess, ...] = ()

    def attempt(self, step_path: str) -> PlannedAttempt:
        try:
            return next(item for item in self.attempts if item.step_path == step_path)
        except StopIteration as exc:
            raise BundleError(f"step {step_path!r} is not in the execution plan") from exc


class SubmittedJob(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    step_path: str
    # The immutable planned identity. A requeued invocation publishes its
    # derived identity below attempts/<this id>/restart-*.json.
    attempt_id: UUID
    job_id: str


class SubmissionClaim(WireModel):
    """One-shot proof that scheduler handoff has begun for this workflow."""

    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str


class SubmittedFinalizer(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    job_id: str


class WorkerHandleRecord(WireModel):
    schema_version: Literal[1] = 1
    engine: Literal["slurm-worker"] = "slurm-worker"
    recipe: str
    submission: str
    jobs: dict[str, str]
    finalizer: str | None = None


class FinalizedStep(WireModel):
    step_path: str
    attempt_id: UUID
    job_id: str | None = None
    scheduler_state: str = "UNKNOWN"
    state: Literal["running", "succeeded", "failed", "cancelled", "cached", "skipped", "unknown"]
    record: str | None = None


class Finalization(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    complete: bool
    steps: tuple[FinalizedStep, ...]
    manifest: str | None = None


def worker_platform(
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    libc: tuple[str, str] | None = None,
) -> str:
    """Stable compatibility tag that does not include the host kernel name."""
    libc_name, libc_version = libc or platform.libc_ver()
    return f"{platform_name or sys.platform}-{machine or platform.machine()}-{libc_name}-{libc_version}"


def _load_identity(submission_dir: Path) -> tuple[Submission, RecipeBundle, ExecutionPlan]:
    submission_dir = submission_dir.resolve()
    submission = Submission.model_validate_json((submission_dir / "submission.json").read_text())
    bundle = RecipeBundle.read(submission_dir / "bundle.json")
    plan = ExecutionPlan.model_validate_json((submission_dir / "execution.json").read_text())
    identity = (submission.workflow_id, submission.bundle_digest)
    if identity != (plan.workflow_id, plan.bundle_digest) or submission.bundle_digest != bundle.digest:
        raise BundleError("submission, execution plan and bundle identities disagree")
    return submission, bundle, plan


def _verify_environment(bundle: RecipeBundle, plan: ExecutionPlan) -> None:
    if plan.worker.shinobi_version != __version__:
        raise BundleError(f"worker/bundle environment mismatch: staged worker is {plan.worker.shinobi_version}, running worker is {__version__}")
    if plan.worker.python_version != platform.python_version() or plan.worker.platform != worker_platform():
        raise BundleError("running worker platform does not match the staged worker environment")
    if plan.worker.distributions_digest is not None:
        from shinobi.backends.venv import digest_of_dists, freeze_dists

        distributions = freeze_dists(Path(sys.executable))
        if distributions is None or digest_of_dists(distributions) != plan.worker.distributions_digest:
            raise BundleError("running worker dependencies do not match the staged worker environment")
    if source_tree_digest(Path(plan.worker.source)) != plan.worker.source_digest:
        raise BundleError("staged worker source no longer matches its recorded digest")
    for step in bundle.steps:
        image = unpack(step.scope.settings["image"]) if "image" in step.scope.settings else None
        if image and step.backend in ("docker", "podman", "apptainer") and step.image_digest is None:
            raise BundleError(f"step {step.name!r}: worker execution refuses an image without a submission-time pin")


def _verify_tool_environment(frozen) -> None:
    """Verify only this allocation's tool environment.

    A changed environment on one branch must not prevent an unrelated branch
    from reaching its own cache decision or running successfully.
    """
    if frozen.backend == "venv":
        from shinobi.backends.venv import inspect_venv_digest

        actual = inspect_venv_digest(Path(frozen.tool_venv))
        if frozen.tool_venv_digest is None or actual != frozen.tool_venv_digest:
            raise BundleError(f"step {frozen.name!r}: tool venv no longer matches its submission-time fingerprint")


def _load(submission_dir: Path) -> tuple[Submission, RecipeBundle, ExecutionPlan]:
    submission, bundle, plan = _load_identity(submission_dir)
    _verify_environment(bundle, plan)
    return submission, bundle, plan


def _record_path(submission_dir: Path, attempt_id: UUID, phase: str = "final") -> Path:
    return submission_dir / "attempts" / str(attempt_id) / f"{phase}.json"


def _attempt_id_for_restart(planned_attempt_id: UUID, restart_count: int) -> UUID:
    return planned_attempt_id if restart_count == 0 else uuid5(planned_attempt_id, f"slurm-restart:{restart_count}")


def _invocation_attempt(submission_dir: Path, planned: PlannedAttempt) -> UUID:
    """Return and publish the attempt identity for this Slurm invocation.

    Slurm reuses the submitted script and job id on requeue, but increments
    ``SLURM_RESTART_COUNT``.  Deriving a UUID from that durable generation
    gives every invocation a fresh success oracle without mutating the frozen
    execution plan.
    """
    raw = os.environ.get("SLURM_RESTART_COUNT", "0")
    try:
        restart_count = int(raw)
    except ValueError as exc:
        raise BundleError(f"invalid SLURM_RESTART_COUNT {raw!r}") from exc
    if restart_count < 0:
        raise BundleError(f"invalid SLURM_RESTART_COUNT {raw!r}")
    attempt_id = _attempt_id_for_restart(planned.attempt_id, restart_count)
    if restart_count:
        invocation = AttemptInvocation(planned_attempt_id=planned.attempt_id, attempt_id=attempt_id, restart_count=restart_count)
        path = submission_dir / "attempts" / str(planned.attempt_id) / f"restart-{restart_count:08d}.json"
        try:
            write_new(path, invocation)
        except FileExistsError:
            existing = AttemptInvocation.model_validate_json(path.read_text())
            if existing != invocation:
                raise BundleError(f"restart generation {restart_count} for {planned.step_path!r} has conflicting identity")
    return attempt_id


def _latest_attempt(submission_dir: Path, planned: PlannedAttempt) -> UUID:
    """Newest published invocation for a planned step."""
    latest = (0, planned.attempt_id)
    root = submission_dir / "attempts" / str(planned.attempt_id)
    for path in sorted(root.glob("restart-*.json")):
        invocation = AttemptInvocation.model_validate_json(path.read_text())
        if invocation.planned_attempt_id != planned.attempt_id:
            raise BundleError(f"attempt invocation {path} does not belong to {planned.step_path!r}")
        expected = _attempt_id_for_restart(planned.attempt_id, invocation.restart_count)
        expected_name = f"restart-{invocation.restart_count:08d}.json"
        if invocation.attempt_id != expected or path.name != expected_name or invocation.restart_count <= 0:
            raise BundleError(f"attempt invocation {path} has an invalid retry identity")
        if invocation.restart_count > latest[0]:
            latest = (invocation.restart_count, invocation.attempt_id)
    return latest[1]


def _result_for(submission_dir: Path, bundle: RecipeBundle, plan: ExecutionPlan, step_path: str) -> StepResult:
    index = next((i for i, step in enumerate(bundle.steps) if step.name == step_path), None)
    if index is None:
        raise BundleError(f"unknown producing step {step_path!r}")
    planned = plan.attempt(step_path)
    attempt_id = _latest_attempt(submission_dir, planned)
    record = AttemptRecord.read(
        _record_path(submission_dir, attempt_id),
        workflow_id=plan.workflow_id,
        attempt_id=attempt_id,
        step_path=step_path,
        bundle_digest=plan.bundle_digest,
    )
    if not record.committed:
        raise BundleError(f"upstream step {step_path!r} has no committed result")
    return record.result(bundle.steps[index].scope.restore())


def _callable(submission_dir: Path, index: int, bundle: RecipeBundle):
    frozen = bundle.steps[index]
    if frozen.code is None:
        return None
    source_root = submission_dir / "code" / str(index)
    for file in frozen.code.files:
        if (source_root / file.path).read_text(encoding="utf-8") != file.source:
            raise BundleError(f"staged pystep source {file.path!r} no longer matches the frozen bundle")
    if source_tree_digest(source_root) != frozen.code.source_digest:
        raise BundleError("staged pystep source tree contains files outside the frozen bundle")
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    module = importlib.import_module(frozen.code.module)
    obj = module
    for part in frozen.code.qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise BundleError(f"staged pystep {frozen.code.module}.{frozen.code.qualname} is not callable")
    setattr(obj, "__shinobi_staged_code_root__", str(source_root))
    scope = frozen.scope.restore()
    return _make_adapter(obj, scope.outputs_model, bool(frozen.pystep_is_empty), bool(frozen.pystep_wants_ctx))


def _resolved_inputs(submission_dir: Path, bundle: RecipeBundle, plan: ExecutionPlan, index: int) -> tuple[dict, dict[str, StepResult], dict]:
    frozen = bundle.steps[index]
    recipe_inputs = unpack(bundle.inputs)
    kwargs = dict(unpack(frozen.params))
    loaded: dict[str, StepResult] = {}

    def one(source: InputRef | OutputRef):
        if isinstance(source, InputRef):
            return recipe_inputs[source.field]
        if source.step not in loaded:
            loaded[source.step] = _result_for(submission_dir, bundle, plan, source.step)
        result = loaded[source.step]
        return getattr(result.outputs, source.field)

    ref = frozen.declaration()
    for field, source in ref.wiring.items():
        kwargs[field] = [one(item) for item in source] if isinstance(source, list) else one(source)
    if ref.loop is not None:
        for related in (ref.loop.sentinel_step, ref.loop.prev_step):
            if related and related not in loaded:
                loaded[related] = _result_for(submission_dir, bundle, plan, related)
    return kwargs, loaded, resolve_input_keys(ref, {}, loaded)


def _check_scratch_filesystems(scope: Scope, prepared: dict, workspace: Path) -> None:
    """Refuse absolute declared scratch on another filesystem in M1."""
    workspace_device = workspace.stat().st_dev
    for directory, source in declared_output_dirs(scope, prepared):
        if not source.startswith("scratch pattern") or not directory.is_absolute():
            continue
        existing = directory
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        if existing.stat().st_dev != workspace_device:
            raise BundleError(
                f"{source} resolves through {directory}, which is on a different filesystem from the shared workspace; "
                "node-local/cross-filesystem scratch is not supported by the M1 worker"
            )


def execute_step(submission_dir: Path, step_path: str, attempt_id: UUID) -> int:
    """Execute one planned step and atomically publish its terminal record."""
    submission_dir = submission_dir.resolve()
    submission, bundle, plan = _load_identity(submission_dir)
    planned = plan.attempt(step_path)
    if planned.attempt_id != attempt_id:
        raise BundleError(f"attempt id for {step_path!r} does not match the execution plan")
    attempt_id = _invocation_attempt(submission_dir, planned)
    index = next(i for i, step in enumerate(bundle.steps) if step.name == step_path)
    frozen = bundle.steps[index]
    job_id = os.environ.get("SLURM_JOB_ID")
    # Attempts never share a scratch root. Besides making provenance exact
    # under concurrent Slurm jobs, this gives later retry work a natural
    # ownership boundary without changing the sandbox implementation.
    sandbox_root = submission_dir / "sandboxes" / str(attempt_id)
    final_path = _record_path(submission_dir, attempt_id)
    common = {
        "workflow_id": submission.workflow_id,
        "attempt_id": attempt_id,
        "step_path": step_path,
        "bundle_digest": bundle.digest,
        "job_id": job_id,
        "code_digest": frozen.code.digest if frozen.code else None,
        "worker_digest": plan.worker.source_digest,
    }

    def publish_failure(exc: BaseException, *, phase: str = "final") -> None:
        diagnostic = "".join(traceback.format_exception(exc))
        failure = AttemptRecord(state="failed", error=str(exc), stderr=diagnostic, sandbox=retained_sandbox(), **common)
        path = _record_path(submission_dir, attempt_id, phase)
        try:
            write_new(path, failure)
        except FileExistsError:
            # A concurrently-published terminal record always wins. Preserve
            # the diagnostic separately when possible, never overwrite it.
            if phase == "final":
                _write_or_read(_record_path(submission_dir, attempt_id, "publication-error"), failure)

    def retained_sandbox() -> str | None:
        if not sandbox_root.is_dir():
            return None
        candidates = list(sandbox_root.iterdir())
        return str(candidates[0]) if len(candidates) == 1 else None

    def commit_result(result: StepResult, record_cache) -> None:
        """Publish the worker success oracle before updating its cache index.

        SnapshotGuard invokes this between journal commit and trash/marker
        cleanup.  Once the immutable attempt record is visible, a cache-index
        failure may cost a future rerun but cannot revoke completed scientific
        work or turn it into an ambiguous attempt.
        """
        result.sandbox_path = result.sandbox_path or retained_sandbox()
        if frozen.image_digest is not None and result.image_digest != frozen.image_digest:
            raise BundleError(f"step {step_path!r}: executed image digest {result.image_digest!r} does not match submission pin {frozen.image_digest!r}")
        if frozen.tool_venv_digest is not None and result.venv_digest != frozen.tool_venv_digest:
            raise BundleError(f"step {step_path!r}: executed tool venv digest {result.venv_digest!r} does not match submission fingerprint {frozen.tool_venv_digest!r}")
        result.code_digest = common["code_digest"]
        result.worker_digest = common["worker_digest"]
        result.job_id = job_id
        terminal = AttemptRecord.from_result(result, sandbox=result.sandbox_path, **common)
        try:
            terminal.write(submission_dir.parent)
        except BaseException:
            if not final_path.exists():
                raise
            visible = AttemptRecord.read(
                final_path,
                workflow_id=plan.workflow_id,
                attempt_id=attempt_id,
                step_path=step_path,
                bundle_digest=plan.bundle_digest,
            )
            if visible != terminal:
                raise
            # The atomic link is visible and contains the exact terminal
            # record. Avoid reporting a failed Slurm job beside that commit;
            # a later disappearance is still handled as unknown, never success.
            logger.warning("terminal record %s is visible but its directory sync could not be confirmed", final_path)
        if terminal.committed:
            faults("W_RESULT")
            try:
                record_cache()
            except BaseException:  # noqa: BLE001 -- the attempt already committed
                logger.exception("step %s committed, but its reusable cache index could not be updated", step_path)
            faults("W_CACHE")

    try:
        ownership_path = submission_dir / "ownership.json"
        if plan.ownership_required and not ownership_path.exists():
            raise BundleError("write-declaring detached workflow has no immutable workspace-ownership requirement")
        if ownership_path.exists():
            from shinobi.ownership import WorkspaceOwner, require_workspace_owner

            required = WorkspaceOwner.model_validate_json(ownership_path.read_text())
            ownership_workspace = Path(plan.ownership_workspace) if plan.ownership_workspace is not None else None
            if (
                required.workflow_id != str(plan.workflow_id)
                or ownership_workspace is None
                or Path(required.workspace) != ownership_workspace
                or plan.ownership_registry is None
                or Path(required.registry) != Path(plan.ownership_registry)
                or required.accesses != plan.accesses
            ):
                raise BundleError("workspace-ownership requirement disagrees with the execution plan")
            require_workspace_owner(ownership_workspace, str(plan.workflow_id))
        _verify_environment(bundle, plan)
        _verify_tool_environment(frozen)
        if Path(bundle.workspace).stat().st_dev != submission_dir.stat().st_dev:
            raise BundleError("submission sandboxes and workspace are on different filesystems; node-local staging is not supported")
    except BaseException as exc:
        publish_failure(exc)
        return 1

    if final_path.exists():
        logger.error("attempt %s for step %r already has a terminal record; refusing to execute it again", attempt_id, step_path)
        return 1
    running = AttemptRecord(state="running", sandbox=str(sandbox_root), **common)
    try:
        running.write(submission_dir.parent)
    except FileExistsError:
        exc = BundleError(f"attempt {attempt_id} for step {step_path!r} has already started; a requeue requires a new attempt identity")
        publish_failure(exc, phase="requeue")
        return 1

    try:
        old_cwd = Path.cwd()
        os.chdir(bundle.workspace)
        try:
            restored = frozen.scope.restore()
            scope = restored.model_copy(update={"backend": frozen.backend, "venv": frozen.tool_venv or restored.venv})
            func = _callable(submission_dir, index, bundle)
            kwargs, upstream, input_keys = _resolved_inputs(submission_dir, bundle, plan, index)
            ref = frozen.declaration()
            prepared = _prepare_inputs(scope, kwargs)
            config = AppConfig.model_validate({name: unpack(value) for name, value in bundle.config.items()})
            recipe = bundle.recipe.restore()
            cache_enabled = (
                bundle.cache_override
                if bundle.cache_override is not None
                else scope.cache
                if scope.cache is not None
                else recipe.cache
                if recipe.cache is not None
                else config.cache.enabled
            )
            cache_dir = bundle.cache_dir_override or scope.cache_dir or recipe.cache_dir or config.cache.dir
            snapshots_active = config.cache.snapshots.mode != "off" and (cache_enabled or bool(recipe.cache) or config.cache.enabled)
            if snapshots_active:
                paths = {path for values in mutation_paths(scope, prepared).values() for path in values}
                if paths:
                    # Slurm's inferred mutation dependencies ensure no live
                    # sibling can be writing these same paths. Reconcile
                    # before the loop skip decision as well: pass-through
                    # must not leave a prior interrupted rewrite unresolved.
                    reconcile(cache_dir, get_cache_manifest(cache_dir), paths=paths)
            if should_skip(ref, upstream):
                result = passthrough_result(ref, upstream[ref.loop.prev_step], scope.inputs_model(**prepared))
                commit_result(result, lambda: None)
            else:
                _check_scratch_filesystems(scope, prepared, Path(bundle.workspace))
                config.sandbox.enabled = True
                config.sandbox.dir = str(sandbox_root)
                config.provenance.enabled = True
                result = _dispatch(
                    scope,
                    func,
                    backend=frozen.backend,
                    cache=bundle.cache_override,
                    cache_dir=bundle.cache_dir_override,
                    provenance=True,
                    sandbox=True,
                    stream=False,
                    _recipe_cache=recipe.cache,
                    _recipe_cache_dir=recipe.cache_dir,
                    _cache_path=f"{recipe.name}.{step_path}",
                    _config=config,
                    _run_id=str(attempt_id),
                    _input_keys=input_keys,
                    _wired_fields=set(ref.wiring),
                    _execution_identity=ExecutionIdentity(
                        code_digest=frozen.code.execution_digest if frozen.code else None,
                        image_digest=frozen.image_digest,
                        venv=frozen.tool_venv,
                        venv_digest=frozen.tool_venv_digest,
                    ),
                    _snapshot_success_record=final_path,
                    _snapshot_success_step_path=step_path,
                    _result_commit=commit_result,
                    **kwargs,
                )
        finally:
            os.chdir(old_cwd)
        if result.success:
            try:
                sandbox_root.rmdir()
            except OSError:
                pass
        return 0 if result.success else max(1, abs(result.returncode))
    except BaseException as exc:
        if final_path.exists():
            visible = AttemptRecord.read(
                final_path,
                workflow_id=plan.workflow_id,
                attempt_id=attempt_id,
                step_path=step_path,
                bundle_digest=plan.bundle_digest,
            )
            if visible.committed:
                # The immutable result is the success oracle. A crash or
                # cleanup error after that point may leave a snapshot marker
                # for reconciliation, but must not publish a contradictory
                # failure or make Slurm discard successful dependants.
                logger.exception("step %s raised after its terminal result committed; preserving the committed result", step_path)
                return 0
        publish_failure(exc)
        return 1


def _write_or_read(
    path: Path,
    model: WireModel | RunManifest,
    *,
    compare_exclude: frozenset[str] = frozenset(),
):
    try:
        return write_new(path, model)
    except FileExistsError:
        existing = type(model).model_validate_json(path.read_text())
        if existing.model_dump(mode="json", exclude=compare_exclude) != model.model_dump(mode="json", exclude=compare_exclude):
            raise BundleError(f"existing {path.name} disagrees with reconstructed finalization") from None
        return path


def _scheduler_attempt_state(state: str) -> tuple[str, bool]:
    """Map volatile Slurm text to a typed observation and terminality."""
    normalized = (state.strip().upper().split() or ["UNKNOWN"])[0].split("+")[0]
    if normalized.startswith("CANCELLED"):
        return "cancelled", True
    if normalized in {
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }:
        return "failed", True
    if normalized == "COMPLETED":
        # Scheduler success is never a Shinobi commit.
        return "unknown", True
    if normalized in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED", "STOPPED"}:
        return "running", False
    return "unknown", False


def finalize_submission(submission_dir: Path) -> Finalization:
    """Reconstruct ordered status and, for a complete success, a run manifest."""
    submission_dir = submission_dir.resolve()
    submission, bundle, plan = _load(submission_dir)
    finalization_path = submission_dir / "finalization.json"
    if finalization_path.exists():
        existing = Finalization.model_validate_json(finalization_path.read_text())
        if (existing.workflow_id, existing.bundle_digest) != (submission.workflow_id, bundle.digest):
            raise BundleError("finalization record has the wrong workflow identity")
        if existing.manifest is not None:
            RunManifest.model_validate_json((submission_dir / existing.manifest).read_text())
        return existing
    from shinobi.offload.slurm import status_slurm

    jobs: dict[str, SubmittedJob] = {}
    for path in sorted((submission_dir / "jobs").glob("*.json")):
        job = SubmittedJob.model_validate_json(path.read_text())
        if (job.workflow_id, job.bundle_digest) != (submission.workflow_id, bundle.digest):
            raise BundleError(f"submitted-job record {path} has the wrong workflow identity")
        jobs[job.step_path] = job
    scheduler = status_slurm({name: job.job_id for name, job in jobs.items()}) if jobs else {}
    finalizer_record_path = submission_dir / "finalizer-job.json"
    terminal_context = False
    if finalizer_record_path.exists():
        finalizer_job = SubmittedFinalizer.model_validate_json(finalizer_record_path.read_text())
        if (finalizer_job.workflow_id, finalizer_job.bundle_digest) != (submission.workflow_id, bundle.digest):
            raise BundleError("submitted-finalizer record has the wrong workflow identity")
        terminal_context = os.environ.get("SLURM_JOB_ID") == finalizer_job.job_id
        if not terminal_context:
            finalizer_state = status_slurm({"__finalizer__": finalizer_job.job_id}).get("__finalizer__", "UNKNOWN")
            _outcome, terminal_context = _scheduler_attempt_state(finalizer_state)
    finalized: list[FinalizedStep] = []
    results: dict[str, StepResult] = {}
    all_committed = len(jobs) == len(bundle.steps)
    # A missing job record is not proof that sbatch never accepted the job:
    # the submitter can die between scheduler acceptance and durable publish.
    # Neither finalizer context nor terminal state for the recorded subset may
    # turn that ambiguity into permission to release workspace ownership.
    settled = len(jobs) == len(bundle.steps)
    for index, frozen in enumerate(bundle.steps):
        planned = plan.attempt(frozen.name)
        attempt_id = _latest_attempt(submission_dir, planned)
        final_path = _record_path(submission_dir, attempt_id)
        started_path = _record_path(submission_dir, attempt_id, "started")
        diagnostic_paths = [
            _record_path(submission_dir, attempt_id, "publication-error"),
            _record_path(submission_dir, attempt_id, "requeue"),
        ]
        job = jobs.get(frozen.name)
        scheduler_state = scheduler.get(frozen.name, "UNKNOWN")
        _scheduler_outcome, terminal = _scheduler_attempt_state(scheduler_state)
        if not terminal and terminal_context:
            terminal = True
        settled = settled and terminal
        diagnostic_path = next((path for path in diagnostic_paths if path.exists()), None)
        chosen_path = diagnostic_path or (final_path if final_path.exists() else None)
        if chosen_path is not None:
            record = AttemptRecord.read(chosen_path, workflow_id=plan.workflow_id, attempt_id=attempt_id, step_path=frozen.name, bundle_digest=plan.bundle_digest)
            state = record.state
            record_path: Path | None = chosen_path
            if diagnostic_path is not None:
                # A duplicate/publication diagnostic can be evidence of an
                # unrecorded concurrent invocation. Never release ownership
                # automatically from that ambiguous state.
                settled = False
        else:
            state = _scheduler_outcome
            record = None
            record_path = started_path if started_path.exists() else None
        all_committed = all_committed and record is not None and record.committed
        if record is not None and record.committed:
            result = record.result(frozen.scope.restore())
            result.scheduler_state = scheduler_state
            result.job_id = job.job_id if job else result.job_id
            results[frozen.name] = result
        finalized.append(
            FinalizedStep(
                step_path=frozen.name,
                attempt_id=attempt_id,
                job_id=job.job_id if job else None,
                scheduler_state=scheduler_state,
                state=state,
                record=str(record_path.relative_to(submission_dir)) if record_path is not None else None,
            )
        )

    manifest_name = None
    if all_committed and settled:
        recipe = bundle.declaration()
        output_values = {name: getattr(results[binding.step].outputs, binding.field) for name, binding in recipe.output_wiring.items()}
        root = StepResult(
            name=recipe.name,
            returncode=0,
            inputs=recipe.inputs_model(**unpack(bundle.inputs)),
            outputs=recipe.outputs_model(**output_values),
            kind="recipe",
            backend="slurm-worker",
            sub_results={step.name: results[step.name] for step in bundle.steps},
        )
        manifest = build_manifest(root, backend="slurm-worker")
        manifest_path = submission_dir / "manifest.json"
        # ``generated_at`` is publication time, not run identity. Independent
        # finalizers reconstruct identical run content at different instants;
        # whichever atomically publishes first supplies the canonical time.
        _write_or_read(manifest_path, manifest, compare_exclude=frozenset({"generated_at"}))
        manifest_name = manifest_path.name
    finalization = Finalization(workflow_id=submission.workflow_id, bundle_digest=bundle.digest, complete=all_committed and settled, steps=tuple(finalized), manifest=manifest_name)
    # An early status observation is deliberately not canonical: scheduler
    # state changes and an absent final record may appear moments later. Once
    # every attempt is terminal, persist exactly one stable reconstruction.
    if settled:
        from shinobi.ownership import inspect_workspace, release_workspace

        if plan.ownership_required:
            ownership_workspace = Path(plan.ownership_workspace or bundle.workspace)
            owner = inspect_workspace(ownership_workspace)
            if owner is not None and owner.workflow_id == str(plan.workflow_id):
                release_workspace(ownership_workspace, str(plan.workflow_id))
            elif owner is not None:
                logger.warning("not releasing workspace ownership held by newer workflow %s", owner.workflow_id)
        _write_or_read(finalization_path, finalization)
    return finalization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shinobi.offload.worker")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--submission", type=Path, required=True)
    run.add_argument("--step", required=True)
    run.add_argument("--attempt-id", type=UUID, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return execute_step(args.submission, args.step, args.attempt_id)
    finalized = finalize_submission(args.submission)
    return 0 if finalized.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
