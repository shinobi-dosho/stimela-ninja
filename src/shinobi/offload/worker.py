"""Short-lived compute worker for frozen shared-storage submissions.

Each invocation executes exactly one declared step.  It never reconstructs a
recipe from mutable user source: schemas and wiring come from ``bundle.json``
and pystep code is imported only from the source snapshot staged beside it.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Literal
from uuid import UUID

from shinobi import __version__
from shinobi.config import AppConfig
from shinobi.offload._codec import BundleError, WireModel, unpack
from shinobi.offload.bundle import RecipeBundle, Submission, write_new
from shinobi.offload.code import source_tree_digest
from shinobi.offload.records import AttemptRecord
from shinobi.provenance import RunManifest, build_manifest
from shinobi.results import StepResult
from shinobi.steps.dispatch import _dispatch, _prepare_inputs
from shinobi.steps.loops import passthrough_result, should_skip
from shinobi.steps.pyfunc import _make_adapter
from shinobi.steps.schema import InputRef, OutputRef


class PlannedAttempt(WireModel):
    step_path: str
    attempt_id: UUID


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
    attempt_id: UUID
    job_id: str


class SubmittedFinalizer(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    job_id: str


class FinalizedStep(WireModel):
    step_path: str
    attempt_id: UUID
    job_id: str | None = None
    scheduler_state: str = "UNKNOWN"
    state: Literal["running", "succeeded", "failed", "cached", "skipped", "unknown"]
    record: str


class Finalization(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    complete: bool
    steps: tuple[FinalizedStep, ...]
    manifest: str | None = None


def worker_platform() -> str:
    """Stable compatibility tag that does not include the host kernel name."""
    libc, version = platform.libc_ver()
    return f"{sys.platform}-{platform.machine()}-{libc}-{version}"


def _load(submission_dir: Path) -> tuple[Submission, RecipeBundle, ExecutionPlan]:
    submission_dir = submission_dir.resolve()
    submission = Submission.model_validate_json((submission_dir / "submission.json").read_text())
    bundle = RecipeBundle.read(submission_dir / "bundle.json")
    plan = ExecutionPlan.model_validate_json((submission_dir / "execution.json").read_text())
    identity = (submission.workflow_id, submission.bundle_digest)
    if identity != (plan.workflow_id, plan.bundle_digest) or submission.bundle_digest != bundle.digest:
        raise BundleError("submission, execution plan and bundle identities disagree")
    if plan.worker.shinobi_version != __version__:
        raise BundleError(
            f"worker/bundle environment mismatch: staged worker is {plan.worker.shinobi_version}, running worker is {__version__}"
        )
    if plan.worker.python_version != platform.python_version() or plan.worker.platform != worker_platform():
        raise BundleError("running worker platform does not match the staged worker environment")
    if plan.worker.distributions_digest is not None:
        from shinobi.backends.venv import digest_of_dists, freeze_dists

        distributions = freeze_dists(Path(sys.executable))
        if distributions is None or digest_of_dists(distributions) != plan.worker.distributions_digest:
            raise BundleError("running worker dependencies do not match the staged worker environment")
    if source_tree_digest(Path(plan.worker.source)) != plan.worker.source_digest:
        raise BundleError("staged worker source no longer matches its recorded digest")
    return submission, bundle, plan


def _record_path(submission_dir: Path, attempt_id: UUID, phase: str = "final") -> Path:
    return submission_dir / "attempts" / str(attempt_id) / f"{phase}.json"


def _result_for(submission_dir: Path, bundle: RecipeBundle, plan: ExecutionPlan, step_path: str) -> StepResult:
    index = next((i for i, step in enumerate(bundle.steps) if step.name == step_path), None)
    if index is None:
        raise BundleError(f"unknown producing step {step_path!r}")
    attempt = plan.attempt(step_path)
    record = AttemptRecord.read(
        _record_path(submission_dir, attempt.attempt_id),
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
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
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    module = importlib.import_module(frozen.code.module)
    obj = module
    for part in frozen.code.qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise BundleError(f"staged pystep {frozen.code.module}.{frozen.code.qualname} is not callable")
    scope = frozen.scope.restore()
    return _make_adapter(obj, scope.outputs_model, bool(frozen.pystep_is_empty), bool(frozen.pystep_wants_ctx))


def _resolved_inputs(submission_dir: Path, bundle: RecipeBundle, plan: ExecutionPlan, index: int) -> tuple[dict, dict[str, StepResult]]:
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
    return kwargs, loaded


def execute_step(submission_dir: Path, step_path: str, attempt_id: UUID) -> int:
    """Execute one planned step and atomically publish its terminal record."""
    submission_dir = submission_dir.resolve()
    submission, bundle, plan = _load(submission_dir)
    if Path(bundle.workspace).stat().st_dev != submission_dir.stat().st_dev:
        raise BundleError("submission sandboxes and workspace are on different filesystems; node-local staging is not supported")
    planned = plan.attempt(step_path)
    if planned.attempt_id != attempt_id:
        raise BundleError(f"attempt id for {step_path!r} does not match the execution plan")
    index = next(i for i, step in enumerate(bundle.steps) if step.name == step_path)
    frozen = bundle.steps[index]
    job_id = os.environ.get("SLURM_JOB_ID")
    sandbox_root = submission_dir / "sandboxes"
    before = set(sandbox_root.iterdir()) if sandbox_root.is_dir() else set()
    common = {
        "workflow_id": submission.workflow_id,
        "attempt_id": attempt_id,
        "step_path": step_path,
        "bundle_digest": bundle.digest,
        "job_id": job_id,
        "code_digest": frozen.code.digest if frozen.code else None,
        "worker_digest": plan.worker.source_digest,
    }
    AttemptRecord(state="running", sandbox=str(sandbox_root), **common).write(submission_dir.parent)

    def retained_sandbox() -> str | None:
        if not sandbox_root.is_dir():
            return None
        candidates = [path for path in sandbox_root.iterdir() if path not in before]
        return str(max(candidates, key=lambda path: path.stat().st_mtime_ns)) if candidates else None

    try:
        old_cwd = Path.cwd()
        os.chdir(bundle.workspace)
        try:
            restored = frozen.scope.restore()
            scope = restored.model_copy(update={"backend": frozen.backend, "venv": frozen.tool_venv or restored.venv})
            func = _callable(submission_dir, index, bundle)
            kwargs, upstream = _resolved_inputs(submission_dir, bundle, plan, index)
            ref = frozen.declaration()
            if should_skip(ref, upstream):
                prepared = _prepare_inputs(scope, kwargs)
                result = passthrough_result(ref, upstream[ref.loop.prev_step], scope.inputs_model(**prepared))
            else:
                config = AppConfig.model_validate({name: unpack(value) for name, value in bundle.config.items()})
                config.cache.enabled = False
                config.cache.snapshots.mode = "off"
                config.sandbox.enabled = True
                config.sandbox.dir = str(submission_dir / "sandboxes")
                config.provenance.enabled = True
                result = _dispatch(
                    scope,
                    func,
                    backend=frozen.backend,
                    cache=False,
                    provenance=True,
                    sandbox=True,
                    stream=False,
                    _cache_path=step_path,
                    _config=config,
                    _run_id=str(submission.workflow_id),
                    **kwargs,
                )
        finally:
            os.chdir(old_cwd)
        result.sandbox_path = retained_sandbox()
        result.code_digest = common["code_digest"]
        result.worker_digest = common["worker_digest"]
        result.job_id = job_id
        terminal = AttemptRecord.from_result(result, sandbox=result.sandbox_path, **common)
        terminal.write(submission_dir.parent)
        return 0 if result.success else max(1, abs(result.returncode))
    except BaseException as exc:
        diagnostic = "".join(traceback.format_exception(exc))
        failure = AttemptRecord(state="failed", error=str(exc), stderr=diagnostic, sandbox=retained_sandbox(), **common)
        failure.write(submission_dir.parent)
        return 1


def _write_or_read(path: Path, model: WireModel):
    try:
        return write_new(path, model)
    except FileExistsError:
        existing = type(model).model_validate_json(path.read_text())
        if existing != model:
            raise BundleError(f"existing {path.name} disagrees with reconstructed finalization") from None
        return path


def finalize_submission(submission_dir: Path) -> Finalization:
    """Reconstruct ordered status and, for a complete success, a run manifest."""
    submission_dir = submission_dir.resolve()
    submission, bundle, plan = _load(submission_dir)
    from shinobi.offload.slurm import status_slurm

    jobs: dict[str, SubmittedJob] = {}
    for path in sorted((submission_dir / "jobs").glob("*.json")):
        job = SubmittedJob.model_validate_json(path.read_text())
        if (job.workflow_id, job.bundle_digest) != (submission.workflow_id, bundle.digest):
            raise BundleError(f"submitted-job record {path} has the wrong workflow identity")
        jobs[job.step_path] = job
    scheduler = status_slurm({name: job.job_id for name, job in jobs.items()}) if jobs else {}
    finalized: list[FinalizedStep] = []
    results: dict[str, StepResult] = {}
    all_committed = len(jobs) == len(bundle.steps)
    for index, frozen in enumerate(bundle.steps):
        attempt = plan.attempt(frozen.name)
        final_path = _record_path(submission_dir, attempt.attempt_id)
        unknown_path = _record_path(submission_dir, attempt.attempt_id, "unknown")
        job = jobs.get(frozen.name)
        state = scheduler.get(frozen.name, "UNKNOWN")
        if final_path.exists():
            record = AttemptRecord.read(final_path, workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id,
                                        step_path=frozen.name, bundle_digest=plan.bundle_digest)
        else:
            record = AttemptRecord(workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id,
                                   step_path=frozen.name, bundle_digest=plan.bundle_digest, state="unknown",
                                   error="worker left no committed final record", job_id=job.job_id if job else None,
                                   scheduler_state=state, code_digest=frozen.code.digest if frozen.code else None,
                                   worker_digest=plan.worker.source_digest)
            _write_or_read(unknown_path, record)
        all_committed = all_committed and record.committed
        if record.committed:
            result = record.result(frozen.scope.restore())
            result.scheduler_state = state
            result.job_id = job.job_id if job else result.job_id
            results[frozen.name] = result
        finalized.append(FinalizedStep(step_path=frozen.name, attempt_id=attempt.attempt_id,
                                       job_id=job.job_id if job else None, scheduler_state=state,
                                       state=record.state, record=str((final_path if final_path.exists() else unknown_path).relative_to(submission_dir))))

    manifest_name = None
    if all_committed:
        recipe = bundle.declaration()
        output_values = {name: getattr(results[binding.step].outputs, binding.field) for name, binding in recipe.output_wiring.items()}
        root = StepResult(name=recipe.name, returncode=0, inputs=recipe.inputs_model(**unpack(bundle.inputs)),
                          outputs=recipe.outputs_model(**output_values), kind="recipe", backend="slurm-worker",
                          sub_results={step.name: results[step.name] for step in bundle.steps})
        manifest = build_manifest(root, backend="slurm-worker")
        manifest_path = submission_dir / "manifest.json"
        if not manifest_path.exists():
            write_new(manifest_path, manifest)  # type: ignore[arg-type]
        else:
            RunManifest.model_validate_json(manifest_path.read_text())
        manifest_name = manifest_path.name
    finalization = Finalization(workflow_id=submission.workflow_id, bundle_digest=bundle.digest,
                                complete=all_committed, steps=tuple(finalized), manifest=manifest_name)
    _write_or_read(submission_dir / "finalization.json", finalization)
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
