"""Short-lived compute worker for frozen shared-storage submissions.

Each invocation executes exactly one declared step.  It never reconstructs a
recipe from mutable user source: schemas and wiring come from ``bundle.json``
and pystep code is imported only from the source snapshot staged beside it.
Strict dataset leaves additionally re-resolve the frozen shared-storage plan,
observe and recover on the compute node, and embed terminal lifecycle evidence
in the immutable attempt record before ownership can be released.
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

from pydantic import model_validator

from shinobi import __version__
from shinobi.cache import ExecutionIdentity, get_cache_manifest, resolve_input_keys
from shinobi.config import AppConfig
from shinobi.dataset_access import ResolvedDatasetAccess
from shinobi.dataset_backends import (
    DatasetBackendCapability,
    DatasetBackendStatus,
    SharedStorageQualification,
    load_shared_storage_qualification,
    worker_dataset_backend_capability,
)
from shinobi.dataset_lifecycle import DatasetLifecycleSnapshot
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


class DatasetStepPlan(WireModel):
    """Frozen dataset accesses assigned to one declared worker leaf."""

    step_path: str
    accesses: tuple[ResolvedDatasetAccess, ...]


class DatasetWorkerPlan(WireModel):
    """Compute-side lifecycle contract for one shared-storage workflow."""

    schema_version: Literal[1] = 1
    storage_namespace: str
    qualification: SharedStorageQualification
    qualification_path: str
    qualification_digest: str
    mutation: bool
    capability: DatasetBackendCapability
    tool_capabilities: tuple[DatasetBackendCapability, ...]
    initial_snapshot: DatasetLifecycleSnapshot
    steps: tuple[DatasetStepPlan, ...]
    cache_dir: str

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetWorkerPlan":
        namespace = Path(self.storage_namespace)
        if not namespace.is_absolute():
            raise BundleError("a dataset worker plan needs an absolute shared-storage namespace")
        root = self.qualification.storage_root.resolve()
        qualification_path = Path(self.qualification_path)
        cache_dir = Path(self.cache_dir)
        for label, path in (
            ("storage namespace", namespace),
            ("qualification file", qualification_path),
            ("cache directory", cache_dir),
        ):
            if not path.is_absolute() or (path.resolve() != root and not path.resolve().is_relative_to(root)):
                raise BundleError(f"dataset worker {label} {path} is outside qualified shared-storage root {root}")
        if len(self.qualification_digest) != 64 or any(character not in "0123456789abcdef" for character in self.qualification_digest):
            raise BundleError("dataset worker qualification digest is not a lowercase sha256")
        expected = "contained-native-msv2-mutation/v1" if self.mutation else "contained-native-msv2-read/v1"
        expected_capability = worker_dataset_backend_capability(mutation=self.mutation, qualification=self.qualification)
        if self.capability != expected_capability or self.capability.lifecycle != expected:
            raise BundleError("dataset worker plan has no tested compute-side lifecycle capability")
        if any(capability.lifecycle != expected or capability.status is not DatasetBackendStatus.TESTED for capability in self.tool_capabilities):
            raise BundleError("dataset worker plan contains an untested tool-backend route")
        names = [step.step_path for step in self.steps]
        if len(names) != len(set(names)):
            raise BundleError("dataset worker plan repeats a logical step")
        planned = tuple(access for step in self.steps for access in step.accesses)
        if planned != self.initial_snapshot.accesses:
            raise BundleError("per-step dataset accesses disagree with the frozen workflow snapshot")
        outside = sorted(
            {
                path.resolve()
                for access in planned
                for path in ((access.root,) if access.root is not None else ()) + access.resources
                if path.resolve() != root and not path.resolve().is_relative_to(root)
            },
            key=str,
        )
        if outside:
            raise BundleError("dataset worker plan contains paths outside qualified shared storage: " + ", ".join(map(str, outside)))
        return self

    def step(self, step_path: str) -> DatasetStepPlan:
        try:
            return next(item for item in self.steps if item.step_path == step_path)
        except StopIteration as exc:
            raise BundleError(f"step {step_path!r} is not in the dataset worker plan") from exc


class ExecutionPlan(WireModel):
    schema_version: Literal[1, 2] = 1
    workflow_id: UUID
    bundle_digest: str
    worker: WorkerEnvironment
    attempts: tuple[PlannedAttempt, ...]
    ownership_required: bool = False
    ownership_workspace: str | None = None
    ownership_registry: str | None = None
    accesses: tuple[WorkspaceAccess, ...] = ()
    execution_blocked_reason: str | None = None
    dataset_lifecycle: DatasetWorkerPlan | None = None

    @model_validator(mode="after")
    def _dataset_contract(self) -> "ExecutionPlan":
        if (self.schema_version == 1) != (self.dataset_lifecycle is None):
            raise BundleError("dataset worker lifecycle requires execution-plan schema version 2")
        if self.dataset_lifecycle is not None:
            if self.execution_blocked_reason is not None:
                raise BundleError("an executable dataset worker plan cannot also be marked blocked")
            if not self.ownership_required:
                raise BundleError("dataset worker lifecycle requires durable workflow ownership")
        return self

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


class DatasetSettlement(WireModel):
    """Durable authority for releasing a terminal dataset workflow claim."""

    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    attempts: tuple[UUID, ...]


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
    if plan.execution_blocked_reason is not None:
        raise BundleError(plan.execution_blocked_reason)
    _verify_environment(bundle, plan)
    return submission, bundle, plan


def _record_path(submission_dir: Path, attempt_id: UUID, phase: str = "final") -> Path:
    return submission_dir / "attempts" / str(attempt_id) / f"{phase}.json"


def _dataset_record_path(submission_dir: Path, attempt_id: UUID) -> Path:
    return submission_dir / "attempts" / str(attempt_id) / "dataset-lifecycle.json"


def _require_exact_owner(submission_dir: Path, plan: ExecutionPlan):
    """Return the exact live owner frozen for this submission."""

    ownership_path = submission_dir / "ownership.json"
    if plan.ownership_required and not ownership_path.exists():
        raise BundleError("write- or dataset-declaring detached workflow has no immutable workspace-ownership requirement")
    if not ownership_path.exists():
        return None
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
    live = require_workspace_owner(ownership_workspace, str(plan.workflow_id))
    if live != required:
        raise BundleError("live workspace owner differs from the immutable submission claim")
    return live


def _planned_access_matches(planned: ResolvedDatasetAccess, actual: ResolvedDatasetAccess) -> bool:
    """Allow a provisional CREATE identity to become its real closure."""

    if planned == actual:
        return True
    if planned.closure_status is not None or actual.closure_status is None:
        return False
    stable = (
        "field",
        "declaration",
        "requested_path",
        "root",
        "mode",
        "whole_dataset",
        "path_known",
        "columns_known",
        "fallback",
    )
    if any(getattr(planned, name) != getattr(actual, name) for name in stable):
        return False
    root = actual.root
    return root is not None and all(resource == root or resource.is_relative_to(root) for resource in actual.resources)


def _require_current_invocation(submission_dir: Path, plan: ExecutionPlan, planned: PlannedAttempt, attempt_id: UUID) -> None:
    if _latest_attempt(submission_dir, planned) != attempt_id:
        raise BundleError(f"attempt {attempt_id} for {planned.step_path!r} was superseded by a newer scheduler invocation")


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


class _WorkerDatasetLifecycle:
    """Compute-side controller around one frozen worker leaf.

    The submission plan supplies identities and the durable workflow claim;
    observations and recovery are always repeated by the allocation that is
    about to execute the tool.  Construction only opens the lifecycle record;
    `establish` does the checks, so a refusal there is still recorded as
    terminal evidence by `fail`.
    """

    def __init__(
        self,
        *,
        submission_dir: Path,
        plan: ExecutionPlan,
        attempt_id: UUID,
        step_path: str,
        backend: str,
    ) -> None:
        from shinobi.dataset_lifecycle import DatasetLifecycle

        contract = plan.dataset_lifecycle
        assert contract is not None
        qualification, digest = load_shared_storage_qualification(Path(contract.qualification_path))
        if qualification != contract.qualification or digest != contract.qualification_digest:
            raise BundleError("compute-node shared-storage qualification differs from the immutable execution plan")
        qualified_root = qualification.storage_root.resolve()
        if submission_dir != qualified_root and not submission_dir.is_relative_to(qualified_root):
            raise BundleError(f"submission directory {submission_dir} is outside qualified shared-storage root {qualified_root}")
        self.contract = contract
        self.workspace = Path(contract.storage_namespace)
        self.cache_dir = contract.cache_dir
        self.step_path = step_path
        self.backend = backend
        self.expected = contract.step(step_path).accesses
        # Observe this leaf's own roots and those no leaf in the workflow
        # writes.  A root only a *sibling* writes may legitimately change
        # while this leaf runs (independent writers are not ordered), and the
        # sibling's own lifecycle is what vouches for it.
        workflow_written = {access.root for access in contract.initial_snapshot.accesses if access.writes and access.root is not None}
        own = {access.root for access in self.expected if access.root is not None}
        every = {access.root for access in contract.initial_snapshot.accesses if access.root is not None}
        self.roots = tuple(sorted(own | (every - workflow_written), key=str))
        self.written_roots: set[Path] = set()
        self.lifecycle = DatasetLifecycle.start(
            workspace=self.workspace,
            attempt_id=str(attempt_id),
            scope=step_path,
            backends=("slurm-worker", backend),
            mutation=contract.mutation,
            store_path=_dataset_record_path(submission_dir, attempt_id),
        )

    def establish(self, *, scope: Scope, prepared: dict, owner) -> None:
        from shinobi.dataset_backends import dataset_backend_capability
        from shinobi.dataset_lifecycle import (
            DatasetLifecyclePhase,
            DatasetLifecycleSnapshot,
            claim_covers_accesses,
            pending_dataset_recovery,
            resolve_lifecycle_snapshot,
        )

        contract = self.contract
        step_path = self.step_path
        actual_tool = dataset_backend_capability(self.backend, mutation=contract.mutation)
        expected_tool = contract.tool_capabilities[next(i for i, item in enumerate(contract.steps) if item.step_path == step_path)]
        if actual_tool != expected_tool:
            raise BundleError(f"step {step_path!r}: compute-node tool-backend capability differs from the frozen submission plan")

        values = scope.inputs_model(**prepared)
        self.leaf_snapshot = (
            resolve_lifecycle_snapshot(scope, values, workspace=self.workspace, mutation=contract.mutation)
            if self.expected
            else DatasetLifecycleSnapshot(accesses=(), observations=())
        )
        if len(self.expected) != len(self.leaf_snapshot.accesses) or any(
            not _planned_access_matches(expected, actual) for expected, actual in zip(self.expected, self.leaf_snapshot.accesses)
        ):
            raise BundleError(f"step {step_path!r}: compute-node dataset resolution differs from the frozen access plan")
        uncovered = claim_covers_accesses(owner, self.leaf_snapshot.accesses)
        if uncovered:
            raise BundleError(f"step {step_path!r}: live workflow claim does not cover dataset resources: {', '.join(map(str, uncovered))}")

        self.baseline = self._observe()
        self.lifecycle.transition(
            DatasetLifecyclePhase.PLANNED,
            "compute worker re-resolved the frozen shared-storage dataset plan",
            backends=("slurm-worker", self.backend),
            backend_capabilities=(contract.capability, actual_tool),
            capability_supported=True,
            planned_accesses=contract.initial_snapshot.accesses,
            pre_observations=self.baseline.observations,
            absent_roots=self.baseline.absent_roots if contract.mutation else (),
        )
        self.lifecycle.transition(
            DatasetLifecyclePhase.CLAIMED,
            "verified the exact live detached workflow claim and registry entry",
            claim=owner,
        )

        # Re-resolve after reading the claim.  The first observation never
        # authorizes execution by itself.
        second_leaf = resolve_lifecycle_snapshot(scope, values, workspace=self.workspace, mutation=contract.mutation) if self.expected else self.leaf_snapshot
        if second_leaf != self.leaf_snapshot or self._observe() != self.baseline:
            raise BundleError(f"step {step_path!r}: dataset plan or observation changed while establishing the compute-side lifecycle")

        self.written_roots = {access.root for access in self.leaf_snapshot.accesses if access.writes and access.root is not None}
        recovery_roots = {
            access.root
            for access in self.leaf_snapshot.accesses
            if access.root is not None and any(candidate.root == access.root and candidate.writes for candidate in contract.initial_snapshot.accesses)
        }
        notes = reconcile(self.cache_dir, get_cache_manifest(self.cache_dir), paths=recovery_roots, exact=True) if contract.mutation and recovery_roots else []
        if notes:
            self.baseline = self._observe()
            self.lifecycle.amend(
                recovery=(*self.lifecycle.record.recovery, *notes),
                pre_observations=self.baseline.observations,
                absent_roots=self.baseline.absent_roots,
            )
        pending = pending_dataset_recovery(self.cache_dir, self.leaf_snapshot)
        if pending:
            raise BundleError(f"step {step_path!r}: unresolved dataset mutation recovery remains: {', '.join(map(str, pending))}")
        self.lifecycle.transition(
            DatasetLifecyclePhase.REVALIDATED,
            "compute-side access plan and observations match under the detached claim",
        )
        self.lifecycle.transition(
            DatasetLifecyclePhase.EXECUTING,
            "entering detached worker dispatch under the durable workflow claim",
        )

    @property
    def evidence(self):
        return self.lifecycle.record

    def _observe(self):
        from shinobi.dataset_lifecycle import DatasetLifecycleSnapshot, observe_roots

        observations, absent = observe_roots(self.roots, self.workspace)
        return DatasetLifecycleSnapshot(
            accesses=self.contract.initial_snapshot.accesses,
            observations=observations,
            absent_roots=absent,
        )

    def _read_only_changed(self, post) -> tuple[Path, ...]:
        written = self.written_roots
        before = {item.root: item for item in self.baseline.observations}
        after = {item.root: item for item in post.observations}
        return tuple(
            root for root in self.roots if root not in written and (before.get(root) != after.get(root) or (root in self.baseline.absent_roots) != (root in post.absent_roots))
        )

    def validate(self):
        """Observe now and refuse any change to a root this leaf only reads.

        Called before S1 (so a violating writer is rolled back by its guard
        before anything is snapshotted or committed) and again at
        publication, since the dataset is not frozen in between.
        """
        from shinobi.exceptions import DatasetLifecycleViolationError

        post = self._observe()
        changed = self._read_only_changed(post)
        if changed:
            raise DatasetLifecycleViolationError("detached MSv2 worker changed a read-only dataset: " + ", ".join(map(str, changed)))
        return post

    def finish(self, result: StepResult):
        """Validate a successful leaf; return its unpublished COMMITTED record.

        The caller embeds that record in the immutable attempt record and
        adopts it only once the attempt record is linked.  Until then the
        persisted phase stays VALIDATED, so a failed link is still a failure
        `fail` recovers from.  A failed leaf is recovered here and returns
        ``None``.
        """
        from shinobi.dataset_lifecycle import DatasetLifecyclePhase

        if not result.success:
            self._recover_failure(f"worker returned non-zero status {result.returncode}")
            return None
        post = self.validate()
        if self.lifecycle.record.phase is DatasetLifecyclePhase.EXECUTING:
            self.lifecycle.transition(
                DatasetLifecyclePhase.VALIDATED,
                "compute-side declared postconditions hold",
                post_observations=post.observations,
            )
        elif self.lifecycle.record.phase is not DatasetLifecyclePhase.VALIDATED:
            raise BundleError(
                f"step {self.step_path!r}: detached dataset lifecycle is {self.lifecycle.record.phase.value!r}, expected 'executing' or 'validated' before publication"
            )
        return self.lifecycle.preview(DatasetLifecyclePhase.COMMITTED, "immutable detached attempt is ready for publication")

    def _recover_failure(self, reason: str) -> None:
        from shinobi.dataset_lifecycle import DatasetLifecyclePhase

        notes = reconcile(self.cache_dir, get_cache_manifest(self.cache_dir), paths=self.written_roots, exact=True) if self.contract.mutation and self.written_roots else []
        post = self._observe()
        changed = self._read_only_changed(post)
        detail = reason
        if notes:
            detail += "; recovery: " + "; ".join(notes)
        if changed:
            detail += "; unrecovered read-only change: " + ", ".join(map(str, changed))
        self.lifecycle.transition(
            DatasetLifecyclePhase.FAILED,
            detail,
            post_observations=post.observations,
            recovery=(*self.lifecycle.record.recovery, *notes),
        )

    def fail(self, exc: BaseException) -> None:
        from shinobi.dataset_lifecycle import DatasetLifecyclePhase

        if self.lifecycle.record.phase in {
            DatasetLifecyclePhase.COMMITTED,
            DatasetLifecyclePhase.REFUSED,
            DatasetLifecyclePhase.FAILED,
        }:
            return
        if self.lifecycle.record.phase in {DatasetLifecyclePhase.EXECUTING, DatasetLifecyclePhase.VALIDATED}:
            self._recover_failure(f"execution raised {type(exc).__name__}: {exc}")
        else:
            self.lifecycle.transition(DatasetLifecyclePhase.REFUSED, f"{type(exc).__name__}: {exc}")


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
    dataset_runtime: _WorkerDatasetLifecycle | None = None
    precommitted_result: StepResult | None = None

    def publish_failure(exc: BaseException, *, phase: str = "final") -> None:
        diagnostic = "".join(traceback.format_exception(exc))
        lifecycle = None
        if dataset_runtime is not None and dataset_runtime.evidence.outcome in ("failed", "refused"):
            lifecycle = dataset_runtime.evidence
        failure = AttemptRecord(
            schema_version=2 if lifecycle is not None else 1,
            state="failed",
            error=str(exc),
            stderr=diagnostic,
            sandbox=retained_sandbox(),
            dataset_lifecycle=lifecycle,
            **common,
        )
        try:
            _require_current_invocation(submission_dir, plan, planned, attempt_id)
            _require_exact_owner(submission_dir, plan)
        except BaseException as fence_exc:
            failure = failure.model_copy(
                update={
                    "error": f"{failure.error}; terminal publication refused: {fence_exc}",
                    "stderr": failure.stderr + "\nPublication fence:\n" + "".join(traceback.format_exception(fence_exc)),
                }
            )
            phase = "publication-error"
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

    def precommit_result(result: StepResult) -> None:
        """Fence the invocation and all pinned identities before S1/S2."""

        nonlocal precommitted_result
        _require_current_invocation(submission_dir, plan, planned, attempt_id)
        _require_exact_owner(submission_dir, plan)
        result.sandbox_path = result.sandbox_path or retained_sandbox()
        if frozen.image_digest is not None and result.image_digest != frozen.image_digest:
            raise BundleError(f"step {step_path!r}: executed image digest {result.image_digest!r} does not match submission pin {frozen.image_digest!r}")
        if frozen.tool_venv_digest is not None and result.venv_digest != frozen.tool_venv_digest:
            raise BundleError(f"step {step_path!r}: executed tool venv digest {result.venv_digest!r} does not match submission fingerprint {frozen.tool_venv_digest!r}")
        if result.success and dataset_runtime is not None:
            dataset_runtime.validate()
        result.code_digest = common["code_digest"]
        result.worker_digest = common["worker_digest"]
        result.job_id = job_id
        precommitted_result = result

    def commit_result(result: StepResult, record_cache) -> None:
        """Publish the worker success oracle before updating its cache index.

        SnapshotGuard invokes this between journal commit and trash/marker
        cleanup.  Once the immutable attempt record is visible, a cache-index
        failure may cost a future rerun but cannot revoke completed scientific
        work or turn it into an ambiguous attempt.
        """
        if precommitted_result is not result:
            precommit_result(result)
        committed_lifecycle = dataset_runtime.finish(result) if dataset_runtime is not None else None
        terminal = AttemptRecord.from_result(
            result,
            sandbox=result.sandbox_path,
            dataset_lifecycle=committed_lifecycle or (dataset_runtime.evidence if dataset_runtime is not None else None),
            **common,
        )
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
        if committed_lifecycle is not None:
            try:
                dataset_runtime.lifecycle.adopt(committed_lifecycle)
            except BaseException:  # noqa: BLE001 -- the attempt record is the oracle and embeds this evidence
                logger.exception("step %s committed, but its dataset lifecycle file could not be advanced to committed", step_path)
        if terminal.committed:
            faults("W_RESULT")
            try:
                record_cache()
            except BaseException:  # noqa: BLE001 -- the attempt already committed
                logger.exception("step %s committed, but its reusable cache index could not be updated", step_path)
            faults("W_CACHE")

    try:
        owner = _require_exact_owner(submission_dir, plan)
        _require_current_invocation(submission_dir, plan, planned, attempt_id)
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
            if plan.dataset_lifecycle is not None:
                resolved_cache_dir = Path(cache_dir)
                if not resolved_cache_dir.is_absolute():
                    resolved_cache_dir = Path(bundle.workspace) / resolved_cache_dir
                if resolved_cache_dir.resolve() != Path(plan.dataset_lifecycle.cache_dir):
                    raise BundleError("worker cache directory differs from the frozen strict-dataset recovery store")
                dataset_runtime = _WorkerDatasetLifecycle(
                    submission_dir=submission_dir,
                    plan=plan,
                    attempt_id=attempt_id,
                    step_path=step_path,
                    backend=frozen.backend,
                )
                dataset_runtime.establish(scope=scope, prepared=prepared, owner=owner)
            if snapshots_active and plan.dataset_lifecycle is None:
                paths = {path for values in mutation_paths(scope, prepared).values() for path in values}
                if paths:
                    # Slurm's inferred mutation dependencies ensure no live
                    # sibling can be writing these same paths. Reconcile
                    # before the loop skip decision as well: pass-through
                    # must not leave a prior interrupted rewrite unresolved.
                    reconcile(cache_dir, get_cache_manifest(cache_dir), paths=paths)
            if should_skip(ref, upstream):
                result = passthrough_result(ref, upstream[ref.loop.prev_step], scope.inputs_model(**prepared))
                precommit_result(result)
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
                    _boundary_fields=frozenset(field for field, source in ref.wiring.items() if isinstance(source, InputRef)),
                    _execution_identity=ExecutionIdentity(
                        code_digest=frozen.code.execution_digest if frozen.code else None,
                        image_digest=frozen.image_digest,
                        venv=frozen.tool_venv,
                        venv_digest=frozen.tool_venv_digest,
                    ),
                    _snapshot_success_record=final_path,
                    _snapshot_success_step_path=step_path,
                    _result_precommit=precommit_result,
                    _result_commit=commit_result,
                    _workspace_claimed=dataset_runtime is not None,
                    _dataset_lifecycle=dataset_runtime.lifecycle if dataset_runtime is not None else None,
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
        # Check the oracle first: once the immutable record is committed,
        # rolling the dataset back would revert work it vouches for.
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
        if dataset_runtime is not None:
            try:
                dataset_runtime.fail(exc)
            except BaseException as lifecycle_exc:
                exc.add_note(f"dataset lifecycle failure handling also failed: {type(lifecycle_exc).__name__}: {lifecycle_exc}")
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


def _settle_dataset_submission(
    submission_dir: Path,
    bundle: RecipeBundle,
    plan: ExecutionPlan,
    jobs: dict[str, SubmittedJob],
) -> bool:
    """Recover terminal dataset attempts and publish durable failures.

    Scheduler terminality is only the caller's precondition.  Ownership may
    be released after this function returns true because every mutation
    marker has been reconciled and every non-committed invocation either has
    terminal lifecycle evidence or provably never entered the lifecycle.
    """

    contract = plan.dataset_lifecycle
    if contract is None:
        return True
    settlement_path = submission_dir / "dataset-settlement.json"
    current_attempts = tuple(_latest_attempt(submission_dir, planned) for planned in plan.attempts)
    expected_settlement = DatasetSettlement(
        workflow_id=plan.workflow_id,
        bundle_digest=plan.bundle_digest,
        attempts=current_attempts,
    )
    if settlement_path.exists():
        try:
            return DatasetSettlement.model_validate_json(settlement_path.read_text()) == expected_settlement
        except (OSError, ValueError):
            return False
    try:
        _require_exact_owner(submission_dir, plan)
    except BaseException:
        logger.exception("dataset finalization cannot verify the exact live workflow claim")
        return False

    from shinobi.dataset_lifecycle import (
        DatasetLifecycle,
        DatasetLifecyclePhase,
        DatasetLifecycleStore,
        member_fingerprint,
        observe_roots,
        structural_signature,
    )
    from shinobi.snapshots import HeadStatus, chain_id, get_journal

    written_roots = {access.root for access in contract.initial_snapshot.accesses if access.writes and access.root is not None}
    try:
        if contract.mutation and written_roots:
            notes = reconcile(contract.cache_dir, get_cache_manifest(contract.cache_dir), paths=written_roots, exact=True)
            for note in notes:
                logger.warning("dataset finalizer recovery: %s", note)
        journal = get_journal(contract.cache_dir)
        for root in written_roots:
            chain = journal.get(chain_id(root))
            if chain is not None and (chain.marker is not None or chain.status is not HeadStatus.TRUSTED):
                raise BundleError(f"dataset recovery for {root} did not reach a trusted marker-free state")
    except BaseException:
        logger.exception("dataset finalization could not complete mutation recovery")
        return False

    frozen_by_name = {step.name: step for step in bundle.steps}
    safe = True
    for planned in plan.attempts:
        attempt_id = _latest_attempt(submission_dir, planned)
        final_path = _record_path(submission_dir, attempt_id)
        lifecycle_path = _dataset_record_path(submission_dir, attempt_id)
        if final_path.exists():
            try:
                record = AttemptRecord.read(
                    final_path,
                    workflow_id=plan.workflow_id,
                    attempt_id=attempt_id,
                    step_path=planned.step_path,
                    bundle_digest=plan.bundle_digest,
                )
            except BaseException:
                logger.exception("dataset finalization cannot validate %s", final_path)
                safe = False
                continue
            if record.committed:
                if record.dataset_lifecycle is None or record.dataset_lifecycle.outcome != "committed":
                    safe = False
                continue

        lifecycle = None
        if lifecycle_path.exists():
            try:
                store = DatasetLifecycleStore(lifecycle_path)
                lifecycle = DatasetLifecycle(store, store.attempt(), workspace=Path(contract.storage_namespace))
                record = lifecycle.record
                relevant_roots = {access.root for access in contract.step(planned.step_path).accesses if access.root is not None}
                observations, absent = observe_roots(tuple(sorted(relevant_roots, key=str)), Path(contract.storage_namespace))
                before = {item.root: item for item in record.pre_observations if item.root in relevant_roots}
                after = {item.root: item for item in observations}
                expected_absent = set(record.absent_roots) & relevant_roots

                def matches(root: Path) -> bool:
                    left, right = before.get(root), after.get(root)
                    if left is None or right is None:
                        return left is right
                    if root in written_roots:
                        return structural_signature(left) == structural_signature(right) and member_fingerprint(left) == member_fingerprint(right)
                    return left == right

                # An attempt refused before EXECUTING never launched its tool,
                # so there is nothing of its own to have recovered (and it may
                # have been refused before recording any pre-observation).
                executed = any(event.phase is DatasetLifecyclePhase.EXECUTING for event in record.events)
                if executed and (any(not matches(root) for root in relevant_roots - expected_absent) or set(absent) != expected_absent):
                    raise BundleError(f"dataset attempt {planned.step_path!r} did not recover to its recorded predecessor")
                if record.phase not in {
                    DatasetLifecyclePhase.COMMITTED,
                    DatasetLifecyclePhase.REFUSED,
                    DatasetLifecyclePhase.FAILED,
                }:
                    phase = DatasetLifecyclePhase.FAILED if record.phase in {DatasetLifecyclePhase.EXECUTING, DatasetLifecyclePhase.VALIDATED} else DatasetLifecyclePhase.REFUSED
                    lifecycle.transition(
                        phase,
                        "detached finalizer observed terminal scheduler state and verified dataset recovery",
                        post_observations=observations,
                        recovery=(*record.recovery, "detached finalizer verified the recorded predecessor"),
                    )
            except BaseException:
                logger.exception("dataset finalization cannot establish terminal lifecycle evidence for %s", planned.step_path)
                safe = False
                continue

        if final_path.exists():
            # A worker-published failure is already immutable. Recovery above
            # is the additional condition needed for ownership release.
            continue
        if lifecycle is None and not _record_path(submission_dir, attempt_id, "started").exists():
            # The worker never started (e.g. a dependant Slurm cancelled), so
            # it touched nothing; leave it to scheduler state as "cancelled"
            # rather than inventing a failure for it.
            continue
        frozen = frozen_by_name[planned.step_path]
        job = jobs.get(planned.step_path)
        lifecycle_evidence = lifecycle.record if lifecycle is not None and lifecycle.record.outcome in ("failed", "refused") else None
        failure = AttemptRecord(
            schema_version=2 if lifecycle_evidence is not None else 1,
            workflow_id=plan.workflow_id,
            attempt_id=attempt_id,
            step_path=planned.step_path,
            bundle_digest=plan.bundle_digest,
            state="failed",
            error="scheduler became terminal before the worker published a final result; dataset recovery was verified",
            job_id=job.job_id if job is not None else None,
            code_digest=frozen.code.digest if frozen.code else None,
            worker_digest=plan.worker.source_digest,
            dataset_lifecycle=lifecycle_evidence,
        )
        try:
            failure.write(submission_dir.parent)
        except FileExistsError:
            try:
                AttemptRecord.read(
                    final_path,
                    workflow_id=plan.workflow_id,
                    attempt_id=attempt_id,
                    step_path=planned.step_path,
                    bundle_digest=plan.bundle_digest,
                )
            except BaseException:
                safe = False
    if safe:
        try:
            _write_or_read(settlement_path, expected_settlement)
        except BaseException:
            logger.exception("dataset finalization could not publish release authority")
            return False
    return safe


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
    jobs_terminal = len(jobs) == len(bundle.steps)
    for frozen in bundle.steps:
        _outcome, terminal = _scheduler_attempt_state(scheduler.get(frozen.name, "UNKNOWN"))
        jobs_terminal = jobs_terminal and (terminal or terminal_context)
    dataset_settled = not jobs_terminal or _settle_dataset_submission(submission_dir, bundle, plan, jobs)
    finalized: list[FinalizedStep] = []
    results: dict[str, StepResult] = {}
    all_committed = len(jobs) == len(bundle.steps)
    # A missing job record is not proof that sbatch never accepted the job:
    # the submitter can die between scheduler acceptance and durable publish.
    # Neither finalizer context nor terminal state for the recorded subset may
    # turn that ambiguity into permission to release workspace ownership.
    settled = len(jobs) == len(bundle.steps) and dataset_settled
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
