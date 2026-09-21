"""Durable ownership of a scientific workspace across whole workflows.

The short locks used by :mod:`shinobi.storage` serialize metadata updates;
they do not own the data a tool may be rewriting.  This module records one
workflow owner under the canonical workspace and holds a separate liveness
lock for local processes.  Detached owners remain recorded between Slurm
jobs and after the submitter exits.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from shinobi.exceptions import ShinobiError
from shinobi.storage import JsonFileStore, SharedStorageError, ofd_lock
from shinobi.steps.schema import InputRef, Recipe, Scope, declares_path_writes, path_accesses, paths_overlap


class WorkspaceOwnershipError(ShinobiError):
    """A workflow cannot safely acquire or change workspace ownership."""


def scope_declares_writes(scope: Scope) -> bool:
    """Whether a scope can write any statically declared filesystem path."""
    if isinstance(scope, Recipe):
        return any(scope_declares_writes(step.step) for step in scope.steps)
    from shinobi.dataset_access import DatasetMode

    return declares_path_writes(scope) or any(access.mode is not DatasetMode.READ for access in scope.dataset_accesses)


def scope_requires_ownership(scope: Scope) -> bool:
    """Whether a workflow must register reads or writes before execution."""

    if isinstance(scope, Recipe):
        return any(scope_requires_ownership(step.step) for step in scope.steps)
    from shinobi.dataset_access import scope_has_dataset_contract

    return declares_path_writes(scope) or scope_has_dataset_contract(scope)


def _workspace(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise WorkspaceOwnershipError(f"workspace {resolved} is not a directory")
    return resolved


def ownership_workspace(default: Path, writes: Iterable[Path]) -> Path:
    """Choose one canonical authority containing every writable target.

    A target's parent is used so ownership metadata never lands inside the
    scientific dataset itself. Multiple targets use their common parent,
    making launches from different current directories converge on the same
    authority. A filesystem root is intentionally not accepted as a useful
    ownership boundary.
    """

    paths = [path.resolve() for path in writes]
    authorities: list[str] = []
    for path in paths:
        parent = path.parent
        while not parent.is_dir() and parent.parent != parent:
            parent = parent.parent
        # A not-yet-created absolute output may have no representable
        # authority below /. It contains no existing scientific state to
        # protect, so leave that output under the launch-workspace claim.
        if parent.parent != parent:
            authorities.append(str(parent))
    if not authorities:
        return _workspace(default)
    common = Path(os.path.commonpath(authorities))
    if common.parent == common:
        raise WorkspaceOwnershipError("mutable paths do not share a safe ownership directory; choose paths under one scientific workspace")
    return _workspace(common)


class WorkspaceAccess(BaseModel):
    """One canonical path covered by a workflow ownership claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    writes: bool


def _canonical_accesses(workspace: Path, accesses: Iterable[tuple[Path, bool]]) -> tuple[WorkspaceAccess, ...]:
    merged: dict[Path, bool] = {}
    for path, writes in accesses:
        canonical = (path if path.is_absolute() else workspace / path).resolve()
        merged[canonical] = merged.get(canonical, False) or writes
    return tuple(WorkspaceAccess(path=str(path), writes=writes) for path, writes in sorted(merged.items(), key=lambda item: str(item[0])))


def _model_values(model: BaseModel) -> dict[str, Any]:
    values = {name: getattr(model, name) for name in type(model).model_fields}
    values.update(model.model_extra or {})
    return values


def _resolved_leaf_inputs(
    scope: Scope,
    values: dict[str, Any] | BaseModel,
    *,
    step_inputs: dict[int, tuple[BaseModel, bool]],
    reusable: bool = True,
    unresolved_inputs: set[str] | None = None,
) -> Iterable[tuple[Scope, dict[str, Any], set[str]]]:
    """Yield each leaf with the inputs knowable at a workflow boundary.

    ``step_inputs`` collects each successfully validated StepRef model. Its
    boolean says whether every value was knowable at the ownership boundary;
    an unresolved ``OutputRef`` makes that step and everything below a nested
    recipe runtime-dependent. Fully knowable models are handed to dispatch
    unchanged, so both default factories and custom validators run exactly
    once. Runtime-dependent models still supply their already-evaluated
    defaults, but execution validates the final upstream values normally.
    """
    unresolved_inputs = set(unresolved_inputs or ())
    if not isinstance(scope, Recipe):
        # Top-level dispatch passes its already-validated model so defaults
        # (including default factories) are part of the claim and are not
        # evaluated a second time before execution.
        if isinstance(values, BaseModel):
            yield scope, _model_values(values), unresolved_inputs
            return
        try:
            validated = scope.inputs_model(**values)
            yield scope, _model_values(validated), unresolved_inputs
        except Exception:  # input validation reports the authoritative error in dispatch
            yield scope, values, unresolved_inputs
        return

    if isinstance(values, BaseModel):
        recipe_inputs = _model_values(values)
    else:
        try:
            recipe_inputs = _model_values(scope.inputs_model(**values))
        except Exception:  # input validation reports the authoritative error in dispatch
            recipe_inputs = dict(values)

    for ref in scope.steps:
        known = dict(ref.params)
        runtime_dependent = False
        unresolved_fields: set[str] = set()
        for field, source in ref.wiring.items():
            sources = source if isinstance(source, list) else [source]
            if all(isinstance(item, InputRef) and item.field in recipe_inputs for item in sources):
                resolved = [recipe_inputs[item.field] for item in sources]
                known[field] = resolved if isinstance(source, list) else resolved[0]
            elif any(not isinstance(item, InputRef) for item in sources):
                runtime_dependent = True
                known.pop(field, None)
                unresolved_fields.add(field)
        validated = None
        try:
            validated = ref.step.inputs_model(**known)
            known = {name: getattr(validated, name) for name in ref.step.inputs_model.model_fields}
            for name in unresolved_fields:
                known.pop(name, None)
            step_inputs[id(ref)] = (validated, reusable and not runtime_dependent and ref.scatter is None)
        except Exception:
            pass
        yield from _resolved_leaf_inputs(
            ref.step,
            validated if validated is not None and not runtime_dependent else known,
            step_inputs=step_inputs,
            reusable=reusable and validated is not None and not runtime_dependent and ref.scatter is None,
            unresolved_inputs=unresolved_fields,
        )


def scope_path_accesses(scope: Scope, values: dict[str, Any] | BaseModel, *, workspace: Path) -> tuple[list[tuple[Path, bool]], dict[int, tuple[BaseModel, bool]]]:
    """Resolve every path access known at a workflow boundary.

    Also returns each StepRef's validated inputs model, keyed by StepRef
    identity, plus whether it is safe to reuse unchanged at execution. This
    makes claim-time validation the one validation for fully knowable steps.
    """
    collected: dict[Path, bool] = {}
    step_inputs: dict[int, tuple[BaseModel, bool]] = {}
    from shinobi.dataset_access import ResolvedAccessPlanner, dataset_workspace_accesses

    planner = ResolvedAccessPlanner(workspace)
    for index, (leaf, known, unresolved) in enumerate(_resolved_leaf_inputs(scope, values, step_inputs=step_inputs)):
        generic_accesses = path_accesses(leaf, known, workspace=workspace)
        for path, writes in generic_accesses:
            collected[path] = collected.get(path, False) or writes
        decision = planner.order_after(
            f"ownership[{index}]",
            leaf,
            known,
            unresolved_inputs=unresolved,
            resolved_path_accesses=generic_accesses,
        )
        for path, writes in dataset_workspace_accesses(decision.datasets):
            collected[path] = collected.get(path, False) or writes
    return list(collected.items()), step_inputs


class WorkspaceOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workflow_id: str
    kind: Literal["local", "slurm"]
    workspace: str
    acquired_at: float
    submission: str | None = None
    accesses: tuple[WorkspaceAccess, ...] = ()
    registry: str


def _same_claim(left: WorkspaceOwner, right: WorkspaceOwner) -> bool:
    """Compare stable claim identity, excluding only acquisition time."""
    return (
        left.workflow_id,
        left.kind,
        left.workspace,
        left.submission,
        left.accesses,
        left.registry,
    ) == (
        right.workflow_id,
        right.kind,
        right.workspace,
        right.submission,
        right.accesses,
        right.registry,
    )


class WorkspaceOwnershipStore(JsonFileStore):
    """The one ownership record below a canonical workspace."""

    def __init__(self, workspace: Path):
        self.workspace = _workspace(workspace)
        super().__init__(self.workspace / ".shinobi" / "workspace-owner.json")

    def owner(self) -> WorkspaceOwner | None:
        data = self.read().get("owner")
        return WorkspaceOwner.model_validate(data) if data is not None else None

    def acquire(self, owner: WorkspaceOwner) -> tuple[WorkspaceOwner, bool]:
        """Return the persisted owner and whether this transaction inserted it."""

        def update(data: dict) -> tuple[WorkspaceOwner, bool]:
            current = data.get("owner")
            if current is not None:
                held = WorkspaceOwner.model_validate(current)
                if held.workflow_id == owner.workflow_id:
                    if not _same_claim(held, owner):
                        raise WorkspaceOwnershipError(f"workspace {self.workspace} has conflicting metadata for workflow {owner.workflow_id}")
                    return held, False
                raise WorkspaceOwnershipError(
                    f"workspace {self.workspace} is owned by {held.kind} workflow {held.workflow_id}"
                    + (f" ({held.submission})" if held.submission else "")
                    + "; inspect or reconcile the recorded owner before retrying"
                )
            data["owner"] = owner.model_dump(mode="json")
            return owner, True

        return self.update(update)

    def release(self, workflow_id: str) -> bool:
        removed = False

        def update(data: dict) -> None:
            nonlocal removed
            current = data.get("owner")
            if current is None:
                return
            held = WorkspaceOwner.model_validate(current)
            if held.workflow_id != workflow_id:
                raise WorkspaceOwnershipError(f"workspace {self.workspace} is owned by workflow {held.workflow_id}, not {workflow_id}")
            data.pop("owner", None)
            removed = True

        self.update(update)
        return removed


class WorkspaceRegistryStore(JsonFileStore):
    """Path-scope registry allowing disjoint workflow claims to coexist."""

    def __init__(self, path: Path):
        self.registry_path = path.expanduser().resolve()
        super().__init__(self.registry_path)

    def acquire(self, owner: WorkspaceOwner) -> bool:
        """Return whether this transaction inserted the workflow entry."""

        def update(data: dict) -> bool:
            owners = data.setdefault("owners", {})
            current = owners.get(owner.workflow_id)
            if current is not None:
                if WorkspaceOwner.model_validate(current) != owner:
                    raise WorkspaceOwnershipError(f"registry {self.registry_path} has conflicting metadata for workflow {owner.workflow_id}")
                return False
            for value in owners.values():
                held = WorkspaceOwner.model_validate(value)
                if _owners_conflict(owner, held):
                    raise WorkspaceOwnershipError(
                        f"path ownership conflicts with {held.kind} workflow {held.workflow_id} in {held.workspace}; inspect or reconcile that recorded owner before retrying"
                    )
            owners[owner.workflow_id] = owner.model_dump(mode="json")
            return True

        return self.update(update)

    def owner(self, workflow_id: str) -> WorkspaceOwner | None:
        value = self.read().get("owners", {}).get(workflow_id)
        return WorkspaceOwner.model_validate(value) if value is not None else None

    def release(self, workflow_id: str) -> bool:
        removed = False

        def update(data: dict) -> None:
            nonlocal removed
            owners = data.setdefault("owners", {})
            removed = owners.pop(workflow_id, None) is not None

        self.update(update)
        return removed


_leases: dict[tuple[str, str], "WorkspaceLease"] = {}
_leases_lock = threading.Lock()


def _owners_conflict(left: WorkspaceOwner, right: WorkspaceOwner) -> bool:
    if not left.accesses or not right.accesses:
        # Old/opaque records cannot prove disjointness.
        return True
    return any((a.writes or b.writes) and paths_overlap(Path(a.path), Path(b.path)) for a in left.accesses for b in right.accesses)


def ownership_registry() -> Path:
    """Canonical shared registry path, overridable for site storage."""
    return Path(os.environ.get("SHINOBI_OWNERSHIP_REGISTRY", "~/.shinobi/workspace-owners.json")).expanduser().resolve()


def _require_registry(owner: WorkspaceOwner) -> None:
    registered = WorkspaceRegistryStore(Path(owner.registry)).owner(owner.workflow_id)
    if registered != owner:
        raise WorkspaceOwnershipError(f"workspace owner {owner.workflow_id} has no matching path-registry claim at {owner.registry}")


@dataclass
class WorkspaceLease:
    store: WorkspaceOwnershipStore
    owner: WorkspaceOwner
    _fd: int | None = None

    def release(self) -> None:
        """Release only this workflow's record, then its liveness lock."""
        try:
            WorkspaceRegistryStore(Path(self.owner.registry)).release(self.owner.workflow_id)
            self.store.release(self.owner.workflow_id)
        finally:
            if self._fd is not None:
                try:
                    ofd_lock(self._fd, fcntl.F_UNLCK, blocking=False)
                finally:
                    os.close(self._fd)
                    self._fd = None
            self.store.path.with_name(f"workspace-owner.{self.owner.workflow_id}.live").unlink(missing_ok=True)
            with _leases_lock:
                _leases.pop((self.owner.workspace, self.owner.workflow_id), None)


def acquire_workspace(
    workspace: Path,
    workflow_id: str,
    *,
    kind: Literal["local", "slurm"],
    submission: Path | None = None,
    accesses: Iterable[tuple[Path, bool]] | None = None,
    registry: Path | None = None,
) -> WorkspaceLease:
    """Atomically claim a workspace and its canonical scientific paths.

    Callers with concrete mutation targets choose the authority with
    :func:`ownership_workspace`, so aliases and unrelated launch directories
    converge before this atomic claim.
    """
    store = WorkspaceOwnershipStore(workspace)
    canonical = _canonical_accesses(store.workspace, accesses or [(store.workspace, True)])
    registry_path = (registry or ownership_registry()).expanduser().resolve()
    owner = WorkspaceOwner(
        workflow_id=workflow_id,
        kind=kind,
        workspace=str(store.workspace),
        acquired_at=time.time(),
        submission=str(submission.resolve()) if submission is not None else None,
        accesses=canonical,
        registry=str(registry_path),
    )
    key = (owner.workspace, owner.workflow_id)
    with _leases_lock:
        existing = _leases.get(key)
        if existing is not None:
            if not _same_claim(existing.owner, owner):
                raise WorkspaceOwnershipError(f"workspace {store.workspace} has conflicting metadata for workflow {workflow_id}")
            return existing

    fd = None
    if kind == "local":
        live = store.path.with_name(f"workspace-owner.{workflow_id}.live")
        live.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(live, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            ofd_lock(fd, fcntl.F_WRLCK, blocking=False)
        except BaseException:
            os.close(fd)
            raise
    registered = WorkspaceRegistryStore(registry_path)
    root_added = False
    registry_added = False
    try:
        persisted, root_added = store.acquire(owner)
        owner = persisted
        registry_added = registered.acquire(owner)
    except BaseException:
        if registry_added:
            try:
                registered.release(workflow_id)
            except BaseException:
                pass
        try:
            if root_added:
                store.release(workflow_id)
        except BaseException:
            pass
        if fd is not None:
            os.close(fd)
        raise
    lease = WorkspaceLease(store, owner, fd)
    with _leases_lock:
        _leases[key] = lease
    return lease


def inspect_workspace(workspace: Path) -> WorkspaceOwner | None:
    """Read the current owner without changing it."""
    return WorkspaceOwnershipStore(workspace).owner()


def local_owner_live(owner: WorkspaceOwner) -> bool | None:
    """True/False from the durable liveness lock, or None if uncertain."""
    if owner.kind != "local":
        return None
    store = WorkspaceOwnershipStore(Path(owner.workspace))
    path = store.path.with_name(f"workspace-owner.{owner.workflow_id}.live")
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            ofd_lock(fd, fcntl.F_WRLCK, blocking=False)
        except BlockingIOError:
            return True
        except SharedStorageError:
            return None
        ofd_lock(fd, fcntl.F_UNLCK, blocking=False)
        return False
    except OSError:
        return None
    finally:
        os.close(fd)


def release_workspace(workspace: Path, workflow_id: str) -> bool:
    """Remove exactly one workflow's ownership record, never another's."""
    store = WorkspaceOwnershipStore(workspace)
    current = store.owner()
    if current is not None and current.workflow_id == workflow_id:
        WorkspaceRegistryStore(Path(current.registry)).release(workflow_id)
    removed = store.release(workflow_id)
    if removed and current is not None:
        with _leases_lock:
            lease = _leases.pop((current.workspace, current.workflow_id), None)
        if lease is not None and lease._fd is not None:
            try:
                ofd_lock(lease._fd, fcntl.F_UNLCK, blocking=False)
            finally:
                os.close(lease._fd)
                lease._fd = None
        store.path.with_name(f"workspace-owner.{current.workflow_id}.live").unlink(missing_ok=True)
    return removed


def require_workspace_owner(workspace: Path, workflow_id: str) -> WorkspaceOwner:
    """Fail unless ``workflow_id`` is still the durable workspace owner."""
    owner = inspect_workspace(workspace)
    if owner is None:
        raise WorkspaceOwnershipError(f"workspace {_workspace(workspace)} has no owner; refusing detached workflow {workflow_id}")
    if owner.workflow_id != workflow_id:
        raise WorkspaceOwnershipError(f"workspace {_workspace(workspace)} is now owned by workflow {owner.workflow_id}; refusing stale detached workflow {workflow_id}")
    _require_registry(owner)
    return owner


@dataclass(frozen=True)
class OwnershipInspection:
    owner: WorkspaceOwner | None
    liveness: Literal["free", "live", "dead", "uncertain"]
    detail: str


def inspect_ownership(workspace: Path) -> OwnershipInspection:
    """Inspect storage and scheduler/liveness evidence without changing it."""
    try:
        owner = inspect_workspace(workspace)
    except (OSError, ValueError, SharedStorageError, json.JSONDecodeError) as exc:
        return OwnershipInspection(None, "uncertain", f"workspace-owner evidence cannot be read: {exc}")
    if owner is None:
        return OwnershipInspection(None, "free", "workspace has no recorded owner")
    try:
        _require_registry(owner)
    except (OSError, ValueError, WorkspaceOwnershipError, SharedStorageError, json.JSONDecodeError) as exc:
        return OwnershipInspection(owner, "uncertain", f"path-registry evidence cannot be verified: {exc}")
    if owner.kind == "local":
        live = local_owner_live(owner)
        if live is True:
            return OwnershipInspection(owner, "live", "the owner's durable liveness lock is held")
        if live is False:
            return OwnershipInspection(owner, "dead", "the owner's durable liveness lock is no longer held")
        return OwnershipInspection(owner, "uncertain", "the owner's liveness lock could not be verified")

    if owner.submission is None:
        return OwnershipInspection(owner, "uncertain", "the detached owner has no recorded submission directory")
    submission = Path(owner.submission)
    try:
        from shinobi.offload.bundle import RecipeBundle, Submission
        from shinobi.offload.worker import ExecutionPlan, SubmittedJob

        submitted = Submission.model_validate_json((submission / "submission.json").read_text())
        bundle = RecipeBundle.read(submission / "bundle.json")
        execution = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
        if str(submitted.workflow_id) != owner.workflow_id or execution.workflow_id != submitted.workflow_id:
            raise ValueError("submission workflow identity does not match the recorded owner")
        if submitted.bundle_digest != bundle.digest or execution.bundle_digest != bundle.digest:
            raise ValueError("submission bundle identities disagree")
        if execution.ownership_workspace is None or Path(execution.ownership_workspace).resolve() != Path(owner.workspace):
            raise ValueError("execution ownership workspace does not match the recorded owner")
        if execution.ownership_registry is None or Path(execution.ownership_registry).resolve() != Path(owner.registry):
            raise ValueError("execution ownership registry does not match the recorded owner")
        if not execution.ownership_required or execution.accesses != owner.accesses:
            raise ValueError("execution plan ownership requirement does not match the recorded owner")
        planned = {attempt.step_path: attempt for attempt in execution.attempts}
        job_files = sorted((submission / "jobs").glob("*.json"))
        jobs = [SubmittedJob.model_validate_json(path.read_text()) for path in job_files]
        if any(job.workflow_id != submitted.workflow_id or job.bundle_digest != bundle.digest for job in jobs):
            raise ValueError("a submitted-job record does not belong to the recorded owner")
        recorded = {job.step_path: job for job in jobs}
        if len(recorded) != len(jobs):
            raise ValueError("submission has duplicate scheduler records for one planned step")
        if recorded.keys() - planned.keys():
            raise ValueError("submission has scheduler records for an unplanned step")
        if any(job.attempt_id != planned[name].attempt_id for name, job in recorded.items()):
            raise ValueError("a submitted-job record has the wrong planned attempt identity")
        missing = planned.keys() - recorded.keys()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return OwnershipInspection(owner, "uncertain", f"submission evidence cannot be read: {exc}")
    if not jobs:
        return OwnershipInspection(owner, "uncertain", "submission has no durable scheduler job records")
    try:
        from shinobi.offload.slurm import status_slurm
        from shinobi.offload.worker import _scheduler_attempt_state

        states = status_slurm({job.step_path: job.job_id for job in jobs})
    except Exception as exc:  # scheduler/accounting failure is uncertainty, never death
        return OwnershipInspection(owner, "uncertain", f"scheduler state could not be verified: {exc}")
    outcomes = [_scheduler_attempt_state(states.get(job.step_path, "UNKNOWN")) for job in jobs]
    if any(not terminal and state != "unknown" for state, terminal in outcomes):
        return OwnershipInspection(owner, "live", "at least one accepted Slurm job is queued or running")
    if any(not terminal for _state, terminal in outcomes):
        return OwnershipInspection(owner, "uncertain", "scheduler accounting cannot establish that every accepted job is terminal")
    if missing:
        return OwnershipInspection(
            owner,
            "uncertain",
            f"every recorded Slurm job is terminal, but {len(missing)} planned job(s) have no durable scheduler record; submission may have crashed after scheduler acceptance",
        )
    return OwnershipInspection(owner, "dead", "every planned Slurm job has a durable record and is terminal")


def reconcile_ownership(workspace: Path) -> OwnershipInspection:
    """Release a demonstrably dead owner; refuse live or uncertain owners."""
    inspection = inspect_ownership(workspace)
    if inspection.owner is None:
        if inspection.liveness != "free":
            raise WorkspaceOwnershipError(f"refusing to release workspace ownership: ownership is {inspection.liveness} ({inspection.detail})")
        return inspection
    if inspection.liveness != "dead":
        raise WorkspaceOwnershipError(
            f"refusing to release {inspection.owner.kind} workflow {inspection.owner.workflow_id}: ownership is {inspection.liveness} ({inspection.detail})"
        )
    release_workspace(workspace, inspection.owner.workflow_id)
    return inspection
