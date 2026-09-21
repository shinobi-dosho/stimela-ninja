"""Contained local read lifecycle for strict MSv2 contracts.

This module deliberately implements one small capability: a flat local
workflow may read one ordinary, directory-backed MSv2 through the native
backend.  The access planner remains authoritative for resource identities
and :mod:`shinobi.ownership` remains authoritative for exclusion.  The
lifecycle adds durable observations around that existing claim; it is not a
second planner or lock hierarchy.
"""

from __future__ import annotations

import time
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from shinobi.dataset_access import DatasetMode, ResolvedDatasetAccess, plan_recipe_accesses, resolve_scope_dataset_accesses
from shinobi.dataset_closure import DATASET_CLOSURE_PROFILE, ClosureStatus, DatasetClosure, resolve_dataset_closure
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_PROFILE, inspect_measurement_set_v2
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.ownership import WorkspaceOwner
from shinobi.storage import JsonFileStore
from shinobi.steps.schema import Recipe, Scope


DATASET_READ_CAPABILITY = "contained-native-msv2-read/v1"


class DatasetLifecyclePhase(str, Enum):
    """Durable phases of one strict dataset attempt."""

    PLANNED = "planned"
    CLAIMED = "claimed"
    REVALIDATED = "revalidated"
    EXECUTING = "executing"
    VALIDATED = "validated"
    COMMITTED = "committed"
    REFUSED = "refused"
    FAILED = "failed"


class DatasetLifecycleEvent(BaseModel):
    """One append-only phase observation in an attempt record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: DatasetLifecyclePhase
    observed_at: float
    reason: str


class DatasetFileObservation(BaseModel):
    """Cheap metadata identity for one declared CASA table member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class DatasetObservation(BaseModel):
    """Serializable structural, closure and member observation of one MSv2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structural_profile: Literal["msv2-structural/v1"] = MSV2_STRUCTURAL_PROFILE
    closure_profile: Literal["msv2-dataset-closure/v1"] = DATASET_CLOSURE_PROFILE
    root: Path
    descriptor: DatasetDescriptor
    closure: DatasetClosure
    files: tuple[DatasetFileObservation, ...]


class DatasetLifecycleSnapshot(BaseModel):
    """One complete access-plan and on-disk observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accesses: tuple[ResolvedDatasetAccess, ...]
    observations: tuple[DatasetObservation, ...]


class DatasetLifecycleAttempt(BaseModel):
    """Versioned durable provenance for one contained read workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    attempt_id: str
    scope: str
    workspace: Path
    phase: DatasetLifecyclePhase
    events: tuple[DatasetLifecycleEvent, ...]
    backends: tuple[str, ...]
    capability: Literal["contained-native-msv2-read/v1"] = DATASET_READ_CAPABILITY
    capability_supported: bool
    planned_accesses: tuple[ResolvedDatasetAccess, ...] = ()
    claim: WorkspaceOwner | None = None
    pre_observations: tuple[DatasetObservation, ...] = ()
    post_observations: tuple[DatasetObservation, ...] = ()
    outcome: Literal["pending", "committed", "refused", "failed"] = "pending"
    reason: str

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetLifecycleAttempt":
        if not self.events or self.events[-1].phase is not self.phase:
            raise ValueError("dataset lifecycle phase must match the last event")
        if self.events[0].phase is not DatasetLifecyclePhase.PLANNED:
            raise ValueError("dataset lifecycle must begin in the planned phase")
        allowed = {
            DatasetLifecyclePhase.PLANNED: {
                DatasetLifecyclePhase.PLANNED,
                DatasetLifecyclePhase.CLAIMED,
                DatasetLifecyclePhase.REFUSED,
            },
            DatasetLifecyclePhase.CLAIMED: {
                DatasetLifecyclePhase.REVALIDATED,
                DatasetLifecyclePhase.REFUSED,
            },
            DatasetLifecyclePhase.REVALIDATED: {
                DatasetLifecyclePhase.EXECUTING,
                DatasetLifecyclePhase.REFUSED,
            },
            DatasetLifecyclePhase.EXECUTING: {
                DatasetLifecyclePhase.VALIDATED,
                DatasetLifecyclePhase.FAILED,
            },
            DatasetLifecyclePhase.VALIDATED: {
                DatasetLifecyclePhase.COMMITTED,
                DatasetLifecyclePhase.FAILED,
            },
            DatasetLifecyclePhase.COMMITTED: set(),
            DatasetLifecyclePhase.REFUSED: set(),
            DatasetLifecyclePhase.FAILED: set(),
        }
        for previous, current in zip(self.events, self.events[1:]):
            if current.phase not in allowed[previous.phase]:
                raise ValueError(f"invalid dataset lifecycle transition {previous.phase.value!r} -> {current.phase.value!r}")
        if self.reason != self.events[-1].reason:
            raise ValueError("dataset lifecycle reason must match the last event")
        expected = {
            DatasetLifecyclePhase.COMMITTED: "committed",
            DatasetLifecyclePhase.REFUSED: "refused",
            DatasetLifecyclePhase.FAILED: "failed",
        }.get(self.phase, "pending")
        if self.outcome != expected:
            raise ValueError(f"dataset lifecycle outcome {self.outcome!r} disagrees with phase {self.phase.value!r}")
        return self


def dataset_attempt_path(workspace: Path, attempt_id: str) -> Path:
    """Return the durable local attempt-record path."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", attempt_id) is None:
        raise ValueError(f"invalid dataset lifecycle attempt id {attempt_id!r}")
    return workspace.resolve() / ".shinobi" / "dataset-attempts" / f"{attempt_id}.json"


class DatasetLifecycleStore(JsonFileStore):
    """Transactionally update one lifecycle attempt."""

    def create(self, record: DatasetLifecycleAttempt) -> None:
        def update(data: dict[str, Any]) -> None:
            if "record" in data:
                raise DatasetLifecycleUnavailableError(f"dataset lifecycle attempt {record.attempt_id} already exists")
            data["record"] = record.model_dump(mode="json")

        self.update(update)

    def attempt(self) -> DatasetLifecycleAttempt:
        value = self.read().get("record")
        if value is None:
            raise DatasetLifecycleUnavailableError(f"dataset lifecycle attempt record is missing: {self.path}")
        return DatasetLifecycleAttempt.model_validate(value)

    def replace(self, record: DatasetLifecycleAttempt) -> None:
        self.update(lambda data: data.__setitem__("record", record.model_dump(mode="json")))


def read_dataset_attempt(path: Path) -> DatasetLifecycleAttempt:
    """Read and validate one durable lifecycle attempt record."""

    return DatasetLifecycleStore(path).attempt()


class DatasetLifecycle:
    """Mutable controller whose persisted state is an immutable model."""

    def __init__(self, store: DatasetLifecycleStore, record: DatasetLifecycleAttempt):
        self.store = store
        self.record = record

    @classmethod
    def start(cls, *, workspace: Path, attempt_id: str, scope: str, backends: tuple[str, ...]) -> "DatasetLifecycle":
        reason = "strict MSv2 read lifecycle planning started"
        event = DatasetLifecycleEvent(phase=DatasetLifecyclePhase.PLANNED, observed_at=time.time(), reason=reason)
        record = DatasetLifecycleAttempt(
            attempt_id=attempt_id,
            scope=scope,
            workspace=workspace.resolve(),
            phase=DatasetLifecyclePhase.PLANNED,
            events=(event,),
            backends=backends,
            capability_supported=False,
            reason=reason,
        )
        store = DatasetLifecycleStore(dataset_attempt_path(workspace, attempt_id))
        store.create(record)
        return cls(store, record)

    def transition(self, phase: DatasetLifecyclePhase, reason: str, **changes: Any) -> None:
        outcome = {
            DatasetLifecyclePhase.COMMITTED: "committed",
            DatasetLifecyclePhase.REFUSED: "refused",
            DatasetLifecyclePhase.FAILED: "failed",
        }.get(phase, "pending")
        event = DatasetLifecycleEvent(phase=phase, observed_at=time.time(), reason=reason)
        record = self.record.model_copy(
            update={
                "phase": phase,
                "events": (*self.record.events, event),
                "outcome": outcome,
                "reason": reason,
                **changes,
            }
        )
        # model_copy does not re-run validators; round-trip before publication.
        record = DatasetLifecycleAttempt.model_validate(record.model_dump())
        self.store.replace(record)
        self.record = record


def _model_values(model: BaseModel) -> dict[str, Any]:
    values = {name: getattr(model, name) for name in type(model).model_fields}
    values.update(model.model_extra or {})
    return values


def _file_observation(path: Path) -> DatasetFileObservation:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise DatasetLifecycleUnavailableError(f"cannot observe declared MSv2 member {path}: {type(exc).__name__}: {exc}") from exc
    return DatasetFileObservation(
        path=path,
        device=stat.st_dev,
        inode=stat.st_ino,
        mode=stat.st_mode,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _observe_root(root: Path, workspace: Path) -> DatasetObservation:
    closure = resolve_dataset_closure(root, storage_namespace=workspace)
    if closure.status is not ClosureStatus.VALID or closure.root is None:
        raise DatasetLifecycleUnavailableError(f"contained MSv2 read refused: closure revalidation returned {closure.status.value}: {closure.message}")
    external = tuple(resource.path for resource in closure.resources if resource.external_to_root)
    if external:
        raise DatasetLifecycleUnavailableError("contained MSv2 read refused: external closure resources are not supported: " + ", ".join(map(str, external)))
    descriptor = inspect_measurement_set_v2(closure.root)
    if descriptor.status is not DatasetStatus.VALID:
        raise DatasetLifecycleUnavailableError(f"contained MSv2 read refused: structural validation returned {descriptor.status.value}: {descriptor.message}")
    paths: set[Path] = set()
    for resource in closure.resources:
        paths.add(resource.path)
        paths.update(resource.path / name for name in resource.table_files)
    files = tuple(_file_observation(path) for path in sorted(paths, key=str))
    return DatasetObservation(root=closure.root, descriptor=descriptor, closure=closure, files=files)


def resolve_lifecycle_snapshot(
    scope: Scope,
    values: BaseModel,
    *,
    workspace: Path,
    validated_steps: dict[int, tuple[BaseModel, bool]] | None = None,
) -> DatasetLifecycleSnapshot:
    """Resolve the shared access plan and inspect its one contained MSv2."""

    workspace = workspace.resolve()
    if isinstance(scope, Recipe):
        plan = plan_recipe_accesses(
            scope,
            values,
            workspace=workspace,
            validated_steps=validated_steps,
            validated_inputs=values,
        )
        accesses = tuple(access for ref in scope.steps for access in plan.accesses.get(ref.name, ()))
    else:
        accesses = resolve_scope_dataset_accesses(scope, _model_values(values), workspace=workspace)
    if not accesses:
        raise DatasetLifecycleUnavailableError("contained MSv2 read refused: no concrete dataset access was resolved")
    if any(access.mode is not DatasetMode.READ for access in accesses):
        modes = ", ".join(sorted({access.mode.value for access in accesses if access.mode is not DatasetMode.READ}))
        raise DatasetLifecycleUnavailableError(f"contained MSv2 read refused: {modes} access requires mutation recovery, which is not available")
    if any(not access.path_known or access.root is None or access.closure_status is not ClosureStatus.VALID for access in accesses):
        raise DatasetLifecycleUnavailableError("contained MSv2 read refused: every access must resolve to a validated local closure")
    roots = tuple(sorted({access.root for access in accesses if access.root is not None}, key=str))
    if len(roots) != 1:
        raise DatasetLifecycleUnavailableError(f"contained MSv2 read refused: exactly one closure root is supported, resolved {len(roots)}")
    observations = tuple(_observe_root(root, workspace) for root in roots)
    resources = {resource.path for observation in observations for resource in observation.closure.resources}
    planned_resources = {resource for access in accesses for resource in access.resources}
    if resources != planned_resources:
        raise DatasetLifecycleUnavailableError("contained MSv2 read refused: access plan and closure observation disagree")
    return DatasetLifecycleSnapshot(accesses=accesses, observations=observations)


def claim_covers_snapshot(owner: WorkspaceOwner, snapshot: DatasetLifecycleSnapshot) -> bool:
    """Whether the existing shared claim covers every closure resource read-only."""

    claimed = {(Path(access.path), access.writes) for access in owner.accesses}
    return all((resource, False) in claimed for access in snapshot.accesses for resource in access.resources)


def pending_dataset_recovery(cache_dir: str, snapshot: DatasetLifecycleSnapshot) -> tuple[Path, ...]:
    """Dataset resources carrying an unresolved mutation marker.

    Reconciliation can restore or delete the live path.  A contained reader
    holds only a shared read claim over its closure, so it must never perform
    that recovery itself; it refuses until an exclusive writer or an operator
    reconciles the marker.
    """

    from shinobi.snapshots import get_journal
    from shinobi.steps.schema import paths_overlap

    resources = {resource for access in snapshot.accesses for resource in access.resources}
    pending = {
        Path(chain.path)
        for chain in get_journal(cache_dir).all_chains().values()
        if chain.marker is not None and any(paths_overlap(Path(chain.path), resource) for resource in resources)
    }
    return tuple(sorted(pending, key=str))


__all__ = [
    "DATASET_READ_CAPABILITY",
    "DatasetFileObservation",
    "DatasetLifecycle",
    "DatasetLifecycleAttempt",
    "DatasetLifecycleEvent",
    "DatasetLifecyclePhase",
    "DatasetLifecycleSnapshot",
    "DatasetLifecycleStore",
    "DatasetObservation",
    "claim_covers_snapshot",
    "dataset_attempt_path",
    "pending_dataset_recovery",
    "read_dataset_attempt",
    "resolve_lifecycle_snapshot",
]
