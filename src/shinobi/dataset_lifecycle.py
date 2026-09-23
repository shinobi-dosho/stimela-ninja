"""Contained local read and mutation lifecycles for strict MSv2 contracts.

This module deliberately implements two small capabilities on one route: a
flat local workflow may read, and (with exact recovery) write or create,
ordinary directory-backed MSv2 datasets through the native backend.  The
access planner remains authoritative for resource identities,
:mod:`shinobi.ownership` remains authoritative for exclusion, and
:mod:`shinobi.snapshots` remains authoritative for naming, snapshotting and
restoring states.  The lifecycle adds durable observations and declared
postconditions around those mechanisms; it is not a second planner, lock
hierarchy or snapshot store.

**Four identities, deliberately kept apart** in every mutation record:

- *physical snapshot identity* -- the snapshot directory holding an exact
  copy of a state's closure (``predecessor_snapshot``/``successor_snapshot``);
- *structural identity* -- :func:`structural_signature`, a hash of the
  bounded ``msv2-structural/v1`` observation (row count, column, keyword and
  subtable names, closure membership, table files and storage managers). It
  refuses an incompatible tree; it never proves visibility values unchanged;
- *logical cache identity* -- the step's skip-cache key, which names the
  state it produces;
- *provenance lineage* -- the journal state names (``predecessor_state`` and
  ``successor_state``), i.e. which step and output field produced a state.

A future MSv4/Zarr reusable-state tier adds its own logical-state identity;
it is not any of these and must never replace the native predecessor
snapshot as the exact rollback source.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from shinobi.dataset_access import DatasetFallback, DatasetMode, DatasetTable, ResolvedDatasetAccess, plan_recipe_accesses, resolve_scope_dataset_accesses
from shinobi.dataset_closure import DATASET_CLOSURE_PROFILE, ClosureStatus, DatasetClosure, resolve_dataset_closure
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_PROFILE, inspect_measurement_set_v2
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.ownership import WorkspaceOwner
from shinobi.storage import JsonFileStore
from shinobi.steps.schema import Recipe, Scope


DATASET_READ_CAPABILITY = "contained-native-msv2-read/v1"
DATASET_MUTATION_CAPABILITY = "contained-native-msv2-mutation/v1"
STRUCTURAL_SIGNATURE_VERSION = "msv2-structural-signature/v1"
# What a structural signature covers, recorded beside every use of one so no
# record can be read as a content identity.
STRUCTURAL_SIGNATURE_COVERAGE = (
    "MAIN row count; MAIN column, keyword and subtable names; closure membership, "
    "table files and storage managers. Cell values, keyword values and visibility data are not examined."
)


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
    """One complete access-plan and on-disk observation.

    ``absent_roots`` are CREATE targets, observed as absent: their
    predecessor is absence by contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    accesses: tuple[ResolvedDatasetAccess, ...]
    observations: tuple[DatasetObservation, ...]
    absent_roots: tuple[Path, ...] = ()


def structural_signature(observation: DatasetObservation) -> str:
    """The structural identity of one observed MSv2 closure.

    Paths are recorded relative to the root and stat metadata is excluded, so
    an exact snapshot restored at the same path, or the same state observed
    through an alias, has the same signature. See
    :data:`STRUCTURAL_SIGNATURE_COVERAGE` for what it does *not* cover.
    """

    root = observation.root
    descriptor = observation.descriptor
    resources = []
    for resource in observation.closure.resources:
        resources.append(
            {
                "path": str(resource.path.relative_to(root)) if resource.path.is_relative_to(root) else str(resource.path),
                "members": sorted(resource.members),
                "files": sorted(resource.table_files),
                "managers": sorted(resource.storage_managers),
                "external": resource.external_to_root,
            }
        )
    blob = {
        "version": STRUCTURAL_SIGNATURE_VERSION,
        "structural_profile": observation.structural_profile,
        "closure_profile": observation.closure_profile,
        "status": descriptor.status.value,
        "nrows": descriptor.nrows,
        "columns": sorted(descriptor.columns),
        "keywords": sorted(descriptor.keywords),
        "subtables": sorted(descriptor.subtables),
        "resources": sorted(resources, key=lambda item: item["path"]),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


class DatasetMutationOutcome(str, Enum):
    """What finally happened to one strict dataset a leaf wrote or created."""

    PENDING = "pending"
    COMMITTED = "committed"
    # Failure: the dataset is exactly its predecessor again.
    ROLLED_BACK = "rolled-back"
    ABSENT_RESTORED = "absent-restored"
    # Failure whose rollback also failed: the journal marks the dataset for
    # restore before anything else may use it.
    UNTRUSTED = "untrusted"
    REFUSED = "refused"


class DatasetMutationRecord(BaseModel):
    """Predecessor/successor evidence for one strict write or create.

    The four identities in the module docstring are separate fields here on
    purpose; none of them stands in for another.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    mode: DatasetMode
    root: Path
    predecessor_state: str | None
    predecessor_signature: str | None
    predecessor_snapshot: Path | None
    successor_state: str
    successor_signature: str | None = None
    successor_snapshot: Path | None = None
    signature_coverage: str = STRUCTURAL_SIGNATURE_COVERAGE
    outcome: DatasetMutationOutcome = DatasetMutationOutcome.PENDING


class DatasetCacheDecision(BaseModel):
    """Which identity and fingerprint coverage one leaf's cache decision used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["hit", "miss", "rejected-hit", "disabled"]
    cache_key: str | None
    identity: str
    dataset_coverage: tuple[str, ...]
    reason: str


class DatasetLeafAttempt(BaseModel):
    """One atomic step's strict observations under the workflow's claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_path: str
    accesses: tuple[ResolvedDatasetAccess, ...]
    cache: DatasetCacheDecision | None = None
    pre_observations: tuple[DatasetObservation, ...] = ()
    absent_before: tuple[Path, ...] = ()
    post_observations: tuple[DatasetObservation, ...] = ()
    mutations: tuple[DatasetMutationRecord, ...] = ()
    outcome: Literal["pending", "validated", "committed", "reused", "failed", "refused"] = "pending"
    reason: str = ""


class DatasetLifecycleAttempt(BaseModel):
    """Versioned durable provenance for one contained strict workflow.

    Schema 1 is the read-only baseline. Schema 2 adds per-leaf records and
    is required for the mutation capability; a v2 read-only attempt is also
    valid, so readers inside a mutation workflow share the leaf format.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 1
    attempt_id: str
    scope: str
    workspace: Path
    phase: DatasetLifecyclePhase
    events: tuple[DatasetLifecycleEvent, ...]
    backends: tuple[str, ...]
    capability: Literal["contained-native-msv2-read/v1", "contained-native-msv2-mutation/v1"] = DATASET_READ_CAPABILITY
    capability_supported: bool
    planned_accesses: tuple[ResolvedDatasetAccess, ...] = ()
    claim: WorkspaceOwner | None = None
    pre_observations: tuple[DatasetObservation, ...] = ()
    post_observations: tuple[DatasetObservation, ...] = ()
    absent_roots: tuple[Path, ...] = ()
    leaves: tuple[DatasetLeafAttempt, ...] = ()
    recovery: tuple[str, ...] = ()
    outcome: Literal["pending", "committed", "refused", "failed"] = "pending"
    reason: str

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetLifecycleAttempt":
        if self.schema_version == 1 and (self.capability != DATASET_READ_CAPABILITY or self.leaves or self.absent_roots or self.recovery):
            raise ValueError("dataset lifecycle schema 1 records only the contained read capability")
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


def mutation_committed(path: Path, *, attempt_id: str, step_path: str, cache_key: str | None) -> bool:
    """Whether a strict attempt record committed this exact leaf invocation.

    This is the success oracle a strict mutation's in-flight marker names
    (``Marker.success_kind == "dataset-lifecycle"``). The leaf entry is
    written at S3, after the successor snapshot and journal commit, so a
    crash on either side of it is decided correctly by reconciliation. An
    unreadable record is not success.
    """

    try:
        record = DatasetLifecycleStore(path).attempt()
    except (OSError, ValueError, DatasetLifecycleUnavailableError):
        return False
    if record.attempt_id != attempt_id or cache_key is None:
        return False
    return any(leaf.step_path == step_path and leaf.outcome == "committed" and leaf.cache is not None and leaf.cache.cache_key == cache_key for leaf in record.leaves)


class DatasetLifecycle:
    """Mutable controller whose persisted state is an immutable model.

    Leaves of a flat recipe may run concurrently and each amends its own
    entry, so every update is a read-modify-write under one lock.
    """

    def __init__(self, store: DatasetLifecycleStore, record: DatasetLifecycleAttempt, *, workspace: Path | None = None):
        self.store = store
        self.record = record
        self.workspace = (workspace or record.workspace).resolve()
        self._lock = threading.RLock()

    @property
    def mutation(self) -> bool:
        return self.record.capability == DATASET_MUTATION_CAPABILITY

    @classmethod
    def start(cls, *, workspace: Path, attempt_id: str, scope: str, backends: tuple[str, ...], mutation: bool = False) -> "DatasetLifecycle":
        reason = f"strict MSv2 {'mutation' if mutation else 'read'} lifecycle planning started"
        event = DatasetLifecycleEvent(phase=DatasetLifecyclePhase.PLANNED, observed_at=time.time(), reason=reason)
        record = DatasetLifecycleAttempt(
            schema_version=2 if mutation else 1,
            attempt_id=attempt_id,
            scope=scope,
            workspace=workspace.resolve(),
            phase=DatasetLifecyclePhase.PLANNED,
            events=(event,),
            backends=backends,
            capability=DATASET_MUTATION_CAPABILITY if mutation else DATASET_READ_CAPABILITY,
            capability_supported=False,
            reason=reason,
        )
        store = DatasetLifecycleStore(dataset_attempt_path(workspace, attempt_id))
        store.create(record)
        return cls(store, record, workspace=workspace)

    def _publish(self, update: dict[str, Any]) -> None:
        record = self.record.model_copy(update=update)
        # model_copy does not re-run validators; round-trip before publication.
        record = DatasetLifecycleAttempt.model_validate(record.model_dump())
        self.store.replace(record)
        self.record = record

    def transition(self, phase: DatasetLifecyclePhase, reason: str, **changes: Any) -> None:
        outcome = {
            DatasetLifecyclePhase.COMMITTED: "committed",
            DatasetLifecyclePhase.REFUSED: "refused",
            DatasetLifecyclePhase.FAILED: "failed",
        }.get(phase, "pending")
        with self._lock:
            event = DatasetLifecycleEvent(phase=phase, observed_at=time.time(), reason=reason)
            self._publish(
                {
                    "phase": phase,
                    "events": (*self.record.events, event),
                    "outcome": outcome,
                    "reason": reason,
                    **changes,
                }
            )

    def amend(self, **changes: Any) -> None:
        """Update evidence without a phase transition (e.g. recovery notes)."""

        with self._lock:
            self._publish(changes)

    def record_leaf(self, leaf: DatasetLeafAttempt) -> None:
        """Insert or replace the entry for ``leaf.step_path``."""

        with self._lock:
            leaves = [existing for existing in self.record.leaves if existing.step_path != leaf.step_path]
            self._publish({"leaves": (*leaves, leaf)})

    def leaf(self, step_path: str) -> DatasetLeafAttempt | None:
        with self._lock:
            return next((leaf for leaf in self.record.leaves if leaf.step_path == step_path), None)


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


def observe_dataset(root: Path, workspace: Path) -> DatasetObservation:
    """Observe one contained ordinary MSv2, refusing anything else."""

    return _observe_root(root, workspace.resolve())


def dataset_signature(root: Path, workspace: Path) -> str:
    """:func:`structural_signature` of a freshly observed root."""

    return structural_signature(observe_dataset(root, workspace))


def observe_roots(roots: tuple[Path, ...], workspace: Path) -> tuple[tuple[DatasetObservation, ...], tuple[Path, ...]]:
    """Observe existing roots; report the rest as absent."""

    observations: list[DatasetObservation] = []
    absent: list[Path] = []
    for root in roots:
        if not root.exists() and not root.is_symlink():
            absent.append(root)
        else:
            observations.append(_observe_root(root, workspace))
    return tuple(observations), tuple(absent)


def resolve_lifecycle_snapshot(
    scope: Scope,
    values: BaseModel,
    *,
    workspace: Path,
    validated_steps: dict[int, tuple[BaseModel, bool]] | None = None,
    mutation: bool = False,
) -> DatasetLifecycleSnapshot:
    """Resolve the shared access plan and inspect its contained MSv2 roots.

    A read lifecycle accepts exactly one root. A mutation lifecycle accepts
    several pairwise-disjoint contained roots, and observes a CREATE target
    as absent.
    """

    if mutation:
        return _resolve_mutation_snapshot(scope, values, workspace=workspace.resolve(), validated_steps=validated_steps)
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


def _resolve_mutation_snapshot(
    scope: Scope,
    values: BaseModel,
    *,
    workspace: Path,
    validated_steps: dict[int, tuple[BaseModel, bool]] | None,
) -> DatasetLifecycleSnapshot:
    prefix = "contained MSv2 mutation refused"
    if isinstance(scope, Recipe):
        plan = plan_recipe_accesses(scope, values, workspace=workspace, validated_steps=validated_steps, validated_inputs=values)
        accesses = tuple(access for ref in scope.steps for access in plan.accesses.get(ref.name, ()))
    else:
        accesses = resolve_scope_dataset_accesses(scope, _model_values(values), workspace=workspace)
    if not accesses:
        raise DatasetLifecycleUnavailableError(f"{prefix}: no concrete dataset access was resolved")
    created = {access.root for access in accesses if access.mode is DatasetMode.CREATE and access.root is not None}
    for access in accesses:
        if not access.path_known or access.root is None:
            raise DatasetLifecycleUnavailableError(f"{prefix}: dataset field {access.field!r} has no path known at the claim boundary")
        if access.closure_status is not ClosureStatus.VALID and access.root not in created:
            raise DatasetLifecycleUnavailableError(f"{prefix}: dataset field {access.field!r} does not resolve to a validated local closure")
    roots = tuple(sorted({access.root for access in accesses if access.root is not None}, key=str))
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise DatasetLifecycleUnavailableError(f"{prefix}: dataset roots {left} and {right} are nested")
    existing = tuple(root for root in roots if root not in created)
    observations = tuple(_observe_root(root, workspace) for root in existing)
    for root in created:
        if root.exists() or root.is_symlink():
            raise DatasetLifecycleUnavailableError(f"{prefix}: CREATE target {root} already exists")
    for observation in observations:
        observed = {resource.path for resource in observation.closure.resources}
        planned = {resource for access in accesses if access.root == observation.root for resource in access.resources}
        if observed != planned:
            raise DatasetLifecycleUnavailableError(f"{prefix}: access plan and closure observation disagree for {observation.root}")
    return DatasetLifecycleSnapshot(accesses=accesses, observations=observations, absent_roots=tuple(sorted(created, key=str)))


def _claimed_for(owner: WorkspaceOwner, resource: Path, writes: bool) -> bool:
    """Whether ``owner`` holds ``resource`` at least as strongly as needed.

    Containment counts: a CREATE target is claimed at its root before its
    subtables exist, and a write claim satisfies a read of the same bytes.
    """

    for access in owner.accesses:
        claimed = Path(access.path)
        if (resource == claimed or resource.is_relative_to(claimed)) and (access.writes or not writes):
            return True
    return False


def claim_covers_accesses(owner: WorkspaceOwner, accesses: tuple[ResolvedDatasetAccess, ...]) -> tuple[Path, ...]:
    """Resources ``accesses`` need that ``owner`` does not cover (empty = covered)."""

    return tuple(sorted({resource for access in accesses for resource in access.resources if not _claimed_for(owner, resource, access.writes)}, key=str))


# -- strict leaves ---------------------------------------------------------


class LeafMutationError(DatasetLifecycleUnavailableError):
    """A strict leaf's shape cannot receive exact mutation recovery."""


def strict_mutation_targets(scope: Scope, accesses: tuple[ResolvedDatasetAccess, ...]) -> tuple[dict[str, Path], frozenset[str]]:
    """The guarded ``{field: root}`` of one leaf, and which are CREATE targets.

    Refuses the shapes whose predecessor or successor would have no single
    name: a write addressed through a subtable path, several fields writing
    one root, or a writer whose field is not in the shared mutated set (the
    cache key and the snapshotter would then disagree about it).
    """

    from shinobi.steps.schema import mutated_path_fields

    mutated = mutated_path_fields(scope)
    fields: dict[str, Path] = {}
    creates: set[str] = set()
    for access in accesses:
        if not access.writes:
            continue
        assert access.root is not None and access.requested_path is not None
        if access.requested_path != access.root:
            raise LeafMutationError(
                f"strict MSv2 mutation of {scope.name!r} field {access.field!r} addresses {access.requested_path}, not its dataset root {access.root}; "
                "declare the subtable with DatasetAccess.table and pass the MS root"
            )
        if access.mode is DatasetMode.WRITE and access.field not in mutated:
            raise LeafMutationError(f"strict MSv2 write field {access.field!r} of {scope.name!r} is not an in-place mutation input")
        previous = fields.get(access.field)
        if previous is not None and previous != access.root:
            raise LeafMutationError(f"strict MSv2 field {access.field!r} of {scope.name!r} resolves to two roots")
        fields[access.field] = access.root
        if access.mode is DatasetMode.CREATE:
            creates.add(access.field)
    roots = list(fields.values())
    if len(set(roots)) != len(roots):
        raise LeafMutationError(f"strict MSv2 mutation of {scope.name!r}: several fields write one dataset root; one field per written dataset is supported")
    return fields, frozenset(creates)


def _changed_tables(pre: DatasetObservation, post: DatasetObservation) -> set[str]:
    """Closure members whose own backing files changed between observations."""

    def by_resource(observation: DatasetObservation) -> dict[str, dict[Path, DatasetFileObservation]]:
        grouped: dict[str, dict[Path, DatasetFileObservation]] = {}
        for resource in observation.closure.resources:
            own = {item.path: item for item in observation.files if item.path == resource.path or item.path.parent == resource.path and item.path.name in resource.table_files}
            for member in resource.members:
                grouped[member] = own
        return grouped

    def content(files: dict[Path, DatasetFileObservation]) -> dict[Path, tuple[int, int, int, int]]:
        # ctime is excluded: it also moves for a mere chmod or link change.
        return {path: (item.size, item.mtime_ns, item.inode, item.mode) for path, item in files.items()}

    before, after = by_resource(pre), by_resource(post)
    changed: set[str] = set()
    for member in set(before) | set(after):
        left, right = before.get(member), after.get(member)
        if left is None or right is None or content(left) != content(right):
            changed.add(member)
    return changed


def leaf_postcondition_issues(
    accesses: tuple[ResolvedDatasetAccess, ...],
    pre: dict[Path, DatasetObservation],
    post: dict[Path, DatasetObservation | None],
) -> list[str]:
    """Every way one leaf's observed change exceeds its declared contract.

    Per dataset root: a pure reader must leave the observation identical; a
    CREATE must produce a valid contained MSv2 carrying its declared
    columns; a writer may change row count, schema (columns, subtables,
    closure membership) and MAIN keyword names only where its declarations
    permit, may add/remove only declared columns when its columns are
    known, and may touch only the tables it declares -- unless it was an
    undeclared whole-dataset writer, for which any table is in scope.

    These are structural checks. Which *cells* a writer changed is not
    observable here, so declared column writes are recorded, not verified.
    """

    issues: list[str] = []
    roots = {access.root for access in accesses if access.root is not None}
    for root in sorted(roots, key=str):
        mine = [access for access in accesses if access.root == root]
        writers = [access for access in mine if access.writes]
        before, after = pre.get(root), post.get(root)
        label = root.name or str(root)
        if not writers:
            if after != before:
                issues.append(f"{label}: read-only access changed the dataset")
            continue
        if after is None:
            issues.append(f"{label}: the dataset is absent after the step")
            continue
        creates = [access for access in writers if access.mode is DatasetMode.CREATE]
        # The descriptor observes MAIN's columns only, so only MAIN column
        # declarations can be checked against it.
        main_create = {
            name for access in writers if access.declaration.table is DatasetTable.MAIN and access.declaration.columns is not None for name in access.declaration.columns.create
        }
        main_remove = {
            name for access in writers if access.declaration.table is DatasetTable.MAIN and access.declaration.columns is not None for name in access.declaration.columns.remove
        }
        columns = set(after.descriptor.columns)
        missing = sorted(main_create - columns)
        if missing:
            issues.append(f"{label}: declared created column(s) absent: {', '.join(missing)}")
        lingering = sorted(main_remove & columns)
        if lingering:
            issues.append(f"{label}: declared removed column(s) still present: {', '.join(lingering)}")
        if creates:
            if before is not None:
                issues.append(f"{label}: CREATE target existed before the step")
            continue
        if before is None:
            issues.append(f"{label}: written dataset was absent before the step")
            continue
        allow_rows = any(access.declaration.allow_row_count_change for access in writers)
        allow_schema = any(access.declaration.allow_schema_change for access in writers)
        allow_keywords = any(access.declaration.allow_keyword_change for access in writers)
        if not allow_rows and before.descriptor.nrows != after.descriptor.nrows:
            issues.append(f"{label}: MAIN row count changed {before.descriptor.nrows} -> {after.descriptor.nrows} without allow_row_count_change")
        added = set(after.descriptor.columns) - set(before.descriptor.columns)
        removed = set(before.descriptor.columns) - set(after.descriptor.columns)
        if (added or removed) and not allow_schema:
            issues.append(
                f"{label}: MAIN columns changed without allow_schema_change (added: {', '.join(sorted(added)) or 'none'}; removed: {', '.join(sorted(removed)) or 'none'})"
            )
        elif allow_schema and all(access.declaration.columns is not None for access in writers):
            if added - main_create:
                issues.append(f"{label}: undeclared column(s) created: {', '.join(sorted(added - main_create))}")
            if removed - main_remove:
                issues.append(f"{label}: undeclared column(s) removed: {', '.join(sorted(removed - main_remove))}")
        subtables_changed = set(before.descriptor.subtables) ^ set(after.descriptor.subtables)
        if subtables_changed and not allow_schema:
            issues.append(f"{label}: subtable set changed without allow_schema_change")
        before_members = {str(resource.path.relative_to(root)) for resource in before.closure.resources if resource.path.is_relative_to(root)}
        after_members = {str(resource.path.relative_to(root)) for resource in after.closure.resources if resource.path.is_relative_to(root)}
        if before_members != after_members and not allow_schema:
            issues.append(f"{label}: closure membership changed without allow_schema_change")
        # A subtable is linked by a MAIN keyword of the same name, so that
        # part of a keyword change is the schema change judged above.
        keywords_changed = (set(before.descriptor.keywords) ^ set(after.descriptor.keywords)) - subtables_changed
        if keywords_changed and not allow_keywords:
            issues.append(f"{label}: MAIN keyword name(s) changed without allow_keyword_change: {', '.join(sorted(keywords_changed))}")
        if not any(access.fallback is DatasetFallback.UNDECLARED for access in writers):
            declared_tables = {access.declaration.table.value for access in writers}
            touched = sorted(_changed_tables(before, after) - declared_tables)
            if touched:
                issues.append(f"{label}: undeclared table(s) changed: {', '.join(touched)} (declared: {', '.join(sorted(declared_tables))})")
    return issues


class StrictLeaf:
    """One strict leaf's observations and records inside a mutation lifecycle.

    Created by dispatch for every leaf that carries a dataset contract. The
    call order is fixed: :meth:`decide_reuse` on a cache hit, otherwise
    :meth:`policy` for the snapshot guard, :meth:`observe_before` after the
    guard has restored the predecessor, :meth:`validate` after the tool
    exits zero, :meth:`commit` at S3 as the marker's success oracle, and
    :meth:`fail` on any other ending.
    """

    def __init__(self, lifecycle: DatasetLifecycle, step_path: str, scope: Scope, prepared: dict[str, Any]):
        self.lifecycle = lifecycle
        self.step_path = step_path
        self.scope = scope
        workspace = lifecycle.workspace
        self.accesses = resolve_scope_dataset_accesses(scope, prepared, workspace=workspace)
        claim = lifecycle.record.claim
        if claim is None:
            raise DatasetLifecycleUnavailableError(f"strict leaf {step_path!r} has no workflow claim")
        uncovered = claim_covers_accesses(claim, self.accesses)
        if uncovered:
            raise DatasetLifecycleUnavailableError(f"strict leaf {step_path!r} needs dataset resources its workflow claim does not cover: {', '.join(map(str, uncovered))}")
        self.fields, self.creates = strict_mutation_targets(scope, self.accesses)
        self.roots = tuple(sorted({access.root for access in self.accesses if access.root is not None}, key=str))
        self.cache_key: str | None = None
        self.pre: dict[Path, DatasetObservation] = {}
        self.absent: tuple[Path, ...] = ()
        self.cache: DatasetCacheDecision | None = None
        self.mutations: dict[str, DatasetMutationRecord] = {}

    @property
    def writes(self) -> bool:
        return bool(self.fields)

    def policy(self) -> Any:
        from shinobi.snapshots import StrictMutation

        workspace = self.lifecycle.workspace
        return StrictMutation(signature=lambda path: dataset_signature(path, workspace), create_fields=self.creates)

    def coverage(self, input_keys: dict[str, Any] | None) -> tuple[str, ...]:
        """How each dataset field entered the cache key -- stated, not implied."""

        from shinobi.steps.schema import mutated_path_fields

        mutated = mutated_path_fields(self.scope)
        keys = input_keys or {}
        lines = []
        for access in self.accesses:
            if access.field in mutated or access.mode is DatasetMode.CREATE:
                lines.append(
                    f"{access.field}: {access.mode.value} target; path string only (content excluded from the key), "
                    "its state tracked by the snapshot journal with a structural signature"
                )
            elif access.field in keys:
                lines.append(f"{access.field}: wired; identified by producer lineage {keys[access.field]}")
            else:
                lines.append(f"{access.field}: unwired boundary; path plus per-file (relative path, mtime_ns, size) fingerprint")
        return tuple(dict.fromkeys(lines))

    def decide_reuse(self, cache_key: str, input_keys: dict[str, Any] | None, journal: Any) -> bool:
        """Accept or reject a skip-cache hit, recording which and why."""

        from shinobi.snapshots import state_name, strict_reuse_issue

        self.cache_key = cache_key
        problems = []
        for field, root in sorted(self.fields.items()):
            try:
                signature = dataset_signature(root, self.lifecycle.workspace)
            except DatasetLifecycleUnavailableError as exc:
                problems.append(f"{field}: {exc}")
                continue
            issue = strict_reuse_issue(journal, root, state_name(cache_key, field), signature)
            if issue is not None:
                problems.append(f"{field}: {issue}")
        accepted = not problems
        self.cache = DatasetCacheDecision(
            decision="hit" if accepted else "rejected-hit",
            cache_key=cache_key,
            identity="skip-cache key: tool identity, effective parameters and upstream provenance",
            dataset_coverage=self.coverage(input_keys),
            reason="skip-cache hit; the journal vouches for every written dataset" if accepted else "; ".join(problems),
        )
        self._record("reused" if accepted else "pending", self.cache.reason)
        return accepted

    def decide_run(self, cache_key: str | None, input_keys: dict[str, Any] | None, *, cacheable: bool) -> None:
        self.cache_key = cache_key
        if self.cache is not None and self.cache.decision == "rejected-hit":
            return
        self.cache = DatasetCacheDecision(
            decision="miss" if cacheable else "disabled",
            cache_key=cache_key,
            identity="skip-cache key: tool identity, effective parameters and upstream provenance" if cacheable else "none",
            dataset_coverage=self.coverage(input_keys),
            reason="no reusable entry for this key" if cacheable else "caching is disabled for this step",
        )

    def observe_before(self, guard: Any | None) -> None:
        """The predecessor, observed after any restore and before launch."""

        from shinobi.snapshots import state_name

        observations, absent = observe_roots(self.roots, self.lifecycle.workspace)
        self.pre = {observation.root: observation for observation in observations}
        self.absent = absent
        unexpected = sorted(set(absent) - {self.fields[field] for field in self.creates}, key=str)
        if unexpected:
            raise DatasetLifecycleUnavailableError(f"strict leaf {self.step_path!r}: dataset(s) absent before launch: {', '.join(map(str, unexpected))}")
        plans = {plan.field: plan for plan in guard.plans} if guard is not None else {}
        for field, root in sorted(self.fields.items()):
            plan = plans.get(field)
            required = plan.required if plan is not None else None
            before = self.pre.get(root)
            assert self.cache_key is not None
            self.mutations[field] = DatasetMutationRecord(
                field=field,
                mode=DatasetMode.CREATE if field in self.creates else DatasetMode.WRITE,
                root=root,
                predecessor_state=required,
                predecessor_signature=structural_signature(before) if before is not None else None,
                predecessor_snapshot=guard.journal.snapshot_dir(required) if guard is not None and required is not None else None,
                successor_state=state_name(self.cache_key, field),
            )
        self._record("pending", "predecessor observed under the workflow claim")

    def validate(self, guard: Any | None) -> None:
        """Declared postconditions; hands successor signatures to the guard."""

        from shinobi.exceptions import DatasetLifecycleViolationError

        post: dict[Path, DatasetObservation | None] = {}
        failures: list[str] = []
        for root in self.roots:
            if not root.exists() and not root.is_symlink():
                post[root] = None
                continue
            try:
                post[root] = _observe_root(root, self.lifecycle.workspace)
            except DatasetLifecycleUnavailableError as exc:
                failures.append(f"{root.name or root}: {exc}")
                post[root] = None
        issues = failures + leaf_postcondition_issues(self.accesses, self.pre, post)
        observed = tuple(observation for observation in post.values() if observation is not None)
        if issues:
            reason = "postcondition failed: " + "; ".join(issues)
            self._record("failed", reason, post=observed)
            raise DatasetLifecycleViolationError(f"strict MSv2 step {self.step_path!r} broke its declared contract: " + "; ".join(issues))
        for field, root in self.fields.items():
            observation = post[root]
            assert observation is not None
            signature = structural_signature(observation)
            if guard is not None:
                guard.successor_signatures[field] = signature
            record = self.mutations[field]
            self.mutations[field] = record.model_copy(
                update={"successor_signature": signature, "successor_snapshot": guard.journal.snapshot_dir(record.successor_state) if guard is not None else None}
            )
        self._post = observed
        self._record("validated", "declared postconditions hold", post=observed)

    def commit(self) -> None:
        for field, record in self.mutations.items():
            self.mutations[field] = record.model_copy(update={"outcome": DatasetMutationOutcome.COMMITTED})
        self._record("committed", "successor snapshotted and journalled; this record is the success oracle", post=getattr(self, "_post", ()))

    def fail(self, guard: Any | None, reason: str, *, refused: bool = False) -> None:
        plans = {plan.field: plan for plan in guard.plans} if guard is not None else {}
        for field, record in self.mutations.items():
            plan = plans.get(field)
            outcome = DatasetMutationOutcome.REFUSED if refused else DatasetMutationOutcome(plan.outcome) if plan is not None and plan.outcome != "pending" else record.outcome
            self.mutations[field] = record.model_copy(update={"outcome": outcome})
        self._record("refused" if refused else "failed", reason, post=getattr(self, "_post", ()))

    def _record(self, outcome: str, reason: str, *, post: tuple[DatasetObservation, ...] = ()) -> None:
        self.lifecycle.record_leaf(
            DatasetLeafAttempt(
                step_path=self.step_path,
                accesses=self.accesses,
                cache=self.cache,
                pre_observations=tuple(self.pre.values()),
                absent_before=self.absent,
                post_observations=post,
                mutations=tuple(self.mutations[field] for field in sorted(self.mutations)),
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
            )
        )


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
    "DATASET_MUTATION_CAPABILITY",
    "DATASET_READ_CAPABILITY",
    "STRUCTURAL_SIGNATURE_COVERAGE",
    "STRUCTURAL_SIGNATURE_VERSION",
    "DatasetCacheDecision",
    "DatasetFileObservation",
    "DatasetLeafAttempt",
    "DatasetLifecycle",
    "DatasetLifecycleAttempt",
    "DatasetLifecycleEvent",
    "DatasetLifecyclePhase",
    "DatasetLifecycleSnapshot",
    "DatasetLifecycleStore",
    "DatasetMutationOutcome",
    "DatasetMutationRecord",
    "DatasetObservation",
    "LeafMutationError",
    "StrictLeaf",
    "claim_covers_accesses",
    "claim_covers_snapshot",
    "dataset_attempt_path",
    "dataset_signature",
    "leaf_postcondition_issues",
    "mutation_committed",
    "observe_dataset",
    "observe_roots",
    "pending_dataset_recovery",
    "read_dataset_attempt",
    "resolve_lifecycle_snapshot",
    "strict_mutation_targets",
    "structural_signature",
]
