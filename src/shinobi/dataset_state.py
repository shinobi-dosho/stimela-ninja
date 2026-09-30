"""Explicit local, contained MSv2 state export and fresh materialization.

This store is independent of the skip cache and mutation snapshots. A state is
named by msutils' native logical ID; MSv4 and native Zarr payloads are physical
representations of that state, never runnable datasets. Only exact-logical
fidelity is supported. Ownership is cooperative, and v1 requires local Linux
storage; it is not a worker transport or a hostile-filesystem security boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from shinobi import _state_adapter as adapter
from shinobi.dataset_lifecycle import DatasetObservation, member_fingerprint, observe_dataset, structural_signature
from shinobi.exceptions import ShinobiError
from shinobi.ownership import acquire_workspace, inspect_ownership, ownership_registry, release_workspace
from shinobi.storage import JsonFileStore, SharedFileLock, publish_directory, sync_directory, sync_directory_chain
from shinobi.steps.schema import paths_overlap

LogicalID = Annotated[str, Field(pattern=r"^msutils-logical-hash/v1:[0-9a-f]{64}$")]
RepresentationID = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
MAX_METADATA = 64 * 1024 * 1024
_MSV4_SCHEMA = adapter.MSV4_SCHEMA
NativeModel = Literal["msutils-native-model/v1"]
LogicalHash = Literal["msutils-logical-hash/v1"]
StateProfile = Literal["fixed-shape-defined-or-empty/v1"]
StructuralProfile = Literal["msv2-structural/v1"]
ClosureProfile = Literal["msv2-dataset-closure/v1"]
StateFidelity = Literal["exact-logical"]
MappingProfile = Literal["shinobi-xarray-ms-native/v1"]
PreservationSchema = Literal["msutils-native-preservation/v2"]
StateEvidence = Literal[
    "native-logical-id",
    "msv4-logical-id",
    "payload-logical-id",
    "structural-signature",
    "physical-inventory",
    "zarr-metadata",
]
_STATE_EVIDENCE: tuple[StateEvidence, ...] = (
    "native-logical-id",
    "msv4-logical-id",
    "payload-logical-id",
    "structural-signature",
    "physical-inventory",
    "zarr-metadata",
)


def _complete_state_stack(value: dict[str, str]) -> dict[str, str]:
    required = {*adapter.PACKAGES, "msutils_commit"}
    if set(value) != required:
        missing, extra = sorted(required - set(value)), sorted(set(value) - required)
        raise ValueError(f"state stack keys disagree with the qualified contract (missing={missing}, extra={extra})")
    if any(not version for version in value.values()):
        raise ValueError("state stack versions must be non-empty")
    return value


StateStack = Annotated[dict[str, str], AfterValidator(_complete_state_stack)]


class StateError(ShinobiError):
    """Stable refusal code plus any upstream table/column/row diagnostics."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(f"{code}: {message}")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LogicalState(_Record):
    """Logical identity only; no paths, dates, chunking or implementation versions."""

    schema_version: Literal["shinobi-logical-state/v1"] = "shinobi-logical-state/v1"
    state_id: LogicalID
    native_model: NativeModel = "msutils-native-model/v1"
    hash_algorithm: LogicalHash = "msutils-logical-hash/v1"
    profile: StateProfile = "fixed-shape-defined-or-empty/v1"
    structural_profile: StructuralProfile = "msv2-structural/v1"
    closure_profile: ClosureProfile = "msv2-dataset-closure/v1"
    fidelity: StateFidelity = "exact-logical"


class FileEntry(_Record):
    path: str
    kind: Literal["file", "directory"]
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$|^$")

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or value == "." or path.is_absolute() or ".." in path.parts or str(path) != value or "\\" in value:
            raise ValueError("ledger paths must be normalized relative POSIX paths")
        return value

    @model_validator(mode="after")
    def correct_kind(self):
        if (self.kind == "directory" and (self.size or self.sha256)) or (self.kind == "file" and not self.sha256):
            raise ValueError("invalid file/directory ledger entry")
        return self


class StateRepresentation(_Record):
    """Physical inventory and provenance, cryptographically bound to a native state.

    zarr_metadata retains each complete v3 node document as JSON text, including
    non-finite fill values, without lossy JSON-to-Pydantic float normalization.
    It records actual chunks, shards, codecs, dimensions and exporter attributes.
    """

    schema_version: Literal["shinobi-state-representation/v1"] = "shinobi-state-representation/v1"
    representation_id: RepresentationID
    state_id: LogicalID
    msv4_id: LogicalID
    payload_id: LogicalID
    preservation_schema: PreservationSchema = "msutils-native-preservation/v2"
    adapter: MappingProfile = adapter.ADAPTER
    msv4_schema: Literal[_MSV4_SCHEMA] = _MSV4_SCHEMA
    versions: StateStack
    source_spelling: str
    source: Path
    observation: DatasetObservation
    structural_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    block_rows: int | None = Field(default=None, ge=1)
    export_warnings: tuple[str, ...] = ()
    capture_attempt: uuid.UUID
    captured_at: datetime
    lineage: Literal["native-msv2-to-xarray-ms-plus-native-preservation/v1"] = "native-msv2-to-xarray-ms-plus-native-preservation/v1"
    files: tuple[FileEntry, ...]
    zarr_metadata: dict[str, str]

    @model_validator(mode="after")
    def ordered_files(self):
        names = [entry.path for entry in self.files]
        if names != sorted(set(names)):
            raise ValueError("ledger must be sorted with unique paths")
        if not self.source.is_absolute() or self.source != self.observation.root:
            raise ValueError("source observation disagrees with canonical source")
        if self.captured_at.tzinfo is None:
            raise ValueError("capture date must be timezone-aware")
        return self


class StateResult(_Record):
    state_id: LogicalID
    representation_id: RepresentationID
    fidelity: StateFidelity = "exact-logical"
    physical_restoration: Literal[False] = False
    verified: bool
    destination: Path | None = None
    attempt: Path | None = None


class StateProvenance(_Record):
    """Closed reusable-state evidence attached to one state operation.

    The representation manifest remains the authoritative physical inventory;
    this deliberately small record freezes the compatibility and fidelity
    decision needed to interpret an attempt without conflating the logical
    state, its selected representation, or the native reconstruction sidecar.
    """

    schema_version: Literal["shinobi-state-provenance/v1"] = "shinobi-state-provenance/v1"
    state_id: LogicalID
    representation_id: RepresentationID
    msv4_id: LogicalID
    payload_id: LogicalID
    native_model: NativeModel = "msutils-native-model/v1"
    hash_algorithm: LogicalHash = "msutils-logical-hash/v1"
    profile: StateProfile = "fixed-shape-defined-or-empty/v1"
    structural_profile: StructuralProfile = "msv2-structural/v1"
    closure_profile: ClosureProfile = "msv2-dataset-closure/v1"
    mapping_profile: MappingProfile = adapter.ADAPTER
    msv4_schema: Literal[_MSV4_SCHEMA] = _MSV4_SCHEMA
    preservation_schema: PreservationSchema = "msutils-native-preservation/v2"
    producer_versions: StateStack
    requested_fidelity: StateFidelity = "exact-logical"
    actual_fidelity: StateFidelity | None = None
    physical_restoration: Literal[False] = False
    reconstruction: Literal["native-zarr-preservation-sidecar/v2"] = "native-zarr-preservation-sidecar/v2"
    cache_decision: Literal["store-requested", "selected-representation"]
    materialization_decision: Literal["not-requested", "requested", "validated"]
    replay_decision: Literal["not-requested", "requested", "validated"]
    transformation_decision: Literal["not-requested"] = "not-requested"
    structural_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[StateEvidence, ...] = _STATE_EVIDENCE

    @classmethod
    def from_representation(
        cls,
        representation: "StateRepresentation",
        *,
        operation: Literal["export", "materialize"],
    ) -> "StateProvenance":
        export = operation == "export"
        return cls(
            state_id=representation.state_id,
            representation_id=representation.representation_id,
            msv4_id=representation.msv4_id,
            payload_id=representation.payload_id,
            mapping_profile=representation.adapter,
            msv4_schema=representation.msv4_schema,
            preservation_schema=representation.preservation_schema,
            producer_versions=representation.versions,
            actual_fidelity="exact-logical" if export else None,
            cache_decision="store-requested" if export else "selected-representation",
            materialization_decision="not-requested" if export else "requested",
            replay_decision="not-requested" if export else "requested",
            structural_signature=representation.structural_signature,
        )

    def validated_replay(self) -> "StateProvenance":
        """Record proven reconstruction separately from its earlier request."""

        if self.materialization_decision != "requested" or self.replay_decision != "requested":
            raise ValueError("only a requested materialization/replay can become validated")
        return type(self).model_validate(
            {
                **self.model_dump(),
                "actual_fidelity": "exact-logical",
                "materialization_decision": "validated",
                "replay_decision": "validated",
            }
        )

    @model_validator(mode="after")
    def complete_evidence(self):
        if self.evidence != _STATE_EVIDENCE:
            raise ValueError("exact reusable-state provenance requires complete, unique evidence coverage")
        return self


class StateAttempt(_Record):
    schema_version: Literal["shinobi-state-attempt/v1", "shinobi-state-attempt/v2"] = "shinobi-state-attempt/v2"
    attempt_id: uuid.UUID
    operation: Literal["export", "materialize"]
    store: Path
    authority: Path
    registry: Path
    source: Path | None = None
    destination: Path | None = None
    requested_state_id: str | None = None
    requested_representation_id: str | None = None
    requested_fidelity: str = "exact-logical"
    state_id: LogicalID | None = None
    representation_id: RepresentationID | None = None
    stage: Path
    stage_identity: tuple[int, int] | None = None
    parent_identity: tuple[int, int]
    candidate_identity: tuple[int, int] | None = None
    versions: dict[str, str] | None = None
    fidelity: StateFidelity = "exact-logical"
    physical_restoration: Literal[False] = False
    phase: Literal["planned", "claimed", "staged", "writing", "validating", "ready", "published", "committed", "refused", "failed", "interrupted"] = "planned"
    finished: bool = False
    events: tuple[str, ...] = ()
    error: str | None = None
    structural_signature: str | None = None
    provenance: StateProvenance | None = None

    @model_validator(mode="after")
    def consistent_provenance(self):
        if self.schema_version == "shinobi-state-attempt/v1":
            if self.provenance is not None:
                raise ValueError("state attempt schema v1 cannot carry reusable-state provenance")
            return self
        if self.versions is not None:
            _complete_state_stack(self.versions)
        if self.phase == "committed":
            if self.provenance is None or self.versions is None:
                raise ValueError("a committed state attempt requires reusable-state provenance and a complete runtime stack")
            if self.structural_signature is None or self.structural_signature != self.provenance.structural_signature:
                raise ValueError("a committed state attempt requires its matching structural signature")
            if self.operation == "materialize" and (
                self.provenance.actual_fidelity != "exact-logical" or self.provenance.materialization_decision != "validated" or self.provenance.replay_decision != "validated"
            ):
                raise ValueError("a committed materialization requires validated exact replay provenance")
        if self.provenance is None:
            return self
        evidence = self.provenance
        if self.state_id != evidence.state_id or self.representation_id != evidence.representation_id:
            raise ValueError("state attempt and reusable-state provenance identities disagree")
        if self.operation == "export" and (
            evidence.cache_decision != "store-requested" or evidence.materialization_decision != "not-requested" or evidence.replay_decision != "not-requested"
        ):
            raise ValueError("reusable-state provenance decisions disagree with export")
        if self.operation == "materialize" and (
            evidence.cache_decision != "selected-representation" or evidence.materialization_decision == "not-requested" or evidence.replay_decision == "not-requested"
        ):
            raise ValueError("reusable-state provenance decisions disagree with materialization")
        if self.structural_signature is not None and self.structural_signature != evidence.structural_signature:
            raise ValueError("state attempt and reusable-state provenance structural signatures disagree")
        return self


def read_state_attempt(path: str | Path) -> StateAttempt:
    """Read and strictly validate a durable reusable-state attempt record."""

    return StateAttempt.model_validate(JsonFileStore(Path(path)).read())


def _error(exc: Exception) -> StateError:
    if isinstance(exc, StateError):
        return exc
    if isinstance(exc, ImportError):
        return StateError("state-stack", str(exc))
    if isinstance(exc, ValidationError):
        return StateError("state-contract", str(exc))
    details = {key: getattr(exc, key) for key in ("table", "column", "row", "reason") if getattr(exc, key, None) is not None}
    return StateError(getattr(exc, "code", "state-operation"), str(exc), details=details)


def _digest(value: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _directory_identity(path: Path, *, missing_ok: bool = False) -> tuple[int, int] | None:
    """Inspect the entry itself: a file or symlink is never our published MS."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None


def _identity(path: Path) -> tuple[int, int]:
    identity = _directory_identity(path)
    if identity is None:
        raise StateError("path-kind", f"expected a real directory: {path}")
    return identity


def _read(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA:
        raise StateError("metadata", f"expected a regular metadata file of at most {MAX_METADATA} bytes: {path}")
    return path.read_bytes()


def _write(path: Path, model: BaseModel) -> None:
    # This is inside private staging; publication is one no-replace directory move.
    payload = model.model_dump_json(indent=2).encode()
    if len(payload) > MAX_METADATA:
        raise StateError("metadata-size", f"manifest exceeds {MAX_METADATA} bytes: {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _manifest(path: Path, model):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StateError("metadata", f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    data = json.loads(_read(path), object_pairs_hook=unique)
    if not isinstance(data, dict) or "schema_version" not in data:
        raise StateError("schema-version", f"missing explicit manifest version: {path}")
    return model.model_validate(data)


def _inventory(root: Path) -> tuple[tuple[FileEntry, ...], dict[str, str]]:
    """A complete physical ledger, including otherwise invisible empty directories."""
    _identity(root)
    entries, nodes = [], {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if relative == "representation.json":
                continue
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                entries.append(FileEntry(path=relative, kind="directory", size=0, sha256=""))
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                entries.append(FileEntry(path=relative, kind="file", size=info.st_size, sha256=checksum))
                if name == "zarr.json":
                    raw = _read(path).decode()
                    metadata = json.loads(raw)
                    if metadata.get("zarr_format") != 3:
                        raise StateError("zarr-version", f"expected Zarr v3: {path}")
                    nodes[relative] = raw
            else:
                raise StateError("entry-kind", f"symlink or special file refused: {path}")
    return tuple(sorted(entries, key=lambda item: item.path)), dict(sorted(nodes.items()))


def _sync_tree(root: Path) -> None:
    for parent, _dirs, files in os.walk(root, topdown=False):
        for name in files:
            fd = os.open(Path(parent) / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        sync_directory(Path(parent))


class DatasetStateStore:
    """A local immutable state store with explicit, recoverable operations.

    Paths are local filesystem paths. Source and destination parent directories
    must exist. Stored state references are deliberately not accepted by Recipe.
    ``cache_dir`` is used only to refuse an unfinished Tier 1 mutation on export.
    """

    def __init__(self, directory: str | Path = ".shinobi/states", *, cache_dir: str | Path = ".shinobi/cache"):
        self.root = Path(directory).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if paths_overlap(self.root, self.cache_dir):
            raise StateError("store-cache-overlap", "the reusable state store must be separate from the skip cache")

    def _setup(self):
        for name in ("states", "staging", "attempts", "locks"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
            _identity(self.root / name)
        sync_directory_chain(self.root)

    def _state(self, state_id: str) -> Path:
        if not re.fullmatch(r"msutils-logical-hash/v1:[0-9a-f]{64}", state_id):
            raise StateError("state-id", "expected msutils-logical-hash/v1:<64 lowercase hex digits>")
        return self.root / "states" / state_id.split(":")[1]

    def _representation(self, state_id: str, representation_id: str | None) -> Path:
        state = self._state(state_id)
        try:
            _identity(state)
        except FileNotFoundError as exc:
            raise StateError("state-not-found", f"no committed state {state_id}") from exc
        directory = state / "representations"
        _identity(directory)
        if representation_id is None:
            candidates = sorted(directory.iterdir())
            if not candidates:
                raise StateError("incomplete", "state has no committed representation")
            representation_id = "sha256:" + candidates[0].name
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", representation_id):
            raise StateError("representation-id", "expected sha256:<64 lowercase hex digits>")
        result = directory / representation_id.split(":")[1]
        try:
            _identity(result)
        except FileNotFoundError as exc:
            raise StateError("representation-not-found", f"no committed representation {representation_id} for {state_id}") from exc
        return result

    def _load(self, state_id: str, representation_id: str | None = None):
        state = self._state(state_id)
        rep = self._representation(state_id, representation_id)
        logical = _manifest(state / "logical.json", LogicalState)
        physical = _manifest(rep / "representation.json", StateRepresentation)
        if logical.state_id != state_id or physical.state_id != state_id or physical.representation_id != "sha256:" + rep.name:
            raise StateError("identity", "directory and manifest identities disagree")
        if _digest(physical.model_dump(mode="json", exclude={"representation_id"})) != physical.representation_id:
            raise StateError("representation-digest", "representation manifest changed")
        if {p.name for p in state.iterdir()} != {"logical.json", "representations"}:
            raise StateError("extra-entry", "unexpected entry in logical state directory")
        return rep, logical, physical

    def list(self) -> list[StateResult]:
        """List committed manifest identities; this does not verify stored data."""
        if not (self.root / "states").exists():
            return []
        result = []
        try:
            for path in sorted((self.root / "states").iterdir()):
                state_id = "msutils-logical-hash/v1:" + path.name
                for rep in sorted((path / "representations").iterdir()):
                    _, _, physical = self._load(state_id, "sha256:" + rep.name)
                    result.append(StateResult(state_id=state_id, representation_id=physical.representation_id, verified=False))
        except Exception as exc:
            raise _error(exc) from exc
        return result

    def verify(self, state_id: str, representation_id: str | None = None) -> StateResult:
        """Fully verify contracts, physical bytes and both public logical roots."""
        try:
            versions = adapter.runtime()
            rep, _logical, physical = self._load(state_id, representation_id)
            if physical.versions != versions:
                raise StateError("stack-version", "representation requires the exact recorded implementation stack")
            self._verify_rep(rep, physical)
            return StateResult(state_id=state_id, representation_id=physical.representation_id, verified=True)
        except Exception as exc:
            raise _error(exc) from exc

    @staticmethod
    def _verify_rep(rep: Path, physical: StateRepresentation):
        files, nodes = _inventory(rep)
        if files != physical.files or nodes != physical.zarr_metadata:
            raise StateError("physical-integrity", "representation inventory, bytes or Zarr layout changed")
        ids = adapter.verify(rep / "msv4.zarr", rep / "preservation")
        if (ids.native_logical_id, ids.msv4_logical_id, ids.payload_logical_id) != (physical.state_id, physical.msv4_id, physical.payload_id):
            raise StateError("logical-integrity", "verified native/MSv4/payload roots disagree with the state manifest")
        if structural_signature(physical.observation) != physical.structural_signature:
            raise StateError("structural-signature", "recorded source signature disagrees with observation")

    def _save(self, attempt: StateAttempt, **changes) -> StateAttempt:
        if "phase" in changes:
            changes["events"] = (*attempt.events, f"{datetime.now(timezone.utc).isoformat()} {changes['phase']}")
        updated = StateAttempt.model_validate({**attempt.model_dump(), **changes})
        JsonFileStore(self._attempt_path(attempt.attempt_id)).update(lambda data: data.update(updated.model_dump(mode="json")))
        return updated

    def _attempt_path(self, attempt_id: uuid.UUID) -> Path:
        return self.root / "attempts" / f"{attempt_id}.json"

    def _lock(self, kind: str, identity: str, *, timeout: float = 60):
        key = hashlib.sha256(identity.encode()).hexdigest()
        return SharedFileLock(self.root / "locks" / kind / key, lock_timeout=timeout)

    def _cleanup(self, attempt: StateAttempt):
        stage = attempt.stage
        expected_parent = self.root / "staging" if attempt.operation == "export" else attempt.destination.parent
        expected_name = str(attempt.attempt_id) if attempt.operation == "export" else f".shinobi-state-{attempt.attempt_id}"
        if stage.parent != expected_parent or stage.name != expected_name or _identity(expected_parent) != attempt.parent_identity:
            raise StateError("recovery-path", "staging parent/name changed; preserve for manual inspection")
        if os.path.lexists(stage):
            if attempt.stage_identity is None or _identity(stage) != attempt.stage_identity:
                raise StateError("recovery-identity", "staging identity changed or was never committed; preserve for manual inspection")
            shutil.rmtree(stage)
            sync_directory(stage.parent)

    def _pending_snapshot(self, source: Path):
        from shinobi.snapshots import get_journal

        for chain in get_journal(str(self.cache_dir)).all_chains().values():
            if chain.marker is not None and paths_overlap(source, Path(chain.path)):
                raise StateError("pending-mutation", "source has an interrupted mutation; recover it through its workflow before exporting")

    def export(self, source: str | Path, *, block_rows: int | None = None) -> StateResult:
        """Capture a supported contained MS under a shared READ ownership claim."""
        attempt = lease = None
        try:
            if block_rows is not None and (type(block_rows) is not int or block_rows < 1):
                raise StateError("block-rows", "block_rows must be a positive integer")
            versions = adapter.runtime()
            spelling = str(Path(source).expanduser().absolute())
            source = Path(source).expanduser().resolve(strict=True)
            if paths_overlap(source, self.root) or paths_overlap(source, self.cache_dir):
                raise StateError("source-overlap", "source overlaps state or cache storage")
            observed = observe_dataset(source, source.parent)
            self._setup()
            operation = uuid.uuid4()
            stage = self.root / "staging" / str(operation)
            attempt = StateAttempt(
                attempt_id=operation,
                operation="export",
                store=self.root,
                authority=source.parent,
                registry=ownership_registry(),
                source=source,
                stage=stage,
                parent_identity=_identity(stage.parent),
                versions=versions,
            )
            with self._lock("attempts", str(operation)):
                attempt = self._save(attempt, phase="planned")
                lease = acquire_workspace(source.parent, str(operation), kind="local", accesses=[(source, False)], registry=attempt.registry)
                attempt = self._save(attempt, phase="claimed")
                before = observe_dataset(source, source.parent)
                if structural_signature(before) != structural_signature(observed) or member_fingerprint(before) != member_fingerprint(observed):
                    raise StateError("source-changed", "source changed while acquiring ownership")
                self._pending_snapshot(source)
                state_id = adapter.native_id(source)
                stage.mkdir()
                attempt = self._save(attempt, phase="staged", stage_identity=_identity(stage), state_id=state_id)
                rep = stage / "representation"
                rep.mkdir()
                attempt = self._save(attempt, phase="writing")
                warnings = adapter.export(source, rep / "msv4.zarr")
                adapter.capture(source, rep / "msv4.zarr", rep / "preservation", block_rows)
                attempt = self._save(attempt, phase="validating")
                ids = adapter.verify(rep / "msv4.zarr", rep / "preservation")
                after = observe_dataset(source, source.parent)
                if ids.native_logical_id != state_id or adapter.native_id(source) != state_id or member_fingerprint(before) != member_fingerprint(after):
                    raise StateError("source-changed", "source changed during export/capture")
                files, nodes = _inventory(rep)
                physical = StateRepresentation(
                    representation_id="sha256:" + "0" * 64,
                    state_id=state_id,
                    msv4_id=ids.msv4_logical_id,
                    payload_id=ids.payload_logical_id,
                    versions=versions,
                    source_spelling=spelling,
                    source=source,
                    observation=before,
                    structural_signature=structural_signature(before),
                    block_rows=block_rows,
                    export_warnings=tuple(warnings),
                    capture_attempt=operation,
                    captured_at=datetime.now(timezone.utc),
                    files=files,
                    zarr_metadata=nodes,
                )
                physical = physical.model_copy(update={"representation_id": _digest(physical.model_dump(mode="json", exclude={"representation_id"}))})
                provenance = StateProvenance.from_representation(physical, operation="export")
                _write(rep / "representation.json", physical)
                self._verify_rep(rep, physical)
                publication = stage / "state"
                (publication / "representations").mkdir(parents=True)
                _write(publication / "logical.json", LogicalState(state_id=state_id))
                rep.rename(publication / "representations" / physical.representation_id.split(":")[1])
                _sync_tree(stage)
                attempt = self._save(
                    attempt,
                    phase="ready",
                    representation_id=physical.representation_id,
                    structural_signature=physical.structural_signature,
                    provenance=provenance,
                )
                with self._lock("states", state_id):
                    target = self._state(state_id)
                    if target.exists():
                        self._load(state_id)
                        publish_directory(
                            publication / "representations" / physical.representation_id.split(":")[1], target / "representations" / physical.representation_id.split(":")[1]
                        )
                    else:
                        publish_directory(publication, target)
                attempt = self._save(attempt, phase="committed")
                self._cleanup(attempt)
                lease.release()
                lease = None
                attempt = self._save(attempt, finished=True)
                return StateResult(state_id=state_id, representation_id=physical.representation_id, verified=True, attempt=self._attempt_path(operation))
        except Exception as exc:
            failure = _error(exc)
            if attempt is not None:
                # As with MS publication, a parent fsync failure can occur
                # after the no-replace move. Leave its ready intent recoverable.
                published = (
                    attempt.state_id and attempt.representation_id and (self._state(attempt.state_id) / "representations" / attempt.representation_id.split(":")[1]).exists()
                )
                if not published:
                    phase = "failed" if failure.code == "state-operation" else "refused"
                    attempt = self._save(attempt, phase=phase, error=str(failure))
                    self._cleanup(attempt)
                    if lease is not None:
                        lease.release()
                        lease = None
                    self._save(attempt, finished=True)
            raise failure from exc
        finally:
            if lease is not None:
                lease.release()

    def _target(self, destination: str | Path) -> Path:
        spelling = Path(destination).expanduser().absolute()
        parent = spelling.parent.resolve(strict=True)
        target = parent / spelling.name
        _identity(parent)
        return target

    def _destination(self, destination: str | Path, physical: StateRepresentation) -> Path:
        target = self._target(destination)
        if os.path.lexists(target):
            raise StateError("destination-exists", f"destination already exists (including symlinks): {target}")
        sources = {physical.source, Path(physical.source_spelling).resolve()}
        # Independent exports may capture the same native state at different
        # paths. Every recorded source is protected, not just the selected rep.
        for sibling in (self._state(physical.state_id) / "representations").iterdir():
            _, _, recorded = self._load(physical.state_id, "sha256:" + sibling.name)
            sources.update((recorded.source, Path(recorded.source_spelling).resolve()))
        for excluded in (self.root, self.cache_dir, *sources):
            if paths_overlap(target, excluded):
                raise StateError("destination-alias", f"destination overlaps protected source/store/cache path: {excluded}")
        return target

    def materialize(self, state_id: str, destination: str | Path, *, representation_id: str | None = None, fidelity: str = "exact-logical") -> StateResult:
        """Create a fresh native MS after independent validation, never overwrite."""
        attempt = lease = None
        try:
            self._setup()
            target = self._target(destination)
            with self._lock("destinations", str(target)):
                self._recover(target)
                operation = uuid.uuid4()
                stage = target.parent / f".shinobi-state-{operation}"
                attempt = StateAttempt(
                    attempt_id=operation,
                    operation="materialize",
                    store=self.root,
                    authority=target.parent,
                    registry=ownership_registry(),
                    destination=target,
                    requested_state_id=state_id,
                    requested_representation_id=representation_id,
                    requested_fidelity=fidelity,
                    stage=stage,
                    parent_identity=_identity(target.parent),
                )
                with self._lock("attempts", str(operation)):
                    attempt = self._save(attempt, phase="planned")
                    if fidelity != "exact-logical":
                        raise StateError("fidelity", "only exact-logical is supported; physical restoration is unavailable")
                    versions = adapter.runtime()
                    attempt = self._save(attempt, versions=versions)
                    rep, _, physical = self._load(state_id, representation_id)
                    target = self._destination(destination, physical)
                    provenance = StateProvenance.from_representation(physical, operation="materialize")
                    attempt = self._save(
                        attempt,
                        state_id=state_id,
                        representation_id=physical.representation_id,
                        provenance=provenance,
                    )
                    lease = acquire_workspace(target.parent, str(operation), kind="local", accesses=[(target, True), (rep, False), (stage, True)], registry=attempt.registry)
                    attempt = self._save(attempt, phase="claimed")
                    if self._destination(destination, physical) != target or _identity(target.parent) != attempt.parent_identity:
                        raise StateError("destination-changed", "destination parent changed while acquiring ownership")
                    self.verify(state_id, physical.representation_id)
                    stage.mkdir()
                    attempt = self._save(attempt, phase="writing", stage_identity=_identity(stage))
                    product = stage / "product.ms"
                    adapter.materialize(rep / "msv4.zarr", rep / "preservation", product, state_id)
                    attempt = self._save(attempt, phase="validating")
                    observed = observe_dataset(product, stage)
                    signature = structural_signature(observed)
                    if signature != physical.structural_signature or adapter.native_id(product) != state_id:
                        raise StateError("postcondition", "independent Shinobi structural signature/native logical ID differs from stored state")
                    self._verify_rep(rep, physical)
                    _sync_tree(product)
                    attempt = self._save(
                        attempt,
                        phase="ready",
                        candidate_identity=_identity(product),
                        structural_signature=signature,
                        provenance=provenance.validated_replay(),
                    )
                    if self._destination(destination, physical) != target or _identity(target.parent) != attempt.parent_identity:
                        raise StateError("destination-changed", "destination changed before publication")
                    publish_directory(product, target)
                    attempt = self._save(attempt, phase="published")
                    self._cleanup(attempt)
                    attempt = self._save(attempt, phase="committed")
                    lease.release()
                    lease = None
                    self._save(attempt, finished=True)
                    return StateResult(state_id=state_id, representation_id=physical.representation_id, verified=True, destination=target, attempt=self._attempt_path(operation))
        except Exception as exc:
            failure = _error(exc)
            if attempt is not None:
                # A rename may have committed even if the following fsync or
                # metadata write failed. Preserve ready intent for recovery.
                published = attempt.candidate_identity is not None and _directory_identity(attempt.destination, missing_ok=True) == attempt.candidate_identity
                if not published:
                    phase = "failed" if failure.code == "state-operation" else "refused"
                    attempt = self._save(attempt, phase=phase, error=str(failure))
                    self._cleanup(attempt)
                    if lease is not None:
                        lease.release()
                        lease = None
                    self._save(attempt, finished=True)
            raise failure from exc
        finally:
            if lease is not None:
                lease.release()

    def recover(self, destination: str | Path | None = None) -> list[StateAttempt]:
        """Settle dead attempts and sweep only their recorded private stage trees.

        A sweep is all-or-nothing: any live, uncertain or invalid attempt
        aborts it. Use ``destination`` to recover one target independently.
        """
        try:
            if destination is not None:
                spelling = Path(destination).expanduser().absolute()
                # Resolve parent aliases, but retain the recorded leaf spelling
                # even when an unrelated symlink won the publication race.
                target = spelling.parent.resolve() / spelling.name
                with self._lock("destinations", str(target)):
                    return self._recover(target)
            return self._recover(None)
        except Exception as exc:
            raise _error(exc) from exc

    def _recover(self, destination: Path | None) -> list[StateAttempt]:
        recovered = []
        if not (self.root / "attempts").exists():
            return recovered
        # JsonFileStore's synced log is authoritative; a crash or failed
        # compatibility-view refresh can leave only the persistent .lock file.
        directory = self.root / "attempts"
        paths = set(directory.glob("*.json")) | {path.with_suffix("") for path in directory.glob("*.json.lock")}
        for path in sorted(paths):
            attempt = read_state_attempt(path)
            if attempt.finished or (destination is not None and attempt.destination != destination):
                continue
            with self._lock("attempts", str(attempt.attempt_id), timeout=0.05):
                attempt = read_state_attempt(path)
                if attempt.finished:
                    continue
                if attempt.store != self.root or path != self._attempt_path(attempt.attempt_id):
                    raise StateError("recovery-record", "attempt/store identity disagrees")
                evidence = inspect_ownership(attempt.authority, str(attempt.attempt_id))
                if evidence.owner is not None and Path(evidence.owner.registry) != attempt.registry:
                    raise StateError("recovery-registry", "attempt and ownership registry disagree")
                if evidence.liveness not in {"dead", "free"}:
                    raise StateError("recovery-live", f"cannot settle {attempt.attempt_id}: {evidence.detail}")
                committed = False
                published_identity = _directory_identity(attempt.destination, missing_ok=True) if attempt.operation == "materialize" else None
                if published_identity is not None:
                    if attempt.candidate_identity is None or published_identity != attempt.candidate_identity:
                        raise StateError("recovery-destination", "destination identity changed; preserving destination and staging")
                    adapter.runtime()
                    self.verify(attempt.state_id, attempt.representation_id)
                    observed = observe_dataset(attempt.destination, attempt.destination.parent)
                    if structural_signature(observed) != attempt.structural_signature or adapter.native_id(attempt.destination) != attempt.state_id:
                        raise StateError("recovery-postcondition", "published destination no longer matches the recorded state; preserving it")
                    committed = True
                elif attempt.operation == "export" and attempt.state_id and attempt.representation_id:
                    try:
                        self._load(attempt.state_id, attempt.representation_id)
                        self.verify(attempt.state_id, attempt.representation_id)
                        committed = True
                    except StateError as exc:
                        # A ready export can be interrupted before its first
                        # publication, or before adding a new representation.
                        # Missing committed storage means the intent did not
                        # publish; every other validation failure is unsafe to
                        # classify automatically.
                        if exc.code not in {"state-not-found", "representation-not-found"}:
                            raise
                self._cleanup(attempt)
                attempt = self._save(attempt, phase="committed" if committed else "interrupted")
                if evidence.owner is not None:
                    release_workspace(attempt.authority, str(attempt.attempt_id))
                recovered.append(self._save(attempt, finished=True))
        return recovered
