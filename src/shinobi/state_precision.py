"""Explicit approximate MSv4 artifacts, separate from exact native states."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shinobi import _state_adapter as adapter
from shinobi.dataset_state import (
    FileEntry,
    StateError,
    _digest,
    _directory_identity,
    _identity,
    _inventory,
    _manifest,
    _sync_tree,
    _write,
)
from shinobi.state_codecs import bounded_chunks, tile_slices
from shinobi.storage import JsonFileStore, publish_directory, sync_directory, sync_directory_chain
from shinobi.ownership import acquire_workspace, inspect_ownership, ownership_registry, release_workspace


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrecisionPolicy(_Record):
    schema_version: Literal["shinobi-precision-policy/v1"] = "shinobi-precision-policy/v1"
    algorithm: Literal["mantissa-round-nearest-even/v1"] = "mantissa-round-nearest-even/v1"
    arrays: tuple[str, ...]
    keep_bits: int = Field(ge=0, le=51)
    atol: float = Field(ge=0, allow_inf_nan=False)
    rtol: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_paths(self):
        from pathlib import PurePosixPath

        if not self.arrays or self.arrays != tuple(sorted(set(self.arrays))):
            raise ValueError("array paths must be nonempty, unique and sorted")
        for name in self.arrays:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
                raise ValueError(f"invalid array path: {name}")
        return self


class DerivedLogical(_Record):
    schema_version: Literal["shinobi-derived-logical/v1"] = "shinobi-derived-logical/v1"
    derived_id: str = Field(pattern=r"^shinobi-derived/v1:[0-9a-f]{64}$")
    parent_state_id: str = Field(pattern=r"^msutils-logical-hash/v1:[0-9a-f]{64}$")
    parent_msv4_id: str = Field(pattern=r"^msutils-logical-hash/v1:[0-9a-f]{64}$")
    policy: PrecisionPolicy
    msv4_id: str = Field(pattern=r"^msutils-logical-hash/v1:[0-9a-f]{64}$")
    fidelity: Literal["approximate"] = "approximate"

    def expected_id(self):
        return "shinobi-derived/v1:" + _digest(self.model_dump(mode="json", exclude={"derived_id"})).split(":")[1]


class DerivedPhysical(_Record):
    schema_version: Literal["shinobi-derived-physical/v1"] = "shinobi-derived-physical/v1"
    representation_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    derived_id: str = Field(pattern=r"^shinobi-derived/v1:[0-9a-f]{64}$")
    source_representation_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    versions: dict[str, str]
    created_at: datetime
    files: tuple[FileEntry, ...]
    zarr_metadata: dict[str, str]
    changed_count: int = Field(ge=1)
    max_abs_error: float = Field(ge=0, allow_inf_nan=False)
    max_relative_error: float = Field(ge=0, allow_inf_nan=False)


class DerivedResult(_Record):
    derived_id: str
    representation_id: str
    parent_state_id: str
    msv4_id: str
    fidelity: Literal["approximate"] = "approximate"
    verified: bool
    attempt: Path | None = None


class DerivedAttempt(_Record):
    schema_version: Literal["shinobi-derived-attempt/v1"] = "shinobi-derived-attempt/v1"
    attempt_id: uuid.UUID
    store: Path
    authority: Path
    registry: Path
    parent_state_id: str
    source_representation_id: str
    policy: PrecisionPolicy
    versions: dict[str, str]
    stage: Path
    parent_identity: tuple[int, int]
    stage_identity: tuple[int, int] | None = None
    candidate_identity: tuple[int, int] | None = None
    derived_id: str | None = None
    representation_id: str | None = None
    phase: Literal["planned", "writing", "ready", "committed", "refused", "failed", "interrupted"] = "planned"
    finished: bool = False
    error: str | None = None


def _round(values, keep_bits: int):
    """Round IEEE mantissas by integer bits; untouched NaNs retain payloads."""
    import numpy as np

    if values.dtype.kind not in "fc" or values.dtype.itemsize not in (2, 4, 8, 16):
        raise StateError("precision-dtype", f"unsupported floating/complex dtype {values.dtype}")
    component = values.real.dtype if values.dtype.kind == "c" else values.dtype
    width = component.itemsize * 8
    mantissa = {16: 10, 32: 23, 64: 52}[width]
    if keep_bits >= mantissa:
        return values.copy()
    unsigned = {16: np.uint16, 32: np.uint32, 64: np.uint64}[width]
    bits = values.copy().view(component).view(unsigned)
    old = bits.copy()
    drop = mantissa - keep_bits
    mask = (1 << drop) - 1
    finite = np.repeat(np.isfinite(values), 2).reshape(bits.shape) if values.dtype.kind == "c" else np.isfinite(values)
    # Adding to the low mantissa bits performs carry into the exponent too.
    rounded = (bits + ((1 << (drop - 1)) - 1) + ((bits >> drop) & 1)) & ~np.array(mask, dtype=unsigned)
    bits[finite] = rounded[finite]
    result = bits.view(component).view(values.dtype).reshape(values.shape)
    if np.any(np.isfinite(values) & ~np.isfinite(result)):
        raise StateError("precision-overflow", "finite value rounded to infinity")
    # Preserve bitwise identity of non-finite payloads and signed zero.
    if not np.array_equal(old[~finite], bits[~finite]):
        raise StateError("precision-nonfinite", "non-finite bits changed")
    return result


def _compare(source, transformed, policy: PrecisionPolicy):
    import numpy as np

    if source.shape != transformed.shape or source.dtype != transformed.dtype:
        raise StateError("precision-shape", "transformation changed dtype or shape")
    valid = np.isfinite(source) & np.isfinite(transformed)
    old_bytes = source.copy().view(np.uint8).reshape(source.size, source.dtype.itemsize)
    new_bytes = transformed.copy().view(np.uint8).reshape(source.size, source.dtype.itemsize)
    if np.any(np.any(old_bytes != new_bytes, axis=1).reshape(source.shape) & ~valid):
        raise StateError("precision-nonfinite", "value containing a non-finite component changed")

    def magnitude(value):
        if value.dtype.kind == "c":
            parts = value.astype(np.clongdouble)
            return np.hypot(parts.real, parts.imag)
        return np.abs(value.astype(np.longdouble))

    error = np.zeros(source.shape, dtype=np.longdouble)
    error[valid] = magnitude(
        transformed[valid].astype(np.clongdouble if source.dtype.kind == "c" else np.longdouble)
        - source[valid].astype(np.clongdouble if source.dtype.kind == "c" else np.longdouble)
    )
    old_magnitude = magnitude(source[valid])
    permitted = np.longdouble(policy.atol) + np.longdouble(policy.rtol) * old_magnitude
    if np.any(error[valid] > permitted):
        raise StateError("precision-bound", "per-value absolute/relative error bound exceeded")
    relative = np.zeros(source.shape, dtype=np.longdouble)
    nonzero = valid & (source != 0)
    relative[nonzero] = error[nonzero] / magnitude(source[nonzero])
    changed = int(np.count_nonzero(np.any(old_bytes != new_bytes, axis=1)))
    maximum = float(np.max(error, initial=0))
    maximum_relative = float(np.max(relative, initial=0))
    if not np.isfinite(maximum) or not np.isfinite(maximum_relative):
        raise StateError("precision-metric", "error metric cannot be represented as a finite float")
    return changed, maximum, maximum_relative


def _transform(source: Path, target: Path, policy: PrecisionPolicy, *, verify: bool = False):
    import numpy as np
    import zarr

    src = zarr.open_group(source, mode="r", use_consolidated=False)
    dst = zarr.open_group(target, mode="r" if verify else "a", use_consolidated=False)
    changed_total, max_abs, max_rel = 0, 0.0, 0.0
    for name in policy.arrays:
        try:
            left, right = src[name], dst[name]
        except KeyError as exc:
            raise StateError("precision-path", f"array not found: {name}") from exc
        if not hasattr(left, "dtype") or not hasattr(right, "dtype"):
            raise StateError("precision-path", f"not an array: {name}")
        if left.dtype.kind not in "fc":
            raise StateError("precision-dtype", f"selected array is not floating or complex: {name}")
        roles = left.metadata.dimension_names or ()
        leaf = name.rsplit("/", 1)[-1]
        if leaf not in {"VISIBILITY", "MODEL_VISIBILITY", "CORRECTED_VISIBILITY", "WEIGHT", "WEIGHT_SPECTRUM"} or not {"time", "baseline_id", "frequency", "polarization"}.issubset(
            roles
        ):
            raise StateError("precision-role", f"selected path is not an xarray-ms visibility/weight data array: {name}")
        component_bits = (left.dtype.itemsize // (2 if left.dtype.kind == "c" else 1)) * 8
        if policy.keep_bits >= {16: 10, 32: 23, 64: 52}.get(component_bits, -1):
            raise StateError("precision-noop", f"policy does not reduce precision for {name}")
        chunks = bounded_chunks(left.shape, left.chunks, left.dtype.itemsize) if left.shape else ()
        for region in tile_slices(left.shape, chunks):
            original = np.asarray(left[region])
            expected = _round(original, policy.keep_bits)
            changed, absolute, relative = _compare(original, expected, policy)
            if verify:
                actual = np.asarray(right[region])
                if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
                    raise StateError("precision-algorithm", f"derived values differ from expected rounding: {name}")
            else:
                right[region] = expected
            changed_total += changed
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
    if changed_total == 0:
        raise StateError("precision-noop", "selected values are bitwise unchanged")
    return changed_total, max_abs, max_rel


def _verify_whole_tree(source: Path, target: Path, selected: tuple[str, ...]) -> None:
    """Check all nodes/attributes and untouched values against the exact parent."""
    import numpy as np
    import zarr

    src = zarr.open_group(source, mode="r", use_consolidated=False)
    dst = zarr.open_group(target, mode="r", use_consolidated=False)

    def visit(left, right, prefix=""):
        if left.attrs.asdict() != right.attrs.asdict():
            raise StateError("derived-attributes", f"group attributes changed: {prefix}")
        if {name for name, _ in left.members()} != {name for name, _ in right.members()}:
            raise StateError("derived-nodes", f"node names changed: {prefix}")
        for name, node in left.members():
            other = right[name]
            path = f"{prefix}/{name}".lstrip("/")
            if hasattr(node, "dtype") != hasattr(other, "dtype"):
                raise StateError("derived-nodes", f"node kind changed: {path}")
            if not hasattr(node, "dtype"):
                visit(node, other, path)
                continue
            if (node.dtype, node.shape, node.metadata.dimension_names, node.attrs.asdict()) != (other.dtype, other.shape, other.metadata.dimension_names, other.attrs.asdict()):
                raise StateError("derived-schema", f"array schema changed: {path}")
            if path in selected:
                continue
            chunks = bounded_chunks(node.shape, node.chunks, max(1, node.dtype.itemsize)) if node.shape else ()
            for region in tile_slices(node.shape, chunks):
                before, after = np.asarray(node[region]), np.asarray(other[region])
                equal = np.array_equal(before.view(np.uint8), after.view(np.uint8)) if not node.dtype.hasobject else np.array_equal(before, after)
                if not equal:
                    raise StateError("derived-untouched", f"unselected array changed: {path}")

    visit(src, dst)


class DerivedStateStore:
    """Derived namespace attached to an exact DatasetStateStore."""

    def __init__(self, exact):
        self.exact = exact
        self.root = exact.root

    @staticmethod
    def _real_directory(path: Path, *, missing_ok: bool = False) -> bool:
        """Reject symlink parents before any derived read or publication."""
        identity = _directory_identity(path, missing_ok=True)
        if identity is not None:
            return True
        if os.path.lexists(path):
            raise StateError("derived-path-kind", f"expected a real directory: {path}")
        if not missing_ok:
            raise StateError("derived-missing", f"derived directory is missing: {path}")
        return False

    def _paths(self, derived_id: str, representation_id: str | None = None, *, missing_ok: bool = False):
        if not re.fullmatch(r"shinobi-derived/v1:[0-9a-f]{64}", derived_id):
            raise StateError("derived-id", "expected shinobi-derived/v1:<64 lowercase hex digits>")
        self._real_directory(self.root, missing_ok=missing_ok)
        directory = self.root / "derived"
        self._real_directory(directory, missing_ok=missing_ok)
        base = directory / derived_id.split(":")[1]
        self._real_directory(base, missing_ok=missing_ok)
        representations = base / "representations"
        if self._real_directory(representations, missing_ok=missing_ok):
            for member in representations.iterdir():
                self._real_directory(member)
        if representation_id is None:
            candidates = sorted(representations.iterdir())
            if not candidates:
                raise StateError("derived-missing", "no derived representation")
            representation_id = "sha256:" + candidates[0].name
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", representation_id):
            raise StateError("representation-id", "invalid representation ID")
        rep = representations / representation_id.split(":")[1]
        self._real_directory(rep, missing_ok=missing_ok)
        return base, rep

    def _load(self, derived_id: str, representation_id: str | None = None):
        base, rep = self._paths(derived_id, representation_id)
        logical = _manifest(base / "logical.json", DerivedLogical)
        physical = _manifest(rep / "representation.json", DerivedPhysical)
        if logical.derived_id != derived_id or logical.expected_id() != derived_id or physical.derived_id != derived_id or physical.representation_id != "sha256:" + rep.name:
            raise StateError("derived-identity", "derived manifest identities disagree")
        if _digest(physical.model_dump(mode="json", exclude={"representation_id"})) != physical.representation_id:
            raise StateError("derived-digest", "derived physical manifest changed")
        if {p.name for p in base.iterdir()} != {"logical.json", "representations"}:
            raise StateError("derived-extra", "unexpected derived state entry")
        return rep, logical, physical

    def list(self):
        if not self._real_directory(self.root, missing_ok=True):
            return []
        directory = self.root / "derived"
        if not self._real_directory(directory, missing_ok=True):
            return []
        results = []
        for path in sorted(directory.iterdir()):
            base, _ = self._paths("shinobi-derived/v1:" + path.name)
            for member in sorted((base / "representations").iterdir()):
                _, logical, physical = self._load(logical_id := "shinobi-derived/v1:" + path.name, "sha256:" + member.name)
                results.append(
                    DerivedResult(
                        derived_id=logical_id, representation_id=physical.representation_id, parent_state_id=logical.parent_state_id, msv4_id=logical.msv4_id, verified=False
                    )
                )
        return results

    def verify(self, derived_id: str, representation_id: str | None = None):
        import msutils

        rep, logical, physical = self._load(derived_id, representation_id)
        if adapter.runtime() != physical.versions:
            raise StateError("stack-version", "derived representation requires its recorded runtime stack")
        self.exact.verify(logical.parent_state_id, physical.source_representation_id)
        source, _, parent = self.exact._load(logical.parent_state_id, physical.source_representation_id)
        if parent.msv4_id != logical.parent_msv4_id:
            raise StateError("derived-parent", "parent MSv4 logical identity differs")
        files, nodes = _inventory(rep)
        if files != physical.files or nodes != physical.zarr_metadata:
            raise StateError("derived-physical", "derived files or layout changed")
        if msutils.logical_id(rep / "msv4.zarr") != logical.msv4_id:
            raise StateError("derived-logical", "derived MSv4 logical identity differs")
        metrics = _transform(source / "msv4.zarr", rep / "msv4.zarr", logical.policy, verify=True)
        _verify_whole_tree(source / "msv4.zarr", rep / "msv4.zarr", logical.policy.arrays)
        if metrics != (physical.changed_count, physical.max_abs_error, physical.max_relative_error):
            raise StateError("derived-metrics", "derived error evidence differs")
        return DerivedResult(derived_id=derived_id, representation_id=physical.representation_id, parent_state_id=logical.parent_state_id, msv4_id=logical.msv4_id, verified=True)

    def derive(self, state_id: str, policy: PrecisionPolicy, representation_id: str | None = None):
        import msutils

        policy = PrecisionPolicy.model_validate(policy)
        self._real_directory(self.root, missing_ok=True)
        self.exact._setup()
        self._real_directory(self.root)
        self.exact.verify(state_id, representation_id)
        source, _, parent = self.exact._load(state_id, representation_id)
        operation = uuid.uuid4()
        staging = self.root / "staging"
        stage = staging / f"derived-{operation}"
        attempts = self.root / "derived-attempts"
        self._real_directory(attempts, missing_ok=True)
        attempts.mkdir(exist_ok=True)
        self._real_directory(attempts)
        derived_root = self.root / "derived"
        self._real_directory(derived_root, missing_ok=True)
        derived_root.mkdir(exist_ok=True)
        self._real_directory(derived_root)
        sync_directory_chain(attempts)
        sync_directory_chain(derived_root)
        attempt_path = attempts / f"{operation}.json"
        attempt = DerivedAttempt(
            attempt_id=operation,
            store=self.root,
            authority=self.root,
            registry=ownership_registry(),
            parent_state_id=state_id,
            source_representation_id=parent.representation_id,
            policy=policy,
            versions=adapter.runtime(),
            stage=stage,
            parent_identity=_identity(staging),
        )
        lease = None

        def save(**updates):
            nonlocal attempt
            attempt = DerivedAttempt.model_validate({**attempt.model_dump(), **updates})
            JsonFileStore(attempt_path).update(lambda data: data.update(attempt.model_dump(mode="json")))

        with self.exact._lock("attempts", str(operation)):
            save()
            try:
                lease = acquire_workspace(self.root, str(operation), kind="local", accesses=[(source, False), (stage, True)], registry=attempt.registry)
                stage.mkdir()
                save(stage_identity=_identity(stage), phase="writing")
                rep = stage / "representation"
                rep.mkdir()
                shutil.copytree(source / "msv4.zarr", rep / "msv4.zarr", symlinks=False)
                metrics = _transform(source / "msv4.zarr", rep / "msv4.zarr", policy)
                _verify_whole_tree(source / "msv4.zarr", rep / "msv4.zarr", policy.arrays)
                _transform(source / "msv4.zarr", rep / "msv4.zarr", policy, verify=True)
                self.exact.verify(state_id, parent.representation_id)
                transformed_id = msutils.logical_id(rep / "msv4.zarr")
                if transformed_id == parent.msv4_id:
                    raise StateError("precision-noop", "transformed logical ID is unchanged")
                logical = DerivedLogical(
                    derived_id="shinobi-derived/v1:" + "0" * 64, parent_state_id=state_id, parent_msv4_id=parent.msv4_id, policy=policy, msv4_id=transformed_id
                )
                logical = logical.model_copy(update={"derived_id": logical.expected_id()})
                files, nodes = _inventory(rep)
                physical = DerivedPhysical(
                    representation_id="sha256:" + "0" * 64,
                    derived_id=logical.derived_id,
                    source_representation_id=parent.representation_id,
                    versions=adapter.runtime(),
                    created_at=datetime.now(timezone.utc),
                    files=files,
                    zarr_metadata=nodes,
                    changed_count=metrics[0],
                    max_abs_error=metrics[1],
                    max_relative_error=metrics[2],
                )
                physical = physical.model_copy(update={"representation_id": _digest(physical.model_dump(mode="json", exclude={"representation_id"}))})
                _write(rep / "representation.json", physical)
                publication = stage / "state"
                (publication / "representations").mkdir(parents=True)
                _write(publication / "logical.json", logical)
                rep.rename(publication / "representations" / physical.representation_id.split(":")[1])
                _sync_tree(stage)
                candidate = publication / "representations" / physical.representation_id.split(":")[1]
                save(phase="ready", derived_id=logical.derived_id, representation_id=physical.representation_id, candidate_identity=_identity(candidate))
                with self.exact._lock("derived", logical.derived_id):
                    target, published_rep = self._paths(logical.derived_id, physical.representation_id, missing_ok=True)
                    if self._real_directory(target, missing_ok=True):
                        current = _manifest(target / "logical.json", DerivedLogical)
                        if current != logical:
                            raise StateError("derived-identity", "existing derived logical manifest disagrees")
                        publish_directory(candidate, published_rep)
                    else:
                        publish_directory(publication, target)
                save(phase="committed")
                shutil.rmtree(stage)
                sync_directory(staging)
                lease.release()
                lease = None
                save(finished=True)
                return DerivedResult(
                    derived_id=logical.derived_id,
                    representation_id=physical.representation_id,
                    parent_state_id=state_id,
                    msv4_id=transformed_id,
                    verified=True,
                    attempt=attempt_path,
                )
            except Exception as exc:
                if attempt.phase in {"ready", "committed"} and attempt.derived_id and attempt.representation_id:
                    _, candidate = self._paths(attempt.derived_id, attempt.representation_id, missing_ok=True)
                    if _directory_identity(candidate, missing_ok=True) == attempt.candidate_identity:
                        raise
                save(phase="refused" if isinstance(exc, (StateError, ValueError)) else "failed", error=str(exc))
                if os.path.lexists(stage):
                    if attempt.stage_identity is None or _directory_identity(stage) != attempt.stage_identity:
                        raise StateError("derived-stage-identity", "stage inode changed or was never recorded; preserve for recovery") from exc
                    shutil.rmtree(stage)
                    sync_directory(staging)
                if lease is not None:
                    lease.release()
                    lease = None
                save(finished=True)
                raise
            finally:
                if lease is not None:
                    lease.release()

    def recover(self):
        results = []
        if not self._real_directory(self.root, missing_ok=True):
            return results
        directory = self.root / "derived-attempts"
        if not self._real_directory(directory, missing_ok=True):
            return results
        paths = sorted(set(directory.glob("*.json")) | {p.with_suffix("") for p in directory.glob("*.json.lock")})
        for path in paths:
            attempt = DerivedAttempt.model_validate(JsonFileStore(path).read())
            if attempt.finished:
                continue
            with self.exact._lock("attempts", str(attempt.attempt_id), timeout=0.05):
                attempt = DerivedAttempt.model_validate(JsonFileStore(path).read())
                if attempt.finished:
                    continue
                if (
                    attempt.store != self.root
                    or path != directory / f"{attempt.attempt_id}.json"
                    or attempt.stage != self.root / "staging" / f"derived-{attempt.attempt_id}"
                    or _identity(attempt.stage.parent) != attempt.parent_identity
                ):
                    raise StateError("derived-recovery-path", "derived stage identity/path changed")
                evidence = inspect_ownership(attempt.authority, str(attempt.attempt_id))
                if evidence.owner is not None and Path(evidence.owner.registry) != attempt.registry:
                    raise StateError("derived-recovery-registry", "ownership registry disagrees")
                if evidence.liveness not in {"dead", "free"}:
                    raise StateError("derived-recovery-live", evidence.detail)
                published = False
                if attempt.derived_id and attempt.representation_id:
                    _, candidate = self._paths(attempt.derived_id, attempt.representation_id, missing_ok=True)
                    identity = _directory_identity(candidate, missing_ok=True)
                    if identity is not None:
                        if identity != attempt.candidate_identity:
                            raise StateError("derived-recovery-candidate", "published inode differs from recorded candidate")
                        self.verify(attempt.derived_id, attempt.representation_id)
                        published = True
                if os.path.lexists(attempt.stage):
                    if _identity(attempt.stage) != attempt.stage_identity:
                        raise StateError("derived-recovery-identity", "stage inode changed")
                    shutil.rmtree(attempt.stage)
                    sync_directory(attempt.stage.parent)
                updated = attempt.model_copy(update={"phase": "committed" if published else "interrupted", "finished": False})
                JsonFileStore(path).update(lambda data: data.update(updated.model_dump(mode="json")))
                if evidence.owner is not None:
                    release_workspace(attempt.authority, str(attempt.attempt_id))
                finished = updated.model_copy(update={"finished": True})
                JsonFileStore(path).update(lambda data: data.update(finished.model_dump(mode="json")))
                results.append(finished)
        return results
