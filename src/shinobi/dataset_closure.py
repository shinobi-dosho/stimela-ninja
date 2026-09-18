"""Bounded, serializable physical-resource closure for validated MSv2 data."""

from __future__ import annotations

import errno
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from shinobi.datasets import DatasetStatus, _load_table_factory, inspect_measurement_set_v2

DATASET_CLOSURE_PROFILE = "msv2-dataset-closure/v1"
_OPTIONAL_SUBTABLES = frozenset({"DOPPLER", "FREQ_OFFSET", "SOURCE", "SYSCAL", "WEATHER"})
_MANDATORY_SUBTABLES = frozenset(
    {"ANTENNA", "DATA_DESCRIPTION", "FEED", "FIELD", "FLAG_CMD", "HISTORY", "OBSERVATION", "POINTING", "POLARIZATION", "PROCESSOR", "SPECTRAL_WINDOW", "STATE"}
)
_SUPPORTED_MANAGERS = frozenset({"IncrementalStMan", "StandardStMan", "StManAipsIO", "TiledCellStMan", "TiledColumnStMan", "TiledDataStMan", "TiledShapeStMan"})


class ClosureStatus(str, Enum):
    """Outcome of MSv2 physical closure resolution."""

    VALID = "valid"
    INVALID_ROOT = "invalid-root"
    MISSING_RESOURCE = "missing-resource"
    UNREADABLE_RESOURCE = "unreadable-resource"
    DANGLING_REFERENCE = "dangling-reference"
    CYCLIC_REFERENCE = "cyclic-reference"
    ESCAPED_NAMESPACE = "escaped-namespace"
    CHANGED_DURING_RESOLUTION = "changed-during-resolution"
    UNSUPPORTED = "unsupported"


class ClosureRequirement(str, Enum):
    """Physical requirements of the accepted ordinary-directory profile."""

    TABLE_MEMBERS = "table-members"
    PRESERVE_NAMESPACE_PATHS = "preserve-namespace-paths"
    REWRITE_KEYWORD_REFERENCES = "rewrite-keyword-references"


class ClosureCapabilities(BaseModel):
    """Operation requirements established by the accepted closure profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    copy_requirements: tuple[ClosureRequirement, ...]
    mount_requirements: tuple[ClosureRequirement, ...]
    stage_requirements: tuple[ClosureRequirement, ...]
    materialize_requirements: tuple[ClosureRequirement, ...]
    restore_requirements: tuple[ClosureRequirement, ...]


_ORDINARY_CAPABILITIES = ClosureCapabilities(
    copy_requirements=(ClosureRequirement.TABLE_MEMBERS, ClosureRequirement.PRESERVE_NAMESPACE_PATHS),
    mount_requirements=(ClosureRequirement.TABLE_MEMBERS, ClosureRequirement.PRESERVE_NAMESPACE_PATHS),
    stage_requirements=(ClosureRequirement.TABLE_MEMBERS, ClosureRequirement.PRESERVE_NAMESPACE_PATHS),
    materialize_requirements=(ClosureRequirement.TABLE_MEMBERS, ClosureRequirement.REWRITE_KEYWORD_REFERENCES),
    restore_requirements=(ClosureRequirement.TABLE_MEMBERS, ClosureRequirement.PRESERVE_NAMESPACE_PATHS),
)


class ClosureResource(BaseModel):
    """One canonical physical CASA table in a dataset closure."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    namespace_path: Path
    members: tuple[str, ...]
    table_files: tuple[str, ...]
    external_to_root: bool
    storage_managers: tuple[str, ...]


class DatasetClosure(BaseModel):
    """Serializable observation of an MSv2's physical CASA-table resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    profile: str = DATASET_CLOSURE_PROFILE
    requested_root: Path
    storage_namespace: Path
    root: Path | None = None
    status: ClosureStatus
    message: str
    resources: tuple[ClosureResource, ...] = ()
    capabilities: ClosureCapabilities | None = None
    unsupported_features: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status is ClosureStatus.VALID


def _result(path: Path, namespace: Path, status: ClosureStatus, message: str, **kwargs: Any) -> DatasetClosure:
    return DatasetClosure(requested_root=path, storage_namespace=namespace, status=status, message=message, **kwargs)


def _canonical(path: Path) -> tuple[Path | None, ClosureStatus | None, str | None]:
    try:
        return path.resolve(strict=True), None, None
    except RuntimeError as exc:
        return None, ClosureStatus.CYCLIC_REFERENCE, f"cyclic filesystem reference at {path}: {exc}"
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None, ClosureStatus.CYCLIC_REFERENCE, f"cyclic filesystem reference at {path}: {exc}"
        if exc.errno in {errno.EACCES, errno.EPERM}:
            return None, ClosureStatus.UNREADABLE_RESOURCE, f"resource is unreadable: {path}: {exc}"
        return None, ClosureStatus.DANGLING_REFERENCE, f"referenced resource does not exist: {path}: {exc}"


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _entry_record(item: Path, relative: str) -> tuple[str, int, int, int, int]:
    stat = item.lstat()
    return relative, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _directory_observation(path: Path, *, max_entries: int) -> tuple[tuple[str, int, int, int, int], ...]:
    """Observe direct entries only, so MAIN does not absorb its subtables."""

    entries = sorted(path.iterdir(), key=lambda item: item.name)
    if len(entries) + 1 > max_entries:
        raise OverflowError(f"resource {path} exceeds closure entry limit {max_entries}")
    return (_entry_record(path, "."), *(_entry_record(item, item.name) for item in entries))


def _table_observation(path: Path, table_files: tuple[str, ...]) -> tuple[tuple[str, int, int, int, int], ...]:
    """Observe only one table's backing members, never child subtables."""

    return (_entry_record(path, "."), *(_entry_record(path / name, name) for name in table_files))


def _direct_link_problem(path: Path, namespace: Path) -> tuple[ClosureStatus | None, str | None]:
    """Reject broken or escaped direct links before casacore can follow them."""

    try:
        entries = tuple(path.iterdir())
    except OSError as exc:
        return ClosureStatus.UNREADABLE_RESOURCE, f"cannot enumerate CASA table {path}: {exc}"
    for item in entries:
        if not item.is_symlink():
            continue
        resolved, error, message = _canonical(item)
        if error:
            return error, f"backing member {item}: {message}"
        assert resolved is not None
        if not resolved.is_relative_to(namespace):
            return ClosureStatus.ESCAPED_NAMESPACE, f"backing member escapes storage namespace {namespace}: {item} -> {resolved}"
    return None, None


def _keyword_path(value: Any, owner: Path) -> Path:
    close = getattr(value, "close", None)
    try:
        name = value.name() if callable(getattr(value, "name", None)) else value
        if not isinstance(name, (str, os.PathLike)):
            raise TypeError(f"table keyword has unsupported value {type(value).__name__}")
        target = Path(os.fspath(name).removeprefix("Table: "))
        return target if target.is_absolute() else owner / target
    finally:
        if callable(close):
            close()


def _plain_table_metadata(table: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return metadata only for a plain table, using authoritative parts."""

    try:
        reported_name = Path(table.name())
        parts = tuple(Path(part).resolve(strict=True) for part in table.partnames())
    except Exception as exc:
        raise ValueError(f"cannot establish authoritative CASA table parts: {type(exc).__name__}: {exc}") from exc
    if len(parts) != 1:
        raise ValueError(f"unsupported reference or virtual-concatenation table: name={reported_name}, parts={parts!r}")
    try:
        name = reported_name.resolve(strict=True)
    except Exception as exc:
        raise ValueError(f"cannot establish authoritative CASA table name: {type(exc).__name__}: {exc}") from exc
    if parts != (name,):
        raise ValueError(f"unsupported reference or virtual-concatenation table: name={name}, parts={parts!r}")
    dminfo = table.getdminfo()
    managers = tuple(sorted({str(item.get("TYPE", "")) for item in dminfo.values() if item.get("TYPE")}))
    unsupported = tuple(manager for manager in managers if manager not in _SUPPORTED_MANAGERS)
    if unsupported:
        raise ValueError(f"unsupported opaque storage manager(s): {', '.join(unsupported)}")
    return tuple(sorted(str(keyword) for keyword in table.keywordnames())), managers


def _changed(source: Path, canonical: Path, identity: tuple[int, int]) -> bool:
    current, error, _ = _canonical(source)
    try:
        return error is not None or current != canonical or _identity(canonical) != identity
    except OSError:
        return True


def _classify_table_files(
    physical: Path, *, namespace: Path, table_paths: frozenset[Path], member_paths: frozenset[Path], max_entries: int
) -> tuple[tuple[str, ...] | None, ClosureStatus | None, str | None]:
    entries = sorted(physical.iterdir(), key=lambda item: item.name)
    if len(entries) + 1 > max_entries:
        raise OverflowError(f"resource {physical} exceeds closure entry limit {max_entries}")
    files: list[str] = []
    for item in entries:
        resolved, error, message = _canonical(item)
        if error:
            return None, error, f"backing member {item}: {message}"
        assert resolved is not None
        if not resolved.is_relative_to(namespace):
            return None, ClosureStatus.ESCAPED_NAMESPACE, f"backing member escapes storage namespace {namespace}: {item} -> {resolved}"
        if resolved in table_paths and (item in member_paths or item.is_dir()):
            continue
        if item.is_symlink():
            return None, ClosureStatus.UNSUPPORTED, f"unsupported intra-table backing symlink: {item} -> {resolved}"
        if item.is_dir():
            return None, ClosureStatus.UNSUPPORTED, f"unsupported nested directory in ordinary CASA table: {item}"
        if not item.is_file():
            return None, ClosureStatus.UNSUPPORTED, f"unsupported non-file CASA table member: {item}"
        files.append(item.name)
    return tuple(files), None, None


def resolve_dataset_closure(path: str | Path, *, storage_namespace: str | Path, max_resources: int = 32, max_entries_per_resource: int = 100_000) -> DatasetClosure:
    """Resolve an MSv2 into canonical physical CASA-table resources.

    Resolution detects changes across its reads, but filesystem observation is
    not an atomic snapshot.  Consumers must use a cooperative immutable or
    snapshot boundary and revalidate the observation before acting on it.
    """

    if max_resources < 1 or max_entries_per_resource < 1:
        raise ValueError("closure limits must be positive")
    requested = Path(path)
    namespace_requested = Path(storage_namespace)
    namespace, error, message = _canonical(namespace_requested)
    if error:
        return _result(requested, namespace_requested, error, message or "invalid storage namespace")
    assert namespace is not None
    if not namespace.is_dir():
        return _result(requested, namespace, ClosureStatus.INVALID_ROOT, f"storage namespace is not a directory: {namespace}")
    root, error, message = _canonical(requested)
    if error:
        status = ClosureStatus.MISSING_RESOURCE if error is ClosureStatus.DANGLING_REFERENCE else error
        return _result(requested, namespace, status, message or "invalid dataset root")
    assert root is not None
    if not root.is_relative_to(namespace):
        return _result(requested, namespace, ClosureStatus.ESCAPED_NAMESPACE, f"dataset root escapes storage namespace {namespace}: {root}", root=root)
    if not root.is_dir():
        return _result(requested, namespace, ClosureStatus.INVALID_ROOT, f"MSv2 root is not a directory: {root}", root=root)
    error, message = _direct_link_problem(root, namespace)
    if error:
        return _result(requested, namespace, error, message or "invalid MSv2 backing link", root=root)

    try:
        root_identity = _identity(root)
        initial_root = _directory_observation(root, max_entries=max_entries_per_resource)
        table_factory = _load_table_factory()
        main = table_factory(str(root), readonly=True, ack=False)
        try:
            keywords, root_managers = _plain_table_metadata(main)
            reference_paths = {member: _keyword_path(main.getkeyword(member), root) for member in sorted((_MANDATORY_SUBTABLES | _OPTIONAL_SUBTABLES).intersection(keywords))}
        finally:
            main.close()
    except OverflowError as exc:
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
    except ValueError as exc:
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
    except Exception as exc:
        return _result(requested, namespace, ClosureStatus.UNREADABLE_RESOURCE, f"cannot resolve MSv2 root metadata: {type(exc).__name__}: {exc}", root=root)
    if _changed(requested, root, root_identity):
        return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"MSv2 root changed during metadata discovery: {requested}", root=root)

    by_path: dict[Path, dict[str, Any]] = {root: {"members": ["MAIN"], "managers": root_managers, "source_paths": [requested]}}
    member_paths: set[Path] = set()
    for member, reference in reference_paths.items():
        physical, error, message = _canonical(reference)
        if error:
            return _result(requested, namespace, error, f"subtable {member}: {message}", root=root)
        assert physical is not None
        if not physical.is_relative_to(namespace):
            return _result(requested, namespace, ClosureStatus.ESCAPED_NAMESPACE, f"subtable {member} escapes storage namespace {namespace}: {physical}", root=root)
        if physical == root:
            return _result(requested, namespace, ClosureStatus.CYCLIC_REFERENCE, f"subtable {member} refers back to the MS root {root}", root=root)
        if not physical.is_dir():
            return _result(requested, namespace, ClosureStatus.DANGLING_REFERENCE, f"subtable {member} is not a directory-backed CASA table: {physical}", root=root)
        error, message = _direct_link_problem(physical, namespace)
        if error:
            return _result(requested, namespace, error, f"subtable {member}: {message}", root=root)
        member_paths.add(reference)
        if physical in by_path:
            by_path[physical]["members"].append(member)
            by_path[physical]["source_paths"].append(reference)
            continue
        try:
            identity = _identity(physical)
            initial_direct = _directory_observation(physical, max_entries=max_entries_per_resource)
            subtable = table_factory(str(physical), readonly=True, ack=False)
            try:
                _, managers = _plain_table_metadata(subtable)
            finally:
                subtable.close()
            after_direct = _directory_observation(physical, max_entries=max_entries_per_resource)
        except OverflowError as exc:
            return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
        except ValueError as exc:
            return _result(requested, namespace, ClosureStatus.UNSUPPORTED, f"subtable {member}: {exc}", root=root, unsupported_features=(str(exc),))
        except Exception as exc:
            return _result(requested, namespace, ClosureStatus.UNREADABLE_RESOURCE, f"subtable {member} is unreadable at {physical}: {type(exc).__name__}: {exc}", root=root)
        if _changed(reference, physical, identity) or after_direct != initial_direct:
            return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"subtable {member} changed during metadata discovery: {reference}", root=root)
        by_path[physical] = {"members": [member], "managers": managers, "source_paths": [reference], "identity": identity}

    if len(by_path) > max_resources:
        feature = f"canonical closure resource count {len(by_path)} exceeds limit {max_resources}"
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, feature, root=root, unsupported_features=(feature,))

    table_paths = frozenset(by_path)
    baselines: dict[Path, tuple[tuple[str, int, int, int, int], ...]] = {}
    for physical, data in by_path.items():
        data.setdefault("identity", _identity(physical))
        try:
            files, error, message = _classify_table_files(
                physical, namespace=namespace, table_paths=table_paths, member_paths=frozenset(member_paths), max_entries=max_entries_per_resource
            )
            if error:
                return _result(requested, namespace, error, message or "invalid table member", root=root)
            data["files"] = files
            baselines[physical] = _table_observation(physical, files or ())
        except OverflowError as exc:
            return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
        except OSError as exc:
            return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"resource changed during initial observation: {physical}: {exc}", root=root)

    inspected = inspect_measurement_set_v2(root)
    if inspected.status is not DatasetStatus.VALID:
        return _result(requested, namespace, ClosureStatus.INVALID_ROOT, f"MSv2 root is not structurally valid: {inspected.message}", root=root)

    final_observations: dict[Path, tuple[tuple[str, int, int, int, int], ...]] = {}
    try:
        for physical in sorted(by_path, key=lambda item: item.relative_to(namespace).as_posix()):
            final_observations[physical] = _table_observation(physical, by_path[physical]["files"])
        final_root = _directory_observation(root, max_entries=max_entries_per_resource)
    except (OSError, OverflowError) as exc:
        return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"resource changed during final observation: {type(exc).__name__}: {exc}", root=root)
    changed = [
        physical
        for physical, data in by_path.items()
        if any(_changed(source, physical, data["identity"]) for source in data["source_paths"]) or final_observations[physical] != baselines[physical]
    ]
    if final_root != initial_root and root not in changed:
        changed.append(root)
    if changed:
        names = ", ".join(str(item) for item in sorted(set(changed)))
        return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"resource changed during closure resolution: {names}", root=root)

    resources = tuple(
        ClosureResource(
            path=physical,
            namespace_path=physical.relative_to(namespace),
            members=tuple(sorted(data["members"])),
            table_files=data["files"],
            external_to_root=physical != root and not physical.is_relative_to(root),
            storage_managers=data["managers"],
        )
        for physical, data in sorted(by_path.items(), key=lambda item: item[0].relative_to(namespace).as_posix())
    )
    return _result(
        requested,
        namespace,
        ClosureStatus.VALID,
        f"resolved {len(resources)} canonical physical resources for {DATASET_CLOSURE_PROFILE}",
        root=root,
        resources=resources,
        capabilities=_ORDINARY_CAPABILITIES,
    )


__all__ = ["DATASET_CLOSURE_PROFILE", "ClosureCapabilities", "ClosureRequirement", "ClosureResource", "ClosureStatus", "DatasetClosure", "resolve_dataset_closure"]
