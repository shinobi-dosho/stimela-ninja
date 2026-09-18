"""Bounded, serializable physical-resource closure for validated MSv2 data.

Closure resolution is deliberately separate from the ``MeasurementSetV2``
annotation and from structural inspection.  It observes the files which must
move together; it does not make the annotation executable and never retains
an open casacore table.
"""

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
    {
        "ANTENNA",
        "DATA_DESCRIPTION",
        "FEED",
        "FIELD",
        "FLAG_CMD",
        "HISTORY",
        "OBSERVATION",
        "POINTING",
        "POLARIZATION",
        "PROCESSOR",
        "SPECTRAL_WINDOW",
        "STATE",
    }
)
_SUPPORTED_MANAGERS = frozenset(
    {
        "IncrementalStMan",
        "StandardStMan",
        "StManAipsIO",
        "TiledCellStMan",
        "TiledColumnStMan",
        "TiledDataStMan",
        "TiledShapeStMan",
    }
)


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


class ClosureOperation(str, Enum):
    """Lifecycle operations whose physical requirements were recorded."""

    COPY = "copy"
    MOUNT = "mount"
    STAGE = "stage"
    MATERIALIZE = "materialize"
    RESTORE = "restore"


class ClosureResource(BaseModel):
    """One canonical physical CASA table in a dataset closure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    namespace_path: Path
    members: tuple[str, ...]
    external_to_root: bool
    storage_managers: tuple[str, ...]
    operations: tuple[ClosureOperation, ...] = tuple(ClosureOperation)


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


def _signature(path: Path, *, max_entries: int) -> tuple[tuple[str, int, int, int, int], ...]:
    records: list[tuple[str, int, int, int, int]] = []
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        files.sort()
        for name in [".", *dirs, *files]:
            item = Path(base) if name == "." else Path(base, name)
            stat = item.lstat()
            records.append((str(item.relative_to(path)), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
            if len(records) > max_entries:
                raise OverflowError(f"resource {path} exceeds closure entry limit {max_entries}")
    return tuple(records)


def _keyword_path(value: Any, owner: Path) -> Path:
    close = getattr(value, "close", None)
    try:
        name = value.name() if callable(getattr(value, "name", None)) else value
        if not isinstance(name, (str, os.PathLike)):
            raise TypeError(f"table keyword has unsupported value {type(value).__name__}")
        text = os.fspath(name).removeprefix("Table: ")
        target = Path(text)
        return target if target.is_absolute() else owner / target
    finally:
        if callable(close):
            close()


def _table_metadata(table: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    info = table.info() if callable(getattr(table, "info", None)) else {}
    info_text = " ".join(str(info.get(key, "")) for key in ("type", "subType", "readme")).lower()
    if "reference" in info_text or "concat" in info_text:
        raise ValueError(f"unsupported CASA table form: {info_text.strip()}")
    dminfo = table.getdminfo() if callable(getattr(table, "getdminfo", None)) else {}
    managers = tuple(sorted({str(item.get("TYPE", "")) for item in dminfo.values() if item.get("TYPE")}))
    unsupported = tuple(name for name in managers if name not in _SUPPORTED_MANAGERS)
    if unsupported:
        raise ValueError(f"unsupported opaque storage manager(s): {', '.join(unsupported)}")
    keywords = tuple(sorted(str(name) for name in table.keywordnames()))
    return keywords, managers


def resolve_dataset_closure(
    path: str | Path,
    *,
    storage_namespace: str | Path,
    max_resources: int = 32,
    max_entries_per_resource: int = 100_000,
) -> DatasetClosure:
    """Resolve a validated MSv2 into canonical physical CASA-table resources.

    ``storage_namespace`` is an explicit trust boundary.  The MS root and all
    referenced subtables must resolve inside it, though a subtable may be
    outside the MS directory and may be shared by several MS roots.
    """

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

    inspected = inspect_measurement_set_v2(root)
    if inspected.status is not DatasetStatus.VALID:
        return _result(requested, namespace, ClosureStatus.INVALID_ROOT, f"MSv2 root is not structurally valid: {inspected.message}", root=root)

    try:
        table_factory = _load_table_factory()
        before_root = _signature(root, max_entries=max_entries_per_resource)
        main = table_factory(str(root), readonly=True, ack=False)
        try:
            keywords, root_managers = _table_metadata(main)
            references: list[tuple[str, Path]] = []
            for name in sorted((_MANDATORY_SUBTABLES | _OPTIONAL_SUBTABLES).intersection(keywords)):
                references.append((name, _keyword_path(main.getkeyword(name), root)))
        finally:
            main.close()
    except OverflowError as exc:
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
    except (OSError, PermissionError) as exc:
        return _result(requested, namespace, ClosureStatus.UNREADABLE_RESOURCE, f"cannot read MSv2 root {root}: {type(exc).__name__}: {exc}", root=root)
    except ValueError as exc:
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
    except Exception as exc:
        return _result(requested, namespace, ClosureStatus.UNREADABLE_RESOURCE, f"cannot resolve MSv2 root metadata: {type(exc).__name__}: {exc}", root=root)

    if len(references) + 1 > max_resources:
        feature = f"closure resource count {len(references) + 1} exceeds limit {max_resources}"
        return _result(requested, namespace, ClosureStatus.UNSUPPORTED, feature, root=root, unsupported_features=(feature,))

    by_path: dict[Path, dict[str, Any]] = {root: {"members": ["MAIN"], "managers": root_managers, "before": before_root}}
    for member, reference in references:
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
        if physical in by_path:
            by_path[physical]["members"].append(member)
            continue
        try:
            before = _signature(physical, max_entries=max_entries_per_resource)
            subtable = table_factory(str(physical), readonly=True, ack=False)
            try:
                _, managers = _table_metadata(subtable)
            finally:
                subtable.close()
        except OverflowError as exc:
            return _result(requested, namespace, ClosureStatus.UNSUPPORTED, str(exc), root=root, unsupported_features=(str(exc),))
        except ValueError as exc:
            return _result(requested, namespace, ClosureStatus.UNSUPPORTED, f"subtable {member}: {exc}", root=root, unsupported_features=(str(exc),))
        except Exception as exc:
            return _result(requested, namespace, ClosureStatus.UNREADABLE_RESOURCE, f"subtable {member} is unreadable at {physical}: {type(exc).__name__}: {exc}", root=root)
        by_path[physical] = {"members": [member], "managers": managers, "before": before}

    resources: list[ClosureResource] = []
    try:
        for physical in sorted(by_path, key=lambda item: item.relative_to(namespace).as_posix()):
            data = by_path[physical]
            if _signature(physical, max_entries=max_entries_per_resource) != data["before"]:
                return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"resource changed during closure resolution: {physical}", root=root)
            resources.append(
                ClosureResource(
                    path=physical,
                    namespace_path=physical.relative_to(namespace),
                    members=tuple(sorted(data["members"])),
                    external_to_root=physical != root and not physical.is_relative_to(root),
                    storage_managers=data["managers"],
                )
            )
    except (OSError, OverflowError) as exc:
        return _result(requested, namespace, ClosureStatus.CHANGED_DURING_RESOLUTION, f"resource changed during closure resolution: {type(exc).__name__}: {exc}", root=root)

    return _result(
        requested,
        namespace,
        ClosureStatus.VALID,
        f"resolved {len(resources)} canonical physical resources for {DATASET_CLOSURE_PROFILE}",
        root=root,
        resources=tuple(resources),
    )


__all__ = [
    "DATASET_CLOSURE_PROFILE",
    "ClosureOperation",
    "ClosureResource",
    "ClosureStatus",
    "DatasetClosure",
    "resolve_dataset_closure",
]
