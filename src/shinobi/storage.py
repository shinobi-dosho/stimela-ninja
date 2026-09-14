"""Shared-filesystem publication and mutable-metadata transactions.

Immutable worker observations use atomic no-clobber publication; mutable cache
indexes and snapshot journals use :class:`JsonFileStore`.  Both routes share
the durability primitive here, but only mutable metadata takes a short-lived
transaction lock.  That lock is deliberately not ownership of any scientific
data a step may mutate.

The transaction contract requires 64-bit Linux open-file-description locks
which conflict with the POSIX locks used by NFSv4, plus atomic same-directory
``os.replace`` and honoured file/directory ``fsync``.  System-call failures are
fatal before the log commit rather than silently weakening the contract; a
compatibility-view refresh failure after commit is only a warning. A local
process cannot prove that a remote node observes its locks, so supported
deployments must run the physical cross-node probe in
``tests/slurm_physical/run_m2.py``.
"""

from __future__ import annotations

import fcntl
import errno
import hashlib
import json
import os
import platform
import struct
import sys
import threading
import time
import warnings
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4


class SharedStorageError(RuntimeError):
    """A required shared-storage locking or durability operation failed."""


T = TypeVar("T")

_FLOCK_FORMAT = "hhqqi4x"
_F_OFD_SETLK = getattr(fcntl, "F_OFD_SETLK", None)
_F_OFD_SETLKW = getattr(fcntl, "F_OFD_SETLKW", None)
_open_lock_fds: set[int] = set()
_open_lock_fds_guard = threading.Lock()


def ofd_lock(fd: int, lock_type: int, *, blocking: bool = True) -> None:
    """Take or release a Linux open-file-description lock over all bytes.

    OFD locks conflict with ordinary POSIX locks used by NFSv4, but are owned
    by this open file description: independently opened stores in one process
    exclude each other, and closing an unrelated descriptor cannot release a
    transaction.  The explicit ABI check fails closed on unsupported builds.
    """
    if sys.platform != "linux" or struct.calcsize(_FLOCK_FORMAT) != 32 or struct.calcsize("P") != 8:
        raise SharedStorageError("shared metadata transactions require the 64-bit Linux OFD-lock ABI")
    commands = (_F_OFD_SETLK, _F_OFD_SETLKW)
    if None in commands:
        # python-build-standalone 3.13 in the supported uv environment omits
        # these constants even though its Linux kernel implements them. Linux
        # UAPI values 37/38 are admitted only for the two 64-bit architectures
        # exercised here; every other ABI fails closed above/below.
        if platform.machine().lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
            raise SharedStorageError("this Python omits OFD-lock constants and its Linux architecture is not a verified fallback")
        commands = (37, 38)
    command = commands[1] if blocking and lock_type != fcntl.F_UNLCK else commands[0]
    lock = struct.pack(_FLOCK_FORMAT, lock_type, os.SEEK_SET, 0, 0, 0)
    fcntl.fcntl(fd, command, lock)


def _before_fork() -> None:
    _open_lock_fds_guard.acquire()


def _after_fork_parent() -> None:
    _open_lock_fds_guard.release()


def _after_fork_child() -> None:
    # OFD locks follow the open description across fork. Drop the child's
    # copies so it cannot accidentally prolong a lock owned by the parent.
    for fd in _open_lock_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    _open_lock_fds.clear()
    _open_lock_fds_guard.release()


os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)


def sync_directory_chain(path: Path) -> None:
    """Fsync a directory and every entry that makes it reachable.

    Shared storage is often reached through a symlink, so both the resolved
    ancestry and the spelling supplied by the caller are covered.  This is
    shared with immutable worker-record publication.
    """
    path = path.absolute()
    resolved = path.resolve()
    for parent in dict.fromkeys((resolved, *resolved.parents, path, *path.parents)):
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def sync_directory(path: Path) -> None:
    """Fsync exactly one directory containing a created/replaced entry."""
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ensure_directory(path: Path) -> None:
    """Create ``path`` durably, syncing only entries created by this call."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        sync_directory(directory.parent)


class JsonFileStore:
    """One backward-compatible JSON object with serialized transactions.

    A persistent sibling ``.lock`` file is record-locked before every logged
    read or read-modify-write transaction. A read-only legacy JSON file with no
    sidecar can still be inspected without creating one. The lock must never be unlinked: replacing a
    lock inode while another process still holds it would create two lock
    domains.  The same inode carries an append-only, checksummed transaction
    log.  This is load-bearing on an NFS server whose local clients may reopen
    a stale pathname even after a remote atomic replacement.

    The first record is a full snapshot; later records contain top-level sets
    and deletes.  A partial trailing record is ignored after interruption,
    while corruption of a newline-terminated record is fatal.  After logging
    the commit, the ordinary JSON file is refreshed through an ``O_EXCL``
    random temporary, so equal PIDs on different nodes cannot select the same
    name.  Readers replay the log over that materialized view; replacement is
    therefore compatibility and convenience, not the commit point.

    The JSON payload and target filename are unchanged, so a legacy store is
    imported by the first update. That upgrade is one-way: after the log has
    been created, every writer must use this protocol. A pre-M2 writer cannot
    participate in its lock and its direct JSON replacement is intentionally
    ignored rather than allowed to overwrite committed log state.
    """

    def __init__(self, path: Path, *, lock_timeout: float = 60.0):
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        self.lock_timeout = lock_timeout

    def _read_base_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SharedStorageError(f"cannot read materialized transaction view {self.path}: {exc}") from exc

    def _read_unlocked(self, fd: int) -> tuple[dict[str, Any], bool, int]:
        size = os.fstat(fd).st_size
        chunks = []
        offset = 0
        while offset < size:
            chunk = os.pread(fd, size - offset, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        if not raw:
            return self._read_base_unlocked(), False, 0
        # Every non-empty log begins with a full snapshot. Do not consult the
        # compatibility JSON at all after upgrade: it may lag a committed log,
        # be absent/read-only, or have been replaced by an unsafe old writer.
        data: dict[str, Any] = {}
        lines = raw.splitlines(keepends=True)
        complete = lines if lines[-1].endswith(b"\n") else lines[:-1]
        committed_size = sum(map(len, complete))
        for index, line in enumerate(complete, 1):
            try:
                expected, payload = line[:-1].split(b" ", 1)
                if hashlib.sha256(payload).hexdigest().encode() != expected:
                    raise ValueError("checksum mismatch")
                record = json.loads(payload)
                if record.get("version") != 1:
                    raise ValueError(f"unsupported version {record.get('version')!r}")
                if "snapshot" in record:
                    data = record["snapshot"]
                else:
                    for key in record["delete"]:
                        data.pop(key, None)
                    data.update(record["set"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SharedStorageError(f"corrupt committed transaction {index} in {self.lock_path}: {exc}") from exc
        return data, bool(complete), committed_size

    def _register_fd(self, flags: int, mode: int = 0o666) -> int:
        with _open_lock_fds_guard:
            fd = os.open(self.lock_path, flags, mode)
            _open_lock_fds.add(fd)
        return fd

    def _acquire(self, fd: int, lock_type: int) -> None:
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                ofd_lock(fd, lock_type, blocking=False)
                return
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise SharedStorageError(
                        f"timed out after {self.lock_timeout:g}s waiting for transaction lock {self.lock_path}; the holder may be stalled or disconnected from shared storage"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _open_and_lock(self, operation: int) -> int | None:
        try:
            if operation == fcntl.LOCK_SH:
                try:
                    fd = self._register_fd(os.O_RDONLY)
                except FileNotFoundError:
                    return None
            else:
                _ensure_directory(self.path.parent)
                try:
                    fd = self._register_fd(os.O_RDWR)
                except FileNotFoundError:
                    try:
                        fd = self._register_fd(os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666)
                    except FileExistsError:
                        fd = self._register_fd(os.O_RDWR)
            try:
                self._acquire(fd, fcntl.F_RDLCK if operation == fcntl.LOCK_SH else fcntl.F_WRLCK)
                if operation == fcntl.LOCK_EX:
                    # The persistent inode is part of the protocol. Sync its
                    # immediate parent before every commit, so even recovery
                    # after an earlier creation-sync failure cannot use an
                    # entry whose durability was never established.
                    sync_directory(self.lock_path.parent)
            except BaseException:
                self._close(fd)
                raise
            return fd
        except SharedStorageError:
            raise
        except OSError as exc:
            raise SharedStorageError(f"cannot establish transaction lock {self.lock_path}: {exc}") from exc

    @staticmethod
    def _close(fd: int) -> None:
        with _open_lock_fds_guard:
            try:
                os.close(fd)
            except OSError as exc:
                # Closing releases the OFD lock. Do not report a committed
                # transaction as failed if close itself reports an error.
                warnings.warn(f"could not close transaction descriptor {fd}: {exc}", RuntimeWarning, stacklevel=2)
            finally:
                _open_lock_fds.discard(fd)

    @classmethod
    def _unlock(cls, fd: int) -> None:
        # An OFD lock is released by closing its last descriptor. Avoid a
        # separate post-commit unlock syscall whose failure could falsely
        # report an already durable transaction as failed.
        cls._close(fd)

    def read(self) -> dict[str, Any]:
        """Return one committed version under a shared process/node lock."""
        # Preserve the old miss behaviour: merely checking an unused cache
        # does not create its directory or lock sidecar.
        if not self.path.parent.exists():
            return {}
        fd = self._open_and_lock(fcntl.LOCK_SH)
        if fd is None:
            # A legacy/read-only cache has no sidecar yet. Read it without
            # creating anything, then retry under the lock if a writer raced
            # us through the one-way upgrade.
            data = self._read_base_unlocked()
            if self.lock_path.exists():
                return self.read()
            return data
        try:
            try:
                data, _logged, _committed_size = self._read_unlocked(fd)
            except OSError as exc:
                raise SharedStorageError(f"cannot read transaction store {self.path}: {exc}") from exc
            return data
        finally:
            self._unlock(fd)

    def update(self, mutate: Callable[[dict[str, Any]], T]) -> T:
        """Apply ``mutate`` and commit its resulting in-place JSON update.

        The callback runs while the open-description lock excludes both local
        threads and remote processes. If it raises, no commit is attempted.
        """
        fd = self._open_and_lock(fcntl.LOCK_EX)
        assert fd is not None
        try:
            self._cleanup_temporaries_unlocked()
            try:
                data, logged, committed_size = self._read_unlocked(fd)
            except OSError as exc:
                raise SharedStorageError(f"cannot read transaction store {self.path}: {exc}") from exc
            before = deepcopy(data)
            result = mutate(data)
            changed = {key: value for key, value in data.items() if key not in before or before[key] != value}
            deleted = [key for key in before if key not in data]
            if not changed and not deleted:
                return result
            try:
                if committed_size != os.fstat(fd).st_size:
                    os.ftruncate(fd, committed_size)
            except OSError as exc:
                raise SharedStorageError(f"cannot prepare transaction log {self.lock_path}: {exc}") from exc
            record = {"version": 1, "set": changed, "delete": deleted} if logged else {"version": 1, "snapshot": data}
            self._append_unlocked(fd, record, committed_size)
            try:
                self._write_unlocked(data)
            except OSError as exc:
                # The fsynced log is already the commit point. Reporting
                # failure here would invite a retry after a successful commit.
                warnings.warn(
                    f"committed {self.lock_path}, but could not refresh compatibility view {self.path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return result
        finally:
            self._unlock(fd)

    def _append_unlocked(self, fd: int, record: dict[str, Any], rollback_offset: int) -> None:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        line = hashlib.sha256(payload).hexdigest().encode() + b" " + payload + b"\n"
        try:
            os.lseek(fd, 0, os.SEEK_END)
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                if written == 0:
                    raise OSError("zero-byte transaction write")
                view = view[written:]
            os.fsync(fd)
        except OSError as exc:
            # Best-effort rollback makes ordinary write/fsync failures retain
            # the documented no-commit-on-error contract. If rollback itself
            # fails, say explicitly that the outcome cannot be determined.
            try:
                os.ftruncate(fd, rollback_offset)
                os.fsync(fd)
            except OSError as rollback_exc:
                raise SharedStorageError(f"transaction commit failed for {self.lock_path}, and rollback also failed; outcome is unknown: {rollback_exc}") from exc
            raise SharedStorageError(f"transaction was not committed to {self.lock_path}: {exc}") from exc

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            sync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _cleanup_temporaries_unlocked(self) -> None:
        """Remove leftovers from writers that died while holding this lock."""
        for temporary in self.path.parent.glob(f".{self.path.name}.*.tmp"):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                warnings.warn(f"could not remove stale transaction temporary {temporary}: {exc}", RuntimeWarning, stacklevel=2)

    def reset(self, cleanup: Callable[[], None] | None = None) -> None:
        """Clear this store while preserving its persistent lock inode.

        Used by administrative cleanup. The materialized view is removed
        first; truncating and syncing the held log is the empty-state commit.
        A process that later updates the store reuses this same lock domain.
        Corrupt log contents do not prevent an explicit reset.
        """
        fd = self._open_and_lock(fcntl.LOCK_EX)
        assert fd is not None
        try:
            try:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    sync_directory(self.path.parent)
                os.ftruncate(fd, 0)
                os.fsync(fd)
            except OSError as exc:
                raise SharedStorageError(f"cannot reset transaction store {self.path}: {exc}") from exc
            # Empty metadata is already committed. Payload cleanup follows so
            # an interruption can leave only harmless orphan payloads, never
            # a journal that names state cleanup already removed.
            if cleanup is not None:
                try:
                    cleanup()
                except OSError as exc:
                    raise SharedStorageError(f"metadata reset committed for {self.path}, but payload cleanup failed: {exc}") from exc
            self._cleanup_temporaries_unlocked()
        finally:
            self._unlock(fd)
