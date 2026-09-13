"""Shared-filesystem publication and mutable-metadata transactions.

Immutable worker observations use atomic no-clobber publication; mutable cache
indexes and snapshot journals use :class:`JsonFileStore`.  Both routes share
the durability primitive here, but only mutable metadata takes a short-lived
transaction lock.  That lock is deliberately not ownership of any scientific
data a step may mutate.

The transaction contract requires 64-bit Linux open-file-description locks
which conflict with the POSIX locks used by NFSv4, plus atomic same-directory
``os.replace`` and honoured file/directory ``fsync``.  System-call failures are
fatal rather than silently weakening the contract.  A local process cannot
prove that a remote node observes its locks, so supported deployments must run
the physical cross-node probe in ``tests/slurm_physical/run_m2.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
import threading
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar


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
        # python-build-standalone 3.13 on the supported test host omits these
        # constants even though its Linux kernel implements them. Values 37/38
        # are verified only for these two 64-bit Linux architecture families.
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
    # copies or its next transaction would wait forever on a lock it inherited.
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


class JsonFileStore:
    """One backward-compatible JSON object with serialized transactions.

    A persistent sibling ``.lock`` file is record-locked before every read or
    read-modify-write transaction.  It must never be unlinked: replacing a
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

    The JSON payload and target filename are unchanged, so stores written by
    earlier Shinobi versions remain readable without migration.
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")

    def _read_base_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _read_unlocked(self, fd: int) -> tuple[dict[str, Any], bool, int]:
        data = self._read_base_unlocked()
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
            return data, False, 0
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

    def _open_and_lock(self, operation: int) -> int:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _open_lock_fds_guard:
                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                _open_lock_fds.add(fd)
            try:
                ofd_lock(fd, fcntl.F_RDLCK if operation == fcntl.LOCK_SH else fcntl.F_WRLCK)
            except BaseException:
                self._close(fd)
                raise
            return fd
        except OSError as exc:
            raise SharedStorageError(f"cannot establish transaction lock {self.lock_path}: {exc}") from exc

    @staticmethod
    def _close(fd: int) -> None:
        with _open_lock_fds_guard:
            os.close(fd)
            _open_lock_fds.discard(fd)

    @classmethod
    def _unlock(cls, fd: int) -> None:
        try:
            ofd_lock(fd, fcntl.F_UNLCK, blocking=False)
        finally:
            cls._close(fd)

    def read(self) -> dict[str, Any]:
        """Return one committed version under a shared process/node lock."""
        # Preserve the old miss behaviour: merely checking an unused cache
        # does not create its directory or lock sidecar.
        if not self.path.parent.exists():
            return {}
        fd = self._open_and_lock(fcntl.LOCK_SH)
        try:
            data, _logged, _committed_size = self._read_unlocked(fd)
            return data
        finally:
            self._unlock(fd)

    def update(self, mutate: Callable[[dict[str, Any]], T]) -> T:
        """Apply ``mutate`` and commit its resulting in-place JSON update.

        The callback runs while the open-description lock excludes both local
        threads and remote processes. If it raises, no commit is attempted.
        """
        fd = self._open_and_lock(fcntl.LOCK_EX)
        try:
            data, logged, committed_size = self._read_unlocked(fd)
            before = deepcopy(data)
            result = mutate(data)
            changed = {key: value for key, value in data.items() if key not in before or before[key] != value}
            deleted = [key for key in before if key not in data]
            if not changed and not deleted:
                return result
            if committed_size != os.fstat(fd).st_size:
                os.ftruncate(fd, committed_size)
            record = {"version": 1, "set": changed, "delete": deleted} if logged else {"version": 1, "snapshot": data}
            self._append_unlocked(fd, record)
            self._write_unlocked(data)
            return result
        finally:
            self._unlock(fd)

    def _append_unlocked(self, fd: int, record: dict[str, Any]) -> None:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        line = hashlib.sha256(payload).hexdigest().encode() + b" " + payload + b"\n"
        os.lseek(fd, 0, os.SEEK_END)
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise SharedStorageError(f"short transaction write to {self.lock_path}")
            view = view[written:]
        os.fsync(fd)
        sync_directory_chain(self.lock_path.parent)

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            sync_directory_chain(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)
