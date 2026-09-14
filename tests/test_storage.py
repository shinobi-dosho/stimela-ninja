"""Cross-process contracts for mutable shared-storage metadata."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi.storage as storage
from shinobi.cache import CacheManifest
from shinobi.results import StepResult
from shinobi.snapshots import Chain, ChainJournal
from shinobi.storage import JsonFileStore, SharedStorageError, ofd_lock


class _NoInputs(BaseModel):
    pass


class _ValueOutput(BaseModel):
    value: int


def _update_store(path: str, key: str, ready, start) -> None:
    store = JsonFileStore(Path(path))
    ready.put(key)
    start.wait()

    def mutate(data):
        # Widen the read-modify-write window: without an inter-process lock,
        # all workers read the same initial object and only one key survives.
        time.sleep(0.02)
        data[key] = key

    store.update(mutate)


def _update_journal(root: str, index: int, ready, start) -> None:
    journal = ChainJournal(Path(root))
    cid = f"chain-{index}"
    ready.put(cid)
    start.wait()
    journal.update_chain(cid, lambda _old: Chain(dev=1, ino=index, ctime_ns=index, path=f"/data/{cid}"))


def _record_manifest(path: str, index: int, ready, start) -> None:
    manifest = CacheManifest(Path(path))
    ready.put(index)
    start.wait()
    manifest.record(
        f"step-{index}",
        f"key-{index}",
        StepResult(name=f"step-{index}", returncode=0, inputs=_NoInputs(), outputs=_ValueOutput(value=index)),
    )


def _update_after_fork(path: str) -> None:
    JsonFileStore(Path(path)).update(lambda data: data.update(child=True))


def _wait_after_fork(started, release) -> None:
    started.set()
    release.wait(timeout=10)


def _run_concurrently(target, args: list[tuple], count: int) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    start = context.Event()
    processes = [context.Process(target=target, args=(*one, ready, start)) for one in args]
    for process in processes:
        process.start()
    for _ in range(count):
        ready.get(timeout=10)
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0


def test_independent_process_updates_are_all_retained(tmp_path):
    path = tmp_path / "manifest.json"
    count = 12
    _run_concurrently(_update_store, [(str(path), f"step-{i}") for i in range(count)], count)
    assert JsonFileStore(path).read() == {f"step-{i}": f"step-{i}" for i in range(count)}


def test_independently_constructed_stores_share_one_process_lock(tmp_path):
    path = tmp_path / "manifest.json"
    count = 12
    start = threading.Event()

    def worker(index):
        store = JsonFileStore(path)
        start.wait()

        def mutate(data):
            time.sleep(0.01)
            data[f"step-{index}"] = index

        store.update(mutate)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()
    assert JsonFileStore(path).read() == {f"step-{index}": index for index in range(count)}


def test_different_paths_to_one_lock_inode_still_exclude(tmp_path):
    first_path = tmp_path / "first" / "manifest.json"
    second_path = tmp_path / "second" / "manifest.json"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_lock = first_path.with_name(first_path.name + ".lock")
    second_lock = second_path.with_name(second_path.name + ".lock")
    first_lock.touch()
    os.link(first_lock, second_lock)

    entered = threading.Event()
    release = threading.Event()
    contender_entered = threading.Event()

    def hold(data):
        entered.set()
        release.wait(timeout=10)
        data["holder"] = True

    holder = threading.Thread(target=lambda: JsonFileStore(first_path).update(hold))
    contender = threading.Thread(target=lambda: JsonFileStore(second_path).update(lambda data: (contender_entered.set(), data.update(contender=True))))
    holder.start()
    assert entered.wait(timeout=10)
    contender.start()
    assert not contender_entered.wait(timeout=0.1)
    release.set()
    holder.join(timeout=10)
    contender.join(timeout=10)
    assert not holder.is_alive() and not contender.is_alive()
    assert contender_entered.is_set()


def test_closing_an_unrelated_descriptor_does_not_release_an_ofd_lock(tmp_path):
    path = tmp_path / "held.lock"
    holder = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    unrelated = contender = None
    try:
        ofd_lock(holder, fcntl.F_WRLCK)
        unrelated = os.open(path, os.O_RDWR)
        os.close(unrelated)
        unrelated = None
        contender = os.open(path, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            ofd_lock(contender, fcntl.F_WRLCK, blocking=False)
    finally:
        if unrelated is not None:
            os.close(unrelated)
        if contender is not None:
            os.close(contender)
        os.close(holder)


@pytest.mark.filterwarnings("ignore:This process.*fork.*:DeprecationWarning")
def test_fork_while_transaction_is_held_does_not_deadlock_child(tmp_path):
    path = tmp_path / "manifest.json"
    entered = threading.Event()
    release = threading.Event()

    def hold(data):
        entered.set()
        release.wait(timeout=10)
        data["parent"] = True

    thread = threading.Thread(target=lambda: JsonFileStore(path).update(hold))
    thread.start()
    assert entered.wait(timeout=10)
    process = multiprocessing.get_context("fork").Process(target=_update_after_fork, args=(str(path),))
    process.start()
    release.set()
    thread.join(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    assert JsonFileStore(path).read() == {"parent": True, "child": True}


@pytest.mark.filterwarnings("ignore:This process.*fork.*:DeprecationWarning")
def test_forked_child_does_not_prolong_parent_lock_lifetime(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    fd = store._open_and_lock(fcntl.LOCK_EX)
    assert fd is not None
    context = multiprocessing.get_context("fork")
    started = context.Event()
    release = context.Event()
    child = context.Process(target=_wait_after_fork, args=(started, release))
    child.start()
    assert started.wait(timeout=10)
    store._unlock(fd)

    contender = os.open(store.lock_path, os.O_RDWR)
    try:
        ofd_lock(contender, fcntl.F_WRLCK, blocking=False)
    finally:
        os.close(contender)
        release.set()
        child.join(timeout=10)
    assert child.exitcode == 0


def test_independent_process_cache_records_are_all_retained(tmp_path):
    path = tmp_path / "manifest.json"
    count = 12
    _run_concurrently(_record_manifest, [(str(path), i) for i in range(count)], count)
    manifest = CacheManifest(path)
    assert {name: manifest.entry(name)["cache_key"] for name in (f"step-{i}" for i in range(count))} == {f"step-{i}": f"key-{i}" for i in range(count)}


def test_independent_process_journal_updates_are_all_retained(tmp_path):
    root = tmp_path / "snapshots"
    count = 12
    _run_concurrently(_update_journal, [(str(root), i) for i in range(count)], count)
    assert set(ChainJournal(root).all_chains()) == {f"chain-{i}" for i in range(count)}


def test_interrupted_replacement_leaves_the_previous_commit_readable(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(old="committed"))

    def fail_replace(_source, _target):
        raise OSError("injected refresh failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.warns(RuntimeWarning, match="could not refresh compatibility view"):
        store.update(lambda data: data.update(new="not committed"))

    assert json.loads(path.read_text()) == {"old": "committed"}
    # The checksummed transaction was committed before refreshing the legacy
    # JSON view, so a new reader recovers it rather than false-hitting old data.
    assert JsonFileStore(path).read() == {"old": "committed", "new": "not committed"}
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_partial_trailing_transaction_is_ignored(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(committed=1))
    with store.lock_path.open("ab") as stream:
        stream.write(b'deadbeef {"version":1')
    assert JsonFileStore(path).read() == {"committed": 1}
    JsonFileStore(path).update(lambda data: data.update(recovered=2))
    assert JsonFileStore(path).read() == {"committed": 1, "recovered": 2}


def test_corrupt_complete_transaction_fails_closed(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(committed=1))
    with store.lock_path.open("ab") as stream:
        stream.write(b"deadbeef {}\n")
    with pytest.raises(SharedStorageError, match="corrupt committed transaction"):
        JsonFileStore(path).read()


def test_temporary_names_do_not_depend_on_pid(tmp_path, monkeypatch):
    store = JsonFileStore(tmp_path / "manifest.json")
    monkeypatch.setattr(os, "getpid", lambda: 7)
    store.update(lambda data: data.update(first=1))
    store.update(lambda data: data.update(second=2))
    assert store.read() == {"first": 1, "second": 2}
    assert not (tmp_path / "manifest.json.tmp7").exists()


def test_existing_json_payload_needs_no_migration(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"legacy": {"cache_key": "old"}}')
    assert JsonFileStore(path).read() == {"legacy": {"cache_key": "old"}}


def test_first_update_uses_collaborative_permissions(tmp_path):
    path = tmp_path / "manifest.json"
    previous = os.umask(0o002)
    try:
        JsonFileStore(path).update(lambda data: data.update(value=1))
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o664
    assert path.with_name("manifest.json.lock").stat().st_mode & 0o777 == 0o664


def test_read_only_lock_allows_lookup_without_writing(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(value=1))
    store.lock_path.chmod(0o444)
    try:
        assert JsonFileStore(path).read() == {"value": 1}
    finally:
        store.lock_path.chmod(0o644)


def test_logged_lookup_does_not_require_compatibility_view_access(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(value=1))
    path.chmod(0)
    store.lock_path.chmod(0o444)
    try:
        assert JsonFileStore(path).read() == {"value": 1}
    finally:
        path.chmod(0o644)
        store.lock_path.chmod(0o644)


def test_reading_legacy_json_does_not_create_a_lock(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"legacy": true}')
    store = JsonFileStore(path)
    assert store.read() == {"legacy": True}
    assert not store.lock_path.exists()


def test_old_writer_is_ignored_after_one_way_upgrade(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(committed=1))
    path.write_text('{"old-writer": 2}')
    assert store.read() == {"committed": 1}


def test_store_syncs_immediate_parent_not_every_ancestor(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(storage, "sync_directory", lambda path: synced.append(Path(path)))
    monkeypatch.setattr(storage, "sync_directory_chain", lambda _path: pytest.fail("mutable store used directory-chain sync"))
    path = tmp_path / "cache" / "manifest.json"
    JsonFileStore(path).update(lambda data: data.update(value=1))
    assert synced
    assert set(synced) <= {tmp_path, path.parent}


def test_read_system_call_failure_is_a_shared_storage_error(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    JsonFileStore(path).update(lambda data: data.update(value=1))
    monkeypatch.setattr(os, "pread", lambda *_args: (_ for _ in ()).throw(OSError("read failed")))
    with pytest.raises(SharedStorageError, match="cannot read transaction store"):
        JsonFileStore(path).read()


def test_directory_sync_failure_after_commit_is_only_a_warning(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(first=1))
    real_sync = storage.sync_directory
    calls = {"n": 0}

    def fail_refresh_sync(directory):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("sync failed")
        real_sync(directory)

    monkeypatch.setattr(storage, "sync_directory", fail_refresh_sync)
    with pytest.warns(RuntimeWarning, match="could not refresh compatibility view"):
        store.update(lambda data: data.update(second=2))
    assert store.read() == {"first": 1, "second": 2}


def test_lock_wait_has_a_finite_timeout(tmp_path):
    path = tmp_path / "manifest.json"
    entered = threading.Event()
    release = threading.Event()

    def hold(data):
        entered.set()
        release.wait(timeout=10)
        data["held"] = True

    thread = threading.Thread(target=lambda: JsonFileStore(path).update(hold))
    thread.start()
    assert entered.wait(timeout=10)
    try:
        with pytest.raises(SharedStorageError, match="timed out"):
            JsonFileStore(path, lock_timeout=0.05).update(lambda data: data.update(contender=True))
    finally:
        release.set()
        thread.join(timeout=10)


def test_reset_preserves_lock_inode_and_recovers_corruption(tmp_path):
    path = tmp_path / "manifest.json"
    store = JsonFileStore(path)
    store.update(lambda data: data.update(value=1))
    before = store.lock_path.stat().st_ino
    with store.lock_path.open("ab") as stream:
        stream.write(b"corrupt complete record\n")
    stale = tmp_path / ".manifest.json.stale.tmp"
    stale.write_text("partial")

    store.reset()

    assert store.lock_path.stat().st_ino == before
    assert store.read() == {}
    assert not path.exists()
    assert not stale.exists()


def test_lock_acquisition_failure_is_explicit(tmp_path, monkeypatch):
    store = JsonFileStore(tmp_path / "manifest.json")

    def fail_lock(_fd, _command, _argument):
        raise OSError("locking unavailable")

    monkeypatch.setattr(fcntl, "fcntl", fail_lock)
    with pytest.raises(SharedStorageError, match="cannot establish transaction lock"):
        store.update(lambda data: data.update(value=1))
