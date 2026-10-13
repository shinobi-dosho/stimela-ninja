"""Bounded physical ledgers keep canonical identities and publication rules."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from shinobi import dataset_state as state
from shinobi.dataset_lifecycle import structural_signature
from shinobi.state_codecs import LosslessProfile
from tests.test_dataset_lifecycle import _snapshot
from tests.test_dataset_state import _versions


@pytest.fixture
def physical(tmp_path):
    observation = _snapshot(tmp_path / "source.ms").observations[0]
    return state.StateRepresentation(
        representation_id="sha256:" + "0" * 64,
        state_id="msutils-logical-hash/v1:" + "a" * 64,
        msv4_id="msutils-logical-hash/v1:" + "b" * 64,
        payload_id="msutils-logical-hash/v1:" + "c" * 64,
        versions=_versions(),
        source_spelling=str(observation.root),
        source=observation.root,
        observation=observation,
        structural_signature=structural_signature(observation),
        capture_attempt=uuid.uuid4(),
        captured_at=datetime.now(timezone.utc),
        files=(),
        zarr_metadata={},
    )


def _install(store, physical):
    identity = state._digest(physical.model_dump(mode="json", exclude={"representation_id"}))
    physical = physical.model_copy(update={"representation_id": identity})
    directory = store._state(physical.state_id)
    rep = directory / "representations" / identity.removeprefix("sha256:")
    rep.mkdir(parents=True)
    contract = state._representation_contract(physical)
    state._write(directory / "logical.json", state.LogicalState(state_id=physical.state_id, profile=contract[0], native_model=contract[1]))
    state._write(rep / "representation.json", physical)
    return rep / "representation.json", physical


@pytest.mark.parametrize("version", [1, 2, 3])
def test_physical_manifest_read_once_and_formatting_keeps_digest(tmp_path, monkeypatch, physical, version):
    if version == 2:
        physical = state.StateRepresentationV2(**physical.model_dump(exclude={"schema_version"}), lossless_profile=LosslessProfile())
    elif version == 3:
        physical = state.StateRepresentationV3(**physical.model_dump(exclude={"schema_version"}), profile="fixed-shape-defined-or-empty/v2", native_model="msutils-native-model/v2")
    store = state.DatasetStateStore(tmp_path / "store")
    path, physical = _install(store, physical)
    original = state._read
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(state, "_read", read)
    _, _, compact = store._load(physical.state_id, physical.representation_id)
    assert reads.count(path) == 1
    assert b"\n" not in path.read_bytes()
    path.write_text(physical.model_dump_json(indent=2))
    _, _, pretty = store._load(physical.state_id, physical.representation_id)
    assert reads.count(path) == 2
    assert type(compact) is type(pretty) is type(physical)
    assert compact == pretty == physical
    assert state._digest(pretty.model_dump(mode="json", exclude={"representation_id"})) == physical.representation_id


def test_large_chunk_ledger_above_historical_budget_loads_without_dropping_entries(tmp_path, physical, record_testsuite_property):
    # Actual serialization sizes, not scaled limits: emulate a full MS payload
    # with 400k independently hashed chunk paths and no lossy ledger summary.
    count = 400_000
    files = tuple(state.FileEntry(path=f"preservation/native.zarr/MAIN/CORRECTED_RESIDUAL/c/{index:08d}/0/0", kind="file", size=1, sha256="d" * 64) for index in range(count))
    physical = physical.model_copy(update={"files": files})
    store = state.DatasetStateStore(tmp_path / "store")
    path, physical = _install(store, physical)
    compact_size = path.stat().st_size
    pretty = physical.model_dump_json(indent=2)
    pretty_size = len(pretty.encode())
    assert state.MAX_METADATA == 64 * 1024 * 1024
    assert state.MAX_REPRESENTATION_METADATA == 256 * 1024 * 1024
    assert state.MAX_METADATA < compact_size < pretty_size < state.MAX_REPRESENTATION_METADATA
    _, _, loaded = store._load(physical.state_id, physical.representation_id)
    assert loaded.files == physical.files and len(loaded.files) == count
    del loaded
    path.write_text(pretty)
    del pretty
    _, _, loaded = store._load(physical.state_id, physical.representation_id)
    assert loaded.files == physical.files
    assert loaded.representation_id == physical.representation_id
    record_testsuite_property("large_chunk_ledger_entries", count)
    record_testsuite_property("large_chunk_ledger_compact_bytes", compact_size)
    record_testsuite_property("large_chunk_ledger_pretty_bytes", pretty_size)


def test_metadata_budgets_are_separate_for_reads_and_writes(tmp_path, monkeypatch, physical):
    monkeypatch.setattr(state, "MAX_METADATA", 64)
    monkeypatch.setattr(state, "MAX_REPRESENTATION_METADATA", 8192)
    ordinary = tmp_path / "logical.json"
    with pytest.raises(state.StateError, match="metadata-size"):
        state._write(ordinary, physical)
    assert not ordinary.exists()
    representation = tmp_path / "representation.json"
    state._write(representation, physical)
    assert len(state._read(representation)) > 64
    ordinary.write_bytes(representation.read_bytes())
    with pytest.raises(state.StateError, match="metadata"):
        state._read(ordinary)
    monkeypatch.setattr(state, "MAX_REPRESENTATION_METADATA", representation.stat().st_size - 1)
    with pytest.raises(state.StateError, match="metadata"):
        state._read(representation)
    representation.unlink()
    with pytest.raises(state.StateError, match="metadata-size"):
        state._write(representation, physical)
    assert not representation.exists()


def test_metadata_read_stays_bounded_when_regular_file_grows(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "MAX_METADATA", 8)
    path = tmp_path / "logical.json"
    path.write_bytes(b"x")
    original = Path.open
    reads = []

    class GrowingFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            with original(path, "ab") as output:
                output.write(b"x" * 32)
            return self.stream.read(size)

    def open_file(target, *args, **kwargs):
        stream = original(target, *args, **kwargs)
        return GrowingFile(stream) if target == path and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(state.StateError, match="metadata exceeds"):
        state._read(path)
    assert reads == [9]


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_metadata_read_refuses_nonregular_entries(tmp_path, kind):
    path = tmp_path / "representation.json"
    if kind == "directory":
        path.mkdir()
    else:
        target = tmp_path / "target"
        target.write_text("{}")
        path.symlink_to(target)
    with pytest.raises(state.StateError, match="regular metadata file"):
        state._read(path)


@pytest.mark.parametrize("damage", ["duplicate-version", "duplicate-entry", "unsafe-path", "unknown-version", "unhashable-version"])
def test_physical_manifest_strict_validation_before_inventory_access(tmp_path, physical, damage):
    store = state.DatasetStateStore(tmp_path / "store")
    path, physical = _install(store, physical)
    raw = physical.model_dump_json()
    if damage == "duplicate-version":
        raw = raw.replace('"schema_version":', '"schema_version":"shinobi-state-representation/v1","schema_version":', 1)
    elif damage == "duplicate-entry":
        raw = raw.replace('"files":[]', '"files":[{"path":"x","kind":"file","size":1,"sha256":"' + "d" * 64 + '","size":2}]')
    else:
        data = json.loads(raw)
        if damage == "unsafe-path":
            data["files"] = [{"path": "../outside", "kind": "file", "size": 1, "sha256": "d" * 64}]
        else:
            data["schema_version"] = "shinobi-state-representation/v4" if damage == "unknown-version" else []
        raw = json.dumps(data)
    path.write_text(raw)
    if damage == "unsafe-path":
        with pytest.raises(ValidationError, match="normalized relative"):
            store._load(physical.state_id, physical.representation_id)
    else:
        with pytest.raises(state.StateError, match="duplicate JSON key" if damage.startswith("duplicate") else "unsupported representation"):
            store._load(physical.state_id, physical.representation_id)


def test_oversized_representation_refuses_before_publication(tmp_path, monkeypatch):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    from tests._state_ms import make_state_ms

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    monkeypatch.setattr(state, "MAX_REPRESENTATION_METADATA", 64)
    source = make_state_ms(tmp_path / "source.ms")
    store = state.DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    with pytest.raises(state.StateError, match="metadata-size"):
        store.export(source, block_rows=2)
    assert store.list() == []
    assert list((store.root / "staging").iterdir()) == []
    attempts = list((store.root / "attempts").glob("*.json"))
    assert len(attempts) == 1
    attempt = state.read_state_attempt(attempts[0])
    assert attempt.finished and attempt.phase == "refused"
    assert source.is_dir()


@pytest.mark.parametrize("opened_kind", ["oversized", "nonregular"])
def test_metadata_read_checks_opened_file_before_reading(tmp_path, monkeypatch, opened_kind):
    monkeypatch.setattr(state, "MAX_METADATA", 8)
    path = tmp_path / "logical.json"
    path.write_bytes(b"x")
    original = Path.open
    reads = []

    class CheckedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            raise AssertionError("opened metadata must be checked before reading")

    def changed_open(target, *args, **kwargs):
        if target != path or args != ("rb",):
            return original(target, *args, **kwargs)
        if opened_kind == "nonregular":
            return CheckedFile(original(Path("/dev/null"), "rb"))
        stream = original(target, *args, **kwargs)
        with original(target, "ab") as output:
            output.write(b"x" * 32)
        return CheckedFile(stream)

    monkeypatch.setattr(Path, "open", changed_open)
    with pytest.raises(state.StateError, match="regular metadata file"):
        state._read(path)
    assert reads == []
