"""Exact replay permits regenerated table filenames with distinct durable evidence."""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from shinobi import dataset_state as state
from shinobi.dataset_lifecycle import _structural_signature_document, observe_dataset, structural_signature
from shinobi.ownership import inspect_ownership
from shinobi.storage import JsonFileStore
from tests.test_dataset_lifecycle import _snapshot
from tests.test_dataset_state import _versions


def test_frozen_structural_signature_golden_document():
    observation = _snapshot(Path("/frozen/source.ms")).observations[0]
    assert structural_signature(observation) == "d94da73de493cf37e11ec03bf984d461a0bef02d851c2bbc0464b347d917ed85"
    document = _structural_signature_document(observation)
    assert document["resources"][0]["files"] == ["table.dat"]
    changed = observation.model_copy(
        update={"closure": observation.closure.model_copy(update={"resources": (observation.closure.resources[0].model_copy(update={"table_files": ("table.dat", "table.f9")}),)})}
    )
    assert structural_signature(changed) != structural_signature(observation)
    assert state._state_replay_structure_matches(observation, changed)
    assert _structural_signature_document(observation) == document


@pytest.mark.parametrize("change", ["managers", "members", "path", "external", "rows", "columns", "keywords", "subtables", "closure-profile"])
def test_state_replay_structure_refuses_every_other_frozen_input(tmp_path, change):
    source = _snapshot(tmp_path / "source.ms").observations[0]
    restored = source
    resource = source.closure.resources[0]
    if change in {"managers", "members", "path", "external"}:
        updates = {
            "managers": {"storage_managers": ("TiledColumnStMan",)},
            "members": {"members": ("OTHER",)},
            "path": {"path": source.root / "OTHER"},
            "external": {"external_to_root": True},
        }[change]
        restored = source.model_copy(update={"closure": source.closure.model_copy(update={"resources": (resource.model_copy(update=updates),)})})
    elif change == "closure-profile":
        restored = source.model_copy(update={"closure_profile": "msv2-dataset-closure/v1" if source.closure_profile.endswith("/v2") else "msv2-dataset-closure/v2"})
    else:
        field = {"rows": "nrows", "columns": "columns", "keywords": "keywords", "subtables": "subtables"}[change]
        value = (source.descriptor.nrows or 0) + 1 if field == "nrows" else (*getattr(source.descriptor, field), "ADDED")
        restored = source.model_copy(update={"descriptor": source.descriptor.model_copy(update={field: value})})
    assert not state._state_replay_structure_matches(source, restored)


@pytest.fixture
def ready_attempt(tmp_path):
    provenance = state.StateProvenance(
        state_id="msutils-logical-hash/v1:" + "a" * 64,
        representation_id="sha256:" + "b" * 64,
        msv4_id="msutils-logical-hash/v1:" + "c" * 64,
        payload_id="msutils-logical-hash/v1:" + "d" * 64,
        producer_versions=_versions(),
        actual_fidelity="exact-logical",
        cache_decision="selected-representation",
        materialization_decision="validated",
        replay_decision="validated",
        structural_signature="e" * 64,
    )
    return state.StateAttempt(
        schema_version="shinobi-state-attempt/v3",
        attempt_id=uuid.uuid4(),
        operation="materialize",
        store=tmp_path / "store",
        authority=tmp_path,
        registry=tmp_path / "owners.json",
        destination=tmp_path / "target.ms",
        stage=tmp_path / "stage",
        parent_identity=(1, 2),
        candidate_identity=(3, 4),
        state_id=provenance.state_id,
        representation_id=provenance.representation_id,
        versions=_versions(),
        phase="ready",
        structural_signature="e" * 64,
        restored_structural_signature="f" * 64,
        provenance=provenance,
    )


@pytest.mark.parametrize("phase", ["ready", "published", "committed"])
def test_v3_ready_phases_require_complete_validated_source_and_restored_evidence(ready_attempt, phase):
    raw = {**ready_attempt.model_dump(), "phase": phase}
    assert state.StateAttempt.model_validate(raw).restored_structural_signature == "f" * 64
    for field in ("restored_structural_signature", "structural_signature", "provenance", "versions", "candidate_identity", "state_id", "representation_id"):
        with pytest.raises(ValidationError):
            state.StateAttempt.model_validate({**raw, field: None})
    for change in ({"operation": "export"}, {"restored_structural_signature": "invalid"}, {"structural_signature": "1" * 64}):
        with pytest.raises(ValidationError):
            state.StateAttempt.model_validate({**raw, **change})
    requested = ready_attempt.provenance.model_copy(update={"actual_fidelity": None, "materialization_decision": "requested", "replay_decision": "requested"})
    with pytest.raises(ValidationError, match="validated exact replay"):
        state.StateAttempt.model_validate({**raw, "provenance": requested})
    same = state.StateAttempt.model_validate({**raw, "restored_structural_signature": "e" * 64})
    assert same.restored_structural_signature == same.structural_signature
    failed = state.StateAttempt.model_validate({**raw, "phase": "failed"})
    assert failed.restored_structural_signature == "f" * 64


@pytest.mark.parametrize("version", ["shinobi-state-attempt/v1", "shinobi-state-attempt/v2"])
def test_legacy_attempt_serialization_omits_new_field_and_refuses_new_evidence(ready_attempt, version):
    raw = {**ready_attempt.model_dump(), "schema_version": version, "phase": "interrupted", "restored_structural_signature": None}
    if version.endswith("/v1"):
        raw["provenance"] = None
    legacy = state.StateAttempt.model_validate(raw)
    data = legacy.model_dump(mode="json")
    assert "restored_structural_signature" not in data
    assert "restored_structural_signature" not in legacy.model_dump_json()
    assert legacy.model_dump_json() == json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    assert state.StateAttempt.model_validate_json(legacy.model_dump_json()) == legacy
    with pytest.raises(ValidationError, match="legacy state attempts"):
        state.StateAttempt.model_validate({**raw, "restored_structural_signature": "f" * 64})


@pytest.fixture(scope="module")
def gap_export(tmp_path_factory):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    import numpy as np
    from casacore.tables import table, makearrcoldesc
    from tests._state_ms import make_state_ms

    root = tmp_path_factory.mktemp("gap-state")
    source = make_state_ms(root / "source.ms")
    with table(str(source), readonly=False, ack=False) as tab:
        # Removing an earlier manager leaves table.f* numbering gaps.
        for name in ("REMOVED_SPECTRUM", "CORRECTED_DATA"):
            tab.addcols(
                makearrcoldesc(name, 0j, shape=[4, 4], valuetype="complex"), {"TYPE": "TiledColumnStMan", "NAME": name, "COLUMNS": [name], "SPEC": {"DEFAULTTILESHAPE": [4, 4, 2]}}
            )
            tab.putcol(name, tab.getcol("DATA") + np.complex64(1 + 2j))
        tab.removecols("REMOVED_SPECTRUM")
        assert next(manager["SEQNR"] for manager in tab.getdminfo().values() if manager["NAME"] == "CORRECTED_DATA") == 2
    return _export_source(root, source)


def _export_source(root, source):
    observation = observe_dataset(source, root)
    source_id = state.adapter.native_id(source)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(root / "owners.json"))
        store = state.DatasetStateStore(root / "store", cache_dir=root / "cache")
        exported = store.export(source, block_rows=2)
    assert exported.state_id == source_id
    assert state.read_state_attempt(exported.attempt).schema_version == "shinobi-state-attempt/v2"
    source.rename(root / "hidden.ms")
    return observation, store, exported


@pytest.fixture
def gap_state(gap_export, tmp_path, monkeypatch):
    observation, original, exported = gap_export
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    store = state.DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    shutil.copytree(original.root / "states", store.root / "states")
    return observation, store, exported


def test_real_native_replay_records_distinct_structures_without_changing_source_provenance(gap_state, tmp_path):
    source, store, exported = gap_state
    target = tmp_path / "target.ms"
    result = store.materialize(exported.state_id, target, representation_id=exported.representation_id)
    restored = observe_dataset(target, tmp_path)
    assert structural_signature(source) != structural_signature(restored)
    assert state._state_replay_structure_matches(source, restored)
    assert state.adapter.native_id(target) == exported.state_id
    attempt = state.read_state_attempt(result.attempt)
    assert attempt.schema_version == "shinobi-state-attempt/v3"
    assert attempt.structural_signature == attempt.provenance.structural_signature == structural_signature(source)
    assert attempt.restored_structural_signature == structural_signature(restored)
    assert attempt.finished and attempt.phase == "committed"


def _crash_after_publication(store, exported, target):
    code = """
import os, signal, sys
import shinobi.dataset_state as state
original = state.publish_directory
def publish(source, destination):
    original(source, destination)
    os.kill(os.getpid(), signal.SIGKILL)
state.publish_directory = publish
state.DatasetStateStore(sys.argv[1], cache_dir=sys.argv[4]).materialize(sys.argv[2], sys.argv[3])
"""
    process = subprocess.run([sys.executable, "-c", code, str(store.root), exported.state_id, str(target), str(store.cache_dir)], capture_output=True, timeout=60)
    assert process.returncode == -signal.SIGKILL, process.stderr.decode()
    paths = list((store.root / "attempts").glob("*.json"))
    assert len(paths) == 1
    attempt = state.read_state_attempt(paths[0])
    assert attempt.phase == "ready" and not attempt.finished and attempt.restored_structural_signature is not None
    return paths[0], attempt


def test_v3_recovery_revalidates_published_regenerated_filenames(gap_state, tmp_path):
    source, store, exported = gap_state
    target = tmp_path / "target.ms"
    _, attempt = _crash_after_publication(store, exported, target)
    assert structural_signature(observe_dataset(target, tmp_path)) != structural_signature(source)
    recovered = store.recover(target)
    assert len(recovered) == 1 and recovered[0].phase == "committed" and recovered[0].finished
    assert recovered[0].restored_structural_signature == attempt.restored_structural_signature
    assert not attempt.stage.exists()
    assert inspect_ownership(tmp_path, str(attempt.attempt_id)).liveness == "free"
    assert store.recover(target) == []


@pytest.mark.parametrize("damage", ["recorded-signature", "columns", "rows", "table-files"])
def test_v3_recovery_refuses_changed_published_evidence_and_preserves_candidate(gap_state, tmp_path, damage):
    _, store, exported = gap_state
    target = tmp_path / "target.ms"
    record, attempt = _crash_after_publication(store, exported, target)
    if damage == "recorded-signature":
        JsonFileStore(record).update(lambda data: data.update(restored_structural_signature="0" * 64))
    elif damage == "table-files":
        (target / "table.f999").write_bytes(b"unused")
        assert state.adapter.native_id(target) == exported.state_id
        assert structural_signature(observe_dataset(target, tmp_path)) != attempt.restored_structural_signature
    else:
        from casacore.tables import table, makescacoldesc

        with table(str(target), readonly=False, ack=False) as tab:
            if damage == "columns":
                tab.addcols(makescacoldesc("UNDECLARED", 0.0))
            else:
                tab.addrows(1)
    with pytest.raises(state.StateError, match="recovery-postcondition"):
        store.recover(target)
    assert target.is_dir() and attempt.stage.is_dir()
    assert not state.read_state_attempt(record).finished


@pytest.mark.parametrize("version", ["shinobi-state-attempt/v1", "shinobi-state-attempt/v2"])
def test_legacy_recovery_keeps_exact_signature_comparison(gap_state, tmp_path, version):
    _, store, exported = gap_state
    target = tmp_path / "target.ms"
    record, attempt = _crash_after_publication(store, exported, target)

    def legacy(data):
        data.pop("restored_structural_signature")
        data["schema_version"] = version
        if version.endswith("/v1"):
            data["provenance"] = None

    JsonFileStore(record).update(legacy)
    assert "restored_structural_signature" not in state.read_state_attempt(record).model_dump()
    with pytest.raises(state.StateError, match="recovery-postcondition"):
        store.recover(target)
    assert target.is_dir() and attempt.stage.is_dir()


def test_replay_native_identity_is_independent_of_structural_comparison(gap_state, tmp_path, monkeypatch):
    _, store, exported = gap_state
    from casacore.tables import table

    original = state.adapter.materialize

    def change_value(msv4, bundle, destination, state_id):
        original(msv4, bundle, destination, state_id)
        with table(str(destination), readonly=False, ack=False) as tab:
            values = tab.getcell("DATA", 0)
            values[0, 0] += 1
            tab.putcell("DATA", 0, values)

    monkeypatch.setattr(state.adapter, "materialize", change_value)
    target = tmp_path / "target.ms"
    with pytest.raises(state.StateError, match="postcondition"):
        store.materialize(exported.state_id, target)
    assert not target.exists()
    attempt = state.read_state_attempt(next((store.root / "attempts").glob("*.json")))
    assert attempt.phase == "refused" and attempt.finished
    assert attempt.restored_structural_signature is None
    assert attempt.provenance.actual_fidelity is None


@pytest.fixture(scope="module")
def ordinary_export(tmp_path_factory):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    from tests._state_ms import make_state_ms

    root = tmp_path_factory.mktemp("ordinary-replay")
    return _export_source(root, make_state_ms(root / "source.ms"))


@pytest.mark.parametrize("version", ["shinobi-state-attempt/v1", "shinobi-state-attempt/v2"])
def test_legacy_published_recovery_still_commits_matching_exact_structure(ordinary_export, tmp_path, monkeypatch, version):
    source, original, exported = ordinary_export
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    store = state.DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    shutil.copytree(original.root / "states", store.root / "states")
    target = tmp_path / "target.ms"
    record, attempt = _crash_after_publication(store, exported, target)
    assert structural_signature(observe_dataset(target, tmp_path)) == structural_signature(source)

    def legacy(data):
        data.pop("restored_structural_signature")
        data["schema_version"] = version
        if version.endswith("/v1"):
            data["provenance"] = None

    JsonFileStore(record).update(legacy)
    recovered = store.recover(target)
    assert len(recovered) == 1 and recovered[0].finished and recovered[0].phase == "committed"
    assert recovered[0].schema_version == version
    assert "restored_structural_signature" not in recovered[0].model_dump(mode="json")
    assert not attempt.stage.exists()
