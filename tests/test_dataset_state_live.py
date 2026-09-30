"""Real pinned-stack integration and process-death recovery acceptance tests."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from shinobi import _state_adapter as adapter
from shinobi.dataset_state import DatasetStateStore, StateAttempt, StateError, StateResult, read_state_attempt
from shinobi.ownership import acquire_workspace, inspect_ownership
from shinobi.snapshots import Chain, Marker, chain_id, get_journal
from shinobi.storage import JsonFileStore

from ._state_ms import make_state_ms

pytest.importorskip("msutils", reason="requires measurement-set group")
pytest.importorskip("xarray_ms", reason="requires measurement-set group")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-state")
    old = os.environ.get("SHINOBI_OWNERSHIP_REGISTRY")
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(root / "owners.json")
    source = make_state_ms(root / "source.ms")
    store = DatasetStateStore(root / "store", cache_dir=root / "cache")
    result = store.export(source, block_rows=2)
    if old is None:
        os.environ.pop("SHINOBI_OWNERSHIP_REGISTRY", None)
    else:
        os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = old
    return root, source, store, result


@pytest.fixture
def state(exported, tmp_path, monkeypatch):
    root, source, original, result = exported
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    shutil.copytree(original.root / "states", store.root / "states")
    return source, store, result


def test_exact_round_trip_without_source_and_cli(tmp_path, monkeypatch):
    import numpy as np
    from casacore.tables import table
    from click.testing import CliRunner
    from shinobi.cli import main

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "cache"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    runner = CliRunner()
    response = runner.invoke(main, ["state", "--store", str(store.root), "export", str(source), "--block-rows", "1", "--json"])
    assert response.exit_code == 0, response.output
    exported = StateResult.model_validate_json(response.output)
    assert store.verify(exported.state_id).verified
    physical = store._load(exported.state_id)[2]
    export_attempt = read_state_attempt(exported.attempt)
    assert export_attempt.schema_version == "shinobi-state-attempt/v2"
    assert export_attempt.provenance is not None
    assert export_attempt.provenance.cache_decision == "store-requested"
    assert export_attempt.provenance.materialization_decision == "not-requested"
    assert export_attempt.provenance.replay_decision == "not-requested"
    assert export_attempt.provenance.state_id == exported.state_id
    assert export_attempt.provenance.representation_id == exported.representation_id
    assert export_attempt.provenance.msv4_id == physical.msv4_id
    assert export_attempt.provenance.payload_id == physical.payload_id
    assert export_attempt.provenance.mapping_profile == adapter.ADAPTER
    assert export_attempt.provenance.msv4_schema == adapter.MSV4_SCHEMA
    assert export_attempt.provenance.producer_versions == physical.versions
    assert any("IrregularBaselineGrid" in item for item in physical.export_warnings)
    assert physical.zarr_metadata
    source.rename(tmp_path / "hidden.ms")
    response = runner.invoke(main, ["state", "--store", str(store.root), "materialize", exported.state_id, str(tmp_path / "created.ms"), "--json"])
    assert response.exit_code == 0, response.output
    result = StateResult.model_validate_json(response.output)
    assert result.fidelity == "exact-logical" and not result.physical_restoration
    assert adapter.native_id(tmp_path / "created.ms") == exported.state_id
    for name in ("DATA", "WEIGHT", "SIGMA", "FLAG_ROW", "TIME", "ANTENNA1", "ANTENNA2"):
        with table(str(tmp_path / "hidden.ms"), ack=False) as original, table(str(result.destination), ack=False) as restored:
            np.testing.assert_array_equal(original.getcol(name), restored.getcol(name))
            assert original.getcolkeywords(name) == restored.getcolkeywords(name)
    attempt = read_state_attempt(result.attempt)
    assert attempt.phase == "committed" and attempt.finished
    assert attempt.provenance is not None
    assert attempt.provenance.cache_decision == "selected-representation"
    assert attempt.provenance.materialization_decision == "validated"
    assert attempt.provenance.replay_decision == "validated"
    assert attempt.provenance.requested_fidelity == attempt.provenance.actual_fidelity == "exact-logical"
    assert attempt.provenance.transformation_decision == "not-requested"
    assert not attempt.stage.exists()
    assert inspect_ownership(tmp_path, str(attempt.attempt_id)).liveness == "free"
    response = runner.invoke(main, ["state", "--store", str(store.root), "verify", exported.state_id, "--json"])
    assert response.exit_code == 0, response.output
    assert json.loads(response.output)["verified"]
    response = runner.invoke(main, ["state", "--store", str(store.root), "recover", "--json"])
    assert response.exit_code == 0 and json.loads(response.output) == []


def test_different_layout_same_native_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    first = store.export(source, block_rows=1)
    second = store.export(source, block_rows=3)
    assert first.state_id == second.state_id
    assert first.representation_id != second.representation_id
    assert store._load(first.state_id)[0].name == min(first.representation_id, second.representation_id).split(":")[1]
    assert len(store.list()) == 2 and all(not item.verified for item in store.list())
    assert store._load(first.state_id, first.representation_id)[2].payload_id == store._load(second.state_id, second.representation_id)[2].payload_id
    assert store.verify(second.state_id, second.representation_id).verified
    from casacore.tables import table

    with table(str(source), readonly=False, ack=False) as main:
        main.putcell("SCAN_NUMBER", 0, 7)
    assert adapter.native_id(source) != first.state_id


def test_targeted_stack_mismatch_reports_stack_version(state, monkeypatch):
    _, store, result = state
    qualified = adapter.runtime()
    monkeypatch.setattr(adapter, "runtime", lambda: {**qualified, "zarr": "0.0"})

    with pytest.raises(StateError) as raised:
        store.verify(result.state_id)

    assert raised.value.code == "stack-version"
    with pytest.raises(StateError) as raised:
        store.materialize(result.state_id, store.root.parent / "target.ms")
    assert raised.value.code == "stack-version"
    attempts = [read_state_attempt(path) for path in (store.root / "attempts").glob("*.json")]
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.phase == "refused" and attempt.finished
    assert attempt.versions == {**qualified, "zarr": "0.0"}
    assert attempt.provenance is not None
    assert attempt.provenance.producer_versions == qualified
    assert attempt.provenance.actual_fidelity is None
    assert attempt.provenance.materialization_decision == "requested"
    assert attempt.provenance.replay_decision == "requested"


def test_missing_stack_and_state_have_stable_codes(state, monkeypatch):
    _, store, result = state

    def missing_stack():
        raise ImportError("missing qualified stack")

    monkeypatch.setattr(adapter, "runtime", missing_stack)
    with pytest.raises(StateError) as raised:
        store.verify(result.state_id)
    assert raised.value.code == "state-stack"

    missing = "msutils-logical-hash/v1:" + "f" * 64
    with pytest.raises(StateError) as raised:
        store._representation(missing, None)
    assert raised.value.code == "state-not-found"

    with pytest.raises(StateError) as raised:
        store._representation(result.state_id, "sha256:" + "f" * 64)
    assert raised.value.code == "representation-not-found"


def test_list_refuses_invalid_metadata(state):
    _, store, result = state
    rep, _, _ = store._load(result.state_id)
    (rep / "representation.json").write_text("{}")

    with pytest.raises(StateError) as raised:
        store.list()

    assert raised.value.code == "schema-version"


def test_argument_and_storage_overlap_refusals(state, tmp_path):
    _, store, result = state
    with pytest.raises(StateError) as raised:
        store.export(store.root, block_rows=1)
    assert raised.value.code == "source-overlap"
    with pytest.raises(StateError) as raised:
        store.export(tmp_path / "absent.ms", block_rows=0)
    assert raised.value.code == "block-rows"
    with pytest.raises(StateError) as raised:
        store.materialize(
            result.state_id,
            tmp_path / "target.ms",
            representation_id=result.representation_id,
            fidelity="physical",
        )
    assert raised.value.code == "fidelity"
    attempts = [read_state_attempt(path) for path in (store.root / "attempts").glob("*.json")]
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.phase == "refused" and attempt.finished
    assert attempt.requested_state_id == result.state_id
    assert attempt.requested_representation_id == result.representation_id
    assert attempt.requested_fidelity == "physical"


def test_pending_mutation_refuses_export(state):
    source, store, _ = state
    info = source.stat()
    marker = Marker(step_path="pipe.flag", field="ms", cache_key=None, run_id="interrupted", started_at=0.0)
    get_journal(str(store.cache_dir)).update_chain(
        chain_id(source),
        lambda _old: Chain(dev=info.st_dev, ino=info.st_ino, ctime_ns=info.st_ctime_ns, path=str(source), marker=marker),
    )

    with pytest.raises(StateError) as raised:
        store.export(source)

    assert raised.value.code == "pending-mutation"


@pytest.mark.parametrize("damage", ["chunk", "missing", "extra", "extra-directory", "symlink", "manifest-version", "missing-version", "adapter", "stack", "ledger-path"])
def test_corrupt_and_incompatible_state_refuses_before_write(state, tmp_path, damage):
    source, store, result = state
    rep, _, physical = store._load(result.state_id)
    manifest = rep / "representation.json"
    chunk = next(path for path in (rep / "preservation" / "native.zarr").rglob("*") if path.is_file() and path.name != "zarr.json")
    if damage == "chunk":
        chunk.write_bytes(b"corrupt")
    elif damage == "missing":
        chunk.unlink()
    elif damage == "extra":
        (rep / "unknown").write_text("not inventoried")
    elif damage == "extra-directory":
        (rep / "unknown").mkdir()
    elif damage == "symlink":
        (rep / "link").symlink_to(source)
    else:
        data = json.loads(manifest.read_text())
        if damage == "manifest-version":
            data["schema_version"] = "shinobi-state-representation/v2"
        elif damage == "missing-version":
            del data["schema_version"]
        elif damage == "adapter":
            data["adapter"] = "unqualified/v1"
        elif damage == "stack":
            data["versions"]["zarr"] = "0.0"
        else:
            data["files"][0]["path"] = "../escape"
        manifest.write_text(json.dumps(data))
    expected_codes = {
        "chunk": "physical-integrity",
        "missing": "physical-integrity",
        "extra": "physical-integrity",
        "extra-directory": "physical-integrity",
        "symlink": "entry-kind",
        "manifest-version": "state-contract",
        "missing-version": "schema-version",
        "adapter": "state-contract",
        "stack": "representation-digest",
        "ledger-path": "state-contract",
    }
    with pytest.raises(StateError) as raised:
        store.materialize(result.state_id, tmp_path / "target.ms")
    assert raised.value.code == expected_codes[damage]
    assert not (tmp_path / "target.ms").exists()
    assert not list(tmp_path.glob(".shinobi-state-*"))
    attempts = [read_state_attempt(path) for path in (store.root / "attempts").glob("*.json")]
    assert len(attempts) == 1 and attempts[0].phase == "refused" and attempts[0].finished
    manifest_damage = {"manifest-version", "missing-version", "adapter", "stack", "ledger-path"}
    assert (attempts[0].provenance is None) == (damage in manifest_damage)
    if damage not in manifest_damage:
        assert attempts[0].provenance.actual_fidelity is None
        assert attempts[0].provenance.materialization_decision == "requested"
        assert attempts[0].provenance.replay_decision == "requested"


@pytest.mark.parametrize("target_kind", ["empty", "nonempty", "file", "symlink", "dangling", "source", "inside-store", "above-store", "source-alias"])
def test_destination_refusals(state, tmp_path, target_kind):
    source, store, result = state
    target = tmp_path / "target.ms"
    if target_kind in {"empty", "nonempty"}:
        target.mkdir()
        if target_kind == "nonempty":
            (target / "keep").write_text("keep")
    elif target_kind == "file":
        target.write_text("keep")
    elif target_kind in {"symlink", "dangling"}:
        target.symlink_to(source if target_kind == "symlink" else tmp_path / "absent")
    elif target_kind == "source":
        target = source
    elif target_kind == "inside-store":
        target = store.root / "target.ms"
    elif target_kind == "above-store":
        target = store.root.parent
    else:
        (tmp_path / "alias").symlink_to(source.parent, target_is_directory=True)
        target = tmp_path / "alias" / source.name
    with pytest.raises(StateError):
        store.materialize(result.state_id, target)
    assert not list(tmp_path.glob(".shinobi-state-*"))


def test_independent_check_rejects_writer_success(state, tmp_path, monkeypatch):
    _, store, result = state
    monkeypatch.setattr(adapter, "native_id", lambda _: "msutils-logical-hash/v1:" + "f" * 64)
    with pytest.raises(StateError, match="postcondition"):
        store.materialize(result.state_id, tmp_path / "target.ms")
    assert not (tmp_path / "target.ms").exists()
    assert not list(tmp_path.glob(".shinobi-state-*"))


def test_writer_and_export_conflict_but_readers_coexist(state, tmp_path):
    source, store, _ = state
    reader = acquire_workspace(source.parent, "reader", kind="local", accesses=[(source, False)])
    try:
        result = store.export(source)
        assert result.verified
    finally:
        reader.release()
    writer = acquire_workspace(source.parent, "writer", kind="local", accesses=[(source, True)])
    try:
        with pytest.raises(StateError, match="overlap|conflict|owned"):
            store.export(source)
    finally:
        writer.release()


def test_source_change_between_export_and_capture(tmp_path, monkeypatch):
    from casacore.tables import table

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    export = adapter.export

    def change(source, destination):
        result = export(source, destination)
        with table(str(source), readonly=False, ack=False) as main:
            main.putcell("SCAN_NUMBER", 0, 9)
        return result

    monkeypatch.setattr(adapter, "export", change)
    with pytest.raises(StateError, match="source-changed"):
        store.export(source)
    assert store.list() == []


@pytest.mark.parametrize("boundary", ["write", "verify", "rename"])
def test_sigkill_private_stage_and_publication_recovery(state, tmp_path, boundary):
    _, store, result = state
    target = tmp_path / "target.ms"
    code = r"""
import os, signal, sys
from pathlib import Path
from shinobi.dataset_state import DatasetStateStore
from msutils import _native_preservation as native
import shinobi.dataset_state as state
boundary = sys.argv[4]
def kill():
    os.kill(os.getpid(), signal.SIGKILL)
if boundary == "write":
    original = native.DaskMsWriter.write
    def write(self, plan, destination):
        original(self, plan, destination)
        kill()
    native.DaskMsWriter.write = write
elif boundary == "verify":
    original = native._verify_target
    def verify(plan, destination):
        original(plan, destination)
        kill()
    native._verify_target = verify
else:
    original = state.publish_directory
    def publish(source, destination):
        original(source, destination)
        kill()
    state.publish_directory = publish
DatasetStateStore(sys.argv[1], cache_dir=sys.argv[5]).materialize(sys.argv[2], sys.argv[3])
"""
    process = subprocess.run([sys.executable, "-c", code, str(store.root), result.state_id, str(target), boundary, str(store.cache_dir)], capture_output=True, timeout=60)
    assert process.returncode == -signal.SIGKILL, process.stderr.decode()
    stages = list(tmp_path.glob(".shinobi-state-*"))
    assert len(stages) == 1
    if boundary == "write":
        # The log commit survives even if its materialized JSON view does not.
        for view in (store.root / "attempts").glob("*.json"):
            view.unlink()
    decoy = tmp_path / ".target.ms.unrelated"
    decoy.mkdir()
    (decoy / "keep").write_text("keep")
    recovered = store.recover(target)
    assert len(recovered) == 1
    assert recovered[0].phase == ("committed" if boundary == "rename" else "interrupted")
    assert not stages[0].exists() and (decoy / "keep").read_text() == "keep"
    if boundary != "rename":
        assert not target.exists()
        store.materialize(result.state_id, target)
    assert adapter.native_id(target) == result.state_id
    assert store.recover(target) == []


def test_recovery_cannot_sweep_live_materialization(state, tmp_path, monkeypatch):
    _, store, result = state
    entered, proceed = threading.Event(), threading.Event()
    original = adapter.materialize

    def pause(*args):
        entered.set()
        assert proceed.wait(20)
        return original(*args)

    monkeypatch.setattr(adapter, "materialize", pause)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(store.materialize, result.state_id, tmp_path / "target.ms")
        assert entered.wait(20)
        try:
            with pytest.raises(StateError, match="lock|live"):
                store.recover()
            assert list(tmp_path.glob(".shinobi-state-*"))
        finally:
            proceed.set()
        assert future.result().verified


def test_racing_destination_is_never_replaced(state, tmp_path, monkeypatch):
    import shinobi.dataset_state as module

    _, store, result = state
    original = module.publish_directory

    def race(source, destination):
        destination.mkdir()
        (destination / "winner").write_text("other process")
        original(source, destination)

    monkeypatch.setattr(module, "publish_directory", race)
    with pytest.raises(StateError):
        store.materialize(result.state_id, tmp_path / "target.ms")
    assert (tmp_path / "target.ms" / "winner").read_text() == "other process"


@pytest.mark.parametrize("winner", ["file", "symlink", "dangling-symlink"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_non_directory_publication_winner_is_preserved_and_attempt_settles(state, tmp_path, monkeypatch, winner, interrupted):
    import shinobi.dataset_state as module

    _, store, result = state
    target = tmp_path / "target.ms"
    external = tmp_path / "external"
    if winner == "symlink":
        external.mkdir()
        (external / "keep").write_text("unrelated directory")
    publish = module.publish_directory

    def race(source, destination):
        if winner == "file":
            destination.write_text("unrelated file")
        else:
            destination.symlink_to(external, target_is_directory=True)
        if interrupted:
            # Leave the durable ready record and private candidate as at an
            # abrupt interruption, before no-replace publication is attempted.
            raise KeyboardInterrupt
        publish(source, destination)

    monkeypatch.setattr(module, "publish_directory", race)
    with pytest.raises(KeyboardInterrupt if interrupted else StateError):
        store.materialize(result.state_id, target)
    original_entry = target.lstat()
    if interrupted:
        assert len(list(tmp_path.glob(".shinobi-state-*"))) == 1
        recovered = store.recover(target)
        assert len(recovered) == 1
        assert recovered[0].phase == "interrupted" and recovered[0].finished
        assert store.recover(target) == []
    else:
        records = [StateAttempt.model_validate(JsonFileStore(path).read()) for path in (store.root / "attempts").glob("*.json")]
        assert len(records) == 1 and records[0].phase == "failed" and records[0].finished
        assert records[0].provenance is not None
        assert records[0].provenance.actual_fidelity == "exact-logical"
        assert records[0].provenance.materialization_decision == "validated"
        assert records[0].provenance.replay_decision == "validated"
        assert store.recover(target) == []
    current_entry = target.lstat()
    assert (current_entry.st_dev, current_entry.st_ino, current_entry.st_mode) == (original_entry.st_dev, original_entry.st_ino, original_entry.st_mode)
    if winner == "file":
        assert target.read_text() == "unrelated file"
    else:
        assert target.is_symlink() and target.readlink() == external
        if winner == "symlink":
            assert (external / "keep").read_text() == "unrelated directory"
        else:
            assert not external.exists()
    assert not list(tmp_path.glob(".shinobi-state-*"))


def test_lossless_codec_change_keeps_native_identity(tmp_path, monkeypatch):
    import zarr
    from zarr.codecs import GzipCodec

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    first = store.export(source, block_rows=2)
    create = zarr.create_array
    capture = adapter.capture

    def gzip_array(*args, **kwargs):
        kwargs["compressors"] = [GzipCodec(level=1)]
        return create(*args, **kwargs)

    def alternate_capture(*args):
        with monkeypatch.context() as patch:
            patch.setattr(zarr, "create_array", gzip_array)
            return capture(*args)

    monkeypatch.setattr(adapter, "capture", alternate_capture)
    second = store.export(source, block_rows=2)
    assert second.state_id == first.state_id and second.representation_id != first.representation_id
    assert store._load(first.state_id, first.representation_id)[2].payload_id == store._load(second.state_id, second.representation_id)[2].payload_id
    assert any('"gzip"' in raw for raw in store._load(second.state_id, second.representation_id)[2].zarr_metadata.values())
    assert store.verify(second.state_id, second.representation_id).verified


@pytest.mark.parametrize("case", ["ragged", "mixed", "empty-keyword"])
def test_profile_refusal_preserves_diagnostics(tmp_path, monkeypatch, case):
    import numpy as np
    from casacore.tables import makearrcoldesc, table

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms", set_category=case != "empty-keyword")
    with table(str(source), readonly=False, ack=False) as main:
        if case != "empty-keyword":
            main.addcols(makearrcoldesc("CUSTOM_VARIABLE", 0.0, ndim=1, valuetype="float"))
            for row in range(6 if case == "ragged" else 1):
                main.putcell("CUSTOM_VARIABLE", row, np.arange(row + 1, dtype=np.float32))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    with pytest.raises(StateError) as error:
        store.export(source)
    assert error.value.code == {"ragged": "ragged-cell", "mixed": "mixed-definedness", "empty-keyword": "empty-typed-keyword"}[case]
    if case != "empty-keyword":
        assert error.value.details["table"] == "MAIN" and error.value.details["column"] == "CUSTOM_VARIABLE"
    assert store.list() == []
    assert not list((store.root / "staging").iterdir())


def test_fixed_custom_and_undefined_columns(tmp_path, monkeypatch):
    import numpy as np
    from casacore.tables import makearrcoldesc, table

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        main.addcols(makearrcoldesc("CUSTOM_FIXED", 0.0, shape=[2], valuetype="double"))
        main.addcols(makearrcoldesc("CUSTOM_UNDEFINED", 0.0, ndim=1, valuetype="double"))
        values = np.array([[-0.0, float("inf")]] * 6)
        main.putcol("CUSTOM_FIXED", values)
        main.putcolkeyword("CUSTOM_FIXED", "QuantumUnits", np.asarray(["m"]))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    result = store.export(source)
    target = tmp_path / "target.ms"
    store.materialize(result.state_id, target)
    with table(str(target), ack=False) as restored:
        np.testing.assert_array_equal(restored.getcol("CUSTOM_FIXED").view(np.uint64), values.view(np.uint64))
        assert all(not restored.iscelldefined("CUSTOM_UNDEFINED", row) for row in range(6))
    assert adapter.native_id(target) == result.state_id


def test_missing_original_source_is_still_protected(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    result = store.export(source)
    source.rename(tmp_path / "hidden.ms")
    with pytest.raises(StateError, match="destination-alias"):
        store.materialize(result.state_id, source)


def test_coherently_rebound_bundle_does_not_change_native_store_key(state, tmp_path, monkeypatch):
    """Even a freshly valid preservation bundle cannot redefine the requested key."""
    from casacore.tables import table
    from shinobi.dataset_state import _digest, _inventory

    source, store, result = state
    changed = tmp_path / "changed.ms"
    shutil.copytree(source, changed)
    with table(str(changed), readonly=False, ack=False) as main:
        main.putcell("SCAN_NUMBER", 0, 99)
    rep, _, physical = store._load(result.state_id)
    shutil.rmtree(rep / "preservation")
    adapter.capture(changed, rep / "msv4.zarr", rep / "preservation", 2)
    ids = adapter.verify(rep / "msv4.zarr", rep / "preservation")
    assert ids.native_logical_id != result.state_id
    files, nodes = _inventory(rep)
    # Update every physical binding consistently, retaining the independent
    # requested state ID and directory (the attack a bundle-only check misses).
    physical = physical.model_copy(update={"files": files, "zarr_metadata": nodes, "payload_id": ids.payload_logical_id})
    physical = physical.model_copy(update={"representation_id": _digest(physical.model_dump(mode="json", exclude={"representation_id"}))})
    (rep / "representation.json").write_text(physical.model_dump_json())
    rep.rename(rep.parent / physical.representation_id.split(":")[1])
    with pytest.raises(StateError, match="logical-integrity"):
        store.verify(result.state_id)


def test_cache_maintenance_does_not_touch_states(state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from shinobi.cli import main

    _, store, result = state
    config = tmp_path / "config.yaml"
    config.write_text(f"cache:\n  dir: {store.cache_dir}\nstate:\n  dir: {store.root}\n")
    before = {str(path.relative_to(store.root)): path.read_bytes() for path in store.root.rglob("*") if path.is_file()}
    runner = CliRunner()
    for args in (["cache", "evict", "--bytes", "1000"], ["cache", "invalidate", "absent"], ["clean", "--no-runs", "--no-sandboxes"]):
        response = runner.invoke(main, ["--config", str(config), *args])
        assert response.exit_code == 0, response.output
    after = {str(path.relative_to(store.root)): path.read_bytes() for path in store.root.rglob("*") if path.is_file()}
    assert after == before
    assert store.verify(result.state_id).verified


def test_concurrent_exports_publish_two_complete_representations(state):
    source, store, original = state
    with ThreadPoolExecutor(max_workers=2) as pool:
        exports = list(pool.map(lambda rows: store.export(source, block_rows=rows), (1, 3)))
    assert {item.state_id for item in exports} == {original.state_id}
    assert len({item.representation_id for item in exports}) == 2
    assert len(store.list()) == 3
    assert all(store.verify(item.state_id, item.representation_id).verified for item in exports)


def test_export_after_rename_failure_is_recovered(state, monkeypatch):
    import shinobi.dataset_state as module

    source, store, _ = state
    publish = module.publish_directory

    def fail_after_publish(source, destination):
        publish(source, destination)
        raise OSError("injected directory fsync failure after rename")

    monkeypatch.setattr(module, "publish_directory", fail_after_publish)
    with pytest.raises(StateError, match="injected"):
        store.export(source)
    recovered = store.recover()
    assert len(recovered) == 1 and recovered[0].phase == "committed"
    assert not recovered[0].stage.exists()
    assert store.verify(recovered[0].state_id, recovered[0].representation_id).verified


def test_export_interrupted_before_first_publication_is_recovered(tmp_path, monkeypatch):
    import shinobi.dataset_state as module

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")

    def interrupt_before_publish(_source, _destination):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "publish_directory", interrupt_before_publish)
    with pytest.raises(KeyboardInterrupt):
        store.export(source, block_rows=1)

    unfinished = [read_state_attempt(path) for path in (store.root / "attempts").glob("*.json")]
    assert len(unfinished) == 1 and unfinished[0].phase == "ready" and not unfinished[0].finished
    assert not store._state(unfinished[0].state_id).exists()

    recovered = store.recover()
    assert len(recovered) == 1
    assert recovered[0].phase == "interrupted" and recovered[0].finished
    assert not recovered[0].stage.exists()
    assert store.recover() == []


def test_duplicate_destination_materialization_is_serialized(state, tmp_path):
    _, store, original = state
    target = tmp_path / "target.ms"

    def create():
        try:
            return store.materialize(original.state_id, target)
        except StateError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create) for _ in range(2)]
        results = [future.result() for future in futures]
    assert sum(isinstance(item, StateError) for item in results) == 1
    assert adapter.native_id(target) == original.state_id
