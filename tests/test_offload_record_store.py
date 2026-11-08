"""Integrity and bounded-memory monitoring of submission-local v7 evidence."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from shinobi.cache import ProvenanceKey
from shinobi.offload._codec import BundleError
from shinobi.offload.bundle import write_new
from shinobi.offload.record_store import Edge, LegacySource, Record, Store, digest_file, read_terminal
from shinobi.offload.records import AttemptRecord
from shinobi.provenance import build_manifest, load_manifest
from shinobi.results import StepResult
from shinobi.steps.schema import Scope


class Values(BaseModel):
    path: Path
    pair: tuple[int, str] = (3, "data")


def result(name="root", *, children=None, failed=False, stdout=""):
    return StepResult(
        name=name,
        kind="recipe" if children else "cab",
        returncode=int(failed),
        inputs=Values(path=Path("input")),
        outputs=Values(path=Path("output")),
        sub_results=children,
        stdout=stdout,
        output_keys={"path": ProvenanceKey("cache", "original")},
    )


def capture(tmp_path, tree=None):
    store = Store(tmp_path)
    tree = tree or result(children={"z": result("same", stdout="first"), "a": result("same", stdout="second")}, stdout="first\nsecond")
    record = store.capture(tree, workflow_id=uuid4(), attempt_id=uuid4(), step_path=tree.name, bundle_digest="bundle")
    path = tmp_path / "results" / str(record.attempt_id) / "final.json"
    write_new(path, record)
    return store, record, path


def test_roundtrip_shallow_order_keys_and_streamed_manifest(tmp_path):
    store, record, path = capture(tmp_path)
    assert record.observation.steps == [] and record.children is None
    assert len(list((tmp_path / "schemas").iterdir())) == 1
    assert [edge.name for edge in record.edges] == ["z", "a"]
    store.verify(record, reference=store.root_reference(record.attempt_id))
    scope = Scope(name="scope", inputs_model=Values, outputs_model=Values)
    shallow = store.result(record, scope)
    assert shallow.sub_results is None and shallow.stdout == ""
    assert shallow.outputs.pair == (3, "data") and shallow.outputs.path == Path("output")
    assert shallow.provenance_key("path").producer_field == "original"
    restored = store.result(record, scope, full=True)
    assert list(restored.sub_results) == ["z", "a"]
    assert restored.stdout == "first\nsecond"
    manifest = path.with_name("manifest.json")
    store.manifest(manifest, record)
    assert load_manifest(manifest).root == build_manifest(restored, backend="slurm").root


@pytest.mark.parametrize("legacy", [False, True], ids=["fresh-v7", "converted-v6"])
def test_captured_streams_preserve_newlines_and_unicode(tmp_path, legacy):
    first = result("first", stdout="α\r\nβ\rγ\n")
    first.stderr = "警告\r\nprogress\rfinished\n"
    second = result("second", stdout="δ\r\nε\rζ\n")
    second.stderr = "échec\r\nretry\rdone\n"
    tree = result(children={"first": first, "second": second}, stdout=first.stdout + "\n" + second.stdout)
    tree.stderr = first.stderr + "\n" + second.stderr
    if legacy:
        old = AttemptRecord.from_result(tree, workflow_id=uuid4(), attempt_id=uuid4(), step_path="root", bundle_digest="bundle", composite=True)
        path = tmp_path / "results" / str(old.attempt_id) / "final.json"
        write_new(path, old)
        store = Store(tmp_path)
        record = store.convert(old, LegacySource(**digest_file(path).model_dump()))
        write_new(path.with_name("final.v7.json"), record)
    else:
        store, record, path = capture(tmp_path, tree)
    record = read_terminal(path)
    store.verify(record, reference=store.root_reference(record.attempt_id))
    scope = Scope(name="scope", inputs_model=Values, outputs_model=Values)
    restored = store.result(record, scope, full=True)
    for stream in ("stdout", "stderr"):
        assert getattr(restored, stream) == getattr(tree, stream)
        for name, child in tree.sub_results.items():
            assert getattr(restored.sub_results[name], stream) == getattr(child, stream)


@pytest.mark.parametrize("kind", ["child", "schema", "log"])
@pytest.mark.parametrize("damage", ["missing", "truncated", "changed"])
def test_missing_or_corrupt_evidence_refuses(tmp_path, kind, damage):
    store, record, _ = capture(tmp_path)
    child = store.read_one(record.edges[0].reference)
    path = (
        store.node_path(record.edges[0].reference)
        if kind == "child"
        else store.artifact_path("schema", record.schemas[0])
        if kind == "schema"
        else store.artifact_path("log", child.stdout_artifact)
    )
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:2])
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises((OSError, ValueError)):
        Store(tmp_path).verify(record, reference=store.root_reference(record.attempt_id))


def test_wrong_node_identity_and_unsuccessful_child_refuse(tmp_path):
    store, record, _ = capture(tmp_path)
    child = store.read_one(record.edges[0].reference)
    altered = child.model_copy(update={"address": ("wrong",)})
    reference = store.put_node(altered).model_copy(update={"address": ("z",)})
    bad = record.model_copy(update={"edges": (Edge(name="z", reference=reference),)})
    with pytest.raises(BundleError, match="wrong owner or logical address"):
        Store(tmp_path).verify(bad)
    failed = store.capture(
        result("z", failed=True), workflow_id=record.workflow_id, attempt_id=record.attempt_id, step_path="z", bundle_digest=record.bundle_digest, address=("z",)
    )
    bad = record.model_copy(update={"edges": (Edge(name="z", reference=store.put_node(failed)),)})
    with pytest.raises(BundleError, match="unsuccessful child"):
        Store(tmp_path).verify(bad)


def test_duplicate_edges_and_embedded_metadata_refuse(tmp_path):
    _, record, _ = capture(tmp_path)
    for update in ({"edges": [record.edges[0], record.edges[0]]}, {"stdout": "embedded"}, {"observation": record.observation.model_copy(update={"steps": [record.observation]})}):
        with pytest.raises(ValidationError):
            Record.model_validate({**record.model_dump(), **update})


def test_cas_reuse_verifies_existing_bytes_and_terminal_never_replaces(tmp_path):
    store, record, path = capture(tmp_path)
    artifact = store.put_bytes("log", b"original")
    store.artifact_path("log", artifact).write_bytes(b"changed!")
    with pytest.raises(BundleError, match="digest"):
        store.put_bytes("log", b"original")
    with pytest.raises(FileExistsError):
        write_new(path, record)


def test_shared_node_is_read_once_and_contradictory_reference_refuses(tmp_path, monkeypatch):
    store, child, _ = capture(tmp_path)
    edge = Edge(name="one", reference=store.root_reference(child.attempt_id))
    root = store.capture(
        result(children={}),
        edges=(edge, Edge(name="two", reference=edge.reference)),
        workflow_id=child.workflow_id,
        attempt_id=uuid4(),
        step_path="root",
        bundle_digest=child.bundle_digest,
    )
    seen = []
    original = store.read_one

    def read(reference):
        seen.append((reference.owner, reference.address))
        return original(reference)

    monkeypatch.setattr(store, "read_one", read)
    store.verify(root)
    assert len(seen) == len(set(seen)) == 3
    bad_edge = edge.model_copy(update={"name": "two", "reference": edge.reference.model_copy(update={"size": edge.reference.size + 1})})
    bad = root.model_copy(update={"edges": (edge, bad_edge)})
    with pytest.raises(BundleError, match="contradictory"):
        Store(tmp_path).verify(bad)


def test_legacy_conversion_retains_identity_streams_and_source_binding(tmp_path):
    tree = result(children={"leaf": result(stdout="leaf")}, stdout="leaf")
    legacy = AttemptRecord.from_result(tree, workflow_id=uuid4(), attempt_id=uuid4(), step_path="root", bundle_digest="bundle", composite=True)
    path = tmp_path / "results" / str(legacy.attempt_id) / "final.json"
    write_new(path, legacy)
    before = path.read_bytes()
    store = Store(tmp_path)
    source = LegacySource(**digest_file(path).model_dump())
    record = store.convert(legacy, source)
    write_new(path.with_name("final.v7.json"), record)
    assert path.read_bytes() == before
    assert read_terminal(path) == record
    store.verify(record, reference=store.root_reference(record.attempt_id))
    assert store.convert(legacy, source) == record
    sidecar = path.with_name("final.v7.json")
    saved = sidecar.read_bytes()
    sidecar.write_text(record.model_copy(update={"address": ("not-root",)}).model_dump_json())
    with pytest.raises(BundleError, match="root logical address"):
        read_terminal(path)
    sidecar.write_bytes(saved)
    scope = Scope(name="scope", inputs_model=Values, outputs_model=Values)
    assert store.result(record, scope, full=True).stdout == "leaf"
    path.write_bytes(before + b" ")
    with pytest.raises(BundleError, match="digest"):
        read_terminal(path)


def test_corrupt_sidecar_never_falls_back_and_legacy_cap_is_explicit(tmp_path):
    legacy = AttemptRecord.from_result(result(), workflow_id=uuid4(), attempt_id=uuid4(), step_path="root", bundle_digest="bundle", composite=True)
    path = tmp_path / "final.json"
    write_new(path, legacy)
    assert read_terminal(path) == legacy
    with pytest.raises(BundleError, match="convert-records"):
        read_terminal(path, legacy_limit=1)
    path.with_name("final.v7.json").write_bytes(b"broken")
    with pytest.raises(ValueError):
        read_terminal(path)


def test_bound_manifest_corruption_refuses_verification(tmp_path):
    store, record, path = capture(tmp_path)
    manifest = path.with_name("manifest.json")
    artifact = store.manifest(manifest, record)
    record = record.model_copy(update={"manifest": artifact})
    manifest.write_bytes(b"{}")
    with pytest.raises(BundleError, match="digest"):
        store.verify(record)


def test_interrupted_child_publication_has_no_terminal(tmp_path, monkeypatch):
    store = Store(tmp_path)
    original = store.put_node
    calls = 0

    def publish(node):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted")
        return original(node)

    monkeypatch.setattr(store, "put_node", publish)
    tree = result(children={"a": result(), "b": result()})
    identity = {"workflow_id": uuid4(), "attempt_id": uuid4(), "step_path": "root", "bundle_digest": "bundle"}
    with pytest.raises(OSError, match="interrupted"):
        store.capture(tree, **identity)
    assert not list(tmp_path.glob("results/*/final.json"))
    monkeypatch.setattr(store, "put_node", original)
    record = store.capture(tree, **identity)
    store.verify(record)


def test_large_campaign_monitoring_is_bounded(tmp_path):
    import json
    import subprocess
    import sys

    script = Path(__file__).with_name("offload_scale_fixture.py")
    measurements = []
    for factor in (1, 4):
        built = subprocess.run([sys.executable, str(script), "build", str(tmp_path / f"scale-{factor}"), str(factor * 65536)], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        storage = json.loads(built.stdout)
        # A direct child's ru_maxrss can inherit pytest's resident size across
        # fork/exec. Spawn the monitor from a small fresh interpreter so optional
        # imports and earlier tests cannot masquerade as monitoring memory use.
        checked = subprocess.run(
            [sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))", sys.executable, str(script), "monitor", storage["submission"]],
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stderr
        monitoring = json.loads(checked.stdout)
        assert storage["terminal_bytes"] < 256 * 1024
        assert storage["node_schema_bytes"] < 32 * 1024 * 1024
        assert storage["log_bytes"] == 1740 * factor * 65536
        assert storage["log_files"] == 1740
        assert monitoring["peak_rss_kib"] < 256 * 1024
        measurements.append({**storage, **monitoring})
    assert measurements[1]["peak_rss_kib"] < measurements[0]["peak_rss_kib"] + 32 * 1024
    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))


def test_shallow_nested_models_rebind_at_gather_boundary(tmp_path):
    from shinobi.offload.mixed import _declared_result
    from shinobi.steps.dispatch import _aggregate_scatter_results

    class Nested(BaseModel):
        path: Path
        pair: tuple[int, str]

    class Inputs(BaseModel):
        index: int

    class Outputs(BaseModel):
        value: Nested

    scope = Scope(name="nested", inputs_model=Inputs, outputs_model=Outputs)
    original = StepResult(name="nested", returncode=0, inputs=Inputs(index=1), outputs=Outputs(value=Nested(path=Path("file"), pair=(7, "value"))))
    store, record, _ = capture(tmp_path, original)
    shallow = store.result(record, scope)
    assert type(shallow.outputs.value) is not Nested
    rebound = _declared_result(shallow, scope)
    assert isinstance(rebound.outputs.value, Nested)
    gathered = _aggregate_scatter_results(scope, ["index"], {"index": [1]}, [rebound])
    assert gathered.outputs.value[0].path == Path("file")
    assert gathered.outputs.value[0].pair == (7, "value")
