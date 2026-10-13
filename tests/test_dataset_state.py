"""Lightweight state contracts, CLI and atomic storage primitives."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from shinobi import _state_adapter as adapter
from shinobi.cli import main
from shinobi.config import AppConfig
from shinobi.dataset_state import DatasetStateStore, FileEntry, LogicalState, StateAttempt, StateError, StateProvenance, read_state_attempt
from shinobi.storage import publish_directory


def _versions():
    return {**dict.fromkeys(adapter.PACKAGES, "test"), "msutils_commit": "test-commit"}


def test_state_config_and_lazy_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert AppConfig(state={"dir": "datasets"}).state.dir == "datasets"
    assert AppConfig().state.dir != AppConfig().cache.dir
    runner = CliRunner()
    assert runner.invoke(main, ["state", "--help"]).exit_code == 0
    bare = runner.invoke(main, ["state"])
    assert bare.exit_code == 0 and "Commands:" in bare.output
    result = runner.invoke(main, ["state", "--store", str(tmp_path / "store"), "list", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    code = "import sys; from shinobi.cli import main; from shinobi.dataset_state import DatasetStateStore; assert not {'msutils','casacore','xarray','zarr','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("path", ["", ".", "../x", "/x", "a/../x", "a//x", "a\\x"])
def test_manifest_refuses_unsafe_paths(path):
    with pytest.raises(ValidationError):
        FileEntry(path=path, kind="file", size=1, sha256="a" * 64)


def test_identity_validation_and_separate_store(tmp_path):
    with pytest.raises(ValidationError):
        LogicalState(state_id="sha256:" + "a" * 64)
    with pytest.raises(StateError, match="store-cache-overlap"):
        DatasetStateStore(tmp_path / "cache" / "states", cache_dir=tmp_path / "cache")
    with pytest.raises(StateError, match="state-id"):
        DatasetStateStore(tmp_path / "store")._state("../../escape")


def test_state_attempt_v2_closes_exact_replay_provenance(tmp_path):
    state_id = "msutils-logical-hash/v1:" + "a" * 64
    representation_id = "sha256:" + "b" * 64
    versions = _versions()
    provenance = StateProvenance(
        state_id=state_id,
        representation_id=representation_id,
        msv4_id="msutils-logical-hash/v1:" + "c" * 64,
        payload_id="msutils-logical-hash/v1:" + "d" * 64,
        producer_versions=versions,
        actual_fidelity="exact-logical",
        cache_decision="store-requested",
        materialization_decision="not-requested",
        replay_decision="not-requested",
        structural_signature="e" * 64,
    )
    attempt = StateAttempt(
        attempt_id=uuid.uuid4(),
        operation="export",
        store=tmp_path / "store",
        authority=tmp_path,
        registry=tmp_path / "owners.json",
        source=tmp_path / "source.ms",
        state_id=state_id,
        representation_id=representation_id,
        stage=tmp_path / "stage",
        parent_identity=(1, 2),
        versions=versions,
        phase="committed",
        structural_signature="e" * 64,
        provenance=provenance,
    )
    path = tmp_path / "attempt.json"
    path.write_text(attempt.model_dump_json())
    assert read_state_attempt(path) == attempt
    assert provenance.state_id != provenance.representation_id
    assert provenance.closure_profile == "msv2-dataset-closure/v1"
    assert StateProvenance.model_validate({**provenance.model_dump(), "closure_profile": "msv2-dataset-closure/v2"}).closure_profile == "msv2-dataset-closure/v2"
    assert provenance.requested_fidelity == provenance.actual_fidelity == "exact-logical"
    assert provenance.transformation_decision == "not-requested" and not provenance.physical_restoration

    with pytest.raises(ValidationError, match="committed materialization"):
        StateAttempt.model_validate(
            {
                **attempt.model_dump(),
                "operation": "materialize",
            }
        )
    with pytest.raises(ValidationError):
        StateProvenance.model_validate({**provenance.model_dump(), "requested_fidelity": "policy-bounded"})
    with pytest.raises(ValidationError, match="complete, unique evidence"):
        StateProvenance.model_validate({**provenance.model_dump(), "evidence": ["native-logical-id"]})
    with pytest.raises(ValidationError, match="state stack keys"):
        StateProvenance.model_validate({**provenance.model_dump(), "producer_versions": {}})
    with pytest.raises(ValidationError, match="state stack keys"):
        StateAttempt.model_validate({**attempt.model_dump(), "versions": {**versions, "unexpected": "1"}})
    with pytest.raises(ValidationError, match="matching structural signature"):
        StateAttempt.model_validate({**attempt.model_dump(), "structural_signature": None})
    with pytest.raises(ValidationError, match="schema v1 cannot carry"):
        StateAttempt.model_validate({**attempt.model_dump(), "schema_version": "shinobi-state-attempt/v1"})
    with pytest.raises(ValidationError, match="identities disagree"):
        StateAttempt.model_validate(
            {
                **attempt.model_dump(),
                "provenance": {**provenance.model_dump(), "state_id": "msutils-logical-hash/v1:" + "f" * 64},
            }
        )
    with pytest.raises(ValueError, match="requested materialization/replay"):
        provenance.validated_replay()


def test_failed_materialization_records_request_not_success(tmp_path):
    state_id = "msutils-logical-hash/v1:" + "a" * 64
    representation_id = "sha256:" + "b" * 64
    provenance = StateProvenance(
        state_id=state_id,
        representation_id=representation_id,
        msv4_id="msutils-logical-hash/v1:" + "c" * 64,
        payload_id="msutils-logical-hash/v1:" + "d" * 64,
        producer_versions=_versions(),
        cache_decision="selected-representation",
        materialization_decision="requested",
        replay_decision="requested",
        structural_signature="e" * 64,
    )
    attempt = StateAttempt(
        attempt_id=uuid.uuid4(),
        operation="materialize",
        store=tmp_path / "store",
        authority=tmp_path,
        registry=tmp_path / "owners.json",
        destination=tmp_path / "target.ms",
        requested_state_id=state_id,
        requested_fidelity="exact-logical",
        state_id=state_id,
        representation_id=representation_id,
        stage=tmp_path / "stage",
        parent_identity=(1, 2),
        versions=_versions(),
        phase="refused",
        finished=True,
        error="stack-version: incompatible",
        provenance=provenance,
    )
    assert attempt.provenance.actual_fidelity is None
    assert attempt.provenance.materialization_decision == "requested"
    assert attempt.provenance.replay_decision == "requested"
    with pytest.raises(ValidationError, match="decisions disagree with materialization"):
        StateAttempt.model_validate(
            {
                **attempt.model_dump(),
                "provenance": {**provenance.model_dump(), "cache_decision": "store-requested"},
            }
        )


def test_legacy_state_attempt_remains_readable(tmp_path):
    legacy = StateAttempt(
        schema_version="shinobi-state-attempt/v1",
        attempt_id=uuid.uuid4(),
        operation="export",
        store=tmp_path / "store",
        authority=tmp_path,
        registry=tmp_path / "owners.json",
        source=tmp_path / "source.ms",
        stage=tmp_path / "stage",
        parent_identity=(1, 2),
        versions={},
        phase="interrupted",
        finished=True,
    )
    assert StateAttempt.model_validate_json(legacy.model_dump_json()) == legacy


def test_adapter_runtime_reports_missing_stack(monkeypatch):
    def missing(_name):
        raise adapter.metadata.PackageNotFoundError("missing")

    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(ImportError, match="measurement-set development group"):
        adapter.runtime()


@pytest.fixture
def qualified_adapter(monkeypatch):
    import sys
    from types import SimpleNamespace

    versions = {
        "msutils": "3.0.0",
        "xarray-ms": "0.5.8",
        "xarray": "2026.7.0",
        "numcodecs": "0.16.5",
        "arcae": "0.5.4",
        "zarr": "3.1.6" if sys.version_info < (3, 12) else "3.3.0",
    }
    direct = {"url": "https://github.com/shinobi-dosho/msutils.git", "vcs_info": {"vcs": "git", "commit_id": "a" * 40, "requested_revision": "main"}}

    def version(name):
        if name == "dask-ms":
            raise adapter.metadata.PackageNotFoundError(name)
        return versions.get(name, "test")

    monkeypatch.setattr(adapter.metadata, "version", version)
    monkeypatch.setattr(adapter.metadata, "distribution", lambda _: SimpleNamespace(read_text=lambda _: json.dumps(direct)))
    apis = SimpleNamespace(**dict.fromkeys(("native_logical_id", "capture_native_preservation", "verify_native_preservation", "to_msv2"), lambda: None))
    monkeypatch.setitem(sys.modules, "msutils", apis)
    return versions, direct, apis


@pytest.mark.parametrize("commit", ["a" * 40, "25c280a8ab412b5015a80bf4ff6f83b7a6820675"])
@pytest.mark.parametrize("url", ["https://github.com/shinobi-dosho/msutils.git", "https://github.com/shinobi-dosho/msutils"])
@pytest.mark.parametrize("revision", ["main", "25c280a8ab412b5015a80bf4ff6f83b7a6820675"])
@pytest.mark.parametrize("version", ["3.0.0", "3.1.0"])
def test_adapter_runtime_qualifies_upstream_git(qualified_adapter, commit, url, revision, version):
    versions, direct, _ = qualified_adapter
    versions["msutils"] = version
    direct["url"] = url
    direct["vcs_info"].update(commit_id=commit, requested_revision=revision)
    qualified = adapter.runtime()
    assert qualified["dask-ms"] == "not-installed"
    assert qualified["arcae"] == "0.5.4"
    assert qualified["msutils"] == version
    assert qualified["msutils_commit"] == commit
    assert set(qualified) == {*adapter.PACKAGES, "msutils_commit"}


@pytest.mark.parametrize("direct", [None, "{", "[]", "null", '"text"', "{}", '{"vcs_info": []}', '{"vcs_info": null}'])
def test_adapter_runtime_refuses_invalid_provenance(qualified_adapter, monkeypatch, direct):
    from types import SimpleNamespace

    monkeypatch.setattr(adapter.metadata, "distribution", lambda _: SimpleNamespace(read_text=lambda _: direct))
    with pytest.raises(ImportError, match="measurement-set development group"):
        adapter.runtime()


@pytest.mark.parametrize(
    "url",
    ["file:///local/msutils", "https://example.org/msutils.git", "https://secret:token@github.com/shinobi-dosho/msutils.git", "ssh://git@github.com/shinobi-dosho/msutils.git"],
)
def test_adapter_runtime_refuses_wrong_source(qualified_adapter, url):
    _, direct, _ = qualified_adapter
    direct["url"] = url
    with pytest.raises(ImportError, match="upstream Git") as raised:
        adapter.runtime()
    assert url not in str(raised.value)


@pytest.mark.parametrize("vcs", [None, {}, [], {"vcs": "hg", "commit_id": "a" * 40}, {"vcs": "git"}])
def test_adapter_runtime_refuses_wrong_vcs(qualified_adapter, vcs):
    _, direct, _ = qualified_adapter
    direct["vcs_info"] = vcs
    with pytest.raises(ImportError, match="upstream Git"):
        adapter.runtime()


@pytest.mark.parametrize("commit", [None, 123, "main", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_adapter_runtime_refuses_unresolved_revision(qualified_adapter, commit):
    _, direct, _ = qualified_adapter
    direct["vcs_info"]["commit_id"] = commit
    with pytest.raises(ImportError, match="upstream Git"):
        adapter.runtime()


def test_adapter_runtime_refuses_wheel_provenance(qualified_adapter):
    _, direct, _ = qualified_adapter
    direct.pop("vcs_info")
    direct["archive_info"] = {"hash": "sha256=" + "a" * 64}
    with pytest.raises(ImportError, match="upstream Git"):
        adapter.runtime()


@pytest.mark.parametrize("api", ["native_logical_id", "capture_native_preservation", "verify_native_preservation", "to_msv2"])
def test_adapter_runtime_refuses_missing_apis(qualified_adapter, api):
    _, _, apis = qualified_adapter
    setattr(apis, api, None)
    with pytest.raises(ImportError, match="public APIs"):
        adapter.runtime()


@pytest.mark.parametrize("package", ["xarray-ms", "xarray", "numcodecs", "arcae", "zarr"])
def test_adapter_runtime_retains_other_version_gates(qualified_adapter, package):
    versions, _, _ = qualified_adapter
    versions[package] = "0.0"
    with pytest.raises(ImportError, match="unqualified state stack"):
        adapter.runtime()


def test_no_replace_including_empty_directory(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "content").write_text("new")
    target.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(source, target)
    assert source.is_dir() and not list(target.iterdir())
    target.rmdir()
    publish_directory(source, target)
    assert (target / "content").read_text() == "new"
    assert not source.exists()


def test_historical_nested_observation_digest_loads_unchanged(tmp_path):
    from datetime import datetime, timezone
    from shinobi.dataset_lifecycle import structural_signature
    from shinobi.dataset_state import StateRepresentation, _digest
    from tests.test_dataset_lifecycle import _snapshot

    observation = _snapshot(tmp_path / "source.ms").observations[0]
    state_id = "msutils-logical-hash/v1:" + "a" * 64
    physical = StateRepresentation(
        representation_id="sha256:" + "0" * 64,
        state_id=state_id,
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
    historical = physical.model_dump(mode="json", exclude={"representation_id"})
    assert "table_columns" not in historical["observation"]
    representation_id = _digest(historical)
    historical["representation_id"] = representation_id
    store = DatasetStateStore(tmp_path / "store")
    state = store._state(state_id)
    rep = state / "representations" / representation_id.removeprefix("sha256:")
    rep.mkdir(parents=True)
    (state / "logical.json").write_text(LogicalState(state_id=state_id).model_dump_json())
    (rep / "representation.json").write_text(json.dumps(historical))
    _, _, loaded = store._load(state_id, representation_id)
    assert loaded.model_dump(mode="json") == historical
    assert _digest(loaded.model_dump(mode="json", exclude={"representation_id"})) == representation_id


def test_native_profile_pairs_and_legacy_representation_shape(tmp_path):
    from datetime import datetime, timezone
    from shinobi.dataset_lifecycle import structural_signature
    from shinobi.dataset_state import StateRepresentation, StateRepresentationV2, StateRepresentationV3
    from shinobi.state_codecs import LosslessProfile
    from tests.test_dataset_lifecycle import _snapshot

    state_id = "msutils-logical-hash/v1:" + "a" * 64
    with pytest.raises(ValidationError, match="profile and model versions disagree"):
        LogicalState(state_id=state_id, profile="fixed-shape-defined-or-empty/v2")
    observation = _snapshot(tmp_path / "source.ms").observations[0]
    common = {
        "representation_id": "sha256:" + "0" * 64,
        "state_id": state_id,
        "msv4_id": "msutils-logical-hash/v1:" + "b" * 64,
        "payload_id": "msutils-logical-hash/v1:" + "c" * 64,
        "versions": _versions(),
        "source_spelling": str(observation.root),
        "source": observation.root,
        "observation": observation,
        "structural_signature": structural_signature(observation),
        "capture_attempt": uuid.uuid4(),
        "captured_at": datetime.now(timezone.utc),
        "files": (),
        "zarr_metadata": {},
    }
    old = StateRepresentation(**common)
    explicit = StateRepresentationV2(**common, lossless_profile=LosslessProfile())
    assert "profile" not in old.model_dump() and "native_model" not in old.model_dump()
    assert "profile" not in explicit.model_dump() and "native_model" not in explicit.model_dump()
    current = StateRepresentationV3(**common, profile="fixed-shape-defined-or-empty/v2", native_model="msutils-native-model/v2")
    assert current.lossless_profile is None
    with pytest.raises(ValidationError, match="profile and model versions disagree"):
        StateRepresentationV3(**common, profile="fixed-shape-defined-or-empty/v2", native_model="msutils-native-model/v1")
