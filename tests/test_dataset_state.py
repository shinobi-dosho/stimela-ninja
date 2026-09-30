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
