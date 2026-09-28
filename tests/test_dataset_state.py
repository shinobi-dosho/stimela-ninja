"""Lightweight state contracts, CLI and atomic storage primitives."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from shinobi import _state_adapter as adapter
from shinobi.cli import main
from shinobi.config import AppConfig
from shinobi.dataset_state import DatasetStateStore, FileEntry, LogicalState, StateError
from shinobi.storage import publish_directory


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
