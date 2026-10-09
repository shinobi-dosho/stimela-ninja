"""Restart a killed strict MS/model invocation through production dispatch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("casacore.tables")
pytest.importorskip("numpy")

from shinobi.snapshots import chain_id, get_journal
from shinobi.ownership import ownership_workspace, reconcile_ownership, scope_path_accesses
from shinobi.steps.dispatch import _dispatch
from tests._dataset_fixtures import ROWS, make_ms, scans
from tests.test_derived_mutations_live import cab


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4", "S5"])
def test_dispatch_recovers_ms_and_model_after_process_death(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    ms = make_ms(tmp_path / "obs.ms")
    model = tmp_path / "image-model.fits"
    model.write_text("0")
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repository / "src"), str(repository), env.get("PYTHONPATH", "")])
    script = """
import os, sys
from pathlib import Path
from shinobi.snapshots import faults
from shinobi.steps.dispatch import _dispatch
from tests.test_derived_mutations_live import cab
faults.hooks[sys.argv[1]] = lambda: os._exit(73)
_dispatch(cab(), None, ms=Path('obs.ms'), cache=True, cache_dir=str(Path('cache').resolve()))
"""
    killed = subprocess.run([sys.executable, "-c", script, stage], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert killed.returncode == 73, killed.stderr
    cache = str(tmp_path / "cache")
    journal = get_journal(cache)
    if stage != "S5":
        assert all(journal.get(chain_id(path)).marker is not None for path in (ms, model))
    accesses, _ = scope_path_accesses(cab(), cab().inputs_model(ms=ms), workspace=tmp_path)
    writes = [path for path, writable in accesses if writable]
    claim_root = ownership_workspace(tmp_path, writes)
    assert reconcile_ownership(claim_root).liveness == "dead"
    resumed = _dispatch(cab(), None, ms=ms, cache=True, cache_dir=cache)
    assert resumed.success, resumed.stderr
    assert model.read_text() == "1"
    assert scans(ms) == [2] * ROWS
    assert all(journal.get(chain_id(path)).marker is None for path in (ms, model))
    assert _dispatch(cab(), None, ms=ms, cache=True, cache_dir=cache).cached
