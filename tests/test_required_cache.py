"""Required hits never fall through to execution or mutation recovery hooks."""

from pathlib import Path

import pytest
from pydantic import BaseModel
from shinobi import DatasetAccess, MeasurementSetV2, Recipe, pystep
from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import InputRef, OutputRef
from tests.test_cache import _cab


def test_required_cache_exact_hit_and_misses(tmp_path):
    cab, recorder = _cab(tmp_path / "cache")
    first = _dispatch(cab, None, x=1)
    config = AppConfig()
    config.cache.required = {"tool": first.cache_key}
    assert _dispatch(cab, None, x=1, _config=config).cached
    for value in (2, 3):
        with pytest.raises(DatasetLifecycleUnavailableError, match="key mismatch"):
            _dispatch(cab, None, x=value, _config=config)
    get_cache_manifest(str(tmp_path / "cache")).remove({"tool"})
    with pytest.raises(DatasetLifecycleUnavailableError, match="no valid reusable hit"):
        _dispatch(cab, None, x=1, _config=config)
    assert len(recorder.calls) == 1


def test_required_unknown_and_disabled_refuse(tmp_path):
    cab, recorder = _cab(tmp_path / "cache")
    config = AppConfig()
    config.cache.required = {"typo": "key"}
    with pytest.raises(DatasetLifecycleUnavailableError, match="unknown leaf"):
        _dispatch(cab, None, _config=config)
    config.cache.required = {"tool": "key"}
    with pytest.raises(DatasetLifecycleUnavailableError, match="disabled"):
        _dispatch(cab, None, cache=False, _config=config)
    assert recorder.calls == []


class MSInputs(BaseModel):
    ms: MeasurementSetV2


class Empty(BaseModel):
    pass


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
def acquire(ms: MeasurementSetV2) -> MSInputs:
    from tests._dataset_fixtures import make_ms

    return MSInputs(ms=make_ms(ms))


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
def read(ms: MeasurementSetV2) -> None:
    pass


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write")])
def prepare(ms: MeasurementSetV2) -> MSInputs:
    from tests._dataset_fixtures import set_scans

    set_scans(ms, 3)
    return MSInputs(ms=ms)


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write")])
def save(ms: MeasurementSetV2) -> MSInputs:
    return MSInputs(ms=ms)


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write")])
def image(ms: MeasurementSetV2, value: int) -> MSInputs:
    from tests._dataset_fixtures import set_scans

    set_scans(ms, value)
    import os

    if os.environ.get("REQUIRED_CACHE_KILL_IMAGE"):
        if os.environ.get("REQUIRED_CACHE_REMOVE_MAIN"):
            (ms / "table.dat").unlink()
        os._exit(73)
    return MSInputs(ms=ms)


def pipeline(value=7):
    recipe = Recipe(name="continuum", inputs_model=MSInputs, outputs_model=Empty)
    recipe.add_step("acquire", acquire, ms=InputRef(field="ms"))
    recipe.add_step("read", read, ms=OutputRef(step="acquire", field="ms"))
    recipe.add_step("prepare", prepare, ms=OutputRef(step="acquire", field="ms"))
    recipe.steps[-1].after = ["read"]
    recipe.add_step("save", save, ms=OutputRef(step="prepare", field="ms"))
    recipe.add_step("image", image, ms=OutputRef(step="save", field="ms"), value=value)
    return recipe


def test_exact_create_prefix_replay_then_changed_image(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    from tests._dataset_fixtures import scans, ROWS, attempts

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    config = AppConfig()
    config.cache.enabled = True
    config.cache.dir = str(tmp_path / "cache")
    ms = tmp_path / "obs.ms"
    assert _dispatch(pipeline(), None, ms=ms, _config=config).success
    manifest = get_cache_manifest(config.cache.dir)
    config.cache.required = {f"continuum.{name}": manifest.entry(f"continuum.{name}")["cache_key"] for name in ("acquire", "read", "prepare", "save")}
    assert _dispatch(pipeline(9), None, ms=ms, _config=config).success
    assert scans(ms) == [9] * ROWS
    latest = attempts(tmp_path)[-1]
    assert {leaf.step_path: leaf.cache.decision for leaf in latest.leaves} == {
        "continuum.acquire": "hit",
        "continuum.read": "hit",
        "continuum.prepare": "hit",
        "continuum.save": "hit",
        "continuum.image": "miss",
    }


@pytest.mark.parametrize("marker_kind", ["group", "historical-scalar", "corrupt-source"])
@pytest.mark.parametrize("corrupt_live", [False, True], ids=["valid-live", "missing-main-table-dat"])
def test_interrupted_creator_prefix_recovers_same_workspace(tmp_path, monkeypatch, marker_kind, corrupt_live, claim_tamper="none"):
    pytest.importorskip("casacore.tables")
    import os
    import subprocess
    import sys
    from shinobi.ownership import reconcile_ownership
    from shinobi.snapshots import get_journal, chain_id
    from tests._dataset_fixtures import scans, ROWS, attempts

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repository), env["PYTHONPATH"]])
    env["REQUIRED_CACHE_KILL_IMAGE"] = "1"
    if corrupt_live:
        env["REQUIRED_CACHE_REMOVE_MAIN"] = "1"
    script = """
from pathlib import Path
from tests.test_required_cache import pipeline
from shinobi.config import AppConfig
from shinobi.steps.dispatch import _dispatch
config = AppConfig()
config.cache.enabled = True
config.cache.dir = str(Path('cache').resolve())
_dispatch(pipeline(), None, ms=Path('obs.ms').resolve(), _config=config)
"""
    child = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert child.returncode == 73, child.stderr
    config = AppConfig()
    config.cache.enabled = True
    config.cache.dir = str(tmp_path / "cache")
    ms = tmp_path / "obs.ms"
    journal = get_journal(config.cache.dir)
    assert journal.get(chain_id(ms)).marker is not None
    if corrupt_live:
        assert not (ms / "table.dat").exists()
        from shinobi.dataset_closure import resolve_dataset_closure

        assert not resolve_dataset_closure(ms, storage_namespace=tmp_path).valid
    if marker_kind != "group":

        def change_marker(chain):
            chain.marker.strict_group = None
            if marker_kind == "corrupt-source":
                chain.marker.strict_rollback_fingerprint = "corrupt"
            return chain

        journal.update_chain(chain_id(ms), change_marker)
    manifest = get_cache_manifest(config.cache.dir)
    config.cache.required = {f"continuum.{name}": manifest.entry(f"continuum.{name}")["cache_key"] for name in ("acquire", "read", "prepare", "save")}
    assert reconcile_ownership(tmp_path).liveness == "dead"
    if marker_kind == "corrupt-source":
        with pytest.raises(DatasetLifecycleUnavailableError, match="frozen identity"):
            _dispatch(pipeline(9), None, ms=ms, _config=config)
        if corrupt_live:
            assert not (ms / "table.dat").exists()
        else:
            assert scans(ms) == [7] * ROWS
        assert journal.get(chain_id(ms)).marker is not None
        return
    if claim_tamper != "none":
        import shinobi.ownership as ownership
        from tests._dataset_fixtures import set_scans

        acquire_workspace = ownership.acquire_workspace

        def acquire_then_tamper(*args, **kwargs):
            lease = acquire_workspace(*args, **kwargs)
            if claim_tamper == "marker":

                def swap(chain):
                    chain.marker.run_id = "swapped-after-claim"
                    return chain

                journal.update_chain(chain_id(ms), swap)
            else:
                source = Path(journal.get(chain_id(ms)).marker.strict_rollback_source)
                set_scans(source, 99)
            return lease

        monkeypatch.setattr(ownership, "acquire_workspace", acquire_then_tamper)
        with pytest.raises(DatasetLifecycleUnavailableError, match="changed after claim|frozen identity"):
            _dispatch(pipeline(9), None, ms=ms, _config=config)
        assert not (ms / "table.dat").exists()
        assert journal.get(chain_id(ms)).marker is not None
        assert attempts(tmp_path)[-1].recovery == ()
        return
    assert _dispatch(pipeline(9), None, ms=ms, _config=config).success
    assert scans(ms) == [9] * ROWS
    assert journal.get(chain_id(ms)).marker is None
    latest = attempts(tmp_path)[-1]
    assert latest.recovery
    decisions = {leaf.step_path: leaf.cache.decision for leaf in latest.leaves}
    assert decisions == {"continuum.acquire": "hit", "continuum.read": "hit", "continuum.prepare": "hit", "continuum.save": "hit", "continuum.image": "miss"}


def test_required_missing_recorded_product_refuses(tmp_path):
    cab, recorder = _cab(tmp_path / "cache")
    first = _dispatch(cab, None, x=1)
    config = AppConfig()
    config.cache.required = {"tool": first.cache_key}
    manifest = get_cache_manifest(str(tmp_path / "cache"))
    manifest.update(lambda entries: entries["tool"]["products"].append(str(tmp_path / "missing.fits")))
    with pytest.raises(DatasetLifecycleUnavailableError, match="no valid reusable hit"):
        _dispatch(cab, None, x=1, _config=config)
    assert len(recorder.calls) == 1


@pytest.mark.parametrize("damage", ["run_id", "key", "bytes", "symlink", "foreign_workspace"])
def test_create_replay_refuses_unproved_authority(tmp_path, monkeypatch, damage):
    pytest.importorskip("casacore.tables")
    from shinobi.dataset_access import DatasetAccessError
    from tests._dataset_fixtures import set_scans, scans, ROWS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    config = AppConfig()
    config.cache.enabled = True
    config.cache.dir = str(tmp_path / "cache")
    ms = tmp_path / "obs.ms"
    assert _dispatch(pipeline(), None, ms=ms, _config=config).success
    manifest = get_cache_manifest(config.cache.dir)
    config.cache.required = {"continuum.acquire": manifest.entry("continuum.acquire")["cache_key"]}
    if damage == "run_id":
        manifest.update(lambda entries: entries["continuum.acquire"].update(run_id="foreign"))
    elif damage == "key":
        config.cache.required["continuum.acquire"] = "wrong"
    elif damage == "bytes":
        set_scans(ms, 99)
    elif damage == "symlink":
        actual = tmp_path / "actual.ms"
        ms.rename(actual)
        ms.symlink_to(actual, target_is_directory=True)
    else:
        other = tmp_path / "other-workspace"
        other.mkdir()
        monkeypatch.chdir(other)
    with pytest.raises((DatasetAccessError, DatasetLifecycleUnavailableError)):
        _dispatch(pipeline(9), None, ms=ms, _config=config)
    assert scans(ms) == ([99] if damage == "bytes" else [7]) * ROWS


def test_required_writer_rejected_lineage_never_restores(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    from shinobi.snapshots import get_journal, chain_id
    from tests._dataset_fixtures import make_ms, scans, ROWS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    ms = make_ms(tmp_path / "obs.ms")
    config = AppConfig()
    config.cache.enabled = True
    config.cache.dir = str(tmp_path / "cache")
    first = _dispatch(image.step, image.func, ms=ms, value=7, _config=config)
    config.cache.required = {"image": first.cache_key}
    journal = get_journal(config.cache.dir)

    def disconnect(chain):
        chain.head = chain.generations[0].name
        return chain

    journal.update_chain(chain_id(ms), disconnect)
    # The stale head also disagrees with live bytes. Required reuse must stop
    # before SnapshotGuard can restore that predecessor or execute the writer.
    with pytest.raises(DatasetLifecycleUnavailableError, match="no valid reusable hit"):
        _dispatch(image.step, image.func, ms=ms, value=7, _config=config)
    assert scans(ms) == [7] * ROWS


def test_required_scatter_name_cannot_be_silently_ignored(tmp_path):
    from shinobi.steps.schema import ScatterSpec

    cab, recorder = _cab(tmp_path / "cache")
    recipe = Recipe(name="scattered", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("tool", cab, x=[1, 2])
    recipe.steps[-1].scatter = ScatterSpec(fields=["x"])
    config = AppConfig()
    config.cache.required = {"scattered.tool": "key"}
    with pytest.raises(DatasetLifecycleUnavailableError, match="scattered leaf"):
        _dispatch(recipe, None, _config=config)
    assert recorder.calls == []


@pytest.mark.parametrize("tamper", ["marker", "source"])
def test_provisional_recovery_revalidates_frozen_authority_after_claim(tmp_path, monkeypatch, tamper):
    test_interrupted_creator_prefix_recovers_same_workspace(tmp_path, monkeypatch, "historical-scalar", True, claim_tamper=tamper)
