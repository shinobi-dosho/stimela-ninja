from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi.dataset_lifecycle as lifecycle_module
import shinobi.dataset_access as access_module
import shinobi.ownership as ownership_module
from shinobi import DatasetAccess, DatasetMode, MeasurementSetV2, Recipe, pystep, read_dataset_attempt
from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.dataset_access import DatasetFallback, ResolvedDatasetAccess
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.dataset_lifecycle import (
    DatasetFileObservation,
    DatasetLifecycle,
    DatasetLifecycleAttempt,
    DatasetLifecyclePhase,
    DatasetLifecycleSnapshot,
    DatasetLifecycleStore,
    DatasetObservation,
    resolve_lifecycle_snapshot,
)
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_V1
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.ownership import WorkspaceLease, WorkspaceOwnershipStore, WorkspaceRegistryStore, acquire_workspace, release_workspace
from shinobi.snapshots import SnapshotGuard, chain_id, get_journal
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import InputRef, OutputRef
from shinobi.steps.schema import ScatterSpec, StepRef


class Empty(BaseModel):
    pass


class MSInput(BaseModel):
    ms: MeasurementSetV2


class RuntimeReport(BaseModel):
    report: Path


class DefaultReport(BaseModel):
    report: Path = Path("claimed-safe.txt")


def _snapshot(root: Path, *, size: int = 4) -> DatasetLifecycleSnapshot:
    root = root.resolve()
    member = root / "table.dat"
    requirements = (ClosureRequirement.TABLE_MEMBERS,)
    closure = DatasetClosure(
        requested_root=root,
        storage_namespace=root.parent,
        root=root,
        status=ClosureStatus.VALID,
        message="test closure",
        resources=(
            ClosureResource(
                path=root,
                namespace_path=Path(root.name),
                members=("MAIN",),
                table_files=(member.name,),
                external_to_root=False,
                storage_managers=("StandardStMan",),
            ),
        ),
        capabilities=ClosureCapabilities(
            copy_requirements=requirements,
            mount_requirements=requirements,
            stage_requirements=requirements,
            materialize_requirements=requirements,
            restore_requirements=requirements,
        ),
    )
    declaration = DatasetAccess(field="ms", mode=DatasetMode.READ)
    access = ResolvedDatasetAccess(
        field="ms",
        declaration=declaration,
        requested_path=root,
        root=root,
        resources=(root,),
        mode=DatasetMode.READ,
        whole_dataset=True,
        path_known=True,
        columns_known=False,
        fallback=DatasetFallback.UNKNOWN_COLUMNS,
        closure_status=ClosureStatus.VALID,
        reason=f"read whole dataset: {root}",
    )
    descriptor = DatasetDescriptor(
        path=root,
        expected=MSV2_STRUCTURAL_V1,
        status=DatasetStatus.VALID,
        message="valid",
    )
    observation = DatasetObservation(
        root=root,
        descriptor=descriptor,
        closure=closure,
        files=(
            DatasetFileObservation(
                path=member,
                device=1,
                inode=2,
                mode=0o100644,
                size=size,
                mtime_ns=3,
                ctime_ns=4,
            ),
        ),
    )
    return DatasetLifecycleSnapshot(accesses=(access,), observations=(observation,))


def _install_lifecycle(monkeypatch, workspace: Path, snapshots: list[DatasetLifecycleSnapshot]) -> None:
    sequence = iter(snapshots)
    last = snapshots[-1]

    def resolve(*args, **kwargs):
        nonlocal last
        last = next(sequence, last)
        return last

    root = snapshots[0].accesses[0].root
    assert root is not None
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(workspace / "registry.json"))
    monkeypatch.setattr(
        ownership_module,
        "scope_path_accesses",
        lambda *args, **kwargs: ([(root, False)], None),
    )
    monkeypatch.setattr(lifecycle_module, "resolve_lifecycle_snapshot", resolve)


def _attempts(workspace: Path) -> list[DatasetLifecycleAttempt]:
    paths = sorted((workspace / ".shinobi" / "dataset-attempts").glob("*.json"))
    return [DatasetLifecycleStore(path).attempt() for path in paths]


def _install_real_resolution(monkeypatch, workspace: Path, root: Path) -> None:
    """Keep the real planner/re-resolution/stat path, replacing casacore only."""

    snapshot = _snapshot(root)
    closure = snapshot.observations[0].closure
    descriptor = snapshot.observations[0].descriptor
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(workspace / "registry.json"))
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: closure)
    monkeypatch.setattr(lifecycle_module, "resolve_dataset_closure", lambda *args, **kwargs: closure)
    monkeypatch.setattr(lifecycle_module, "inspect_measurement_set_v2", lambda *args, **kwargs: descriptor)


def test_direct_native_reader_commits_versioned_baseline_and_releases_lease(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        assert ms == ms.resolve()

    result = read(ms=ms)

    assert result.success
    attempt = _attempts(tmp_path)[0]
    assert attempt.schema_version == 1
    assert attempt.phase is DatasetLifecyclePhase.COMMITTED
    assert attempt.outcome == "committed"
    assert attempt.capability_supported
    assert attempt.planned_accesses == snapshot.accesses
    assert attempt.pre_observations == snapshot.observations
    assert attempt.post_observations == snapshot.observations
    record_path = next((tmp_path / ".shinobi" / "dataset-attempts").glob("*.json"))
    assert read_dataset_attempt(record_path) == attempt
    assert [event.phase for event in attempt.events] == [
        DatasetLifecyclePhase.PLANNED,
        DatasetLifecyclePhase.PLANNED,
        DatasetLifecyclePhase.CLAIMED,
        DatasetLifecyclePhase.REVALIDATED,
        DatasetLifecyclePhase.EXECUTING,
        DatasetLifecyclePhase.VALIDATED,
        DatasetLifecyclePhase.COMMITTED,
    ]
    assert WorkspaceOwnershipStore(tmp_path).read().get("owners") == {}
    assert WorkspaceRegistryStore(tmp_path / "registry.json").read().get("owners") == {}


def test_flat_recipe_with_annotated_boundary_executes_under_one_read_claim(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.chdir(tmp_path)
    seen = []

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        seen.append(ms)

    recipe = Recipe(name="flat", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("read", read, ms=InputRef(field="ms"))

    assert recipe(ms=ms).success
    assert seen == [ms.resolve()]
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.COMMITTED


def test_reader_may_write_an_ordinary_non_dataset_product(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    report = tmp_path / "report.txt"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.setattr(
        ownership_module,
        "scope_path_accesses",
        lambda *args, **kwargs: ([(ms.resolve(), False), (report.resolve(), True)], None),
    )
    monkeypatch.chdir(tmp_path)

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> None:
        report.write_text(ms.name)

    assert read(ms=ms, report=report).success
    assert report.read_text() == ms.name
    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.COMMITTED
    assert any(access.writes and Path(access.path) == report.resolve() for access in attempt.claim.accesses)


def test_generic_write_overlapping_dataset_closure_is_refused(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    report = ms / "report.txt"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.setattr(
        ownership_module,
        "scope_path_accesses",
        lambda *args, **kwargs: ([(ms.resolve(), False), (report.resolve(), True)], None),
    )
    monkeypatch.chdir(tmp_path)

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> None:
        pytest.fail("overlapping generic write executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="generic write overlaps"):
        read(ms=ms, report=report)
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.REFUSED


def test_concurrent_native_readers_share_claim_and_clean_up_independently(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.chdir(tmp_path)
    entered = 0
    entered_lock = threading.Lock()
    both_entered = threading.Event()
    release = threading.Event()

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 2:
                both_entered.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read, ms=ms) for _ in range(2)]
        assert both_entered.wait(5)
        owners = WorkspaceOwnershipStore(tmp_path).read()["owners"]
        assert len(owners) == 2
        assert all(not access["writes"] for owner in owners.values() for access in owner["accesses"])
        release.set()
        assert all(future.result().success for future in futures)

    assert WorkspaceOwnershipStore(tmp_path).read()["owners"] == {}
    assert WorkspaceRegistryStore(tmp_path / "registry.json").read()["owners"] == {}
    assert {attempt.phase for attempt in _attempts(tmp_path)} == {DatasetLifecyclePhase.COMMITTED}


def test_existing_writer_refuses_reader_before_execution_and_preserves_writer(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.chdir(tmp_path)
    writer = acquire_workspace(tmp_path, "writer", kind="slurm", accesses=[(ms, True)])
    called = False

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        nonlocal called
        called = True

    try:
        with pytest.raises(DatasetLifecycleUnavailableError, match="claim refused"):
            read(ms=ms)
        assert not called
        assert WorkspaceOwnershipStore(tmp_path).owner("writer") == writer.owner
        attempt = _attempts(tmp_path)[0]
        assert attempt.phase is DatasetLifecyclePhase.REFUSED
        assert attempt.outcome == "refused"
    finally:
        writer.release()


def test_post_claim_change_refuses_before_execution_and_releases_reader(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    before = _snapshot(ms)
    changed = _snapshot(ms, size=5)
    _install_lifecycle(monkeypatch, tmp_path, [before, changed])
    monkeypatch.chdir(tmp_path)
    called = False

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        nonlocal called
        called = True

    with pytest.raises(DatasetLifecycleUnavailableError, match="changed after claim"):
        read(ms=ms)

    assert not called
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.REFUSED
    assert WorkspaceOwnershipStore(tmp_path).read()["owners"] == {}
    assert WorkspaceRegistryStore(tmp_path / "registry.json").read()["owners"] == {}


def test_changed_read_only_postcondition_is_a_durable_failure(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    before = _snapshot(ms)
    changed = _snapshot(ms, size=5)
    _install_lifecycle(monkeypatch, tmp_path, [before, before, changed])
    monkeypatch.chdir(tmp_path)
    called = False

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        nonlocal called
        called = True

    with pytest.raises(DatasetLifecycleViolationError, match="changed its dataset"):
        read(ms=ms)

    assert called
    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.FAILED
    assert attempt.outcome == "failed"
    assert attempt.post_observations == changed.observations
    assert WorkspaceOwnershipStore(tmp_path).read()["owners"] == {}


def test_reader_exception_records_failure_without_masking_original(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    snapshot = _snapshot(ms)
    _install_lifecycle(monkeypatch, tmp_path, [snapshot])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        raise RuntimeError("reader failed")

    with pytest.raises(RuntimeError, match="reader failed"):
        read(ms=ms)

    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.FAILED
    assert "RuntimeError" in attempt.reason
    assert attempt.post_observations == snapshot.observations
    assert WorkspaceOwnershipStore(tmp_path).read()["owners"] == {}


@pytest.mark.parametrize("mode", [DatasetMode.WRITE, DatasetMode.CREATE])
def test_mutation_without_nameable_states_is_refused_before_execution(tmp_path, monkeypatch, mode):
    # Write/create now runs under the mutation lifecycle (see
    # test_dataset_mutation.py), but only when exact recovery can name every
    # state -- which it cannot for an uncached writer.
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode=mode)])
    def mutate(ms: MeasurementSetV2) -> None:
        pytest.fail("mutating strict route executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="not cacheable"):
        mutate(ms=ms)
    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.REFUSED
    assert attempt.capability == "contained-native-msv2-mutation/v1"


def test_nested_and_scattered_dataset_recipes_remain_refused(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("unsupported recipe shape executed")

    inner = Recipe(
        name="inner",
        inputs_model=MSInput,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=read.step, func=read.func, wiring={"ms": InputRef(field="ms")})],
    )
    outer = Recipe(
        name="outer",
        inputs_model=MSInput,
        outputs_model=Empty,
        steps=[StepRef(name="inner", step=inner, wiring={"ms": InputRef(field="ms")})],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="nested dataset recipe"):
        outer(ms=ms)

    class ManyMS(BaseModel):
        ms: list[MeasurementSetV2]

    scattered = Recipe(
        name="scattered",
        inputs_model=ManyMS,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="read",
                step=read.step,
                func=read.func,
                wiring={"ms": InputRef(field="ms")},
                scatter=ScatterSpec(fields=["ms"]),
            )
        ],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="scatters a dataset contract"):
        scattered(ms=[ms])


def test_multi_root_external_and_unsupported_closures_remain_refused(tmp_path, monkeypatch):
    left = tmp_path / "left.ms"
    right = tmp_path / "right.ms"
    external = tmp_path / "shared" / "ANTENNA"
    for root in (left, right, external):
        root.mkdir(parents=True)
        (root / "table.dat").write_text("data")
    left_access = _snapshot(left).accesses[0]
    right_access = _snapshot(right).accesses[0]

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pass

    values = read.step.inputs_model(ms=left)
    monkeypatch.setattr(lifecycle_module, "resolve_scope_dataset_accesses", lambda *args, **kwargs: (left_access, right_access))
    with pytest.raises(DatasetLifecycleUnavailableError, match="exactly one closure root"):
        resolve_lifecycle_snapshot(read.step, values, workspace=tmp_path)

    external_resource = _snapshot(external).observations[0].closure.resources[0].model_copy(update={"members": ("ANTENNA",), "external_to_root": True})
    external_closure = _snapshot(left).observations[0].closure.model_copy(update={"resources": (*_snapshot(left).observations[0].closure.resources, external_resource)})
    monkeypatch.setattr(lifecycle_module, "resolve_scope_dataset_accesses", lambda *args, **kwargs: (left_access,))
    monkeypatch.setattr(lifecycle_module, "resolve_dataset_closure", lambda *args, **kwargs: external_closure)
    with pytest.raises(DatasetLifecycleUnavailableError, match="external closure resources"):
        resolve_lifecycle_snapshot(read.step, values, workspace=tmp_path)

    unsupported = external_closure.model_copy(update={"status": ClosureStatus.UNSUPPORTED, "message": "reference table", "root": None, "resources": ()})
    monkeypatch.setattr(lifecycle_module, "resolve_dataset_closure", lambda *args, **kwargs: unsupported)
    with pytest.raises(DatasetLifecycleUnavailableError, match="closure revalidation returned unsupported"):
        resolve_lifecycle_snapshot(read.step, values, workspace=tmp_path)


@pytest.mark.parametrize("backend", ["venv", "docker", "slurm"])
def test_non_native_reader_routes_remain_refused(tmp_path, monkeypatch, backend):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("unsupported route executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="not the supported native local route"):
        read(ms=ms, backend=backend)

    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.REFUSED


def test_postcondition_precedes_cache_and_run_manifest_publication(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    member = ms / "table.dat"
    member.write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "runs"
    config = AppConfig.model_validate({"provenance": {"enabled": True, "dir": str(run_dir)}})

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def dishonest_reader(ms: MeasurementSetV2) -> None:
        (ms / "table.dat").write_text("changed")

    with pytest.raises(DatasetLifecycleViolationError, match="changed its dataset"):
        _dispatch(
            dishonest_reader.step,
            dishonest_reader.func,
            ms=ms,
            cache=True,
            cache_dir=str(cache_dir),
            provenance=True,
            _config=config,
        )

    assert get_cache_manifest(str(cache_dir)).entry("dishonest_reader") is None
    assert not list(run_dir.glob("*.run.json"))
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.FAILED


@pytest.mark.parametrize("snapshots_mode", ["auto", "off"])
def test_pending_dataset_snapshot_is_refused_without_recovery_under_read_claim(tmp_path, monkeypatch, snapshots_mode):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    member = ms / "table.dat"
    member.write_text("complete")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    journal = get_journal(str(cache_dir))
    interrupted = SnapshotGuard(journal, "writer", "writer-key", "dead-writer", {"ms": ms}, {}, set(), force_copy=True)
    interrupted.before_run()
    member.write_text("PARTIAL")
    called = False

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        nonlocal called
        called = True

    config = AppConfig()
    config.cache.snapshots.mode = snapshots_mode
    with pytest.raises(DatasetLifecycleUnavailableError, match="pending mutation recovery"):
        _dispatch(read.step, read.func, ms=ms, cache=True, cache_dir=str(cache_dir), _config=config)

    assert not called
    assert member.read_text() == "PARTIAL"
    assert journal.get(chain_id(ms)).marker is not None


def test_runtime_path_output_and_outputref_write_are_refused_before_execution(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)

    called = []

    @pystep()
    def choose_report() -> RuntimeReport:
        called.append("choose")
        return RuntimeReport(report=ms / "report.txt")

    @pystep(write_paths=["report"])
    def write_report(report: Path) -> None:
        called.append("write")

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        called.append("read")

    recipe = Recipe(name="dynamic", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("choose", choose_report)
    recipe.add_step("write", write_report, report=OutputRef(step="choose", field="report"))
    recipe.add_step("read", read, ms=InputRef(field="ms"))

    with pytest.raises(DatasetLifecycleUnavailableError, match="runtime-dependent generic path input|runtime Python output"):
        recipe(ms=ms)
    assert called == []


def test_pystep_path_output_default_cannot_drift_after_claim(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)

    called = False

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> DefaultReport:
        nonlocal called
        called = True
        return DefaultReport(report=ms / "runtime-drift.txt")

    with pytest.raises(DatasetLifecycleUnavailableError, match="runtime Python output"):
        read(ms=ms)
    assert not called


def test_non_contract_leaf_is_also_bounded_to_native_backend(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("strict leaf executed")

    @pystep(backend="venv")
    def unrelated() -> None:
        pytest.fail("non-contract leaf executed")

    recipe = Recipe(name="mixed", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("read", read, ms=InputRef(field="ms"))
    recipe.add_step("unrelated", unrelated)
    with pytest.raises(DatasetLifecycleUnavailableError, match="every leaf under the lifecycle"):
        recipe(ms=ms)


def test_real_casacore_contained_read_when_available(tmp_path, monkeypatch):
    tables = pytest.importorskip("casacore.tables")
    if not hasattr(tables, "default_ms"):
        pytest.skip("installed python-casacore has no default_ms fixture builder")
    ms_path = tmp_path / "real.ms"
    ms = tables.default_ms(str(ms_path))
    ms.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
    ms.close()
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        assert (ms / "table.dat").is_file()

    assert read(ms=ms_path).success
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.COMMITTED


def test_lifecycle_store_failure_does_not_mask_backend_exception(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)
    original = DatasetLifecycle.transition

    def fail_terminal(self, phase, reason, **changes):
        if phase is DatasetLifecyclePhase.FAILED:
            raise OSError("attempt store unavailable")
        return original(self, phase, reason, **changes)

    monkeypatch.setattr(DatasetLifecycle, "transition", fail_terminal)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        raise RuntimeError("backend primary")

    with pytest.raises(RuntimeError, match="backend primary") as caught:
        read(ms=ms)
    assert any("attempt store unavailable" in note for note in getattr(caught.value, "__notes__", ()))
    assert _attempts(tmp_path)[0].phase is DatasetLifecyclePhase.EXECUTING
    assert WorkspaceOwnershipStore(tmp_path).read()["owners"] == {}


def test_lease_cleanup_failure_is_not_allowed_to_mask_backend_exception(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)
    original = WorkspaceLease.release

    def fail_release(self):
        raise OSError("lease store unavailable")

    monkeypatch.setattr(WorkspaceLease, "release", fail_release)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        raise RuntimeError("backend primary")

    with pytest.raises(RuntimeError, match="backend primary") as caught:
        read(ms=ms)
    assert any("lease store unavailable" in note for note in getattr(caught.value, "__notes__", ()))
    attempt = _attempts(tmp_path)[0]
    assert WorkspaceOwnershipStore(tmp_path).owner(attempt.claim.workflow_id) == attempt.claim

    monkeypatch.setattr(WorkspaceLease, "release", original)
    assert release_workspace(tmp_path, attempt.claim.workflow_id)
