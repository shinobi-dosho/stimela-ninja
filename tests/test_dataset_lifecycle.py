from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi.dataset_lifecycle as lifecycle_module
import shinobi.dataset_access as access_module
import shinobi.ownership as ownership_module
import shinobi.steps.dispatch as dispatch_module
from shinobi import Cab, DatasetAccess, DatasetBackendStatus, DatasetColumns, DatasetMode, DatasetNamespaceMode, MeasurementSetV2, Recipe, pystep, read_dataset_attempt
from shinobi.backends.recording import RecordingBackend
from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.dataset_access import DatasetAccessError, DatasetFallback, ResolvedDatasetAccess
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.dataset_lifecycle import (
    DatasetFileObservation,
    DatasetLifecycle,
    DatasetLifecycleAttempt,
    DatasetLifecyclePhase,
    DatasetLifecycleSnapshot,
    DatasetLifecycleStore,
    DatasetObservation,
    StrictLeaf,
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


class SelectedColumn(BaseModel):
    column: str | None


class ColumnInput(MSInput):
    column: str | None


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
    _install_real_resolution(monkeypatch, workspace, root)
    monkeypatch.setattr(lifecycle_module, "resolve_lifecycle_snapshot", resolve)


def _install_local_container_configuration(monkeypatch, workspace: Path) -> None:
    """Keep capability unit tests independent of the caller's CLI context."""

    for name in (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "CONTAINER_HOST",
        "CONTAINER_CONNECTION",
        "PODMAN_CONNECTION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(workspace / "empty-docker-config"))
    monkeypatch.setenv("PODMAN_CONNECTIONS_CONF", str(workspace / "empty-podman-connections.json"))
    monkeypatch.setenv("CONTAINERS_CONF", str(workspace / "empty-containers.conf"))


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
    assert attempt.schema_version == 2
    [leaf] = attempt.leaves
    assert leaf.outcome == "committed"
    assert leaf.mutations == ()
    assert leaf.accesses == access_module.resolve_scope_dataset_accesses(read.step, {"ms": ms}, workspace=tmp_path)
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


def test_historical_version_one_read_attempt_roundtrips(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)

    @pystep()
    def read(ms: MeasurementSetV2) -> None:
        pass

    assert read(ms=ms).success
    historical = _attempts(tmp_path)[0].model_dump(mode="json")
    historical["schema_version"] = 1
    for field in ("leaves", "absent_roots", "recovery"):
        historical.pop(field)
    attempt = DatasetLifecycleAttempt.model_validate(historical)
    assert attempt.schema_version == 1
    assert attempt.leaves == ()
    store = DatasetLifecycleStore(tmp_path / "historical.json")
    store.create(attempt)
    assert read_dataset_attempt(store.path) == attempt


@pytest.mark.parametrize("route", ["pystep-native", "cab-native", "cab-docker"])
@pytest.mark.parametrize("column", ["DATA", None, "BAD-NAME"])
def test_runtime_reader_columns_resolve_before_execution_and_cache(tmp_path, monkeypatch, route, column):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    _install_local_container_configuration(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    called = []
    checks = []
    backend = RecordingBackend()
    backend_name = "docker" if route == "cab-docker" else "native"
    monkeypatch.setitem(dispatch_module._STEP_BACKENDS, backend_name, backend)
    declaration = DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("{column}",)))

    @pystep()
    def select() -> SelectedColumn:
        return SelectedColumn(column=column)

    if route == "pystep-native":

        @pystep(dataset_accesses=[declaration])
        def reader(ms: MeasurementSetV2, column: str | None) -> None:
            called.append(column)
    else:
        reader = Cab(
            name="reader",
            command="read-tool",
            image="reader:latest" if backend_name == "docker" else None,
            backend=backend_name,
            inputs_model=ColumnInput,
            outputs_model=Empty,
            dataset_accesses=[declaration],
        )

    recipe = Recipe(name="dynamic-reader", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("select", select)
    recipe.add_step("read", reader, ms=InputRef(field="ms"), column=OutputRef(step="select", field="column"))
    manifest = get_cache_manifest(str(cache_dir))
    original_check = manifest.check

    def check(step_path, *args, **kwargs):
        checks.append(step_path)
        return original_check(step_path, *args, **kwargs)

    monkeypatch.setattr(manifest, "check", check)
    if column == "BAD-NAME":
        with pytest.raises(DatasetAccessError, match="column|BAD-NAME"):
            recipe(ms=ms, cache=True, cache_dir=str(cache_dir))
        [attempt] = _attempts(tmp_path)
        [leaf] = attempt.leaves
        assert leaf.outcome == "refused"
        assert leaf.accesses == ()
        assert "BAD-NAME" in leaf.reason
        assert not called and not backend.calls
        assert "dynamic-reader.read" not in checks
        assert manifest.entry("dynamic-reader.read") is None
        return

    assert recipe(ms=ms, cache=True, cache_dir=str(cache_dir)).success
    assert recipe(ms=ms, cache=True, cache_dir=str(cache_dir)).success
    records = _attempts(tmp_path)
    assert len(records) == 2
    assert {record.leaves[0].outcome for record in records} == {"committed", "reused"}
    for attempt in records:
        assert attempt.schema_version == 2
        [planned] = attempt.planned_accesses
        assert planned.declaration.columns is None
        assert planned.fallback is DatasetFallback.UNKNOWN_COLUMNS
        [leaf] = attempt.leaves
        [resolved] = leaf.accesses
        assert resolved.declaration.columns == (DatasetColumns(read=(column,)) if column is not None else None)
        assert resolved.fallback is (None if column is not None else DatasetFallback.UNKNOWN_COLUMNS)
        assert leaf.mutations == ()
        assert "journal" not in leaf.reason and "snapshot" not in leaf.reason
        assert leaf.cache.decision in {"hit", "miss"}
    if route == "pystep-native":
        assert called == [column]
    else:
        assert len(backend.calls) == 1
        [plan] = backend.dataset_plans
        assert all(not mount.writable for mount in plan.mounts)
        if backend_name == "docker":
            assert [mount.source for mount in plan.mounts] == [ms.resolve()]


def test_runtime_reader_resolution_exception_survives_refused_leaf_record_failure(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)
    monkeypatch.chdir(tmp_path)

    @pystep()
    def select() -> SelectedColumn:
        return SelectedColumn(column="BAD-NAME")

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("{column}",)))])
    def reader(ms: MeasurementSetV2, column: str | None) -> None:
        pytest.fail("refused reader executed")

    def unavailable(*args, **kwargs):
        raise OSError("leaf store unavailable")

    monkeypatch.setattr(DatasetLifecycle, "record_leaf", unavailable)
    recipe = Recipe(name="dynamic-reader", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("select", select)
    recipe.add_step("read", reader, ms=InputRef(field="ms"), column=OutputRef(step="select", field="column"))
    with pytest.raises(DatasetAccessError, match="BAD-NAME") as caught:
        recipe(ms=ms, cache=False)
    assert any("leaf store unavailable" in note for note in caught.value.__cause__.__notes__)


@pytest.mark.parametrize("claim_present", [False, True])
def test_reader_claim_refusal_records_no_unclaimed_accesses(tmp_path, monkeypatch, claim_present):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_real_resolution(monkeypatch, tmp_path, ms)

    @pystep()
    def reader(ms: MeasurementSetV2) -> None:
        pass

    lifecycle = DatasetLifecycle.start(workspace=tmp_path, attempt_id="claim-refusal", scope="reader", backends=("native",))
    lease = acquire_workspace(tmp_path, "unrelated", kind="local", accesses=[(tmp_path / "unrelated.ms", False)]) if claim_present else None
    try:
        if lease is not None:
            lifecycle.amend(claim=lease.owner)
        with pytest.raises(DatasetLifecycleUnavailableError, match="no workflow claim|does not cover"):
            StrictLeaf(lifecycle, "reader", reader.step, {"ms": ms})
        attempt = read_dataset_attempt(lifecycle.store.path)
        assert attempt.schema_version == 2
        [leaf] = attempt.leaves
        assert leaf.accesses == ()
        assert leaf.outcome == "refused"
        assert "claim" in leaf.reason
    finally:
        if lease is not None:
            lease.release()


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
    # state -- which it cannot for an explicitly uncached writer.
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode=mode)])
    def mutate(ms: MeasurementSetV2) -> None:
        pytest.fail("mutating strict route executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="caching explicitly disabled"):
        mutate(ms=ms, cache=False)
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


def test_multi_root_reads_and_external_unsupported_closure_refusals(tmp_path, monkeypatch):
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
    monkeypatch.setattr(lifecycle_module, "_observe_root", lambda root, workspace: _snapshot(root).observations[0])
    assert len(resolve_lifecycle_snapshot(read.step, values, workspace=tmp_path).observations) == 2
    monkeypatch.undo()
    monkeypatch.setattr(lifecycle_module, "resolve_scope_dataset_accesses", lambda *args, **kwargs: (left_access,))

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


@pytest.mark.parametrize("backend", ["venv", "docker", "podman", "apptainer", "singularity"])
def test_local_adapter_reader_routes_commit_with_capability_evidence(tmp_path, monkeypatch, backend):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    _install_local_container_configuration(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    recording = RecordingBackend()
    monkeypatch.setitem(dispatch_module._STEP_BACKENDS, backend, recording)
    reader = Cab(
        name="reader",
        command="reader",
        image="reader:latest" if backend != "venv" else None,
        inputs_model=MSInput,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )

    assert reader(ms=ms, backend=backend).success
    assert len(recording.calls) == 1
    [plan] = recording.dataset_plans
    assert plan.capability.backend == backend
    if backend == "venv":
        assert plan.mounts == ()
    else:
        assert [(mount.source, mount.target, mount.writable) for mount in plan.mounts] == [(ms.resolve(), ms.resolve(), False)]
    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.COMMITTED
    [capability] = attempt.backend_capabilities
    assert capability.backend == backend
    assert capability.status is DatasetBackendStatus.TESTED
    assert capability.namespace_mode is (DatasetNamespaceMode.LOCAL if backend == "venv" else DatasetNamespaceMode.IDENTITY_BIND)


@pytest.mark.parametrize("backend", ["slurm", "kubernetes"])
def test_distributed_reader_routes_record_an_explicit_capability_refusal(tmp_path, monkeypatch, backend):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("unsupported route executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="storage namespace"):
        read(ms=ms, backend=backend)

    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.REFUSED
    [capability] = attempt.backend_capabilities
    assert capability.backend == backend
    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert capability.namespace_mode is DatasetNamespaceMode.UNPROVEN


def test_remote_launch_surface_cannot_be_selected_as_a_step_backend(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("remote launch surface executed as a local backend")

    with pytest.raises(DatasetLifecycleUnavailableError, match="launcher makes no local dataset claim"):
        read(ms=ms, backend="remote")

    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.REFUSED
    [capability] = attempt.backend_capabilities
    assert capability.namespace_mode is DatasetNamespaceMode.REMOTE_DELEGATED


def test_remote_container_daemon_is_refused_before_leaf_execution(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCKER_HOST", "ssh://worker.example")

    @pystep(image="reader:latest", dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pytest.fail("remote Docker route executed")

    with pytest.raises(DatasetLifecycleUnavailableError, match="DOCKER_HOST"):
        read(ms=ms, backend="docker")

    attempt = _attempts(tmp_path)[0]
    assert attempt.phase is DatasetLifecyclePhase.REFUSED
    [capability] = attempt.backend_capabilities
    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert capability.namespace_mode is DatasetNamespaceMode.UNPROVEN


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


def test_non_contract_leaf_may_use_another_tested_local_route(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    monkeypatch.chdir(tmp_path)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        called.append("read")

    @pystep(backend="venv")
    def unrelated() -> None:
        called.append("unrelated")

    called = []
    recipe = Recipe(name="mixed", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("read", read, ms=InputRef(field="ms"))
    recipe.add_step("unrelated", unrelated)
    with pytest.warns(UserWarning, match="no venv is declared"):
        assert recipe(ms=ms).success
    assert called == ["read", "unrelated"]
    attempt = _attempts(tmp_path)[0]
    assert {capability.backend for capability in attempt.backend_capabilities} == {"native", "venv"}


def test_unannotated_container_leaf_receives_the_workflow_closure_plan(tmp_path, monkeypatch):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("data")
    _install_lifecycle(monkeypatch, tmp_path, [_snapshot(ms)])
    _install_local_container_configuration(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    native = RecordingBackend()
    container = RecordingBackend()
    monkeypatch.setitem(dispatch_module._STEP_BACKENDS, "native", native)
    monkeypatch.setitem(dispatch_module._STEP_BACKENDS, "docker", container)
    reader = Cab(
        name="reader",
        command="reader",
        inputs_model=MSInput,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )
    unrelated = Cab(
        name="unrelated",
        command="unrelated",
        image="unrelated:latest",
        backend="docker",
        inputs_model=Empty,
        outputs_model=Empty,
    )
    recipe = Recipe(name="mixed", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("reader", reader, ms=InputRef(field="ms"))
    recipe.add_step("unrelated", unrelated)

    assert recipe(ms=ms).success
    [plan] = container.dataset_plans
    assert [(mount.source, mount.target, mount.writable) for mount in plan.mounts] == [(ms.resolve(), ms.resolve(), False)]


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
