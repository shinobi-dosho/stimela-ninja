"""Exclusive reads reserve ownership without becoming scientific mutations."""

from pathlib import Path
import sys

import pytest
from pydantic import BaseModel, ValidationError

from shinobi import Cab, DatasetAccess, MSv2, Recipe
from shinobi.dataset_access import DatasetAccessError, ResolvedAccessPlanner, dataset_workspace_accesses, resolve_scope_dataset_accesses, plan_recipe_accesses
from shinobi.dataset_lifecycle import DatasetLifecycleSnapshot, claim_covers_accesses, claim_covers_snapshot
from shinobi.offload._codec import pack, unpack
from shinobi.offload.bundle import ScopeSpec
from shinobi.ownership import WorkspaceAccess, WorkspaceOwner
from shinobi.steps.schema import InputRef, ScatterSpec, StepRef
from tests.test_dataset_access import _closure


class Inputs(BaseModel):
    ms: MSv2


class Empty(BaseModel):
    pass


def reader(exclusive=False, **kwargs):
    return Cab(name="read", command="true", inputs_model=Inputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", exclusive=exclusive, **kwargs)])


def test_exclusive_is_bounded_and_default_wire_shape_unchanged():
    ordinary = DatasetAccess(field="ms", mode="read")
    assert "exclusive" not in ordinary.model_dump(mode="json")
    for mode in ["write", "create"]:
        with pytest.raises(ValidationError, match="exclusive"):
            DatasetAccess(field="ms", mode=mode, exclusive=True)
    with pytest.raises(ValidationError):
        DatasetAccess(field="ms", mode="read", exclusive="yes")
    exclusive = DatasetAccess(field="ms", mode="read", exclusive=True)
    assert unpack(pack(exclusive)) == exclusive
    cab = reader(True)
    assert ScopeSpec.capture(cab).restore().dataset_accesses == [exclusive]


def test_exclusive_covers_linked_closure_and_claim_strength(tmp_path, monkeypatch):
    import shinobi.dataset_access as module

    root = tmp_path / "obs.ms"
    external = tmp_path / "linked" / "ANTENNA"
    monkeypatch.setattr(module, "resolve_dataset_closure", lambda value, **kwargs: _closure(Path(value), tmp_path, external))
    (access,) = resolve_scope_dataset_accesses(reader(True), {"ms": root}, workspace=tmp_path)
    assert access.requires_exclusive and not access.writes
    assert set(dataset_workspace_accesses([access])) == {(root, True), (external, True)}
    snapshot = DatasetLifecycleSnapshot(accesses=(access,), observations=())

    def owner(exclusive):
        return WorkspaceOwner(
            workflow_id="test",
            kind="local",
            workspace=str(tmp_path),
            acquired_at=0,
            registry=str(tmp_path / "owners"),
            accesses=tuple(WorkspaceAccess(path=str(p), writes=exclusive) for p in (root, external)),
        )

    assert set(claim_covers_accesses(owner(False), (access,))) == {root, external}
    assert not claim_covers_snapshot(owner(False), snapshot)
    assert not claim_covers_accesses(owner(True), (access,))
    assert claim_covers_snapshot(owner(True), snapshot)
    (ordinary,) = resolve_scope_dataset_accesses(reader(), {"ms": root}, workspace=tmp_path)
    assert claim_covers_snapshot(owner(True), DatasetLifecycleSnapshot(accesses=(ordinary,), observations=()))
    planner = ResolvedAccessPlanner(tmp_path)
    assert not planner.order_after("reader1", reader(), {"ms": root}).dependencies
    assert not planner.order_after("reader2", reader(), {"ms": tmp_path / "other.ms"}).dependencies
    assert planner.order_after("exclusive", reader(True), {"ms": root}).dependencies == {"reader1", "reader2"}
    assert planner.order_after("later", reader(), {"ms": root}).dependencies == {"exclusive"}


def test_unknown_exclusive_read_retains_reservation_strength(tmp_path):
    reserved = tmp_path / "reserved"
    (access,) = resolve_scope_dataset_accesses(reader(True, reservation=reserved), {}, workspace=tmp_path, unresolved_inputs={"ms"})
    assert not access.path_known and not access.writes and access.requires_exclusive
    assert dataset_workspace_accesses([access]) == [(reserved, True)]
    with pytest.raises(DatasetAccessError):
        resolve_scope_dataset_accesses(reader(True), {}, workspace=tmp_path, unresolved_inputs={"ms"})


def test_exclusive_read_scatter_conflicts_only_on_shared_closure(tmp_path, monkeypatch):
    import shinobi.dataset_access as module

    monkeypatch.setattr(module, "resolve_dataset_closure", lambda value, **kwargs: _closure(Path(value), tmp_path))

    class Many(BaseModel):
        ms: list[MSv2]

    recipe = Recipe(
        name="scatter",
        inputs_model=Many,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=reader(True), wiring={"ms": InputRef(field="ms")}, scatter=ScatterSpec(fields=["ms"]))],
    )
    with pytest.raises(DatasetAccessError, match="overlap|collision"):
        plan_recipe_accesses(recipe, {"ms": [tmp_path / "one.ms"] * 2}, workspace=tmp_path)
    assert plan_recipe_accesses(recipe, {"ms": [tmp_path / "one.ms", tmp_path / "two.ms"]}, workspace=tmp_path)


def test_native_flaghistory_repeated_dispatch_and_worker_preparation(tmp_path, monkeypatch):
    tables = pytest.importorskip("casacore.tables", reason="requires real MS tools")
    pytest.importorskip("msutils", reason="requires real MS tools")
    import numpy as np
    import dosho
    from tests._dataset_fixtures import make_ms
    from tests.test_offload_dataset_lifecycle import _prepared
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    root = make_ms(tmp_path / "obs.ms")
    with tables.table(str(root), readonly=False, ack=False) as table:
        table.putcol("FLAG", np.zeros((table.nrows(), 1, 1), dtype=bool))
        table.putcol("FLAG_ROW", np.zeros(table.nrows(), dtype=bool))
    # Use the environment's native entry point, without a container dependency.
    executable = Path(sys.executable).with_name("msutils")
    backup = dosho.get("msutils-flags-backup").model_copy(update={"backend": "native", "command": f"{executable} flags backup"})
    delete = dosho.get("msutils-flags-delete").model_copy(update={"backend": "native", "command": f"{executable} flags delete"})
    for _ in range(2):
        assert not _dispatch(backup, None, ms=root, name="roundtrip").cached
        assert (Path(str(root) + ".flagversions") / "flags.roundtrip").exists()
        assert not _dispatch(delete, None, ms=root, name="roundtrip").cached
        assert not (Path(str(root) + ".flagversions") / "flags.roundtrip").exists()
    recipe = Recipe(
        name="history",
        inputs_model=Inputs,
        outputs_model=Empty,
        steps=[
            StepRef(name="backup", step=backup, params={"name": "worker"}, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="delete", step=delete, params={"name": "worker"}, wiring={"ms": InputRef(field="ms")}),
        ],
    )
    workflow, plan, lease = _prepared(tmp_path, recipe, root, qualification_root=tmp_path.parent)
    try:
        assert any(Path(a.path) == root and a.writes for a in plan.accesses)
        assert workflow.execution_blocked_reason is None
    finally:
        lease.release()


def test_flaghistory_exclusive_read_keeps_ms_mount_readonly(tmp_path):
    import dosho
    from shinobi.backends.container import bind_dir_modes

    root = tmp_path / "obs.ms"
    root.mkdir()
    cab = dosho.get("msutils-flags-backup")
    mounts = bind_dir_modes(cab, {"ms": root, "name": "version"}, str(tmp_path))
    assert (str(root), False) in mounts
    assert (str(tmp_path), True) in mounts  # the separately owned sibling history
    assert not cab.input_mutability


@pytest.mark.parametrize("name", ["msutils-info", "msutils-summary", "msutils-flagstats", "msutils-flags-list"])
def test_existing_msutils_readers_respect_exclusive_owner_before_backend(name, tmp_path, monkeypatch):
    import dosho
    from shinobi.backends.recording import RecordingBackend
    from shinobi.ownership import acquire_workspace, WorkspaceOwnershipError, scope_requires_ownership
    from shinobi.steps.dispatch import _STEP_BACKENDS, _dispatch

    root = tmp_path / "obs.ms"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    backend = RecordingBackend()
    monkeypatch.setitem(_STEP_BACKENDS, "readonly-probe", backend)
    cab = dosho.get(name).model_copy(update={"backend": "readonly-probe"})
    assert scope_requires_ownership(cab)
    lease = acquire_workspace(tmp_path, "exclusive-owner", kind="local", accesses=[(root, True)])
    try:
        with pytest.raises(WorkspaceOwnershipError):
            _dispatch(cab, None, ms=root)
    finally:
        lease.release()
    assert not backend.calls


@pytest.mark.parametrize("mode", ["read", "write", "create"])
def test_fits_reservation_protocol_and_claim_strength(tmp_path, mode):
    from shinobi import FITSAccess, FITSImage
    from shinobi.fits_access import resolve_scope_fits_accesses

    class ImageInputs(BaseModel):
        image: FITSImage

    cab = Cab(name="fits", command="true", inputs_model=ImageInputs, outputs_model=Empty, fits_accesses=[FITSAccess(field="image", mode=mode)])
    path = tmp_path / "image.fits"
    (access,) = resolve_scope_fits_accesses(cab, {"image": path}, workspace=tmp_path)
    exclusive = mode != "read"
    assert access.requires_exclusive is exclusive
    assert dataset_workspace_accesses([access]) == [(path, exclusive)]
    owner = WorkspaceOwner(
        workflow_id="fits", kind="local", workspace=str(tmp_path), acquired_at=0, registry=str(tmp_path / "owners"), accesses=(WorkspaceAccess(path=str(path), writes=False),)
    )
    assert claim_covers_accesses(owner, (access,)) == ((path,) if exclusive else ())
    owner = owner.model_copy(update={"accesses": (WorkspaceAccess(path=str(path), writes=True),)})
    assert not claim_covers_accesses(owner, (access,))
    if mode != "create":
        planner = ResolvedAccessPlanner(tmp_path)
        assert not planner.order_after("first", cab, {"image": path}).dependencies
        assert planner.order_after("second", cab, {"image": path}).dependencies == ({"first"} if exclusive else set())
