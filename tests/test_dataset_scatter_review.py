"""Regression checks for bounded scatter planning and scheduler failure paths."""

import threading

import pytest
from pydantic import BaseModel, Field

from shinobi import CasaTable, DatasetAccess, DatasetColumns, FITSImage, MeasurementSetV2, Recipe, pystep
import shinobi.dataset_access as access_module
import shinobi.steps.dispatch as dispatch_module
from shinobi.dataset_access import DatasetAccessError, ResolvedAccessPlanner, _validate_scatter_collisions, plan_recipe_accesses
from shinobi.exceptions import DatasetLifecycleUnavailableError, ParameterError
from shinobi.steps.schema import InputRef, ParamMeta, Scope
from tests._dataset_fixtures import attempts, make_ms
from tests.test_dataset_access import _closure


class Empty(BaseModel):
    pass


class MSInput(BaseModel):
    ms: MeasurementSetV2


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))


@pytest.mark.parametrize("shape", ["atomic", "nested", "nested-scatter"])
def test_fits_scatter_remains_plannable_but_cannot_execute(tmp_path, shape):
    @pystep()
    def read(image: FITSImage) -> None:
        pytest.fail("unsupported FITS execution launched")

    image = tmp_path / "absent.fits"
    recipe = Recipe(name="fits", inputs_model=Empty, outputs_model=Empty)
    if shape == "atomic":
        recipe.add_step("read", read, scatter=["image"], image=[image])
    elif shape == "nested":
        inner = Recipe(name="inner", inputs_model=Empty, outputs_model=Empty)
        inner.add_step("read", read, scatter=["image"], image=[image])
        recipe.add_step("inner", inner)
    else:

        class ImageInput(BaseModel):
            image: FITSImage

        inner = Recipe(name="inner", inputs_model=ImageInput, outputs_model=Empty)
        inner.add_step("read", read, image=InputRef(field="image"))
        recipe.add_step("inner", inner, scatter=["image"], image=[image])
    assert plan_recipe_accesses(recipe, {}, workspace=tmp_path).graph.names == [recipe.steps[0].name]
    with pytest.raises(DatasetLifecycleUnavailableError, match="FITS dataset contract|nested dataset recipe"):
        recipe()


def test_casa_table_write_inference_keeps_non_fits_declarations():
    class Table(BaseModel):
        table: CasaTable

    creator = Scope(name="create", inputs_model=Empty, outputs_model=Table)
    writer = Scope(name="write", inputs_model=Table, outputs_model=Empty, field_meta={"table": ParamMeta(write_path=True)})
    assert dispatch_module._leaf_dataset_writes(creator) == {"table"}
    assert dispatch_module._leaf_dataset_writes(writer) == {"table"}


def test_shared_reads_deduplicate_exact_accesses_and_memoize_closure_per_plan(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    calls = []

    def resolve(path, **kwargs):
        calls.append(path)
        return _closure(path, tmp_path)

    monkeypatch.setattr(access_module, "resolve_dataset_closure", resolve)

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        pass

    recipe = Recipe(name="shared", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=list(range(200)))
    plan = plan_recipe_accesses(recipe, {"ms": root}, workspace=tmp_path)
    assert len(plan.accesses["read"]) == 1
    assert calls == [root]
    # Revalidation owns a new planner rather than retaining stale closures.
    plan_recipe_accesses(recipe, {"ms": root}, workspace=tmp_path)
    assert calls == [root, root]


def test_dedup_preserves_distinct_column_declarations_and_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda path, **kwargs: _closure(path, tmp_path))
    left, right = tmp_path / "left.ms", tmp_path / "right.ms"

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("{column}",)))])
    def read(ms: MeasurementSetV2, column: str) -> None:
        pass

    recipe = Recipe(name="distinct", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["ms", "column"], ms=[left, left, left, right], column=["DATA", "FLAG", "DATA", "DATA"])
    accesses = plan_recipe_accesses(recipe, {}, workspace=tmp_path).accesses["read"]
    assert [(access.root, access.declaration.columns.read) for access in accesses] == [(left, ("DATA",)), (left, ("FLAG",)), (right, ("DATA",))]


def test_closure_cache_preserves_declaration_specific_subtable_checks(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    calls = []

    def resolve(path, **kwargs):
        calls.append(path)
        return _closure(path, tmp_path)

    monkeypatch.setattr(access_module, "resolve_dataset_closure", resolve)
    planner = ResolvedAccessPlanner(tmp_path)
    reader = Scope(name="read", inputs_model=MSInput, outputs_model=Empty)
    planner.order_after("read", reader, {"ms": root})
    antenna = Scope(name="antenna", inputs_model=MSInput, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", table="ANTENNA")])
    with pytest.raises(DatasetAccessError, match="ANTENNA"):
        planner.order_after("antenna", antenna, {"ms": root})
    assert calls == [root]


@pytest.mark.parametrize("parent_writes", [False, True])
def test_ancestor_scatter_overlap_refuses_either_write_direction(tmp_path, parent_writes):
    parent, child = tmp_path / "tree", tmp_path / "tree" / "child"
    with pytest.raises(DatasetAccessError, match="scatter slices 0 and 1"):
        _validate_scatter_collisions("read", [[(parent, parent_writes)], [(child, not parent_writes)]])
    # Repeated accesses belonging to one slice are not a cross-slice hazard.
    _validate_scatter_collisions("read", [[(parent, parent_writes), (child, not parent_writes)]])


def test_collision_groups_keep_a_write_owner_outside_the_first_two_readers(tmp_path):
    path = tmp_path / "shared"
    with pytest.raises(DatasetAccessError, match="overlapping"):
        _validate_scatter_collisions("read", [[(path, False)], [(path, False)], [(path, True)]])
    _validate_scatter_collisions("read", [[(path, False)] for _ in range(1000)])


def test_failed_batch_names_recipe_and_step_and_drains_inflight_peer(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    called = []

    class Indices(BaseModel):
        indices: list[int] = [1, 2]

    class Limited(MSInput):
        index: int = Field(le=10)

    @pystep()
    def peer(ms: MeasurementSetV2) -> None:
        started.set()
        assert release.wait(timeout=10)
        finished.set()

    @pystep()
    def producer() -> Indices:
        assert started.wait(timeout=10)
        threading.Timer(0.1, release.set).start()
        return Indices(indices=[1, 200])

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        called.append(index)

    read.step.inputs_model = Limited
    recipe = Recipe(name="outer", inputs_model=Empty, outputs_model=Empty, max_workers=2)
    recipe.add_step("peer", peer, ms=ms)
    recipe.add_step("producer", producer)
    recipe.add_step("read", read, scatter=["index"], ms=ms, index=recipe.outputs.producer.indices)
    try:
        with pytest.raises(ParameterError, match="step 'read' in recipe 'outer' failed: read: parameter validation failed"):
            recipe()
    finally:
        release.set()
    assert not called
    assert finished.is_set()
    assert {leaf.step_path for leaf in attempts(tmp_path)[0].leaves} == {"outer.peer"}


def test_product_free_slices_keep_planned_inputs_none(tmp_path, monkeypatch):
    ms = make_ms(tmp_path / "obs.ms")
    original = dispatch_module._dispatch
    observed = []

    def dispatch(*args, **kwargs):
        if kwargs.get("_slice_index") is not None:
            observed.append(kwargs.get("_planned_inputs"))
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch_module, "_dispatch", dispatch)

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        pass

    recipe = Recipe(name="free", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["index"], ms=ms, index=[0, 1])
    assert recipe().success
    assert observed == [None, None]
