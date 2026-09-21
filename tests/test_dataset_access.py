from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import shinobi.dataset_access as access_module
from shinobi import (
    DatasetAccess,
    DatasetColumns,
    DatasetFallback,
    DatasetMode,
    DatasetSelection,
    DatasetTable,
    MeasurementSetV2,
)
from shinobi.dataset_access import DatasetAccessError, ResolvedAccessPlanner, plan_recipe_accesses, resolve_scope_dataset_accesses
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.dag import graph_nodes, render_dag
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.graph import RecipeNotOffloadableError, check_offloadable
from shinobi.offload.slurm import MutationOrder, compile_slurm, submit_slurm
from shinobi.steps.pyfunc import pystep
from shinobi.steps.schema import Cab, InputRef, Recipe, ScatterSpec, Scope, StepRef


class MSIn(BaseModel):
    ms: MeasurementSetV2


class Empty(BaseModel):
    pass


def _closure(root: Path, namespace: Path, *extra: Path) -> DatasetClosure:
    resources = []
    for index, path in enumerate((root, *extra)):
        resources.append(
            ClosureResource(
                path=path.resolve(),
                namespace_path=path.resolve().relative_to(namespace.resolve()),
                members=("MAIN",) if index == 0 else ("ANTENNA",),
                table_files=("table.dat",),
                external_to_root=index > 0,
                storage_managers=("StandardStMan",),
            )
        )
    requirements = (ClosureRequirement.TABLE_MEMBERS,)
    return DatasetClosure(
        requested_root=root,
        storage_namespace=namespace.resolve(),
        root=root.resolve(),
        status=ClosureStatus.VALID,
        message="test closure",
        resources=tuple(resources),
        capabilities=ClosureCapabilities(
            copy_requirements=requirements,
            mount_requirements=requirements,
            stage_requirements=requirements,
            materialize_requirements=requirements,
            restore_requirements=requirements,
        ),
    )


def _scope(name: str, mode: DatasetMode, *, columns: DatasetColumns | None = None, table: DatasetTable = DatasetTable.MAIN) -> Scope:
    return Scope(
        name=name,
        inputs_model=MSIn,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode=mode, table=table, columns=columns)],
    )


def test_definition_is_json_serializable_without_resolving(monkeypatch):
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: pytest.fail("construction performed I/O"))
    scope = _scope(
        "flag",
        DatasetMode.WRITE,
        columns=DatasetColumns(read=("DATA",), write=("FLAG",)),
    )
    dumped = scope.model_dump_json()
    assert '"mode":"write"' in dumped
    assert '"write":["FLAG"]' in dumped

    with pytest.raises(DatasetLifecycleUnavailableError, match="declarative"):
        scope(ms="future.ms")

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def python_reader(ms: MeasurementSetV2) -> None:
        pass

    assert python_reader.step.dataset_accesses == [DatasetAccess(field="ms", mode="read")]


def test_offload_execution_eligibility_stays_refused():
    recipe = Recipe(
        name="strict",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=_scope("read", DatasetMode.READ), wiring={"ms": InputRef(field="ms")})],
    )
    with pytest.raises(RecipeNotOffloadableError, match="dataset lifecycle enforcement"):
        check_offloadable(recipe)


def test_declaration_validation_is_bounded_and_explicit():
    with pytest.raises(ValidationError, match="allow_schema_change"):
        DatasetAccess(field="ms", mode="write", columns=DatasetColumns(create=("MODEL_DATA",)))
    with pytest.raises(ValidationError, match="read dataset access"):
        DatasetAccess(field="ms", mode="read", allow_keyword_change=True)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DatasetSelection.model_validate({"taql": "ANTENNA1 == 0"})
    with pytest.raises(ValidationError, match="unknown field"):
        Scope(name="bad", inputs_model=MSIn, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="missing", mode="read")])
    with pytest.raises(ValidationError, match="atomic steps"):
        Recipe(name="bad-root", inputs_model=MSIn, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])


def test_hazards_and_read_read_are_deterministic(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))

    reader = _scope("reader", DatasetMode.READ, columns=DatasetColumns(read=("FLAG",)))
    writer = _scope("writer", DatasetMode.WRITE, columns=DatasetColumns(write=("FLAG",)))
    creator = _scope("creator", DatasetMode.CREATE, columns=DatasetColumns(write=("FLAG",)))

    order = ResolvedAccessPlanner(tmp_path)
    assert not order.order_after("read-1", reader, {"ms": ms}).dependencies
    assert not order.order_after("read-2", reader, {"ms": ms}).dependencies
    write = order.order_after("write", writer, {"ms": ms})
    assert write.dependencies == {"read-1", "read-2"}
    assert write.reasons["read-1"] == ("write-after-read: observation.ms, MAIN.FLAG",)
    read = order.order_after("read-3", reader, {"ms": ms})
    assert read.dependencies == {"write"}
    assert read.reasons["write"] == ("read-after-write: observation.ms, MAIN.FLAG",)
    create = order.order_after("create", creator, {"ms": ms})
    assert create.dependencies == {"read-3"}
    assert create.reasons["read-3"] == ("write-after-read: observation.ms, MAIN.FLAG",)

    writes = ResolvedAccessPlanner(tmp_path)
    assert not writes.order_after("write-1", writer, {"ms": ms}).dependencies
    second = writes.order_after("write-2", writer, {"ms": ms})
    assert second.dependencies == {"write-1"}
    assert second.reasons["write-1"] == ("write-after-write: observation.ms, MAIN.FLAG",)


def test_alias_parent_subtable_and_shared_external_resource_conflict(monkeypatch, tmp_path):
    left = tmp_path / "left.ms"
    right = tmp_path / "right.ms"
    external = tmp_path / "shared" / "ANTENNA"
    for path in (left / "ANTENNA", right, external):
        path.mkdir(parents=True, exist_ok=True)
    alias = tmp_path / "left-alias.ms"
    alias.symlink_to(left, target_is_directory=True)

    def resolve(root, **kwargs):
        root = Path(root).resolve()
        return _closure(root, tmp_path, external) if root in {left.resolve(), right.resolve()} else pytest.fail(str(root))

    monkeypatch.setattr(access_module, "resolve_dataset_closure", resolve)
    main_write = _scope("main", DatasetMode.WRITE)
    antenna_read = _scope("antenna", DatasetMode.READ, table=DatasetTable.ANTENNA)
    order = ResolvedAccessPlanner(tmp_path)
    assert not order.order_after("left", main_write, {"ms": alias}).dependencies
    assert order.order_after("subtable", antenna_read, {"ms": left / "ANTENNA"}).dependencies == {"left"}
    assert order.order_after("shared", antenna_read, {"ms": right}).dependencies == {"left"}


def test_unknown_columns_fall_back_and_unknown_path_needs_reservation(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    scope = _scope("reader", DatasetMode.READ)
    resolved = resolve_scope_dataset_accesses(scope, {"ms": ms}, workspace=tmp_path)
    assert resolved[0].whole_dataset
    assert not resolved[0].columns_known
    assert resolved[0].fallback is DatasetFallback.UNKNOWN_COLUMNS
    assert type(resolved[0]).model_validate_json(resolved[0].model_dump_json()) == resolved[0]

    undeclared = Scope(name="implicit", inputs_model=MSIn, outputs_model=Empty)
    implicit = resolve_scope_dataset_accesses(undeclared, {"ms": ms}, workspace=tmp_path)
    assert implicit[0].fallback is DatasetFallback.UNDECLARED
    assert implicit[0].reason.startswith("undeclared access; read whole dataset")

    with pytest.raises(DatasetAccessError, match="no reservation envelope"):
        resolve_scope_dataset_accesses(scope, {}, workspace=tmp_path)
    reserved = Scope(
        name="dynamic",
        inputs_model=MSIn,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="write", reservation=tmp_path / "reserved")],
    )
    result = resolve_scope_dataset_accesses(reserved, {}, workspace=tmp_path)
    assert not result[0].path_known
    assert result[0].fallback is DatasetFallback.UNKNOWN_PATH
    assert result[0].resources == ((tmp_path / "reserved").resolve(),)


def test_dataset_scatter_and_nested_recipe_are_bounded_refusals(tmp_path):
    class ManyMS(BaseModel):
        ms: list[MeasurementSetV2]

    scattered = Recipe(
        name="scatter",
        inputs_model=ManyMS,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="read",
                step=_scope("read", DatasetMode.READ),
                wiring={"ms": InputRef(field="ms")},
                scatter=ScatterSpec(fields=["ms"]),
            )
        ],
    )
    with pytest.raises(DatasetAccessError, match="scatters a dataset contract"):
        plan_recipe_accesses(scattered, {"ms": [tmp_path / "one.ms"]}, workspace=tmp_path)

    inner = Recipe(
        name="inner",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=_scope("read", DatasetMode.READ), wiring={"ms": InputRef(field="ms")})],
    )
    outer = Recipe(
        name="outer",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[StepRef(name="inner", step=inner, wiring={"ms": InputRef(field="ms")})],
    )
    with pytest.raises(DatasetAccessError, match="nested recipe"):
        plan_recipe_accesses(outer, {"ms": tmp_path / "one.ms"}, workspace=tmp_path)


def test_recipe_plan_adds_no_lineage_and_renders_reason(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    recipe = Recipe(
        name="hazards",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[
            StepRef(name="read", step=_scope("read", DatasetMode.READ), wiring={"ms": InputRef(field="ms")}),
            StepRef(
                name="write",
                step=_scope("write", DatasetMode.WRITE, columns=DatasetColumns(write=("FLAG",))),
                wiring={"ms": InputRef(field="ms")},
            ),
        ],
    )
    plan = plan_recipe_accesses(recipe, {"ms": ms}, workspace=tmp_path)
    assert plan.graph.deps == [set(), {0}]
    assert plan.reasons[("write", "read")] == ("write-after-read: observation.ms, MAIN.FLAG",)
    assert recipe.steps[1].wiring["ms"] == InputRef(field="ms")
    rendered = render_dag(graph_nodes(recipe, {"ms": ms}, workspace=tmp_path))
    assert "write-after-read: observation.ms, MAIN.FLAG" in rendered

    offload = MutationOrder(tmp_path)
    assert not offload.order_after("read", recipe.steps[0].step, {"ms": ms})
    assert offload.order_after("write", recipe.steps[1].step, {"ms": ms}) == {"read"}
    assert offload.last_decision.reasons["read"] == plan.reasons[("write", "read")]


def test_slurm_compilation_uses_same_plan_but_submission_stays_refused(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    reader = Cab(
        name="reader",
        command="read",
        inputs_model=MSIn,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("FLAG",)))],
    )
    writer = Cab(
        name="writer",
        command="write",
        inputs_model=MSIn,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("FLAG",)))],
    )
    recipe = Recipe(
        name="hazards",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[
            StepRef(name="read", step=reader, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="write", step=writer, wiring={"ms": InputRef(field="ms")}),
        ],
    )

    workflow = compile_slurm(recipe, {"ms": ms}, workdir=str(tmp_path), container_runtime=None)
    assert workflow.jobs[1].depends_on == ["read"]
    assert workflow.jobs[1].access_reasons == ["write-after-read: observation.ms, MAIN.FLAG"]
    assert workflow.execution_blocked_reason is not None
    with pytest.raises(DatasetLifecycleUnavailableError, match="planning-only"):
        submit_slurm(workflow, workdir=str(tmp_path))
    assert not workflow.log_dir.exists()


def test_real_casacore_access_resolution_when_available(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    ms_path = tmp_path / "real.ms"
    ms = tables.default_ms(str(ms_path))
    ms.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
    ms.close()

    resolved = resolve_scope_dataset_accesses(
        _scope("read", DatasetMode.READ, columns=DatasetColumns(read=("DATA",))),
        {"ms": ms_path},
        workspace=tmp_path,
    )
    assert resolved[0].closure_status is ClosureStatus.VALID
    assert resolved[0].root == ms_path.resolve()
    assert any(resource.name == "ANTENNA" for resource in resolved[0].resources)
