import sys
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator

import shinobi.dataset_access as access_module
from shinobi import (
    DatasetAccess,
    DatasetColumns,
    DatasetFallback,
    DatasetMode,
    DatasetSelection,
    DatasetTable,
    MeasurementSetV2,
    ResolvedDatasetAccess,
)
from shinobi.config import AppConfig
from shinobi.dataset_access import DatasetAccessError, ResolvedAccessPlanner, plan_recipe_accesses, resolve_scope_dataset_accesses
from shinobi.dataset_closure import ClosureCapabilities, ClosureRequirement, ClosureResource, ClosureStatus, DatasetClosure
from shinobi.dag import graph_nodes, render_dag
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.graph import RecipeNotOffloadableError, check_offloadable
from shinobi.offload.bundle import freeze_recipe
from shinobi.offload.slurm import MutationOrder, compile_slurm, prepare_worker_slurm, submit_slurm
from shinobi.offload.worker import ExecutionPlan
from shinobi.ownership import WorkspaceOwnershipError, acquire_workspace, scope_path_accesses, scope_requires_ownership
from shinobi.steps.pyfunc import pystep
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, ParamMeta, Recipe, ScatterSpec, Scope, StepRef


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
    order = ResolvedAccessPlanner(tmp_path)
    assert not order.order_after("read-1", reader, {"ms": ms}).dependencies
    assert not order.order_after("read-2", reader, {"ms": ms}).dependencies
    write = order.order_after("write", writer, {"ms": ms})
    assert write.dependencies == {"read-1", "read-2"}
    assert write.reasons["read-1"] == ("write-after-read: observation.ms, MAIN.FLAG",)
    read = order.order_after("read-3", reader, {"ms": ms})
    assert read.dependencies == {"write"}
    assert read.reasons["write"] == ("read-after-write: observation.ms, MAIN.FLAG",)
    writes = ResolvedAccessPlanner(tmp_path)
    assert not writes.order_after("write-1", writer, {"ms": ms}).dependencies
    second = writes.order_after("write-2", writer, {"ms": ms})
    assert second.dependencies == {"write-1"}
    assert second.reasons["write-1"] == ("write-after-write: observation.ms, MAIN.FLAG",)


@pytest.mark.parametrize("kind", ["same-output", "mutable", "write-path"])
def test_read_contract_cannot_hide_schema_declared_writes(kind, monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    kwargs = {}
    outputs = Empty
    if kind == "same-output":
        outputs = MSIn
    elif kind == "mutable":
        kwargs["input_mutability"] = {"ms": Mutability.MUTABLE}
    else:
        kwargs["field_meta"] = {"ms": ParamMeta(write_path=True)}
    scope = Scope(
        name=kind,
        inputs_model=MSIn,
        outputs_model=outputs,
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)))],
        **kwargs,
    )

    with pytest.raises(DatasetAccessError, match="schema also declares a filesystem write"):
        ResolvedAccessPlanner(tmp_path).order_after(kind, scope, {"ms": ms})


def test_dataset_contract_keeps_generic_parent_write_for_sibling_ordering(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    sibling = tmp_path / "other.txt"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))

    class DatasetAndParent(BaseModel):
        ms: MeasurementSetV2
        parent: Path

    class GenericPath(BaseModel):
        path: Path

    hybrid = Scope(
        name="hybrid",
        inputs_model=DatasetAndParent,
        outputs_model=Empty,
        field_meta={"parent": ParamMeta(write_path=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("FLAG",)))],
    )
    reader = Scope(name="reader", inputs_model=GenericPath, outputs_model=Empty)

    planner = ResolvedAccessPlanner(tmp_path)
    assert not planner.order_after("hybrid", hybrid, {"ms": ms, "parent": tmp_path}).dependencies
    assert planner.order_after("sibling", reader, {"path": sibling}).dependencies == {"hybrid"}


def test_read_contract_allows_generic_write_to_dataset_parent(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))

    class DatasetAndParent(BaseModel):
        ms: MeasurementSetV2
        parent: Path

    scope = Scope(
        name="read-with-parent-output",
        inputs_model=DatasetAndParent,
        outputs_model=Empty,
        field_meta={"parent": ParamMeta(write_path=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)))],
    )

    assert not ResolvedAccessPlanner(tmp_path).order_after("read", scope, {"ms": ms, "parent": tmp_path}).dependencies


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


def test_unset_optional_dataset_field_declares_no_access(tmp_path):
    class OptionalMS(BaseModel):
        ms: MeasurementSetV2 | None = None

    scope = Scope(
        name="optional",
        inputs_model=OptionalMS,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )

    assert resolve_scope_dataset_accesses(scope, {"ms": None}, workspace=tmp_path) == ()
    assert resolve_scope_dataset_accesses(scope, {}, workspace=tmp_path) == ()


def test_missing_required_nullable_output_still_needs_a_reservation(tmp_path):
    class RequiredNullableMS(BaseModel):
        ms: MeasurementSetV2 | None

    def scope_with(reservation: Path | None) -> Scope:
        return Scope(
            name="required-nullable-output",
            inputs_model=Empty,
            outputs_model=RequiredNullableMS,
            dataset_accesses=[DatasetAccess(field="ms", mode="create", reservation=reservation)],
        )

    with pytest.raises(DatasetAccessError, match="unknown path and no reservation"):
        resolve_scope_dataset_accesses(scope_with(None), {}, workspace=tmp_path)

    reservation = tmp_path / "reserved.ms"
    resolved = resolve_scope_dataset_accesses(scope_with(reservation), {}, workspace=tmp_path)
    assert not resolved[0].path_known
    assert resolved[0].resources == (reservation.resolve(),)
    assert resolve_scope_dataset_accesses(scope_with(None), {"ms": None}, workspace=tmp_path) == ()


def test_unresolved_implicit_optional_output_is_not_treated_as_absent(tmp_path):
    class Stem(BaseModel):
        stem: str

    class OptionalMS(BaseModel):
        ms: MeasurementSetV2 | None = None

    scope = Scope(
        name="implicit-output",
        inputs_model=Stem,
        outputs_model=OptionalMS,
        field_meta={"ms": ParamMeta(implicit="{stem}.ms")},
        dataset_accesses=[DatasetAccess(field="ms", mode="create")],
    )

    with pytest.raises(DatasetAccessError, match="unknown path and no reservation"):
        resolve_scope_dataset_accesses(scope, {}, workspace=tmp_path)


def test_resolved_access_rejects_cross_field_inconsistency(tmp_path):
    root = (tmp_path / "observation.ms").resolve()
    declaration = DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)))
    valid = {
        "field": "ms",
        "declaration": declaration,
        "requested_path": root,
        "root": root,
        "resources": (root,),
        "mode": "read",
        "whole_dataset": False,
        "path_known": True,
        "columns_known": True,
        "closure_status": "valid",
        "reason": "read observation.ms, MAIN.DATA",
    }
    assert ResolvedDatasetAccess.model_validate(valid).resources == (root,)
    for update, message in (
        ({"mode": "write"}, "field/mode"),
        ({"resources": ()}, "at least one resource"),
        ({"resources": ((tmp_path / "other").resolve(),)}, "root must be one"),
        ({"path_known": False}, "unknown dataset path"),
        ({"columns_known": False}, "columns_known"),
        ({"whole_dataset": True}, "fallback reason"),
    ):
        with pytest.raises(ValidationError, match=message):
            ResolvedDatasetAccess.model_validate({**valid, **update})


def test_dataset_access_fields_must_be_direct_paths():
    class Nested(BaseModel):
        value: Path

    class Invalid(BaseModel):
        integer: int
        nested: Nested
        paths: list[Path]
        mixed: Path | int
        ms: Path

    for field in ("integer", "nested", "paths", "mixed"):
        with pytest.raises(ValidationError, match="direct Path or MS-compatible"):
            Scope(name="invalid", inputs_model=Invalid, outputs_model=Empty, dataset_accesses=[DatasetAccess(field=field, mode="read")])
    with pytest.raises(ValidationError, match="root_field.*direct Path"):
        Scope(
            name="invalid-root",
            inputs_model=Invalid,
            outputs_model=Empty,
            dataset_accesses=[DatasetAccess(field="ms", root_field="integer", table="ANTENNA", mode="read")],
        )
    assert Scope(name="valid", inputs_model=MSIn, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    with pytest.raises(DatasetAccessError, match="path-compatible"):
        resolve_scope_dataset_accesses(_scope("bad-value", DatasetMode.READ), {"ms": 7}, workspace=Path.cwd())


def test_create_refuses_an_existing_target(tmp_path):
    existing = tmp_path / "existing.ms"
    existing.mkdir()
    with pytest.raises(DatasetAccessError, match="already exists.*--overwrite"):
        resolve_scope_dataset_accesses(_scope("create", DatasetMode.CREATE), {"ms": existing}, workspace=tmp_path)


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


def test_reusable_validated_step_snapshots_are_authoritative(tmp_path):
    shared = tmp_path / "normalized.ms"

    class RootPaths(BaseModel):
        left: Path
        right: Path

    class NormalizedPath(BaseModel):
        target: ClassVar[Path] = shared
        path: Path

        @field_validator("path")
        @classmethod
        def normalize(cls, _value):
            return cls.target

    writer = Scope(name="writer", inputs_model=NormalizedPath, outputs_model=NormalizedPath)
    recipe = Recipe(
        name="normalized",
        inputs_model=RootPaths,
        outputs_model=Empty,
        steps=[
            StepRef(name="left", step=writer, wiring={"path": InputRef(field="left")}),
            StepRef(name="right", step=writer, wiring={"path": InputRef(field="right")}),
        ],
    )
    raw = {"left": tmp_path / "raw-left.ms", "right": tmp_path / "raw-right.ms"}
    validated = RootPaths(**raw)
    _accesses, snapshots = scope_path_accesses(recipe, validated, workspace=tmp_path)

    local = plan_recipe_accesses(recipe, raw, workspace=tmp_path, validated_inputs=validated, validated_steps=snapshots)
    dryrun = plan_recipe_accesses(recipe, raw, workspace=tmp_path)
    assert local.graph.deps == dryrun.graph.deps == [set(), {0}]
    assert local.reasons == dryrun.reasons


def test_prevalidated_recipe_inputs_are_not_validated_twice(tmp_path):
    calls = 0

    def target_factory():
        nonlocal calls
        calls += 1
        return tmp_path / "generated.ms"

    class GeneratedRoot(BaseModel):
        target: Path = Field(default_factory=target_factory)

    class OnePath(BaseModel):
        target: Path

    reader = Scope(name="reader", inputs_model=OnePath, outputs_model=Empty)
    recipe = Recipe(
        name="single-validation",
        inputs_model=GeneratedRoot,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=reader, wiring={"target": InputRef(field="target")})],
    )
    validated = GeneratedRoot()
    _accesses, snapshots = scope_path_accesses(recipe, validated, workspace=tmp_path)
    plan_recipe_accesses(recipe, validated, workspace=tmp_path, validated_inputs=validated, validated_steps=snapshots)
    assert calls == 1


def test_unresolved_output_ref_does_not_fall_back_to_consumer_default(tmp_path):
    class ProducedPath(BaseModel):
        ms: Path

    class DefaultMS(BaseModel):
        ms: MeasurementSetV2 | None = None

    producer = Scope(name="producer", inputs_model=Empty, outputs_model=ProducedPath)

    def recipe_with(reservation: Path | None):
        consumer = Scope(
            name="consumer",
            inputs_model=DefaultMS,
            outputs_model=Empty,
            dataset_accesses=[DatasetAccess(field="ms", mode="read", reservation=reservation)],
        )
        return Recipe(
            name="dynamic",
            inputs_model=Empty,
            outputs_model=Empty,
            steps=[
                StepRef(name="produce", step=producer),
                StepRef(name="consume", step=consumer, wiring={"ms": OutputRef(step="produce", field="ms")}),
            ],
        )

    reservation = tmp_path / "reserved.ms"
    plan = plan_recipe_accesses(recipe_with(reservation), {}, workspace=tmp_path)
    access = plan.accesses["consume"][0]
    assert not access.path_known
    assert access.resources == (reservation.resolve(),)
    assert access.requested_path is None
    with pytest.raises(DatasetAccessError, match="unknown path and no reservation"):
        plan_recipe_accesses(recipe_with(None), {}, workspace=tmp_path)


def _create_then_read_recipe() -> Recipe:
    class Target(BaseModel):
        target: Path

    class CreatedMS(BaseModel):
        ms: MeasurementSetV2

    creator = Cab(
        name="creator",
        command="create-ms",
        inputs_model=Target,
        outputs_model=CreatedMS,
        field_meta={"ms": ParamMeta(implicit="{target}")},
        dataset_accesses=[DatasetAccess(field="ms", mode="create", columns=DatasetColumns(create=("DATA",)), allow_schema_change=True)],
    )
    reader = Cab(
        name="reader",
        command="read-ms",
        inputs_model=MSIn,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)))],
    )
    return Recipe(
        name="create-read",
        inputs_model=Target,
        outputs_model=Empty,
        steps=[
            StepRef(name="create", step=creator, wiring={"target": InputRef(field="target")}),
            StepRef(name="read", step=reader, wiring={"ms": OutputRef(step="create", field="ms")}),
        ],
    )


def test_planned_create_identity_flows_to_reader_and_all_planners(monkeypatch, tmp_path):
    target = tmp_path / "future.ms"
    recipe = _create_then_read_recipe()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: pytest.fail("planned dataset was inspected before creation"))

    plan = plan_recipe_accesses(recipe, {"target": target}, workspace=tmp_path)
    read = plan.accesses["read"][0]
    assert read.root == target.resolve()
    assert read.resources == (target.resolve(),)
    assert read.closure_status is None
    assert plan.reasons[("read", "create")] == ("read-after-write: future.ms, MAIN.DATA",)
    assert "read-after-write: future.ms, MAIN.DATA" in render_dag(graph_nodes(recipe, {"target": target}, workspace=tmp_path))

    legacy = compile_slurm(recipe, {"target": target}, workdir=str(tmp_path), container_runtime=None)
    assert legacy.jobs[1].depends_on == ["create"]
    assert legacy.jobs[1].access_reasons == ["read-after-write: future.ms, MAIN.DATA"]


def test_worker_dataset_planning_becomes_an_executable_lifecycle_plan(monkeypatch, tmp_path):
    target = tmp_path / "future.ms"
    recipe = _create_then_read_recipe()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: pytest.fail("planned dataset was inspected before creation"))
    bundle = freeze_recipe(recipe, {"target": target}, config=AppConfig(), workspace=tmp_path)
    assert bundle.execution_blocked_reason is not None
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    assert workflow.execution_blocked_reason is None
    assert workflow.jobs[1].depends_on == ["create"]
    assert workflow.jobs[1].access_reasons == ["read-after-write: future.ms, MAIN.DATA"]
    execution = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    assert execution.execution_blocked_reason is None
    assert execution.schema_version == 2
    assert execution.dataset_lifecycle is not None
    assert execution.dataset_lifecycle.mutation
    assert [step.step_path for step in execution.dataset_lifecycle.steps] == ["create", "read"]


def test_worker_preserves_wired_optional_none_as_a_known_absence(monkeypatch, tmp_path):
    class OptionalMS(BaseModel):
        ms: MeasurementSetV2 | None = None

    producer = Cab(name="producer", command="produce", inputs_model=Empty, outputs_model=OptionalMS)
    consumer = Cab(
        name="consumer",
        command="consume",
        inputs_model=OptionalMS,
        outputs_model=Empty,
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )
    recipe = Recipe(
        name="optional-none",
        inputs_model=Empty,
        outputs_model=Empty,
        steps=[
            StepRef(name="produce", step=producer),
            StepRef(name="consume", step=consumer, wiring={"ms": OutputRef(step="produce", field="ms")}),
        ],
    )
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: pytest.fail("absent optional dataset was inspected"))
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path)

    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))

    assert [job.name for job in workflow.jobs] == ["produce", "consume"]
    assert workflow.jobs[1].depends_on == ["produce"]


def test_worker_dataset_refusal_removes_staged_submission(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    contradictory = Cab(
        name="contradictory",
        command="read-ms",
        inputs_model=MSIn,
        outputs_model=MSIn,
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)))],
    )
    recipe = Recipe(
        name="contradictory",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=contradictory, wiring={"ms": InputRef(field="ms")})],
    )
    bundle = freeze_recipe(recipe, {"ms": ms}, config=AppConfig(), workspace=tmp_path)
    submission_root = tmp_path / "runs"

    with pytest.raises(DatasetAccessError, match="schema also declares a filesystem write"):
        prepare_worker_slurm(bundle, submission_root=submission_root, worker_python=Path(sys.executable))

    assert submission_root.is_dir()
    assert list(submission_root.iterdir()) == []


def test_read_only_dataset_plan_registers_claim_that_excludes_writer(monkeypatch, tmp_path):
    ms = tmp_path / "observation.ms"
    ms.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda root, **kwargs: _closure(Path(root), tmp_path))
    reader = _scope("reader", DatasetMode.READ, columns=DatasetColumns(read=("DATA",)))
    assert scope_requires_ownership(reader)
    accesses, _snapshots = scope_path_accesses(reader, {"ms": ms}, workspace=tmp_path)
    assert accesses and all(not writes for _path, writes in accesses)

    reader_root = tmp_path / "reader-workspace"
    writer_root = tmp_path / "writer-workspace"
    reader_root.mkdir()
    writer_root.mkdir()
    registry = tmp_path / "registry.json"
    lease = acquire_workspace(reader_root, "reader", kind="local", accesses=accesses, registry=registry)
    try:
        with pytest.raises(WorkspaceOwnershipError, match="conflicts"):
            acquire_workspace(writer_root, "writer", kind="slurm", accesses=[(ms, True)], registry=registry)
    finally:
        lease.release()


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
