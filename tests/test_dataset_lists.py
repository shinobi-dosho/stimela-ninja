"""Read-only list dataset contracts, through actual tables and lifecycle routes."""

from pathlib import Path
from typing import Sequence

import pytest
from pydantic import BaseModel, ValidationError

from shinobi import MSv2, Cab, DatasetAccess, DatasetColumns, DatasetSelection, pystep
from shinobi.dataset_access import DatasetAccessError, ResolvedDatasetAccess, resolve_scope_dataset_accesses
from shinobi.dataset_backends import plan_dataset_backend
from shinobi.datasets import DatasetDeclarationError, dataset_declarations, executable_dataset_fields
from shinobi.exceptions import DatasetLifecycleViolationError
from shinobi.loaders.yaml_cab import loads
from shinobi.steps.schema import Mutability, ParamMeta, Recipe, StepRef
from tests._dataset_fixtures import attempts, make_ms, set_scans


def run_kwargs(path):
    return {"cache": True, "cache_dir": str(path / "cache")}


class Inputs(BaseModel):
    ms: list[MSv2]


class OptionalInputs(BaseModel):
    ms: list[MSv2] | None = None


class Empty(BaseModel):
    pass


def cab(**kwargs):
    return Cab(name="list-reader", command="true", inputs_model=Inputs, outputs_model=Empty, **kwargs)


def test_python_metadata_and_indexed_records(tmp_path, monkeypatch):
    from tests.test_dataset_access import _closure
    import shinobi.dataset_access as module

    monkeypatch.setattr(module, "resolve_dataset_closure", lambda path, **kwargs: _closure(Path(path), tmp_path))
    roots = [tmp_path / f"{i}.ms" for i in range(2)]
    for root in roots:
        root.mkdir()
    assert list(dataset_declarations(Inputs)) == ["ms[]"]
    assert executable_dataset_fields(Inputs)["ms"].is_list
    assert Inputs(ms=roots).model_dump(mode="json") == {"ms": [str(root) for root in roots]}
    declaration = DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("DATA",)), selection=DatasetSelection(row_ranges=((0, 1),)))
    resolved = resolve_scope_dataset_accesses(cab(dataset_accesses=[declaration]), {"ms": roots}, workspace=tmp_path)
    assert [access.field for access in resolved] == ["ms[0]", "ms[1]"]
    assert [access.element_index for access in resolved] == [0, 1]
    assert all(access.declaration == declaration for access in resolved)
    assert all(ResolvedDatasetAccess.model_validate_json(access.model_dump_json()) == access for access in resolved)
    for changes in ({"element_index": -1}, {"element_index": True}, {"element_index": 0.0}, {"field": "ms"}):
        with pytest.raises(ValidationError):
            ResolvedDatasetAccess.model_validate({**resolved[0].model_dump(), **changes})
    for backend in ("docker", "podman", "apptainer", "singularity"):
        plan = plan_dataset_backend(backend, resolved, workspace=tmp_path, mutation=False)
        assert [(mount.source, mount.writable) for mount in plan.mounts] == [(root, False) for root in roots]


def test_optional_empty_unknown_and_alias(tmp_path, monkeypatch):
    from tests.test_dataset_access import _closure
    from shinobi.dataset_closure import ClosureStatus
    import shinobi.dataset_access as module

    def resolve(path, **kwargs):
        closure = _closure(Path(path), tmp_path)
        return closure.model_copy(update={"status": ClosureStatus.INVALID_ROOT}) if Path(path).name == "absent.ms" else closure

    monkeypatch.setattr(module, "resolve_dataset_closure", resolve)
    optional = Cab(name="optional", command="true", inputs_model=OptionalInputs, outputs_model=Empty)
    assert resolve_scope_dataset_accesses(optional, {}, workspace=tmp_path) == ()
    with pytest.raises(DatasetAccessError, match="non-empty"):
        resolve_scope_dataset_accesses(cab(), {"ms": []}, workspace=tmp_path)
    reserved = cab(dataset_accesses=[DatasetAccess(field="ms", mode="read", reservation=tmp_path)])
    unknown = resolve_scope_dataset_accesses(reserved, {}, workspace=tmp_path, unresolved_inputs={"ms"})
    assert len(unknown) == 1 and unknown[0].element_index is None and not unknown[0].path_known
    root = tmp_path / "obs.ms"
    root.mkdir()
    alias = tmp_path / "alias.ms"
    alias.symlink_to(root)
    with pytest.raises(DatasetAccessError, match=r"ms\[0\].*ms\[1\].*overlapping"):
        resolve_scope_dataset_accesses(cab(), {"ms": [root, alias]}, workspace=tmp_path)
    with pytest.raises(DatasetAccessError, match=r"ms\[1\].*invalid-root"):
        resolve_scope_dataset_accesses(cab(), {"ms": [root, tmp_path / "absent.ms"]}, workspace=tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_accesses": [DatasetAccess(field="ms", mode="write")]},
        {"dataset_accesses": [DatasetAccess(field="ms", mode="create")]},
        {"input_mutability": {"ms": Mutability.MUTABLE}},
        {"field_meta": {"ms": ParamMeta(write_path=True)}},
    ],
)
def test_list_writes_refused(kwargs):
    with pytest.raises((DatasetDeclarationError, ValidationError), match="read-only"):
        cab(**kwargs)


def test_list_outputs_refused():
    with pytest.raises((DatasetDeclarationError, ValidationError), match="read-only"):
        Cab(name="output", command="true", inputs_model=Empty, outputs_model=Inputs)


@pytest.mark.parametrize("dtype", ["List[MSv2]", "list:MSv2"])
def test_yaml_lists(dtype, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded = loads(f"""reader:
  command: 'true'
  inputs:
    ms:
      dtype: {dtype}
      required: true
      policies:
        positional: true
        repeat: list
  dataset_accesses:
    - field: ms
      mode: read
      columns:
        read: [DATA]
""")["reader"]
    roots = [make_ms(tmp_path / f"{i}.ms") for i in range(2)]
    from shinobi.policies import build_argv

    argv = build_argv(loaded, {"ms": roots})
    assert argv[-2:] == [str(path) for path in roots]
    assert loaded(ms=roots, **run_kwargs(tmp_path)).success
    record = attempts(tmp_path)[-1]
    assert record.outcome == "committed"
    assert len(record.pre_observations) == len(record.post_observations) == 2
    assert record.claim is not None
    assert {Path(access.path) for access in record.claim.accesses if not access.writes}.issuperset(roots)
    from shinobi.ownership import inspect_workspace

    assert inspect_workspace(tmp_path, str(record.claim.workflow_id)) is None
    assert [access.element_index for access in record.planned_accesses] == [0, 1]


def test_real_list_pystep_and_second_root_violation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots = [make_ms(tmp_path / f"{i}.ms") for i in range(2)]

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=("{column}",)))])
    def reader(ms: list[MSv2], column: str = "SCAN_NUMBER") -> None:
        assert all(path.is_dir() for path in ms)

    assert reader(ms=roots, **run_kwargs(tmp_path)).success
    record = attempts(tmp_path)[-1]
    assert record.schema_version == 2
    assert len(record.planned_accesses) == 2
    [leaf] = record.leaves
    assert leaf.mutations == ()
    assert [access.element_index for access in leaf.accesses] == [0, 1]
    assert [access.declaration.columns for access in leaf.accesses] == [DatasetColumns(read=("SCAN_NUMBER",))] * 2

    @pystep()
    def bad_reader(ms: list[MSv2]) -> None:
        set_scans(ms[1], 99)

    with pytest.raises(DatasetLifecycleViolationError):
        bad_reader(ms=roots, **run_kwargs(tmp_path))
    assert attempts(tmp_path)[-1].outcome == "failed"


@pytest.mark.parametrize("annotation", [Sequence[MSv2], tuple[MSv2, ...], list[MSv2 | None], list[list[MSv2]], list[MSv2] | MSv2])
def test_unsupported_shapes(annotation):
    from pydantic import create_model

    model = create_model("Unsupported", ms=(annotation, ...))
    with pytest.raises(DatasetDeclarationError):
        executable_dataset_fields(model)


def test_mixed_scalar_writer_list_reader_orders_each_root(tmp_path, monkeypatch):
    from shinobi.dataset_access import plan_recipe_accesses

    class Scalar(BaseModel):
        ms: MSv2

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))])
    def renumber(ms: MSv2, value: int) -> Scalar:
        set_scans(ms, value)
        return Scalar(ms=ms)

    monkeypatch.chdir(tmp_path)
    roots = [make_ms(tmp_path / f"{i}.ms") for i in range(2)]

    @pystep()
    def reader(ms: list[MSv2]) -> None:
        assert len(ms) == 2

    recipe = Recipe(
        name="mixed",
        inputs_model=Inputs,
        outputs_model=Empty,
        steps=[
            StepRef(name="left", step=renumber.step, func=renumber.func, params={"ms": roots[0], "value": 7}),
            StepRef(name="right", step=renumber.step, func=renumber.func, params={"ms": roots[1], "value": 8}),
            StepRef(name="read", step=reader.step, func=reader.func, params={"ms": roots}),
        ],
    )
    plan = plan_recipe_accesses(recipe, {"ms": roots}, workspace=tmp_path)
    assert plan.graph.deps[2] == {0, 1}
    assert recipe(ms=roots, **run_kwargs(tmp_path)).success
    record = attempts(tmp_path)[-1]
    read = next(leaf for leaf in record.leaves if leaf.step_path.endswith("read"))
    from shinobi.dataset_lifecycle import StrictLeaf

    coverage = StrictLeaf.__new__(StrictLeaf)
    coverage.scope = reader.step
    coverage.accesses = read.accesses
    assert all("wired; identified by producer lineage" in line for line in coverage.coverage({"ms": ["left-key", "right-key"]}))
    assert [access.element_index for access in read.accesses] == [0, 1]


@pytest.mark.parametrize("mutate_second", [False, True])
def test_worker_freezes_indexed_list_and_checks_second_root(tmp_path, monkeypatch, mutate_second):
    import sys
    from shinobi.config import AppConfig
    from shinobi.offload.bundle import RecipeBundle, freeze_recipe
    from shinobi.offload.worker import execute_step
    from shinobi.steps.schema import InputRef
    from tests.test_offload_dataset_lifecycle import _prepared, _record

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    roots = [make_ms(tmp_path / f"{i}.ms") for i in range(2)]

    class ToolInputs(BaseModel):
        script: str
        ms: list[MSv2]

    script = "from pathlib import Path;import sys;[Path(p,'table.dat').read_bytes() for p in sys.argv[1:]]"
    if mutate_second:
        script += ";from casacore.tables import table;t=table(sys.argv[2],readonly=False,ack=False);t.putcell('SCAN_NUMBER',0,99);t.close()"
    tool = Cab(
        name="tool",
        command=f"{sys.executable} -c",
        inputs_model=ToolInputs,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True, repeat_as_tokens=True)},
    )
    recipe = Recipe(
        name="list-worker",
        inputs_model=Inputs,
        outputs_model=Empty,
        steps=[StepRef(name="tool", step=tool, params={"script": script}, wiring={"ms": InputRef(field="ms")})],
        cache_dir=str(tmp_path / "cache"),
    )
    bundle = freeze_recipe(recipe, {"ms": roots}, config=AppConfig(), workspace=tmp_path)
    assert RecipeBundle.model_validate_json(bundle.model_dump_json()) == bundle
    workflow, plan, lease = _prepared(tmp_path, recipe, roots)
    try:
        frozen = plan.dataset_lifecycle
        assert frozen is not None
        attempt = plan.attempts[0]
        code = execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id)
        record = _record(workflow, plan)
        assert record.committed is (not mutate_second)
        assert (code == 0) is (not mutate_second)
        assert record.dataset_lifecycle is not None
        assert [access.element_index for access in record.dataset_lifecycle.planned_accesses] == [0, 1]
        assert len(record.dataset_lifecycle.pre_observations) == 2
    finally:
        lease.release()


@pytest.mark.parametrize("case", ["shared", "subtree", "scalar"])
def test_overlapping_list_closures_refuse_distinct_identities(tmp_path, monkeypatch, case):
    from tests.test_dataset_access import _closure
    import shinobi.dataset_access as module

    left = tmp_path / "left.ms"
    right = left / "nested.ms" if case == "subtree" else tmp_path / "right.ms"
    shared = tmp_path / "shared" / "ANTENNA"
    monkeypatch.setattr(module, "resolve_dataset_closure", lambda path, **kwargs: _closure(Path(path), tmp_path, *([shared] if case == "shared" else [])))
    if case == "scalar":

        class Mixed(BaseModel):
            ms: list[MSv2]
            other: MSv2

        tool = Cab(name="mixed", command="true", inputs_model=Mixed, outputs_model=Empty)
        values = {"ms": [left], "other": left}
    else:
        tool = cab()
        values = {"ms": [left, right]}
    with pytest.raises(DatasetAccessError, match="overlapping closure resources"):
        resolve_scope_dataset_accesses(tool, values, workspace=tmp_path)


def test_same_input_output_and_root_field_list_refusals():
    with pytest.raises(ValidationError, match="read-only"):
        Cab(name="inplace", command="true", inputs_model=Inputs, outputs_model=Inputs)

    class Mixed(BaseModel):
        ms: list[MSv2]
        other: MSv2

    with pytest.raises(ValidationError, match="root_field"):
        Cab(name="root", command="true", inputs_model=Mixed, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="other", mode="read", root_field="ms")])
    with pytest.raises(ValidationError, match="root_field"):
        Cab(name="root", command="true", inputs_model=Mixed, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", root_field="other")])


def test_list_hazards_cover_each_writer_but_keep_readers_independent(tmp_path, monkeypatch):
    from tests.test_dataset_access import _closure, _scope
    from shinobi.dataset_access import ResolvedAccessPlanner, DatasetMode
    import shinobi.dataset_access as module

    monkeypatch.setattr(module, "resolve_dataset_closure", lambda path, **kwargs: _closure(Path(path), tmp_path))
    left, right, other = [tmp_path / f"{name}.ms" for name in ("left", "right", "other")]
    planner = ResolvedAccessPlanner(tmp_path)
    for name, path in (("left", left), ("right", right), ("other", other)):
        assert not planner.order_after(name, _scope(name, DatasetMode.WRITE), {"ms": path}).dependencies
    first = planner.order_after("reader1", cab(), {"ms": [left, right]})
    second = planner.order_after("reader2", cab(), {"ms": [left, right]})
    assert first.dependencies == second.dependencies == {"left", "right"}
    assert all("read-after-write" in reason for reasons in first.reasons.values() for reason in reasons)


@pytest.mark.parametrize("annotation", [dict[str, MSv2], list[list[MSv2]]])
@pytest.mark.parametrize("explicit", [False, True])
def test_supported_list_access_preserves_unrelated_composite_metadata(tmp_path, monkeypatch, annotation, explicit):
    from pydantic import create_model
    from shinobi import read_dataset_attempt
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    import shinobi.steps.dispatch as dispatch

    monkeypatch.chdir(tmp_path)
    model = create_model("MixedMetadata", ms=(list[MSv2], ...), unrelated=(annotation, ...))
    accesses = [DatasetAccess(field="ms", mode="read")] if explicit else []
    tool = Cab(name="mixed-metadata", command="true", inputs_model=model, outputs_model=Empty, dataset_accesses=accesses)
    monkeypatch.setattr(dispatch, "_run_cab", lambda *args, **kwargs: pytest.fail("unsupported metadata dispatched a tool"))
    unrelated = {"nested": "other.ms"} if annotation == dict[str, MSv2] else [["other.ms"]]
    with pytest.raises(DatasetLifecycleUnavailableError, match="unrelated"):
        tool(ms=["obs.ms"], unrelated=unrelated)
    records = list((tmp_path / ".shinobi" / "dataset-attempts").glob("*.json"))
    assert len(records) == 1
    assert read_dataset_attempt(records[0]).outcome == "refused"


def test_optional_list_none_execution_without_concrete_access_is_refused(tmp_path, monkeypatch):
    from shinobi.exceptions import DatasetLifecycleUnavailableError

    monkeypatch.chdir(tmp_path)
    optional = Cab(name="optional-only", command="true", inputs_model=OptionalInputs, outputs_model=Empty)
    with pytest.raises(DatasetLifecycleUnavailableError, match="no concrete dataset access"):
        optional(**run_kwargs(tmp_path))
    assert attempts(tmp_path)[-1].outcome == "refused"


def test_optional_list_none_execution_with_scalar_access(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Mixed(BaseModel):
        ms: list[MSv2] | None = None
        scalar: MSv2

    root = make_ms(tmp_path / "concrete.ms")
    tool = Cab(name="optional-with-scalar", command="true", inputs_model=Mixed, outputs_model=Empty)
    assert tool(scalar=root, **run_kwargs(tmp_path)).success
    record = attempts(tmp_path)[-1]
    assert record.outcome == "committed"
    assert [access.field for access in record.planned_accesses] == ["scalar"]
