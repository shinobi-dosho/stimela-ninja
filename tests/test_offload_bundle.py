from __future__ import annotations

import importlib.util
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

import shinobi.offload.bundle as bundle_module
from shinobi import pystep
from shinobi.cache import ProvenanceKey
from shinobi.config import AppConfig
from shinobi.graph import RecipeNotOffloadableError, build_graph, check_offloadable
from shinobi.offload._codec import BundleError, ModelSpec, pack, unpack
from shinobi.offload.bundle import RecipeBundle, freeze_recipe
from shinobi.offload.code import CodeFile, capture_code
from shinobi.offload.records import AttemptRecord
from shinobi.policies import build_argv
from shinobi.results import StepResult
from shinobi.steps.schema import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


class Inputs(BaseModel):
    ms: Path = Path("relative/data.ms")
    n: int = Field(2, ge=1)
    mode: Literal["fast", "slow"] = "fast"


class Outputs(BaseModel):
    ms: Path


class Empty(BaseModel):
    pass


def recipe():
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"n": ParamMeta(nom_de_guerre="count")},
        harvest=["products/*.fits"],
        scratch=["scratch/*"],
    )
    return Recipe(
        name="pipeline",
        inputs_model=Inputs,
        outputs_model=Outputs,
        sandbox=True,
        steps=[StepRef(name="first", step=cab, wiring={"ms": InputRef(field="ms")}), StepRef(name="second", step=cab, wiring={"ms": OutputRef(step="first", field="ms")})],
        output_wiring={"ms": OutputRef(step="second", field="ms")},
    )


def freeze(value, tmp_path, **kwargs):
    return freeze_recipe(value, {}, config=AppConfig(), workspace=tmp_path, **kwargs)


def test_binary_bundle_round_trip_is_independent_and_pure(tmp_path):
    original = recipe()
    bundle = freeze(original, tmp_path)
    assert list(tmp_path.iterdir()) == []
    loaded = RecipeBundle.model_validate_json(bundle.model_dump_json())
    restored = loaded.declaration()
    assert build_graph(restored) == build_graph(original)
    assert isinstance(unpack(loaded.inputs)["ms"], Path)
    for before, after in zip(original.steps, restored.steps):
        assert build_argv(before.step, unpack(bundle.inputs)) == build_argv(after.step, unpack(loaded.inputs))
        assert after.step.harvest == before.step.harvest
        assert after.step.scratch == before.step.scratch
        assert after.step.field_meta == before.step.field_meta
    assert restored.sandbox is True
    with pytest.raises(ValidationError):
        restored.inputs_model(n=0)
    original.steps[0].step.harvest.append("new/*")
    original.steps.clear()
    assert loaded.digest == bundle.digest
    assert len(restored.steps) == 2


def test_bundle_preserves_explicit_runtime_cache_overrides(tmp_path):
    bundle = freeze_recipe(
        recipe(),
        {},
        config=AppConfig(),
        workspace=tmp_path,
        cache=True,
        cache_dir="shared/cache",
    )
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json())
    assert restored.cache_override is True
    assert restored.cache_dir_override == "shared/cache"


def test_unique_submission_directories_and_no_original_source_dependency(tmp_path):
    bundle = freeze(recipe(), tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        directories = list(pool.map(lambda _: bundle.stage(tmp_path), range(12)))
    assert len(set(directories)) == 12
    assert all(RecipeBundle.read(d / "bundle.json").digest == bundle.digest for d in directories)
    assert all((d / "submission.json").is_file() for d in directories)


def test_stage_removes_its_uuid_directory_when_record_publication_fails(monkeypatch, tmp_path):
    bundle = freeze(recipe(), tmp_path)
    root = tmp_path / "runs"
    real_write_new = bundle_module.write_new

    def fail_submission(path, model):
        if path.name == "submission.json":
            raise OSError("injected publication failure")
        return real_write_new(path, model)

    monkeypatch.setattr(bundle_module, "write_new", fail_submission)

    with pytest.raises(OSError, match="injected publication failure"):
        bundle.stage(root)

    assert root.is_dir()
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("worker_protocol", 2), ("unknown", True)])
def test_unknown_protocol_rejected(tmp_path, field, value):
    data = freeze(recipe(), tmp_path).model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        RecipeBundle.model_validate(data)


def test_custom_validators_and_factories_never_run_at_freeze(tmp_path):
    calls = []

    class Custom(BaseModel):
        ms: Path = Path("data.ms")
        value: int = 1

        @field_validator("value")
        @classmethod
        def custom(cls, value):
            calls.append(value)
            return value

    bad = recipe()
    bad.steps[0].step.inputs_model = Custom
    bad.steps[0].wiring = {}
    with pytest.raises(BundleError, match="custom validators"):
        freeze(bad, tmp_path)
    Factory = create_model("Factory", value=(list[int], Field(default_factory=lambda: calls.append(1))))
    with pytest.raises(BundleError, match="default factory"):
        ModelSpec.capture(Factory)
    assert calls == []


def test_nested_models_constraints_and_value_tags_round_trip():
    class Child(BaseModel):
        path: Path

    class Parent(BaseModel):
        child: Child
        pair: tuple[int, str] = (2, "two")
        items: list[Path] = Field(default_factory=list)
        optional: int | None = None

    restored = ModelSpec.model_validate_json(ModelSpec.capture(Parent).model_dump_json()).restore()
    value = restored(child={"path": "thing"})
    assert value.child.path == Path("thing")
    assert value.pair == (2, "two")
    value.items.append(Path("x"))
    assert restored(child={"path": "thing"}).items == []
    tagged = {"path": (Path("relative"), [1, True]), "scalar": ["dict", {}]}
    assert unpack(pack(tagged)) == tagged
    with pytest.raises(BundleError):
        pack(object())


def source_function(tmp_path, source=None):
    entry = tmp_path / "computation.py"
    entry.write_text(
        source
        or "from pathlib import Path\nfrom pydantic import BaseModel\nclass Result(BaseModel):\n    ms: Path\ndef compute(ms: Path = Path('data.ms')) -> Result:\n    import helper\n    return Result(ms=ms)\n"
    )
    (tmp_path / "helper.py").write_text("raise RuntimeError('must not import helper during freeze')\n")
    spec = importlib.util.spec_from_file_location("computation", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute


@pytest.mark.parametrize("backend", ["docker", "venv"])
def test_pystep_explicit_identity_source_and_environment_round_trip(tmp_path, backend):
    func = source_function(tmp_path)
    options = {"backend": backend}
    if backend == "docker":
        options["image"] = "tool:1"
    else:
        (tmp_path / "tool-env" / "bin").mkdir(parents=True)
        (tmp_path / "tool-env" / "bin" / "python").touch()
        options["venv"] = str(tmp_path / "tool-env")
    ref = pystep(**options)(func)
    assert inspect.unwrap(ref.func) is func
    value = Recipe(name="python", inputs_model=Empty, outputs_model=Empty, steps=[ref])
    check_offloadable(value, worker=True)
    with pytest.raises(RecipeNotOffloadableError):
        check_offloadable(value)
    bundle = freeze(value, tmp_path, code_roots=(tmp_path,))
    staged = bundle.stage(tmp_path / "runs")
    loaded = RecipeBundle.read(staged / "bundle.json")
    code = loaded.steps[0].code
    assert code.qualname == "compute"
    assert {f.path for f in code.files} == {"computation.py", "helper.py"}
    assert "pydantic" in code.environment_imports
    assert loaded.steps[0].tool_venv == options.get("venv")
    (tmp_path / "computation.py").unlink()
    (tmp_path / "helper.py").write_text("CHANGED = True\n")
    materialized = code.write(staged / "source")
    assert "def compute" in materialized.read_text()
    assert "must not import" in (materialized.parent / "helper.py").read_text()
    assert loaded.digest == bundle.digest


def test_helper_change_changes_code_identity(tmp_path):
    func = source_function(tmp_path)
    first = capture_code(func, roots=(tmp_path,))
    (tmp_path / "helper.py").write_text("CHANGED = True\n")
    changed = capture_code(func, roots=(tmp_path,))
    assert first.digest != changed.digest
    assert first.execution_digest != changed.execution_digest


def test_unrelated_callable_change_does_not_change_pystep_execution_identity(tmp_path):
    first_func = source_function(
        tmp_path,
        "def compute(value: int = 1):\n    return value + 1\n\ndef unrelated():\n    return 'first'\n",
    )
    first = capture_code(first_func, roots=(tmp_path,))
    second_func = source_function(
        tmp_path,
        "def compute(value: int = 1):\n    return value + 1\n\ndef unrelated():\n    return 'changed'\n",
    )
    second = capture_code(second_func, roots=(tmp_path,))
    assert first.digest != second.digest
    assert first.execution_digest == second.execution_digest


def test_unrelated_deferred_annotation_change_does_not_change_pystep_execution_identity(tmp_path):
    before = "from __future__ import annotations\ndef compute(value: int = 1):\n    return value + 1\n\ndef unrelated(value: int) -> str:\n    return 'same'\n"
    after = before.replace("unrelated(value: int)", "unrelated(value: bytes)")
    first = capture_code(source_function(tmp_path, before), roots=(tmp_path,))
    second = capture_code(source_function(tmp_path, after), roots=(tmp_path,))
    assert first.digest != second.digest
    assert first.execution_digest == second.execution_digest


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "X = 5\nTABLE = list(range(X))\ndef _table():\n    return TABLE\n",
            "X = 6\nTABLE = list(range(X))\ndef _table():\n    return TABLE\n",
            id="literal-feeding-unconditional-statement",
        ),
        pytest.param(
            "A, B = 1, 2\ndef _table():\n    return A\n",
            "A, B = 3, 2\ndef _table():\n    return A\n",
            id="tuple-unpacking",
        ),
        pytest.param(
            "CFG = {'k': 0}\nCFG['k'] = 1\ndef _table():\n    return CFG['k']\n",
            "CFG = {'k': 0}\nCFG['k'] = 2\ndef _table():\n    return CFG['k']\n",
            id="item-assignment",
        ),
        pytest.param(
            "class Settings:\n    level = 1\nSettings.level = 3\ndef _table():\n    return Settings.level\n",
            "class Settings:\n    level = 1\nSettings.level = 4\ndef _table():\n    return Settings.level\n",
            id="attribute-assignment",
        ),
        pytest.param(
            "X = 1\nY = [X]\nX = 2\ndef _table():\n    return Y\n",
            "X = 7\nY = [X]\nX = 2\ndef _table():\n    return Y\n",
            id="shadowed-earlier-binding",
        ),
        pytest.param(
            "STATE = {'value': 0}\ndef _set(value):\n    STATE['value'] = value\n    return int\n\nclass Unrelated:\n    _set(1)\ndef _table():\n    return STATE['value']\n",
            "STATE = {'value': 0}\ndef _set(value):\n    STATE['value'] = value\n    return int\n\nclass Unrelated:\n    _set(2)\ndef _table():\n    return STATE['value']\n",
            id="unreferenced-class-body",
        ),
        pytest.param(
            "STATE = {'value': 0}\ndef _set(value):\n    STATE['value'] = value\n    return int\ndef unrelated(value=_set(1)):\n    return value\ndef _table():\n    return STATE['value']\n",
            "STATE = {'value': 0}\ndef _set(value):\n    STATE['value'] = value\n    return int\ndef unrelated(value=_set(2)):\n    return value\ndef _table():\n    return STATE['value']\n",
            id="unreferenced-function-default",
        ),
        pytest.param(
            "STATE = {'value': 0}\ndef _decorate(value):\n    def apply(func):\n        STATE['value'] = value\n        return func\n    return apply\n@_decorate(1)\ndef unrelated():\n    pass\ndef _table():\n    return STATE['value']\n",
            "STATE = {'value': 0}\ndef _decorate(value):\n    def apply(func):\n        STATE['value'] = value\n        return func\n    return apply\n@_decorate(2)\ndef unrelated():\n    pass\ndef _table():\n    return STATE['value']\n",
            id="unreferenced-function-decorator",
        ),
    ],
)
def test_module_state_literal_change_changes_pystep_execution_identity(tmp_path, before, after):
    call = "def compute(value: int = 1):\n    return _table()\n"
    first = capture_code(source_function(tmp_path, before + call), roots=(tmp_path,))
    changed = capture_code(source_function(tmp_path, after + call), roots=(tmp_path,))
    assert first.execution_digest != changed.execution_digest


def test_tuple_unpacked_scalar_global_can_be_captured(tmp_path):
    func = source_function(tmp_path, "LEFT, RIGHT = 1, 2\ndef compute(value: int = 1):\n    return value + LEFT\n")
    bundle = capture_code(func, roots=(tmp_path,))
    assert bundle.qualname == "compute"


def test_unsupported_callable_and_environment_fail_before_staging(tmp_path):
    value = recipe()
    value.steps[0].func = lambda ctx: ctx.run()
    with pytest.raises(RecipeNotOffloadableError, match="orchestration"):
        freeze(value, tmp_path)
    func = source_function(tmp_path)
    with pytest.raises(BundleError, match="code_roots"):
        capture_code(func)
    with pytest.raises(BundleError, match="module-addressable"):
        capture_code(lambda: None, roots=(tmp_path,))
    with pytest.raises(BundleError, match="native fallback"):
        freeze(Recipe(name="python", inputs_model=Empty, outputs_model=Empty, steps=[pystep(backend="venv")(func)]), tmp_path, code_roots=(tmp_path,))
    with pytest.raises(ValidationError, match="never in-process"):
        freeze(Recipe(name="python", inputs_model=Empty, outputs_model=Empty, steps=[pystep(image="tool:1", backend="native")(func)]), tmp_path, code_roots=(tmp_path,))


@pytest.mark.parametrize("path", ["../escape.py", "/escape.py", "a/../escape.py", "a\\escape.py"])
def test_unsafe_code_paths_rejected(path):
    with pytest.raises(ValidationError):
        CodeFile(path=path, source="")


@pytest.mark.parametrize("state", ["succeeded", "cached", "skipped", "failed"])
def test_attempt_round_trip_and_identity(tmp_path, state):
    scope = recipe().steps[0].step
    result = StepResult(
        name="tool",
        returncode=1 if state == "failed" else 0,
        inputs=Inputs(),
        outputs=Outputs(ms=Path("data.ms")),
        stdout="captured",
        cached=state == "cached",
        skipped=state == "skipped",
        cache_key="own",
        output_keys={"ms": ProvenanceKey("upstream-key", "original_ms")},
    )
    identity = {"workflow_id": uuid4(), "attempt_id": uuid4(), "step_path": "selfcal.2.image", "bundle_digest": "bundle-key"}
    record = AttemptRecord.from_result(result, **identity)
    assert record.state == state
    path = record.write(tmp_path)
    loaded = AttemptRecord.read(path, **identity)
    restored = loaded.result(scope)
    assert restored.outputs.ms == result.outputs.ms
    assert restored.stdout == "captured"
    assert loaded.committed is (state != "failed")
    if state != "failed":
        assert restored.provenance_key("ms") == "upstream-key"
        assert restored.provenance_key("ms").producer_field == "original_ms"
    with pytest.raises(FileExistsError):
        record.write(tmp_path)
    with pytest.raises(BundleError, match="does not belong"):
        AttemptRecord.read(path, **{**identity, "attempt_id": uuid4()})


def test_missing_and_unfinished_records_are_not_success(tmp_path):
    identity = {"workflow_id": uuid4(), "attempt_id": uuid4(), "step_path": "step", "bundle_digest": "key"}
    for state in ("running", "unknown"):
        record = AttemptRecord(**identity, state=state)
        assert not record.committed
        with pytest.raises(BundleError, match="no final"):
            record.result(recipe().steps[0].step)
    with pytest.raises(FileNotFoundError):
        AttemptRecord.read(tmp_path / "missing.json", **identity)
    with pytest.raises(ValidationError):
        AttemptRecord(**identity, state="succeeded")


def test_declared_loop_topology_round_trip(tmp_path):
    from tests.test_steps_loops import make_recipe

    value = make_recipe("native", tmp_path, max_iter=3)
    bundle = freeze_recipe(value, {"ms": Path("data.ms")}, config=AppConfig(), workspace=tmp_path)
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json()).declaration()
    assert build_graph(restored) == build_graph(value)
    assert [ref.loop for ref in restored.steps] == [ref.loop for ref in value.steps]


def test_nested_model_input_round_trip(tmp_path):
    class Nested(BaseModel):
        settings: Inputs

    value = Recipe(name="nested", inputs_model=Nested, outputs_model=Empty)
    bundle = freeze_recipe(value, {"settings": {"ms": "nested.ms"}}, config=AppConfig(), workspace=tmp_path)
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json())
    inputs = restored.declaration().inputs_model(**unpack(restored.inputs))
    assert inputs.settings.ms == Path("nested.ms")


def test_dynamic_helper_explicit_inclusion_and_symlink_escape(tmp_path):
    func = source_function(tmp_path)
    (tmp_path / "dynamic.py").write_text("VALUE = 1\n")
    code = capture_code(func, roots=(tmp_path,), include=("dynamic",))
    assert "dynamic.py" in {f.path for f in code.files}
    with pytest.raises(BundleError, match="missing"):
        capture_code(func, roots=(tmp_path,), include=("missing",))
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("VALUE = 1\n")
    (tmp_path / "escaped.py").symlink_to(outside)
    with pytest.raises(BundleError, match="symlink"):
        capture_code(func, roots=(tmp_path,), include=("escaped",))


def test_mutable_and_changed_global_state_rejected(tmp_path):
    func = source_function(tmp_path, "STATE = []\ndef compute():\n    return STATE\n")
    with pytest.raises(BundleError, match="mutable or changed"):
        capture_code(func, roots=(tmp_path,))
    func = source_function(tmp_path, "STATE = 1\ndef compute():\n    return STATE\n")
    capture_code(func, roots=(tmp_path,))
    func.__globals__["STATE"] = 2
    with pytest.raises(BundleError, match="mutable or changed"):
        capture_code(func, roots=(tmp_path,))


def test_pystep_param_metadata_survives_model_codec():
    model = create_model("AnnotatedInput", ms=(Path, Field(Path("data.ms"), json_schema_extra={"param_meta": ParamMeta(write_path=True)})))
    restored = ModelSpec.capture(model).restore()
    assert restored.model_fields["ms"].json_schema_extra["param_meta"].write_path is True


def test_failed_or_mismatched_record_cannot_claim_commit(tmp_path):
    result = StepResult(name="tool", returncode=1, inputs=Inputs(), outputs=Outputs(ms=Path("data.ms")))
    record = AttemptRecord.from_result(result, workflow_id=uuid4(), attempt_id=uuid4(), step_path="step", bundle_digest="key")
    data = record.model_dump()
    data["state"] = "succeeded"
    with pytest.raises(ValidationError, match="disagrees"):
        AttemptRecord.model_validate(data)
    data["state"] = "failed"
    data["cache_key"] = "must-not-publish"
    with pytest.raises(ValidationError, match="failed attempt"):
        AttemptRecord.model_validate(data)
