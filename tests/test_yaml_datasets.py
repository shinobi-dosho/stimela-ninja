"""Strict dataset metadata follows the same contract in YAML and Python."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml
from pydantic import BaseModel, ValidationError

import shinobi
from shinobi import Cab, CasaTab, DatasetAccess, DatasetMode, MSv2
from shinobi.dataset_access import DatasetFallback, resolve_scope_dataset_accesses
from shinobi.datasets import CASA_TABLE_V1, MSV2_STRUCTURAL_V1, dataset_declarations
from shinobi.exceptions import CabLoadError, DatasetLifecycleUnavailableError
from shinobi.loaders._modelgen import build_model, dtype_to_type, is_file_dtype, narrow_choices
from shinobi.loaders.worker_schema import load_worker_schema
from shinobi.loaders.yaml_cab import loads
from shinobi.offload.bundle import ScopeSpec
from shinobi.policies import build_argv
from shinobi.steps import ParamMeta, ParamPattern, ParamSegment
from shinobi.steps.schema import Mutability, path_fields, readonly_path_fields


def load_cab(**spec):
    return loads(yaml.safe_dump({"cabs": {"tool": {"command": "true", **spec}}}))["tool"]


def test_concise_names_are_identical_public_compatibility_aliases():
    import shinobi.datasets as datasets
    import shinobi.steps as steps

    for module in (shinobi, datasets, steps):
        assert module.MSv2 is module.MeasurementSetV2 is MSv2
        assert module.CasaTab is module.CasaTable is CasaTab
        assert not hasattr(module, "MSv4")
    model = build_model(
        "Strict",
        {
            "ms": ("mSv2", True, None),
            "long_ms": ("MeasurementSetV2", False, None),
            "tab": ("cAsAtAb", False, None),
            "long_tab": ("CasaTable", False, None),
        },
    )
    assert dataset_declarations(model) == {
        "ms": MSV2_STRUCTURAL_V1,
        "long_ms": MSV2_STRUCTURAL_V1,
        "tab": CASA_TABLE_V1,
        "long_tab": CASA_TABLE_V1,
    }
    schema = model.model_json_schema()
    assert schema["properties"]["ms"]["x-shinobi-dataset"]["profile"] == "msv2-structural/v1"
    assert model(ms="future.ms").model_dump(mode="json") == {"ms": "future.ms", "long_ms": None, "tab": None, "long_tab": None}
    loaded = load_cab(inputs={"ms": {"dtype": "MeasurementSetV2"}, "tab": {"dtype": "CasaTable"}})
    assert dataset_declarations(loaded.inputs_model) == {"ms": MSV2_STRUCTURAL_V1, "tab": CASA_TABLE_V1}
    assert path_fields(loaded.inputs_model) == {"ms", "tab"}


@pytest.mark.parametrize("dtype", ["MSv4", "list:MSv4", "List[MSv4]", "Tuple[int, MSv4]", "Union[MS, List[MSv4]]"])
def test_reserved_dtype_refused_recursively(dtype):
    with pytest.raises(ValueError, match="MSv4.*reserved.*unsupported"):
        dtype_to_type(dtype)
    with pytest.raises(CabLoadError, match="cab 'tool'.*MSv4"):
        load_cab(inputs={"ms": {"dtype": dtype}})


def test_registry_keeps_atomic_file_detection_and_config_composites(tmp_path):
    assert dtype_to_type("MS") is Path
    assert dtype_to_type("unknown") is str
    assert dtype_to_type("MSv2") is MSv2
    assert dtype_to_type("CasaTab") is CasaTab
    assert dtype_to_type("List[MSv2]") == list[MSv2]
    for dtype in ("MS", " mSv2 ", "MeasurementSetV2", "CasaTab", "CasaTable"):
        assert is_file_dtype(dtype)
    assert not is_file_dtype("List[MSv2]")
    source = tmp_path / "config.yaml"
    source.write_text("name: config\ninputs: {group: {members: {dtype: 'List[MSv2]'}}}")
    config = load_worker_schema(source)
    assert dataset_declarations(config.inputs_model) == {"group.members[]": MSV2_STRUCTURAL_V1}


def test_yaml_matches_python_accesses_and_argv_with_sanitized_names():
    raw_access = {
        "field": "data_ms",
        "mode": "write",
        "table": "MAIN",
        "columns": {"read": ["DATA"], "write": ["FLAG"], "create": ["MODEL_DATA"], "remove": ["OLD"]},
        "selection": {"row_ranges": [[0, 10]], "field_ids": [0]},
        "allow_schema_change": True,
        "allow_row_count_change": True,
        "allow_keyword_change": True,
        "reservation": "products",
    }
    loaded = load_cab(inputs={"data.ms": {"dtype": "MSv2", "required": True, "mutable": True}}, dataset_accesses=[raw_access])

    class Inputs(BaseModel):
        data_ms: MSv2

    class Outputs(BaseModel):
        pass

    native = Cab(
        name="tool",
        command="true",
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"data_ms": ParamMeta(nom_de_guerre="data.ms")},
        input_mutability={"data_ms": Mutability.MUTABLE},
        dataset_accesses=[DatasetAccess.model_validate(raw_access)],
    )
    assert loaded.dataset_accesses == native.dataset_accesses
    assert loaded.input_mutability == native.input_mutability
    assert dataset_declarations(loaded.inputs_model) == dataset_declarations(native.inputs_model)
    assert build_argv(loaded, loaded.inputs_model(data_ms="obs.ms").model_dump()) == build_argv(native, Inputs(data_ms="obs.ms").model_dump())
    restored = ScopeSpec.model_validate_json(ScopeSpec.capture(loaded).model_dump_json()).restore()
    assert restored.dataset_accesses == loaded.dataset_accesses
    assert dataset_declarations(restored.inputs_model) == dataset_declarations(loaded.inputs_model)
    assert build_argv(restored, restored.inputs_model(data_ms="obs.ms").model_dump()) == build_argv(loaded, loaded.inputs_model(data_ms="obs.ms").model_dump())


@pytest.mark.parametrize(
    "raw, message",
    [
        ({}, "dataset_accesses must be a list"),
        ([None], r"dataset_accesses\[0\].*mapping"),
        ([{"field": "ms", "mode": "read"}, {"field": "ms", "mode": "bogus"}], r"dataset_accesses\[1\]"),
        ([{"field": "data.ms", "mode": "read"}], r"dataset_accesses\[0\].*unknown field 'data.ms'.*data_ms"),
        ([{"field": "data_ms", "mode": "read", "root_field": "data.ms"}], r"dataset_accesses\[0\].*unknown root_field"),
    ],
)
def test_dataset_errors_name_cab_and_index(raw, message):
    with pytest.raises(CabLoadError, match=message) as error:
        load_cab(inputs={"data.ms": {"dtype": "MSv2"}}, dataset_accesses=raw)
    assert "cab 'tool'" in str(error.value)


def test_unrelated_cab_validation_errors_keep_their_type():
    with pytest.raises(ValidationError):
        load_cab(sandbox="invalid")


def test_mutable_field_cannot_be_downgraded_to_explicit_read():
    with pytest.raises(CabLoadError, match=r"dataset_accesses\[0\].*mutable input field 'ms'.*only read"):
        load_cab(
            inputs={"ms": {"dtype": "MSv2", "mutable": True}},
            dataset_accesses=[{"field": "ms", "mode": "read"}],
        )

    loaded = load_cab(
        inputs={"ms": {"dtype": "MSv2", "mutable": True}},
        dataset_accesses=[{"field": "ms", "mode": "read"}, {"field": "ms", "mode": "write", "table": "ANTENNA"}],
    )
    assert [access.mode for access in loaded.dataset_accesses] == [DatasetMode.READ, DatasetMode.WRITE]


@pytest.mark.parametrize("dtype", ["List[MSv2]", "list:CasaTab", "Tuple[int, MSv2]", "Union[MSv2, List[MSv2]]"])
def test_executable_strict_containers_are_refused(dtype):
    with pytest.raises(CabLoadError, match="strict dataset field.*(nested|containers)"):
        load_cab(inputs={"ms": {"dtype": dtype}})


@pytest.mark.parametrize("dtype", ["Union[str, MSv2]", "Union[MSv2, File]", "Union[CasaTab, int]"])
def test_executable_mixed_strict_unions_are_refused(dtype):
    with pytest.raises(CabLoadError, match="cab 'tool'.*field 'ms'.*mixed unions"):
        load_cab(inputs={"ms": {"dtype": dtype}})
    model = build_model("Config", {"ms": (dtype, False, None)})
    assert "ms" in dataset_declarations(model)


def test_conflicting_strict_union_has_loader_context_only():
    with pytest.raises(CabLoadError, match="cab 'tool'.*field tool_Inputs.ms has conflicting dataset declarations"):
        load_cab(inputs={"ms": {"dtype": "Union[MSv2, CasaTab]"}})
    model = build_model("Python", {"ms": ("Union[MSv2, CasaTab]", False, None)})
    with pytest.raises(TypeError, match="field Python.ms has conflicting dataset declarations"):
        dataset_declarations(model)


@pytest.mark.parametrize("dtype", ["MSv2", "CasaTab"])
def test_direct_optional_strict_scalar_retains_path_and_none(dtype):
    loaded = load_cab(inputs={"ms": {"dtype": dtype, "default": None}})
    assert loaded.inputs_model().ms is None
    assert loaded.inputs_model(ms="obs.ms").ms == Path("obs.ms")
    assert "ms" in dataset_declarations(loaded.inputs_model)


@pytest.mark.parametrize("dtype", ["MSv2", "CasaTab", "List[MSv2]"])
def test_strict_choices_cannot_erase_declarations(dtype):
    with pytest.raises(ValueError, match="strict dataset choices"):
        narrow_choices(dtype_to_type(dtype), ["obs.ms"])
    with pytest.raises(CabLoadError, match="strict dataset choices"):
        load_cab(inputs={"ms": {"dtype": dtype, "choices": ["obs.ms"]}})


@pytest.mark.parametrize("key", ["input_patterns", "output_patterns"])
@pytest.mark.parametrize("dtype", ["MSv2", "List[CasaTab]", "Union[str, MSv2]", "MSv4"])
def test_dynamic_strict_patterns_refused_in_yaml_and_python(key, dtype):
    pattern = {"segments": [{"regex": ".+"}, {"attrs": {"ms": {"dtype": dtype}}}]}
    with pytest.raises(CabLoadError, match="strict dataset patterns|MSv4"):
        load_cab(**{key: [pattern]})
    built = ParamPattern(segments=[ParamSegment(regex=".+"), ParamSegment(attrs={"ms": ParamMeta(dtype=dtype)})])
    with pytest.raises(ValidationError, match="strict dataset patterns|MSv4"):
        Cab(name="tool", command="true", inputs_model=BaseModel, outputs_model=BaseModel, **{key: [built]})


def test_optional_strict_paths_and_legacy_shapes_remain_supported():
    loaded = load_cab(
        inputs={"ms": {"dtype": "MSv2", "writable": False}, "legacy": {"dtype": "List[MS]"}, "choice": {"dtype": "MS", "choices": ["a.ms"]}},
        input_patterns=[{"segments": [{"regex": ".+"}, {"attrs": {"ms": {"dtype": "MS"}}}]}],
        dataset_accesses=None,
    )
    assert loaded.inputs_model().ms is None
    assert dataset_declarations(loaded.inputs_model) == {"ms": MSV2_STRUCTURAL_V1}
    assert path_fields(loaded.inputs_model) == {"ms", "legacy"}
    assert readonly_path_fields(loaded.inputs_model, loaded.field_meta) == {"ms"}
    assert loaded.dataset_accesses == []
    assert loaded.inputs_model(choice="a.ms", legacy=["b.ms"]).legacy == [Path("b.ms")]


def test_access_lists_replace_in_use():
    loaded = loads("""
lib:
  base:
    command: true
    inputs: {ms: {dtype: MSv2}}
    dataset_accesses: [{field: ms, mode: read}]
cabs:
  writer:
    _use: lib.base
    command: 'true'
    dataset_accesses: [{field: ms, mode: write}]
""")["writer"]
    assert loaded.dataset_accesses == [DatasetAccess(field="ms", mode="write")]


def test_mutable_omitted_access_reserves_whole_dataset(tmp_path, monkeypatch):
    from tests.test_dataset_access import _closure
    import shinobi.dataset_access as access_module

    root = tmp_path / "obs.ms"
    root.mkdir()
    monkeypatch.setattr(access_module, "resolve_dataset_closure", lambda *args, **kwargs: _closure(root, tmp_path))
    loaded = load_cab(inputs={"ms": {"dtype": "MSv2", "required": True, "mutable": True}})
    (access,) = resolve_scope_dataset_accesses(loaded, {"ms": root}, workspace=tmp_path)
    assert access.mode is DatasetMode.WRITE
    assert access.whole_dataset
    assert access.fallback is DatasetFallback.UNDECLARED


@pytest.mark.parametrize("outputs", [{}, {"antenna": {"dtype": "MSv2"}}])
def test_casatab_subtable_root_field_cannot_bypass_dispatch(tmp_path, outputs):
    loaded = load_cab(
        inputs={"ms": {"dtype": "MSv2"}, "antenna": {"dtype": "CasaTab"}},
        outputs=outputs,
        dataset_accesses=[{"field": "antenna", "root_field": "ms", "mode": "read", "table": "ANTENNA"}],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="accepts only MeasurementSetV2.*antenna"):
        loaded(ms=tmp_path / "obs.ms", antenna=tmp_path / "ANTENNA", workspace=tmp_path)


def test_loaded_real_ms_reader_commits(tmp_path, monkeypatch):
    from tests._state_ms import make_state_ms

    pytest.importorskip("casacore.tables")
    root = make_state_ms(tmp_path / "obs.ms")
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.chdir(tmp_path)
    loaded = load_cab(
        command=f"{sys.executable} -c",
        inputs={
            "script": {
                "dtype": "str",
                "default": "import sys; from casacore.tables import table; t=table(sys.argv[1],readonly=True,ack=False); assert t.getcol('DATA').shape[0] == 6; t.close()",
                "policies": {"positional_head": True},
            },
            "ms": {"dtype": "MSv2", "required": True, "policies": {"positional": True}},
        },
        dataset_accesses=[{"field": "ms", "mode": "read", "columns": {"read": ["DATA"]}}],
    )
    result = loaded(ms=root, workspace=tmp_path)
    assert result.success
    records = list((tmp_path / ".shinobi" / "dataset-attempts").glob("*.json"))
    assert len(records) == 1
    assert shinobi.read_dataset_attempt(records[0]).outcome == "committed"
