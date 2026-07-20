from pathlib import Path
from typing import get_args

import pytest

from shinobi.exceptions import ConfigLoadError
from shinobi.loaders.worker_schema import load_worker_schema

FIXTURES = Path(__file__).parent / "fixtures" / "worker_schemas"


def test_flat_schema_loads_real_getdata_file():
    schema = load_worker_schema(FIXTURES / "getdata_schema.yaml")
    assert schema.name == "getdata"
    fields = schema.inputs_model.model_fields
    assert "dataid" in fields
    assert fields["dataid"].annotation == list[str]
    # nested group ("untar.enable"/"untar.tar_options") becomes a submodel field
    untar_model = fields["untar"].annotation
    assert untar_model.model_fields["enable"].default is True
    assert untar_model().tar_options == "-xvf"
    # "cabs" has no dtype at its own level but its children do -> still a group
    cabs_model = fields["cabs"].annotation
    assert "name" in cabs_model.model_fields


def test_nested_groups_round_trip_choices_and_defaults():
    schema = load_worker_schema(FIXTURES / "crosscal_schema.yaml")
    inputs = schema.inputs_model.model_fields
    assert "rewind_flags" in inputs
    rewind_flags_model = inputs["rewind_flags"].annotation
    mode_field = rewind_flags_model.model_fields["mode"]
    # annotation is Literal[...] | None (choices wrapped in Literal, optional since
    # not `required` in the source schema) -- unwrap the Union to get at the Literal
    literal_type = next(a for a in get_args(mode_field.annotation) if a is not type(None))
    assert set(get_args(literal_type)) >= {"reset_worker", "rewind_to_version"}
    assert mode_field.default == "reset_worker"
    # a group field defaults to an instance of its own sub-model
    default_group = rewind_flags_model()
    assert default_group.mode == "reset_worker"


def test_include_and_use_produce_merged_base_and_ms_base_fields():
    schema = load_worker_schema(FIXTURES / "obsconf_schema.yaml")
    inputs = schema.inputs_model.model_fields
    outputs = schema.outputs_model.model_fields
    # from libs.base.inputs / libs.ms_base.inputs (caracal_base.yaml)
    assert "prefix" in inputs
    assert "msdir" in inputs
    assert "ms" in inputs
    # obsconf's own inputs still present alongside the _use-merged ones
    assert "obsinfo" in inputs
    assert "refant" in inputs
    # from libs.base.outputs
    assert "output" in outputs


def test_writable_false_is_carried_onto_the_field_json_schema_extra():
    # `input` (Directory, writable: false) in caracal_base.yaml -> the container
    # backend mounts it read-only. writable: true / unmarked fields carry nothing.
    from shinobi.steps.schema import readonly_path_fields

    schema = load_worker_schema(FIXTURES / "obsconf_schema.yaml")
    inputs = schema.inputs_model.model_fields
    assert inputs["input"].json_schema_extra == {"writable": False}
    assert inputs["msdir"].json_schema_extra == {"writable": True}
    assert readonly_path_fields(schema.inputs_model) == {"input"}


def test_abbreviation_is_carried_onto_the_field_json_schema_extra(tmp_path):
    # `abbreviation` rides the same json_schema_extra channel as `writable`,
    # so clickutil.build_options can emit a short flag for a worker-config CLI.
    schema_file = tmp_path / "abbrev.yaml"
    schema_file.write_text("name: thing\ninputs:\n  refant:\n    dtype: str\n    abbreviation: ra\n  plain:\n    dtype: str\n")
    inputs = load_worker_schema(schema_file).inputs_model.model_fields
    assert inputs["refant"].json_schema_extra == {"abbreviation": "ra"}
    assert inputs["plain"].json_schema_extra is None


def test_use_missing_path_raises_config_load_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  _use: does.not.exist\n")
    with pytest.raises(ConfigLoadError):
        load_worker_schema(bad)


def test_missing_name_raises_config_load_error(tmp_path):
    noname = tmp_path / "noname.yaml"
    noname.write_text("inputs:\n  x:\n    dtype: str\n")
    with pytest.raises(ConfigLoadError, match="no top-level 'name'"):
        load_worker_schema(noname)


def test_plain_relative_include_merges_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("shared:\n  x:\n    dtype: int\n    default: 1\n")
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: base.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")
    schema = load_worker_schema(main)
    assert "x" in schema.inputs_model.model_fields
    assert schema.inputs_model.model_fields["x"].default == 1


def test_package_scoped_include_resolves_via_importlib(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "shared.yaml").write_text("shared:\n  y:\n    dtype: bool\n    default: true\n")

    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    main = schema_dir / "main.yaml"
    main.write_text("libs:\n  _include: (mypkg)shared.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    schema = load_worker_schema(main)
    assert "y" in schema.inputs_model.model_fields
    assert schema.inputs_model.model_fields["y"].default is True


def test_sibling_keys_win_over_use_merged_keys(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("shared:\n  x:\n    dtype: int\n    default: 1\n")
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: base.yaml\nname: thing\ninputs:\n  _use: libs.shared\n  x:\n    dtype: int\n    default: 99\n")
    schema = load_worker_schema(main)
    assert schema.inputs_model.model_fields["x"].default == 99


def test_non_mapping_param_spec_raises_config_load_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  x: not-a-mapping\n")
    with pytest.raises(ConfigLoadError, match="param/group mapping"):
        load_worker_schema(bad)


def test_scalar_choices_raises_instead_of_exploding_into_per_character_literal(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  x:\n    dtype: str\n    choices: abc\n")
    with pytest.raises(ConfigLoadError, match="'choices' must be a list"):
        load_worker_schema(bad)


def test_list_top_level_document_raises_config_load_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(ConfigLoadError, match="must be a mapping"):
        load_worker_schema(bad)


def test_list_inputs_section_raises_config_load_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n- a\n- b\n")
    with pytest.raises(ConfigLoadError, match="expected a mapping"):
        load_worker_schema(bad)


def test_non_mapping_include_target_raises_config_load_error(tmp_path):
    sub = tmp_path / "sub.yaml"
    sub.write_text("- a\n- b\n")
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: sub.yaml\nname: thing\ninputs: {}\n")
    with pytest.raises(ConfigLoadError, match="must be a mapping"):
        load_worker_schema(main)
