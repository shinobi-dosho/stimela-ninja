from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

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


def test_use_self_cycle_raises_config_load_error_naming_cycle(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nlibs:\n  loop:\n    _use: libs.loop\ninputs: {}\n")
    with pytest.raises(ConfigLoadError, match=r"_use cycle detected: libs\.loop -> libs\.loop"):
        load_worker_schema(bad)


def test_use_multi_node_cycle_raises_config_load_error_naming_cycle(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nlibs:\n  left:\n    _use: libs.right\n  right:\n    _use: libs.left\ninputs: {}\n")
    cycle = r"_use cycle detected: libs\.(?:left -> libs\.right -> libs\.left|right -> libs\.left -> libs\.right)"
    with pytest.raises(ConfigLoadError, match=cycle):
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


def test_include_self_cycle_raises_config_load_error_naming_cycle(tmp_path):
    main = tmp_path / "main.yaml"
    main.write_text("_include: main.yaml\nname: thing\ninputs: {}\n")
    with pytest.raises(ConfigLoadError, match=r"_include cycle detected: .*main\.yaml -> .*main\.yaml"):
        load_worker_schema(main)


def test_include_multi_file_cycle_raises_config_load_error_naming_cycle(tmp_path):
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text("_include: right.yaml\nname: thing\ninputs: {}\n")
    right.write_text("_include: left.yaml\n")
    cycle = r"_include cycle detected: .*left\.yaml -> .*right\.yaml -> .*left\.yaml"
    with pytest.raises(ConfigLoadError, match=cycle):
        load_worker_schema(left)


def _pkg_include_fixture(tmp_path):
    """A package dir with a schema fragment, plus a main schema that pulls
    it in with the package-scoped `(mypkg)shared.yaml` form. The `__init__.py`
    raises: nothing may import this package to resolve the include.
    """
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("raise AssertionError('worker_schema imported a package to resolve an _include')\n")
    (pkg_dir / "shared.yaml").write_text("shared:\n  y:\n    dtype: bool\n    default: true\n")

    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    main = schema_dir / "main.yaml"
    main.write_text("libs:\n  _include: (mypkg)shared.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")
    return pkg_dir, main


def test_package_scoped_include_resolves_via_package_roots(tmp_path):
    pkg_dir, main = _pkg_include_fixture(tmp_path)
    schema = load_worker_schema(main, package_roots={"mypkg": pkg_dir})
    assert "y" in schema.inputs_model.model_fields
    assert schema.inputs_model.model_fields["y"].default is True


def test_package_scoped_include_never_imports_the_package(tmp_path, monkeypatch):
    """The include names a real, importable package whose `__init__.py`
    would blow up if executed. Without a `package_roots` entry the load must
    fail with a clear error rather than importing it -- the loader has no
    business running code named by a config file (SECURITY.md).
    """
    _pkg_dir, main = _pkg_include_fixture(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ConfigLoadError, match="no filesystem root was supplied"):
        load_worker_schema(main)


def test_package_scoped_include_resolves_longest_prefix(tmp_path):
    """`(mypkg.sub)file` against a root registered for `mypkg` descends the
    remainder as a subdirectory, same as the cultcargo dialect.
    """
    pkg_dir = tmp_path / "mypkg"
    (pkg_dir / "sub").mkdir(parents=True)
    (pkg_dir / "sub" / "shared.yaml").write_text("shared:\n  z:\n    dtype: int\n    default: 7\n")
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: (mypkg.sub)shared.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")

    schema = load_worker_schema(main, package_roots={"mypkg": pkg_dir})
    assert schema.inputs_model.model_fields["z"].default == 7


def test_package_scoped_include_cannot_traverse_out_of_its_root(tmp_path):
    """This dialect shares the unguarded join the cultcargo one had: the
    `(?P<file>.+)` part of `_PKG_INCLUDE_RE` cannot contain `..` only because
    nothing checked. See `test_yaml_cab_loader`'s equivalents.
    """
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.yaml").write_text("shared:\n  stolen:\n    dtype: bool\n")
    pkg_dir = tmp_path / "pkg" / "mypkg"
    pkg_dir.mkdir(parents=True)
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: (mypkg)../../outside/secret.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")

    with pytest.raises(ConfigLoadError, match="outside the package root"):
        load_worker_schema(main, package_roots={"mypkg": pkg_dir})


def test_package_file_cannot_re_escape_via_a_nested_plain_include(tmp_path):
    """Transitive, as in the cultcargo dialect: the included package file's
    own relative `_include` resolves against its directory, inside the root.
    """
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.yaml").write_text("stolen:\n  dtype: bool\n")
    pkg_dir = tmp_path / "pkg" / "mypkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "shared.yaml").write_text("shared:\n  _include: ../../outside/secret.yaml\n")
    main = tmp_path / "main.yaml"
    main.write_text("libs:\n  _include: (mypkg)shared.yaml\nname: thing\ninputs:\n  _use: libs.shared\n")

    with pytest.raises(ConfigLoadError, match="outside the package root"):
        load_worker_schema(main, package_roots={"mypkg": pkg_dir})


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


# ---- `_each`: a group whose keys come from the config --------------------

_EACH_SCHEMA = """
name: calibrate
inputs:
  refant:
    dtype: str
    default: m000
  chains:
    info: Named calibration chains.
    _key_pattern: '^[A-Za-z][A-Za-z0-9_]*$'
    _each:
      field:
        dtype: str
        default: target
      order:
        dtype: str
        default: KGB
      solve:
        engine:
          dtype: str
          choices: [casa, quartical]
          default: casa
    default:
      primary:
        field: fcal
        order: KGB
      secondary:
        field: gcal
        order: KGAF
        solve:
          engine: quartical
"""


def _each_model(tmp_path, text=_EACH_SCHEMA):
    path = tmp_path / "each.yaml"
    path.write_text(text)
    return load_worker_schema(path).inputs_model


def test_each_group_builds_a_mapping_of_submodels(tmp_path):
    model = _each_model(tmp_path)
    config = model(chains={"tertiary": {"field": "gcal", "order": "G"}})
    # keys are the config's, values validated against the one sub-schema
    assert set(config.chains) == {"tertiary"}
    assert config.chains["tertiary"].order == "G"
    # sub-schema defaults still apply to an entry that omits them
    assert config.chains["tertiary"].solve.engine == "casa"


def test_each_group_default_entries_load_and_are_not_shared(tmp_path):
    model = _each_model(tmp_path)
    first, second = model(), model()
    assert list(first.chains) == ["primary", "secondary"]
    assert first.chains["secondary"].solve.engine == "quartical"
    # the default is rebuilt per instance -- mutating one config's entry must
    # not reach into another's (a plain `default=` mapping would share them)
    first.chains["primary"].order = "KG"
    assert second.chains["primary"].order == "KGB"


def test_each_group_validates_entries_against_the_sub_schema(tmp_path):
    model = _each_model(tmp_path)
    with pytest.raises(ValidationError):
        model(chains={"primary": {"solve": {"engine": "nonesuch"}}})


def test_each_group_rejects_a_key_that_does_not_match_key_pattern(tmp_path):
    model = _each_model(tmp_path)
    with pytest.raises(ValidationError, match="does not match"):
        model(chains={"not an identifier": {"order": "G"}})


def test_each_group_without_key_pattern_accepts_any_key(tmp_path):
    model = _each_model(tmp_path, _EACH_SCHEMA.replace("    _key_pattern: '^[A-Za-z][A-Za-z0-9_]*$'\n", ""))
    assert set(model(chains={"not an identifier": {}}).chains) == {"not an identifier"}


def test_each_wins_over_the_leaf_test_despite_info_and_default(tmp_path):
    # `info`/`default` are leaf-descriptor keys; a `_each` group carrying them
    # must still be a group, or it silently becomes a `str` field
    model = _each_model(tmp_path)
    assert model.model_fields["chains"].annotation is not str


def test_each_group_with_a_broken_default_raises_at_load(tmp_path):
    bad = _EACH_SCHEMA.replace("          engine: quartical", "          engine: nonesuch")
    with pytest.raises(ConfigLoadError, match="do not match"):
        _each_model(tmp_path, bad)


def test_each_group_with_a_non_mapping_each_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    _each: [a, b]\n")
    with pytest.raises(ConfigLoadError, match="must be a mapping"):
        load_worker_schema(bad)


def test_each_group_with_a_non_mapping_default_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    _each:\n      order: {dtype: str}\n    default: [a, b]\n")
    with pytest.raises(ConfigLoadError, match="must be a mapping of entries"):
        load_worker_schema(bad)


def test_each_group_with_dtype_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    dtype: str\n    _each:\n      order: {dtype: str}\n")
    with pytest.raises(ConfigLoadError, match="no dtype of its own"):
        load_worker_schema(bad)


def test_each_group_with_a_bad_key_pattern_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    _key_pattern: '['\n    _each:\n      order: {dtype: str}\n")
    with pytest.raises(ConfigLoadError, match="not a valid regex"):
        load_worker_schema(bad)


def test_each_sub_schema_resolves_use_directives(tmp_path):
    path = tmp_path / "each_use.yaml"
    path.write_text("libs:\n  solve:\n    order: {dtype: str, default: KGB}\nname: calibrate\ninputs:\n  chains:\n    _each:\n      solve:\n        _use: libs.solve\n")
    model = load_worker_schema(path).inputs_model
    assert model(chains={"primary": {}}).chains["primary"].solve.order == "KGB"


def test_each_group_with_a_leaf_sub_schema_names_the_real_mistake(tmp_path):
    # `_each: {dtype: str}` used to fail one frame deeper, complaining about a
    # param called 'dtype' in the *entry* model -- true, but not the mistake
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    _each:\n      dtype: str\n")
    with pytest.raises(ConfigLoadError, match="must be a group of parameters"):
        load_worker_schema(bad)


@pytest.mark.parametrize("key, value", [("required", "true"), ("choices", "[a, b]"), ("writable", "false"), ("implicit", "'{current.x}'")])
def test_each_group_rejects_leaf_keys_that_would_be_inert(tmp_path, key, value):
    # a mapping group is always optional and has no dtype, so `required: true`
    # (etc.) would be silently ignored -- reject it the way `dtype` is
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"name: bad\ninputs:\n  chains:\n    {key}: {value}\n    _each:\n      order: {{dtype: str}}\n")
    with pytest.raises(ConfigLoadError, match="may only carry"):
        load_worker_schema(bad)


def test_each_group_rejects_an_unrecognised_key(tmp_path):
    # a misspelt modifier (`_key_patthern` for `_key_pattern`) must not be
    # silently dropped -- that would quietly switch off the key validation
    # the schema asked for
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninputs:\n  chains:\n    _key_patthern: '^[A-Z]+$'\n    _each:\n      order: {dtype: str}\n")
    with pytest.raises(ConfigLoadError, match="_key_patthern"):
        load_worker_schema(bad)


def test_each_group_still_accepts_info_and_default(tmp_path):
    model = _each_model(tmp_path)
    assert model.model_fields["chains"].description == "Named calibration chains."
    assert list(model().chains) == ["primary", "secondary"]
