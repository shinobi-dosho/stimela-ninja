
from pathlib import Path

import pydantic
import pytest

from shinobi.exceptions import CabLoadError
from shinobi.loaders.cultcargo import load_file, loads
from shinobi.steps.schema import path_fields

BREIZORRO_YAML = """
cabs:
    breizorro:
        command: breizorro
        image:
            name: breizorro
        policies:
            replace: {'_': '-'}
        inputs:
            restored-image:
                dtype: File
            threshold:
                dtype: float
                default: 6.5
        outputs:
            mask:
                dtype: File
                nom_de_guerre: outfile
                required: true

    casa.flagsummary:
        info: Uses CASA flagdata to obtain a flag summary
        command: flagdata
        flavour: casa-task
        image:
            name: casa
        inputs:
            ms:
                dtype: MS
                required: true
                nom_de_guerre: vis
            mode:
                implicit: summary
        outputs:
            percentage:
                dtype: float
        management:
            wranglers:
                'Total Flagged: .* Total Counts: .* \\((?P<percentage>[\\d.]+)%\\)':
                  - PARSE_OUTPUT:percentage:float
"""


def test_loads_basic_cab():
    breizorro = loads(BREIZORRO_YAML)["breizorro"]
    assert breizorro.command == "breizorro"
    assert breizorro.image == "breizorro"
    assert breizorro.policies.replace == {"_": "-"}
    fields = breizorro.inputs_model.model_fields
    assert fields["threshold"].default == 6.5
    # "restored-image" is sanitised to a valid identifier, original kept as nom
    assert "restored_image" in fields
    assert breizorro.field_meta["restored_image"].nom_de_guerre == "restored-image"
    assert "restored_image" in path_fields(breizorro.inputs_model)
    assert "mask" in breizorro.outputs_model.model_fields


def test_loads_flavour_and_wranglers():
    flagsummary = loads(BREIZORRO_YAML)["casa.flagsummary"]
    assert flagsummary.flavour == "casa-task"
    assert flagsummary.field_meta["ms"].nom_de_guerre == "vis"
    assert flagsummary.field_meta["mode"].implicit == "summary"
    assert len(flagsummary.wranglers) == 1


POSITIONAL_YAML = """
cabs:
    telsim:
        command: simms telsim
        inputs:
            ms:
                dtype: MS
                required: true
                policies:
                    positional: true
            telescope:
                dtype: str
                required: true
"""


def test_param_positional_policy_parsed_into_meta():
    telsim = loads(POSITIONAL_YAML)["telsim"]
    assert telsim.field_meta["ms"].positional is True
    assert "telescope" not in telsim.field_meta or telsim.field_meta["telescope"].positional is False


REPEAT_YAML = """
cabs:
    wsclean:
        command: wsclean
        inputs:
            size:
                dtype: list:int
                required: true
                policies:
                    repeat: list
            multiscale-scales:
                dtype: list:int
                required: false
"""


def test_param_repeat_list_policy_parsed_into_meta():
    wsclean = loads(REPEAT_YAML)["wsclean"]
    assert wsclean.field_meta["size"].repeat_as_tokens is True
    # a field with no `policies.repeat: list` is unaffected (default comma-join)
    assert (
        "multiscale_scales" not in wsclean.field_meta
        or wsclean.field_meta["multiscale_scales"].repeat_as_tokens is False
    )


# -- _use / _include resolution --

USE_ON_IMAGE_YAML = """
vars:
  cult-cargo:
    images:
      registry: quay.io/stimela2
      version: cc0.2.1

cabs:
  breizorro:
    command: breizorro
    image:
      _use: vars.cult-cargo.images
      name: breizorro
"""


def test_use_deep_merges_with_sibling_keys_winning():
    assert loads(USE_ON_IMAGE_YAML)["breizorro"].image == "breizorro"


USE_INHERITS_WHOLE_BLOCK_YAML = """
lib:
  misc:
    casa6:
      command-data:
        command: flagmanager
        flavour:
          kind: casa-task

cabs:
  casa.flagman:
    _use: lib.misc.casa6.command-data
    info: "saves/restores flags"
"""


def test_use_can_inherit_entire_command_block():
    flagman = loads(USE_INHERITS_WHOLE_BLOCK_YAML)["casa.flagman"]
    assert flagman.command == "flagmanager"
    assert flagman.flavour == "casa-task"


def test_use_missing_path_raises_cab_load_error():
    with pytest.raises(CabLoadError):
        loads("cabs:\n  broken:\n    _use: does.not.exist\n")


def test_include_merges_files_relative_to_including_file(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text("vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n")
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n  - base.yml\ncabs:\n  breizorro:\n    command: breizorro\n"
        "    image:\n      _use: vars.cult-cargo.images\n      name: breizorro\n"
    )
    cabs = load_file(main)
    assert cabs["breizorro"].command == "breizorro"
    assert cabs["breizorro"].image == "breizorro"


def test_shared_include_is_only_read_from_disk_once(tmp_path, monkeypatch):
    """Regression test: `_load_raw` used to re-read and re-parse an
    `_include`-d file from disk on every reference, unlike
    `worker_schema._load_include_file`'s `lru_cache`'d equivalent -- a real
    cab library commonly has many cab files all `_include`-ing the same
    shared base. Now cached the same way, keyed on the resolved path.
    """
    base = tmp_path / "shared_base.yml"
    base.write_text("vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n")
    base_resolved = base.resolve()

    def make_main(cab_name: str) -> Path:
        main = tmp_path / f"{cab_name}.yml"
        main.write_text(
            f"_include:\n  - shared_base.yml\ncabs:\n  {cab_name}:\n    command: {cab_name}\n"
            "    image:\n      _use: vars.cult-cargo.images\n      name: x\n"
        )
        return main

    read_calls = []
    original_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self == base_resolved:
            read_calls.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    cabs_a = load_file(make_main("cab_a"))
    cabs_b = load_file(make_main("cab_b"))

    assert cabs_a["cab_a"].image == "x"
    assert cabs_b["cab_b"].image == "x"
    assert len(read_calls) == 1


def test_package_scoped_include_raises_clear_error_without_package_roots(tmp_path):
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n  - (cultcargo):\n      - genesis/cult-cargo-base.yml\n"
        "cabs:\n  plain:\n    command: echo\n"
    )
    with pytest.raises(CabLoadError, match="package_roots"):
        load_file(main)


def test_package_scoped_include_resolves_via_explicit_package_roots(tmp_path):
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis").mkdir(parents=True)
    (pkg_dir / "genesis" / "cult-cargo-base.yml").write_text(
        "vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n"
    )
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n  - (cultcargo):\n      - genesis/cult-cargo-base.yml\n"
        "cabs:\n  breizorro:\n    command: breizorro\n"
        "    image:\n      _use: vars.cult-cargo.images\n      name: breizorro\n"
    )
    cabs = load_file(main, package_roots={"cultcargo": pkg_dir})
    assert cabs["breizorro"].image == "breizorro"


def test_package_scoped_include_via_combined_string_form(tmp_path):
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis" / "cubical").mkdir(parents=True)
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text(
        "data:\n  ms:\n    dtype: MS\n    required: true\n"
    )
    main = tmp_path / "main.yml"
    main.write_text(
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
    )
    cabs = load_file(main, package_roots={"cultcargo": pkg_dir})
    assert "data_ms" in cabs["cubical"].inputs_model.model_fields


def test_dynamic_schema_warns_but_still_loads_static_inputs():
    text = (
        "cabs:\n  tool:\n    command: tool\n    dynamic_schema: some.module.make_schema\n"
        "    inputs:\n      size:\n        dtype: int\n"
    )
    with pytest.warns(UserWarning, match="dynamic_schema"):
        cabs = loads(text)
    assert "size" in cabs["tool"].inputs_model.model_fields


def test_nested_package_scoped_include_inside_inputs_raises_clear_error_without_roots():
    text = (
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
    )
    with pytest.raises(CabLoadError, match="package_roots"):
        loads(text)


# -- section-flattening (stimela2-style CLI-section-nested inputs) --------

SECTIONED_YAML = """
cabs:
    cubical:
        command: gocubical
        policies:
            prefix: '--'
            replace: {'.': '-'}
        inputs:
            data:
                ms:
                    dtype: MS
                    required: true
                column:
                    dtype: str
                    default: DATA
            sel:
                field:
                    dtype: int
"""


def test_section_nested_inputs_flatten_to_dotted_field_names():
    cubical = loads(SECTIONED_YAML)["cubical"]
    fields = cubical.inputs_model.model_fields
    assert "data_ms" in fields and "data_column" in fields and "sel_field" in fields
    assert "data" not in fields  # the section itself must not become a bogus field
    assert cubical.field_meta["data_ms"].nom_de_guerre == "data.ms"
    assert fields["data_column"].default == "DATA"


# -- dynamic_schema cabs: no stopgap tables, always just warn ------------
#
# wsclean/cubical/quartical's real, cross-checked static schemas now live
# in dosho (the native shinobi cab repository, a sibling project) instead
# of a per-cab ParamPattern table in this loader -- any dynamic_schema cab
# loaded through this module (including those three) just gets the
# generic warning and whatever static inputs/outputs are present, same as
# any other dynamic_schema cab. See dosho/cabs/{wsclean,cubical,quartical}.py
# for the real schemas.


def test_dynamic_schema_cab_gets_no_special_case_treatment(tmp_path):
    """Even a cab shaped exactly like cubical.yml (package-scoped _include
    + dynamic_schema) gets no per-cab pattern/allow_extra treatment
    anymore -- it just warns and loads its static fields as-is.
    """
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis" / "cubical").mkdir(parents=True)
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text(
        "data:\n  ms:\n    dtype: MS\n    required: true\n"
    )
    main = tmp_path / "cubical.yml"
    main.write_text(
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
        "    dynamic_schema: cultcargo.genesis.cubical.make_stimela_schema.make_stimela_schema\n"
    )
    with pytest.warns(UserWarning, match="dynamic_schema"):
        cubical = load_file(main, package_roots={"cultcargo": pkg_dir})["cubical"]
    assert "data_ms" in cubical.inputs_model.model_fields
    assert cubical.input_patterns == []
    assert cubical.inputs_model.model_config.get("extra") is None


def test_wsclean_shaped_dynamic_schema_cab_gets_no_output_pattern():
    text = "cabs:\n  wsclean:\n    command: wsclean\n    dynamic_schema: cultcargo.genesis.wsclean.make_stimela_schema\n"
    with pytest.warns(UserWarning, match="dynamic_schema"):
        wsclean = loads(text)["wsclean"]
    assert wsclean.output_patterns == []
    assert wsclean.match_output_pattern("dirty.per-band") is None


def test_bracket_list_dtype_resolves_on_real_simms_example():
    """Regression test for `_modelgen.dtype_to_type`'s `List[<inner>]` support:
    `examples/input-dir/simms-cabs.yaml`'s `telsim` cab declares `subarray-list`/
    `subarray-range` with bracket-syntax dtypes that, before that support was
    added, silently fell back to `str`. Locks in the now-correct `list[str]`/
    `list[int]` resolution so a future change to dtype_to_type can't silently
    re-break this real, already-shipped example without a test noticing.
    """
    simms_yaml = Path(__file__).parent.parent / "examples" / "input-dir" / "simms-cabs.yaml"
    cabs = load_file(simms_yaml)
    telsim_inputs = cabs["telsim"].inputs_model.model_fields
    assert telsim_inputs["subarray_list"].annotation == list[str] | None
    assert telsim_inputs["subarray_range"].annotation == list[int] | None


CHOICES_YAML = """
cabs:
    pick:
        command: pick
        inputs:
            mode:
                dtype: str
                choices: [auto, spw, scan]
                default: auto
"""


def test_choices_are_recorded_on_field_meta():
    cab = loads(CHOICES_YAML)["pick"]
    assert cab.field_meta["mode"].choices == ["auto", "spw", "scan"]


def test_choices_are_enforced_by_the_generated_model():
    cab = loads(CHOICES_YAML)["pick"]
    cab.inputs_model(mode="spw")
    with pytest.raises(pydantic.ValidationError):
        cab.inputs_model(mode="not-a-choice")


def test_non_list_choices_raise():
    bad = "cabs:\n  pick:\n    command: pick\n    inputs:\n      mode:\n        dtype: str\n        choices: auto\n"
    with pytest.raises(CabLoadError, match="'choices' must be a list"):
        loads(bad)
