
from pathlib import Path

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


def test_package_scoped_include_is_skipped_with_warning(tmp_path):
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n  - (cultcargo):\n      - genesis/cult-cargo-base.yml\n"
        "cabs:\n  plain:\n    command: echo\n"
    )
    with pytest.warns(UserWarning, match="package-scoped"):
        cabs = load_file(main)
    assert cabs["plain"].command == "echo"


def test_dynamic_schema_warns_but_still_loads_static_inputs():
    text = (
        "cabs:\n  tool:\n    command: tool\n    dynamic_schema: some.module.make_schema\n"
        "    inputs:\n      size:\n        dtype: int\n"
    )
    with pytest.warns(UserWarning, match="dynamic_schema"):
        cabs = loads(text)
    assert "size" in cabs["tool"].inputs_model.model_fields


def test_nested_package_scoped_include_inside_inputs_raises_clear_error():
    text = (
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
    )
    with pytest.raises(CabLoadError, match="param spec mapping"):
        loads(text)


def test_bracket_list_dtype_resolves_on_real_simms_example():
    """Regression test for `_modelgen.dtype_to_type`'s `List[<inner>]` support:
    `examples/simms/simms-cabs.yaml`'s `telsim` cab declares `subarray-list`/
    `subarray-range` with bracket-syntax dtypes that, before that support was
    added, silently fell back to `str`. Locks in the now-correct `list[str]`/
    `list[int]` resolution so a future change to dtype_to_type can't silently
    re-break this real, already-shipped example without a test noticing.
    """
    simms_yaml = Path(__file__).parent.parent / "examples" / "simms" / "simms-cabs.yaml"
    cabs = load_file(simms_yaml)
    telsim_inputs = cabs["telsim"].inputs_model.model_fields
    assert telsim_inputs["subarray_list"].annotation == list[str] | None
    assert telsim_inputs["subarray_range"].annotation == list[int] | None
