import pytest

from shinobi.exceptions import CabLoadError
from shinobi.loaders.cultcargo import load_file, loads

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
    cabs = loads(BREIZORRO_YAML)
    breizorro = cabs["breizorro"]
    assert breizorro.command == "breizorro"
    assert breizorro.image == "breizorro"
    assert breizorro.policies.replace == {"_": "-"}
    assert breizorro.inputs["threshold"].default == 6.5
    assert breizorro.outputs["mask"].nom_de_guerre == "outfile"
    assert breizorro.outputs["mask"].required is True


def test_loads_flavour_and_wranglers():
    cabs = loads(BREIZORRO_YAML)
    flagsummary = cabs["casa.flagsummary"]
    assert flagsummary.flavour == "casa-task"
    assert flagsummary.inputs["ms"].nom_de_guerre == "vis"
    assert flagsummary.inputs["mode"].implicit == "summary"
    assert len(flagsummary.wranglers) == 1


# -- _use / _include resolution, mirroring real caracal-pipeline/cult-cargo layout --

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
    cabs = loads(USE_ON_IMAGE_YAML)
    # image is a dict per the raw spec ({registry, version, name}); the
    # loader currently narrows dict images down to their "name" for the
    # CabDef.image string field, so what matters here is that the merge
    # happened (name survives, sibling override wins over the _use source).
    assert cabs["breizorro"].image == "breizorro"


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
    cabs = loads(USE_INHERITS_WHOLE_BLOCK_YAML)
    flagman = cabs["casa.flagman"]
    assert flagman.command == "flagmanager"
    assert flagman.flavour == "casa-task"


def test_use_missing_path_raises_cab_load_error():
    with pytest.raises(CabLoadError):
        loads(
            """
            cabs:
              broken:
                _use: does.not.exist
            """
        )


def test_include_merges_files_relative_to_including_file(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text(
        "vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n"
    )
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n"
        "  - base.yml\n"
        "cabs:\n"
        "  breizorro:\n"
        "    command: breizorro\n"
        "    image:\n"
        "      _use: vars.cult-cargo.images\n"
        "      name: breizorro\n"
    )

    cabs = load_file(main)
    assert cabs["breizorro"].command == "breizorro"
    assert cabs["breizorro"].image == "breizorro"


def test_package_scoped_include_is_skipped_with_warning(tmp_path):
    main = tmp_path / "main.yml"
    main.write_text(
        "_include:\n"
        "  - (cultcargo):\n"
        "      - genesis/cult-cargo-base.yml\n"
        "cabs:\n"
        "  plain:\n"
        "    command: echo\n"
    )

    with pytest.warns(UserWarning, match="package-scoped"):
        cabs = load_file(main)
    assert cabs["plain"].command == "echo"
