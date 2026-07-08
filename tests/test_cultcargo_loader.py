
import warnings
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


# -- dynamic_schema cabs: per-cab static ParamPattern catch-alls ----------


def _write_cubical_fixture(tmp_path: Path) -> Path:
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis" / "cubical").mkdir(parents=True)
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text(
        "data:\n  ms:\n    dtype: MS\n    required: true\n"
    )
    (pkg_dir / "genesis" / "cubical" / "schema_JONES_TEMPLATE.yaml").write_text(
        "JONES_TEMPLATE:\n"
        "  solvable:\n    info: whether this term is solvable\n"
        "  time-int:\n    info: time solution interval\n"
    )
    main = tmp_path / "cubical.yml"
    main.write_text(
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    policies:\n      prefix: '--'\n      replace: {'.': '-'}\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
        "    dynamic_schema: cultcargo.genesis.cubical.make_stimela_schema.make_stimela_schema\n"
    )
    return main


def test_dynamic_cab_gets_input_pattern_and_allow_extra(tmp_path):
    from shinobi.loaders._modelgen import build_model

    main = _write_cubical_fixture(tmp_path)
    cubical = load_file(main, package_roots={"cultcargo": tmp_path / "cultcargo"})["cubical"]

    assert "data_ms" in cubical.inputs_model.model_fields
    assert cubical.inputs_model.model_config.get("extra") == "allow"
    meta = cubical.match_pattern("g1.solvable")
    assert meta is not None
    assert cubical.match_pattern("g1.time-int") is not None
    assert cubical.match_pattern("g1.not-a-real-attr") is None
    # a genuinely unrelated model with allow_extra shouldn't matter here --
    # just sanity-checking build_model itself isn't broken (used elsewhere)
    assert build_model("X", {}).model_config.get("extra") is None


def test_dynamic_cab_argv_uses_replace_policy_for_pattern_matched_field(tmp_path):
    from shinobi.policies import build_argv

    main = _write_cubical_fixture(tmp_path)
    cubical = load_file(main, package_roots={"cultcargo": tmp_path / "cultcargo"})["cubical"]
    resolved = {"data_ms": "foo.ms", "g1.solvable": True}
    argv = build_argv(cubical, resolved)
    assert "--g1-solvable" in argv


def test_dynamic_input_patterns_reads_each_template_file_once_per_load(tmp_path, monkeypatch):
    """Regression guard for the redundant-per-cab-I/O review finding:
    _dynamic_input_patterns must be computed once per load_file()/loads()
    call, not once per cab defined in the loaded document.
    """
    import shinobi.loaders.cultcargo as cultcargo_loader

    main = _write_cubical_fixture(tmp_path)
    # a second, unrelated cab in the same document -- if _dynamic_input_
    # patterns were still being called once per cab, loading this
    # two-cab file would re-read cubical's template file twice.
    main.write_text(main.read_text() + "  plain:\n    command: echo\n    inputs:\n      x:\n        dtype: str\n")

    calls = 0
    real = cultcargo_loader._load_template_attrs

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cultcargo_loader, "_load_template_attrs", counting)
    cabs = load_file(main, package_roots={"cultcargo": tmp_path / "cultcargo"})
    assert set(cabs) == {"cubical", "plain"}
    # one call per real template source (cubical, quartical) -- not one
    # per cab defined in the document.
    assert calls == 2


def test_dynamic_cab_template_attrs_carry_policies_and_implicit(tmp_path):
    """_load_template_attrs must forward nom_de_guerre/implicit/
    positional/repeat_as_tokens from a template attr's own spec, not just
    info/dtype -- the "drops template fields" review finding.
    """
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis" / "cubical").mkdir(parents=True)
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text(
        "data:\n  ms:\n    dtype: MS\n    required: true\n"
    )
    (pkg_dir / "genesis" / "cubical" / "schema_JONES_TEMPLATE.yaml").write_text(
        "JONES_TEMPLATE:\n"
        "  solvable:\n    info: whether this term is solvable\n"
        "  positions:\n    info: positional attr\n    policies:\n      positional: true\n"
        "  tags:\n    info: repeated attr\n    policies:\n      repeat: list\n"
    )
    main = tmp_path / "cubical.yml"
    main.write_text(
        "cabs:\n  cubical:\n    command: gocubical\n"
        "    policies:\n      prefix: '--'\n      replace: {'.': '-'}\n"
        "    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
        "    dynamic_schema: cultcargo.genesis.cubical.make_stimela_schema.make_stimela_schema\n"
    )
    cubical = load_file(main, package_roots={"cultcargo": tmp_path / "cultcargo"})["cubical"]
    pattern = cubical.input_patterns[0]
    attrs = next(seg.attrs for seg in pattern.segments if seg.attrs)
    assert attrs["positions"].positional is True
    assert attrs["tags"].repeat_as_tokens is True
    assert attrs["solvable"].positional is False
    assert attrs["solvable"].repeat_as_tokens is False


def test_cab_without_template_file_loads_without_dynamic_pattern(tmp_path):
    """cubical.yml still loads fine (allow_extra=False, no input_patterns)
    if package_roots is supplied but the JONES_TEMPLATE file itself is
    missing -- graceful degrade, not a hard failure.
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
    assert cubical.input_patterns == []
    assert cubical.inputs_model.model_config.get("extra") is None


# -- wsclean output_patterns (validation-only) -----------------------------


def test_wsclean_dynamic_schema_gets_output_pattern_no_warning():
    text = "cabs:\n  wsclean:\n    command: wsclean\n    dynamic_schema: cultcargo.genesis.wsclean.make_stimela_schema\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wsclean = loads(text)["wsclean"]
    assert wsclean.match_output_pattern("dirty.per-band") is not None
    assert wsclean.match_output_pattern("restored.i.per-interval.mfs") is not None
    assert wsclean.match_output_pattern("totally-unknown-shape") is None


def test_wsclean_image_is_not_a_real_imagetype_key():
    """"restored" is the real outputs-dict key prefix; "image" is
    img_output()'s own real on-disk filename component for it (its
    nom_de_guerre, see cultcargo.py's own _WSCLEAN_IMAGETYPES comment)
    -- "image" itself must not be accepted as a key.
    """
    text = "cabs:\n  wsclean:\n    command: wsclean\n    dynamic_schema: cultcargo.genesis.wsclean.make_stimela_schema\n"
    wsclean = loads(text)["wsclean"]
    assert wsclean.match_output_pattern("image.per-band") is None
    restored_meta = next(
        seg.attrs["restored"] for seg in wsclean.output_patterns[0].segments if seg.attrs and "restored" in seg.attrs
    )
    assert restored_meta.nom_de_guerre == "image"


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
