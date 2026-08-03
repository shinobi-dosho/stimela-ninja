from pathlib import Path

import pydantic
import pytest

from shinobi.exceptions import CabLoadError
from shinobi.loaders.yaml_cab import load_file, loads
from shinobi.policies import build_argv
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


POSITIONAL_HEAD_YAML = """
cabs:
    cubical:
        command: gocubical
        inputs:
            parset:
                dtype: File
                required: false
                policies:
                    positional_head: true
            data-ms:
                dtype: MS
                required: true
"""


def test_param_positional_head_policy_parsed_into_meta():
    cubical = loads(POSITIONAL_HEAD_YAML)["cubical"]
    assert cubical.field_meta["parset"].positional_head is True
    assert "data_ms" not in cubical.field_meta or cubical.field_meta["data_ms"].positional_head is False


def test_positional_head_policy_produces_head_positional_in_argv():
    # end-to-end: a YAML cab shaped like cubical.yml emits parset as argv[1],
    # before every flag -- the whole point of the positional_head policy.
    cubical = loads(POSITIONAL_HEAD_YAML)["cubical"]
    argv = build_argv(cubical, {"parset": "base.parset", "data_ms": "foo.ms"})
    assert argv == ["gocubical", "base.parset", "--data-ms", "foo.ms"]


def test_param_repeat_list_policy_parsed_into_meta():
    wsclean = loads(REPEAT_YAML)["wsclean"]
    assert wsclean.field_meta["size"].repeat_as_tokens is True
    # a field with no `policies.repeat: list` is unaffected (default comma-join)
    assert "multiscale_scales" not in wsclean.field_meta or wsclean.field_meta["multiscale_scales"].repeat_as_tokens is False


ABBREV_CHOICE_YAML = """
cabs:
    skysim:
        command: simms skysim
        inputs:
            ms:
                dtype: MS
                required: true
            ascii-sky:
                dtype: File
                abbreviation: as
            mode:
                dtype: str
                default: sim
                choices: [sim, add, subtract]
                abbreviation: m
            column:
                dtype: str
                default: DATA
"""


def test_abbreviation_flows_to_meta_and_field_json_schema_extra():
    skysim = loads(ABBREV_CHOICE_YAML)["skysim"]
    # captured on the ParamMeta ...
    assert skysim.field_meta["ascii_sky"].abbreviation == "as"
    assert skysim.field_meta["mode"].abbreviation == "m"
    # ... and carried onto the model field so build_options can read it.
    fields = skysim.inputs_model.model_fields
    assert fields["ascii_sky"].json_schema_extra == {"abbreviation": "as"}
    assert fields["mode"].json_schema_extra == {"abbreviation": "m"}
    # a field with no abbreviation carries no extra
    assert fields["column"].json_schema_extra is None


def test_choices_narrow_field_annotation_to_literal():
    from typing import get_args

    skysim = loads(ABBREV_CHOICE_YAML)["skysim"]
    # mode is a choice-with-default -> Optional[Literal[...]]; the Literal's
    # allowed values are exactly the `choices:` list.
    assert set(get_args(get_args(skysim.inputs_model.model_fields["mode"].annotation)[0])) == {
        "sim",
        "add",
        "subtract",
    }
    assert skysim.field_meta["mode"].choices == ["sim", "add", "subtract"]
    # an out-of-set value fails pydantic validation
    with pytest.raises(pydantic.ValidationError):
        skysim.inputs_model(ms="x.ms", mode="bogus")


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
    main.write_text("_include:\n  - base.yml\ncabs:\n  breizorro:\n    command: breizorro\n    image:\n      _use: vars.cult-cargo.images\n      name: breizorro\n")
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
        main.write_text(f"_include:\n  - shared_base.yml\ncabs:\n  {cab_name}:\n    command: {cab_name}\n    image:\n      _use: vars.cult-cargo.images\n      name: x\n")
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
    main.write_text("_include:\n  - (cultcargo):\n      - genesis/cult-cargo-base.yml\ncabs:\n  plain:\n    command: echo\n")
    with pytest.raises(CabLoadError, match="package_roots"):
        load_file(main)


def test_package_scoped_include_resolves_via_explicit_package_roots(tmp_path):
    pkg_dir = tmp_path / "cultcargo"
    (pkg_dir / "genesis").mkdir(parents=True)
    (pkg_dir / "genesis" / "cult-cargo-base.yml").write_text("vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n")
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
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text("data:\n  ms:\n    dtype: MS\n    required: true\n")
    main = tmp_path / "main.yml"
    main.write_text("cabs:\n  cubical:\n    command: gocubical\n    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n")
    cabs = load_file(main, package_roots={"cultcargo": pkg_dir})
    assert "data_ms" in cabs["cubical"].inputs_model.model_fields


def _secret_outside(tmp_path: Path) -> Path:
    """A YAML-parseable file outside any package root, plus the package root
    a traversal would have to escape to reach it."""
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.yaml").write_text("vars:\n  stolen: yes\n")
    pkg_dir = tmp_path / "pkg" / "cultcargo"
    (pkg_dir / "genesis").mkdir(parents=True)
    return pkg_dir


def test_package_scoped_include_cannot_traverse_out_of_its_root_combined_form(tmp_path):
    """Regression test: `resolve_package_root` constrains only the dotted
    part; the file part was joined unguarded, so
    `(cultcargo)../../outside/secret.yaml` read straight out of the root that
    `package_roots` exists to define.
    """
    pkg_dir = _secret_outside(tmp_path)
    main = tmp_path / "main.yml"
    main.write_text("cabs:\n  tool:\n    command: echo\n    inputs:\n      _include: (cultcargo)../../outside/secret.yaml\n")
    with pytest.raises(CabLoadError, match="outside the package root"):
        load_file(main, package_roots={"cultcargo": pkg_dir})


def test_package_scoped_include_cannot_traverse_out_of_its_root_dict_form(tmp_path):
    """The dict form takes its filenames from a YAML *list* that never passes
    through `_PKG_INCLUDE_RE` at all -- a regex-level fix leaves this open,
    which is why the check lives at the join.
    """
    pkg_dir = _secret_outside(tmp_path)
    main = tmp_path / "main.yml"
    main.write_text("_include:\n  - (cultcargo):\n      - ../../outside/secret.yaml\ncabs:\n  tool:\n    command: echo\n")
    with pytest.raises(CabLoadError, match="outside the package root"):
        load_file(main, package_roots={"cultcargo": pkg_dir})


def test_package_file_cannot_re_escape_via_a_nested_plain_include(tmp_path):
    """The guard is transitive: hop one lands legitimately inside the root,
    but the included file's own plain relative `_include` resolves against
    *its* directory -- by then inside the package -- and would otherwise walk
    back out, defeating the containment hop one just enforced.
    """
    pkg_dir = _secret_outside(tmp_path)
    (pkg_dir / "genesis" / "base.yml").write_text("_include:\n  - ../../../outside/secret.yaml\n")
    main = tmp_path / "main.yml"
    main.write_text("_include:\n  - (cultcargo)genesis/base.yml\ncabs:\n  tool:\n    command: echo\n")
    with pytest.raises(CabLoadError, match="outside the package root"):
        load_file(main, package_roots={"cultcargo": pkg_dir})


def test_package_scoped_include_cannot_escape_via_a_symlink(tmp_path):
    """Both sides are resolved before comparing, so a symlink planted inside
    the package root is caught -- a textual `..` check would pass this.
    """
    pkg_dir = _secret_outside(tmp_path)
    (pkg_dir / "genesis" / "base.yml").symlink_to(tmp_path / "outside" / "secret.yaml")
    main = tmp_path / "main.yml"
    main.write_text("_include:\n  - (cultcargo)genesis/base.yml\ncabs:\n  tool:\n    command: echo\n")
    with pytest.raises(CabLoadError, match="outside the package root"):
        load_file(main, package_roots={"cultcargo": pkg_dir})


def test_plain_relative_include_may_still_reach_a_sibling_directory(tmp_path):
    """Deliberate scope limit: only `package_roots` ever promised
    containment. A plain relative `_include: ../common/base.yml` below no
    package root is real, widely-used schema layout and stays allowed.
    """
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "base.yml").write_text("vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n")
    (tmp_path / "cabs").mkdir()
    main = tmp_path / "cabs" / "main.yml"
    main.write_text("_include:\n  - ../common/base.yml\ncabs:\n  breizorro:\n    command: breizorro\n    image:\n      _use: vars.cult-cargo.images\n      name: breizorro\n")
    assert load_file(main)["breizorro"].image == "breizorro"


def test_nested_package_scoped_include_rebases_containment_on_the_new_root(tmp_path):
    """A package file may include *another* registered package's file: that
    hop re-enters through `package_roots`, which the caller controls, so the
    root is replaced rather than treated as an escape.
    """
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    pkg_a.mkdir()
    pkg_b.mkdir()
    (pkg_b / "shared.yml").write_text("vars:\n  cult-cargo:\n    images:\n      registry: quay.io/stimela2\n")
    (pkg_a / "base.yml").write_text("_include:\n  - (pkg_b)shared.yml\n")
    main = tmp_path / "main.yml"
    main.write_text("_include:\n  - (pkg_a)base.yml\ncabs:\n  breizorro:\n    command: breizorro\n    image:\n      _use: vars.cult-cargo.images\n      name: breizorro\n")
    cabs = load_file(main, package_roots={"pkg_a": pkg_a, "pkg_b": pkg_b})
    assert cabs["breizorro"].image == "breizorro"


def test_dynamic_schema_warns_but_still_loads_static_inputs():
    text = "cabs:\n  tool:\n    command: tool\n    dynamic_schema: some.module.make_schema\n    inputs:\n      size:\n        dtype: int\n"
    with pytest.warns(UserWarning, match="dynamic_schema"):
        cabs = loads(text)
    assert "size" in cabs["tool"].inputs_model.model_fields


def test_nested_package_scoped_include_inside_inputs_raises_clear_error_without_roots():
    text = "cabs:\n  cubical:\n    command: gocubical\n    inputs:\n      _include: (cultcargo.genesis.cubical)schema.yaml\n"
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
    (pkg_dir / "genesis" / "cubical" / "schema.yaml").write_text("data:\n  ms:\n    dtype: MS\n    required: true\n")
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


def test_output_implicit_template_survives_the_load():
    """An `implicit` on an *output* is how a cab declares where it writes.

    Only input metas used to reach the built `Cab`, so an output template was
    dropped on the floor: nothing filled the output's value, and
    `declared_output_dirs` saw no write directory, so a container run left the
    tool's products inside the container. Silent in both directions -- the
    existing coverage only exercised `implicit` on an input, which worked.
    """
    from shinobi.steps.schema import declared_output_dirs

    cab = loads(
        """
        cabs:
          probe:
            command: probe
            inputs:
              prefix: {dtype: str}
            outputs:
              dirty: {dtype: File, implicit: "{prefix}-dirty.fits"}
        """
    )["probe"]
    assert cab.field_meta["dirty"].implicit == "{prefix}-dirty.fits"
    dirs = declared_output_dirs(cab, {"prefix": "/data/out/img"})
    assert [(str(d), s) for d, s in dirs] == [("/data/out", "output 'dirty'")]


def test_output_meta_wins_over_an_input_of_the_same_name():
    """Pins the merge order against `dosho._builder.define_cab`'s, so a cab
    built from a document and the same cab built in Python agree.
    """
    cab = loads(
        """
        cabs:
          probe:
            command: probe
            inputs:
              out: {dtype: str, info: from-the-input}
            outputs:
              out: {dtype: File, info: from-the-output}
        """
    )["probe"]
    assert cab.field_meta["out"].info == "from-the-output"


# --------------------------------------------------------------------------
# shinobi-native keys: the scabha dialect plus what shinobi's own Cab carries
# --------------------------------------------------------------------------


def _shinobi_doc(extra_field: str = "") -> str:
    return f"""
    cabs:
      probe:
        command: probe
        sandbox: true
        harvest: ["{{prefix}}-*.fits"]
        scratch: ["{{cache}}/*"]
        inputs:
          prefix: {{dtype: str, write_path: true}}
          cache:  {{dtype: str}}
          ms:     {{dtype: MS, mutable: true}}
          {extra_field}
        outputs:
          dirty: {{dtype: File, implicit: "{{prefix}}-dirty.fits"}}
    """


def test_shinobi_native_cab_level_keys():
    cab = loads(_shinobi_doc())["probe"]
    assert cab.sandbox is True
    assert cab.harvest == ["{prefix}-*.fits"]
    assert cab.scratch == ["{cache}/*"]


def test_mutable_marks_an_input_mutable():
    from shinobi.steps.schema import Mutability

    cab = loads(_shinobi_doc())["probe"]
    assert cab.input_mutability == {"ms": Mutability.MUTABLE}


def test_write_path_reaches_the_field_meta():
    cab = loads(_shinobi_doc())["probe"]
    assert cab.field_meta["prefix"].write_path is True
    # and the Cab validator is satisfied by the harvest/implicit declarations
    from shinobi.steps.schema import declared_output_dirs

    assert declared_output_dirs(cab, {"prefix": "/out/img", "cache": "/c"})


def test_write_path_survives_a_same_named_output():
    """The dual declaration -- `mstransform(outputvis=...) -> outputvis`, the
    shape `write_path` exists to disambiguate -- is exactly the shape whose
    input meta the loader used to throw away, because the output side's meta
    replaced the whole object. The marker then reached nothing, and
    `clear_stale_outputs` read the cab as "the caller's data, leave it".

    The output entry carries an `info` deliberately: a spec that declares
    nothing but a dtype produces a *default* `ParamMeta`, which the loader
    never stores, so it could not clobber anything and the bug would not
    reproduce. Real cabs repeat the description on both sides -- which is
    what made this fire on 14 of dosho's documents.
    """
    cab = loads(
        """
        cabs:
          mstransform:
            command: mstransform
            inputs:
              vis: {dtype: MS, required: true}
              outputvis: {dtype: MS, required: true, write_path: true, info: Output MS path.}
            outputs:
              outputvis: {dtype: MS, info: Output MS path.}
        """
    )["mstransform"]
    from shinobi.steps.schema import write_path_fields

    assert write_path_fields(cab) == {"outputvis"}


def test_an_output_side_implicit_still_wins_over_the_input_side():
    """The other half of the merge: attribute-wise, so the output's own
    declarations keep overriding. A whole-object merge got this right and
    every loaded cab depends on it.
    """
    cab = loads(
        """
        cabs:
          probe:
            command: probe
            inputs:
              stem: {dtype: File, write_path: true, info: the input side}
            outputs:
              stem: {dtype: File, implicit: "{stem}-image.fits", info: the output side}
        """
    )["probe"]
    meta = cab.field_meta["stem"]
    assert meta.implicit == "{stem}-image.fits"
    assert meta.info == "the output side"
    assert meta.write_path is True


def test_an_input_only_attribute_survives_an_output_that_is_silent_on_it():
    """A dual-declared name whose output entry omits what the input said --
    the common shape, since cab authors rarely repeat a flag name on both
    sides -- keeps the input's value rather than resetting it to the default.
    """
    cab = loads(
        """
        cabs:
          probe:
            command: probe
            inputs:
              out_file: {dtype: File, nom_de_guerre: out, abbreviation: o}
            outputs:
              out_file: {dtype: File, info: the product}
        """
    )["probe"]
    assert cab.field_meta["out_file"].nom_de_guerre == "out"
    assert cab.field_meta["out_file"].abbreviation == "o"


def test_a_spec_carrying_only_a_shinobi_key_is_a_leaf_not_a_section():
    """`_is_section` decides leaf-vs-section by whether a mapping has any known
    param-spec key. A new key that isn't registered makes a spec using only it
    look like a nested CLI section, and the field silently disappears -- so
    both new keys are in `_LEAF_SPEC_KEYS`, and this is what checks it.
    """
    cab = loads(
        """
        cabs:
          probe:
            command: probe
            inputs:
              stem: {write_path: true}
              inplace: {mutable: true}
            outputs:
              out: {dtype: File, implicit: "{stem}.fits"}
        """
    )["probe"]
    assert "stem" in cab.inputs_model.model_fields
    assert "inplace" in cab.inputs_model.model_fields
    assert cab.field_meta["stem"].write_path is True


def test_the_new_keys_are_optional():
    """A plain scabha document -- cult-cargo's own files -- is unaffected."""
    cab = loads(
        """
        cabs:
          plain:
            command: plain
            inputs: {x: {dtype: int}}
        """
    )["plain"]
    assert cab.input_mutability == {}
    assert cab.harvest == [] and cab.scratch == [] and cab.sandbox is None
    assert cab.field_meta.get("x") is None


# --------------------------------------------------------------------------
# Symbolic image keys
# --------------------------------------------------------------------------

_IMG_DOC = """
cabs:
  probe:
    command: probe
    image: WSCLEAN
"""


def test_image_key_resolves_through_the_caller_mapping():
    cab = loads(_IMG_DOC, images={"WSCLEAN": "ghcr.io/org/wsclean:3.6"})["probe"]
    assert cab.image == "ghcr.io/org/wsclean:3.6"


def test_image_key_is_left_alone_without_a_mapping():
    """No mapping means no resolution -- shinobi has no manifest of its own."""
    assert loads(_IMG_DOC)["probe"].image == "WSCLEAN"


def test_a_literal_reference_passes_through_the_mapping():
    """The older form. cult-cargo's own files carry references directly, and a
    mapping supplied for other cabs must not disturb them.
    """
    doc = """
    cabs:
      probe:
        command: probe
        image: quay.io/stimela2/breizorro:0.1.2
    """
    cab = loads(doc, images={"WSCLEAN": "ghcr.io/org/wsclean:3.6"})["probe"]
    assert cab.image == "quay.io/stimela2/breizorro:0.1.2"


def test_an_unmapped_bare_name_is_not_rejected():
    """`ubuntu` is a legitimate image name. Refusing bare names to catch a
    misspelled key would break more than it caught, so a typo instead fails
    at pull time -- loudly, and in the runtime that can actually tell.
    """
    doc = "cabs:\n  probe:\n    command: probe\n    image: ubuntu\n"
    assert loads(doc, images={"WSCLEAN": "x"})["probe"].image == "ubuntu"


def test_the_dict_image_form_still_resolves():
    """cult-cargo writes `image: {_use: ..., name: breizorro}`; the name that
    falls out of that is resolved like any other.
    """
    doc = """
    cabs:
      probe:
        command: probe
        image: {name: BREIZORRO}
    """
    cab = loads(doc, images={"BREIZORRO": "ghcr.io/org/breizorro:0.2"})["probe"]
    assert cab.image == "ghcr.io/org/breizorro:0.2"


# --------------------------------------------------------------------------
# ParamPattern: dynamically-named inputs and outputs
# --------------------------------------------------------------------------

_PATTERN_DOC = """
cabs:
  probe:
    command: probe
    input_patterns:
      - separator: "-"
        segments:
          - regex: ".+?"
          - attrs:
              solvable: {}
              time-int: {dtype: int}
              load-from: {dtype: File, info: gain table}
    output_patterns:
      - separator: "."
        segments:
          - attrs:
              dirty: {dtype: File}
          - regex: ".+"
"""


def test_input_pattern_round_trips_into_a_matchable_pattern():
    cab = loads(_PATTERN_DOC)["probe"]
    pattern = cab.input_patterns[0]
    assert pattern.separator == "-"
    assert pattern.segments[0].regex == ".+?"
    assert sorted(pattern.segments[1].attrs) == ["load-from", "solvable", "time-int"]


def test_matched_attrs_carry_their_dtype():
    """The reason `ParamMeta.dtype` exists: a dynamically-named input has no
    model field, so this is the only thing telling a backend it is file-like.
    """
    cab = loads(_PATTERN_DOC)["probe"]
    assert cab.match_pattern("g1-time-int").dtype == "int"
    assert cab.match_pattern("g1-load-from").dtype == "File"
    assert cab.match_pattern("g1-load-from").info == "gain table"


def test_an_attr_with_an_empty_spec_survives():
    """An all-default `ParamMeta` is dropped for a declared field, where it says
    nothing. For an attr the *name* is the information -- dropping it would
    delete the attr from the pattern and stop it matching.
    """
    cab = loads(_PATTERN_DOC)["probe"]
    assert cab.match_pattern("g1-solvable") is not None


def test_the_attrs_segment_need_not_be_last():
    """wsclean's output names are the opposite shape from cubical's inputs: a
    known image type followed by an open-ended tail.
    """
    cab = loads(_PATTERN_DOC)["probe"]
    assert cab.match_output_pattern("dirty.per-band") is not None


def test_patterns_are_optional():
    cab = loads("cabs:\n  p:\n    command: p\n")["p"]
    assert cab.input_patterns == [] and cab.output_patterns == []


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ("input_patterns: {}", "must be a list of patterns"),
        ("input_patterns:\n      - segments: []", "non-empty 'segments'"),
        ("input_patterns:\n      - segments: [{regex: 'x'}]", "not a valid pattern"),
        (
            "input_patterns:\n      - segments: [{attrs: {a: {}}}, {attrs: {b: {}}}]",
            "not a valid pattern",
        ),
        ("input_patterns:\n      - segments: [{attrs: []}]", "attrs. must be a mapping"),
    ],
)
def test_malformed_patterns_are_rejected_by_name(doc, match):
    """A pattern is the one part of a cab a reader cannot check by eye against
    the tool's own --help, so the errors name the cab and the key.
    """
    with pytest.raises(CabLoadError, match=match):
        loads(f"cabs:\n  probe:\n    command: probe\n    {doc}\n")


def test_input_patterns_make_the_model_accept_extras():
    """A pattern exists to accept names no field declares. Without
    `allow_extra` the model rejects every value the pattern was written to
    match, so the pattern silently does nothing -- the cab loads, looks
    correct, and fails at the first dynamic parameter.

    Not caught by the round-trip comparator: it compares `model_fields`, and
    this lives in `model_config`.
    """
    cab = loads(_PATTERN_DOC)["probe"]
    assert cab.inputs_model.model_config.get("extra") == "allow"
    validated = cab.inputs_model(**{"g1-time-int": 4})
    assert validated.model_extra == {"g1-time-int": 4}


def test_a_cab_without_patterns_still_rejects_extras():
    """Scoped to cabs that need it: an unexpected parameter should still be an
    error everywhere else, which is most cabs.
    """
    cab = loads("cabs:\n  p:\n    command: p\n    inputs: {x: {dtype: int}}\n")["p"]
    assert cab.inputs_model.model_config.get("extra") is None
