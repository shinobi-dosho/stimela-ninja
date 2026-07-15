import pytest

from shinobi.exceptions import UnsupportedFlavourError
from shinobi.loaders import build_model
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern, ParamSegment, Policies


def make_cab(model, **kwargs) -> Cab:
    return Cab(
        name="tool", command="tool", inputs_model=model, outputs_model=build_model("Out", {}), **kwargs
    )


def test_scalar_value_becomes_flag_and_value():
    cab = make_cab(build_model("In", {"size": ("int", False, 1024)}))
    assert build_argv(cab, {"size": 1024}) == ["tool", "--size", "1024"]


def test_default_is_used_when_supplied_by_prepared_dict():
    cab = make_cab(build_model("In", {"size": ("int", False, 1024)}))
    # dispatch prepares defaults into the dict; build_argv just formats them
    assert build_argv(cab, {"size": 1024}) == ["tool", "--size", "1024"]


def test_none_value_is_skipped():
    cab = make_cab(build_model("In", {"size": ("int", False, None)}))
    assert build_argv(cab, {"size": None}) == ["tool"]


def test_bool_flag_only_emitted_when_true():
    cab = make_cab(build_model("In", {"verbose": ("bool", False, False)}))
    assert build_argv(cab, {"verbose": True}) == ["tool", "--verbose"]
    assert build_argv(cab, {"verbose": False}) == ["tool"]


def test_list_param_joined():
    cab = make_cab(build_model("In", {"scales": ("list:int", False, None)}))
    assert build_argv(cab, {"scales": [0, 2, 4]}) == ["tool", "--scales", "0,2,4"]


def test_implicit_value_always_used_regardless_of_prepared_dict():
    cab = make_cab(
        build_model("In", {"mode": ("str", False, None)}),
        field_meta={"mode": ParamMeta(implicit="summary")},
    )
    assert build_argv(cab, {}) == ["tool", "--mode", "summary"]
    assert build_argv(cab, {"mode": "other"}) == ["tool", "--mode", "summary"]


def test_nom_de_guerre_used_as_flag_name():
    cab = make_cab(
        build_model("In", {"ms": ("MS", True, None)}),
        field_meta={"ms": ParamMeta(nom_de_guerre="vis")},
    )
    assert build_argv(cab, {"ms": "foo.ms"}) == ["tool", "--vis", "foo.ms"]


# -- command splitting / positional args (subcommand-style CLIs, e.g. `simms telsim`) --


def test_single_word_command_unchanged():
    cab = make_cab(build_model("In", {}))
    assert build_argv(cab, {})[0:1] == ["tool"]


def test_multiword_command_is_split_into_argv():
    cab = Cab(
        name="tool",
        command="simms telsim",
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {}),
    )
    assert build_argv(cab, {})[:2] == ["simms", "telsim"]


def test_positional_field_emitted_as_bare_value_last():
    cab = make_cab(
        build_model("In", {"vis": ("MS", True, None), "telescope": ("str", True, None)}),
        field_meta={"vis": ParamMeta(positional=True)},
    )
    argv = build_argv(cab, {"vis": "foo.ms", "telescope": "meerkat"})
    assert "--vis" not in argv
    assert "--telescope" in argv
    assert argv[-1] == "foo.ms"


def test_positional_with_multiword_command_end_to_end():
    cab = Cab(
        name="telsim",
        command="simms telsim",
        inputs_model=build_model("In", {"ms": ("MS", True, None), "telescope": ("str", True, None)}),
        outputs_model=build_model("Out", {}),
        field_meta={"ms": ParamMeta(positional=True)},
    )
    argv = build_argv(cab, {"ms": "sim.ms", "telescope": "meerkat"})
    assert argv[:2] == ["simms", "telsim"]
    assert argv[-1] == "sim.ms"
    assert "--ms" not in argv


def test_multiple_positionals_keep_field_declaration_order():
    cab = make_cab(
        build_model("In", {"first": ("str", True, None), "second": ("str", True, None)}),
        field_meta={"first": ParamMeta(positional=True), "second": ParamMeta(positional=True)},
    )
    argv = build_argv(cab, {"first": "a", "second": "b"})
    assert argv[-2:] == ["a", "b"]


# -- positional_head (real cult-cargo `cubical.yml`'s `parset: {policies:
# {positional_head: true}}` -- for tools like CubiCal/killMS whose own CLI
# only recognises a bare token as argv[1], never as a trailing leftover) --


def test_positional_head_emitted_before_every_flag():
    cab = make_cab(
        build_model("In", {"parset": ("File", False, None), "data_ms": ("MS", True, None)}),
        field_meta={"parset": ParamMeta(positional_head=True)},
    )
    argv = build_argv(cab, {"parset": "base.parset", "data_ms": "foo.ms"})
    assert argv == ["tool", "base.parset", "--data_ms", "foo.ms"]


def test_positional_head_omitted_when_none_optional():
    cab = make_cab(
        build_model("In", {"parset": ("File", False, None), "data_ms": ("MS", True, None)}),
        field_meta={"parset": ParamMeta(positional_head=True)},
    )
    argv = build_argv(cab, {"data_ms": "foo.ms"})
    assert argv == ["tool", "--data_ms", "foo.ms"]


def test_positional_head_and_positional_tail_bracket_the_flags():
    cab = make_cab(
        build_model("In", {"parset": ("File", False, None), "ms": ("MS", True, None), "telescope": ("str", True, None)}),
        field_meta={"parset": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True)},
    )
    argv = build_argv(cab, {"parset": "base.parset", "ms": "foo.ms", "telescope": "meerkat"})
    assert argv == ["tool", "base.parset", "--telescope", "meerkat", "foo.ms"]


# -- repeat_as_tokens (e.g. wsclean's "-size 4096 4096"/"-weight briggs 0") --


def test_repeat_as_tokens_flagged_emits_flag_once_then_bare_items():
    cab = make_cab(
        build_model("In", {"size": ("list:int", True, None)}),
        field_meta={"size": ParamMeta(repeat_as_tokens=True)},
    )
    argv = build_argv(cab, {"size": [4096, 4096]})
    assert argv == ["tool", "--size", "4096", "4096"]


def test_repeat_as_tokens_string_items():
    cab = make_cab(
        build_model("In", {"weight": ("list:str", True, None)}),
        field_meta={"weight": ParamMeta(repeat_as_tokens=True)},
    )
    argv = build_argv(cab, {"weight": ["briggs", "0"]})
    assert argv == ["tool", "--weight", "briggs", "0"]


def test_repeat_as_tokens_positional_emits_bare_items_no_flag():
    cab = make_cab(
        build_model("In", {"ms": ("list:MS", True, None)}),
        field_meta={"ms": ParamMeta(positional=True, repeat_as_tokens=True)},
    )
    argv = build_argv(cab, {"ms": ["a.ms", "b.ms"]})
    assert argv == ["tool", "a.ms", "b.ms"]
    assert "--ms" not in argv


def test_repeat_as_tokens_ignored_for_a_scalar_value():
    # repeat_as_tokens only kicks in for an actual list/tuple value --
    # a scalar falls through to the ordinary single --flag value emission.
    cab = make_cab(
        build_model("In", {"weight": ("str", True, None)}),
        field_meta={"weight": ParamMeta(repeat_as_tokens=True)},
    )
    argv = build_argv(cab, {"weight": "natural"})
    assert argv == ["tool", "--weight", "natural"]


def test_non_binary_flavour_rejected_before_building_argv():
    cab = Cab(
        name="tool",
        command="import os\nos.system('echo hi')",
        flavour="python-code",
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {}),
    )
    with pytest.raises(UnsupportedFlavourError):
        build_argv(cab, {})


# -- pattern-matched (dynamically-named) params, e.g. QuartiCal's K.type --


def make_quartical_like_cab() -> Cab:
    return Cab(
        name="quartical",
        command="goquartical",
        inputs_model=build_model("In", {"input_ms": ("MS", True, None)}, allow_extra=True),
        outputs_model=build_model("Out", {}),
        input_patterns=[
            ParamPattern(
                segments=[
                    ParamSegment(regex=r".+?"),
                    ParamSegment(
                        attrs={
                            "type": ParamMeta(),
                            "time_interval": ParamMeta(),
                            "solvable": ParamMeta(),
                        }
                    ),
                ]
            )
        ],
    )


def test_pattern_matched_param_is_built():
    cab = make_quartical_like_cab()
    argv = build_argv(cab, {"input_ms": "foo.ms", "K.type": "diag_complex", "G.time_interval": 10})
    assert argv[0] == "goquartical"
    assert argv[argv.index("--K.type") + 1] == "diag_complex"
    assert argv[argv.index("--G.time_interval") + 1] == "10"


def test_pattern_matched_bool_param_is_a_flag():
    cab = make_quartical_like_cab()
    argv = build_argv(cab, {"input_ms": "foo.ms", "K.solvable": True})
    assert "--K.solvable" in argv
    assert argv[-1] == "--K.solvable"


def test_unmatched_dynamic_name_is_not_emitted():
    cab = make_quartical_like_cab()
    argv = build_argv(cab, {"input_ms": "foo.ms", "K.nonexistent_attr": "x"})
    assert "--K.nonexistent_attr" not in argv


# -- key_value / bracket-list policies (real quartical.yml shape:
# policies: {key_value: true, repeat: '[]', prefix: ''}) --


def make_quartical_key_value_cab() -> Cab:
    return Cab(
        name="quartical",
        command="goquartical",
        inputs_model=build_model(
            "In",
            {
                "input_ms.path": ("MS", True, None),
                "input_ms.data_column": ("str", False, "DATA"),
                "solver.terms": ("list:str", False, None),
                "dask.scheduler": ("bool", False, None),
            },
        ),
        outputs_model=build_model("Out", {}),
        policies=Policies(prefix="", key_value=True, repeat="[]"),
    )


def test_key_value_policy_emits_single_equals_token():
    cab = make_quartical_key_value_cab()
    argv = build_argv(cab, {"input_ms.path": "foo.ms", "input_ms.data_column": "DATA"})
    assert "input_ms.path=foo.ms" in argv
    assert "input_ms.data_column=DATA" in argv
    # never the two-token --flag/value shape, and no bare "input_ms.path"
    # token separate from its value either
    assert "--input_ms.path" not in argv
    assert "input_ms.path" not in argv


def test_key_value_policy_formats_list_as_bracket_literal():
    cab = make_quartical_key_value_cab()
    argv = build_argv(cab, {"input_ms.path": "foo.ms", "solver.terms": ["K", "G"]})
    assert "solver.terms=[K,G]" in argv


def test_key_value_policy_formats_bool_as_true_false_not_a_bare_flag():
    cab = make_quartical_key_value_cab()
    argv = build_argv(cab, {"input_ms.path": "foo.ms", "dask.scheduler": True})
    assert "dask.scheduler=true" in argv
    argv = build_argv(cab, {"input_ms.path": "foo.ms", "dask.scheduler": False})
    assert "dask.scheduler=false" in argv


def test_key_value_policy_applies_to_pattern_matched_dynamic_fields_too():
    cab = Cab(
        name="quartical",
        command="goquartical",
        inputs_model=build_model("In", {"input_ms.path": ("MS", True, None)}, allow_extra=True),
        outputs_model=build_model("Out", {}),
        policies=Policies(prefix="", key_value=True, repeat="[]"),
        input_patterns=[
            ParamPattern(
                segments=[ParamSegment(regex=r".+?"), ParamSegment(attrs={"type": ParamMeta()})]
            )
        ],
    )
    argv = build_argv(cab, {"input_ms.path": "foo.ms", "K.type": "diag_complex"})
    assert "K.type=diag_complex" in argv


def test_loading_real_quartical_policies_dict_preserves_key_value_and_repeat():
    """Regression test for the actual bug: `Policies(**policies_spec)` used
    to silently drop `key_value`/`repeat` since the model had no such
    fields (pydantic's default extra="ignore"), even though real
    quartical.yml declares `policies: {key_value: true, repeat: '[]',
    prefix: ''}` -- the exact info needed to build correct argv.
    """
    policies = Policies(**{"key_value": True, "repeat": "[]", "prefix": ""})
    assert policies.key_value is True
    assert policies.repeat == "[]"
    assert policies.prefix == ""


# -- explicit_true / explicit_false (real cubical.yml shape: policies:
# {prefix: '--', explicit_true: true, explicit_false: false}) --


def make_cubical_like_cab() -> Cab:
    return Cab(
        name="cubical",
        command="gocubical",
        inputs_model=build_model(
            "In", {"out_overwrite": ("bool", False, None), "out_derotate": ("bool", False, None)}, allow_extra=True
        ),
        outputs_model=build_model("Out", {}),
        policies=Policies(prefix="--", explicit_true=True, explicit_false=False),
        input_patterns=[
            ParamPattern(
                separator="-",
                segments=[ParamSegment(regex=r".+?"), ParamSegment(attrs={"solvable": ParamMeta()})],
            )
        ],
    )


def test_explicit_true_emits_flag_and_true_token_not_a_bare_flag():
    cab = make_cubical_like_cab()
    argv = build_argv(cab, {"out_overwrite": True})
    assert "--out_overwrite" in argv
    assert argv[argv.index("--out_overwrite") + 1] == "true"


def test_explicit_false_default_still_omits_the_flag_entirely():
    """cubical.yml's real explicit_false: false -- a False value should
    still just be omitted, not emitted as `--flag false`, since only
    explicit_true is set.
    """
    cab = make_cubical_like_cab()
    argv = build_argv(cab, {"out_overwrite": False})
    assert "--out_overwrite" not in argv
    assert "false" not in argv


def test_explicit_true_applies_to_pattern_matched_dynamic_fields_too():
    cab = make_cubical_like_cab()
    argv = build_argv(cab, {"g-solvable": True})
    assert "--g-solvable" in argv
    assert argv[argv.index("--g-solvable") + 1] == "true"


def test_explicit_false_policy_emits_flag_and_false_token_when_enabled():
    cab = Cab(
        name="t",
        command="t",
        inputs_model=build_model("In", {"flag": ("bool", False, None)}),
        outputs_model=build_model("Out", {}),
        policies=Policies(prefix="--", explicit_false=True),
    )
    argv = build_argv(cab, {"flag": False})
    assert "--flag" in argv
    assert argv[argv.index("--flag") + 1] == "false"


def test_loading_real_cubical_policies_dict_preserves_explicit_true_and_false():
    """Regression test for the actual bug this was fixing:
    `Policies(**policies_spec)` used to silently drop `explicit_true`/
    `explicit_false` since the model had no such fields (pydantic's
    default extra="ignore"), even though real cubical.yml declares
    `policies: {prefix: '--', explicit_true: true, explicit_false:
    false}` -- causing every boolean to build as a bare/omitted flag,
    which real gocubical's optparse-derived CLI doesn't tolerate (a bare
    `--flag` with no value corrupts parsing of everything after it).
    """
    policies = Policies(**{"prefix": "--", "explicit_true": True, "explicit_false": False})
    assert policies.explicit_true is True
    assert policies.explicit_false is False
