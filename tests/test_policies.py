import pytest

from shinobi.exceptions import UnsupportedFlavourError
from shinobi.loaders._modelgen import build_model
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern, ParamSegment


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
