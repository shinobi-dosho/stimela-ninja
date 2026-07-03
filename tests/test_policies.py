import pytest

from shinobi.exceptions import ParameterError, UnsupportedFlavourError
from shinobi.policies import build_args, build_argv, resolve_params
from shinobi.schema import CabDef, ParamPattern, ParamSchema


def make_cab(**inputs: ParamSchema) -> CabDef:
    return CabDef(name="tool", command="tool", inputs=inputs)


def test_required_param_missing_raises():
    cab = make_cab(size=ParamSchema(dtype="int", required=True))
    with pytest.raises(ParameterError):
        build_args(cab, {})


def test_unknown_param_raises():
    cab = make_cab(size=ParamSchema(dtype="int"))
    with pytest.raises(ParameterError):
        build_args(cab, {"bogus": 1})


def test_default_is_used():
    cab = make_cab(size=ParamSchema(dtype="int", default=1024))
    assert build_args(cab, {}) == ["tool", "--size", "1024"]


def test_bool_flag_only_emitted_when_true():
    cab = make_cab(verbose=ParamSchema(dtype="bool"))
    assert build_args(cab, {"verbose": True}) == ["tool", "--verbose"]
    assert build_args(cab, {"verbose": False}) == ["tool"]


def test_list_param_joined():
    cab = make_cab(scales=ParamSchema(dtype="list:int"))
    assert build_args(cab, {"scales": [0, 2, 4]}) == ["tool", "--scales", "0,2,4"]


def test_implicit_cannot_be_overridden():
    cab = make_cab(mode=ParamSchema(dtype="str", implicit="summary"))
    with pytest.raises(ParameterError):
        build_args(cab, {"mode": "other"})
    assert build_args(cab, {}) == ["tool", "--mode", "summary"]


def test_nom_de_guerre_used_as_flag_name():
    cab = make_cab(ms=ParamSchema(dtype="MS", nom_de_guerre="vis", required=True))
    assert build_args(cab, {"ms": "foo.ms"}) == ["tool", "--vis", "foo.ms"]


def test_non_binary_flavour_rejected_before_building_argv():
    # a "python-code"/"casa-task" cab's `command` is inline source or a
    # dotted function reference, not an executable name -- must never
    # reach subprocess as argv[0]
    cab = CabDef(name="tool", command="import os\nos.system('echo hi')", flavour="python-code")
    with pytest.raises(UnsupportedFlavourError):
        build_argv(cab, resolve_params(cab, {}))


# -- pattern-matched (dynamically-named) params, e.g. QuartiCal's K.type --


def make_quartical_like_cab() -> CabDef:
    return CabDef(
        name="quartical",
        command="goquartical",
        inputs={"input_ms": ParamSchema(dtype="MS", required=True)},
        input_patterns=[
            ParamPattern(
                attrs={
                    "type": ParamSchema(dtype="str"),
                    "time_interval": ParamSchema(dtype="int"),
                    "solvable": ParamSchema(dtype="bool"),
                }
            )
        ],
    )


def test_pattern_matched_param_is_resolved_and_built():
    cab = make_quartical_like_cab()
    argv = build_args(cab, {"input_ms": "foo.ms", "K.type": "diag_complex", "G.time_interval": 10})
    assert argv[0] == "goquartical"
    assert "--K.type" in argv
    assert argv[argv.index("--K.type") + 1] == "diag_complex"
    assert "--G.time_interval" in argv
    assert argv[argv.index("--G.time_interval") + 1] == "10"


def test_pattern_matched_bool_param_is_a_flag():
    cab = make_quartical_like_cab()
    argv = build_args(cab, {"input_ms": "foo.ms", "K.solvable": True})
    assert "--K.solvable" in argv
    # a bool flag has no separate value token following it
    assert argv[-1] == "--K.solvable"


def test_unmatched_dynamic_name_still_rejected():
    cab = make_quartical_like_cab()
    with pytest.raises(ParameterError):
        build_args(cab, {"input_ms": "foo.ms", "K.nonexistent_attr": "x"})


def test_name_with_no_prefix_before_separator_is_not_a_pattern_match():
    cab = make_quartical_like_cab()
    with pytest.raises(ParameterError):
        build_args(cab, {"input_ms": "foo.ms", ".type": "x"})
