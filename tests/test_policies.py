import pytest

from shinobi.exceptions import ParameterError
from shinobi.policies import build_args
from shinobi.schema import CabDef, ParamSchema


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
