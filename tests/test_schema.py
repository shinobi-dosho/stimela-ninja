from shinobi.schema import CabDef, ParamPattern, ParamSchema, Policies


def test_cabdef_defaults():
    cab = CabDef(name="echo", command="/bin/echo")
    assert cab.flavour == "binary"
    assert cab.inputs == {}
    assert cab.policies.prefix == "--"


def test_param_name_uses_nom_de_guerre():
    cab = CabDef(
        name="tool",
        command="tool",
        inputs={"ms": ParamSchema(dtype="MS", nom_de_guerre="vis")},
    )
    assert cab.param_name("ms", cab.inputs["ms"]) == "vis"


def test_policies_arg_name_replace():
    policies = Policies(replace={"_": "-"})
    assert policies.arg_name("image_size") == "--image-size"


def test_param_pattern_matches_prefix_dot_attr():
    pattern = ParamPattern(attrs={"type": ParamSchema(dtype="str")})
    assert pattern.matches("K.type") is not None
    assert pattern.matches("K.type").dtype == "str"


def test_param_pattern_rejects_unknown_attr():
    pattern = ParamPattern(attrs={"type": ParamSchema(dtype="str")})
    assert pattern.matches("K.bogus") is None


def test_param_pattern_rejects_names_without_separator():
    pattern = ParamPattern(attrs={"type": ParamSchema(dtype="str")})
    assert pattern.matches("type") is None


def test_param_pattern_rejects_empty_prefix():
    pattern = ParamPattern(attrs={"type": ParamSchema(dtype="str")})
    assert pattern.matches(".type") is None


def test_param_pattern_custom_separator():
    pattern = ParamPattern(separator="-", attrs={"solvable": ParamSchema(dtype="bool")})
    assert pattern.matches("g1-solvable") is not None
    assert pattern.matches("g1.solvable") is None


def test_param_pattern_attr_containing_the_separator():
    # regression: cubical's real attrs include "time-int"/"freq-int",
    # which themselves contain the "-" separator -- a blind rpartition()
    # would incorrectly split "g1-time-int" into prefix="g1-time",
    # attr="int" instead of prefix="g1", attr="time-int"
    pattern = ParamPattern(
        separator="-",
        attrs={"time-int": ParamSchema(dtype="int"), "solvable": ParamSchema(dtype="bool")},
    )
    schema = pattern.matches("g1-time-int")
    assert schema is not None
    assert schema.dtype == "int"


def test_cabdef_match_pattern_checks_all_patterns_in_order():
    cab = CabDef(
        name="tool",
        command="tool",
        input_patterns=[
            ParamPattern(attrs={"type": ParamSchema(dtype="str")}),
            ParamPattern(separator="-", attrs={"solvable": ParamSchema(dtype="bool")}),
        ],
    )
    assert cab.match_pattern("K.type") is not None
    assert cab.match_pattern("g1-solvable") is not None
    assert cab.match_pattern("nope") is None
