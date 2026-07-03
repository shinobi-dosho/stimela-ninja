from shinobi.schema import CabDef, ParamSchema, Policies


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
