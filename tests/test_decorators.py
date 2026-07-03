from shinobi.decorators import cab, recipe
from shinobi.recipe import call
from shinobi.schema import CabDef, ParamPattern, ParamSchema, RecipeInfo


def test_cab_decorator_produces_a_cabdef():
    @cab("breizorro", image="breizorro:latest")
    def breizorro(restored_image: str, threshold: float = 6.5):
        """Mask creation and manipulation for radio astronomy images."""

    assert isinstance(breizorro, CabDef)
    assert breizorro.name == "breizorro"
    assert breizorro.command == "breizorro"
    assert breizorro.image == "breizorro:latest"
    assert breizorro.info == "Mask creation and manipulation for radio astronomy images."


def test_dtype_inferred_from_annotations():
    @cab("tool")
    def tool(name: str, size: int, threshold: float, verbose: bool, scales: list[int]):
        pass

    assert tool.inputs["name"].dtype == "str"
    assert tool.inputs["size"].dtype == "int"
    assert tool.inputs["threshold"].dtype == "float"
    assert tool.inputs["verbose"].dtype == "bool"
    assert tool.inputs["scales"].dtype == "list:int"


def test_no_default_means_required_default_means_not():
    @cab("tool")
    def tool(ms: str, threshold: float = 6.5):
        pass

    assert tool.inputs["ms"].required is True
    assert tool.inputs["ms"].default is None
    assert tool.inputs["threshold"].required is False
    assert tool.inputs["threshold"].default == 6.5


def test_missing_annotation_defaults_to_str():
    @cab("tool")
    def tool(anything):
        pass

    assert tool.inputs["anything"].dtype == "str"


def test_inputs_override_replaces_derived_entry():
    @cab(
        "tool",
        inputs={"ms": ParamSchema(dtype="MS", required=True, nom_de_guerre="vis")},
    )
    def tool(ms: str = "unused"):
        pass

    assert tool.inputs["ms"].nom_de_guerre == "vis"
    assert tool.inputs["ms"].dtype == "MS"


def test_outputs_and_wranglers_pass_through():
    @cab(
        "tool",
        outputs={"mask": ParamSchema(dtype="File", required=True)},
        wranglers={r"done: (?P<n>\d+)": ["PARSE_OUTPUT:n:int"]},
    )
    def tool():
        pass

    assert tool.outputs["mask"].dtype == "File"
    assert len(tool.wranglers) == 1


def test_decorated_cab_is_interchangeable_with_call(native):
    @cab("/bin/echo")
    def greet(text: str = "hello from a python-native cab"):
        pass

    result = call(greet, native)
    assert result.success
    assert "hello from a python-native cab" in result.stdout


# -- @recipe: unlike @cab, must stay a directly-callable plain function --


def test_recipe_decorator_does_not_replace_function():
    @recipe()
    def selfcal(ms: str, threshold: float = 6.5):
        """Run selfcal."""
        return (ms, threshold)

    assert callable(selfcal)
    assert selfcal("data.ms") == ("data.ms", 6.5)
    assert selfcal("data.ms", threshold=7.0) == ("data.ms", 7.0)


def test_recipe_metadata_attached():
    @recipe(name="my-recipe")
    def foo(ms: str, threshold: float = 6.5):
        """Docs here."""

    info = foo.__shinobi_recipe__
    assert isinstance(info, RecipeInfo)
    assert info.name == "my-recipe"
    assert info.info == "Docs here."
    assert info.inputs["ms"].required is True
    assert info.inputs["threshold"].default == 6.5


def test_recipe_name_defaults_to_function_name():
    @recipe()
    def selfcal():
        pass

    assert selfcal.__shinobi_recipe__.name == "selfcal"


def test_recipe_dtype_inferred_from_annotations():
    @recipe()
    def tool(name: str, size: int, scales: list[int]):
        pass

    inputs = tool.__shinobi_recipe__.inputs
    assert inputs["name"].dtype == "str"
    assert inputs["size"].dtype == "int"
    assert inputs["scales"].dtype == "list:int"


def test_recipe_inputs_override_replaces_derived_entry():
    @recipe(inputs={"ms": ParamSchema(dtype="MS", required=True)})
    def tool(ms: str = "unused"):
        pass

    assert tool.__shinobi_recipe__.inputs["ms"].dtype == "MS"


def test_cab_input_patterns_passthrough():
    @cab(
        "goquartical",
        input_patterns=[ParamPattern(attrs={"type": ParamSchema(dtype="str")})],
    )
    def quartical(input_ms: str):
        pass

    assert quartical.match_pattern("K.type") is not None
    assert quartical.match_pattern("K.bogus") is None
