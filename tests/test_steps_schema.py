from pydantic import BaseModel

from shinobi.steps import CabDef, InputRef, Mutability, OutputRef, RecipeDef, Step, StepRef, step


class Inputs(BaseModel):
    x: int = 0


class Outputs(BaseModel):
    y: int = 0


def test_cabdef_stores_inputs_outputs_model_classes():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    assert cab.inputs_model is Inputs
    assert cab.outputs_model is Outputs
    assert cab.inputs_model(x=5).x == 5


def test_default_mutability_is_immutable():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    assert cab.mutability_of("x") is Mutability.IMMUTABLE


def test_explicit_mutable_entry_is_respected():
    cab = CabDef(
        name="tool",
        command="tool",
        inputs_model=Inputs,
        outputs_model=Outputs,
        input_mutability={"x": Mutability.MUTABLE},
    )
    assert cab.mutability_of("x") is Mutability.MUTABLE


def test_cabdef_backend_defaults_to_none():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    assert cab.backend is None


def test_recipedef_backend_defaults_to_none():
    recipe = RecipeDef(name="r", inputs_model=Inputs, outputs_model=Outputs)
    assert recipe.backend is None


def test_recipedef_steps_and_output_wiring_round_trip():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    recipe = RecipeDef(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    assert recipe.steps[0].name == "a"
    assert recipe.steps[0].step is cab
    assert recipe.output_wiring["y"].step == "a"
    assert recipe.output_wiring["y"].field == "y"


def test_stepref_accepts_bare_cabdef():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    ref = StepRef(name="a", step=cab)
    assert isinstance(ref.step, CabDef)


def test_stepref_accepts_bare_recipedef():
    recipe = RecipeDef(name="r", inputs_model=Inputs, outputs_model=Outputs)
    ref = StepRef(name="a", step=recipe)
    assert isinstance(ref.step, RecipeDef)


def test_stepref_accepts_decorated_step():
    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)

    @step(cab)
    def tool_step(x: int):
        return None

    ref = StepRef(name="a", step=tool_step)
    assert isinstance(ref.step, Step)


def test_inputref_and_outputref_are_minimal():
    assert InputRef(field="x").field == "x"
    out = OutputRef(step="a", field="y")
    assert out.step == "a"
    assert out.field == "y"
