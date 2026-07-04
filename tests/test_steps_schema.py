import pytest
from pydantic import BaseModel

from shinobi.steps import (
    Cab,
    InputRef,
    Mutability,
    OutputRef,
    Recipe,
    Scope,
    StepRef,
    step,
)


class Inputs(BaseModel):
    x: int = 0


class Outputs(BaseModel):
    y: int = 0


def make_cab(**kwargs) -> Cab:
    return Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, **kwargs)


def test_cab_is_a_scope_and_stores_model_classes():
    cab = make_cab()
    assert isinstance(cab, Scope)
    assert cab.inputs_model is Inputs
    assert cab.outputs_model is Outputs
    assert cab.inputs_model(x=5).x == 5


def test_default_mutability_is_immutable():
    assert make_cab().mutability_of("x") is Mutability.IMMUTABLE


def test_explicit_mutable_entry_is_respected():
    cab = make_cab(input_mutability={"x": Mutability.MUTABLE})
    assert cab.mutability_of("x") is Mutability.MUTABLE


def test_backends_default_to_none():
    assert make_cab().backend is None
    assert Recipe(name="r", inputs_model=Inputs, outputs_model=Outputs).backend is None


def test_recipe_steps_and_output_wiring_round_trip():
    cab = make_cab()
    recipe = Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    assert recipe.steps[0].name == "a"
    assert recipe.steps[0].step is cab  # concrete Cab preserved, not coerced to Scope
    assert recipe.output_wiring["y"].step == "a"


def test_stepref_carries_func_and_params():
    cab = make_cab()

    def orchestrate(ctx):
        return None

    ref = StepRef(name="a", step=cab, func=orchestrate, params={"x": 7})
    assert ref.func is orchestrate
    assert ref.params == {"x": 7}
    assert isinstance(ref.step, Cab)


def test_decorator_returns_a_stepref_with_the_func_attached():
    cab = make_cab()

    @step(scope=cab)
    def my_step(ctx):
        return None

    assert isinstance(my_step, StepRef)
    assert my_step.name == "my_step"
    assert my_step.func is not None
    assert my_step.step is cab


def test_decorator_backend_binds_a_copy_and_never_mutates_the_scope():
    cab = make_cab()

    @step(scope=cab, backend="native")
    def my_step(ctx):
        return None

    assert my_step.step is not cab
    assert my_step.step.backend == "native"
    assert cab.backend is None


def test_two_functions_over_one_scope_get_independent_steprefs():
    cab = make_cab()

    @step(scope=cab)
    def a(ctx):
        return None

    @step(scope=cab)
    def b(ctx):
        return None

    assert a is not b
    assert a.func is not b.func
    assert a.step is cab and b.step is cab


def test_inputref_and_outputref_are_minimal():
    assert InputRef(field="x").field == "x"
    out = OutputRef(step="a", field="y")
    assert out.step == "a"
    assert out.field == "y"


# -- wiring proxies --


def test_inputs_proxy_validates_field_names():
    recipe = Recipe(name="r", inputs_model=Inputs, outputs_model=Outputs)
    assert recipe.inputs.x == InputRef(field="x")
    assert recipe.inputs("x") == InputRef(field="x")
    with pytest.raises(AttributeError):
        recipe.inputs.nope


def test_outputs_proxy_validates_step_and_field_names():
    cab = make_cab()
    recipe = Recipe(name="r", inputs_model=Inputs, outputs_model=Outputs)
    recipe.add_step("a", cab)
    assert recipe.outputs.a.y == OutputRef(step="a", field="y")
    assert recipe.outputs("a", "y") == OutputRef(step="a", field="y")
    with pytest.raises(AttributeError):
        recipe.outputs.missing_step
    with pytest.raises(AttributeError):
        recipe.outputs.a.not_a_field


def test_builder_splits_wiring_from_params():
    cab = make_cab()
    recipe = Recipe(name="r", inputs_model=Inputs, outputs_model=Outputs)
    recipe.add_step("a", cab, x=recipe.inputs.x)
    recipe.add_step("b", cab, x=99)
    assert recipe.steps[0].wiring == {"x": InputRef(field="x")}
    assert recipe.steps[0].params == {}
    assert recipe.steps[1].wiring == {}
    assert recipe.steps[1].params == {"x": 99}
