import pytest
from pydantic import BaseModel

from shinobi.backends.recording import RecordingBackend
from shinobi.results import StepResult
from shinobi.steps import Cab, Recipe, StepRef, register_step_backend, step


class Inputs(BaseModel):
    x: int = 0


class Outputs(BaseModel):
    y: str | None = None


def make_recording_cab(**kwargs) -> tuple[Cab, RecordingBackend]:
    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    cab = Cab(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record", **kwargs
    )
    return cab, recorder


def test_decorator_returns_a_callable_stepref():
    cab, _ = make_recording_cab()

    @step(scope=cab)
    def my_step(ctx):
        """Docs."""
        return None

    assert isinstance(my_step, StepRef)
    assert callable(my_step)
    assert my_step.name == "my_step"


def test_passthrough_function_dispatches_inputs_unchanged():
    cab, recorder = make_recording_cab()

    @step(scope=cab)
    def my_step(ctx):
        return None

    my_step(x=5)
    assert len(recorder.calls) == 1
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 5


def test_ctx_run_overrides_change_what_reaches_backend():
    cab, recorder = make_recording_cab()

    @step(scope=cab)
    def my_step(ctx):
        return ctx.run(x=ctx.inputs.x * 10)

    my_step(x=5)
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 50


def test_per_step_params_are_merged_under_caller_kwargs():
    cab, recorder = make_recording_cab()

    @step(scope=cab, x=3)
    def my_step(ctx):
        return None

    my_step()  # no kwargs -> param default used
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 3

    my_step(x=7)  # caller kwarg wins over the param
    _, _, inputs = recorder.calls[1]
    assert inputs["x"] == 7


def test_returning_a_stepresult_passes_it_through():
    cab, recorder = make_recording_cab()

    @step(scope=cab)
    def my_step(ctx):
        result = ctx.run(x=1)
        return result

    out = my_step(x=9)
    assert isinstance(out, StepResult)
    assert out.inputs.x == 1


def test_returning_non_stepresult_raises_typeerror():
    cab, _ = make_recording_cab()

    @step(scope=cab)
    def my_step(ctx):
        return 42

    with pytest.raises(TypeError):
        my_step(x=1)


def test_decorator_wraps_a_recipe_the_same_way():
    recipe = Recipe(name="r", inputs_model=Inputs, outputs_model=Outputs)

    @step(scope=recipe)
    def my_recipe_step(ctx):
        return None

    assert isinstance(my_recipe_step, StepRef)
    assert my_recipe_step.step is recipe


def test_same_named_functions_in_two_recipes_do_not_collide():
    cab_a, rec_a = make_recording_cab()
    cab_b = Cab(name="tool_b", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record")

    r1 = Recipe(name="r1", inputs_model=Inputs, outputs_model=Outputs)
    r2 = Recipe(name="r2", inputs_model=Inputs, outputs_model=Outputs)

    @r1.step(scope=cab_a)
    def build(ctx):
        return ctx.run(x=1)

    @r2.step(scope=cab_b)
    def build(ctx):  # noqa: F811 -- deliberately same name, different recipe
        return ctx.run(x=2)

    assert r1.steps[0].func is not r2.steps[0].func
    assert r1.steps[0].step is cab_a
    assert r2.steps[0].step is cab_b
