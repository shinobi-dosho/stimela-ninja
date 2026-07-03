from pydantic import BaseModel

from shinobi.steps import CabDef, RecipeDef, RecordingStepBackend, Step, register_step_backend, step


class Inputs(BaseModel):
    x: int = 0


class Outputs(BaseModel):
    y: str | None = None


def make_recording_cab(**kwargs) -> tuple[CabDef, RecordingStepBackend]:
    recorder = RecordingStepBackend()
    register_step_backend("record", recorder)
    cab = CabDef(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record", **kwargs
    )
    return cab, recorder


def test_step_stays_callable_and_preserves_name_and_docstring():
    cab, _ = make_recording_cab()

    @step(cab)
    def my_step(x: int):
        """Docs."""
        return None

    assert isinstance(my_step, Step)
    assert callable(my_step)
    assert my_step.__name__ == "my_step"
    assert my_step.__doc__ == "Docs."


def test_step_with_passthrough_function_dispatches_inputs_unchanged():
    cab, recorder = make_recording_cab()

    @step(cab)
    def my_step(x: int):
        return None

    my_step(x=5)
    assert len(recorder.calls) == 1
    _, inputs = recorder.calls[0]
    assert inputs.x == 5


def test_step_function_returned_overrides_change_what_reaches_backend():
    cab, recorder = make_recording_cab()

    @step(cab)
    def my_step(x: int):
        return {"x": x * 10}

    my_step(x=5)
    _, inputs = recorder.calls[0]
    assert inputs.x == 50


def test_step_wraps_recipedef_the_same_way():
    recipe = RecipeDef(name="r", inputs_model=Inputs, outputs_model=Outputs)

    @step(recipe)
    def my_recipe_step(x: int):
        return None

    assert isinstance(my_recipe_step, Step)
    assert my_recipe_step.defn is recipe


def test_schema_always_comes_from_defn_not_function_signature():
    # the decorated function's own signature is irrelevant to validation --
    # defn.inputs_model governs, full stop, even with an unrelated **kwargs
    # signature that couldn't itself express any schema
    cab, recorder = make_recording_cab()

    @step(cab)
    def my_step(**kwargs):
        return None

    my_step(x=7)
    _, inputs = recorder.calls[0]
    assert inputs.x == 7
