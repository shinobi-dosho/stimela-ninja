import pytest
from pydantic import BaseModel, ValidationError

from shinobi.config import AppConfig
from shinobi.steps import CabDef, InputRef, OutputRef, RecipeDef, RecordingStepBackend, StepRef
from shinobi.steps.dispatch import register_step_backend, run_step
from tests.fixtures.sample_steps import (
    append_to_immutable,
    append_to_mutable,
    chained_recipe,
    make_value_cab,
    use_value_cab,
)


class Inputs(BaseModel):
    x: int


class Outputs(BaseModel):
    y: str | None = None


def make_recording_cab(**kwargs) -> tuple[CabDef, RecordingStepBackend]:
    recorder = RecordingStepBackend()
    register_step_backend("record", recorder)
    cab = CabDef(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record", **kwargs
    )
    return cab, recorder


def test_missing_required_field_raises_before_backend_runs():
    cab, recorder = make_recording_cab()
    with pytest.raises(ValidationError):
        run_step(cab, None)
    assert recorder.calls == []


def test_immutable_input_mutation_does_not_propagate_to_caller():
    original = [1, 2, 3]
    append_to_immutable(items=original)
    assert original == [1, 2, 3]


def test_mutable_input_mutation_does_propagate_to_caller():
    original = [1, 2, 3]
    append_to_mutable(items=original)
    assert original == [1, 2, 3, 99]


def test_function_overrides_are_merged_before_dispatch():
    cab, recorder = make_recording_cab()
    run_step(cab, lambda x: {"x": x + 1}, x=1)
    _, inputs = recorder.calls[0]
    assert inputs.x == 2


def test_bad_override_raises_at_the_step_boundary():
    cab, recorder = make_recording_cab()
    with pytest.raises(ValidationError):
        run_step(cab, lambda x: {"x": "not an int"}, x=1)
    assert recorder.calls == []


def test_cabdef_dispatch_calls_the_resolved_backend_with_final_inputs():
    cab, recorder = make_recording_cab()
    run_step(cab, None, x=5)
    defn, inputs = recorder.calls[0]
    assert defn is cab
    assert inputs.x == 5


def test_recipedef_dispatch_runs_substeps_end_to_end():
    result = run_step(chained_recipe, None, name="whatever.txt")
    assert result.ok is True


def test_recipedef_wires_a_real_output_value_into_the_next_steps_input():
    # chained_recipe's real cabs run via /bin/echo, whose captured stdout
    # doesn't happen to populate "path" -- this test controls the first
    # step's output directly, to prove a genuine, non-None Python value
    # (not just "no error occurred") flows from one step's output into
    # the next step's input.
    class MakeInputs(BaseModel):
        name: str = "x"

    class PathOut(BaseModel):
        path: str | None = None

    class UseInputs(BaseModel):
        path: str | None = None

    class OkOut(BaseModel):
        ok: bool = True

    class FixedOutputBackend:
        def __init__(self, outputs: dict):
            self.outputs = outputs

        def run(self, defn, inputs):
            return self.outputs

    use_recorder = RecordingStepBackend()
    register_step_backend("fixed-output", FixedOutputBackend({"path": "/tmp/real-value.txt"}))
    register_step_backend("use-recorder", use_recorder)

    make_cab = CabDef(
        name="make", command="x", inputs_model=MakeInputs, outputs_model=PathOut, backend="fixed-output"
    )
    use_cab = CabDef(
        name="use", command="x", inputs_model=UseInputs, outputs_model=OkOut, backend="use-recorder"
    )
    recipe = RecipeDef(
        name="r",
        inputs_model=MakeInputs,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make_cab, wiring={"name": InputRef(field="name")}),
            StepRef(name="use", step=use_cab, wiring={"path": OutputRef(step="make", field="path")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )

    run_step(recipe, None, name="whatever")

    _, use_inputs = use_recorder.calls[0]
    assert use_inputs.path == "/tmp/real-value.txt"


def test_recipedef_output_wiring_surfaces_substep_output():
    # use_value_cab's OkOutputs.ok defaults to True (RecordingStepBackend-
    # style backends return {}); confirm the recipe's own output field is
    # actually sourced via output_wiring, not just coincidentally present
    assert "ok" in chained_recipe.output_wiring
    assert chained_recipe.output_wiring["ok"].step == "use"
    assert chained_recipe.output_wiring["ok"].field == "ok"


# -- backend resolution priority --


def test_cabdef_backend_wins_over_recipe_backend():
    cab_recorder = RecordingStepBackend()
    recipe_recorder = RecordingStepBackend()
    register_step_backend("cab-choice", cab_recorder)
    register_step_backend("recipe-choice", recipe_recorder)

    cab = CabDef(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="cab-choice"
    )
    recipe = RecipeDef(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="recipe-choice",
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    run_step(recipe, None, x=1)
    assert len(cab_recorder.calls) == 1
    assert recipe_recorder.calls == []


def test_recipe_backend_wins_over_appconfig_default_when_cab_has_none():
    recipe_recorder = RecordingStepBackend()
    register_step_backend("recipe-only-choice", recipe_recorder)

    cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)  # no backend
    recipe = RecipeDef(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="recipe-only-choice",
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    run_step(recipe, None, x=1)
    assert len(recipe_recorder.calls) == 1


def test_falls_through_to_appconfig_default_when_neither_declares_a_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__DEFAULT", raising=False)
    fallback_recorder = RecordingStepBackend()
    register_step_backend("native", fallback_recorder)  # temporarily shadow "native"
    try:
        cab = CabDef(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
        assert AppConfig.load(config_file=tmp_path / "missing.yml").backend.default == "native"
        run_step(cab, None, x=1)
        assert len(fallback_recorder.calls) == 1
    finally:
        from shinobi.steps.backend import NativeStepBackend

        register_step_backend("native", NativeStepBackend())


def test_make_value_and_use_value_cabs_are_reused_from_fixtures():
    # sanity check that the fixtures module's cabs used above are the same
    # ones the RecipeDef in chained_recipe actually wires together
    assert chained_recipe.steps[0].step is make_value_cab
    assert chained_recipe.steps[1].step is use_value_cab
