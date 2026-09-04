"""Tests for `@shinobi.pystep` (src/shinobi/steps/pyfunc.py)."""

from typing import Annotated

import click
import pytest
from click.testing import CliRunner
from pydantic import BaseModel, Field

from shinobi.backends.recording import RecordingBackend
from shinobi.clickutil import build_options
from shinobi.steps import InputRef, Recipe, StepRef, pystep, register_step_backend
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import ParamMeta
from .fixtures.sample_steps import use_value_cab

# Module-level models -- required for typing.get_type_hints to resolve them.


class OffsetOutputs(BaseModel):
    shifted: str


def add_offset(ms: str, offset: float = 0.0) -> OffsetOutputs:
    return OffsetOutputs(shifted=f"{ms}+{offset}")


def choose_mode(mode: Annotated[str, ParamMeta(choices=["analytic", "subsample", "none"])] = "analytic") -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


def choose_mode_from_field(mode: str = Field("analytic", json_schema_extra={"param_meta": ParamMeta(choices=["analytic", "subsample", "none"])})) -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


def test_inputs_model_derived_from_signature_required_and_optional():
    ref = pystep()(add_offset)
    fields = ref.step.inputs_model.model_fields
    assert set(fields) == {"ms", "offset"}
    assert fields["ms"].is_required()
    assert not fields["offset"].is_required()
    assert fields["offset"].default == 0.0


def test_annotated_param_meta_choices_validate_and_build_click_choice():
    ref = pystep()(choose_mode)
    field = ref.step.inputs_model.model_fields["mode"]

    assert ref.step.field_meta["mode"].choices == ["analytic", "subsample", "none"]
    assert field.json_schema_extra["param_meta"].choices == ["analytic", "subsample", "none"]
    assert list(build_options(ref.step.inputs_model)[0].type.choices) == ["analytic", "subsample", "none"]

    @click.command()
    def command(**kwargs):
        click.echo(kwargs["mode"])

    command.params.append(build_options(ref.step.inputs_model)[0])
    runner = CliRunner()
    assert runner.invoke(command, ["--mode", "none"]).output.strip() == "none"
    assert runner.invoke(command, ["--mode", "invalid"]).exit_code != 0

    assert ref(mode="subsample").outputs.shifted == "subsample"
    with pytest.raises(ValueError, match="Input should be 'analytic', 'subsample' or 'none'"):
        ref.step.inputs_model(mode="invalid")


def test_field_param_meta_choices_validate_and_build_click_choice():
    ref = pystep()(choose_mode_from_field)
    option = build_options(ref.step.inputs_model)[0]

    assert list(option.type.choices) == ["analytic", "subsample", "none"]
    with pytest.raises(ValueError, match="Input should be 'analytic', 'subsample' or 'none'"):
        ref.step.inputs_model(mode="invalid")


def test_happy_path_standalone_call():
    ref = pystep()(add_offset)
    result = ref(ms="x.ms", offset=2.0)
    assert result.success
    assert result.returncode == 0
    assert result.outputs.shifted == "x.ms+2.0"
    assert result.shifted == "x.ms+2.0"  # StepResult.__getattr__ read-through


def test_wraps_an_existing_function_without_decorator_syntax():
    # matches the precedent already established for shinobi.step
    ref = pystep()(add_offset)
    assert isinstance(ref, StepRef)
    assert ref.name == "add_offset"


def test_step_ref_json_dumps_without_crashing_on_func():
    """`func` is a live callable, not JSON-serializable by default -- this
    is what `ninja cabs show <name>` exercises for a pystep-backed
    `shinobi.cabs` provider entry (e.g. dosho's CASA-task pysteps).
    """
    import json

    ref = pystep()(add_offset)
    dumped = json.loads(ref.model_dump_json())
    assert dumped["func"] == "_adapter"
    assert dumped["name"] == "add_offset"


def test_unannotated_parameter_raises_at_decoration():
    def bad(x) -> OffsetOutputs:  # no type hint on x
        return OffsetOutputs(shifted="x")

    with pytest.raises(TypeError, match="no type hint"):
        pystep()(bad)


def test_var_positional_raises_at_decoration():
    def bad(*args: int) -> OffsetOutputs:
        return OffsetOutputs(shifted="x")

    with pytest.raises(TypeError, match="positional"):
        pystep()(bad)


def test_var_keyword_raises_at_decoration():
    def bad(**kwargs: int) -> OffsetOutputs:
        return OffsetOutputs(shifted="x")

    with pytest.raises(TypeError):
        pystep()(bad)


def test_bad_return_annotation_raises_at_decoration():
    def bad(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="BaseModel"):
        pystep()(bad)


def test_no_return_annotation_means_empty_outputs():
    def side_effect(x: int):
        assert x == 1

    ref = pystep()(side_effect)
    assert ref.step.outputs_model.model_fields == {}
    result = ref(x=1)
    assert result.success


def test_empty_outputs_function_returning_non_none_raises_at_call():
    def side_effect(x: int):
        return "not none"

    ref = pystep()(side_effect)
    with pytest.raises(TypeError, match="None"):
        ref(x=1)


def test_wrong_outputs_type_raises_at_call():
    def wrong(x: int) -> OffsetOutputs:
        return "not an OffsetOutputs"  # type: ignore[return-value]

    ref = pystep()(wrong)
    with pytest.raises(TypeError, match="OffsetOutputs"):
        ref(x=1)


def test_inputs_are_immutable_by_default():
    def mutate(items: list[int]) -> OffsetOutputs:
        items.append(99)
        return OffsetOutputs(shifted=str(items))

    ref = pystep()(mutate)
    original = [1, 2, 3]
    ref(items=original)
    assert original == [1, 2, 3]  # deep-copied before the function ran


def test_params_prebind_a_constant_like_step():
    ref = pystep(offset=5.0)(add_offset)
    result = ref(ms="x.ms")
    assert result.outputs.shifted == "x.ms+5.0"


def test_pystep_wires_into_a_recipe_feeding_a_cab():
    class RecipeInputs(BaseModel):
        ms: str = "in.ms"

    class RecipeOutputs(BaseModel):
        ok: bool | None = None

    offset_ref = pystep()(add_offset)

    recipe = Recipe(name="r", inputs_model=RecipeInputs, outputs_model=RecipeOutputs)
    recipe.add_step("offset", offset_ref, ms=InputRef(field="ms"))
    recipe.add_step("use", use_value_cab, path=recipe.outputs("offset", "shifted"))
    recipe.set_output("ok", recipe.outputs("use", "ok"))

    recorder = RecordingBackend()
    register_step_backend("recording", recorder)
    result = _dispatch(recipe, None, backend="recording")

    assert result.success
    cab, argv, inputs = recorder.calls[0]
    assert cab.name == "use_value"
    assert inputs["path"] == "in.ms+0.0"
