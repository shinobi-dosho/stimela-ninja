"""Tests for `@shinobi.pystep` (src/shinobi/steps/pyfunc.py)."""

from pathlib import Path
from typing import Annotated, Literal, Optional

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


# -- ParamMeta declarations that must not silently change a field's shape --


def nested_in_optional(x: Optional[Annotated[str, ParamMeta(choices=["a", "b"])]] = None) -> OffsetOutputs:
    return OffsetOutputs(shifted=str(x))


def nested_in_list(cols: list[Annotated[str, ParamMeta(choices=["I", "Q"])]] = ["I"]) -> OffsetOutputs:
    return OffsetOutputs(shifted=",".join(cols))


def optional_choices(x: Annotated[str | None, ParamMeta(choices=["a", "b"])] = None) -> OffsetOutputs:
    return OffsetOutputs(shifted=str(x))


def list_choices(cols: Annotated[list[str], ParamMeta(choices=["I", "Q"])] = ["I"]) -> OffsetOutputs:
    return OffsetOutputs(shifted=",".join(cols))


def path_choices(out: Annotated[Path, ParamMeta(choices=["/a", "/b"])] = Path("/a")) -> OffsetOutputs:
    return OffsetOutputs(shifted=str(out))


def int_choices(n: Annotated[int, ParamMeta(choices=[1, 2])] = 1) -> OffsetOutputs:
    return OffsetOutputs(shifted=str(n))


def documented_choice(mode: Annotated[str, ParamMeta(choices=["a", "b"], abbreviation="m", info="pick one")] = "a") -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


def written_and_chosen(out: str = "x", keep: Annotated[str, ParamMeta(choices=["a", "b"])] = "a") -> OffsetOutputs:
    return OffsetOutputs(shifted=out)


SHARED_MODE_FIELD = Field("a", json_schema_extra={"choices": ["a", "b"]})


def borrows_shared_field(mode: str = SHARED_MODE_FIELD) -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


def annotates_shared_field(mode: Annotated[str, ParamMeta(choices=["x", "y"])] = SHARED_MODE_FIELD) -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


@pytest.mark.parametrize("func", [nested_in_optional, nested_in_list])
def test_param_meta_below_the_top_level_is_rejected(func):
    # `get_type_hints(..., include_extras=True)` keeps every Annotated layer,
    # but only the outermost one can be lifted off before `create_model` -- a
    # deeper ParamMeta would otherwise reach pydantic, which would then
    # validate the parameter's value as a ParamMeta instance.
    with pytest.raises(TypeError, match="nests ParamMeta inside its annotation"):
        pystep()(func)


def test_optional_choices_keep_their_none_arm():
    model = pystep()(optional_choices).step.inputs_model
    assert model(x="a").x == "a"
    assert model(x=None).x is None
    with pytest.raises(ValueError, match="Input should be 'a' or 'b'"):
        model(x="zzz")


def test_list_choices_narrow_the_element_not_the_container():
    model = pystep()(list_choices).step.inputs_model
    assert model(cols=["I", "Q"]).cols == ["I", "Q"]
    with pytest.raises(ValueError, match="Input should be 'I' or 'Q'"):
        model(cols=["Z"])

    # the field is still a list, so click still renders a repeatable option
    option = build_options(model)[0]
    assert option.multiple is True

    @click.command()
    def command(**kwargs):
        click.echo(",".join(kwargs["cols"]))

    command.params.append(option)
    assert CliRunner().invoke(command, ["--cols", "I", "--cols", "Q"]).output.strip() == "I,Q"


def test_choices_on_a_non_scalar_leaf_are_refused():
    # Narrowing a Path away to a Literal would leave `path_fields` unable to
    # see that the field is a path at all -- no `click.Path`, no bind-mount.
    with pytest.raises(TypeError, match="choices narrow a str/int/float/bool leaf"):
        pystep()(path_choices)


def test_non_string_choices_round_trip_through_click():
    model = pystep()(int_choices).step.inputs_model

    @click.command()
    def command(**kwargs):
        click.echo(repr(model(**kwargs).n))

    command.params.append(build_options(model)[0])
    runner = CliRunner()
    assert runner.invoke(command, ["--n", "2"]).output.strip() == "2"
    assert runner.invoke(command, []).output.strip() == "1"
    assert runner.invoke(command, ["--n", "9"]).exit_code != 0


def test_shared_field_spec_is_not_mutated_by_a_neighbouring_pystep():
    # One module-level `Field(...)` is the default of two functions: the
    # first one's annotation metadata must not be written into the shared
    # object and so picked up by the second.
    pystep()(annotates_shared_field)
    field = pystep()(borrows_shared_field).step.inputs_model.model_fields["mode"]
    assert field.annotation == Literal["a", "b"]


def test_param_meta_abbreviation_and_info_reach_the_cli():
    option = build_options(pystep()(documented_choice).step.inputs_model)[0]
    assert option.opts == ["--mode", "-m"]
    assert option.help == "pick one"


def test_write_paths_agree_between_scope_and_model_metadata():
    ref = pystep(write_paths=["out"])(written_and_chosen)
    assert ref.step.field_meta["out"].write_path is True
    assert ref.step.inputs_model.model_fields["out"].json_schema_extra["param_meta"].write_path is True
    assert ref.step.field_meta["keep"].write_path is False


def test_pystep_carries_its_cache_policy_onto_the_scope():
    ref = pystep(cache=False, cache_dir="/shared/cache")(add_offset)
    assert ref.step.cache is False
    assert ref.step.cache_dir == "/shared/cache"


def annotated_over_field_default(
    ascii_sky: Annotated[Optional[str], ParamMeta(abbreviation="as", info="ignored")] = Field(None, description="Catalogue of sources."),
    smearing_subsamples: Annotated[int, ParamMeta(abbreviation="sss")] = Field(8, ge=1, description="Sub-sample cap."),
    mode: Annotated[str, ParamMeta(choices=["sim", "add"], abbreviation="m")] = Field("sim", description="What to do."),
    described_by_meta: Annotated[str, ParamMeta(abbreviation="d", info="from ParamMeta")] = Field("x"),
    cab_style: Annotated[str, ParamMeta(choices=["a", "b"])] = Field("a", description="Flat extra.", json_schema_extra={"abbreviation": "cs"}),
) -> OffsetOutputs:
    return OffsetOutputs(shifted=mode)


def test_param_meta_survives_a_field_default():
    # `Annotated[T, ParamMeta(...)] = Field(...)` spells both halves. The
    # `ParamMeta` has to reach the built model's field, not just
    # `Scope.field_meta`: `create_model` rebuilds a `FieldInfo` default out
    # of its constructor arguments, so metadata attached afterwards would be
    # dropped and every CLI-facing key but `choices` (which rewrites the
    # annotation) would silently vanish.
    fields = pystep()(annotated_over_field_default).step.inputs_model.model_fields

    assert fields["ascii_sky"].json_schema_extra["param_meta"].abbreviation == "as"
    assert fields["smearing_subsamples"].json_schema_extra["param_meta"].abbreviation == "sss"
    # the `Field(...)` half survives the merge intact
    assert fields["ascii_sky"].description == "Catalogue of sources."
    assert fields["smearing_subsamples"].default == 8
    assert fields["smearing_subsamples"].metadata  # ge=1 is not dropped
    assert fields["mode"].annotation == Literal["sim", "add"]


def test_field_default_abbreviations_reach_the_cli():
    options = {option.name: option for option in build_options(pystep()(annotated_over_field_default).step.inputs_model)}

    assert options["ascii_sky"].opts == ["--ascii-sky", "-as"]
    assert options["smearing_subsamples"].opts == ["--smearing-subsamples", "-sss"]
    assert options["mode"].opts == ["--mode", "-m"]
    assert list(options["mode"].type.choices) == ["sim", "add"]
    # a `Field(description=...)` wins over `ParamMeta.info`, which only fills a gap
    assert options["ascii_sky"].help == "Catalogue of sources."
    assert options["described_by_meta"].help == "from ParamMeta"
    # a flat cab-style key already on the field is kept alongside `param_meta`
    assert options["cab_style"].opts == ["--cab-style", "-cs"]


def test_field_default_is_not_mutated_by_the_pystep():
    # The `Field(...)` object lives in the function's `__defaults__`; the
    # decorator must leave it as the author wrote it.
    default = annotated_over_field_default.__defaults__[0]
    pystep()(annotated_over_field_default)
    assert default.json_schema_extra is None
