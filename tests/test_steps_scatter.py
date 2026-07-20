import pytest
from pydantic import BaseModel, Field

from shinobi.backends.recording import RecordingBackend
from shinobi.exceptions import CabRunError
from shinobi.results import BackendRun
from shinobi.steps import (
    Cab,
    InputRef,
    OutputRef,
    Recipe,
    ScatterSpec,
    StepRef,
    register_step_backend,
    step,
)
from shinobi.steps.dispatch import ScatterError, _dispatch
from shinobi.graph import RecipeGraphError, RecipeNotOffloadableError, build_graph, check_offloadable


class ListIn(BaseModel):
    items: list[int]


class ScalarIn(BaseModel):
    item: int
    flag: bool = False


class ScalarOut(BaseModel):
    out: int | None = None


class ListOut(BaseModel):
    outs: list[int | None]


class BoolOut(BaseModel):
    ok: bool = True


class EchoBackend:
    """Backend that echoes the input 'item' value as an 'out' wrangler output."""

    def run(self, cab, argv, inputs, **kwargs):
        return BackendRun(0, f"out={inputs['item']}", "")


def make_recording_cab() -> tuple[Cab, RecordingBackend]:
    recorder = RecordingBackend()
    register_step_backend("scatter-record", recorder)
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=ScalarIn,
        outputs_model=ScalarOut,
        backend="scatter-record",
        wranglers={r"out=(?P<out>\d+)": ["PARSE_OUTPUT:out:int"]},
    )
    return cab, recorder


def make_echo_cab() -> Cab:
    register_step_backend("echo", EchoBackend())
    return Cab(
        name="tool",
        command="tool",
        inputs_model=ScalarIn,
        outputs_model=ScalarOut,
        backend="echo",
        wranglers={r"out=(?P<out>\d+)": ["PARSE_OUTPUT:out:int"]},
    )


# -- builder API --


def test_add_step_accepts_scatter_list():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    assert recipe.steps[0].scatter == ScatterSpec(fields=["item"])


def test_add_step_accepts_scatter_spec():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=ScatterSpec(fields=["item"]), item=recipe.inputs.items)
    assert recipe.steps[0].scatter.fields == ["item"]


def test_step_decorator_accepts_scatter():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)

    @recipe.step(cab, scatter=["item"], item=recipe.inputs.items)
    def a(ctx):
        return None

    assert recipe.steps[0].scatter.fields == ["item"]


def test_free_step_decorator_accepts_scatter():
    cab, _ = make_recording_cab()

    @step(scope=cab, scatter=["item"])
    def a(ctx):
        return None

    assert a.scatter.fields == ["item"]


def test_add_step_preserves_scatter_from_stepref():
    cab, _ = make_recording_cab()
    ref = StepRef(name="src", step=cab, scatter=ScatterSpec(fields=["item"]))
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", ref, item=recipe.inputs.items)
    assert recipe.steps[0].scatter.fields == ["item"]


def test_add_step_can_override_scatter_on_stepref():
    cab, _ = make_recording_cab()
    ref = StepRef(name="src", step=cab, scatter=ScatterSpec(fields=["item"]))
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", ref, scatter=["flag"], item=recipe.inputs.items, flag=True)
    assert recipe.steps[0].scatter.fields == ["flag"]


# -- graph validation --


def test_graph_rejects_scatter_over_unknown_field():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["nope"], item=recipe.inputs.items)
    with pytest.raises(RecipeGraphError, match="scatter over 'nope'"):
        build_graph(recipe)


def test_graph_rejects_empty_scatter_spec():
    cab, _ = make_recording_cab()
    with pytest.raises(ValueError, match="at least one field"):
        ScatterSpec(fields=[])


# -- runtime fan-out --


def test_scatter_runs_once_per_list_element():
    cab = make_echo_cab()
    recipe = Recipe(
        name="r",
        inputs_model=ListIn,
        outputs_model=ListOut,
        steps=[
            StepRef(
                name="a",
                step=cab,
                scatter=ScatterSpec(fields=["item"]),
                wiring={"item": InputRef(field="items")},
            ),
        ],
        output_wiring={"outs": OutputRef(step="a", field="out")},
    )
    result = _dispatch(recipe, None, items=[10, 20, 30])
    assert result.success
    assert result.outputs.outs == [10, 20, 30]


def test_scatter_gathers_outputs_as_list():
    cab = make_echo_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[1, 2, 3])
    assert result.outputs.outs == [1, 2, 3]


def test_scatter_can_chain_over_gathered_list():
    """A downstream step can scatter over a gathered output from a scattered
    upstream step.
    """
    cab = make_echo_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.add_step("b", cab, scatter=["item"], item=recipe.outputs.a.out)
    recipe.set_output("outs", recipe.outputs.b.out)
    result = _dispatch(recipe, None, items=[1, 2, 3])
    assert result.outputs.outs == [1, 2, 3]


def test_scatter_with_multiple_fields_same_length():
    class TwoIn(BaseModel):
        x: int
        y: int

    class TwoOut(BaseModel):
        z: int | None = None

    class TwoBackend:
        def run(self, cab, argv, inputs, **kwargs):
            return BackendRun(0, f"z={inputs['x'] + inputs['y']}", "")

    register_step_backend("two", TwoBackend())
    cab = Cab(
        name="two",
        command="two",
        inputs_model=TwoIn,
        outputs_model=TwoOut,
        backend="two",
        wranglers={r"z=(?P<z>\d+)": ["PARSE_OUTPUT:z:int"]},
    )

    class RIn(BaseModel):
        xs: list[int]
        ys: list[int]

    class ROut(BaseModel):
        zs: list[int | None]

    recipe = Recipe(name="r", inputs_model=RIn, outputs_model=ROut)
    recipe.add_step(
        "a",
        cab,
        scatter=["x", "y"],
        x=recipe.inputs.xs,
        y=recipe.inputs.ys,
    )
    recipe.set_output("zs", recipe.outputs.a.z)
    result = _dispatch(recipe, None, xs=[1, 2], ys=[10, 20])
    assert result.success
    assert result.outputs.zs == [11, 22]


def test_scatter_mismatched_lengths_raise():
    class TwoIn(BaseModel):
        item: int
        other: int

    class TwoOut(BaseModel):
        z: int

    recorder = RecordingBackend()
    register_step_backend("two-record", recorder)
    cab = Cab(
        name="two",
        command="two",
        inputs_model=TwoIn,
        outputs_model=TwoOut,
        backend="two-record",
    )

    class RIn(BaseModel):
        items: list[int]
        others: list[int]

    recipe = Recipe(name="r", inputs_model=RIn, outputs_model=BoolOut)
    recipe.add_step(
        "a",
        cab,
        scatter=["item", "other"],
        item=recipe.inputs.items,
        other=recipe.inputs.others,
    )

    with pytest.raises(ScatterError, match="different lengths"):
        _dispatch(recipe, None, items=[1, 2, 3], others=[10, 20])


def test_scatter_non_list_value_raises():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ScalarIn, outputs_model=BoolOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.item)
    with pytest.raises(ScatterError, match="not a list"):
        _dispatch(recipe, None, item=5)


def test_zero_length_scatter_produces_empty_lists():
    cab = make_echo_cab()
    recorder = RecordingBackend()
    register_step_backend("echo-record", recorder)
    cab = cab.model_copy(update={"backend": "echo-record"})
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[])
    assert result.success
    assert result.outputs.outs == []
    assert recorder.calls == []


def test_scatter_preserves_default_factory_on_unsupplied_field():
    """A non-scattered Field(default_factory=...) the caller doesn't supply
    must survive aggregation (the factory, not PydanticUndefined)."""

    class FactoryIn(BaseModel):
        item: int
        tags: list[str] = Field(default_factory=list)

    cab = make_echo_cab()
    cab = cab.model_copy(update={"inputs_model": FactoryIn})
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[1, 2])
    assert result.success
    assert result.outputs.outs == [1, 2]
    assert result.sub_results["a"].inputs.tags == []


# -- failure handling --


def test_scatter_failure_stops_dependents_and_drains_slices():
    class FlakyBackend:
        def __init__(self, fail_on):
            self.fail_on = fail_on

        def run(self, cab, argv, inputs, **kwargs):
            if inputs["item"] in self.fail_on:
                return BackendRun(1, "", "boom")
            return BackendRun(0, f"ok={inputs['item']}", "")

    register_step_backend("flaky", FlakyBackend(fail_on={20}))
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=ScalarIn,
        outputs_model=ScalarOut,
        backend="flaky",
        wranglers={r"ok=(?P<out>\d+)": ["PARSE_OUTPUT:out:int"]},
    )

    down_recorder = RecordingBackend()
    register_step_backend("down-record", down_recorder)
    down_cab = Cab(
        name="down",
        command="down",
        inputs_model=ScalarIn,
        outputs_model=BoolOut,
        backend="down-record",
    )

    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=BoolOut, max_workers=3)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.add_step("b", down_cab, item=recipe.outputs.a.out)
    recipe.set_output("ok", recipe.outputs.b.ok)

    with pytest.raises(CabRunError, match="step 'a'.*returncode 1"):
        _dispatch(recipe, None, items=[10, 20, 30])
    assert len(down_recorder.calls) == 0


# -- offload --


def test_scatter_blocks_offload():
    cab, _ = make_recording_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    with pytest.raises(RecipeNotOffloadableError, match="scatter"):
        check_offloadable(recipe)


# -- concurrency --


def test_scatter_slices_run_concurrently_with_max_workers():
    import threading

    barrier = threading.Barrier(3, timeout=5)

    class BarrierBackend:
        def run(self, cab, argv, inputs, **kwargs):
            barrier.wait()
            return BackendRun(0, f"out={inputs['item']}", "")

    register_step_backend("scatter-barrier", BarrierBackend())
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=ScalarIn,
        outputs_model=ScalarOut,
        backend="scatter-barrier",
        wranglers={r"out=(?P<out>\d+)": ["PARSE_OUTPUT:out:int"]},
    )
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut, max_workers=3)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[1, 2, 3])
    assert result.success
    assert sorted(result.outputs.outs) == [1, 2, 3]


# -- provenance / backend metadata --


def test_scatter_aggregates_containerized_flag():
    class ContainerizedBackend:
        def run(self, cab, argv, inputs, **kwargs):
            return BackendRun(0, f"out={inputs['item']}", "", containerized=True, image_digest="sha256:abc")

    register_step_backend("scatter-container", ContainerizedBackend())
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=ScalarIn,
        outputs_model=ScalarOut,
        backend="scatter-container",
        image="x",
        wranglers={r"out=(?P<out>\d+)": ["PARSE_OUTPUT:out:int"]},
    )
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[1, 2])
    assert result.success
    assert result.outputs.outs == [1, 2]
    assert result.sub_results["a"].containerized is True
    assert result.sub_results["a"].image == "x"


# -- outputs wiring from scatter --


def test_recipe_output_wiring_from_scatter_gathers_list():
    cab = make_echo_cab()
    recipe = Recipe(name="r", inputs_model=ListIn, outputs_model=ListOut)
    recipe.add_step("a", cab, scatter=["item"], item=recipe.inputs.items)
    recipe.set_output("outs", recipe.outputs.a.out)
    result = _dispatch(recipe, None, items=[5, 6, 7])
    assert result.outputs.outs == [5, 6, 7]
