import pytest
from pydantic import BaseModel

from shinobi.graph import RecipeGraph, RecipeGraphError, build_graph
from shinobi.steps.schema import Cab, InputRef, OutputRef, Recipe, StepRef


class In(BaseModel):
    name: str = "x"


class PathOut(BaseModel):
    path: str | None = None


class UseIn(BaseModel):
    path: str | None = None


class OkOut(BaseModel):
    ok: bool = True


def _cab(name, im, om):
    return Cab(name=name, command="x", inputs_model=im, outputs_model=om)


def _recipe(steps, output_wiring=None, inputs_model=In):
    return Recipe(
        name="r",
        inputs_model=inputs_model,
        outputs_model=OkOut,
        steps=steps,
        output_wiring=output_wiring or {},
    )


# -- true edges (no artificial chaining) --


def test_output_ref_creates_true_dependency_edge():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    graph = build_graph(
        _recipe(
            [
                StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
                StepRef(name="use", step=use, wiring={"path": OutputRef(step="make", field="path")}),
            ]
        )
    )
    assert isinstance(graph, RecipeGraph)
    assert graph.names == ["make", "use"]
    assert graph.deps == [set(), {0}]
    assert graph.dependents == [{1}, set()]


def test_independent_steps_have_no_edges():
    a = _cab("a", In, PathOut)
    graph = build_graph(
        _recipe(
            [
                StepRef(name="a", step=a, wiring={"name": InputRef(field="name")}),
                StepRef(name="b", step=a, wiring={"name": InputRef(field="name")}),
            ]
        )
    )
    # unlike the display graph, the executor's graph does NOT chain b after a
    assert graph.deps == [set(), set()]


# -- validation --


def test_duplicate_step_name_is_rejected():
    a = _cab("a", In, PathOut)
    with pytest.raises(RecipeGraphError, match="more than one step named 'dup'"):
        build_graph(
            _recipe(
                [
                    StepRef(name="dup", step=a, wiring={"name": InputRef(field="name")}),
                    StepRef(name="dup", step=a, wiring={"name": InputRef(field="name")}),
                ]
            )
        )


def test_input_ref_to_unknown_recipe_field_is_rejected():
    a = _cab("a", In, PathOut)
    with pytest.raises(RecipeGraphError, match="not a field of In"):
        build_graph(_recipe([StepRef(name="a", step=a, wiring={"name": InputRef(field="nope")})]))


def test_output_ref_to_unknown_step_is_rejected():
    use = _cab("use", UseIn, OkOut)
    with pytest.raises(RecipeGraphError, match="output of step 'ghost'"):
        build_graph(
            _recipe([StepRef(name="use", step=use, wiring={"path": OutputRef(step="ghost", field="path")})])
        )


def test_output_wiring_to_unknown_step_is_rejected():
    a = _cab("a", In, PathOut)
    with pytest.raises(RecipeGraphError, match="output 'ok' is wired from step 'ghost'"):
        build_graph(
            _recipe(
                [StepRef(name="a", step=a, wiring={"name": InputRef(field="name")})],
                output_wiring={"ok": OutputRef(step="ghost", field="ok")},
            )
        )


def test_dependency_cycle_is_rejected():
    a = _cab("a", UseIn, PathOut)
    b = _cab("b", UseIn, PathOut)
    with pytest.raises(RecipeGraphError, match="dependency cycle involving: a, b"):
        build_graph(
            _recipe(
                [
                    StepRef(name="a", step=a, wiring={"path": OutputRef(step="b", field="path")}),
                    StepRef(name="b", step=b, wiring={"path": OutputRef(step="a", field="path")}),
                ]
            )
        )
