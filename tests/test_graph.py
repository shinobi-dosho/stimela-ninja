from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.graph import (
    RecipeGraph,
    RecipeGraphError,
    RecipeNotOffloadableError,
    build_graph,
    check_offloadable,
)
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, Recipe, StepRef


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


def test_list_of_output_refs_creates_one_edge_per_producer():
    """A single wiring value can be a list of refs (e.g. applycal's
    gaintable=[k.caltable, g.caltable], accumulating a variable number of
    upstream outputs into one list-typed input) -- each ref in the list
    contributes its own dependency edge.
    """
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    graph = build_graph(
        _recipe(
            [
                StepRef(name="k", step=make, wiring={"name": InputRef(field="name")}),
                StepRef(name="g", step=make, wiring={"name": InputRef(field="name")}),
                StepRef(
                    name="apply",
                    step=use,
                    wiring={
                        "path": [
                            OutputRef(step="k", field="path"),
                            OutputRef(step="g", field="path"),
                        ]
                    },
                ),
            ]
        )
    )
    assert graph.deps == [set(), set(), {0, 1}]
    assert graph.dependents == [{2}, {2}, set()]


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


# -- offload eligibility (check_offloadable) --
#
# Offloadable data flow is filesystem paths only, so these fixtures use
# real pathlib.Path-typed output fields (File/MS/Directory dtypes map to
# Path); a plain `str` output is NOT a path field and must be rejected.


class MakePathIn(BaseModel):
    where: Path = Path("out.ms")


class MSOut(BaseModel):
    ms: Path | None = None


class UsePathIn(BaseModel):
    ms: Path | None = None


class StrOut(BaseModel):
    value: str | None = None


def _make_cab(name="make"):
    return Cab(name=name, command="mk", inputs_model=MakePathIn, outputs_model=MSOut)


def _use_cab(name="use"):
    return Cab(name=name, command="use", inputs_model=UsePathIn, outputs_model=OkOut)


def _path_wired_recipe(make, use):
    return _recipe(
        [
            StepRef(name="make", step=make, wiring={"where": InputRef(field="name")}),
            StepRef(name="use", step=use, wiring={"ms": OutputRef(step="make", field="ms")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )


def test_pure_path_wired_recipe_is_offloadable():
    check_offloadable(_path_wired_recipe(_make_cab(), _use_cab()))  # does not raise


def test_orchestration_function_blocks_offload():
    make = _make_cab()
    use = _use_cab()
    recipe = _path_wired_recipe(make, use)
    recipe.steps[1].func = lambda ctx: ctx.run()
    with pytest.raises(RecipeNotOffloadableError, match="orchestration function"):
        check_offloadable(recipe)


def test_mutable_input_blocks_offload():
    make = _make_cab().model_copy(update={"input_mutability": {"where": Mutability.MUTABLE}})
    with pytest.raises(RecipeNotOffloadableError, match=r"MUTABLE input\(s\) \['where'\]"):
        check_offloadable(_path_wired_recipe(make, _use_cab()))


def test_non_cab_step_blocks_offload():
    from shinobi.steps.schema import Scope

    scope = Scope(name="bare", inputs_model=MakePathIn, outputs_model=MSOut)
    recipe = _recipe(
        [
            StepRef(name="make", step=scope, func=lambda ctx: None, wiring={"where": InputRef(field="name")}),
        ]
    )
    with pytest.raises(RecipeNotOffloadableError, match="not a Cab"):
        check_offloadable(recipe)


def test_non_path_output_ref_blocks_offload():
    make = Cab(name="make", command="mk", inputs_model=MakePathIn, outputs_model=StrOut)
    use = Cab(name="use", command="use", inputs_model=UsePathIn, outputs_model=OkOut)
    recipe = _recipe(
        [
            StepRef(name="make", step=make, wiring={"where": InputRef(field="name")}),
            StepRef(name="use", step=use, wiring={"ms": OutputRef(step="make", field="value")}),
        ]
    )
    with pytest.raises(RecipeNotOffloadableError, match="non-path output 'make.value'"):
        check_offloadable(recipe)


def test_list_wired_non_path_output_ref_blocks_offload():
    """check_offloadable must look inside a list-valued wiring entry too,
    not just scalar ones -- each ref in the list is independently checked.
    """
    make = Cab(name="make", command="mk", inputs_model=MakePathIn, outputs_model=StrOut)
    use = Cab(name="use", command="use", inputs_model=UsePathIn, outputs_model=OkOut)
    recipe = _recipe(
        [
            StepRef(name="make", step=make, wiring={"where": InputRef(field="name")}),
            StepRef(
                name="use",
                step=use,
                wiring={"ms": [OutputRef(step="make", field="value")]},
            ),
        ]
    )
    with pytest.raises(RecipeNotOffloadableError, match="non-path output 'make.value'"):
        check_offloadable(recipe)


def test_wrangler_derived_output_ref_blocks_offload():
    # `ms` is a path field, but it's filled by a wrangler -> unavailable offloaded
    make = Cab(
        name="make",
        command="mk",
        inputs_model=MakePathIn,
        outputs_model=MSOut,
        wranglers={r"ms=(?P<ms>\S+)": ["PARSE_OUTPUT:ms:str"]},
    )
    recipe = _path_wired_recipe(make, _use_cab())
    with pytest.raises(RecipeNotOffloadableError, match="wrangler-derived output 'make.ms'"):
        check_offloadable(recipe)


def test_offload_check_reports_all_reasons_at_once():
    make = _make_cab().model_copy(update={"input_mutability": {"where": Mutability.MUTABLE}})
    use = _use_cab()
    recipe = _path_wired_recipe(make, use)
    recipe.steps[1].func = lambda ctx: ctx.run()
    with pytest.raises(RecipeNotOffloadableError) as exc:
        check_offloadable(recipe)
    msg = str(exc.value)
    assert "MUTABLE" in msg and "orchestration function" in msg


# -- additional wiring/field validation --


def test_wiring_field_not_on_consumer_inputs_is_rejected():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    with pytest.raises(RecipeGraphError, match="step 'use' wires input 'nope'"):
        build_graph(
            _recipe(
                [
                    StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
                    StepRef(name="use", step=use, wiring={"nope": OutputRef(step="make", field="path")}),
                ]
            )
        )


def test_output_ref_to_missing_field_is_rejected():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    with pytest.raises(RecipeGraphError, match="output 'nope' of step 'make'"):
        build_graph(
            _recipe(
                [
                    StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
                    StepRef(name="use", step=use, wiring={"path": OutputRef(step="make", field="nope")}),
                ]
            )
        )


def test_output_wiring_field_not_on_recipe_outputs_is_rejected():
    make = _cab("make", In, PathOut)
    with pytest.raises(RecipeGraphError, match="output 'nope' is not a field of OkOut"):
        build_graph(
            _recipe(
                [StepRef(name="make", step=make, wiring={"name": InputRef(field="name")})],
                output_wiring={"nope": OutputRef(step="make", field="path")},
            )
        )


def test_output_wiring_to_missing_step_field_is_rejected():
    make = _cab("make", In, PathOut)
    with pytest.raises(RecipeGraphError, match="output 'ok' is wired from output 'nope' of step 'make'"):
        build_graph(
            _recipe(
                [StepRef(name="make", step=make, wiring={"name": InputRef(field="name")})],
                output_wiring={"ok": OutputRef(step="make", field="nope")},
            )
        )


def test_constant_param_not_on_step_inputs_is_rejected():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    with pytest.raises(RecipeGraphError, match="step 'use' sets constant param 'nope'"):
        build_graph(
            _recipe(
                [
                    StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
                    StepRef(
                        name="use",
                        step=use,
                        wiring={"path": OutputRef(step="make", field="path")},
                        params={"nope": "x"},
                    ),
                ]
            )
        )
