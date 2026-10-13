import pytest
from pydantic import BaseModel

from shinobi.dag import TraceStep, graph_nodes, render_dag
from shinobi.graph import RecipeGraphError
from shinobi.resources import Resources
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


# -- graph_nodes: build the declared graph from a Recipe --


def test_graph_nodes_detects_output_dependency_edge():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
            StepRef(name="use", step=use, wiring={"path": OutputRef(step="make", field="path")}),
        ],
    )
    nodes = graph_nodes(recipe)
    assert [n.name for n in nodes] == ["make", "use"]
    assert nodes[0].depends_on == set()
    assert nodes[1].depends_on == {0}  # use depends on make via OutputRef


def test_independent_steps_are_separate_roots_in_declaration_order():
    a = _cab("a", In, PathOut)
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="a", step=a, wiring={"name": InputRef(field="name")}),
            StepRef(name="b", step=a, wiring={"name": InputRef(field="name")}),
        ],
    )
    nodes = graph_nodes(recipe)
    # no invented edge: b does not wait on a, and the tree says so
    assert nodes[1].depends_on == set()
    assert render_dag(nodes).splitlines() == ["a", "b"]


def test_graph_nodes_surfaces_a_cycle_like_the_executor_would():
    a = _cab("a", UseIn, PathOut)
    b = _cab("b", UseIn, PathOut)
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="a", step=a, wiring={"path": OutputRef(step="b", field="path")}),
            StepRef(name="b", step=b, wiring={"path": OutputRef(step="a", field="path")}),
        ],
    )
    with pytest.raises(RecipeGraphError, match="dependency cycle"):
        graph_nodes(recipe)


# -- render_dag: an indented tree --


def test_render_dag_empty():
    assert render_dag([]) == "(no steps traced)"


def test_render_linear_chain():
    steps = [
        TraceStep(id=0, name="Commit", depends_on=set()),
        TraceStep(id=1, name="Build App", depends_on={0}),
    ]
    assert render_dag(steps).splitlines() == ["Commit", "└── Build App"]


def test_render_fan_out_and_fan_in_diamond():
    steps = [
        TraceStep(id=0, name="Commit", depends_on=set()),
        TraceStep(id=1, name="Build App", depends_on={0}),
        TraceStep(id=2, name="Run Tests", depends_on={0}),
        TraceStep(id=3, name="Deploy to QA", depends_on={1, 2}),
    ]
    # the fan-in step sits under its last parent and names the other one
    assert render_dag(steps).splitlines() == [
        "Commit",
        "├── Build App",
        "└── Run Tests",
        "    └── Deploy to QA  (also waits on: Build App)",
    ]


def test_render_keeps_deep_branches_aligned():
    steps = [
        TraceStep(id=0, name="A", depends_on=set()),
        TraceStep(id=1, name="B", depends_on={0}),
        TraceStep(id=2, name="C", depends_on={0}),
        TraceStep(id=3, name="D", depends_on={1}),
    ]
    assert render_dag(steps).splitlines() == [
        "A",
        "├── B",
        "│   └── D",
        "└── C",
    ]


def test_dryrun_shows_a_declared_footprint():
    """`--dryrun` shows what is declared, and a footprint is a declaration
    that changes a run's shape -- it is what turns parallel branches into a
    queue -- so it belongs in the diagram, not just in the timings.
    """

    def cab(n, resources=None):
        return Cab(name=n, command=n, inputs_model=In, outputs_model=PathOut, resources=resources)

    recipe = Recipe(
        name="p",
        inputs_model=In,
        outputs_model=PathOut,
        steps=[
            StepRef(name="cheap", step=cab("cheap")),
            StepRef(name="heavy", step=cab("heavy", Resources(cpus=16, memory="200GiB"))),
        ],
    )
    nodes = graph_nodes(recipe)
    assert nodes[0].resources == ""
    assert nodes[1].resources == "cpus=16, memory=200.0GiB"
    assert render_dag(nodes).splitlines() == ["cheap", "heavy  (cpus=16, memory=200.0GiB)"]


# -- nested recipes are drawn as their own branch --


def test_nested_recipe_is_a_branch_holding_its_own_steps():
    make = _cab("make", In, PathOut)
    use = _cab("use", UseIn, OkOut)
    inner = Recipe(
        name="inner",
        inputs_model=UseIn,
        outputs_model=PathOut,
        steps=[
            StepRef(name="a", step=use, wiring={"path": InputRef(field="path")}),
            StepRef(name="b", step=make, after=["a"]),
        ],
        output_wiring={"path": OutputRef(step="b", field="path")},
    )
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make, wiring={"name": InputRef(field="name")}),
            StepRef(name="sub", step=inner, wiring={"path": OutputRef(step="make", field="path")}),
            StepRef(name="use", step=use, wiring={"path": OutputRef(step="sub", field="path")}),
        ],
    )
    nodes = graph_nodes(recipe)
    assert [(n.name, n.recipe, n.group) for n in nodes] == [("make", False, None), ("sub", True, None), ("a", False, 1), ("b", False, 1), ("use", False, None)]
    # edges stay within a level: sub waits on make, b on a, use on sub
    assert [n.depends_on for n in nodes] == [set(), {0}, set(), {2}, {1}]
    # a is one of sub's steps (dashed); b waits on a, use waits on sub (solid)
    assert render_dag(nodes).splitlines() == [
        "make",
        "└── sub  (recipe)",
        "    ├╌╌ a",
        "    │   └── b",
        "    └── use",
    ]


def test_recipes_nested_two_deep():
    a = _cab("a", In, OkOut)
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="x", step=a)])
    middle = Recipe(name="middle", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="inner", step=inner), StepRef(name="y", step=a)])
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="mid", step=middle)])
    assert render_dag(graph_nodes(recipe)).splitlines() == [
        "mid  (recipe)",
        "├╌╌ inner  (recipe)",
        "│   └╌╌ x",
        "└╌╌ y",
    ]


def test_dotted_step_name_cannot_pass_for_a_nested_step():
    """Step names are free strings, and `add_loop` declares dotted ones, so a
    top-level `sub.a` waiting on `sub` must not look like `sub`'s own `a`.
    Unqualified names plus the membership connector keep them apart."""
    a = _cab("a", In, OkOut)
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="a", step=a)])
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="sub", step=inner),
            StepRef(name="sub.a", step=a, after=["sub"]),
            StepRef(name="last", step=a, after=["sub", "sub.a"]),
        ],
    )
    assert render_dag(graph_nodes(recipe)).splitlines() == [
        "sub  (recipe)",
        "├╌╌ a",
        "└── sub.a",
        "    └── last  (also waits on: sub)",
    ]


def test_empty_nested_recipe_is_a_leaf():
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut)
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="sub", step=inner)])
    assert render_dag(graph_nodes(recipe)).splitlines() == ["sub  (recipe)"]


def test_mis_wired_nested_recipe_fails_the_dryrun_graph():
    inner = Recipe(
        name="inner",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[StepRef(name="a", step=_cab("a", UseIn, OkOut), wiring={"path": OutputRef(step="nope", field="path")})],
    )
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="sub", step=inner)])
    with pytest.raises(RecipeGraphError, match="invalid nested recipe"):
        graph_nodes(recipe)
