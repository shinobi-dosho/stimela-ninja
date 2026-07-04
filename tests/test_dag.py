from pydantic import BaseModel

from shinobi.dag import TraceStep, graph_nodes, render_dag
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


def test_graph_nodes_chains_independent_steps_sequentially():
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
    # b has no output dependency, so it's chained after a
    assert nodes[1].depends_on == {0}


# -- render_dag (kept verbatim from the old model) --


def test_render_dag_empty():
    assert render_dag([]) == "(no steps traced)"


def test_render_linear_chain():
    steps = [
        TraceStep(id=0, name="Commit", depends_on=set()),
        TraceStep(id=1, name="Build App", depends_on={0}),
    ]
    out = render_dag(steps)
    assert "[ Commit ]" in out
    assert "[ Build App ]" in out
    assert "v" in out
    assert out.index("Commit") < out.index("Build App")


def test_render_fan_out_and_fan_in_diamond():
    steps = [
        TraceStep(id=0, name="Commit", depends_on=set()),
        TraceStep(id=1, name="Build App", depends_on={0}),
        TraceStep(id=2, name="Run Tests", depends_on={0}),
        TraceStep(id=3, name="Deploy to QA", depends_on={1, 2}),
    ]
    out = render_dag(steps)
    lines = out.splitlines()
    sibling_line = next(ln for ln in lines if "Build App" in ln and "Run Tests" in ln)
    assert sibling_line
    bracket_lines = [ln for ln in lines if "-" in ln and ln.count("+") == 3]
    assert len(bracket_lines) == 2
    assert out.index("Commit") < out.index("Build App") < out.index("Deploy to QA")


def test_render_falls_back_to_plain_chain_without_false_fan_structure():
    steps = [
        TraceStep(id=0, name="A", depends_on=set()),
        TraceStep(id=1, name="B", depends_on={0}),
        TraceStep(id=2, name="C", depends_on={0}),
        TraceStep(id=3, name="D", depends_on={1}),
    ]
    out = render_dag(steps)
    lines = out.splitlines()
    assert not any("-" in ln for ln in lines[-4:])
