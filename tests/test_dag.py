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
    assert render_dag(nodes).split() == ["a", "b"]
    assert len(render_dag(nodes).splitlines()) == 1


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


# -- render_dag: centered dependency layers --


def test_render_dag_empty():
    assert render_dag([]) == "(no steps traced)"


def test_render_linear_chain():
    rendered = render_dag([TraceStep(0, "Commit"), TraceStep(1, "Build App", {0})])
    lines = rendered.splitlines()
    assert [line.strip() for line in lines] == ["Commit", "│", "│", "▼", "Build App"]
    assert lines[0].index("Commit") + 3 == lines[3].index("▼")
    assert lines[4].index("Build App") + 4 == lines[3].index("▼")


def test_render_fan_out_and_fan_in_diamond():
    steps = [TraceStep(0, "Commit"), TraceStep(1, "Build App", {0}), TraceStep(2, "Run Tests", {0}), TraceStep(3, "Deploy to QA", {1, 2})]
    rendered = render_dag(steps)
    lines = rendered.splitlines()
    parallel = next(line for line in lines if "Build App" in line)
    assert "Run Tests" in parallel
    assert parallel.index("Build App") < parallel.index("Run Tests")
    assert rendered.index("Commit") < rendered.index("Build App") < rendered.index("Deploy to QA")
    assert rendered.count("▼") == 3
    assert "also waits on" not in rendered


def test_render_keeps_deep_branches_aligned():
    rendered = render_dag([TraceStep(0, "A"), TraceStep(1, "B", {0}), TraceStep(2, "C", {0}), TraceStep(3, "D", {1})])
    lines = rendered.splitlines()
    parallel = next(line for line in lines if "B" in line)
    descendant = next(line for line in lines if "D" in line)
    assert "C" in parallel
    assert parallel.index("B") == descendant.index("D")
    assert rendered.count("▼") == 3


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
    assert "cheap" in render_dag(nodes)
    assert "heavy" in render_dag(nodes)
    assert "cpus=16, memory=200.0GiB" in render_dag(nodes)


# -- nested recipes have explicitly scoped diagrams --


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
    rendered = render_dag(nodes)
    outer, inner = rendered.split("Contents of recipe 'sub':")
    assert "make" in outer and "sub" in outer and "use" in outer
    assert "sub: path <- make.path" in outer
    assert "use: path <- sub.path" in outer
    assert [line.strip() for line in inner.splitlines() if line.strip()] == ["a", "│", "│", "▼", "b"]


def test_recipes_nested_two_deep():
    a = _cab("a", In, OkOut)
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="x", step=a)])
    middle = Recipe(name="middle", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="inner", step=inner), StepRef(name="y", step=a)])
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="mid", step=middle)])
    rendered = render_dag(graph_nodes(recipe))
    assert "Contents of recipe 'mid':" in rendered
    assert "Contents of recipe 'mid' > 'inner':" in rendered
    middle, inner = rendered.split("Contents of recipe 'mid' > 'inner':")
    assert "inner" in middle and "y" in middle
    assert inner.strip() == "x"


def test_dotted_step_name_cannot_pass_for_a_nested_step():
    """Step names are free strings, and `add_loop` declares dotted ones, so a
    top-level `sub.x` waiting on `sub` must not look like one of `sub`'s own
    steps. Unqualified names and explicit scope headings keep them apart."""
    a = _cab("a", In, OkOut)
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="a", step=a)])
    recipe = Recipe(
        name="r",
        inputs_model=In,
        outputs_model=OkOut,
        steps=[
            StepRef(name="sub", step=inner),
            StepRef(name="sub.x", step=a, after=["sub"]),
            StepRef(name="last", step=a, after=["sub", "sub.x"]),
        ],
    )
    outer, inner = render_dag(graph_nodes(recipe)).split("Contents of recipe 'sub':")
    assert "sub.x" in outer and "last" in outer
    assert inner.strip() == "a"


def test_dryrun_refuses_a_step_whose_path_collides_with_a_nested_step():
    a = _cab("a", In, OkOut)
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="a", step=a)])
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="sub", step=inner), StepRef(name="sub.a", step=a)])
    with pytest.raises(RecipeGraphError, match="would share the step path 'sub.a'"):
        graph_nodes(recipe)


def test_empty_nested_recipe_is_a_leaf():
    inner = Recipe(name="inner", inputs_model=In, outputs_model=OkOut)
    recipe = Recipe(name="r", inputs_model=In, outputs_model=OkOut, steps=[StepRef(name="sub", step=inner)])
    assert [line.strip() for line in render_dag(graph_nodes(recipe)).splitlines()] == ["sub", "(recipe)"]


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


class ScatterInputs(BaseModel):
    weight: str
    prefix: str = "image"
    temp_dir: str = "tmp"


class ScatterBoundary(BaseModel):
    weights: list[str] = []


def _scatter_recipe(**params):
    recipe = Recipe(name="scatter", inputs_model=ScatterBoundary, outputs_model=OkOut)
    recipe.add_step("image", _cab("image", ScatterInputs, OkOut), scatter=["weight"], **params)
    return recipe


def test_scatter_constants_preview_retains_one_node_and_dependencies():
    recipe = _scatter_recipe(weight=["briggs -1.0", "briggs -0.5", "briggs 0.0", "briggs 0.5", "briggs 1.0"])
    recipe.add_step("consume", _cab("consume", In, OkOut))
    recipe.steps[-1].after = ["image"]
    nodes = graph_nodes(recipe)
    assert [(node.name, node.depends_on) for node in nodes] == [("image", set()), ("consume", {0})]
    compact = render_dag(nodes)
    assert "[scatter; 5 slices]" in compact and "▼" in compact
    assert "briggs" not in compact
    assert "weight = ['briggs -1.0'" in render_dag(nodes, details=True)


def test_scatter_inputref_is_resolved_only_from_supplied_boundary_values():
    recipe = _scatter_recipe(weight=InputRef(field="weights"))
    assert "[scatter]" in render_dag(graph_nodes(recipe))
    assert "InputRef(field='weights') [scatter]" in render_dag(graph_nodes(recipe), details=True)
    nodes = graph_nodes(recipe, {"weights": ["natural", "uniform"]})
    assert "[scatter; 2 slices]" in render_dag(nodes)
    assert "image: weight = ['natural', 'uniform'] [scatter; 2 slices]" in render_dag(nodes, details=True)


def test_scatter_outputref_stays_symbolic_even_with_static_output_defaults():
    class Weights(BaseModel):
        weights: list[str] = ["natural", "uniform"]

    recipe = Recipe(name="scatter", inputs_model=In, outputs_model=OkOut)
    recipe.add_step("produce", _cab("produce", In, Weights))
    recipe.add_step("image", _cab("image", ScatterInputs, OkOut), scatter=["weight"], weight=recipe.outputs.produce.weights)
    nodes = graph_nodes(recipe, {})
    assert nodes[1].depends_on == {0}
    compact = render_dag(nodes)
    assert "[scatter]" in compact and "slices" not in compact
    assert "image: weight <- produce.weights" in compact
    assert "weight = OutputRef(step='produce', field='weights')" in render_dag(nodes, details=True)


def test_scatter_outputref_selection_is_visible():
    from pathlib import Path

    from shinobi import FamilySpec, MemberRule, ProductFamily
    from shinobi.steps.schema import ParamMeta

    class Families(BaseModel):
        files: ProductFamily[Path]

    recipe = Recipe(name="scatter", inputs_model=In, outputs_model=OkOut)
    recipe.add_step(
        "produce",
        Cab(
            name="produce",
            command="x",
            inputs_model=In,
            outputs_model=Families,
            field_meta={"files": ParamMeta(family=FamilySpec(coordinates={"band": "int"}, rules=(MemberRule(path="output", coordinates={"band": 1}),)))},
        ),
    )
    recipe.add_step("image", _cab("image", ScatterInputs, OkOut), scatter=["weight"], weight=recipe.outputs.produce.files.select(band=1))
    nodes = graph_nodes(recipe)
    assert "weight <- produce.files select(band)" in render_dag(nodes)
    assert "weight <- produce.files select {'band': 1}" in render_dag(nodes, details=True)
    assert "OutputRef(step='produce', field='files', selection={'band': 1}) [scatter]" in render_dag(nodes, details=True)


@pytest.mark.parametrize("many", [False, True])
def test_selection_values_are_only_shown_in_detailed_wiring(many):
    from pathlib import Path

    from shinobi import FamilySpec, MemberRule, ProductFamily
    from shinobi.steps.schema import ParamMeta

    class Families(BaseModel):
        files: ProductFamily[Path]

    class FileInput(BaseModel):
        path: Path

    class FilesInput(BaseModel):
        paths: list[Path]

    recipe = Recipe(name="selection", inputs_model=In, outputs_model=OkOut)
    producer = Cab(
        name="produce",
        command="x",
        inputs_model=In,
        outputs_model=Families,
        field_meta={"files": ParamMeta(family=FamilySpec(coordinates={"partition": "str"}, rules=(MemberRule(path="output.fits", coordinates={"partition": "/secret/path"}),)))},
    )
    recipe.add_step("produce", producer)
    selected = recipe.outputs.produce.files.select(partition="/secret/path")
    recipe.add_step("use", _cab("use", FilesInput if many else FileInput, OkOut), **({"paths": [selected]} if many else {"path": selected}))
    nodes = graph_nodes(recipe)
    compact = render_dag(nodes)
    assert "produce.files select(partition)" in compact
    assert "/secret/path" not in compact
    assert "output.fits" not in compact
    assert "produce.files select {'partition': '/secret/path'}" in render_dag(nodes, details=True)


def test_wired_product_lists_preserve_order_and_boundary_inputs():
    class Boundary(BaseModel):
        initial: str

    class Many(BaseModel):
        paths: list[str]

    recipe = Recipe(name="wiring", inputs_model=Boundary, outputs_model=OkOut)
    recipe.add_step("produce", _cab("produce", In, PathOut))
    recipe.add_step("consume", _cab("consume", Many, OkOut), paths=[InputRef(field="initial"), recipe.outputs.produce.path])
    assert "consume: paths <- [inputs.initial, produce.path]" in render_dag(graph_nodes(recipe))


def test_zip_scatter_details_preserve_arrows_resources_and_reasons():
    recipe = Recipe(name="zip", inputs_model=In, outputs_model=OkOut)
    recipe.add_step("start", _cab("start", In, OkOut))
    cab = _cab("image", ScatterInputs, OkOut)
    cab.resources = Resources(cpus=2)
    recipe.add_step(
        "image",
        cab,
        scatter=["weight", "prefix", "temp_dir"],
        weight=["briggs -1.0", "briggs -0.5", "briggs 0.0", "briggs 0.5", "briggs 1.0"],
        prefix=["w0", "w1", "w2", "w3", "w4"],
        temp_dir=["t0", "t1", "t2", "t3", "t4"],
    )
    recipe.steps[-1].after = ["start"]
    recipe.add_step("consume", _cab("consume", In, OkOut))
    recipe.steps[-1].after = ["image"]
    recipe.add_step("peer", _cab("peer", In, OkOut))
    recipe.steps[-1].after = ["start"]
    nodes = graph_nodes(recipe)
    nodes[1].access_reasons = ("read-after-write: observation.ms",)
    compact = render_dag(nodes)
    assert "[scatter; 5 slices]" in compact and "cpus=2" in compact
    assert "read-after-write" not in compact
    assert compact.count("▼") == 3
    detailed = render_dag(nodes, details=True)
    assert "read-after-write: observation.ms" in detailed
    assert "weight = ['briggs -1.0'" in detailed
    assert "prefix = ['w0', 'w1', 'w2', 'w3', 'w4']" in detailed
    assert "temp_dir = ['t0', 't1', 't2', 't3', 't4']" in detailed


@pytest.mark.parametrize("weights,count,preview", [([], 0, "[]"), ([str(i) for i in range(8)], 8, "['0', '1', '2', '3', '4', ...]")])
def test_scatter_empty_and_truncated_previews_show_full_known_count(weights, count, preview):
    nodes = graph_nodes(_scatter_recipe(weight=weights))
    assert f"[scatter; {count} slices]" in render_dag(nodes)
    assert f"image: weight = {preview} [scatter; {count} slices]" in render_dag(nodes, details=True)


def test_zip_unknown_or_unequal_lengths_never_fabricates_a_count():
    recipe = Recipe(name="zip", inputs_model=ScatterBoundary, outputs_model=OkOut)
    recipe.add_step("image", _cab("image", ScatterInputs, OkOut), scatter=["weight", "prefix"], weight=InputRef(field="weights"), prefix=["one"])
    assert "slices" not in render_dag(graph_nodes(recipe))
    # The display does not execute or validate aggregate scatter inputs.
    recipe.steps[0].wiring.clear()
    recipe.steps[0].params["weight"] = ["a", "b"]
    assert "slices" not in render_dag(graph_nodes(recipe))


def test_nested_scatter_labels_use_parent_inputs_without_evaluating_factories():
    from pydantic import Field

    def forbidden():
        pytest.fail("rendering evaluated a default factory")

    class Boundary(BaseModel):
        weights: list[str] = Field(default_factory=forbidden)

    inner = Recipe(name="inner", inputs_model=Boundary, outputs_model=OkOut)
    inner.add_step("image", _cab("image", ScatterInputs, OkOut), scatter=["weight"], weight=InputRef(field="weights"))
    outer = Recipe(name="outer", inputs_model=Boundary, outputs_model=OkOut)
    outer.add_step("sub", inner, weights=["natural", "uniform"])
    nodes = graph_nodes(outer)
    compact = render_dag(nodes)
    assert "Contents of recipe 'sub':" in compact and "[scatter; 2 slices]" in compact
    assert "weight = ['natural', 'uniform']" in render_dag(nodes, details=True)


def test_nested_recipe_scatter_does_not_treat_whole_lists_as_slice_inputs():
    inner = _scatter_recipe(weight=InputRef(field="weights"))
    outer = Recipe(name="outer", inputs_model=In, outputs_model=OkOut)
    outer.add_step("sub", inner, scatter=["weights"], weights=[["a"], ["b"]])
    nodes = graph_nodes(outer)
    compact = render_dag(nodes)
    assert compact.count("[scatter; 2 slices]") == 1
    assert "[scatter]" in compact
    assert "weight = InputRef(field='weights')" in render_dag(nodes, details=True)


def test_scatter_preview_bounds_long_values_and_escapes_newlines():
    rendered = render_dag(graph_nodes(_scatter_recipe(weight=["first\nsecond", "x" * 1000])), details=True)
    assert "first\\nsecond" in rendered
    assert "..." in rendered
    assert max(map(len, rendered.splitlines())) < 200
