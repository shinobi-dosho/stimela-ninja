from shinobi.dag import TraceStep, find_dependencies, placeholder, render_dag


def test_placeholder_is_found_as_exact_value():
    deps = find_dependencies({"path": placeholder(3, "image")})
    assert deps == {3}


def test_placeholder_is_found_inside_a_concatenated_string():
    # recipes often combine a prior output into a bigger string, e.g.
    # f"MODEL_DATA+{result.image}" in examples/ninja_recipe.py
    deps = find_dependencies({"model": f"MODEL_DATA+{placeholder(2, 'lsm')}"})
    assert deps == {2}


def test_placeholder_is_found_inside_a_list():
    deps = find_dependencies({"mslist": [placeholder(0, "ms"), "plain/path.ms"]})
    assert deps == {0}


def test_multiple_placeholders_across_params_all_detected():
    deps = find_dependencies({"a": placeholder(1, "x"), "b": placeholder(4, "y")})
    assert deps == {1, 4}


def test_no_placeholder_means_no_dependencies():
    assert find_dependencies({"threshold": 6.5, "name": "plain string"}) == set()


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
    assert "▼" in out
    # Commit's line comes before Build App's in the rendered text
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

    # Build App and Run Tests are siblings on the same row (fan-out)
    sibling_line = next(ln for ln in lines if "Build App" in ln and "Run Tests" in ln)
    assert sibling_line

    # a fan-out bracket (single parent -> two children) appears
    assert any("┌" in ln and "┐" in ln and "┴" in ln for ln in lines)
    # a fan-in bracket (two parents -> single child) appears
    assert any("└" in ln and "┘" in ln and "┬" in ln for ln in lines)

    assert out.index("Commit") < out.index("Build App") < out.index("Deploy to QA")


def test_render_chained_diamonds_matches_mockup_shape():
    steps = [
        TraceStep(id=0, name="Commit", depends_on=set()),
        TraceStep(id=1, name="Build App", depends_on={0}),
        TraceStep(id=2, name="Run Tests", depends_on={0}),
        TraceStep(id=3, name="Deploy to QA", depends_on={1, 2}),
        TraceStep(id=4, name="Approve to Staging", depends_on={3}),
        TraceStep(id=5, name="Run Load Tests", depends_on={3}),
        TraceStep(id=6, name="Deploy to Production", depends_on={4, 5}),
    ]
    out = render_dag(steps)
    for name in (
        "Commit",
        "Build App",
        "Run Tests",
        "Deploy to QA",
        "Approve to Staging",
        "Run Load Tests",
        "Deploy to Production",
    ):
        assert f"[ {name} ]" in out
    # two independent diamonds -> two fan-out and two fan-in brackets
    lines = out.splitlines()
    fan_outs = [ln for ln in lines if "┌" in ln and "┐" in ln and "┴" in ln]
    fan_ins = [ln for ln in lines if "└" in ln and "┘" in ln and "┬" in ln]
    assert len(fan_outs) == 2
    assert len(fan_ins) == 2


def test_render_falls_back_to_plain_chain_without_false_fan_structure():
    # C depends on A only, not on B -- B and C don't share a dependency
    # set with each other or exactly match a previous batch's ids, so this
    # must not be rendered as a clean fan-in/fan-out.
    steps = [
        TraceStep(id=0, name="A", depends_on=set()),
        TraceStep(id=1, name="B", depends_on={0}),
        TraceStep(id=2, name="C", depends_on={0}),
        TraceStep(id=3, name="D", depends_on={1}),  # only depends on B, not C
    ]
    out = render_dag(steps)
    lines = out.splitlines()
    # no fan-in bracket for the B/C -> D transition (D isn't a clean merge of {1, 2})
    assert not any("└" in ln and "┘" in ln for ln in lines[-4:])
