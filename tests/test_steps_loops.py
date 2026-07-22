"""Declared loops: `Recipe.add_loop` unrolls a body into the parent graph and
short-circuits once the convergence sentinel appears (see
`shinobi.steps.loops`).
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.graph import RecipeGraphError, build_graph
from shinobi.results import BackendRun
from shinobi.steps import Cab, InputRef, OutputRef, Recipe, register_step_backend
from shinobi.steps.dispatch import _dispatch


class WorkIn(BaseModel):
    ms: Path


class WorkOut(BaseModel):
    ms: Path | None = None


class AssessIn(BaseModel):
    ms: Path
    flag: Path


class AssessOut(BaseModel):
    flag: Path | None = None


class BodyIn(BaseModel):
    ms: Path
    flag: Path


class BodyOut(BaseModel):
    ms: Path | None = None
    converged: Path | None = None


class ConvergeAfter:
    """Fake backend: records every step it runs, and writes the sentinel file
    once `assess` has run `after` times -- i.e. the loop converges then.
    """

    def __init__(self, after: int):
        self.after = after
        self.calls: list[str] = []

    def run(self, cab, argv, inputs, **kwargs):
        label = kwargs.get("label") or cab.name
        self.calls.append(label)
        if cab.name == "assess":
            if sum(1 for c in self.calls if "assess" in c) >= self.after:
                Path(inputs["flag"]).write_text("converged")
        return BackendRun(0, "", "")

    @property
    def ran(self) -> list[str]:
        return self.calls


def make_body(backend_name: str) -> Recipe:
    """A two-step loop body: `work` then `assess`, whose `converged` output is
    the sentinel `assess` writes.
    """
    work = Cab(name="work", command="work", inputs_model=WorkIn, outputs_model=WorkOut, backend=backend_name)
    assess = Cab(name="assess", command="assess", inputs_model=AssessIn, outputs_model=AssessOut, backend=backend_name)
    body = Recipe(name="cycle", inputs_model=BodyIn, outputs_model=BodyOut)
    body.add_step("work", work, ms=InputRef(field="ms"))
    body.add_step("assess", assess, ms=OutputRef(step="work", field="ms"), flag=InputRef(field="flag"))
    body.set_output("ms", OutputRef(step="work", field="ms"))
    body.set_output("converged", OutputRef(step="assess", field="flag"))
    return body


def make_recipe(backend_name: str, tmp_path: Path, max_iter: int = 4) -> Recipe:
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    recipe.add_loop(
        "selfcal",
        make_body(backend_name),
        max_iter=max_iter,
        until="converged",
        carry={"ms": "ms"},
        ms=InputRef(field="ms"),
        flag=tmp_path / "converged.flag",
    )
    return recipe


def test_unrolls_body_into_flattened_steps(tmp_path):
    recipe = make_recipe("loop-unroll", tmp_path, max_iter=3)
    assert [ref.name for ref in recipe.steps] == [
        "selfcal.1.work",
        "selfcal.1.assess",
        "selfcal.2.work",
        "selfcal.2.assess",
        "selfcal.3.work",
        "selfcal.3.assess",
    ]


def test_iterations_are_chained_by_real_graph_edges(tmp_path):
    """The unrolled chain must be real dependency edges, not bookkeeping --
    otherwise iterations land in the ready set together and race.
    """
    recipe = make_recipe("loop-edges", tmp_path, max_iter=3)
    graph = build_graph(recipe)
    by_name = {name: i for i, name in enumerate(graph.names)}

    # Only the very first step is dependency-free.
    assert [name for name, i in by_name.items() if not graph.deps[i]] == ["selfcal.1.work"]
    # Carry edge: iteration 2's work reads iteration 1's work output...
    assert by_name["selfcal.1.work"] in graph.deps[by_name["selfcal.2.work"]]
    # ...and an ordering edge makes it wait for iteration 1's *sentinel*,
    # which the carry alone would not (assess runs after work).
    assert by_name["selfcal.1.assess"] in graph.deps[by_name["selfcal.2.work"]]


def test_loop_outputs_resolve_to_the_final_iteration(tmp_path):
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    loop = recipe.add_loop(
        "selfcal",
        make_body("loop-outputs"),
        max_iter=5,
        until="converged",
        carry={"ms": "ms"},
        ms=InputRef(field="ms"),
        flag=tmp_path / "f",
    )
    assert loop.outputs.ms == OutputRef(step="selfcal.5.work", field="ms")
    assert loop.outputs.converged == OutputRef(step="selfcal.5.assess", field="flag")


def test_converged_iterations_do_no_work(tmp_path):
    """The point of the feature: once the sentinel exists, later iterations
    complete without touching the backend.
    """
    backend = ConvergeAfter(after=2)
    register_step_backend("loop-run", backend)
    recipe = make_recipe("loop-run", tmp_path, max_iter=4)

    result = _dispatch(recipe, None, ms=tmp_path / "input.ms")

    assert result.returncode == 0
    # Iterations 1 and 2 ran (2 steps each); 3 and 4 were skipped entirely.
    assert len(backend.ran) == 4
    skipped = {name: sub.skipped for name, sub in result.sub_results.items()}
    assert skipped == {
        "selfcal.1.work": False,
        "selfcal.1.assess": False,
        "selfcal.2.work": False,
        "selfcal.2.assess": False,
        "selfcal.3.work": True,
        "selfcal.3.assess": True,
        "selfcal.4.work": True,
        "selfcal.4.assess": True,
    }


def test_skipped_steps_pass_the_converged_outputs_through(tmp_path):
    backend = ConvergeAfter(after=1)
    register_step_backend("loop-pass", backend)
    recipe = make_recipe("loop-pass", tmp_path, max_iter=3)

    result = _dispatch(recipe, None, ms=tmp_path / "input.ms")

    converged = result.sub_results["selfcal.1.work"].outputs.ms
    assert result.sub_results["selfcal.3.work"].outputs.ms == converged
    # `kind` must stay the scope's real kind -- provenance replay asserts on it.
    assert result.sub_results["selfcal.3.work"].kind == "cab"


def test_loop_runs_every_iteration_when_it_never_converges(tmp_path):
    backend = ConvergeAfter(after=99)
    register_step_backend("loop-full", backend)
    recipe = make_recipe("loop-full", tmp_path, max_iter=3)

    result = _dispatch(recipe, None, ms=tmp_path / "input.ms")

    assert len(backend.ran) == 6
    assert not any(sub.skipped for sub in result.sub_results.values())


def test_rejects_non_path_sentinel(tmp_path):
    class BoolBodyOut(BaseModel):
        ms: Path | None = None
        converged: bool = False

    body = make_body("loop-bad")
    body.outputs_model = BoolBodyOut
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    with pytest.raises(ValueError, match="must be a path-typed output"):
        recipe.add_loop("selfcal", body, max_iter=2, until="converged", carry={"ms": "ms"}, ms=InputRef(field="ms"))


def test_rejects_unknown_carry_field(tmp_path):
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    with pytest.raises(ValueError, match="carry key 'nope' is not an output"):
        recipe.add_loop("selfcal", make_body("loop-bad"), max_iter=2, until="converged", carry={"nope": "ms"}, ms=InputRef(field="ms"))


def test_rejects_max_iter_below_one(tmp_path):
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    with pytest.raises(ValueError, match="max_iter=0"):
        recipe.add_loop("selfcal", make_body("loop-bad"), max_iter=0, until="converged", carry={"ms": "ms"}, ms=InputRef(field="ms"))


def test_rejects_colliding_step_names(tmp_path):
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    recipe.add_step("selfcal.1.work", Cab(name="x", command="x", inputs_model=WorkIn, outputs_model=WorkOut))
    with pytest.raises(ValueError, match="already exists"):
        recipe.add_loop("selfcal", make_body("loop-bad"), max_iter=2, until="converged", carry={"ms": "ms"}, ms=InputRef(field="ms"))


def test_after_edge_must_name_a_real_step():
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    recipe.add_step("a", Cab(name="x", command="x", inputs_model=WorkIn, outputs_model=WorkOut))
    recipe.steps[0].after = ["ghost"]
    with pytest.raises(RecipeGraphError, match="after='ghost'"):
        build_graph(recipe)


def test_manifest_from_an_early_converging_run_still_replays(tmp_path):
    """`kind` must stay the scope's real kind on a skipped step: replay
    asserts the manifest's kind still matches the target's shape, so a
    "skipped" kind would make every early-converging run unreplayable.
    """
    from shinobi.provenance import apply_manifest_pins, build_manifest

    backend = ConvergeAfter(after=1)
    register_step_backend("loop-replay", backend)
    recipe = make_recipe("loop-replay", tmp_path, max_iter=3)

    result = _dispatch(recipe, None, ms=tmp_path / "input.ms")
    manifest = build_manifest(result, backend="loop-replay")

    # Every declared iteration is recorded, with the skipped ones marked.
    assert [(s.name, s.skipped) for s in manifest.root.steps] == [
        ("selfcal.1.work", False),
        ("selfcal.1.assess", False),
        ("selfcal.2.work", True),
        ("selfcal.2.assess", True),
        ("selfcal.3.work", True),
        ("selfcal.3.assess", True),
    ]
    apply_manifest_pins(recipe, manifest.root)  # must not raise


def test_dryrun_renders_every_declared_iteration(tmp_path):
    """The graph shows what is *declared*, not what will run -- all
    max_iter iterations are real nodes.
    """
    from shinobi.dag import graph_nodes, render_dag

    recipe = make_recipe("loop-dag", tmp_path, max_iter=3)
    nodes = graph_nodes(recipe)
    assert [n.name for n in nodes] == [
        "selfcal.1.work",
        "selfcal.1.assess",
        "selfcal.2.work",
        "selfcal.2.assess",
        "selfcal.3.work",
        "selfcal.3.assess",
    ]
    assert "selfcal.3.assess" in render_dag(nodes)


def test_loop_demo_example_runs(tmp_path, monkeypatch):
    """The runnable example actually converges early: three cycles do work,
    the remaining two are skipped. Uses only `sh`, so it runs anywhere.
    """
    import importlib.util
    import sys

    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / "examples" / "loop_demo.py"
    spec = importlib.util.spec_from_file_location("loop_demo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loop_demo"] = module
    spec.loader.exec_module(module)

    result = module.pipeline()

    assert result.returncode == 0
    assert (tmp_path / "loop-demo-work.txt").read_text().splitlines() == [
        "refined on cycle 1",
        "refined on cycle 2",
        "refined on cycle 3",
    ]
    ran = [n for n, s in result.sub_results.items() if not s.skipped]
    assert ran == [
        "prepare",
        "refine_until_good.1.refine",
        "refine_until_good.1.assess",
        "refine_until_good.2.refine",
        "refine_until_good.2.assess",
        "refine_until_good.3.refine",
        "refine_until_good.3.assess",
    ]


class SelfIn(BaseModel):
    ms: Path
    flag: Path


class SelfOut(BaseModel):
    ms: Path | None = None
    flag: Path | None = None


def test_single_cab_body_is_one_step_per_iteration(tmp_path):
    """A non-Recipe body needs no flattening: `selfcal.2`, not `selfcal.2.x`."""

    class OneShot:
        def __init__(self):
            self.n = 0

        def run(self, cab, argv, inputs, **kwargs):
            self.n += 1
            if self.n >= 2:
                Path(inputs["flag"]).write_text("done")
            return BackendRun(0, "", "")

    backend = OneShot()
    register_step_backend("loop-single", backend)
    cab = Cab(name="solo", command="solo", inputs_model=SelfIn, outputs_model=SelfOut, backend="loop-single")
    recipe = Recipe(name="outer", inputs_model=WorkIn, outputs_model=WorkOut)
    recipe.add_loop(
        "cycle",
        cab,
        max_iter=4,
        until="flag",
        carry={"ms": "ms"},
        ms=InputRef(field="ms"),
        flag=tmp_path / "single.flag",
    )
    assert [ref.name for ref in recipe.steps] == ["cycle.1", "cycle.2", "cycle.3", "cycle.4"]

    result = _dispatch(recipe, None, ms=tmp_path / "in.ms")
    assert backend.n == 2
    assert [s.skipped for s in result.sub_results.values()] == [False, False, True, True]


def test_iterations_stay_ordered_under_parallel_workers(tmp_path):
    """With max_workers > 1 a missing inter-iteration edge would let cycles
    run concurrently and the skip read a stale sentinel. The ordering edge
    must keep them strictly sequential.
    """
    backend = ConvergeAfter(after=2)
    register_step_backend("loop-parallel", backend)
    recipe = make_recipe("loop-parallel", tmp_path, max_iter=4)
    recipe.max_workers = 4

    result = _dispatch(recipe, None, ms=tmp_path / "input.ms")

    assert [name.removeprefix("outer.") for name in backend.ran] == [
        "selfcal.1.work",
        "selfcal.1.assess",
        "selfcal.2.work",
        "selfcal.2.assess",
    ]
    assert result.returncode == 0
