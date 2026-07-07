"""Smoke tests for examples/meerkat_simulation.py -- must pass with none of
simms/wsclean/cubical actually installed (RecordingBackend intercepts every
step, including the two that have `backend="native"` baked onto their Cab).
"""

import importlib.util
from pathlib import Path

from shinobi.backends.recording import RecordingBackend
from shinobi.dag import graph_nodes, render_dag
from shinobi.steps import Recipe, register_step_backend
from shinobi.steps.dispatch import _dispatch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "meerkat_simulation.py"


def load_example():
    # No `dynamic_schema` warning expected here anymore: wsclean.yml's
    # `dynamic_schema` now resolves to a real (validation-only)
    # `output_patterns` entry (see shinobi.loaders.cultcargo's
    # `_dynamic_output_patterns`), which suppresses the "possibly
    # incomplete schema" warning `_build_cabdef` used to emit unconditionally.
    spec = importlib.util.spec_from_file_location("meerkat_simulation", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_builds_recipe():
    mod = load_example()
    assert isinstance(mod.recipe, Recipe)
    assert [ref.name for ref in mod.recipe.steps] == [
        "make_ms",
        "simulate",
        "calibrate",
        "image_robust_2",
        "image_robust_0",
        "image_robust_m2",
    ]


def test_dryrun_dag_renders():
    mod = load_example()
    rendered = render_dag(graph_nodes(mod.recipe))
    for name in ["make_ms", "simulate", "calibrate", "image_robust_2"]:
        assert name in rendered


def test_recipe_dispatches_with_correct_argv_shape():
    mod = load_example()
    recorder = RecordingBackend()
    # telsim/skysim have backend="native" baked onto the Cab (no docker
    # image exists for simms yet) -- that beats any backend override
    # passed to _dispatch, so intercept "native" too, not just a custom
    # name, to guarantee this test never shells out regardless of whether
    # simms happens to be installed in the current environment.
    register_step_backend("native", recorder)
    register_step_backend("recording", recorder)

    res = _dispatch(mod.recipe, None, backend="recording")
    assert res.success

    calls_by_name: dict[str, list[list[str]]] = {}
    for cab, argv, _ in recorder.calls:
        calls_by_name.setdefault(cab.name, []).append(argv)
    assert set(calls_by_name) == {"telsim", "skysim", "cubical", "wsclean"}
    assert len(calls_by_name["wsclean"]) == 3  # one per Briggs robust value

    # multi-word `command` split + positional `ms` (no --ms flag, bare
    # value last) for both simms steps.
    telsim_argv = calls_by_name["telsim"][0]
    assert telsim_argv[:2] == ["simms", "telsim"]
    assert "--ms" not in telsim_argv
    assert "--telescope" in telsim_argv
    assert telsim_argv[-1] == "meerkat_simulation.ms"

    skysim_argv = calls_by_name["skysim"][0]
    assert skysim_argv[:2] == ["simms", "skysim"]
    assert "--ms" not in skysim_argv
    assert "--ascii-sky" in skysim_argv
    assert skysim_argv[skysim_argv.index("--ascii-sky") + 1] == str(
        mod._SIMMS_DIR / "testsky.txt"
    )
    # skysim's ms is wired from telsim's own (positional) ms output.
    assert skysim_argv[-1] == "meerkat_simulation.ms"

    # wsclean (robust=2, the first image step): repeat_as_tokens for
    # -size/-weight (bare tokens, not comma-joined), real nom_de_guerre
    # flags (-data-column/-name), ms wired all the way from calibrate's
    # data_ms passthrough output.
    wsclean_argv = calls_by_name["wsclean"][0]
    assert "-size" in wsclean_argv
    size_i = wsclean_argv.index("-size")
    assert wsclean_argv[size_i + 1 : size_i + 3] == ["4096", "4096"]
    assert "-weight" in wsclean_argv
    weight_i = wsclean_argv.index("-weight")
    assert wsclean_argv[weight_i + 1 : weight_i + 3] == ["briggs", "2"]
    assert "-data-column" in wsclean_argv
    assert "-name" in wsclean_argv
    assert wsclean_argv[-1] == "meerkat_simulation.ms"

    assert calls_by_name["cubical"][0][0] == "gocubical"
