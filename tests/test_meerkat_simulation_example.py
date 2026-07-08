"""Smoke tests for examples/meerkat_simulation.py -- must pass with none of
simms/wsclean/cubical actually installed (RecordingBackend intercepts every
step, including the two that have `backend="native"` baked onto their Cab).

The example itself imports `dosho` (the native shinobi cab repository) for
its real wsclean/cubical/simms cabs -- only installed via the optional
`examples` dependency group (`uv sync --group examples`), not CI's default
`--group dev`. Skip cleanly rather than erroring out collection when it
isn't present.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("dosho")

from shinobi.backends.recording import RecordingBackend  # noqa: E402
from shinobi.dag import graph_nodes, render_dag  # noqa: E402
from shinobi.steps import Recipe, register_step_backend  # noqa: E402
from shinobi.steps.dispatch import _dispatch  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "meerkat_simulation.py"


def load_example():
    # Register in sys.modules *before* exec_module -- every module here
    # uses `from __future__ import annotations`, so pydantic resolves
    # field type annotations lazily against `sys.modules[__module__]`
    # (matching what `shinobi.cli._resolve_target` itself does for a real
    # `ninja run`; skipping this step here would make outputs_model
    # construction fail with a "not fully defined" PydanticUserError as
    # soon as a field actually gets a real (non-None) value -- which only
    # started happening once this example's `image` output resolved to a
    # real path via dosho's wsclean implicit-template outputs).
    spec = importlib.util.spec_from_file_location("meerkat_simulation", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["meerkat_simulation"] = mod
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
    assert str(res.outputs.image) == "meerkat-sim-robust2-image.fits"

    calls_by_name: dict[str, list[list[str]]] = {}
    for cab, argv, _ in recorder.calls:
        calls_by_name.setdefault(cab.name, []).append(argv)
    assert set(calls_by_name) == {"simms-telsim", "simms-skysim", "cubical", "wsclean"}
    assert len(calls_by_name["wsclean"]) == 3  # one per Briggs robust value

    # multi-word `command` split + positional `ms` (no --ms flag, bare
    # value last) for both simms steps.
    telsim_argv = calls_by_name["simms-telsim"][0]
    assert telsim_argv[:2] == ["simms", "telsim"]
    assert "--ms" not in telsim_argv
    assert "--telescope" in telsim_argv
    assert telsim_argv[-1] == "meerkat_simulation.ms"

    skysim_argv = calls_by_name["simms-skysim"][0]
    assert skysim_argv[:2] == ["simms", "skysim"]
    assert "--ms" not in skysim_argv
    assert "--ascii-sky" in skysim_argv
    assert skysim_argv[skysim_argv.index("--ascii-sky") + 1] == str(
        mod._SIMMS_DIR / "testsky.txt"
    )
    # skysim's ms is wired from telsim's own (positional) ms output.
    assert skysim_argv[-1] == "meerkat_simulation.ms"

    # real dosho cubical: flattened flag names, per-Jones-term
    # ParamPattern-matched extras (lowercase g-solvable/g-type -- real
    # CubiCal per-term CLI flags are always lowercase regardless of the
    # term label's own case, unlike --sol-jones' own uppercase "G" value),
    # a real `ms` output (implicit={data_ms} passthrough) wired all the
    # way into wsclean below.
    cubical_argv = calls_by_name["cubical"][0]
    assert cubical_argv[0] == "gocubical"
    assert "--data-ms" in cubical_argv
    assert "--sol-jones" in cubical_argv
    assert "--g-solvable" in cubical_argv
    i = cubical_argv.index("--g-type")
    assert cubical_argv[i : i + 2] == ["--g-type", "complex-2x2"]

    # wsclean (robust=2, the first image step): repeat_as_tokens for
    # -size (bare tokens, not comma-joined), a real Union[str, Tuple[str,
    # float]] `weight` value, real nom_de_guerre flags (-data-column/
    # -name), ms wired all the way from cubical's real `ms` output.
    wsclean_argv = calls_by_name["wsclean"][0]
    assert "-size" in wsclean_argv
    size_i = wsclean_argv.index("-size")
    assert wsclean_argv[size_i + 1 : size_i + 3] == ["4096", "4096"]
    assert "-weight" in wsclean_argv
    weight_i = wsclean_argv.index("-weight")
    assert wsclean_argv[weight_i + 1 : weight_i + 3] == ["briggs", "2.0"]
    assert "-data-column" in wsclean_argv
    assert "-name" in wsclean_argv
    assert wsclean_argv[-1] == "meerkat_simulation.ms"
