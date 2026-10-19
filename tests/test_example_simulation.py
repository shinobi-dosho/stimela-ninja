"""Smoke tests for examples/example-simulation.py.

The example imports `dosho` (the native shinobi cab repository) for its real
wsclean/quartical/simms cabs. dosho is now a plain dependency group --
`uv sync --group examples` -- since it dropped the [tool.uv.sources] redirect
that used to make it unresolvable alongside this project (dosho #42). It is
still not in CI's default `--group dev`, so skip cleanly rather than erroring
out collection when it is absent.

`RecordingBackend` intercepts every *Cab* step, including the ones carrying
`backend="native"`, so wsclean and quartical need nothing installed. It cannot
intercept a **pystep**: as of simms 3.0, dosho's `skysim`/`telsim` are
`@shinobi.pystep` `StepRef`s rather than binary Cabs, and a pystep runs its
own Python body (`ctx.import_callable("runit", "simms.apps.telsim")`) instead of
dispatching through a backend. A backend override cannot reach it, so the one
test that dispatches those steps needs the real `simms` module -- which has no
docker image and is not on PyPI, hence still a manual install:

    uv pip install --no-deps simms @ git+https://github.com/wits-cfa/simms.git
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("dosho")

from shinobi.backends.recording import RecordingBackend  # noqa: E402
from shinobi.dataset_access import scope_tree_has_dataset_contract  # noqa: E402
from shinobi.dag import graph_nodes, render_dag  # noqa: E402
from shinobi.policies import build_argv  # noqa: E402
from shinobi.steps import Recipe  # noqa: E402
from shinobi.steps.dispatch import _dispatch  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example-simulation.py"


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
    spec = importlib.util.spec_from_file_location("example_simulation", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["example_simulation"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_builds_recipe():
    mod = load_example()
    assert isinstance(mod.recipe, Recipe)
    assert [ref.name for ref in mod.recipe.steps] == [
        "make_ms",
        "simulate",
        "image_sim",
        "calibrate",
        "image_robust_2",
        "image_robust_0",
        "image_robust_m2",
    ]


def test_dryrun_dag_renders():
    mod = load_example()
    rendered = render_dag(graph_nodes(mod.recipe))
    for name in ["make_ms", "simulate", "image_sim", "calibrate", "image_robust_2"]:
        assert name in rendered


def test_quartical_command_requests_corrected_data():
    mod = load_example()
    calibration = next(ref for ref in mod.recipe.steps if ref.name == "calibrate")
    argv = build_argv(calibration.step, {**calibration.params, "input_ms_path": "example-simulation.ms"})
    assert argv[0] == "goquartical"
    for option in (
        "input_ms.path=example-simulation.ms",
        "input_model.recipe=MODEL_DATA",
        "output.products=[corrected_data]",
        "output.columns=[CORRECTED_DATA]",
        "solver.terms=[G]",
        "G.type=diag_complex",
    ):
        assert option in argv


@pytest.mark.skipif(
    importlib.util.find_spec("simms") is None,
    reason="dosho's skysim/telsim are pysteps, not Cabs -- a backend override cannot intercept them, so dispatching this recipe imports simms for real",
)
def test_recipe_dispatches_with_correct_argv_shape(monkeypatch, tmp_path):
    # Run somewhere disposable. When simms *is* installed the pysteps below
    # are not intercepted by anything -- telsim writes a real MeasurementSet
    # (~6 MB) and skysim computes real visibilities into it. Without this the
    # repo gains an untracked example-simulation.ms/ every run.
    monkeypatch.chdir(tmp_path)
    mod = load_example()
    # Newer dosho cabs declare strict MSv2 contracts. This example also has
    # path-only MS passthroughs, so the lifecycle can refuse dispatch before
    # there is any argv to inspect.
    if scope_tree_has_dataset_contract(mod.recipe):
        pytest.skip("installed dosho declares strict MSv2 contracts that this example's path-only steps cannot share")
    recorder = RecordingBackend()
    # wsclean/quartical are Cabs, so intercepting "native" keeps them from
    # shelling out (that override beats any backend passed to _dispatch).
    # Shadow it via monkeypatch so the override can't leak into later tests
    # that run real native commands (_STEP_BACKENDS is module-global).
    from shinobi.steps.dispatch import _STEP_BACKENDS

    monkeypatch.setitem(_STEP_BACKENDS, "native", recorder)

    # "native" rather than a "recording" route: a route name the dataset
    # lifecycle has no adapter for is refused whenever a step carries a strict
    # dataset contract. The override above already routes every native Cab here.
    res = _dispatch(mod.recipe, None, backend="native")
    assert res.success
    assert str(res.outputs.image) == "example-sim-robust2-image.fits"

    calls_by_name: dict[str, list[list[str]]] = {}
    for cab, argv, _ in recorder.calls:
        calls_by_name.setdefault(cab.name, []).append(argv)
    # Only the Cab steps reach a backend. simms-telsim/simms-skysim are
    # pysteps, which run their own body instead of dispatching through one,
    # so the recorder never sees them -- asserting otherwise was asserting
    # that dosho had not yet moved simms to @shinobi.pystep, which it has.
    assert set(calls_by_name) == {"quartical", "wsclean"}
    # image_sim (pre-calibration, populates MODEL_DATA) + one per Briggs
    # robust value.
    assert len(calls_by_name["wsclean"]) == 4

    # The simms steps are pysteps, so there is no argv to inspect -- what
    # remains checkable, and is the part worth checking, is that their output
    # still flows through the wiring into the Cab steps. QuartiCal is wired the
    # MS telsim produced.
    quartical_argv = calls_by_name["quartical"][0]
    assert "input_ms.path=example-simulation.ms" in quartical_argv

    # wsclean's own real outputs are FITS image products (image/image_mfs/
    # ...), not the MS -- image_sim's `input_ms` is an *outputs_model-only*
    # field (see wsclean_with_model in the example itself), resolved via
    # an implicit "{ms[0]}" template indexing into wsclean's own real
    # (list) `ms` input, never added to inputs_model. This is a
    # regression guard for a real bug: adding it to inputs_model too (the
    # shape caracal2's own line/selfcal workers use) makes build_argv
    # emit it as a bogus `-input-ms <path>` flag real wsclean doesn't
    # recognise.
    image_sim_argv = calls_by_name["wsclean"][0]
    assert "-input-ms" not in image_sim_argv
    assert "-data-column" in image_sim_argv
    assert image_sim_argv[image_sim_argv.index("-data-column") + 1] == "DATA"
    assert image_sim_argv[-1] == "example-simulation.ms"

    # QuartiCal uses dotted key=value options and writes corrected visibilities
    # only when the product and its MS column are requested explicitly.
    assert quartical_argv[0] == "goquartical"
    assert "input_model.recipe=MODEL_DATA" in quartical_argv
    assert "output.products=[corrected_data]" in quartical_argv
    assert "output.columns=[CORRECTED_DATA]" in quartical_argv
    assert "solver.terms=[G]" in quartical_argv
    assert "G.type=diag_complex" in quartical_argv

    # wsclean (robust=2, the first *post-calibration* image step, i.e.
    # index 1 -- index 0 is image_sim, checked above): repeat_as_tokens
    # for -size (bare tokens, not comma-joined), a real Union[str,
    # Tuple[str, float]] `weight` value, real nom_de_guerre flags
    # (-data-column/-name), ms wired all the way from QuartiCal's real `ms`
    # output.
    wsclean_argv = calls_by_name["wsclean"][1]
    assert "-size" in wsclean_argv
    size_i = wsclean_argv.index("-size")
    assert wsclean_argv[size_i + 1 : size_i + 3] == ["2048", "2048"]
    assert "-weight" in wsclean_argv
    weight_i = wsclean_argv.index("-weight")
    assert wsclean_argv[weight_i + 1 : weight_i + 3] == ["briggs", "2.0"]
    assert "-data-column" in wsclean_argv
    assert "-name" in wsclean_argv
    assert wsclean_argv[-1] == "example-simulation.ms"
