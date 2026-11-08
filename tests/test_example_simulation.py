"""Declaration, argv and optional native-dispatch checks for the simulation example.

Dosho is a runtime dependency. The example explicitly selects native execution
for its adapted simms functions, so the dispatch check needs simms installed.
Argv checks run independently of simms and dataset-lifecycle eligibility.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from shinobi.backends.recording import RecordingBackend
from shinobi.dataset_access import scope_tree_has_dataset_contract
from shinobi.dag import graph_nodes, render_dag
from shinobi.policies import build_argv
from shinobi.steps import Recipe
from shinobi.steps.dispatch import _dispatch

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


IMAGING_CASES = [
    ("image_sim", "DATA", 1024, 1.5, "example-simdata", "simulate", "ms"),
    ("image_robust_2", "CORRECTED_DATA", 2048, 2.0, "example-sim-robust2", "calibrate", "ms"),
    ("image_robust_0", "CORRECTED_DATA", 2048, 0.0, "example-sim-robust0", "calibrate", "ms"),
    ("image_robust_m2", "CORRECTED_DATA", 2048, -2.0, "example-sim-robust-2", "calibrate", "ms"),
]


def assert_imaging_argv(argv, column, size, weight, prefix, ms):
    assert argv[0] == "wsclean"
    assert "-input-ms" not in argv
    assert argv[argv.index("-size") + 1 : argv.index("-size") + 3] == [str(size), str(size)]
    assert argv[argv.index("-weight") + 1 : argv.index("-weight") + 3] == ["briggs", str(weight)]
    assert argv[argv.index("-data-column") + 1] == column
    assert argv[argv.index("-name") + 1] == prefix
    assert argv[-1] == ms


@pytest.mark.parametrize(
    "name,column,size,weight,prefix,producer,field",
    IMAGING_CASES,
)
def test_imaging_argv_and_declared_ms_wiring(name, column, size, weight, prefix, producer, field):
    from shinobi import OutputRef

    mod = load_example()
    ref = next(ref for ref in mod.recipe.steps if ref.name == name)
    assert ref.wiring["ms"] == [OutputRef(step=producer, field=field)]
    argv = build_argv(ref.step, {**ref.params, "ms": ["example-simulation.ms"]})
    assert "input_ms" not in ref.step.inputs_model.model_fields
    assert_imaging_argv(argv, column, size, weight, prefix, "example-simulation.ms")


@pytest.mark.skipif(
    importlib.util.find_spec("simms") is None,
    reason="this example selects native simms functions, so dispatch requires simms installed",
)
def test_native_recipe_dispatches_with_wired_ms_and_image_output(monkeypatch, tmp_path):
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
    # that dosho had not yet moved simms to Python functions, which it has.
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

    for argv, (_, column, size, weight, prefix, _, _) in zip(calls_by_name["wsclean"], IMAGING_CASES, strict=True):
        assert_imaging_argv(argv, column, size, weight, prefix, "example-simulation.ms")
