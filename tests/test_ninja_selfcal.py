"""Smoke tests for examples/ninja_selfcal.py -- a *declared DAG* example.

Unlike examples/example-simulation.py, this one is deliberately standalone:
it vendors its cab schemas (examples/cultcargo/, examples/stimela_classic/)
and hand-declares breizorro/cubical, so it imports *no* `dosho`. That means
these tests run unconditionally in CI's default dependency group -- there is
nothing to `importorskip` -- and catch shinobi API drift the moment it
happens.

The example's contract is dryrun/DAG rendering, not execution: its stub
wsclean produces no real `restored` output, so a full `_dispatch` can't run
end-to-end (that's what dosho's implicit-template outputs are for, and why
the simulation example -- not this one -- exercises dispatch). These tests
therefore assert what the two documented commands actually do:

    python examples/ninja_selfcal.py            # renders the DAG
    ninja run examples/ninja_selfcal.py:selfcal --dryrun
"""

import importlib.util
import sys
from pathlib import Path

from shinobi.dag import graph_nodes, render_dag
from shinobi.policies import build_argv
from shinobi.steps import Recipe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ninja_selfcal.py"


def load_example():
    # Register in sys.modules *before* exec_module -- the example uses
    # `from __future__ import annotations`, so pydantic resolves field type
    # annotations lazily against `sys.modules[__module__]` (mirroring what
    # shinobi.cli._resolve_target does for a real `ninja run`).
    spec = importlib.util.spec_from_file_location("ninja_selfcal", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ninja_selfcal"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_and_builds_recipe():
    mod = load_example()
    assert isinstance(mod.selfcal, Recipe)
    # a 2-round selfcal: image -> mask -> calibrate, per round.
    assert [ref.name for ref in mod.selfcal.steps] == [
        "image1",
        "mask1",
        "cal1",
        "image2",
        "mask2",
        "cal2",
    ]


def test_dryrun_dag_renders():
    mod = load_example()
    rendered = render_dag(graph_nodes(mod.selfcal))
    for name in ["image1", "mask1", "cal1", "image2", "mask2", "cal2"]:
        assert name in rendered


def test_vendored_cab_schemas_load():
    # guards the loader entry points the example depends on: wsclean via the
    # cult-cargo loader, the CASA/msutils tasks via the stimela-classic one.
    mod = load_example()
    assert mod.wsclean.command == "wsclean"
    assert "ms" in mod.wsclean.inputs_model.model_fields
    assert mod.casa_mstransform.flavour == "casa-task"
    assert mod.msutils.command  # a real binary cab loaded from parameters.json


def test_cubical_argv_restores_hyphenated_flag_names():
    # guards the sanitize_unique + nom_de_guerre + build_argv path: the
    # example's cubical is built from hyphenated tool option names, which
    # must round-trip back to `--data-ms`/`--out-name` (not `--data_ms`).
    mod = load_example()
    argv = build_argv(
        mod.cubical,
        {"data_ms": "foo.ms", "out_name": "r1", "model_list": "r1-model", "data_column": "DATA"},
    )
    assert argv[0] == "gocubical"
    assert "--data-ms" in argv and "--out-name" in argv and "--model-list" in argv
    assert argv[argv.index("--data-ms") + 1] == "foo.ms"
