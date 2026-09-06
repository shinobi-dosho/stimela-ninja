"""Smoke-test the minimal example used by the quickstart."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from shinobi.graph import build_graph
from shinobi.policies import build_argv
from shinobi.steps import Recipe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "simple_selfcal.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("simple_selfcal_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quickstart_example_builds_a_valid_wired_recipe():
    module = _load_example()
    assert isinstance(module.selfcal, Recipe)
    graph = build_graph(module.selfcal)
    assert graph.names == ["image", "mask"]
    assert graph.deps == [set(), {0}]


def test_quickstart_cabs_declare_real_output_values_and_tool_names():
    module = _load_example()
    assert module.wsclean.field_meta["restored"].implicit == "{prefix}-MFS-image.fits"
    argv = build_argv(
        module.breizorro,
        {
            "restored_image": Path("img-MFS-image.fits"),
            "outfile": Path("mask.fits"),
        },
    )
    assert argv == [
        "breizorro",
        "--restored-image",
        "img-MFS-image.fits",
        "--outfile",
        "mask.fits",
    ]
    assert module.breizorro.field_meta["mask"].implicit == "{outfile}"
