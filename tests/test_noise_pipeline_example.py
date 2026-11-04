"""The noise experiment is a configurable, statically declared DAG."""

import importlib
from pathlib import Path

import pytest
import yaml

from shinobi.graph import build_graph
from shinobi.dag import graph_nodes, render_dag
from shinobi.steps.schema import OutputRef


@pytest.fixture
def example(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "examples" / "noise_comparison"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("noise_pipeline")


def test_default_experiment_keeps_full_grid(example, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    recipe = example.recipe(**yaml.safe_load(Path(example.__file__).with_name("settings.yaml").read_text()))
    build_graph(recipe)
    simulations = [ref for ref in recipe.steps if ref.name.endswith("_simulate")]
    imagers = [ref for ref in recipe.steps if ref.name.endswith(("_wsclean", "_ddfacet", "_casa"))]
    assert len(simulations) == 30
    assert sum(len(ref.params[ref.scatter.fields[0]]) for ref in imagers) == 1260
    assert all(ref.params["ntime"] == 450 for ref in simulations)
    assert all(ref.placement is None for ref in recipe.steps)


def test_settings_select_grid_without_rewriting_recipe(example, tmp_path):
    source = Path(example.__file__).with_name("settings.yaml")
    data = yaml.safe_load(source.read_text())
    data["arrays"] = {"meerkat": data["arrays"]["meerkat"]}
    data["arrays"]["meerkat"]["layout"] = str(Path(data["arrays"]["meerkat"]["layout"]).resolve())
    data.update(declinations=[-25, 10], robust=[-1, 0.5], nchan=8, single_channel=3, duration_seconds=480, integration_seconds=8)
    recipe = example.recipe(**data)
    graph = build_graph(recipe)
    assert len(recipe.steps) == 12
    for dec in data["declinations"]:
        name = f"meerkat_dec{dec:+d}"
        refs = {ref.name.removeprefix(name + "_"): ref for ref in recipe.steps if ref.name.startswith(name + "_")}
        assert refs["simulate"].params["ntime"] == 60
        assert refs["simulate"].params["startha"] == pytest.approx(-1 / 15)
        ms = OutputRef(step=name + "_simulate", field="ms")
        assert refs["wsclean"].wiring["ms"] == [ms]
        assert refs["ddfacet"].wiring["data_ms"] == [ms]
        assert refs["casa"].wiring["ms"] == ms
        assert refs["wsclean"].params["channel_range"] == [(3, 4), (3, 4), (0, 8), (0, 8)]
        assert refs["casa"].params["spw"] == ["0:3~3", "0:3~3", "0:0~7", "0:0~7"]
        assert refs["ddfacet"].params["weight_robust"] == [-1, 0.5, -1, 0.5]
        assert refs["casa_fits"].wiring["products"] == OutputRef(step=name + "_casa", field="products")
        inventory_index = graph.names.index(name + "_inventory")
        assert {graph.names[i] for i in graph.deps[inventory_index]} == {name + suffix for suffix in ("_wsclean", "_ddfacet", "_casa_fits")}
        assert len(refs["inventory"].params["directories"]) == 12


def test_dryrun_keeps_product_wiring_without_expanding_paths(example):
    source = Path(example.__file__).with_name("settings.yaml")
    data = yaml.safe_load(source.read_text())
    data.update(arrays={"meerkat": data["arrays"]["meerkat"]}, declinations=[-25], robust=[-1, 0.5])
    nodes = graph_nodes(example.recipe(**data), {})
    compact = render_dag(nodes)
    assert "ms <- [meerkat_dec-25_simulate.ms]" in compact
    assert "products <- meerkat_dec-25_casa.products" in compact
    assert "[scatter; 4 slices]" in compact
    parallel = next(line for line in compact.splitlines() if "meerkat_dec-25_wsclean" in line)
    assert "meerkat_dec-25_ddfacet" in parallel and "meerkat_dec-25_casa" in parallel
    assert compact.count("▼") == 5
    assert "noise-products" not in compact
    assert "robust" not in compact
    assert "read-after-write" not in compact
    detailed = render_dag(nodes, details=True)
    assert "noise-products" in detailed
    assert "read-after-write" in detailed


def test_external_settings_layout_is_relative_to_invocation(example, tmp_path, monkeypatch):
    source = Path(example.__file__).with_name("settings.yaml")
    data = yaml.safe_load(source.read_text())
    array = data["arrays"]["meerkat"]
    layout = Path(array["layout"]).resolve()
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "layout.yaml").write_text(layout.read_text())
    array["layout"] = "layout.yaml"
    data.update(arrays={"meerkat": array}, declinations=[-25], robust=[0])
    settings = settings_dir / "settings.yaml"
    settings.write_text(yaml.safe_dump(data))
    monkeypatch.chdir(workspace)
    recipe = example.recipe(**yaml.safe_load(settings.read_text()))
    simulation = next(ref for ref in recipe.steps if ref.name.endswith("_simulate"))
    assert simulation.params["telescope"] == workspace / "layout.yaml"
    assert not (settings_dir / "layout.yaml").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"declinations": [5, 5]}, "declinations must be unique"),
        ({"declinations": [95]}, "less than or equal to 90"),
        ({"robust": [3]}, "less than or equal to 2"),
        ({"integration_seconds": 7}, "exact number of integrations"),
        ({"single_channel": 4}, "single_channel must be smaller than nchan"),
        ({"casa": {"robust": 0}}, "casa: robust are set per job"),
    ],
)
def test_settings_reject_inconsistent_experiments(example, change, message):
    source = Path(example.__file__).with_name("settings.yaml")
    data = {**yaml.safe_load(source.read_text()), **change}
    with pytest.raises(ValueError, match=message):
        example.Settings.model_validate(data)
