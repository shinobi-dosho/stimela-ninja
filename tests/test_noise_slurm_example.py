"""The Slurm noise variant places the local example's case declaration per case."""

import importlib
from pathlib import Path

import pytest
import yaml

from shinobi.graph import build_graph
from shinobi.config import AppConfig
from shinobi.offload.mixed import freeze_mixed, plan_mixed
from shinobi.steps.schema import InputRef


ROOT = Path(__file__).resolve().parents[1] / "examples" / "noise_comparison"


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("noise_pipeline"), importlib.import_module("noise_slurm")


@pytest.fixture
def settings():
    data = yaml.safe_load((ROOT / "settings.yaml").read_text())
    arrays = {name: data["arrays"][name] for name in ("meerkat", "gmrt")}
    for array in arrays.values():
        array["layout"] = str((ROOT.parents[1] / array["layout"]).resolve())
    data.update(arrays=arrays, declinations=[-25, 10], robust=[-1, 0.5])
    return data


def test_each_case_is_one_placed_scatter_slice(examples, settings):
    _, slurm = examples
    recipe = slurm.recipe(**settings)
    build_graph(recipe)
    (cases,) = recipe.steps
    assert cases.placement.sbatch_opts == settings["sbatch_opts"]
    assert cases.placement.max_parallel == settings["max_parallel"]
    assert len(cases.params["ms"]) == 4
    branch = cases.step
    build_graph(branch)
    assert [ref.name for ref in branch.steps] == ["simulate", "wsclean", "ddfacet", "casa", "casa_fits", "inventory"]
    assert set(cases.scatter.fields) == set(branch.inputs_model.model_fields)


def test_cases_declare_the_local_parameters(examples, settings):
    local, slurm = examples
    flat = {ref.name: ref for ref in local.recipe(**settings).steps}
    (cases,) = slurm.recipe(**settings).steps
    names = [f"{instrument}_dec{dec:+d}_" for dec in settings["declinations"] for instrument in settings["arrays"]]
    for index, prefix in enumerate(names):
        for ref in cases.step.steps:
            expected = flat[prefix + ref.name]
            inputs = {field: source for field, source in ref.wiring.items() if isinstance(source, InputRef)}
            resolved = {field: cases.params[source.field][index] for field, source in inputs.items()}
            assert {**ref.params, **resolved} == expected.params
            assert ref.step.name == expected.step.name and ref.step.image == expected.step.image


def test_compiles_to_one_driver_per_case(examples, settings, tmp_path, monkeypatch):
    # The same freeze and plan as `ninja compile` without --submit, including
    # capture of the CASA and inventory pystep source.
    _, slurm = examples
    monkeypatch.chdir(tmp_path)
    config = AppConfig.model_validate(yaml.safe_load((ROOT / "config.yaml").read_text()))
    bundle = freeze_mixed(slurm.recipe(**settings), {}, config=config, workspace=tmp_path, code_roots=(ROOT,))
    plan = plan_mixed(bundle)
    assert [unit.name for unit in plan.units] == [f"array-noise.cases[{index}]" for index in range(4)]
    assert not any(tmp_path.iterdir())
