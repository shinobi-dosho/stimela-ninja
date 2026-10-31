"""CREATE outputs resume only through verified cache evidence."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi import DatasetAccess, MeasurementSetV2, Recipe, pystep
from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.dataset_access import DatasetAccessError
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.steps.schema import InputRef, OutputRef
from tests._dataset_fixtures import make_ms, ms_fixture_tables, scans, set_scans

# Skip before dispatch: a skip raised inside a task becomes a workflow failure.
ms_fixture_tables()


class Inputs(BaseModel):
    ms: MeasurementSetV2


class Outputs(BaseModel):
    report: Path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
    def simulate(ms: MeasurementSetV2, value: int) -> Inputs:
        calls.append("simulate")
        return Inputs(ms=make_ms(ms, scan=value))

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")], write_paths=["report"])
    def image(ms: MeasurementSetV2, report: str) -> Outputs:
        calls.append("image")
        path = Path(report)
        path.write_text(json.dumps(scans(ms)))
        return Outputs(report=path)

    def recipe(value=7):
        result = Recipe(name="noise", inputs_model=Inputs, outputs_model=Outputs, cache=True, sandbox=True)
        result.add_step("simulate", simulate, ms=InputRef(field="ms"), value=value)
        result.add_step("image", image, ms=OutputRef(step="simulate", field="ms"), report=str(tmp_path / "image.json"))
        result.set_output("report", OutputRef(step="image", field="report"))
        return result

    config = AppConfig()
    config.cache.dir = str(tmp_path / "cache")
    config.cache.enabled = True
    return recipe, config, tmp_path / "noise.ms", calls


def test_create_resumes_and_missing_descendant_reruns(campaign):
    recipe, config, ms, calls = campaign
    first = recipe()(ms=ms, _config=config)
    assert first.success
    before = scans(ms)
    for _ in range(2):
        assert recipe()(ms=ms, _config=config).success
    assert calls == ["simulate", "image"]
    first.outputs.report.unlink()
    assert recipe()(ms=ms, _config=config).success
    assert calls == ["simulate", "image", "image"]
    assert scans(ms) == before


def test_explicit_overwrite_regenerates_and_invalidates_descendant(campaign):
    recipe, config, ms, calls = campaign
    assert recipe()(ms=ms, _config=config).success
    result = recipe(9)(ms=ms, _config=config, overwrite_steps=["simulate"])
    assert result.success
    assert scans(ms) == [9] * 4
    assert json.loads(result.outputs.report.read_text()) == [9] * 4
    assert calls == ["simulate", "image", "simulate", "image"]
    assert recipe(9)(ms=ms, _config=config).success
    assert len(calls) == 4


@pytest.mark.parametrize("damage", ["unrelated", "modified", "cache_removed", "changed_inputs"])
def test_existing_create_never_falls_through_to_execution(campaign, damage):
    recipe, config, ms, calls = campaign
    if damage == "unrelated":
        make_ms(ms, scan=42)
    else:
        assert recipe()(ms=ms, _config=config).success
        if damage == "modified":
            set_scans(ms, 42)
        elif damage == "cache_removed":
            get_cache_manifest(config.cache.dir).remove({"noise.simulate"})
    before = scans(ms)
    count = len(calls)
    with pytest.raises((DatasetAccessError, DatasetLifecycleUnavailableError)):
        recipe(9 if damage == "changed_inputs" else 7)(ms=ms, _config=config)
    assert len(calls) == count
    assert scans(ms) == before


@pytest.mark.parametrize("corrupt_live", [False, True])
def test_automatic_create_resume_recovers_interrupted_descendant(tmp_path, monkeypatch, corrupt_live):
    from tests.test_required_cache import test_interrupted_creator_prefix_recovers_same_workspace

    test_interrupted_creator_prefix_recovers_same_workspace(tmp_path, monkeypatch, "group", corrupt_live, automatic=True)


def test_overwrite_image_preserves_created_noise(campaign):
    recipe, config, ms, calls = campaign
    assert recipe()(ms=ms, _config=config).success
    before = scans(ms)
    assert recipe()(ms=ms, _config=config, overwrite_steps=["image"]).success
    assert calls == ["simulate", "image", "image"]
    assert scans(ms) == before
    assert recipe()(ms=ms, _config=config).success
    assert len(calls) == 3


def test_overwrite_scattered_images_keeps_ms_and_invalidates_all_slices(campaign, tmp_path):
    from shinobi.steps.schema import ScatterSpec

    recipe, config, ms, calls = campaign
    pipeline = recipe()
    pipeline.outputs_model = type("EmptyOutputs", (BaseModel,), {})
    pipeline.output_wiring = {}
    image = pipeline.steps[1]
    image.scatter = ScatterSpec(fields=["report"])
    image.params["report"] = [str(tmp_path / "first.json"), str(tmp_path / "second.json")]
    assert pipeline(ms=ms, _config=config).success
    assert pipeline(ms=ms, _config=config, overwrite_steps=["image"]).success
    assert calls == ["simulate", "image", "image", "image", "image"]
    assert pipeline(ms=ms, _config=config, overwrite_steps=["simulate"]).success
    assert calls.count("simulate") == 2
    assert calls.count("image") == 6


def test_static_scatter_products_feed_export_in_ms_workflow(campaign, tmp_path):
    from shinobi.steps.schema import ScatterSpec

    recipe, config, ms, calls = campaign
    pipeline = recipe()
    pipeline.outputs_model = type("EmptyOutputs", (BaseModel,), {})
    pipeline.output_wiring = {}
    image = pipeline.steps[1]
    image.scatter = ScatterSpec(fields=["report"])
    image.params["report"] = [str(tmp_path / "first.json"), str(tmp_path / "second.json")]

    class Exported(BaseModel):
        destination: Path

    @pystep(write_paths=["destination"])
    def export(report: Path, destination: str) -> Exported:
        target = Path(destination)
        target.write_text(report.read_text())
        return Exported(destination=target)

    destinations = [str(tmp_path / "export1.json"), str(tmp_path / "export2.json")]
    pipeline.add_step("export", export, scatter=["report", "destination"], report=OutputRef(step="image", field="report"), destination=destinations)
    assert pipeline(ms=ms, _config=config).success
    assert all(json.loads(Path(path).read_text()) == [7] * 4 for path in destinations)
    assert pipeline(ms=ms, _config=config).success
    assert calls == ["simulate", "image", "image"]
