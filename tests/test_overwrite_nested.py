"""Run-time overwrite selects declared nested paths and their dependents."""

from collections import Counter

import pytest
from pydantic import BaseModel

from shinobi import Recipe, pystep
from shinobi.steps.schema import OutputRef


class Empty(BaseModel):
    pass


class Value(BaseModel):
    value: int


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = Counter()

    @pystep()
    def work(label: str, value: int = 1) -> Value:
        calls[label] += 1
        return Value(value=value)

    child = Recipe(name="internal", inputs_model=Empty, outputs_model=Value)
    child.add_step("source", work, label="source")
    child.add_step("child", work, label="child", value=OutputRef(step="source", field="value"))
    child.add_step("sibling", work, label="sibling")
    child.set_output("value", OutputRef(step="child", field="value"))
    outer = Recipe(name="outer", inputs_model=Empty, outputs_model=Empty)
    outer.add_step("step", child)
    outer.add_step("consumer", work, label="consumer", value=OutputRef(step="step", field="value"))
    outer.add_step("ordered", work, label="ordered")
    outer.steps[-1].after = ["step"]
    outer.add_step("unrelated", work, label="unrelated")
    kwargs = {"cache": True, "cache_dir": str(tmp_path / "cache")}
    assert outer(**kwargs).success
    assert outer(**kwargs).success
    assert set(calls.values()) == {1}
    return outer, kwargs, calls


@pytest.mark.parametrize(
    "selected, rerun",
    [
        ("step.child", {"child", "consumer", "ordered"}),
        ("step.source", {"source", "child", "consumer", "ordered"}),
        ("step", {"source", "child", "sibling", "consumer", "ordered"}),
        ("consumer", {"consumer"}),
    ],
)
def test_nested_overwrite_preserves_unrelated_cache(pipeline, selected, rerun):
    recipe, kwargs, calls = pipeline
    assert recipe(**kwargs, overwrite_steps=[selected]).success
    assert calls == {name: 2 if name in rerun else 1 for name in calls}
    before = calls.copy()
    assert recipe(**kwargs).success
    assert calls == before


def test_invalid_selection_leaves_cache_untouched(pipeline):
    recipe, kwargs, calls = pipeline
    before = calls.copy()
    with pytest.raises(ValueError, match="unknown step"):
        recipe(**kwargs, overwrite_steps=["step.child", "step.missing"])
    assert recipe(**kwargs).success
    assert calls == before


def test_cli_accepts_nested_path(pipeline, monkeypatch):
    from click.testing import CliRunner
    from shinobi import cli

    recipe, kwargs, calls = pipeline
    recipe.cache = True
    monkeypatch.setattr(cli, "_resolve_target", lambda target: recipe)
    result = CliRunner().invoke(cli.main, ["run", "pipeline:recipe", "--cache-dir", kwargs["cache_dir"], "--overwrite", "step.child"])
    assert result.exit_code == 0, result.output
    assert calls["child"] == calls["consumer"] == 2
    assert calls["source"] == calls["sibling"] == 1


def test_overwrite_uses_snapshot_rollback(tmp_path, monkeypatch):
    from tests.test_snapshots import _pipeline, _read

    monkeypatch.chdir(tmp_path)
    ms = tmp_path / "data.ms"
    calls = {}
    kwargs = {"spw": "*", "cache": True, "cache_dir": str(tmp_path / "cache")}
    assert _pipeline(ms, calls)(**kwargs).success
    assert _pipeline(ms, calls)(**kwargs, overwrite_steps=["flag"]).success
    assert calls["flag_saw"] == ["vis[*]", "vis[*]"]
    assert calls["cal"] == 2
    assert _read(ms) == "vis[*]|flag[default]|cal"


def test_scatter_family_and_custom_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = Counter()

    @pystep(cache_dir=str(tmp_path / "leaf-cache"))
    def work(value: int) -> Value:
        calls[value] += 1
        return Value(value=value)

    inner = Recipe(name="internal", inputs_model=Empty, outputs_model=Empty)
    inner.add_step("child", work, scatter=["value"], value=[1, 2, 3])
    outer = Recipe(name="outer", inputs_model=Empty, outputs_model=Empty)
    outer.add_step("step", inner)
    kwargs = {"cache": True, "cache_dir": str(tmp_path / "cache")}
    assert outer(**kwargs).success
    assert outer(**kwargs).success
    assert calls == {1: 1, 2: 1, 3: 1}
    assert outer(**kwargs, overwrite_steps=["step.child"]).success
    assert calls == {1: 2, 2: 2, 3: 2}
