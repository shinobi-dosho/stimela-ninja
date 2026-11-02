"""Construction stays explicit and separate from runtime inputs."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import BaseModel, ImportString

from shinobi import Recipe, recipe_builder
from shinobi.cli import main


class Empty(BaseModel):
    pass


def test_python_boundary():
    calls = []

    class Inputs(BaseModel):
        count: int

    @recipe_builder(inputs_model=Inputs)
    def builder(inputs):
        calls.append(inputs.count)
        return Recipe(name="built", inputs_model=Empty, outputs_model=Empty)

    with pytest.raises(ValueError):
        builder(count="bad")
    assert calls == []
    assert isinstance(builder(count=2), Recipe)
    assert calls == [2]


def test_builder_guards():
    class Unsafe(BaseModel):
        implementation: ImportString

    with pytest.raises(ValueError, match="ImportString"):
        recipe_builder(inputs_model=Unsafe)(lambda inputs: None)

    @recipe_builder(inputs_model=Empty)
    def wrong(inputs):
        return None

    with pytest.raises(TypeError, match="return a Recipe"):
        wrong()

    class Required(BaseModel):
        value: str

    @recipe_builder(inputs_model=Empty)
    def required(inputs):
        return Recipe(name="required", inputs_model=Required, outputs_model=Empty)

    with pytest.raises(ValueError, match="runtime inputs: value"):
        required()

    class Defaulted(BaseModel):
        value: str = "ok"

    @recipe_builder(inputs_model=Empty)
    def defaulted(inputs):
        return Recipe(name="defaulted", inputs_model=Defaulted, outputs_model=Empty)

    assert defaulted().inputs_model().value == "ok"


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "builder_target.py"
    path.write_text("""from pathlib import Path
from pydantic import BaseModel
from shinobi import Cab, Recipe, recipe_builder
class Inputs(BaseModel):
    settings: Path
class ToolInputs(BaseModel):
    value: str
class Empty(BaseModel):
    pass
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    with Path("calls").open("a") as stream:
        stream.write("called\\n")
    value = inputs.settings.read_text()
    graph = Recipe(name="built", inputs_model=Empty, outputs_model=Empty)
    graph.add_step("echo", Cab(name="echo", command="echo", inputs_model=ToolInputs, outputs_model=Empty), value=value.strip())
    return graph
""")
    return f"{path}:recipe"


def test_help_no_body_or_settings(target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["run", target, "--help"])
    assert result.exit_code == 0, result.output
    assert "--settings" in result.output
    result = CliRunner().invoke(main, ["compile", target, "--help"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "calls").exists()
    result = CliRunner().invoke(main, ["run", target])
    assert result.exit_code != 0
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("command,flags", [("run", []), ("run", ["--dryrun"]), ("compile", [])])
def test_cli_constructs_once(command, flags, target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "settings").write_text("built")
    result = CliRunner().invoke(main, [command, target, "--settings", "settings", *flags])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "calls").read_text() == "called\n"


def test_compile_workdir_refuses_before_body(target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["compile", target, "--workdir", str(tmp_path / "elsewhere"), "--settings", "absent"])
    assert result.exit_code != 0
    assert "invocation working directory" in result.output
    assert not (tmp_path / "calls").exists()


def test_remote_marker_refuses_before_body(target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["run", target, "--static-target-only", "--settings", "absent"])
    assert result.exit_code != 0
    assert "static targets only" in result.output
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("field", ["quiet: bool = False", "workdir: str = 'x'", "help: str = 'x'", "no_cache: bool = False"])
def test_outer_option_collisions(tmp_path, field):
    command = "compile" if field.startswith("workdir") else "run"
    path = tmp_path / "collision.py"
    path.write_text(f"""from pydantic import BaseModel
from shinobi import recipe_builder
class Inputs(BaseModel):
    {field}
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    raise AssertionError("body called")
""")
    result = CliRunner().invoke(main, [command, f"{path}:recipe", "--help"])
    assert result.exit_code != 0
    assert "collides" in result.output


def test_graph_validation_includes_children():
    from shinobi.graph import RecipeGraphError
    from shinobi.steps.schema import StepRef

    @recipe_builder(inputs_model=Empty)
    def invalid(inputs):
        child = Recipe(name="child", inputs_model=Empty, outputs_model=Empty)
        root = Recipe(name="root", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="child", step=child)])
        child.steps.append(StepRef(name="bad", step=Recipe(name="leaf", inputs_model=Empty, outputs_model=Empty), after=["missing"]))
        return root

    with pytest.raises(RecipeGraphError):
        invalid()


def test_nested_builder_options_and_flattening_collisions(tmp_path):
    path = tmp_path / "nested_builder.py"
    path.write_text("""from pydantic import BaseModel
from shinobi import Recipe, recipe_builder
class Group(BaseModel):
    count: int
class Inputs(BaseModel):
    group: Group
class Empty(BaseModel):
    pass
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    return Recipe(name=f"built-{inputs.group.count}", inputs_model=Empty, outputs_model=Empty)
""")
    result = CliRunner().invoke(main, ["run", f"{path}:recipe", "--group-count", "3", "--dryrun"])
    assert result.exit_code == 0, result.output
    path.write_text(path.read_text().replace("    group: Group", "    group: Group\n    group_count: int = 1"))
    result = CliRunner().invoke(main, ["run", f"{path}:recipe", "--help"])
    assert "collides" in result.output


@pytest.mark.parametrize("placed", [False, True])
def test_provenance_origin_and_replay_override_guard(target, tmp_path, monkeypatch, placed):
    import json

    monkeypatch.chdir(tmp_path)
    if placed:
        source = Path(target.rsplit(":", 1)[0])
        source.write_text(
            source.read_text()
            .replace("from shinobi import Cab, Recipe, recipe_builder", "from shinobi import Cab, Recipe, recipe_builder, SlurmPlacement")
            .replace("value=value.strip()", "placement=SlurmPlacement(), value=value.strip()")
        )
    (tmp_path / "settings").write_text("built")
    result = CliRunner().invoke(main, ["run", target, "--settings", "settings", "--provenance"])
    assert result.exit_code == 0, result.output
    from shinobi.config import AppConfig

    manifest = next(path for path in Path(AppConfig.load().provenance.dir).glob("built.*.run.json") if json.loads(path.read_text())["root"]["name"] == "built")
    data = json.loads(manifest.read_text())
    assert data["builder_origin"] == {"target": target, "arguments": {"settings": "settings"}}
    assert data["root"]["inputs"] == {}
    (tmp_path / "settings").unlink()
    result = CliRunner().invoke(main, ["replay", str(manifest), "--target", "does.not.exist:recipe"])
    assert result.exit_code != 0
    assert "replay of recipe-builder runs is unsupported" in result.output
    assert (tmp_path / "calls").read_text() == "called\n"
    data.pop("builder_origin")
    manifest.write_text(json.dumps(data))
    result = CliRunner().invoke(main, ["replay", str(manifest)])
    assert "recipe builders cannot be replayed" in result.output
    assert (tmp_path / "calls").read_text() == "called\n"


def test_frozen_recipe_needs_no_settings(target, tmp_path, monkeypatch):
    from shinobi.cli import _resolve_target
    from shinobi.config import AppConfig
    from shinobi.offload.bundle import RecipeBundle, freeze_recipe
    from shinobi.graph import build_graph

    monkeypatch.chdir(tmp_path)
    (tmp_path / "settings").write_text("built")
    recipe = _resolve_target(target)(settings=Path("settings"))
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path)
    (tmp_path / "settings").unlink()
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json()).declaration()
    assert restored.name == "built"
    build_graph(restored)
    result = restored(backend="native", _config=AppConfig())
    assert result.success
    assert "--value built" in result.stdout
    assert (tmp_path / "calls").read_text() == "called\n"


def test_validation_failure_does_not_construct(tmp_path):
    path = tmp_path / "validation.py"
    path.write_text("""from pydantic import BaseModel, Field
from shinobi import recipe_builder
class Inputs(BaseModel):
    count: int = Field(gt=0)
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    raise AssertionError("body called")
""")
    result = CliRunner().invoke(main, ["run", f"{path}:recipe", "--count", "-1"])
    assert result.exit_code != 0
    assert "greater than 0" in result.output
    assert "body called" not in result.output


def test_settings_contents_use_invocation_directory(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    path = source / "paths.py"
    path.write_text("""from pathlib import Path
import yaml
from pydantic import BaseModel
from shinobi import Recipe, recipe_builder
class Inputs(BaseModel):
    settings: Path
class Settings(BaseModel):
    label: Path
class Empty(BaseModel):
    pass
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    settings = Settings.model_validate(yaml.safe_load(inputs.settings.read_text()))
    return Recipe(name=settings.label.read_text(), inputs_model=Empty, outputs_model=Empty)
""")
    (source / "settings.yaml").write_text("label: label.txt\n")
    (source / "label.txt").write_text("wrong")
    (invocation / "label.txt").write_text("right")
    monkeypatch.chdir(invocation)
    from shinobi.cli import _resolve_target

    assert _resolve_target(f"{path}:recipe")(settings=source / "settings.yaml").name == "right"


def test_short_alias_collisions(tmp_path):
    path = tmp_path / "aliases.py"
    path.write_text("""from pydantic import BaseModel, Field
from shinobi import recipe_builder
class Inputs(BaseModel):
    one: str = Field("one", json_schema_extra={"abbreviation": "h"})
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    raise AssertionError("body called")
""")
    result = CliRunner().invoke(main, ["run", f"{path}:recipe", "--help"])
    assert "'-h' collides" in result.output
