"""Positional inputs files supply a target's inputs; typed flags override them."""

import json

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from shinobi.cli import main
from shinobi.clickutil import build_options


@pytest.fixture
def cab_target(tmp_path):
    path = tmp_path / "cab_target.py"
    path.write_text("""from pydantic import BaseModel
from shinobi import Cab
class Inputs(BaseModel):
    name: str
    count: int = 1
    loud: bool = False
class Empty(BaseModel):
    pass
cab = Cab(name="echo", command="echo", inputs_model=Inputs, outputs_model=Empty)
""")
    return f"{path}:cab"


@pytest.fixture
def builder_target(tmp_path):
    path = tmp_path / "builder_target.py"
    path.write_text("""import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from shinobi import Recipe, recipe_builder
class Group(BaseModel):
    count: int
    label: str = "default"
class Inputs(BaseModel):
    group: Group
    tool: dict[str, Any] = {}
class Empty(BaseModel):
    pass
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    Path("built.json").write_text(inputs.model_dump_json())
    return Recipe(name="built", inputs_model=Empty, outputs_model=Empty)
""")
    return f"{path}:recipe"


def _run(*args):
    return CliRunner().invoke(main, ["run", *args])


def test_file_supplies_inputs_and_typed_flags_override(cab_target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs.yaml").write_text("name: file\ncount: 5\nloud: true\n")
    result = _run(cab_target, "inputs.yaml", "--dryrun")
    assert result.exit_code == 0, result.output
    assert "--name file" in result.output and "--count 5" in result.output and "--loud" in result.output
    # An untyped boolean flag keeps the file's value; a typed one overrides it.
    result = _run(cab_target, "inputs.yaml", "--count", "7", "--dryrun")
    assert result.exit_code == 0, result.output
    assert "--count 7" in result.output and "--loud" in result.output
    result = _run(cab_target, "inputs.yaml", "--no-loud", "--dryrun")
    assert "--loud" not in result.output


def test_files_layer_in_order_and_groups_merge(builder_target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base.yaml").write_text("group: {count: 1, label: base}\ntool: {niter: 0, psf: true}\n")
    (tmp_path / "variant.yaml").write_text("group: {count: 2}\ntool: {niter: 10}\n")
    result = _run(builder_target, "base.yaml", "variant.yaml", "--group-label", "flag", "--dryrun")
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "built.json").read_text()) == {"group": {"count": 2, "label": "flag"}, "tool": {"niter": 10, "psf": True}}


def test_bad_inputs_are_usage_errors_before_construction(builder_target, cab_target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run(builder_target, "--dryrun")
    assert result.exit_code == 2
    assert "missing required inputs: --group-count" in result.output
    (tmp_path / "typo.yaml").write_text("group: {count: 1, lable: x}\n")
    result = _run(builder_target, "typo.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "sets unknown inputs: group.lable" in result.output
    (tmp_path / "list.yaml").write_text("- name\n")
    result = _run(cab_target, "list.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "must map input names to values" in result.output
    result = _run(cab_target, "absent.yaml", "--dryrun")
    assert result.exit_code == 2
    assert not (tmp_path / "built.json").exists()


def test_help_marks_required_and_names_files(cab_target):
    result = _run(cab_target, "--help")
    assert result.exit_code == 0, result.output
    assert "[INPUTS.yaml]..." in result.output
    assert next(line for line in result.output.splitlines() if "--name" in line).split() == ["--name", "TEXT", "[required]"]


def test_dict_fields_have_no_flag():
    class Inputs(BaseModel):
        tool: dict[str, int] = {}
        tools: list[dict[str, int]] | None = None
        either: dict[str, int] | str = ""

    assert [option.name for option in build_options(Inputs)] == ["either"]


def _write(path, source):
    path.write_text(source)
    return path


def test_typos_inside_mapping_and_list_entries_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = _write(
        tmp_path / "entries.py",
        """from pydantic import BaseModel
from shinobi import Recipe, recipe_builder
class Entry(BaseModel):
    npix: int = 512
class Inputs(BaseModel):
    arrays: dict[str, Entry] = {}
    jobs: list[Entry] = []
class Empty(BaseModel):
    pass
@recipe_builder(inputs_model=Inputs)
def recipe(inputs):
    raise AssertionError("body called")
""",
    )
    (tmp_path / "mapping.yaml").write_text("arrays: {meerkat: {npxi: 1024}}\n")
    result = _run(f"{target}:recipe", "mapping.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "sets unknown inputs: arrays.meerkat.npxi" in result.output
    (tmp_path / "list.yaml").write_text("jobs: [{npix: 1}, {npxi: 2}]\n")
    result = _run(f"{target}:recipe", "list.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "sets unknown inputs: jobs[1].npxi" in result.output


def test_unreadable_inputs_file_is_a_usage_error_naming_it(cab_target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "good.yaml").write_text("name: x\n")
    (tmp_path / "broken.yaml").write_text("name: [unclosed\n")
    result = _run(cab_target, "good.yaml", "broken.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "cannot read inputs file broken.yaml" in result.output
    (tmp_path / "binary.yaml").write_bytes(b"\xff\xfe\x00")
    result = _run(cab_target, "binary.yaml", "--dryrun")
    assert result.exit_code == 2
    assert "cannot read inputs file binary.yaml" in result.output


def test_click_defaults_still_satisfy_required_bools_and_defaulted_groups(tmp_path, monkeypatch):
    # Before inputs files, an unset required bool flag passed False and a
    # required group took its defaults; both must keep working untyped.
    monkeypatch.chdir(tmp_path)
    target = _write(
        tmp_path / "defaults.py",
        """from pydantic import BaseModel
from shinobi import Cab
class Group(BaseModel):
    a: int = 1
class Inputs(BaseModel):
    name: str
    loud: bool
    g: Group
class Empty(BaseModel):
    pass
cab = Cab(name="echo", command="echo", inputs_model=Inputs, outputs_model=Empty)
""",
    )
    result = _run(f"{target}:cab", "--name", "x", "--dryrun")
    assert result.exit_code == 0, result.output
    assert "--name x" in result.output and "--loud" not in result.output
    result = _run(f"{target}:cab", "--name", "x", "--loud", "--g-a", "3", "--dryrun")
    assert result.exit_code == 0, result.output
    assert "--loud" in result.output and "a=3" in result.output


def test_step_constants_sit_under_files_and_flags_and_stay_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = _write(
        tmp_path / "constants.py",
        """from pydantic import BaseModel
from shinobi import Cab
from shinobi.steps.schema import StepRef
class Group(BaseModel):
    a: int = 1
    b: int = 2
class Inputs(BaseModel):
    name: str
    loud: bool = False
    g: Group = Group()
class Empty(BaseModel):
    pass
cab = Cab(name="echo", command="echo", inputs_model=Inputs, outputs_model=Empty)
ref = StepRef(name="ref", step=cab, params={"name": "p", "loud": True, "g": {"a": 10, "b": 20}})
""",
    )
    result = _run(f"{target}:ref", "--g-a", "5", "--dryrun")
    assert result.exit_code == 0, result.output
    # The StepRef supplies `name`; an untyped bool keeps its True; a typed
    # group leaf overrides one value and keeps the other.
    assert "--name p" in result.output and "--loud" in result.output and "a=5 b=20" in result.output
    (tmp_path / "inputs.yaml").write_text("name: file\n")
    result = _run(f"{target}:ref", "inputs.yaml", "--dryrun")
    assert "--name file" in result.output and "a=10 b=20" in result.output


def test_compile_takes_inputs_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = _write(
        tmp_path / "static.py",
        """from pydantic import BaseModel
from shinobi import Cab, Recipe
from shinobi.steps.schema import InputRef
class Inputs(BaseModel):
    label: str
    loud: bool = False
class Empty(BaseModel):
    pass
cab = Cab(name="echo", command="echo", inputs_model=Inputs, outputs_model=Empty)
recipe = Recipe(name="static", inputs_model=Inputs, outputs_model=Empty)
recipe.add_step("echo", cab, label=InputRef(field="label"), loud=InputRef(field="loud"))
""",
    )
    (tmp_path / "inputs.yaml").write_text("label: fromfile\nloud: true\n")
    result = CliRunner().invoke(main, ["compile", f"{target}:recipe", "inputs.yaml", "--label", "flag"])
    assert result.exit_code == 0, result.output
    assert "echo --label flag --loud" in result.output
    result = CliRunner().invoke(main, ["compile", f"{target}:recipe"])
    assert result.exit_code == 2
    assert "missing required inputs: --label" in result.output
