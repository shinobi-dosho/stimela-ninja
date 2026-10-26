"""Neutral definitions acquire execution semantics only at Shinobi binding."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import dosho
from dosho.validation import CabSpec
from shinobi import Recipe, step
from shinobi.cabs import adapt_definition, get
from shinobi.config import AppConfig
from shinobi.offload.bundle import RecipeBundle, freeze_recipe
from shinobi.offload.worker import _callable
from shinobi.steps.pyfunc import PystepCallable, _inputs_model_from_signature
from shinobi.steps.schema import ParamMeta, StepRef


class Empty(BaseModel):
    pass


class Result(BaseModel):
    value: int
    stop: Path | None = None


@dosho.pystep(name="increment", cache=False, value=3)
def increment(value: int = 0) -> Result:
    return Result(value=value + 1)


@pytest.mark.parametrize("name", dosho.list_cabs())
def test_every_catalogue_definition_adapts(name):
    definition = dosho.get(name)
    runtime = adapt_definition(definition)
    assert runtime is adapt_definition(runtime)
    provider = get(name)
    scope = runtime.step if isinstance(runtime, StepRef) else runtime
    other = provider.step if isinstance(provider, StepRef) else provider
    assert scope.inputs_model.model_json_schema() == other.inputs_model.model_json_schema()
    assert scope.field_meta == other.field_meta
    assert scope.dataset_accesses == other.dataset_accesses
    assert scope.fits_accesses == other.fits_accesses
    if isinstance(definition, CabSpec):
        assert scope.name == definition.name
    else:
        assert isinstance(runtime.func, PystepCallable)
        assert inspect.unwrap(runtime.func) is definition


def test_native_function_and_recipe_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = adapt_definition(increment)
    assert ref().outputs.value == 4
    assert ref.func.__wrapped__ is increment
    assert ref.step.cache is False
    recipe = Recipe(name="pipeline", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("one", increment, value=8)
    assert recipe.steps[0]().outputs.value == 9
    recipe.add_loop("loop", increment, max_iter=2, until="stop", carry={"value": "value"}, value=2)
    assert all(isinstance(s.func, PystepCallable) for s in recipe.steps)


def test_scope_decorators_adapt_and_preserve_bound_constants():
    @step(increment)
    def orchestrate(ctx):
        return None

    assert orchestrate.params == {"value": 3}
    recipe = Recipe(name="pipeline", inputs_model=Empty, outputs_model=Empty)

    @recipe.step(increment, value=9)
    def inside(ctx):
        return None

    assert inside.params == {"value": 9}
    binary = dosho.define_cab("echo", "echo", "image:1", {"text": ("str", True, None)})

    @step(binary)
    def command(ctx):
        return ctx.run()

    assert command.step.command == "echo"


def test_arbitrary_callables_are_rejected():
    with pytest.raises(TypeError, match="@dosho.pystep"):
        adapt_definition(lambda: None)


def test_richer_metadata_and_fieldinfo_survive_shared_derivation():
    class RichMeta(ParamMeta):
        extra_contract: str = "preserved"

    metadata = RichMeta(info="help", choices=["fast", "slow"])

    def func(value: str = Field("fast", json_schema_extra={"param_meta": metadata})) -> None:
        pass

    _, _, fields = _inputs_model_from_signature(func)
    assert fields["value"] is metadata
    assert fields["value"].extra_contract == "preserved"


def source(tmp_path, monkeypatch, *, venv=None):
    path = tmp_path / "neutral_compute.py"
    path.write_text("""from pathlib import Path
from pydantic import BaseModel
import dosho
class Result(BaseModel):
    value: int
@dosho.pystep(cache=False)
def compute(value: int = 3) -> Result:
    return Result(value=value + 2)
""")
    spec = importlib.util.spec_from_file_location("neutral_compute", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "neutral_compute", module)
    spec.loader.exec_module(module)
    if venv:
        dosho.pystep(venv=str(venv), backend="venv", cache=False)(module.compute)
    return module.compute


def test_neutral_function_runs_in_bare_venv(make_venv, tmp_path, monkeypatch):
    func = source(tmp_path, monkeypatch, venv=make_venv())
    result = adapt_definition(func)(value=7)
    assert result.outputs.value == 9
    assert result.backend == "venv"


def test_frozen_worker_restores_original_function_and_identity(make_venv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    func = source(tmp_path, monkeypatch, venv=make_venv())
    recipe = Recipe(name="pipeline", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("compute", func)
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path, code_roots=(tmp_path,))
    staged = bundle.stage(tmp_path / "runs")
    restored = RecipeBundle.read(staged / "bundle.json")
    assert restored.steps[0].code.qualname == func.__qualname__
    assert restored.steps[0].code.module == func.__module__
    monkeypatch.delitem(sys.modules, "neutral_compute")
    adapter = _callable(staged, 0, restored)
    assert isinstance(adapter, PystepCallable)
    assert adapter.__wrapped__.__name__ == func.__name__
    scope = restored.steps[0].scope.restore()
    ref = StepRef(name="restored", step=scope, func=adapter)
    assert ref(value=11).outputs.value == 13
    assert bundle.steps[0].code.execution_digest == restored.steps[0].code.execution_digest


@pytest.mark.parametrize("target,option", [("dosho.cabs:wsclean", "--ms"), ("dosho.cabs.casatasks:listobs", "--vis")])
def test_cli_resolves_neutral_definitions_for_help(target, option):
    from click.testing import CliRunner
    from shinobi.cli import main

    result = CliRunner().invoke(main, ["run", target, "--help"])
    assert result.exit_code == 0, result.output
    assert option in result.output


def test_real_dosho_function_uses_container_stubs(tmp_path, monkeypatch):
    """Execute the generated runner on the host with framework imports blocked."""
    import subprocess
    from unittest.mock import patch

    monkeypatch.chdir(tmp_path)
    bootstrap = """
import runpy, sys, types
from pathlib import Path
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('dosho', 'shinobi', 'pydantic'):
            raise ImportError('blocked in container: ' + name)
sys.meta_path.insert(0, Block())
package = types.ModuleType('fitstoolz')
package.__path__ = []
apps = types.ModuleType('fitstoolz.apps')
apps.__path__ = []
module = types.ModuleType('fitstoolz.apps.header')
def runit(opts):
    assert isinstance(opts.fname, str) and isinstance(opts.outfile, str)
    assert opts.edit == ['CRVAL4=1'] and opts.replace is False
    Path(opts.outfile).write_text('edited ' + Path(opts.fname).read_text())
    return opts.outfile
module.runit = runit
package.apps = apps
apps.header = module
sys.modules.update({'fitstoolz': package, 'fitstoolz.apps': apps,
                    'fitstoolz.apps.header': module})
runpy.run_path(sys.argv[1], run_name='__main__')
"""

    def run_script(argv, **kwargs):
        runner = next(value for value in argv if value.endswith("runner.py"))
        return subprocess.run([sys.executable, "-c", bootstrap, runner], capture_output=True, text=True, check=False)

    source = tmp_path / "input.fits"
    source.write_text("native input")
    ref = adapt_definition(dosho.get("fitstoolz-header"))
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=run_script):
        result = ref(fname=source, edit=["CRVAL4=1"], outfile=tmp_path / "edited.fits", backend="docker")
    assert result.success, result.stderr
    assert result.outputs.outfile == tmp_path / "edited.fits"
    assert result.outputs.outfile.read_text() == "edited native input"
    assert source.read_text() == "native input"
