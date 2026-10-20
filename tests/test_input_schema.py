from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Literal

import pytest
from click.testing import CliRunner
from pydantic import BaseModel, Field, ImportString, RootModel, create_model
from typing_extensions import TypeAliasType, TypedDict

from shinobi.cli import main
from shinobi.clickutil import build_options
from shinobi.exceptions import ParameterError
from shinobi.ownership import WorkspaceOwnershipError, scope_path_accesses, scope_requires_ownership
from shinobi.steps import Cab, ParamMeta, Recipe, Scope, StepRef, pystep
from shinobi.steps.dispatch import _dispatch, _prepare_inputs


class Empty(BaseModel):
    pass


class NestedImports(BaseModel):
    transform: ImportString[Callable]


class ImportRecord(TypedDict):
    transform: ImportString[Callable]


@dataclass
class ImportOptions:
    transform: ImportString[Callable]


class RecursiveImports(BaseModel):
    children: list[RecursiveImports] = Field(default_factory=list)
    transform: ImportString[Callable] | None = None


class RecursiveData(BaseModel):
    children: list[RecursiveData] = Field(default_factory=list)
    value: int = 0


ImportedTransform = TypeAliasType("ImportedTransform", ImportString[Callable])


@pytest.mark.parametrize("scope_type", [Scope, Cab, Recipe])
@pytest.mark.parametrize(
    "annotation",
    [
        ImportString,
        ImportString[Callable],
        ImportString[int],
        ImportString[Callable] | None,
        list[ImportString[Callable]],
        dict[str, ImportString[Callable]],
        dict[ImportString, str],
        tuple[int, ImportString[Callable]],
        NestedImports,
        RootModel[ImportString[Callable]],
        ImportRecord,
        ImportOptions,
        RecursiveImports,
        ImportedTransform,
    ],
)
def test_input_definitions_reject_import_string_in_all_supported_shapes(scope_type, annotation):
    inputs = create_model("Inputs", transform=(annotation, ...))
    kwargs = {"command": "unused"} if scope_type is Cab else {}
    with pytest.raises(ValueError, match="ImportString.*not allowed"):
        scope_type(name="probe", inputs_model=inputs, outputs_model=Empty, **kwargs)


def test_pystep_rejects_import_resolving_parameter():
    with pytest.raises(ValueError, match="ImportString.*not allowed"):

        @pystep()
        def transform(value: ImportString[Callable]) -> None:
            value(9)


def test_cli_option_generation_rejects_import_resolving_model():
    with pytest.raises(ValueError, match="ImportString.*not allowed"):
        build_options(NestedImports)


def test_empty_base_model_placeholder_has_no_cli_options():
    assert build_options(BaseModel) == []


@pytest.fixture
def import_target(tmp_path, monkeypatch):
    module = "shinobi_input_import_probe"
    marker = tmp_path / "imported.txt"
    (tmp_path / f"{module}.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\nvalue = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module, marker
    sys.modules.pop(module, None)


def test_dispatch_rechecks_replaced_input_model_before_import(import_target):
    module, marker = import_target
    scope = Recipe(name="probe", inputs_model=Empty, outputs_model=Empty)
    scope = scope.model_copy(update={"inputs_model": NestedImports})
    with pytest.raises(ParameterError, match="ImportString.*not allowed"):
        _prepare_inputs(scope, {"transform": f"{module}.value"})
    assert module not in sys.modules
    assert not marker.exists()


def test_importing_default_is_rejected_before_default_validation(import_target):
    module, marker = import_target
    inputs = create_model("Inputs", value=(ImportString[int], Field(default=f"{module}.value", validate_default=True)))
    with pytest.raises(ValueError, match="ImportString.*not allowed"):
        Recipe(name="probe", inputs_model=inputs, outputs_model=Empty)
    assert module not in sys.modules
    assert not marker.exists()


@pytest.mark.parametrize("command", ["run", "compile"])
def test_cli_refuses_import_schema_without_importing_input_module(tmp_path, import_target, command):
    module, marker = import_target
    target = tmp_path / "recipe.py"
    target.write_text(
        "from typing import Callable\n"
        "from pydantic import BaseModel, ImportString\n"
        "from shinobi import Recipe\n"
        "class Inputs(BaseModel):\n"
        "    transform: ImportString[Callable]\n"
        "class Outputs(BaseModel):\n"
        "    pass\n"
        "recipe = Recipe(name='probe', inputs_model=Inputs, outputs_model=Outputs)\n"
    )
    result = CliRunner().invoke(main, [command, f"{target}:recipe", "--transform", f"{module}.value"])
    assert result.exit_code != 0
    assert "ImportString" in result.output
    assert "not allowed" in result.output
    assert "Traceback" not in result.output
    assert module not in sys.modules
    assert not marker.exists()


def test_plain_dotted_strings_and_declared_choices_remain_data(import_target):
    module, marker = import_target
    inputs = create_model("Inputs", transform=(str, ...), method=(Literal["sqrt", "square"], "sqrt"))
    scope = Recipe(name="probe", inputs_model=inputs, outputs_model=Empty)
    supplied = f"{module}.value"
    assert _prepare_inputs(scope, {"transform": supplied}) == {"transform": supplied, "method": "sqrt"}
    assert module not in sys.modules
    assert not marker.exists()


def test_recursive_data_model_still_validates_nested_values():
    scope = Recipe(name="probe", inputs_model=RecursiveData, outputs_model=Empty)
    prepared = _prepare_inputs(scope, {"children": [{"value": "3"}]})
    assert prepared["children"][0].value == 3
    assert prepared["children"][0].children == []


@pytest.mark.parametrize("tag", ["default", "metadata", "serialization"])
@pytest.mark.parametrize("entrypoint", ["definition", "cli", "dispatch"])
def test_reserved_union_tags_do_not_hide_imports(tag, entrypoint, import_target):
    module, marker = import_target
    branch = create_model("ImportBranch", kind=(Literal[tag], tag), value=(ImportString[int], ...))
    safe = create_model("DataBranch", kind=(Literal["data"], "data"), value=(int, ...))
    inputs = create_model("Inputs", selection=(Annotated[branch | safe, Field(discriminator="kind")], ...))
    error = ParameterError if entrypoint == "dispatch" else ValueError
    with pytest.raises(error, match="ImportString.*not allowed"):
        if entrypoint == "definition":
            Recipe(name="probe", inputs_model=inputs, outputs_model=Empty)
        elif entrypoint == "cli":
            build_options(inputs)
        else:
            scope = Recipe(name="probe", inputs_model=Empty, outputs_model=Empty).model_copy(update={"inputs_model": inputs})
            _prepare_inputs(scope, {"selection": {"kind": tag, "value": f"{module}.value"}})
    assert module not in sys.modules
    assert not marker.exists()


@pytest.mark.parametrize("replacement", ["assignment", "model_copy"])
@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("entrypoint", ["ownership", "dispatch", "prevalidated_dispatch"])
def test_replaced_child_is_checked_before_ownership_validation(tmp_path, monkeypatch, import_target, replacement, scatter, nested, entrypoint):
    module, marker = import_target
    safe_inputs = create_model("SafeInputs", transform=(str, ...), out=(Path, ...))
    importing_inputs = create_model("ImportInputs", transform=(ImportString[int], ...), out=(Path, ...))
    leaf = Scope(name="leaf", inputs_model=safe_inputs, outputs_model=Empty, field_meta={"out": ParamMeta(write_path=True)})

    def body(ctx):
        pytest.fail("an import-resolving input schema reached the step body")

    recipe = Recipe(name="probe", inputs_model=Empty, outputs_model=Empty)
    target = Recipe(name="inner", inputs_model=Empty, outputs_model=Empty) if nested else recipe
    target.add_step(
        "leaf",
        StepRef(name="leaf", step=leaf, func=body),
        transform=[f"{module}.value"] if scatter else f"{module}.value",
        out=tmp_path / "out.txt",
        scatter=["transform"] if scatter else None,
    )
    if nested:
        recipe.add_step("inner", target)
    ref = target.steps[0]
    if replacement == "assignment":
        ref.step.inputs_model = importing_inputs
    else:
        ref.step = ref.step.model_copy(update={"inputs_model": importing_inputs})
    assert scope_requires_ownership(recipe)
    monkeypatch.chdir(tmp_path)
    error = WorkspaceOwnershipError if entrypoint == "ownership" else ParameterError
    with pytest.raises(error, match="ImportString.*not allowed"):
        if entrypoint == "ownership":
            scope_path_accesses(recipe, {}, workspace=tmp_path)
        else:
            kwargs = {"_validated_inputs": Empty()} if entrypoint == "prevalidated_dispatch" else {}
            _dispatch(recipe, None, **kwargs)
    assert module not in sys.modules
    assert not marker.exists()
