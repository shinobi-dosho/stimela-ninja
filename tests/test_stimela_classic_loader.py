import json

import pydantic
import pytest

from shinobi.exceptions import CabLoadError
from shinobi.loaders.stimela_classic import load_file, loads
from shinobi.steps.schema import path_fields

MSTRANSFORM_JSON = json.dumps(
    {
        "task": "casa_mstransform",
        "base": "stimela/casa",
        "binary": "mstransform",
        "description": "Split the MS, combine/separate/regrid spws",
        "prefix": "-",
        "msdir": True,
        "parameters": [
            {
                "info": "Name of input visibility file",
                "name": "msname",
                "io": "msfile",
                "dtype": "file",
                "required": True,
                "mapping": "vis",
            },
            {
                "info": "Axis to do parallelization across.",
                "dtype": "str",
                "default": "auto",
                "name": "separationaxis",
                "choices": ["auto", "spw", "scan", "baseline"],
            },
            {
                "info": "The number of Sub-MSs to create",
                "dtype": ["str", "int"],
                "default": "auto",
                "name": "numsubms",
            },
            {
                "info": "Create a multi-MS output from an input MS",
                "dtype": "bool",
                "default": False,
                "name": "createmms",
            },
        ],
    }
)

MSUTILS_JSON = json.dumps(
    {
        "task": "msutils",
        "base": "stimela/msutils",
        "binary": "msutils",
        "description": "Tools for manipulating measurement sets (MSs)",
        "parameters": [
            {
                "info": "MSUtils command to execute",
                "name": "command",
                "dtype": "str",
                "required": True,
                "choices": ["addcol", "sumcols", "copycol", "summary"],
            },
            {"info": "MS name", "dtype": "file", "name": "msname", "io": "msfile"},
        ],
    }
)

NO_BINARY_JSON = json.dumps({"task": "bare", "parameters": []})


def test_loads_basic_cab_fields():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.name == "casa_mstransform"
    assert cab.command == "mstransform"
    assert cab.info == "Split the MS, combine/separate/regrid spws"
    assert cab.image == "stimela/casa"


def test_casa_base_gets_casa_task_flavour():
    assert loads(MSTRANSFORM_JSON).flavour == "casa-task"


def test_non_casa_base_gets_binary_flavour():
    assert loads(MSUTILS_JSON).flavour == "binary"


def test_msfile_io_forces_ms_dtype_regardless_of_raw_dtype():
    cab = loads(MSTRANSFORM_JSON)
    # raw dtype was "file", but io: "msfile" overrides it -> a path field
    assert "msname" in path_fields(cab.inputs_model)


def test_mapping_becomes_nom_de_guerre():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.field_meta["msname"].nom_de_guerre == "vis"


def test_required_and_default_map_directly():
    cab = loads(MSTRANSFORM_JSON)
    fields = cab.inputs_model.model_fields
    assert fields["msname"].is_required()
    assert fields["separationaxis"].default == "auto"
    assert fields["createmms"].default is False


def test_choices_are_appended_to_info():
    cab = loads(MSTRANSFORM_JSON)
    info = cab.field_meta["separationaxis"].info
    assert "Axis to do parallelization across." in info
    assert "auto" in info and "baseline" in info


def test_choices_alone_become_info_when_no_info_given():
    cab = loads(MSUTILS_JSON)
    assert "sumcols" in cab.field_meta["command"].info


def test_choices_are_recorded_on_field_meta():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.field_meta["separationaxis"].choices == ["auto", "spw", "scan", "baseline"]


def test_choices_are_enforced_by_the_generated_model():
    cab = loads(MSTRANSFORM_JSON)
    cab.inputs_model(msname="x.ms", separationaxis="scan")
    with pytest.raises(pydantic.ValidationError):
        cab.inputs_model(msname="x.ms", separationaxis="not-a-choice")


def test_non_list_choices_raise():
    """Regression test: `list("auto")` used to silently explode a string
    `choices` value into per-character choices (`['a', 'u', 't', 'o']`)
    instead of failing loudly -- now routed through the same
    `_modelgen.validate_choices` the other loaders use.
    """
    bad = json.dumps(
        {
            "task": "bad",
            "parameters": [{"name": "mode", "dtype": "str", "choices": "auto"}],
        }
    )
    with pytest.raises(CabLoadError, match="'choices' must be a list"):
        loads(bad)


def test_dtype_list_uses_first_alternative():
    cab = loads(MSTRANSFORM_JSON)
    # ["str", "int"] narrows to str -> not a path, scalar string field
    assert cab.inputs_model.model_fields["numsubms"].annotation is not None
    assert "numsubms" not in path_fields(cab.inputs_model)


def test_missing_binary_falls_back_to_task_name():
    assert loads(NO_BINARY_JSON).command == "bare"


def test_load_file_reads_from_disk(tmp_path):
    path = tmp_path / "parameters.json"
    path.write_text(MSUTILS_JSON)
    cab = load_file(path)
    assert cab.name == "msutils"
    assert cab.inputs_model.model_fields["command"].is_required()
