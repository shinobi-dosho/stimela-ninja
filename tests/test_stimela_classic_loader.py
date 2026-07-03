import json

from shinobi.loaders.stimela_classic import load_file, loads

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
    cab = loads(MSTRANSFORM_JSON)
    assert cab.flavour == "casa-task"


def test_non_casa_base_gets_binary_flavour():
    cab = loads(MSUTILS_JSON)
    assert cab.flavour == "binary"


def test_msfile_io_forces_ms_dtype_regardless_of_raw_dtype():
    cab = loads(MSTRANSFORM_JSON)
    # raw dtype was "file", but io: "msfile" overrides it
    assert cab.inputs["msname"].dtype == "MS"


def test_mapping_becomes_nom_de_guerre():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.inputs["msname"].nom_de_guerre == "vis"


def test_required_and_default_map_directly():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.inputs["msname"].required is True
    assert cab.inputs["separationaxis"].default == "auto"
    assert cab.inputs["createmms"].default is False


def test_choices_are_appended_to_info():
    cab = loads(MSTRANSFORM_JSON)
    info = cab.inputs["separationaxis"].info
    assert "Axis to do parallelization across." in info
    assert "auto" in info and "baseline" in info


def test_choices_alone_become_info_when_no_info_given():
    cab = loads(MSUTILS_JSON)
    info = cab.inputs["command"].info
    assert "sumcols" in info


def test_dtype_list_uses_first_alternative():
    cab = loads(MSTRANSFORM_JSON)
    assert cab.inputs["numsubms"].dtype == "str"


def test_missing_binary_falls_back_to_task_name():
    cab = loads(NO_BINARY_JSON)
    assert cab.command == "bare"


def test_load_file_reads_from_disk(tmp_path):
    path = tmp_path / "parameters.json"
    path.write_text(MSUTILS_JSON)
    cab = load_file(path)
    assert cab.name == "msutils"
    assert cab.inputs["command"].required is True
