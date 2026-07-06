import click
from click.testing import CliRunner
from pydantic import BaseModel, Field

from shinobi.clickutil import bool_option_flag, build_options, iter_leaf_fields, option_flag, unflatten_kwargs


class _Plotelev(BaseModel):
    enable: bool = True
    plotter: str | None = "owlcat"


class _Obsinfo(BaseModel):
    enable: bool
    listobs: bool | None = True
    plotelev: _Plotelev = Field(default_factory=_Plotelev)


class _Inputs(BaseModel):
    ms: str
    obsinfo: _Obsinfo
    refant: str | None = "auto"


def test_iter_leaf_fields_flattens_nested_groups_with_dotted_path():
    leaves = iter_leaf_fields(_Inputs)
    flat_names = {name for name, _path, _field in leaves}
    assert flat_names == {"ms", "obsinfo_enable", "obsinfo_listobs", "obsinfo_plotelev_enable",
                           "obsinfo_plotelev_plotter", "refant"}
    paths = {name: path for name, path, _field in leaves}
    assert paths["obsinfo_plotelev_plotter"] == ("obsinfo", "plotelev", "plotter")


def test_flat_model_iter_leaf_fields_matches_single_level_walk():
    # a model with no nested BaseModel fields (every real cab's inputs_model)
    # degenerates to exactly a flat, single-level walk.
    class Flat(BaseModel):
        a: str
        b: int = 1

    leaves = iter_leaf_fields(Flat)
    assert [(name, path) for name, path, _field in leaves] == [("a", ("a",)), ("b", ("b",))]


def test_build_options_produces_dotted_flag_names_for_nested_fields():
    options = build_options(_Inputs)
    flags = {opt.opts[0] for opt in options}
    assert "--obsinfo-plotelev-plotter" in flags
    assert "--obsinfo-enable" in flags


def test_option_flag_round_trips_through_click_kwarg_naming():
    # click derives the callback kwarg name from the flag by replacing "-" -> "_"
    assert option_flag("obsinfo_plotelev_plotter").replace("--", "").replace("-", "_") == (
        "obsinfo_plotelev_plotter"
    )


def test_bool_option_flag_produces_negatable_pair():
    assert bool_option_flag("obsinfo_listobs") == "--obsinfo-listobs/--no-obsinfo-listobs"


def test_build_options_bool_field_can_be_explicitly_negated_via_cli():
    # obsinfo.listobs defaults True -- a bare is_flag option could only ever
    # turn it on, never override the default back to False.
    options = build_options(_Inputs)

    @click.command()
    def cmd(**kwargs):
        click.echo(repr(kwargs.get("obsinfo_listobs")))

    for opt in options:
        cmd.params.append(opt)

    runner = CliRunner()
    default_result = runner.invoke(cmd, ["--ms", "x", "--obsinfo-enable"])
    assert default_result.output.strip() == "True"
    negated_result = runner.invoke(cmd, ["--ms", "x", "--obsinfo-enable", "--no-obsinfo-listobs"])
    assert negated_result.output.strip() == "False"


def test_unflatten_kwargs_reconstructs_nested_dict_for_model_construction():
    flat = {
        "ms": "test.ms",
        "obsinfo_enable": True,
        "obsinfo_listobs": False,
        "obsinfo_plotelev_enable": None,  # not provided by the user -> omitted
        "obsinfo_plotelev_plotter": "plotms",
        "refant": None,  # not provided -> omitted, so the model's own default applies
    }
    nested = unflatten_kwargs(_Inputs, flat)
    assert nested == {
        "ms": "test.ms",
        "obsinfo": {"enable": True, "listobs": False, "plotelev": {"plotter": "plotms"}},
    }
    inputs = _Inputs(**nested)
    assert inputs.refant == "auto"
    assert inputs.obsinfo.plotelev.enable is True  # submodel's own default, since omitted
    assert inputs.obsinfo.plotelev.plotter == "plotms"
