from typing import Literal

import click
from click.testing import CliRunner
from pydantic import BaseModel, Field

from shinobi.clickutil import (
    bool_option_flag,
    build_options,
    click_type,
    iter_leaf_fields,
    option_flag,
    unflatten_kwargs,
)


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


class _TupleInputs(BaseModel):
    ms: str
    channel_range: tuple[int, int] | None = None
    weight: str | tuple[str, float] | None = None
    tags: list[str] | None = None


def test_unflatten_kwargs_drops_click_multiple_empty_tuple_default():
    # build_options renders every list/tuple field as a `multiple=True`
    # click option, which defaults to `()` when the user omits it. That `()`
    # must be treated as "not provided" -- otherwise it reaches the model and
    # an optional Tuple/Union-of-tuple field rejects it instead of falling
    # back to its own default. (Regression: wsclean's channel-range/interval/
    # weight failing validation when left unset.)
    flat = {
        "ms": "test.ms",
        "channel_range": (),
        "weight": (),
        "tags": (),
    }
    nested = unflatten_kwargs(_TupleInputs, flat)
    assert nested == {"ms": "test.ms"}
    inputs = _TupleInputs(**nested)
    assert inputs.channel_range is None
    assert inputs.weight is None
    assert inputs.tags is None


class _ChoiceInputs(BaseModel):
    ms: str
    mode: Literal["sim", "add", "subtract"] = "sim"
    # a choice field with a default is Optional -> Literal[...] | None, the
    # shape narrow_choices produces for an optional/default cab `choices:` field.
    interp: Literal["nearest", "linear", "cubic"] | None = "nearest"


def test_click_type_maps_literal_to_choice():
    # a bare Literal and an Optional[Literal] (choice-with-default) both map
    # to a click.Choice listing exactly the allowed values.
    bare = click_type(Literal["sim", "add", "subtract"], is_path=False)
    optional = click_type(Literal["nearest", "linear", "cubic"] | None, is_path=False)
    assert isinstance(bare, click.Choice) and list(bare.choices) == ["sim", "add", "subtract"]
    assert isinstance(optional, click.Choice) and list(optional.choices) == ["nearest", "linear", "cubic"]


def test_build_options_choice_field_becomes_click_choice():
    options = build_options(_ChoiceInputs)
    by_name = {opt.name: opt for opt in options}
    assert isinstance(by_name["mode"].type, click.Choice)
    assert list(by_name["mode"].type.choices) == ["sim", "add", "subtract"]
    assert isinstance(by_name["interp"].type, click.Choice)


def test_build_options_choice_field_rejects_out_of_set_value_via_cli():
    options = build_options(_ChoiceInputs)

    @click.command()
    def cmd(**kwargs):
        click.echo(kwargs["mode"])

    for opt in options:
        cmd.params.append(opt)

    runner = CliRunner()
    assert runner.invoke(cmd, ["--ms", "x", "--mode", "add"]).output.strip() == "add"
    rejected = runner.invoke(cmd, ["--ms", "x", "--mode", "bogus"])
    assert rejected.exit_code != 0
    assert "not one of" in rejected.output


class _AbbrevInputs(BaseModel):
    ms: str
    ascii_sky: str | None = Field(default=None, json_schema_extra={"abbreviation": "as"})
    polarisation: bool = Field(default=True, json_schema_extra={"abbreviation": "pol"})
    refant: str | None = "auto"  # no abbreviation


def test_build_options_emits_short_alias_from_abbreviation():
    options = build_options(_AbbrevInputs)
    by_name = {opt.name: opt for opt in options}
    assert "-as" in by_name["ascii_sky"].opts
    assert "--ascii-sky" in by_name["ascii_sky"].opts
    # the bool field's abbreviation rides alongside its --flag/--no-flag pair
    assert "-pol" in by_name["polarisation"].opts
    # a field with no abbreviation gets only its long flag
    assert by_name["refant"].opts == ["--refant"]


def test_short_alias_round_trips_to_same_field_as_long_flag():
    options = build_options(_AbbrevInputs)

    @click.command()
    def cmd(**kwargs):
        click.echo(f"{kwargs['ascii_sky']!r} {kwargs['polarisation']!r}")

    for opt in options:
        cmd.params.append(opt)

    runner = CliRunner()
    via_long = runner.invoke(cmd, ["--ms", "x", "--ascii-sky", "cat.txt"])
    via_short = runner.invoke(cmd, ["--ms", "x", "-as", "cat.txt", "-pol"])
    assert via_long.output.strip() == "'cat.txt' True"
    assert via_short.output.strip() == "'cat.txt' True"


def test_unflatten_kwargs_keeps_populated_multiple_values():
    # a non-empty tuple from a `multiple=True` option is a real value and
    # must survive unflattening.
    flat = {"ms": "test.ms", "channel_range": (10, 20), "weight": ("briggs", 0.5)}
    nested = unflatten_kwargs(_TupleInputs, flat)
    assert nested == {"ms": "test.ms", "channel_range": (10, 20), "weight": ("briggs", 0.5)}
    inputs = _TupleInputs(**nested)
    assert inputs.channel_range == (10, 20)
    assert inputs.weight == ("briggs", 0.5)
