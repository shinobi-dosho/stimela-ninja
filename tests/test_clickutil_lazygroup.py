"""`shinobi.clickutil.LazyGroup` -- the three things it adds to click's recipe.

Click's published version is covered by click; what needs testing here is
where this one differs: factories as declarations, `short_help` that keeps
`--help` from building anything, a lazily-resolved mapping, and caching.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from shinobi.clickutil import LazyGroup, LazySubcommand


@click.command("built")
def _target():
    """A real command, reachable by import path."""
    click.echo("ran built")


def _make(name: str, calls: list[str]) -> click.Command:
    @click.command(name)
    def command():
        click.echo(f"ran {name}")

    calls.append(name)
    return command


# -- factories --------------------------------------------------------------


def test_a_factory_is_not_called_until_the_command_is_invoked():
    calls: list[str] = []

    @click.group(cls=LazyGroup, lazy_subcommands={"a": lambda: _make("a", calls), "b": lambda: _make("b", calls)})
    def cli():
        pass

    assert calls == []
    result = CliRunner().invoke(cli, ["a"])
    assert result.exit_code == 0, result.output
    assert "ran a" in result.output
    assert calls == ["a"], "invoking one subcommand must not build the others"


def test_a_factory_result_is_cached():
    """The published recipe reloads per `get_command`, which is free for a
    module-level object and is not for a factory that does real work.
    """
    calls: list[str] = []

    @click.group(cls=LazyGroup, lazy_subcommands={"a": LazySubcommand(lambda: _make("a", calls))})
    def cli():
        pass

    ctx = click.Context(cli)
    first = cli.get_command(ctx, "a")
    second = cli.get_command(ctx, "a")
    assert first is second
    assert calls == ["a"]


def test_a_non_command_declaration_is_refused():
    @click.group(cls=LazyGroup, lazy_subcommands={"bad": lambda: "not a command"})
    def cli():
        pass

    with pytest.raises(ValueError, match="non-command object"):
        cli.get_command(click.Context(cli), "bad")


# -- click's own string form ------------------------------------------------


def test_the_import_path_form_still_works():
    for path in ("tests.test_clickutil_lazygroup._target", "tests.test_clickutil_lazygroup:_target"):

        @click.group(cls=LazyGroup, lazy_subcommands={"built": path})
        def cli():
            pass

        result = CliRunner().invoke(cli, ["built"])
        assert result.exit_code == 0, result.output
        assert "ran built" in result.output


def test_a_malformed_import_path_says_so():
    @click.group(cls=LazyGroup, lazy_subcommands={"x": "nodots"})
    def cli():
        pass

    with pytest.raises(ValueError, match="module.attr"):
        cli.get_command(click.Context(cli), "x")


# -- help without building --------------------------------------------------


def test_declared_short_help_is_used_without_building_the_command():
    calls: list[str] = []

    @click.group(
        cls=LazyGroup,
        lazy_subcommands={"a": LazySubcommand(lambda: _make("a", calls), short_help="does the a thing")},
    )
    def cli():
        pass

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "does the a thing" in result.output
    assert calls == [], "--help must not build a command whose help was declared"


def test_help_falls_back_to_building_when_no_short_help_is_declared():
    calls: list[str] = []

    @click.group(cls=LazyGroup, lazy_subcommands={"a": lambda: _make("a", calls)})
    def cli():
        pass

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "a" in result.output
    assert calls == ["a"], "with nothing declared it must still produce a correct listing"


def test_a_long_short_help_is_shortened_to_one_line():
    long = "first line of the description\nand a second one that goes on " + "x" * 200

    @click.group(cls=LazyGroup, lazy_subcommands={"a": LazySubcommand(lambda: _make("a", []), short_help=long)})
    def cli():
        pass

    result = CliRunner().invoke(cli, ["--help"])
    listing = [line for line in result.output.splitlines() if line.strip().startswith("a ")]
    assert listing, result.output
    assert len(listing[0]) < 120
    assert listing[0].rstrip().endswith("...")


# -- listing, precedence, lazy mapping --------------------------------------


def test_eager_and_lazy_commands_are_listed_together_and_sorted():
    @click.group(cls=LazyGroup, lazy_subcommands={"zebra": lambda: _make("zebra", []), "alpha": lambda: _make("alpha", [])})
    def cli():
        pass

    @cli.command("middle")
    def _middle():
        pass

    names = cli.list_commands(click.Context(cli))
    assert names == ["alpha", "middle", "zebra"], "the recipe's base+lazy concatenation sorts only within each half"


def test_an_eager_command_shadows_a_lazy_one_of_the_same_name():
    calls: list[str] = []

    @click.group(cls=LazyGroup, lazy_subcommands={"dup": lambda: _make("dup", calls)})
    def cli():
        pass

    @cli.command("dup")
    def _dup():
        click.echo("eager")

    result = CliRunner().invoke(cli, ["dup"])
    assert "eager" in result.output
    assert calls == []


def test_an_unknown_command_is_none_not_an_error():
    @click.group(cls=LazyGroup, lazy_subcommands={"a": lambda: _make("a", [])})
    def cli():
        pass

    assert cli.get_command(click.Context(cli), "nope") is None


def test_the_mapping_itself_can_be_deferred():
    """Discovering the *names* can cost something too -- caracal scans 19
    YAML headers for its worker commands -- and a group is constructed on
    every invocation.
    """
    resolved: list[int] = []

    def discover():
        resolved.append(1)
        return {"a": lambda: _make("a", [])}

    @click.group(cls=LazyGroup, lazy_subcommands=discover)
    def cli():
        pass

    assert resolved == [], "constructing the group must not resolve the mapping"
    ctx = click.Context(cli)
    cli.list_commands(ctx)
    cli.list_commands(ctx)
    assert resolved == [1], "and it must be resolved at most once"
