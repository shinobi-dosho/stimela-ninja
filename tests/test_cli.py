from click.testing import CliRunner

from shinobi.cli import main

FIXTURES = "tests/fixtures/sample_targets.py"


def test_run_cab_target():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hello there"])
    assert result.exit_code == 0, result.output
    assert "hello there" in result.output


def test_run_stepref_target():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet_step", "--text", "via step"])
    assert result.exit_code == 0, result.output
    assert "via step" in result.output


def test_run_recipe_target_executes_substeps():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:chained", "--name", "obs"])
    assert result.exit_code == 0, result.output


def test_run_missing_required_option_errors():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet_image"])
    assert result.exit_code != 0
    assert "--restored-image" in result.output or "Missing option" in result.output


def test_run_unknown_target_errors():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:nope"])
    assert result.exit_code != 0


def test_run_help_shows_dynamic_options():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--help"])
    assert result.exit_code == 0, result.output
    assert "--text" in result.output


def test_run_bare_help_shows_run_commands_own_help():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "TARGET" in result.output
    assert "--dryrun" in result.output


def test_run_with_no_args_shows_help():
    result = CliRunner().invoke(main, ["run"])
    assert "TARGET" in result.output
    assert "--dryrun" in result.output


def test_option_flag_name_roundtrips_underscore():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet_image", "--restored-image", "/data/img.fits"]
    )
    assert result.exit_code == 0, result.output
    assert "/data/img.fits" in result.output


def test_run_failing_cab_reports_nonzero_exit():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:fail"])
    assert result.exit_code != 0


def test_run_cab_target_dryrun_shows_argv_and_does_not_execute():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet", "--dryrun", "--text", "echo-me"]
    )
    assert result.exit_code == 0, result.output
    # a bare Cab dryrun echoes its build_argv (see AGENTS.md), not a box graph
    assert "/bin/echo" in result.output
    assert "--text" in result.output


def test_run_recipe_target_dryrun_shows_declared_graph():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:chained", "--dryrun"])
    assert result.exit_code == 0, result.output
    assert "[ make_file ]" in result.output
    assert "[ use_file ]" in result.output
    assert "v" in result.output  # a dependency edge was drawn
