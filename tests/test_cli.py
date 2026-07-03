from click.testing import CliRunner

from shinobi.cli import main

FIXTURES = "tests/fixtures/sample_targets.py"


def test_run_cab_target():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hello there"])
    assert result.exit_code == 0, result.output
    assert "hello there" in result.output


def test_run_recipe_target_calls_function_directly():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:double", "--n", "21"])
    assert result.exit_code == 0, result.output
    assert "42" in result.output


def test_run_missing_required_option_errors():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:double"])
    assert result.exit_code != 0
    assert "--n" in result.output or "Missing option" in result.output


def test_run_unknown_target_errors():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:nope"])
    assert result.exit_code != 0


def test_run_help_shows_dynamic_options():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:double", "--help"])
    assert result.exit_code == 0, result.output
    assert "--n" in result.output


def test_option_flag_name_roundtrips_underscore():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet_image", "--restored-image", "/data/img.fits"]
    )
    assert result.exit_code == 0, result.output
    assert "/data/img.fits" in result.output


def test_run_failing_cab_reports_nonzero_exit():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:fail"])
    assert result.exit_code != 0
