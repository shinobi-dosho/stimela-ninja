from click.testing import CliRunner

from shinobi.cli import main

FIXTURES = "tests/fixtures/sample_targets.py"


# -- ninja compile (offload) --


def test_compile_recipe_prints_dependency_chained_scripts():
    result = CliRunner().invoke(
        main,
        ["compile", f"{FIXTURES}:path_pipe", "--ms", "/scratch/obs.ms", "--container-runtime", "none"],
    )
    assert result.exit_code == 0, result.output
    assert "===== ms_make =====" in result.output
    assert "===== ms_use  (afterok: ms_make) =====" in result.output
    # the input path flows statically from ms_make's output into ms_use's argv
    assert "use --ms /scratch/obs.ms" in result.output


def test_compile_rejects_non_offloadable_recipe_cleanly():
    # `chained` wires a str (non-path) output between steps -> not offloadable
    result = CliRunner().invoke(main, ["compile", f"{FIXTURES}:chained", "--name", "obs"])
    assert result.exit_code != 0
    assert "cannot be offloaded" in result.output
    assert "non-path output" in result.output


def test_compile_rejects_non_recipe_target():
    result = CliRunner().invoke(main, ["compile", f"{FIXTURES}:greet", "--text", "hi"])
    assert result.exit_code != 0
    assert "not a Recipe" in result.output


def test_compile_rejects_unknown_engine():
    result = CliRunner().invoke(main, ["compile", f"{FIXTURES}:path_pipe", "--engine", "argo"])
    assert result.exit_code != 0
    assert "unknown engine" in result.output


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


# -- caching options --


def test_run_help_shows_cache_options():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--cache-dir" in result.output
    assert "--no-cache" in result.output


def test_run_with_no_cache_flag_still_executes():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hi", "--no-cache"])
    assert result.exit_code == 0, result.output


def test_run_with_cache_dir_option_still_executes(tmp_path):
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet", "--text", "hi", "--cache-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output


# -- ninja run --remote --


def test_run_remote_rejects_dotted_module_target():
    result = CliRunner().invoke(
        main, ["run", "shinobi.cli:main", "--remote", "user@host:/path"]
    )
    assert result.exit_code != 0
    assert "local file target" in result.output


def test_run_remote_rejects_dryrun_combo():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet", "--remote", "user@host:/path", "--dryrun"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_run_remote_rejects_cache_options():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet", "--remote", "user@host:/path", "--no-cache"]
    )
    assert result.exit_code != 0
    assert "local runs only" in result.output


def test_run_remote_rejects_malformed_spec():
    result = CliRunner().invoke(
        main, ["run", f"{FIXTURES}:greet", "--remote", "no-colon-here"]
    )
    assert result.exit_code != 0
    assert "user@host:/path" in result.output


def test_run_help_shows_remote_options():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--remote" in result.output
    assert "--add-venv" in result.output
    assert "--include" in result.output
