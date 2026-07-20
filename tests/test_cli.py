import pytest
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
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet_image", "--restored-image", "/data/img.fits"])
    assert result.exit_code == 0, result.output
    assert "/data/img.fits" in result.output


def test_run_failing_cab_reports_nonzero_exit():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:fail"])
    assert result.exit_code != 0


def test_run_failing_recipe_reports_nonzero_exit():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:fail_recipe"])
    assert result.exit_code != 0


def test_run_cab_target_dryrun_shows_argv_and_does_not_execute():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--dryrun", "--text", "echo-me"])
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


def test_run_flattened_group_option_reaches_the_nested_model():
    """Regression test: `build_options` flattens a nested-`BaseModel` group
    field to a `--parent-child` option (see `clickutil`'s docstring), but
    `cli.py`'s callbacks used to pass the flat kwargs straight to the model
    without its matching inverse (`unflatten_kwargs`) -- silently dropping
    a `--sub-value` override instead of nesting it back under `sub`. No
    shipped cab loader produces this shape today, but `clickutil` is
    documented as reusable by a downstream project's own CLI, so the gap
    was real even if unexercised.
    """
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:group_cab", "--dryrun", "--sub-value", "hi-nested"])
    assert result.exit_code == 0, result.output
    assert "hi-nested" in result.output


# -- caching options --


def test_run_help_shows_cache_options():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--cache-dir" in result.output
    assert "--no-cache" in result.output


def test_run_with_no_cache_flag_still_executes():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hi", "--no-cache"])
    assert result.exit_code == 0, result.output


# -- live stdout/stderr streaming (default on) / --quiet --


def test_run_default_streams_live_with_label_prefix():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hello there"])
    assert result.exit_code == 0, result.output
    assert "[greet] --text hello there" in result.output
    # streamed live -> no separate end-of-run dump repeating the same text
    assert result.output.count("hello there") == 1


def test_run_quiet_flag_suppresses_streaming_and_dumps_at_end():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hello there", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "[greet]" not in result.output
    assert "hello there" in result.output


def test_run_help_shows_quiet_option():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--quiet" in result.output


def test_run_with_cache_dir_option_still_executes(tmp_path):
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--text", "hi", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output


# -- ninja run --remote --


def test_run_remote_rejects_dotted_module_target():
    result = CliRunner().invoke(main, ["run", "shinobi.cli:main", "--remote", "user@host:/path"])
    assert result.exit_code != 0
    assert "local file target" in result.output


def test_run_remote_rejects_dryrun_combo():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--remote", "user@host:/path", "--dryrun"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_run_remote_rejects_cache_options():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--remote", "user@host:/path", "--no-cache"])
    assert result.exit_code != 0
    assert "local runs only" in result.output


def test_run_remote_rejects_malformed_spec():
    result = CliRunner().invoke(main, ["run", f"{FIXTURES}:greet", "--remote", "no-colon-here"])
    assert result.exit_code != 0
    assert "user@host:/path" in result.output


def test_run_help_shows_remote_options():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--remote" in result.output
    assert "--add-venv" in result.output
    assert "--include" in result.output


# -- ninja --log-file/--log-dir/--log-level (main group) --


def _invoke_with_config_probe(args):
    """Invoke `ninja <args> <probe>` where the probe command echoes the
    resolved AppConfig.log settings, then unregister the probe."""
    import click

    @main.command("probe-log-config", hidden=True)
    @click.pass_context
    def probe(ctx: click.Context) -> None:
        log = ctx.obj.log
        click.echo(f"{log.file} {log.dir} {log.level}")

    try:
        return CliRunner().invoke(main, [*args, "probe-log-config"])
    finally:
        del main.commands["probe-log-config"]


def test_log_options_override_config(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("log:\n  dir: from-file\n  level: WARNING\n")
    result = _invoke_with_config_probe(["--config", str(cfg), "--log-file", "run.log", "--log-dir", "logs", "--log-level", "debug"])
    assert result.exit_code == 0, result.output
    # --log-level is case-insensitive and normalizes to the canonical name.
    assert result.output.strip() == "run.log logs DEBUG"


def test_log_options_absent_leave_config_untouched(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("log:\n  dir: from-file\n")
    result = _invoke_with_config_probe(["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "None from-file INFO"


def test_log_level_rejects_unknown_name():
    result = CliRunner().invoke(main, ["--log-level", "CHATTY", "version"])
    assert result.exit_code != 0
    assert "CHATTY" in result.output


# -- run-log file (ninja --log-file ... run ...) --


@pytest.fixture
def _detach_file_logging():
    """Detach the file handler the invoked CLI attached, so later tests in
    this process don't keep appending to a stale tmp file."""
    yield
    from shinobi.config import LogConfig
    from shinobi.logsetup import setup_file_logging

    setup_file_logging(LogConfig())


def test_run_writes_log_file(tmp_path, _detach_file_logging):
    result = CliRunner().invoke(
        main,
        ["--log-file", "run.log", "--log-dir", str(tmp_path), "run", f"{FIXTURES}:greet", "--text", "hello log"],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "run.log").read_text()
    assert "INFO step greet: starting" in text
    assert "[greet] --text hello log" in text
    assert "INFO step greet: finished (returncode 0)" in text


def test_run_log_debug_includes_argv(tmp_path, _detach_file_logging):
    result = CliRunner().invoke(
        main,
        ["--log-file", "run.log", "--log-dir", str(tmp_path), "--log-level", "DEBUG", "run", f"{FIXTURES}:greet", "--text", "hi"],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "run.log").read_text()
    assert "DEBUG step greet: backend=native argv: /bin/echo --text hi" in text


def test_run_log_records_failure_as_error(tmp_path, _detach_file_logging):
    result = CliRunner().invoke(
        main,
        ["--log-file", "run.log", "--log-dir", str(tmp_path), "run", f"{FIXTURES}:fail"],
    )
    assert result.exit_code != 0
    text = (tmp_path / "run.log").read_text()
    assert "ERROR step fail: failed (returncode 1)" in text


def test_run_log_recipe_logs_each_substep_once(tmp_path, _detach_file_logging):
    result = CliRunner().invoke(
        main,
        ["--log-file", "run.log", "--log-dir", str(tmp_path), "run", f"{FIXTURES}:chained", "--name", "obs"],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "run.log").read_text()
    assert "step chained: starting" in text
    assert "step chained.make_file: finished (returncode 0)" in text
    assert "step chained.use_file: finished (returncode 0)" in text
    assert "step chained: finished (returncode 0)" in text
    # Sub-step output appears under the sub-step's dotted label only -- the
    # recipe's own aggregated stdout is not re-logged under `[chained]`.
    assert "[chained.make_file] " in text
    assert "[chained] " not in text


def test_run_without_log_file_writes_nothing(tmp_path, _detach_file_logging):
    result = CliRunner().invoke(
        main,
        ["--log-dir", str(tmp_path), "run", f"{FIXTURES}:greet", "--text", "hi"],
    )
    assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == []
