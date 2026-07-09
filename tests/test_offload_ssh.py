from pathlib import Path

import pytest

from shinobi.exceptions import BackendError
from shinobi.offload.ssh import (
    RemoteSpec,
    find_cab_deps,
    launch_remote,
    parse_remote,
    status_ssh,
    sync_to_remote,
)

FIXTURE_DIR = Path("tests/fixtures/remote_target")


# -- parse_remote --


def test_parse_remote_splits_host_and_path():
    spec = parse_remote("user@host:/path/to/run")
    assert spec.host == "user@host"
    assert spec.path == "/path/to/run"


def test_parse_remote_rejects_missing_colon():
    with pytest.raises(ValueError, match="user@host:/path"):
        parse_remote("no-colon-here")


# -- find_cab_deps --


def test_find_cab_deps_resolves_path_dot_parent_expression_and_follows_include():
    deps, warnings = find_cab_deps(FIXTURE_DIR / "recipe.py")
    assert warnings == []
    assert (FIXTURE_DIR / "cabs" / "tool.yml").resolve() in deps
    # tool.yml's _include: [vars.yml] should be followed too
    assert (FIXTURE_DIR / "cabs" / "vars.yml").resolve() in deps


def test_find_cab_deps_follows_include_nested_under_inputs():
    """Regression test: real cult-cargo cabs (cubical.yml/quartical.yml)
    nest `_include:` under `inputs:`/`outputs:`, not just at the top level
    (see cultcargo.py's own module docstring) -- `_include_deps` used to
    only scan the top level, silently missing this dependency for
    `--remote` syncs.
    """
    deps, warnings = find_cab_deps(FIXTURE_DIR / "recipe_nested_include.py")
    assert warnings == []
    assert (FIXTURE_DIR / "cabs" / "nested_include_tool.yml").resolve() in deps
    assert (FIXTURE_DIR / "cabs" / "nested_vars.yml").resolve() in deps


def test_find_cab_deps_warns_instead_of_raising_on_unresolvable_call():
    deps, warnings = find_cab_deps(FIXTURE_DIR / "recipe_unresolvable.py")
    assert deps == []
    assert len(warnings) == 1
    assert "could not statically resolve" in warnings[0]


# -- sync_to_remote / launch_remote / status_ssh (subprocess mocked) --


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_sync_to_remote_mkdirs_then_rsyncs_with_relative_paths(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _FakeProc(returncode=0)

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    sync_to_remote(FIXTURE_DIR, [Path("recipe.py"), Path("cabs/tool.yml")], RemoteSpec("host", "/remote/path"))

    assert calls[0][0] == ["ssh", "host", "bash -lc 'mkdir -p /remote/path'"]
    rsync_args, rsync_kwargs = calls[1]
    assert rsync_args[0] == "rsync"
    assert "-R" in rsync_args
    assert "recipe.py" in rsync_args
    assert "cabs/tool.yml" in rsync_args
    assert rsync_args[-1] == "host:/remote/path/"
    assert rsync_kwargs["cwd"] == FIXTURE_DIR


def test_sync_to_remote_raises_backend_error_on_rsync_failure(monkeypatch):
    def fake_run(args, **kwargs):
        if args[0] == "ssh":
            return _FakeProc(returncode=0)
        return _FakeProc(returncode=1, stderr="connection refused")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    with pytest.raises(BackendError, match="connection refused"):
        sync_to_remote(FIXTURE_DIR, [Path("recipe.py")], RemoteSpec("host", "/remote/path"))


def test_launch_remote_captures_pid_from_echoed_output(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="12345\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    handle = launch_remote(RemoteSpec("host", "/remote/path"), "recipe.py:tool", ["--text", "hi"], add_venv=True)

    assert handle.pid == "12345"
    assert handle.host == "host"
    assert handle.path == "/remote/path"
    args = captured["args"]
    assert args[:2] == ["ssh", "host"]
    assert len(args) == 3  # single trailing arg -- see _ssh()'s docstring on why
    remote_cmd = args[-1]
    assert remote_cmd.startswith("bash -lc ")
    assert "setsid bash -c" in remote_cmd
    assert "source venv/bin/activate" in remote_cmd
    assert "recipe.py:tool" in remote_cmd
    assert "/remote/path/ninja-run-" in remote_cmd  # log/exit paths are absolute, not cwd-relative


def test_launch_remote_raises_on_non_pid_output(monkeypatch):
    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="not-a-pid\n"),
    )
    with pytest.raises(BackendError, match="unexpected launch output"):
        launch_remote(RemoteSpec("host", "/remote/path"), "recipe.py:tool", [], add_venv=False)


def test_status_ssh_sends_a_single_trailing_arg_and_uses_absolute_exit_path(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="RUNNING\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    handle = {"host": "host", "path": "/remote/path", "pid": "1", "log_file": "l.log", "exit_file": "e.exit"}
    status_ssh(handle)

    args = captured["args"]
    assert args[:2] == ["ssh", "host"]
    assert len(args) == 3
    assert "/remote/path/e.exit" in args[-1]


def test_status_ssh_reports_running(monkeypatch):
    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="RUNNING\n"),
    )
    handle = {"host": "host", "path": "/remote/path", "pid": "1", "log_file": "l.log", "exit_file": "e.exit"}
    assert status_ssh(handle) == "RUNNING"


def test_status_ssh_reports_success_and_failure(monkeypatch):
    handle = {"host": "host", "path": "/remote/path", "pid": "1", "log_file": "l.log", "exit_file": "e.exit"}

    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="0\n"),
    )
    assert status_ssh(handle) == "FINISHED (success)"

    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="1\n"),
    )
    assert "FINISHED (exit 1)" in status_ssh(handle)
