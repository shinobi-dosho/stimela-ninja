from pathlib import Path

import subprocess

import pytest

from shinobi.exceptions import BackendError
from shinobi.offload.ssh import (
    _venv_activation,
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


@pytest.mark.parametrize(
    "spec",
    [
        # ssh/rsync take the host positionally, so a leading '-' is read as
        # an option -- `-oProxyCommand=...` runs a command on the local box.
        "-oProxyCommand=touch /tmp/pwned:/path",
        "-J evil:/path",
        # nothing shell-ish should survive into an argv element either
        "host;touch /tmp/x:/path",
        "$(id):/path",
    ],
)
def test_parse_remote_rejects_option_shaped_and_exotic_hosts(spec):
    with pytest.raises(ValueError, match="not a plain"):
        parse_remote(spec)


def test_parse_remote_still_accepts_ordinary_hosts():
    assert parse_remote("host:/p").host == "host"
    assert parse_remote("user@host.example.com:/p").host == "user@host.example.com"
    assert parse_remote("user-name@ilifu-slurm-1:/p").host == "user-name@ilifu-slurm-1"


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

    assert calls[0][0] == ["ssh", "--", "host", "bash -lc 'mkdir -p /remote/path'"]
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
    assert args[:3] == ["ssh", "--", "host"]
    assert len(args) == 4  # single trailing arg after `-- host` -- see _ssh()'s docstring on why
    remote_cmd = args[-1]
    assert remote_cmd.startswith("bash -lc ")
    assert "setsid bash -c" in remote_cmd
    assert ". venv/bin/activate" in remote_cmd
    assert "recipe.py:tool" in remote_cmd
    assert "/remote/path/ninja-run-" in remote_cmd  # log/exit paths are absolute, not cwd-relative


def test_launch_remote_honours_a_custom_launcher(monkeypatch):
    """A downstream CLI runs its own entry point, and its log/exit files
    are named after it so they can't collide with a `ninja run`'s.
    """
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="4321\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    handle = launch_remote(
        RemoteSpec("host", "/remote/path"),
        "line.yaml",
        ["--backend", "apptainer"],
        add_venv=False,
        launcher=["caracal", "run"],
    )

    remote_cmd = captured["args"][-1]
    assert "caracal run line.yaml --backend apptainer" in remote_cmd
    assert "ninja run" not in remote_cmd
    assert handle.log_file.startswith("caracal-run-")
    assert handle.exit_file.startswith("caracal-run-")
    assert "/remote/path/caracal-run-" in remote_cmd


def test_launch_remote_defaults_to_ninja_run(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    handle = launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], add_venv=False)

    assert "ninja run recipe.py:tool" in captured["args"][-1]
    assert handle.log_file.startswith("ninja-run-")


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
    assert args[:3] == ["ssh", "--", "host"]
    assert len(args) == 4  # single trailing arg
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


# --------------------------------------------------------------------------
# The venv activation snippet
# --------------------------------------------------------------------------


def _run_snippet(tmp_path, snippet: str, *, present: tuple[str, ...]) -> str:
    """Run the fragment for real, against a given on-disk shape.

    Executed rather than pattern-matched: the bug it replaces was a shell
    *precedence* error -- `A && B || C && D` parses as `((A && B) || C) && D` --
    which reads perfectly plausibly and which no amount of substring assertion
    would have caught.
    """
    for name in present:
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin" / "activate").write_text(f'echo "sourced {name}"\n')
    out = subprocess.run(["bash", "-c", snippet], cwd=tmp_path, capture_output=True, text=True, check=False)
    return (out.stdout + out.stderr).strip()


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (("venv", ".venv"), "sourced venv"),
        (("venv",), "sourced venv"),
        ((".venv",), "sourced .venv"),
    ],
)
def test_exactly_one_venv_is_sourced(tmp_path, present, expected):
    """`venv/` wins when both exist, and only one is ever sourced.

    The `&&`/`||` chain this replaced sourced *both* when both were present,
    `.venv` last -- so the documented `venv/`-first order was backwards in
    exactly the case where it mattered.
    """
    out = _run_snippet(tmp_path, _venv_activation("/remote/path"), present=present)
    assert out == expected


def test_a_missing_second_venv_is_not_an_error(tmp_path):
    """With only `venv/` present the old chain still ran `source .venv/...`,
    putting "No such file or directory" in the log of a run that was fine.
    """
    out = _run_snippet(tmp_path, _venv_activation("/remote/path"), present=("venv",))
    assert "No such file" not in out


def test_no_venv_at_all_says_so(tmp_path):
    """`--add-venv` defaults to on, so this is the common case on a host that
    has no venv -- and it used to be silent, leaving `ninja run` to resolve
    against whatever the login shell's PATH held.
    """
    out = _run_snippet(tmp_path, _venv_activation("/some/where"), present=())
    assert "no venv found under /some/where" in out
    assert "venv/, .venv/" in out


def test_the_activation_reaches_the_calling_shell(tmp_path):
    """`source` inside `( ... )` would change only that subshell's environment,
    discarded the instant it exits and long before `ninja run` sees it.

    Tested by running it and looking, rather than by asserting the fragment
    has no parentheses -- it does have some, in the message text, and a
    structural check would either fail on those or be weakened until it proved
    nothing. What matters is whether the activation survives into the next
    command, so that is what this asks.
    """
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "activate").write_text("export NINJA_PROBE=activated\n")
    out = subprocess.run(
        ["bash", "-c", _venv_activation("/remote/path") + 'echo "$NINJA_PROBE"'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout.strip() == "activated"
