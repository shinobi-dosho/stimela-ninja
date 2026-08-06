from pathlib import Path

import json
import shlex
import subprocess

import pytest

from shinobi.backends.venv import digest_of_dists
from shinobi.exceptions import BackendError
from shinobi.offload.remote_venv import (
    COLLIDED,
    PUBLISHED,
    EnvInputs,
    PlatformTriple,
    Sentinel,
    env_id,
    lock_source_for,
    sha256_hex,
    venv_dir,
)
from shinobi.offload.ssh import (
    _venv_activation,
    RemoteSpec,
    resolve_remote_venv,
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
    handle = launch_remote(RemoteSpec("host", "/remote/path"), "recipe.py:tool", ["--text", "hi"], venv="use")

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


def test_launch_remote_defaults_to_activating_a_venv(monkeypatch):
    """`venv` is keyword-only with a default, where `add_venv` was required.
    A caller that names neither gets `use`, which is what every existing
    caller passing `add_venv=True` was getting."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [])
    assert ". venv/bin/activate" in captured["args"][-1]


@pytest.mark.parametrize("add_venv,activates", [(True, True), (False, False)])
def test_launch_remote_still_accepts_the_deprecated_boolean(monkeypatch, add_venv, activates):
    """caracal's own `--remote` wrapper passes `add_venv=` into this
    function (`caracal/remote.py`), so dropping it here would break a
    downstream release rather than deprecate one."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    with pytest.warns(DeprecationWarning, match="add-venv"):
        launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], add_venv=add_venv)
    assert (". venv/bin/activate" in captured["args"][-1]) is activates


def test_launch_remote_refuses_two_venv_flags_that_disagree(monkeypatch):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(returncode=0, stdout="1\n"))
    with pytest.raises(ValueError, match="different things"):
        launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], venv="off", add_venv=True)


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
        venv="off",
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
    handle = launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], venv="off")

    assert "ninja run recipe.py:tool" in captured["args"][-1]
    assert handle.log_file.startswith("ninja-run-")


def test_launch_remote_exports_env_after_the_venv_activation(monkeypatch):
    """Order matters: the venv's own `activate` sets PATH and can set more,
    so a caller-supplied override has to come after it to win.
    """
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="7\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    launch_remote(
        RemoteSpec("host", "/remote/path"),
        "recipe.py:tool",
        [],
        venv="use",
        env={"APPTAINER_CACHEDIR": "/home/u/.apptainer/cache"},
    )

    remote_cmd = captured["args"][-1]
    assert "export APPTAINER_CACHEDIR=/home/u/.apptainer/cache" in remote_cmd
    assert remote_cmd.index("bin/activate") < remote_cmd.index("export APPTAINER_CACHEDIR")
    assert remote_cmd.index("export APPTAINER_CACHEDIR") < remote_cmd.index("ninja run")


def test_launch_remote_quotes_env_values(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="7\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    launch_remote(
        RemoteSpec("host", "/remote/path"),
        "recipe.py:tool",
        [],
        venv="off",
        env={"EVIL": "x; rm -rf /"},
    )

    # Two layers of quoting sit between here and the export -- `bash -lc
    # <inner>` and `setsid bash -c <wrapped>` -- so unwrap them the way the
    # remote shells will, rather than matching the doubly-escaped text.
    inner = shlex.split(captured["args"][-1])[2]
    wrapped = shlex.split(inner)[3]
    assert "export EVIL='x; rm -rf /'" in wrapped  # one word, not a command separator


@pytest.mark.parametrize("name", ["A B", "1BAD", "A=B", "", "A;rm -rf /"])
def test_launch_remote_refuses_a_malformed_env_name(monkeypatch, name):
    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="7\n"),
    )
    with pytest.raises(ValueError, match="not a valid environment variable name"):
        launch_remote(RemoteSpec("host", "/p"), "r.py:t", [], venv="off", env={name: "x"})


def test_launch_remote_raises_on_non_pid_output(monkeypatch):
    monkeypatch.setattr(
        "shinobi.offload.ssh.subprocess.run",
        lambda args, **kwargs: _FakeProc(returncode=0, stdout="not-a-pid\n"),
    )
    with pytest.raises(BackendError, match="unexpected launch output"):
        launch_remote(RemoteSpec("host", "/remote/path"), "recipe.py:tool", [], venv="off")


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


# -- resolve_remote_venv --


class _FakeRemote:
    """A scripted stand-in for the remote host.

    Keyed on what each command *is* rather than on call order: the number of
    round-trips is an implementation detail this suite should not pin, but
    which command produces which answer is exactly the contract under test.
    """

    def __init__(
        self,
        *,
        uv="uv 0.11.21",
        can_bootstrap=True,
        bootstrap_uv="uv 0.12.2",
        sentinel="",
        publish=PUBLISHED,
        build_rc=0,
        sentinel_after_collision=None,
        dists=None,
    ):
        self.uv = uv
        self.can_bootstrap = can_bootstrap
        self.bootstrap_uv = bootstrap_uv
        self.dists = dists
        self.sentinel = sentinel
        self.publish = publish
        self.build_rc = build_rc
        self.sentinel_after_collision = sentinel_after_collision
        self.commands: list[str] = []
        self._sentinel_reads = 0

    def __call__(self, host, command):
        self.commands.append(command)
        if "shinobi-platform" in command:
            ensurepip = "yes" if self.can_bootstrap else "no"
            return _FakeProc(
                returncode=0,
                stdout=f"shinobi-platform:x86_64/glibc-2.39/3.11\nshinobi-uv:{self.uv}\nshinobi-ensurepip:{ensurepip}\n",
            )
        if "python3 -m venv" in command:
            return _FakeProc(returncode=0, stdout=f"shinobi-uv:{self.bootstrap_uv}\n")
        if command.startswith("cat "):
            self._sentinel_reads += 1
            if self._sentinel_reads > 1 and self.sentinel_after_collision is not None:
                return _FakeProc(returncode=0, stdout=self.sentinel_after_collision)
            return _FakeProc(returncode=0, stdout=self.sentinel)
        if "venv --relocatable" in command:
            stdout = "shinobi-venv-python:3.11.9\n"
            if self.dists is not None:
                stdout += f"shinobi-dists:{json.dumps(self.dists)}\n"
            return _FakeProc(returncode=self.build_rc, stdout=stdout, stderr="build blew up")
        if "os.rename" in command:
            return _FakeProc(returncode=0, stdout=f"{self.publish}\n")
        return _FakeProc(returncode=0, stdout="")

    def ran(self, needle):
        return any(needle in c for c in self.commands)


@pytest.fixture
def fake_remote(monkeypatch):
    def install(**kwargs):
        remote = _FakeRemote(**kwargs)
        monkeypatch.setattr("shinobi.offload.ssh._ssh", remote)
        monkeypatch.setattr("shinobi.offload.ssh.sync_to_remote", lambda *a, **k: None)
        return remote

    return install


@pytest.fixture
def lock_source(tmp_path):
    (tmp_path / "uv.lock").write_text("# lock\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    return lock_source_for(tmp_path / "uv.lock")


def _matching_sentinel(remote_path, source):
    """The sentinel a previous `sync` of `source` would have left behind."""
    lock, pyproject = source.read()
    inputs = EnvInputs(
        lock=lock,
        pyproject=pyproject,
        extras=(),
        groups=(),
        python_request="",
        mode=source.mode,
        platform=PlatformTriple("x86_64", "glibc-2.39", "3.11"),
    )
    eid = env_id(inputs)
    sentinel = Sentinel(
        env_id=eid,
        lock_sha256=sha256_hex(inputs.lock),
        pyproject_sha256=sha256_hex(inputs.pyproject),
        extras=(),
        groups=(),
        python_request="",
        mode=inputs.mode,
        platform_triple=str(inputs.platform),
    )
    return sentinel.to_json(), venv_dir(remote_path, eid)


def test_off_resolves_to_nothing_without_touching_the_host(fake_remote, lock_source):
    remote = fake_remote()
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "off", lock_source)
    assert resolved.path is None
    assert remote.commands == []


def test_use_without_a_lock_costs_no_round_trips(fake_remote):
    """Most recipes are not in a repository with a lock, and `use` is the
    default -- so the common case must not have acquired an ssh call."""
    remote = fake_remote()
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", None)
    assert resolved.path is None
    assert remote.commands == []


def test_sync_without_a_lock_is_refused(fake_remote):
    remote = fake_remote()
    with pytest.raises(BackendError, match="needs a lock to provision from"):
        resolve_remote_venv(RemoteSpec("host", "/p"), "sync", None)
    assert remote.commands == []


def test_sync_bootstraps_uv_on_a_host_that_has_none(fake_remote, lock_source):
    """uv ships manylinux wheels, so a host with `python3 -m venv` can
    install one from the same index the packages come from -- no
    `curl | sh`, and `--relocatable` survives, unlike a plain venv+pip."""
    remote = fake_remote(uv="", can_bootstrap=True, sentinel="")
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.provisioned
    assert remote.ran("python3 -m venv")
    assert remote.ran("pip install --quiet uv")
    # and the bootstrapped uv is the one that builds, not a bare `uv`
    build = next(c for c in remote.commands if "venv --relocatable" in c)
    assert ".uv-bootstrap/bin/uv" in build


def test_the_bootstrap_venv_is_thrown_away_with_the_staging_dir(fake_remote, lock_source):
    """A host is left exactly as it was found."""
    remote = fake_remote(uv="", can_bootstrap=True, sentinel="")
    resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert remote.ran(".uv-bootstrap")
    assert any("rm -rf" in c and "/.partial-" in c for c in remote.commands)


def test_the_bootstrapped_uv_version_is_what_gets_recorded(fake_remote, lock_source):
    """Not the probe's answer -- there wasn't one -- and not nothing:
    which uv built an environment is exactly what someone diagnosing a
    divergent build needs to see."""
    remote = fake_remote(uv="", can_bootstrap=True, sentinel="", bootstrap_uv="uv 0.12.2")
    resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    published = next(c for c in remote.commands if "os.rename" in c)
    assert "uv 0.12.2" in published


def test_sync_refuses_when_uv_can_be_neither_found_nor_bootstrapped(fake_remote, lock_source):
    """Debian and Ubuntu ship the stdlib venv module without ensurepip's
    bundled wheels, so `python3 -m venv` exists and fails partway."""
    remote = fake_remote(uv="", can_bootstrap=False)
    with pytest.raises(BackendError, match="python3-venv"):
        resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not remote.ran("uv venv")


def test_a_host_that_already_has_uv_is_not_bootstrapped(fake_remote, lock_source):
    remote = fake_remote(sentinel="")
    resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not remote.ran("pip install --quiet uv")


def test_use_adopts_a_matching_environment(fake_remote, lock_source):
    sentinel, expected = _matching_sentinel("/p", lock_source)
    fake_remote(sentinel=sentinel)
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert resolved.path == expected
    assert not resolved.provisioned


def test_sync_does_not_rebuild_what_is_already_there(fake_remote, lock_source):
    """Idempotence: the sentinel is present, so there is nothing to do. No
    stamp file, no freshness comparison, nothing that can disagree."""
    sentinel, expected = _matching_sentinel("/p", lock_source)
    remote = fake_remote(sentinel=sentinel)
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.path == expected
    assert not remote.ran("uv venv")


def test_use_falls_back_when_nothing_is_provisioned(fake_remote, lock_source):
    remote = fake_remote(sentinel="")
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert resolved.path is None
    assert any("--venv sync" in note for note in resolved.notes)
    assert not remote.ran("uv venv")


def test_use_never_writes_to_the_remote(fake_remote, lock_source):
    """Invariant 4. Only `sync` provisions."""
    remote = fake_remote(sentinel="")
    resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert not any(w in c for c in remote.commands for w in ("uv venv", "uv sync", "os.rename", "rm -rf"))


def test_use_degrades_rather_than_failing_when_the_host_misbehaves(fake_remote, lock_source, monkeypatch):
    """`use` is the default and a lock is discovered by walking up, so this
    path runs for people who have never heard of provisioning. Acquiring a
    new way to fail there would be a regression dressed as a feature."""
    fake_remote()
    monkeypatch.setattr("shinobi.offload.ssh._ssh", lambda h, c: _FakeProc(returncode=0, stdout="not a probe"))
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert resolved.path is None
    assert any("falling back" in note for note in resolved.notes)


def test_sync_provisions_and_publishes(fake_remote, lock_source):
    _sentinel_json, expected = _matching_sentinel("/p", lock_source)
    remote = fake_remote(sentinel="")
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.path == expected
    assert resolved.provisioned
    assert remote.ran("uv venv --relocatable")
    assert remote.ran("os.rename")


def test_a_failed_build_fails_the_launch_and_cleans_up(fake_remote, lock_source):
    """Invariant 7: a `sync` that cannot provision fails the launch rather
    than quietly launching into something else."""
    remote = fake_remote(sentinel="", build_rc=1)
    with pytest.raises(BackendError, match="build blew up"):
        resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert remote.ran("rm -rf")


def test_the_staging_directory_is_removed_on_success_too(fake_remote, lock_source):
    remote = fake_remote(sentinel="")
    resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert remote.ran("rm -rf")
    assert all("/.partial-" in c for c in remote.commands if "rm -rf" in c)


def test_a_concurrent_launch_winning_the_rename_is_adopted(fake_remote, lock_source):
    """The good case, and it needs no apology: whoever won built from the
    same inputs, which is what env_id means."""
    sentinel, expected = _matching_sentinel("/p", lock_source)
    fake_remote(sentinel="", publish=COLLIDED, sentinel_after_collision=sentinel)
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.path == expected
    assert any("adopting it" in note for note in resolved.notes)


def test_a_collision_with_no_sentinel_is_refused_not_cleared(fake_remote, lock_source):
    """That path may be an environment someone built by hand. Deleting it
    unasked is not this tool's decision to make."""
    remote = fake_remote(sentinel="", publish=COLLIDED, sentinel_after_collision="")
    with pytest.raises(BackendError, match="will not be overwritten"):
        resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not any("rm -rf" in c and "/.partial-" not in c for c in remote.commands)


def test_sync_refuses_to_provision_over_a_newer_clients_environment(fake_remote, lock_source):
    """FOREIGN, not ABSENT. Merging the two would make `sync` overwrite an
    environment a newer ninja owns."""
    sentinel, _expected = _matching_sentinel("/p", lock_source)
    remote = fake_remote(sentinel=sentinel.replace('"schema": 1', '"schema": 99'))
    with pytest.raises(BackendError, match="refusing to provision over"):
        resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not remote.ran("uv venv")


def test_use_ignores_a_newer_clients_environment_and_carries_on(fake_remote, lock_source):
    sentinel, _expected = _matching_sentinel("/p", lock_source)
    fake_remote(sentinel=sentinel.replace('"schema": 1', '"schema": 99'))
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert resolved.path is None
    assert any("ignoring" in note for note in resolved.notes)


def test_a_sentinel_for_another_platform_is_never_activated(fake_remote, lock_source):
    """Invariant 3. At this path it also means something is wrong -- the
    triple is *in* env_id, so a mismatch here is not our environment."""
    sentinel, _expected = _matching_sentinel("/p", lock_source)
    fake_remote(sentinel=sentinel.replace("x86_64/glibc-2.39/3.11", "aarch64/glibc-2.31/3.11"))
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "use", lock_source)
    assert resolved.path is None


# -- the activation fragment, with a resolved environment --


def test_a_resolved_environment_becomes_the_first_branch():
    fragment = _venv_activation("/p", "/p/.shinobi/venvs/deadbeef")
    assert fragment.index("deadbeef") < fragment.index("venv/bin/activate")
    assert fragment.count("if [ ") == 3  # one `if`, two `elif` -- exactly one branch runs
    # No subshell anywhere a `source` could land in: it has to change the
    # PATH of the same shell that later runs `ninja run`. The only paren in
    # the fragment is inside the single-quoted not-found message.
    assert "(" not in fragment.split("else echo '")[0]


def test_the_not_found_message_names_the_resolved_path_too():
    assert "deadbeef" in _venv_activation("/p", "/p/.shinobi/venvs/deadbeef").split("no venv found")[1]


def test_launch_remote_activates_a_resolved_environment(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], venv="sync", venv_path="/p/.shinobi/venvs/deadbeef")
    assert "/p/.shinobi/venvs/deadbeef/bin/activate" in captured["args"][-1]


def test_off_ignores_a_resolved_environment(monkeypatch):
    """`off` means source nothing, and it means it whatever else was passed."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    launch_remote(RemoteSpec("host", "/p"), "recipe.py:tool", [], venv="off", venv_path="/p/.shinobi/venvs/deadbeef")
    assert "bin/activate" not in captured["args"][-1]


# -- the divergence digest (design 4.6) --


def _local_venv(project_dir, dists):
    """A venv-shaped directory whose `bin/python` reports `dists`.

    A real one would take a `uv venv` per test; what `venv_digest` actually
    needs is an interpreter that prints the freeze JSON, so that is what this
    provides -- a shell script standing in for `bin/python`.
    """
    bindir = project_dir / ".venv" / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.write_text(f"#!/bin/sh\necho {shlex.quote(json.dumps(dists))}\n")
    python.chmod(0o755)
    return project_dir / ".venv"


def test_the_provisioned_digest_lands_in_the_sentinel(fake_remote, lock_source):
    """Recorded once at provisioning time rather than recomputed per launch
    -- which is what makes the comparison free."""
    remote = fake_remote(sentinel="", dists=["click==8.1.7"])
    resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    published = [c for c in remote.commands if "os.rename" in c][0]
    assert digest_of_dists(["click==8.1.7"]) in published


def test_a_matching_local_venv_says_nothing(fake_remote, lock_source):
    """Silence is the answer when there is nothing to report. A note on every
    provision is a note nobody reads."""
    _local_venv(lock_source.project_dir, ["click==8.1.7"])
    fake_remote(sentinel="", dists=["click==8.1.7"])
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not any("differs from" in note for note in resolved.notes)


def test_a_diverging_local_venv_is_reported_with_both_digests(fake_remote, lock_source):
    """Informational, never a check: the same version list can sit on
    different compiled C-extensions, which is why a venv step is reported
    *unpinned* in the run manifest."""
    _local_venv(lock_source.project_dir, ["click==8.1.7"])
    fake_remote(sentinel="", dists=["click==9.0.0"])
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    note = next(n for n in resolved.notes if "differs from" in n)
    assert digest_of_dists(["click==9.0.0"])[:12] in note
    assert digest_of_dists(["click==8.1.7"])[:12] in note


def test_divergence_does_not_fail_the_launch(fake_remote, lock_source):
    """A digest is not a pin, and a mismatch is expected across platforms."""
    _local_venv(lock_source.project_dir, ["click==8.1.7"])
    fake_remote(sentinel="", dists=["click==9.0.0"])
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.provisioned and resolved.path


def test_no_local_venv_means_no_comparison(fake_remote, lock_source):
    """Often the very reason someone is provisioning remotely: the laptop
    cannot build the environment at all."""
    fake_remote(sentinel="", dists=["click==8.1.7"])
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert not any("differs from" in note for note in resolved.notes)


def test_a_remote_that_cannot_describe_itself_says_so(fake_remote, lock_source):
    """An honest null. The environment is still published -- it was built."""
    fake_remote(sentinel="", dists=None)
    resolved = resolve_remote_venv(RemoteSpec("host", "/p"), "sync", lock_source)
    assert resolved.provisioned
    assert any("did not report a distribution list" in note for note in resolved.notes)
