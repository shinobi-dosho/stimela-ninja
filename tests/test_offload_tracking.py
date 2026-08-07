"""Tests for `shinobi.offload.tracking` -- discovering the runs a
workspace has offloaded, asking each engine what became of them, and
following a running one's remote log.

The engines are mocked throughout: `probe_ssh` is one `_ssh` round trip
and `follow` is one `ssh ... tail` child, so both are exercised by
substituting the subprocess layer rather than by needing a cluster.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from shinobi.exceptions import BackendError
from shinobi.offload import tracking
from shinobi.offload.tracking import FINISHED, RUNNING, UNKNOWN, Launch, RunState, discover, find, probe_all, probe_slurm, probe_ssh

_SSH_HANDLE = {"engine": "ssh", "host": "host", "path": "/remote/path", "pid": "1", "log_file": "l.log", "exit_file": "e.exit"}


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_handle(base, name, handle):
    path = base / ".shinobi" / name / "handle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(handle))
    return path


# --- discovery --------------------------------------------------------


def test_discover_finds_every_handle_in_the_launch_layout(tmp_path):
    _write_handle(tmp_path, "a.recipe", {**_SSH_HANDLE, "launched_at": 100})
    _write_handle(tmp_path, "b.recipe", {**_SSH_HANDLE, "launched_at": 200})
    launches = discover(tmp_path)
    assert [launch.name for launch in launches] == ["a.recipe", "b.recipe"]
    assert all(launch.engine == "ssh" for launch in launches)


def test_discover_orders_oldest_first(tmp_path):
    _write_handle(tmp_path, "newer", {**_SSH_HANDLE, "launched_at": 900})
    _write_handle(tmp_path, "older", {**_SSH_HANDLE, "launched_at": 100})
    assert [launch.name for launch in discover(tmp_path)] == ["older", "newer"]


def test_discover_skips_a_corrupt_handle_rather_than_failing(tmp_path):
    """One launch interrupted mid-write must not take down the listing of
    every other run -- the failure mode that makes a tracking command
    useless exactly when something has gone wrong.
    """
    _write_handle(tmp_path, "good", {**_SSH_HANDLE, "launched_at": 100})
    bad = tmp_path / ".shinobi" / "truncated" / "handle.json"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"engine": "ssh", "host":')
    assert [launch.name for launch in discover(tmp_path)] == ["good"]


def test_discover_skips_a_handle_that_is_not_an_object(tmp_path):
    bad = tmp_path / ".shinobi" / "listy" / "handle.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("[1, 2, 3]")
    assert discover(tmp_path) == []


def test_discover_on_an_empty_workspace_is_empty_not_an_error(tmp_path):
    assert discover(tmp_path) == []


def test_launched_at_prefers_the_recorded_field(tmp_path):
    _write_handle(tmp_path, "r", {**_SSH_HANDLE, "launched_at": 12345, "log_file": "ninja-run-99999.log"})
    assert discover(tmp_path)[0].launched_at == 12345


def test_launched_at_falls_back_to_the_log_filename_stamp(tmp_path):
    """Handles written before `launched_at` existed still carry the launch
    time -- `launch_remote` has always embedded it in the log filename.
    """
    handle = dict(_SSH_HANDLE)
    handle["log_file"] = "ninja-run-1700000000.log"
    _write_handle(tmp_path, "old", handle)
    assert discover(tmp_path)[0].launched_at == 1700000000


def test_launched_at_falls_back_to_the_handle_mtime(tmp_path):
    handle = dict(_SSH_HANDLE)
    handle["log_file"] = "no-timestamp-here.log"
    path = _write_handle(tmp_path, "older", handle)
    assert discover(tmp_path)[0].launched_at == pytest.approx(path.stat().st_mtime)


# --- resolving one launch ---------------------------------------------


def test_find_resolves_by_the_name_runs_lists(tmp_path):
    _write_handle(tmp_path, "my.recipe", _SSH_HANDLE)
    assert find("my.recipe", tmp_path).name == "my.recipe"


def test_find_resolves_by_handle_path(tmp_path):
    """`ninja status` takes a path and `ninja runs` prints a name; each
    command accepting the other's output is the point.
    """
    path = _write_handle(tmp_path, "my.recipe", _SSH_HANDLE)
    assert find(str(path)).name == "my.recipe"


def test_find_raises_lookuperror_for_an_unknown_name(tmp_path):
    _write_handle(tmp_path, "my.recipe", _SSH_HANDLE)
    with pytest.raises(LookupError):
        find("nope", tmp_path)


# --- probing ssh ------------------------------------------------------


def test_probe_ssh_reports_running(monkeypatch):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="RUNNING\n"))
    assert probe_ssh(_SSH_HANDLE) == RunState(RUNNING)


def test_probe_ssh_reports_the_exit_code_and_the_finish_time(monkeypatch):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="137\n1700000000\n"))
    state = probe_ssh(_SSH_HANDLE)
    assert state.state == FINISHED
    assert state.exit_code == 137
    assert state.finished_at == 1700000000
    assert state.failed


def test_probe_ssh_tolerates_a_host_whose_stat_said_nothing(monkeypatch):
    """`stat -c` is GNU and `-f` is BSD; a host with neither still yields a
    perfectly good state, just no end time.
    """
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="0\n"))
    state = probe_ssh(_SSH_HANDLE)
    assert state.state == FINISHED
    assert state.exit_code == 0
    assert state.finished_at is None


def test_probe_ssh_reports_unknown_when_nothing_wrote_an_exit_code(monkeypatch):
    """No exit file and no live process: the run was killed outright, and
    the shell that would have recorded the code died with it.
    """
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="UNKNOWN\n"))
    state = probe_ssh(_SSH_HANDLE)
    assert state.state == UNKNOWN
    assert "no exit file" in state.detail


def test_probe_ssh_raises_when_the_host_is_unreachable(monkeypatch):
    """Not an UNKNOWN state: a cluster you cannot reach and a run that
    vanished are different facts and must not render identically.
    """
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(returncode=255, stderr="no route to host"))
    with pytest.raises(BackendError, match="no route to host"):
        probe_ssh(_SSH_HANDLE)


def test_probe_ssh_asks_for_the_exit_code_and_mtime_in_one_round_trip(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _FakeProc(stdout="0\n123\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    probe_ssh(_SSH_HANDLE)
    assert len(calls) == 1
    command = calls[0][-1]
    assert "/remote/path/e.exit" in command
    assert "stat -c %Y" in command and "stat -f %m" in command


# --- probing slurm ----------------------------------------------------


def test_probe_slurm_is_running_while_any_job_is_pending(monkeypatch):
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"a": "COMPLETED", "b": "PENDING"})
    assert probe_slurm({"jobs": {"a": "1", "b": "2"}}).state == RUNNING


def test_probe_slurm_fails_if_any_job_did(monkeypatch):
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"a": "COMPLETED", "b": "FAILED"})
    state = probe_slurm({"jobs": {"a": "1", "b": "2"}})
    assert state.failed
    assert "b=FAILED" in state.detail


def test_probe_slurm_succeeds_only_when_every_job_completed(monkeypatch):
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"a": "COMPLETED", "b": "COMPLETED"})
    state = probe_slurm({"jobs": {"a": "1", "b": "2"}})
    assert state.state == FINISHED
    assert state.exit_code == 0


def test_probe_slurm_with_no_jobs_is_unknown(monkeypatch):
    assert probe_slurm({"jobs": {}}).state == UNKNOWN


# --- probing many -----------------------------------------------------


def test_probe_all_fills_in_every_state(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="RUNNING\n"))
    _write_handle(tmp_path, "a", _SSH_HANDLE)
    _write_handle(tmp_path, "b", _SSH_HANDLE)
    launches = probe_all(discover(tmp_path))
    assert [launch.state.state for launch in launches] == [RUNNING, RUNNING]


def test_probe_all_turns_one_unreachable_host_into_a_row_not_a_crash(monkeypatch, tmp_path):
    """A single dead host must not empty the table -- the listing exists to
    show you the others.
    """

    def fake_run(args, **kwargs):
        return _FakeProc(returncode=255, stderr="unreachable") if "dead" in args[2] else _FakeProc(stdout="RUNNING\n")

    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", fake_run)
    _write_handle(tmp_path, "alive", {**_SSH_HANDLE, "host": "live", "launched_at": 1})
    _write_handle(tmp_path, "gone", {**_SSH_HANDLE, "host": "dead", "launched_at": 2})
    states = {launch.name: launch.state for launch in probe_all(discover(tmp_path))}
    assert states["alive"].state == RUNNING
    assert states["gone"].state == UNKNOWN
    assert "unreachable" in states["gone"].detail


def test_probe_all_on_nothing_does_nothing(monkeypatch):
    assert probe_all([]) == []


def test_probe_of_an_unknown_engine_says_so(tmp_path):
    launch = Launch(name="x", handle_path=tmp_path / "h.json", engine="argo", handle={"engine": "argo"})
    assert "argo" in tracking.probe(launch).detail


# --- rendering --------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (RunState(RUNNING), "RUNNING"),
        (RunState(FINISHED, exit_code=0), "FINISHED (success)"),
        (RunState(FINISHED, exit_code=3), "FINISHED (exit 3)"),
        (RunState(UNKNOWN), "UNKNOWN"),
    ],
)
def test_describe_matches_what_ninja_status_has_always_printed(state, expected):
    assert state.describe() == expected


def test_launch_host_falls_back_for_slurm(tmp_path):
    launch = Launch(name="x", handle_path=tmp_path / "h.json", engine="slurm", handle={"engine": "slurm", "jobs": {}})
    assert launch.host == "(slurm)"
    assert launch.log_path is None


def test_launch_log_path_joins_the_remote_dir_and_filename(tmp_path):
    launch = Launch(name="x", handle_path=tmp_path / "h.json", engine="ssh", handle=_SSH_HANDLE)
    assert launch.log_path == "/remote/path/l.log"


# --- following --------------------------------------------------------


class _FakePopen:
    """A stand-in for the `ssh ... tail` child. `readline` blocks on a
    queue the way a real pipe blocks on a quiet log, so a test can prove
    the follower stops without waiting for a real timeout.
    """

    def __init__(self, lines, block=False):
        self._lines = list(lines)
        self._block = block
        self._terminated = threading.Event()
        self.stdout = self
        self.stderr = self
        self.returncode = None

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        if not self._block:
            return ""
        # Quiet log: block until something terminates the child, which is
        # exactly what a finished run's `tail -F` does.
        self._terminated.wait(10)
        return ""

    def close(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._terminated.set()

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


def _ssh_launch(tmp_path):
    return Launch(name="x", handle_path=tmp_path / "h.json", engine="ssh", handle=_SSH_HANDLE)


def test_follow_yields_the_remote_log_lines(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *a, **k: _FakePopen(["one\n", "two\n"]))
    assert list(tracking.follow(_ssh_launch(tmp_path), wait=False)) == ["one", "two"]


def test_follow_asks_for_a_tail_of_the_right_file(monkeypatch, tmp_path):
    seen = {}

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        return _FakePopen([])

    monkeypatch.setattr(tracking.subprocess, "Popen", fake_popen)
    list(tracking.follow(_ssh_launch(tmp_path), lines=7, wait=True))
    argv = seen["argv"]
    assert argv[:3] == ["ssh", "--", "host"]
    assert len(argv) == 4  # a single trailing arg, as offload.ssh._ssh does
    assert "tail -n 7" in argv[-1]
    assert "-F" in argv[-1]
    assert "/remote/path/l.log" in argv[-1]


def test_follow_without_wait_does_not_pass_dash_f(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda argv, **k: (seen.__setitem__("argv", argv), _FakePopen([]))[1])
    list(tracking.follow(_ssh_launch(tmp_path), wait=False))
    assert "-F" not in seen["argv"][-1]


def test_setting_stop_ends_a_follow_of_a_quiet_log(monkeypatch, tmp_path):
    """The bug this exists to pin: a finished run's log goes quiet rather
    than closing, so `tail -F` waits forever and a flag checked between
    yields never gets a turn. `stop` must terminate the child.
    """
    proc = _FakePopen(["one\n"], block=True)
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *a, **k: proc)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()

    started = time.monotonic()
    lines = list(tracking.follow(_ssh_launch(tmp_path), wait=True, stop=stop))
    elapsed = time.monotonic() - started

    assert lines == ["one"]
    assert elapsed < 5, "follow did not stop when its stop event was set"
    assert proc.returncode == -15


def test_follow_terminates_the_child_even_when_it_ends_normally(monkeypatch, tmp_path):
    proc = _FakePopen(["one\n"])
    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *a, **k: proc)
    list(tracking.follow(_ssh_launch(tmp_path), wait=False))
    assert proc.returncode == -15


def test_follow_refuses_an_engine_with_no_single_log(tmp_path):
    launch = Launch(name="x", handle_path=tmp_path / "h.json", engine="slurm", handle={"engine": "slurm", "jobs": {}})
    with pytest.raises(BackendError, match="no single log file"):
        list(tracking.follow(launch))
