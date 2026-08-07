"""Tests for `ninja runs` and `ninja logs` -- the plural half of run
tracking, alongside the single-handle `ninja status`.

The engine is mocked at the subprocess layer, as in
`test_offload_tracking`; what is under test here is what the commands
put on the terminal and what they do with a run that has already
finished.
"""

from __future__ import annotations

import json
import threading

from click.testing import CliRunner

from shinobi.cli import main
from shinobi.offload import tracking

_HANDLE = {"engine": "ssh", "host": "cluster", "path": "/scratch/run", "pid": "1", "log_file": "l.log", "exit_file": "e.exit"}


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_handle(base, name, **extra):
    path = base / ".shinobi" / name / "handle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**_HANDLE, **extra}))
    return path


def test_runs_lists_nothing_gracefully(tmp_path):
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no launches under" in result.output


def test_runs_tabulates_name_host_and_state(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="RUNNING\n"))
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path)])
    assert result.exit_code == 0
    assert "my.selfcal" in result.output
    assert "cluster" in result.output
    assert "RUNNING" in result.output
    assert "1 launch, 1 running" in result.output


def test_runs_reports_a_failing_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="137\n1700000100\n"))
    _write_handle(tmp_path, "big.image", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path)])
    assert "FINISHED (137)" in result.output
    # Measured to when it finished (100s), not to now -- otherwise the
    # number grows every time the table is drawn.
    assert "00:01:40" in result.output


def test_runs_json_carries_the_structured_state(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="0\n1700000100\n"))
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["name"] == "my.selfcal"
    assert payload[0]["state"] == "FINISHED"
    assert payload[0]["exit_code"] == 0


def test_runs_no_probe_makes_no_ssh_calls(monkeypatch, tmp_path):
    """The offline listing: useful on a laptop with no route to the
    cluster, and it must genuinely not reach for one.
    """
    calls = []
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: calls.append(a) or _FakeProc())
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path), "--no-probe"])
    assert result.exit_code == 0
    assert calls == []
    assert "my.selfcal" in result.output


def test_runs_survives_one_unreachable_host(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(returncode=255, stderr="no route"))
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["runs", "--workdir", str(tmp_path)])
    assert result.exit_code == 0
    assert "UNKNOWN" in result.output


def test_logs_names_an_unknown_run_helpfully(tmp_path):
    result = CliRunner().invoke(main, ["logs", "nope", "--workdir", str(tmp_path)])
    assert result.exit_code != 0
    assert "no launch named 'nope'" in result.output
    assert "ninja runs" in result.output


def test_logs_prints_a_header_and_the_tail(monkeypatch, tmp_path):
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="0\n1700000100\n"))

    class _Tail:
        stdout = stderr = None

        def __init__(self):
            self._lines = ["first line\n", "second line\n"]
            self.returncode = None
            self.stdout = self
            self.stderr = self

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def close(self):
            pass

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(tracking.subprocess, "Popen", lambda *a, **k: _Tail())
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)
    result = CliRunner().invoke(main, ["logs", "my.selfcal", "--workdir", str(tmp_path)])
    assert result.exit_code == 0
    assert "my.selfcal" in result.output
    assert "cluster" in result.output
    assert "first line" in result.output
    assert "second line" in result.output


def test_logs_follow_of_a_finished_run_does_not_wait(monkeypatch, tmp_path):
    """`--follow` on a run that already ended must print its tail and
    return. A `tail -F` of a file nobody will append to again never ends
    on its own, so this is the difference between a command and a hang.
    """
    monkeypatch.setattr("shinobi.offload.ssh.subprocess.run", lambda *a, **k: _FakeProc(stdout="0\n1700000100\n"))
    seen = {}

    class _Tail:
        def __init__(self):
            self._lines = ["done\n"]
            self.returncode = None
            self.stdout = self
            self.stderr = self

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def close(self):
            pass

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            return self.returncode

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        return _Tail()

    monkeypatch.setattr(tracking.subprocess, "Popen", fake_popen)
    _write_handle(tmp_path, "my.selfcal", launched_at=1_700_000_000)

    finished = threading.Event()

    def _run():
        try:
            return CliRunner().invoke(main, ["logs", "my.selfcal", "--follow", "--workdir", str(tmp_path)])
        finally:
            finished.set()

    result_box = {}
    thread = threading.Thread(target=lambda: result_box.update(r=_run()), daemon=True)
    thread.start()
    assert finished.wait(20), "`logs --follow` hung on a run that had already finished"

    result = result_box["r"]
    assert result.exit_code == 0
    assert "already finished" in result.output
    # No `-F`: there is nothing left to wait for.
    assert "-F" not in seen["argv"][-1]
