"""Finding, querying and following the runs `ninja` has offloaded.

`ninja status HANDLE` answers one question about one handle you already
know the path of. That is the whole of run tracking today, which is thin
for a tool whose runs last hours: a `--remote` launch prints a handle
path and an `ssh host tail -f ...` line, and from then on keeping track
of what is running where is the operator's problem.

This module is the plural version. It discovers the handles a workspace
has accumulated (`discover`), asks each engine what became of its run
(`probe`), and streams a running one's remote log locally (`follow`).
Presentation lives in the CLI; nothing here imports `rich` or `click`.

The contract `shinobi.offload` established for status holds throughout:
state is reconstructed *fresh* on every call, from the handle plus one
round trip to the engine. No daemon, no local state file that can
disagree with the cluster, nothing to leave running. A `ninja runs` on a
laptop that has been asleep for two days is as accurate as one issued a
second after launch.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from shinobi.exceptions import BackendError

# Slurm states that mean "not finished yet". Everything else `sacct` can
# report is terminal, and anything unrecognised is treated as terminal
# rather than left spinning forever in a table.
_SLURM_PENDING = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING"}
# The timestamp `launch_remote` embeds in `ninja-run-<ts>.log`, for handles
# written before `launched_at` was recorded as a field of its own.
_LOG_TS = re.compile(r"-run-(\d+)\.log\Z")

RUNNING = "RUNNING"
FINISHED = "FINISHED"
UNKNOWN = "UNKNOWN"


@dataclass
class RunState:
    """What an engine says about one offloaded run, right now.

    Attributes:
        state: `RUNNING`, `FINISHED` or `UNKNOWN`.
        exit_code: The run's exit status, when it finished and the engine
            could report one. `None` while running, and for a finished run
            whose exit file never appeared.
        detail: Engine-specific extra, e.g. the per-step Slurm states
            behind an aggregate, or why a probe came back `UNKNOWN`.
        finished_at: Unix time the run ended, where the engine could say.
            Without it a finished run's "elapsed" counts to *now*, which
            grows every time you list it and describes your own patience
            rather than how long the job took.
    """

    state: str
    exit_code: int | None = None
    detail: str = ""
    finished_at: float | None = None

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    @property
    def failed(self) -> bool:
        return self.state == FINISHED and self.exit_code not in (0, None)

    def describe(self) -> str:
        """One-line rendering, matching what `ninja status` has always
        printed for an ssh handle so the existing command's output does
        not shift under anyone parsing it.
        """
        if self.state == RUNNING:
            return RUNNING
        if self.state == FINISHED:
            return "FINISHED (success)" if self.exit_code == 0 else f"FINISHED (exit {self.exit_code})"
        return UNKNOWN


@dataclass
class Launch:
    """A discovered handle, its identity, and the state of the run behind
    it. `state` is None until something probes it -- `discover` is cheap
    and offline by design, so a caller that only wants to list what exists
    never pays for a round trip per launch.
    """

    name: str
    handle_path: Path
    engine: str
    handle: dict[str, Any] = field(default_factory=dict)
    launched_at: float | None = None
    state: RunState | None = None

    @property
    def host(self) -> str:
        """Where the run went. Slurm handles name no host -- the scheduler
        chose the nodes -- so they report the engine instead.
        """
        return self.handle.get("host") or "(slurm)"

    @property
    def log_path(self) -> str | None:
        """Remote path of the combined stdout/stderr log, for engines that
        have one. Slurm's output is per-job and lives wherever the batch
        script put it, so there is no single file to point at.
        """
        path, log_file = self.handle.get("path"), self.handle.get("log_file")
        return f"{path.rstrip('/')}/{log_file}" if path and log_file else None


def discover(base: Path | str | None = None) -> list[Launch]:
    """Every launch handle under `base`, oldest first.

    Handles live at `.shinobi/<name>/handle.json` -- the layout
    `ninja run --remote` and `ninja compile --submit` already write and
    `ninja clean --launches` already removes. `<name>` is the launch's
    identity here, which is why it can be typed instead of a path.

    A handle that will not parse is skipped rather than fatal: one
    truncated file (a launch interrupted mid-write) must not take down the
    listing of every other run.
    """
    root = Path(base) if base is not None else Path.cwd()
    launches: list[Launch] = []
    for path in sorted(root.glob(".shinobi/*/handle.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        launches.append(
            Launch(
                name=path.parent.name,
                handle_path=path,
                engine=str(data.get("engine", UNKNOWN)),
                handle=data,
                launched_at=_launched_at(data, path),
            )
        )
    launches.sort(key=lambda launch: (launch.launched_at or 0.0, launch.name))
    return launches


def _launched_at(handle: dict[str, Any], path: Path) -> float | None:
    """When the run was launched, best-effort, in three descending orders
    of directness: the field `launch_remote` records, the timestamp it
    embeds in the log filename (for handles written before that field
    existed), and failing both the handle file's own mtime -- which is
    written once, at launch, and never touched again.
    """
    stamp = handle.get("launched_at")
    if isinstance(stamp, (int, float)):
        return float(stamp)
    match = _LOG_TS.search(str(handle.get("log_file", "")))
    if match:
        return float(match.group(1))
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def find(name_or_path: str, base: Path | str | None = None) -> Launch:
    """Resolve one launch by name (as `ninja runs` lists it) or by the
    path of its handle file, so the two commands accept each other's
    output without the operator translating between them.
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        launches = [launch for launch in discover(candidate.parent.parent.parent) if launch.handle_path == candidate.resolve() or launch.handle_path == candidate]
        if launches:
            return launches[0]
        data = json.loads(candidate.read_text())
        return Launch(
            name=candidate.parent.name,
            handle_path=candidate,
            engine=str(data.get("engine", UNKNOWN)),
            handle=data,
            launched_at=_launched_at(data, candidate),
        )
    for launch in discover(base):
        if launch.name == name_or_path:
            return launch
    raise LookupError(name_or_path)


def probe(launch: Launch) -> RunState:
    """Ask `launch`'s engine what became of it. One round trip; raises
    `BackendError` if the engine could not be reached at all, since a
    silent `UNKNOWN` for an unreachable cluster is indistinguishable from
    a run that genuinely vanished.
    """
    if launch.engine == "ssh":
        return probe_ssh(launch.handle)
    if launch.engine == "slurm":
        return probe_slurm(launch.handle)
    return RunState(UNKNOWN, detail=f"unknown engine {launch.engine!r}")


def probe_all(launches: list[Launch], *, max_workers: int = 8) -> list[Launch]:
    """Probe every launch concurrently, filling in `state` in place.

    Concurrent because each probe is an ssh round trip that is almost
    entirely latency: a serial sweep of a dozen launches against a
    cluster on the other side of an ocean takes long enough that the
    listing feels broken. A probe that raises becomes an `UNKNOWN` row
    carrying the reason -- one unreachable host must not empty the table.
    """
    if not launches:
        return launches

    def _one(launch: Launch) -> None:
        try:
            launch.state = probe(launch)
        except (BackendError, OSError) as exc:
            launch.state = RunState(UNKNOWN, detail=str(exc))

    with ThreadPoolExecutor(max_workers=min(max_workers, len(launches))) as pool:
        list(pool.map(_one, launches))
    return launches


def probe_ssh(handle: dict[str, Any]) -> RunState:
    """Structured form of `offload.ssh.status_ssh`: same single round trip
    and the same shell test, returning the state rather than a sentence
    about it, so a table can align on it and a follower can wait for it.
    """
    from shinobi.offload.ssh import _ssh

    host, path, pid = handle["host"], handle["path"], handle["pid"]
    exit_path = f"{path.rstrip('/')}/{handle['exit_file']}"
    quoted = shlex.quote(exit_path)
    # The exit file's mtime is when the run ended -- it is written once, by
    # the shell wrapper, immediately after the recipe returns. Asked for in
    # the same round trip as the exit code, and tolerant of failure: `-c` is
    # GNU stat and `-f` is BSD's, and a host with neither still gets a
    # perfectly good state, just no end time.
    check = f"if [ -f {quoted} ]; then cat {quoted}; stat -c %Y {quoted} 2>/dev/null || stat -f %m {quoted} 2>/dev/null; else kill -0 {shlex.quote(pid)} 2>/dev/null && echo RUNNING || echo UNKNOWN; fi"
    proc = _ssh(host, check)
    if proc.returncode != 0:
        raise BackendError(f"could not query status on {host}: {proc.stderr.strip()}")
    fields = proc.stdout.split()
    result = fields[0] if fields else ""
    if result == RUNNING:
        return RunState(RUNNING)
    if result.isdigit():
        stamp = fields[1] if len(fields) > 1 and fields[1].isdigit() else None
        return RunState(FINISHED, exit_code=int(result), finished_at=float(stamp) if stamp else None)
    # No exit file and no live process. The usual cause is a run killed
    # outright -- OOM, node reboot, `scancel` -- which writes no exit code
    # because the shell that would have written it died too.
    return RunState(UNKNOWN, detail="no exit file and no live process")


def probe_slurm(handle: dict[str, Any]) -> RunState:
    """Aggregate a compiled DAG's per-job Slurm states into one row.

    A submitted recipe is many jobs; a listing has one line. Pending or
    running anywhere means the workflow is still going, any failure means
    it failed (an `afterok` chain will not proceed past one anyway), and
    only all-terminal-and-clean is success.
    """
    from shinobi.offload.slurm import status_slurm

    jobs = handle.get("jobs") or {}
    if not jobs:
        return RunState(UNKNOWN, detail="handle lists no jobs")
    states = status_slurm(jobs)
    detail = ", ".join(f"{name}={state}" for name, state in states.items())
    if any(state.split()[0] in _SLURM_PENDING for state in states.values()):
        return RunState(RUNNING, detail=detail)
    if any(state.split()[0] != "COMPLETED" for state in states.values()):
        return RunState(FINISHED, exit_code=1, detail=detail)
    return RunState(FINISHED, exit_code=0, detail=detail)


def follow(launch: Launch, *, lines: int = 40, wait: bool = True, stop: threading.Event | None = None) -> Iterator[str]:
    """Yield the remote log of `launch`, line by line, as it is written.

    Starts `lines` back from the end so a follower joining a run in
    progress sees context rather than an empty screen until the tool next
    speaks. `wait=False` prints that tail and returns, which is the right
    default for a run that has already finished.

    The `tail -F` (not `-f`) is deliberate: a log that is rotated or
    replaced under a long run is exactly the case a plain `-f` silently
    keeps reading the wrong inode for.

    `stop` is how a caller ends a follow that would otherwise never end. A
    finished run's log does not close its stream, it just goes quiet, and
    `tail -F` will wait on it forever; meanwhile this generator is blocked
    inside `readline`, where checking a flag between yields never gets a
    turn. Setting the event therefore *terminates the ssh child*, which
    ends the read with EOF and the iteration with it. Nothing short of
    that interrupts a blocking read on a pipe nobody is writing to.
    """
    log_path = launch.log_path
    if log_path is None:
        raise BackendError(f"{launch.name}: this launch has no single log file to follow (engine {launch.engine!r})")
    host = launch.handle["host"]
    remote_cmd = f"tail -n {int(lines)} {'-F ' if wait else ''}{shlex.quote(log_path)}"
    # `--` and the single pre-joined trailing argument for the same reasons
    # `offload.ssh._ssh` uses them: ssh concatenates every trailing element
    # with a space before the remote shell sees it, and a `--` keeps an
    # option-shaped host from being read as an option. Not routed through
    # `_ssh` itself because that one blocks on `subprocess.run`, and the
    # whole point here is to read the pipe while it fills.
    argv = ["ssh", "--", host, f"bash -lc {shlex.quote(remote_cmd)}"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _end() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    closer: threading.Thread | None = None
    if stop is not None:
        closer = threading.Thread(target=lambda: (stop.wait(), _end()), daemon=True)
        closer.start()

    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
    finally:
        # Release the closer thread whether we got here by exhaustion, by
        # the caller closing the generator, or by an exception -- it is
        # parked on an event that may never be set otherwise.
        if stop is not None:
            stop.set()
        _end()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if closer is not None:
            closer.join(timeout=6)
