"""Shared subprocess-running helper for backends that want live stdout/
stderr echo (`native`, `container`, and `steps.pyfunc`'s own inline
container-subprocess call for pysteps) without changing the
`Backend.run()` contract every caller depends on: a blocking call that
returns a complete `BackendRun(returncode, stdout, stderr)`.

`stream=True` adds a side channel (each line echoed to the terminal, via
`click.echo`, as it arrives, prefixed with a caller-supplied label) on top
of that same contract -- it does not change what's captured or returned.

`display_label` is the label callers should derive from a step's dotted
cache path -- see its own docstring.

This module is also where a step's work is torn down, because it is where the
*locally-executing* backends launch their processes: native, venv, container
and pystep all arrive here. `run_streaming` cannot return or raise while its
child is still alive, and `terminate_all` reaches every child of a run at
once -- which is what a Ctrl-C during a parallel recipe needs, since the
KeyboardInterrupt lands only in the main thread while the workers sit
blocked on their own children. See `_teardown` for why signalling a process
group is necessary but, for docker and podman, nowhere near sufficient.

**Not** the slurm or kubernetes backends: those hand work to a scheduler and
poll it (`sbatch`/`sacct`, `kubectl`), so what needs stopping is a job on a
cluster, not a child of this process. Nothing here registers or cancels one,
and an interrupt leaves such a job running -- as it did before any of this
existed. Fixing that means `scancel`/`kubectl delete job` on the way out, and
is deliberately not attempted here.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import click

from shinobi.results import BackendRun

logger = logging.getLogger(__name__)

# How long a child gets to honour SIGTERM before SIGKILL, and how long we
# then wait for the kernel to finish with it. Module-level so tests can
# shorten them; a tool mid-write deserves the first, and the second is
# generous only because an uninterruptible (`D` state) process cannot die
# until its I/O completes no matter what we send.
TERM_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 5.0


def display_label(cache_path: str) -> str:
    """A step's dotted cache path as shown to a human, i.e. without its root
    scope name.

    The cache path is rooted at the top-level scope (`myrecipe.stepA.stepB`)
    because that is what keys the cache manifest, and it must stay that way.
    But the root segment is the *same on every line of a run*, and this
    prefix is repeated once per line of forwarded tool output -- so in a log
    it is pure column width. A 36-character root (caracal generates
    `caracal_pipeline_<sha256[:16]>` to keep two pipeline configs from
    colliding in one cache dir) pushed real content off the right of an
    80-column terminal entirely.

    A path with no dot is a step run outside any recipe: there is no root to
    drop, and it is returned unchanged.
    """
    root, dot, rest = cache_path.partition(".")
    return rest if dot else cache_path


@dataclass
class RuntimeStop:
    """How to stop work ninja cannot signal, and how to tell whether it
    worked.

    Both halves are required, and the second is not decoration. `docker rm
    -f` on a container that does not exist *yet* exits 0 -- so a teardown
    racing container creation gets a success it has not earned, the daemon
    goes on to start the container, and `--rm` being daemon-side means
    killing the client leaves it running. Measured: with a cold daemon that
    window is wide enough to leak a real container. `running()` is what turns
    "we asked" into "it is gone", and it is why `stop()` is retried rather
    than called once.
    """

    stop: Callable[[], None]
    running: Callable[[], bool]


@dataclass
class _Child:
    """A live subprocess and everything needed to make it stop."""

    proc: subprocess.Popen
    label: str
    # For a container that is not our descendant (docker/podman); None where
    # signalling the process group is enough (apptainer-likes, native, venv).
    runtime_stop: RuntimeStop | None


_LIVE: dict[int, _Child] = {}
_LIVE_LOCK = threading.Lock()
# Set the moment teardown is *asked for*, not when it finishes: a second
# Ctrl-C arrives while the first teardown is still working through its grace
# periods, and that is precisely when the user wants the escalation. Setting
# it at the end (as this first did) meant the flag could only ever be read by
# a teardown that no longer needed it.
_IMPATIENT = threading.Event()
# How many times teardown has been asked for. Only the transition past the
# first matters (see `terminate_all`); it is reset with `_IMPATIENT` when a
# fresh burst of work starts, so that in a long-lived process -- a notebook, a
# library caller -- one interrupted run does not permanently strip the
# SIGTERM grace from every run after it.
_TERMINATE_CALLS = 0


def _group_alive(pgid: int | None) -> bool:
    """Is anything still in this process group?

    Signal 0 is the standard existence probe. `PermissionError` means the
    group is there but no longer ours to signal, which for our purpose is
    still "alive": we have not established that the work stopped.
    """
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int | None, proc: subprocess.Popen, sig: int) -> None:
    """Send `sig` to the child's whole process group, falling back to the
    child alone where there is no group to signal (non-POSIX, or a child
    that already reparented out of ours)."""
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.send_signal(sig)
    except (ProcessLookupError, OSError):
        pass


def _stopped(child: _Child) -> bool:
    """Has everything this child stands for actually stopped?

    Three questions, not one. The direct child must be reaped; its process
    group must be empty (reaping what we launched says nothing about what it
    left behind); and, where the work lives in a container we cannot signal,
    the runtime must agree the container is gone. Dropping the third is how a
    teardown that raced container creation reports success over a running
    container.
    """
    if child.proc.poll() is None or _group_alive(_pgid_of(child.proc)):
        return False
    if child.runtime_stop is not None:
        try:
            return not child.runtime_stop.running()
        except Exception:  # cannot ask -> cannot claim it stopped
            return False
    return True


def _wait_stopped(child: _Child, seconds: float) -> bool:
    """Poll `_stopped` until it holds or `seconds` elapse.

    Never lets `KeyboardInterrupt` out: a second Ctrl-C landing in this sleep
    used to unwind the whole teardown loop, abandoning every child not yet
    reached. It now means "stop waiting", and the caller escalates.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _stopped(child):
            return True
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            _IMPATIENT.set()
            break
    return _stopped(child)


def _pgid_of(proc: subprocess.Popen) -> int | None:
    """The child's process group, or `None` when there is none we may signal.

    Returns `None` for our *own* group as well: a child that failed to get
    its own session shares ours, and `killpg` there would take ninja -- and
    the user's whole shell job -- down with the step.
    """
    if proc.returncode is not None:
        # Already reaped. Its pid may since have been recycled onto an
        # unrelated process, and signalling that would be an unforced error.
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError, AttributeError):
        return None
    try:
        return None if pgid == os.getpgid(0) else pgid
    except (OSError, AttributeError):  # non-POSIX: no groups to compare
        return None


def _try_runtime_stop(child: _Child) -> bool:
    """Ask the runtime to stop the container. Returns whether the ask itself
    succeeded -- not whether the container is gone, which only
    `RuntimeStop.running` can answer."""
    if child.runtime_stop is None:
        return True
    try:
        child.runtime_stop.stop()
    except Exception as exc:
        logger.warning("teardown: stopping %s's container failed: %s", child.label, exc)
        return False
    return True


def _teardown(child: _Child) -> bool:
    """Stop a child and everything it started, and do not come back until
    that is a fact -- or report that it is beyond us.

    The order is deliberate and was arrived at empirically, per runtime:

    * **docker and podman**: the container is not a descendant of the client
      at all -- containerd-shim (root-owned, own session) or conmon supervises
      it. Signalling the client, its group, or anything else we can reach does
      **nothing**; the container keeps running. `RuntimeStop` (`<engine> rm
      -f`) is the only thing that stops it, which is why it goes first -- and
      why it is *repeated* at every rung rather than called once. `rm -f` on a
      container the daemon has not created yet exits 0, and a teardown that
      believed that exit code would leave the container running behind it.
    * **apptainer and singularity**: the runtime parent and the payload are in
      our process group, and killing the group does tear the container down --
      including a payload that ignores SIGINT and SIGTERM, which SIGINT alone
      demonstrably does not (a casacore call sits in C for minutes without
      Python ever running its handler, and the runtime parent then reparents
      to init and outlives the run).
    * **native and venv**: the plain case, where the group is the tool and
      whatever it forked.

    Returns:
        Whether the work is confirmed stopped. `False` is not advisory: the
        caller's workspace cleanup runs next, and deleting files out from
        under a process still writing them is the failure this whole path
        exists to prevent -- so a `False` here must suppress that cleanup.
    """
    _try_runtime_stop(child)
    if _wait_stopped(child, 0.5):
        return True

    for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (getattr(signal, "SIGKILL", signal.SIGTERM), KILL_GRACE_SECONDS)):
        if sig != signal.SIGTERM:
            logger.info("teardown: %s did not stop on SIGTERM -- killing", child.label)
        # Re-ask the runtime each time round: the container may only have come
        # into existence since the last attempt (see the docstring).
        _try_runtime_stop(child)
        _signal_group(_pgid_of(child.proc), child.proc, sig)
        if _wait_stopped(child, 0.0 if _IMPATIENT.is_set() else grace):
            return True

    logger.error(
        "teardown: %s (pid %d) did NOT stop -- most likely blocked in uninterruptible I/O, or a container "
        "runtime that will not release it. Its workspace is being left as-is: deleting files a live process "
        "is still writing does not stop the writer, it just makes the result silently incomplete. "
        "Check `ps -p %d` (and your container runtime) before re-running.",
        child.label,
        child.proc.pid,
        child.proc.pid,
    )
    return False


def terminate_all(*, reason: str = "interrupted") -> int:
    """Tear down every child currently running under this process.

    The entry point for an interrupt: a `KeyboardInterrupt` is delivered to
    the main thread only, but a recipe's steps run on worker threads that are
    each blocked on a child of their own and will never see it. Without this,
    Ctrl-C during a parallel recipe stops the scheduler and leaves every
    in-flight container running.

    Signals go to every child *first*, and only then does anyone wait. Waiting
    out one child's full ladder before touching the next made Ctrl-C take
    `n * (TERM_GRACE + KILL_GRACE)` to respond across a wide recipe -- long
    enough that the user reasonably hits Ctrl-C again, which is the case that
    used to abandon the remainder.

    Returns:
        How many children this was asked to stop (not how many complied --
        see `_teardown`'s return, which is logged per child).
    """
    global _TERMINATE_CALLS
    with _LIVE_LOCK:
        children = list(_LIVE.values())
        # A second request means the user asked twice: skip the grace periods
        # from here on. Counted under the lock and *before* any waiting, so it
        # is set by the time the first call's ladder reads it.
        _TERMINATE_CALLS += 1
        if _TERMINATE_CALLS > 1:
            _IMPATIENT.set()
    if not children:
        return 0
    logger.warning("%s -- stopping %d running step(s)", reason, len(children))
    for child in children:
        _try_runtime_stop(child)
        _signal_group(_pgid_of(child.proc), child.proc, signal.SIGTERM)
    for child in children:
        _teardown(child)
    return len(children)


def install_signal_handlers() -> None:
    """Make SIGTERM and SIGHUP tear the run down, the way Ctrl-C does.

    Necessary *because* children now get their own session. Before that they
    shared ninja's process group -- the terminal's foreground group -- so a
    dropped ssh connection (SIGHUP) or a plain `kill <ninja-pid>` reached the
    tool as a side effect of reaching ninja. Isolating the child took that
    away: without these handlers ninja would die instantly, unwind nothing,
    and leave the container running, which is worse than what it replaced.

    SIGINT is deliberately not handled here: Python's default already raises
    `KeyboardInterrupt`, which the run paths catch and turn into teardown.
    Only the main thread may install handlers, so this is a no-op elsewhere
    (a library caller on a worker thread keeps whatever its host installed).
    """
    if threading.current_thread() is not threading.main_thread():
        return

    def _handler(signum, _frame):
        name = signal.Signals(signum).name
        terminate_all(reason=f"received {name}")
        # Re-raise as the interrupt the run paths already understand, so the
        # rest of the unwind (workspace cleanup, reservations, the snapshot
        # guard) is the same one Ctrl-C gets.
        raise KeyboardInterrupt(f"{name} received")

    for signame in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, signame, None)
        if sig is None:  # non-POSIX
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not the main thread, or not permitted
            pass


def _pump(stream, sink: list[str], *, label: str, err: bool) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        click.echo(f"[{label}] {line.rstrip()}", err=err)
    stream.close()


class TeardownIncomplete(BaseException):
    """Raised out of `run_streaming` when an interrupted child could not be
    confirmed stopped.

    A `BaseException`, deliberately: it must not be swallowed by the
    `except Exception` handlers that turn a step's failure into a tidy error
    message, because the thing it reports is that *the tool is still running*
    and nothing downstream may tidy up around it. It carries the interrupt it
    replaces as `__cause__`.
    """


def run_streaming(argv: list[str], *, label: str, stream: bool, stop: RuntimeStop | None = None, **popen_kwargs: Any) -> BackendRun:
    """Run `argv`, returning a `BackendRun` with the complete captured
    stdout/stderr either way.

    `stream=False`: captures both streams and returns them whole, as
    `subprocess.run(argv, capture_output=True, text=True)` does.

    `stream=True`: one reader thread per stream echoes each line (prefixed
    `"[{label}] "`) to the terminal as it arrives while also accumulating it,
    so the returned `BackendRun` is byte-for-byte the same text a
    non-streaming run would have captured.

    Either way the child gets **its own session** (`start_new_session`), and
    is torn down before this function returns or propagates an exception. Both
    halves are load-bearing:

    * Its own session means the terminal's Ctrl-C no longer goes to the tool
      and to ninja simultaneously. That race is not theoretical -- SIGINT
      reaching a payload that does not handle it leaves the payload running
      and the container runtime waiting on it, while ninja unwinds and starts
      deleting the very directories that payload has open.
    * Tearing down first makes "the child is dead" a fact the caller can rely
      on, which is what lets `steps.pyfunc` keep using a plain
      `TemporaryDirectory` for the runner's I/O directory: by the time the
      `with` block exits, there is nothing left to pull the rug from under.
      Where that fact cannot be established, `TeardownIncomplete` says so
      instead of letting the unwind proceed on a false premise.

    Args:
        argv: The command to run.
        label: Prefix for streamed output lines, and the name used in
            teardown diagnostics.
        stream: Whether to echo output live as it arrives.
        stop: How to stop, and check, work ninja cannot signal. Container
            backends whose containers are not our descendants (docker,
            podman) pass one; nothing else needs to.
        **popen_kwargs: Forwarded to `subprocess.Popen` (`cwd`, `env`, ...).

    Returns:
        The completed `BackendRun`.

    Raises:
        TeardownIncomplete: If the run was interrupted and the child could
            not be confirmed stopped. Replaces the original interrupt (kept
            as `__cause__`) so that no caller unwinding past this point
            treats the workspace as safe to clean up.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # POSIX only; on anything else the child stays in our group and
        # teardown falls back to signalling the process itself.
        start_new_session=os.name == "posix",
        **popen_kwargs,
    )
    child = _Child(proc=proc, label=label, runtime_stop=stop)
    global _TERMINATE_CALLS
    with _LIVE_LOCK:
        if not _LIVE:
            # A fresh burst of work: whatever happened to the last one, this
            # run starts entitled to the full grace period again.
            _TERMINATE_CALLS = 0
            _IMPATIENT.clear()
        _LIVE[proc.pid] = child

    try:
        if not stream:
            stdout, stderr = proc.communicate()
            return BackendRun(returncode=proc.returncode, stdout=stdout, stderr=stderr)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, stdout_lines), kwargs={"label": label, "err": False}),
            threading.Thread(target=_pump, args=(proc.stderr, stderr_lines), kwargs={"label": label, "err": True}),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        returncode = proc.wait()
        return BackendRun(returncode=returncode, stdout="".join(stdout_lines), stderr="".join(stderr_lines))
    except BaseException as exc:
        # BaseException, not Exception: KeyboardInterrupt is the case this
        # exists for, and it is not an Exception.
        if not _teardown(child):
            raise TeardownIncomplete(f"'{label}' could not be stopped (pid {proc.pid}); leaving its workspace untouched. See the log above.") from exc
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(proc.pid, None)
