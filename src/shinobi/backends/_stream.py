"""Shared subprocess-running helper for backends that want live stdout/
stderr echo (`native`, `container`, and `steps.pyfunc`'s own inline
container-subprocess call for pysteps) without changing the
`Backend.run()` contract every caller depends on: a blocking call that
returns a complete `BackendRun(returncode, stdout, stderr)`.

`stream=True` adds a side channel (each line echoed to the terminal, via
`click.echo`, as it arrives, prefixed with a caller-supplied label) on top
of that same contract -- it does not change what's captured or returned.

What *is* capped is how much of that output is held in memory. Radio tools
are not shy: a single wsclean or CASA step emits hundreds of MB of
progress chatter, and a `list[str]` of every line of it is held for the
whole run, then again in the `StepResult`, then again in the recipe-level
aggregate. `LineBuffer` keeps the head and the tail of each stream and
elides the middle -- see its docstring for why the middle is the part you
can afford to lose, and for the one class of line it never drops.

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
import re
import signal
import subprocess
import threading
import time
from collections import deque
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

# Defaults for how much of a stream to keep. Deliberately generous: the
# point is to survive a tool that emits a progress line per millisecond,
# not to be frugal about a tool that emits a few thousand lines total. At
# ~100 bytes a line these bound a stream at roughly 1 MB, against the
# hundreds of MB an uncapped run can reach.
DEFAULT_HEAD_LINES = 5_000
DEFAULT_TAIL_LINES = 5_000

# The limits in force for this process, seeded from the defaults and
# replaced once per CLI invocation by `set_capture_limits`. Module state
# rather than a `run_streaming` argument because the alternative is
# threading two ints through `Backend.run()` -- a protocol five backends
# implement -- to reach a value that is the same for every step of a run.
# Dispatch resolves `AppConfig` once and workers never call `load()`, so
# reading config here instead is not an option either.
_head_limit = DEFAULT_HEAD_LINES
_tail_limit = DEFAULT_TAIL_LINES


def set_capture_limits(head: int, tail: int) -> None:
    """Set how many lines of each stream `run_streaming` retains, from
    `AppConfig.log.capture_head_lines` / `capture_tail_lines`.

    Called by the CLI alongside `logsetup.setup_file_logging`. A library
    caller that never calls it gets the module defaults.
    """
    global _head_limit, _tail_limit
    _head_limit = max(0, head)
    _tail_limit = max(0, tail)


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


def elision_marker(count: int) -> str:
    """The single line standing in for `count` dropped lines.

    Deliberately unlike any tool's own output, and deliberately *not*
    silent: a truncated log that doesn't say it was truncated sends the
    next reader hunting for a step that looks like it stopped mid-run.
    """
    return f"... [shinobi] {count} line{'s' if count != 1 else ''} elided ..."


class LineBuffer:
    """A bounded, order-preserving record of one stream's lines.

    Keeps the first `head_max` lines and the last `tail_max`, which between
    them hold what output is actually read for: a tool's banner, its
    resolved parameters and its early failures live at the top; its result,
    its summary and the traceback that killed it live at the bottom. What
    gets dropped is the progress chatter in between -- the major/minor
    cycle counters and per-channel percentages that are only ever read as
    they scroll past, which `stream=True` has already echoed live.

    `keep_matching` is the exception that keeps truncation from changing a
    run's *results* rather than just its readability. A cab's wranglers
    (`shinobi.wranglers`) pull structured outputs out of console lines by
    regex, and dispatch applies them to the text this buffer returns -- so a
    dropped line that a wrangler would have matched silently costs an
    output value. Lines matching any `keep_matching` pattern are therefore
    retained wherever they occur, in position. Backends pass their cab's
    wrangler patterns; a tool whose wranglers match nearly every line
    degrades to `keep_max` retained matches and elides beyond that, which
    is reported like any other drop.

    Not thread-safe: one buffer belongs to one pump thread.
    """

    def __init__(
        self,
        *,
        head_max: int = DEFAULT_HEAD_LINES,
        tail_max: int = DEFAULT_TAIL_LINES,
        keep_matching: tuple[str, ...] = (),
        keep_max: int | None = None,
    ) -> None:
        self._head_max = max(0, head_max)
        self._tail_max = max(0, tail_max)
        # A wrangler pattern that doesn't compile is the cab author's bug
        # and `apply_wranglers` will raise on it soon enough with a better
        # message than a pump thread can give. Retention is best-effort:
        # skip it here rather than kill the thread reading the process.
        self._keep: list[re.Pattern[str]] = []
        for pattern in keep_matching:
            try:
                self._keep.append(re.compile(pattern))
            except re.error:
                continue
        self._keep_max = self._head_max + self._tail_max if keep_max is None else max(0, keep_max)

        self._head: list[tuple[int, str]] = []
        self._tail: deque[tuple[int, str]] = deque(maxlen=self._tail_max) if self._tail_max else deque(maxlen=1)
        self._matched: list[tuple[int, str]] = []
        self._count = 0
        self._matches_dropped = False

    def append(self, line: str) -> None:
        index = self._count
        self._count += 1
        if index < self._head_max:
            self._head.append((index, line))
            return
        # Checked before the tail, not after: a line the tail will evict
        # later still has to be retained now, because eviction is silent.
        if self._keep and any(pattern.search(line) for pattern in self._keep):
            if len(self._matched) < self._keep_max:
                self._matched.append((index, line))
            else:
                self._matches_dropped = True
        if self._tail_max:
            self._tail.append((index, line))

    @property
    def total(self) -> int:
        """Every line seen, including those dropped."""
        return self._count

    def _retained(self) -> list[tuple[int, str]]:
        seen: dict[int, str] = {}
        for index, line in (*self._head, *self._matched, *(self._tail if self._tail_max else ())):
            seen[index] = line
        return sorted(seen.items())

    @property
    def dropped(self) -> int:
        """How many lines this buffer saw but will not return."""
        return self._count - len(self._retained())

    @property
    def matches_dropped(self) -> bool:
        """True when `keep_max` was hit, i.e. a line matching a caller's
        keep pattern was dropped anyway. The one drop that can change a
        run's outputs rather than just its readability.
        """
        return self._matches_dropped

    def text(self) -> str:
        """The retained lines, in order, with each run of dropped lines
        replaced by a single `elision_marker`.

        Line terminators are preserved as they arrived, so an uncapped
        stream returns byte-for-byte what it was given.
        """
        retained = self._retained()
        if not retained:
            return "" if not self._count else elision_marker(self._count) + "\n"
        out: list[str] = []
        expected = 0
        for index, line in retained:
            if index > expected:
                out.append(elision_marker(index - expected) + "\n")
            out.append(line)
            expected = index + 1
        if self._count > expected:
            out.append(elision_marker(self._count - expected) + "\n")
        return "".join(out)


def _pump(stream, sink: LineBuffer, *, label: str, err: bool, echo: bool) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        if echo:
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


def run_streaming(
    argv: list[str],
    *,
    label: str,
    stream: bool,
    stop: RuntimeStop | None = None,
    keep_matching: tuple[str, ...] = (),
    head_lines: int | None = None,
    tail_lines: int | None = None,
    **popen_kwargs: Any,
) -> BackendRun:
    """Run `argv`, returning a `BackendRun` whose stdout/stderr are the
    captured output, capped at `head_lines` + `tail_lines` per stream.

    `stream=True` additionally echoes each line (prefixed `"[{label}] "`) to
    the terminal as it arrives. That echo is a side channel, not a buffer: a
    chatty tool still scrolls past in full whatever the caps say.

    Both modes run through `subprocess.Popen` with one reader thread per
    stream. `stream=False` used to be a plain
    `subprocess.run(capture_output=True)`, which holds the whole of stdout in
    memory with no ceiling -- the very thing being capped here, so it could
    not stay.

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
        keep_matching: The cab's wrangler patterns. Lines matching them are
            never elided (see `LineBuffer`). Pass `()` where output is not
            wrangled.
        head_lines: Lines kept from the start of each stream. `None` takes
            the process-wide limit set by `set_capture_limits`.
        tail_lines: Lines kept from the end of each stream, likewise.
        **popen_kwargs: Forwarded to `subprocess.Popen` (`cwd`, `env`, ...).

    Returns:
        The completed `BackendRun`.

    Raises:
        TeardownIncomplete: If the run was interrupted and the child could
            not be confirmed stopped. Replaces the original interrupt (kept
            as `__cause__`) so that no caller unwinding past this point
            treats the workspace as safe to clean up.
    """
    head = _head_limit if head_lines is None else head_lines
    tail = _tail_limit if tail_lines is None else tail_lines
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

    def _buffer() -> LineBuffer:
        return LineBuffer(head_max=head, tail_max=tail, keep_matching=keep_matching)

    out_buf, err_buf = _buffer(), _buffer()
    try:
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, out_buf), kwargs={"label": label, "err": False, "echo": stream}),
            threading.Thread(target=_pump, args=(proc.stderr, err_buf), kwargs={"label": label, "err": True, "echo": stream}),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        returncode = proc.wait()
        return BackendRun(
            returncode=returncode,
            stdout=out_buf.text(),
            stderr=err_buf.text(),
            stdout_dropped=out_buf.dropped,
            stderr_dropped=err_buf.dropped,
            wrangler_lines_dropped=out_buf.matches_dropped or err_buf.matches_dropped,
        )
    except BaseException as exc:
        # BaseException, not Exception: KeyboardInterrupt is the case this
        # exists for, and it is not an Exception.
        if not _teardown(child):
            raise TeardownIncomplete(f"'{label}' could not be stopped (pid {proc.pid}); leaving its workspace untouched. See the log above.") from exc
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(proc.pid, None)
