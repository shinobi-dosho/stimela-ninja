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
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections import deque
from typing import Any

import click

from shinobi.results import BackendRun

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


def run_streaming(
    argv: list[str],
    *,
    label: str,
    stream: bool,
    keep_matching: tuple[str, ...] = (),
    head_lines: int | None = None,
    tail_lines: int | None = None,
    **popen_kwargs: Any,
) -> BackendRun:
    """Run `argv`, returning a `BackendRun` whose stdout/stderr are the
    captured output, capped at `head_lines` + `tail_lines` per stream.

    `stream=True` additionally echoes each line (prefixed `"[{label}] "`)
    to the terminal as it arrives.

    Both modes run via `subprocess.Popen` with one reader thread per
    stream. `stream=False` used to be a plain
    `subprocess.run(capture_output=True)`, which holds the whole of stdout
    in memory with no ceiling -- the very thing being capped here, so it
    could not stay.

    `keep_matching` is the cab's wrangler patterns: lines matching them are
    never elided (see `LineBuffer`). Pass `()` where output is not
    wrangled.
    """
    head = _head_limit if head_lines is None else head_lines
    tail = _tail_limit if tail_lines is None else tail_lines
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs)

    def _buffer() -> LineBuffer:
        return LineBuffer(head_max=head, tail_max=tail, keep_matching=keep_matching)

    out_buf, err_buf = _buffer(), _buffer()
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
