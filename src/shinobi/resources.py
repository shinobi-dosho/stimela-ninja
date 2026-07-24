"""Declared resource footprints, the machine's real budget, and the
admission control that keeps the two consistent.

`_run_recipe` (`steps/dispatch.py`) schedules by counting slots: a step is
"running" or "not running", and `max_workers` bounds how many run at once.
That is fine when steps are cheap and interchangeable. It is not fine for
radio-astronomy pipelines, where a single `wsclean`/DDFacet/QuartiCal step
is routinely sized to use most of one machine on its own -- five of those
running concurrently because five slots were free will oversubscribe the
CPU and blow through a memory cap, and the step that gets OOM-killed takes
its innocent siblings' shared memory down with it.

So a `Scope` may declare a `Resources` footprint, and the scheduler admits
work against a `Budget` instead of a slot count. Three properties are worth
stating up front, because they are what make this honest rather than
decorative:

**An undeclared step is free.** Nothing declares a footprint by default, so
a recipe that declares nothing is admitted exactly as before. A declaration
is only ever as good as its coverage -- this models what you tell it, and
tells you nothing you did not.

**The budget comes from the cgroup, not from `/proc/meminfo`.** This is the
whole point. A tool that sizes itself from `/proc/meminfo` sees the host's
memory, not the slice quota it is actually confined to, and cannot
self-limit even if it wants to -- DDFacet on a 251.7 GiB-capped user slice
of a much larger box is the motivating case. `detect_budget` walks the
**full cgroup ancestor chain**, because the cap that matters is almost never
on the leaf cgroup a process is in (see its docstring).

**The declaration is not enforcement.** Locally, shinobi decides *whether to
start* a step; it cannot hold a running process to its word. Container and
cluster backends do enforce it (`--memory`/`--cpus`, `#SBATCH --mem`, k8s
`resources.limits`), which is a real difference -- a containerised runaway
dies in its own cgroup instead of eating the shared slice. A native or venv
step has no such backstop at all.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Decimal (SI) and binary (IEC) suffixes. `GB` is 1000^3 and `GiB` is 1024^3,
# per the standards -- deliberately not docker's convention (where `200g` means
# GiB), because everything here converts to bytes immediately and each backend
# emitter re-formats into its own grammar, so there is no reason to inherit any
# one tool's ambiguity. Write `GiB` when you mean 1024^3.
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "pb": 1000**5,
    "k": 1000,
    "m": 1000**2,
    "g": 1000**3,
    "t": 1000**4,
    "p": 1000**5,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "pib": 1024**5,
}

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")

# cgroup v1 writes a near-`INT64_MAX` sentinel rather than a word when a
# controller is unlimited; anything this large is not a real quota.
_V1_UNLIMITED = 1 << 62


def parse_size(value: float | str) -> int:
    """Parse a byte size: a bare number, or a number with a unit suffix.

    Args:
        value: An int/float count of bytes, or a string like `"200GB"`,
            `"250 GiB"`, `"2T"`, `"1024"`.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string is not a number with an optional known
            unit suffix, or the size is negative.
    """
    # bool is an int subclass, so `memory=True` would otherwise parse as one
    # byte. ValueError rather than the TypeError this arguably is: pydantic
    # only converts ValueError/AssertionError from a validator into a clean
    # ValidationError, and a raw TypeError escaping model construction would
    # be a worse experience than a slightly mislabelled exception.
    if isinstance(value, bool):
        raise ValueError(f"invalid size {value!r}: expected a number or a size string, not a bool")  # noqa: TRY004
    if isinstance(value, (int, float)):
        size = int(value)
    else:
        match = _SIZE_RE.match(value)
        if match is None:
            raise ValueError(f"invalid size {value!r} (expected a number with an optional unit, e.g. '200GB', '250GiB', '2T')")
        number, unit = match.groups()
        multiplier = _SIZE_UNITS.get(unit.lower())
        if multiplier is None:
            raise ValueError(f"invalid size unit {unit!r} in {value!r} (known units: {', '.join(sorted(u for u in _SIZE_UNITS if u))})")
        size = int(float(number) * multiplier)
    if size < 0:
        raise ValueError(f"invalid size {value!r}: must not be negative")
    return size


def format_size(size: int) -> str:
    """Render a byte count for humans, in binary units.

    For log lines and error messages only -- every backend wants its own
    grammar (docker takes raw bytes, kubernetes a `Quantity`, Slurm
    megabytes), so this is deliberately not the shared emitter for those.

    Args:
        size: Size in bytes.

    Returns:
        A string like `"251.7GiB"`.
    """
    for unit, scale in (("PiB", 1024**5), ("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f}{unit}"
    return f"{size}B"


class Resources(BaseModel):
    """A declared resource footprint: what one step needs to run.

    Both fields default to `None`, meaning "undeclared" -- which the
    scheduler reads as *free*, not as *unknown-so-assume-the-worst*.
    Assuming the worst would serialise every undeclared recipe, i.e. punish
    everyone who has not opted in; free keeps the feature genuinely additive
    at the cost of being only as complete as your declarations.

    Shaped so a third dimension (`gpus`) can be added later without a schema
    break -- but not added speculatively.
    """

    cpus: float | None = None
    memory: int | None = None

    @field_validator("memory", mode="before")
    @classmethod
    def _parse_memory(cls, value: object) -> int | None:
        """Accept `"200GB"`-style strings for `memory`, storing bytes."""
        return None if value is None else parse_size(value)  # type: ignore[arg-type]

    def is_empty(self) -> bool:
        """True when nothing is declared, i.e. this step costs the scheduler
        nothing and is admitted on slot count alone.
        """
        return self.cpus is None and self.memory is None

    def describe(self) -> str:
        """A short human rendering for log lines (`"cpus=4, memory=200.0GiB"`)."""
        parts = []
        if self.cpus is not None:
            parts.append(f"cpus={self.cpus:g}")
        if self.memory is not None:
            parts.append(f"memory={format_size(self.memory)}")
        return ", ".join(parts) if parts else "unbounded"


class ResourceBudget(BaseModel):
    """Config-side total: how much of the machine a recipe may occupy.

    `"auto"` (the default) detects the real limit, cgroup-aware;
    `"unbounded"` disables that dimension; anything else is an explicit
    value. Note `None` is deliberately *not* the "unbounded" spelling --
    everywhere else in `shinobi.config` `None` means "unset / fall back /
    off", and quietly inverting that here would be a trap.
    """

    cpus: Literal["auto", "unbounded"] | float | int = "auto"
    memory: Literal["auto", "unbounded"] | int | str = "auto"

    def resolve(self, root: str | Path = "/") -> tuple[Resources, str]:
        """Resolve to concrete totals, detecting only if something asks for it.

        Args:
            root: Filesystem root to detect against. A test seam; `"/"` in
                real use.

        Returns:
            `(totals, source)` -- the resolved budget and a short description
            of where each value came from, for the one log line the scheduler
            emits. Whether the number came from the cgroup or from
            `/proc/meminfo` is exactly the thing worth being able to check.
        """
        detected: Resources | None = None
        detected_source = ""

        def detect() -> Resources:
            nonlocal detected, detected_source
            if detected is None:
                detected, detected_source = detect_budget(root)
            return detected

        cpus = None if self.cpus == "unbounded" else (detect().cpus if self.cpus == "auto" else float(self.cpus))
        memory = None if self.memory == "unbounded" else (detect().memory if self.memory == "auto" else parse_size(self.memory))

        sources = []
        sources.append(f"cpus={'detected via ' + detected_source if self.cpus == 'auto' else 'config'}")
        sources.append(f"memory={'detected via ' + detected_source if self.memory == 'auto' else 'config'}")
        return Resources(cpus=cpus, memory=memory), ", ".join(sources)


def _read_first_line(path: Path) -> str | None:
    """Read one line from a `/proc` or `/sys` file, or None if unreadable."""
    try:
        return path.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def _cgroup_paths(root: Path, controller: str) -> list[str] | None:
    """The cgroup-relative path components for `controller`, from
    `/proc/self/cgroup`.

    Args:
        root: Filesystem root (test seam).
        controller: `""` for the unified (v2) hierarchy, otherwise a v1
            controller name such as `"memory"` or `"cpu"`.

    Returns:
        The path split into components (`[]` means the cgroup root), or None
        if this controller has no entry.
    """
    content = None
    try:
        content = (root / "proc/self/cgroup").read_text()
    except OSError:
        return None
    for line in content.splitlines():
        # Format: hierarchy-ID:controller-list:cgroup-path
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, cgroup_path = parts
        names = controllers.split(",") if controllers else []
        if (controller == "" and controllers == "") or (controller and controller in names):
            return [p for p in cgroup_path.split("/") if p]
    return None


def _walk_limits(base: Path, components: list[str], filename: str, parse) -> int | float | None:
    """Read `filename` at every level from the cgroup root down to the leaf,
    returning the tightest (minimum) real limit found.

    **Walking the whole chain is the entire point.** `/proc/self/cgroup`
    names the *leaf*, and the cap that actually kills your job is almost
    never there: a fair-share memory quota is typically set on
    `user.slice/user-<uid>.slice`, which can be half a dozen levels above
    the leaf a login shell ends up in. Reading only the leaf finds `"max"`,
    falls through to `/proc/meminfo`, and reports the host's full memory --
    silently reproducing the very bug this module exists to prevent.

    Args:
        base: The controller's mount point.
        components: Cgroup path components from the root to the leaf.
        filename: The limit file to read at each level.
        parse: Callable turning the file's contents into a limit, or None
            for "no constraint at this level".

    Returns:
        The minimum limit found, or None if no level constrains anything.
    """
    limits = []
    for depth in range(len(components) + 1):
        raw = _read_first_line(base.joinpath(*components[:depth]) / filename)
        if raw is None:
            continue
        try:
            limit = parse(raw)
        except (ValueError, ZeroDivisionError):
            continue
        if limit is not None:
            limits.append(limit)
    return min(limits) if limits else None


def _parse_v2_memory(raw: str) -> int | None:
    """cgroup v2 `memory.max`: a byte count, or the word `max`."""
    return None if raw == "max" else int(raw)


def _parse_v2_cpu(raw: str) -> float | None:
    """cgroup v2 `cpu.max`: `"<quota> <period>"`, quota `max` when unset."""
    quota, _, period = raw.partition(" ")
    return None if quota == "max" else int(quota) / int(period or 100000)


def _parse_v1_memory(raw: str) -> int | None:
    """cgroup v1 `memory.limit_in_bytes`: a huge sentinel when unlimited."""
    value = int(raw)
    return None if value >= _V1_UNLIMITED else value


def detect_budget(root: str | Path = "/") -> tuple[Resources, str]:
    """Detect how much CPU and memory this process may actually use.

    Tried in order, most authoritative first: cgroup v2, cgroup v1, then the
    host's own totals. The cgroup tiers walk the full ancestor chain (see
    `_walk_limits`) and take the tightest limit at any level.

    Args:
        root: Filesystem root to detect against. A test seam; `"/"` in real
            use.

    Returns:
        `(budget, source)`, where an unset field means "nothing constrains
        this dimension". `source` names the tier that answered, so the
        scheduler's log line can say whether the number is a real cgroup
        quota or just the host's size.
    """
    root = Path(root)
    cpus: float | None = None
    memory: int | None = None
    source = "host"

    v2_base = root / "sys/fs/cgroup"
    if (v2_base / "cgroup.controllers").exists():
        components = _cgroup_paths(root, "")
        if components is not None:
            memory = _walk_limits(v2_base, components, "memory.max", _parse_v2_memory)
            cpus = _walk_limits(v2_base, components, "cpu.max", _parse_v2_cpu)
            if memory is not None or cpus is not None:
                source = "cgroup v2"
    if memory is None and cpus is None:
        mem_components = _cgroup_paths(root, "memory")
        if mem_components is not None:
            memory = _walk_limits(root / "sys/fs/cgroup/memory", mem_components, "memory.limit_in_bytes", _parse_v1_memory)
        cpu_components = _cgroup_paths(root, "cpu")
        if cpu_components is not None:
            cpu_base = root / "sys/fs/cgroup/cpu"
            quota = _walk_limits(cpu_base, cpu_components, "cpu.cfs_quota_us", lambda raw: None if int(raw) <= 0 else int(raw))
            period = _read_first_line(cpu_base.joinpath(*cpu_components) / "cpu.cfs_period_us")
            if quota is not None and period:
                cpus = quota / int(period)
        if memory is not None or cpus is not None:
            source = "cgroup v1"

    # Host fallbacks, per dimension: a cgroup may cap memory but not CPU.
    if memory is None:
        meminfo = None
        try:
            meminfo = (root / "proc/meminfo").read_text()
        except OSError:
            pass
        if meminfo:
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    memory = int(line.split()[1]) * 1024  # MemTotal is in kB
                    break
    if cpus is None:
        # sched_getaffinity, not cpu_count: it respects taskset/cpuset pinning.
        try:
            cpus = float(len(os.sched_getaffinity(0)))
        except AttributeError:  # pragma: no cover -- non-Linux
            cpus = float(os.cpu_count() or 1)

    return Resources(cpus=cpus, memory=memory), source


class Budget:
    """Admission control over a shared pool of CPU and memory.

    One `Budget` is created per top-level run and **shared by every nested
    recipe's scheduler** (threaded down like `AppConfig`). That sharing is
    the point: caracal-shaped pipelines nest each worker as its own `Recipe`,
    so a budget scoped to one `_run_recipe` invocation would leave every
    branch free to independently decide it owns the machine -- exactly the
    failure this is meant to stop.

    Not a pydantic model: this is live mutable state guarded by one
    `threading.Condition`. Every method below takes that lock; nothing reads
    the totals outside it.

    ## The wake protocol

    A scheduler that is refused must park until something changes, but it
    cannot park while holding the lock inside `try_acquire` (it still has
    futures to reap). So refusal and parking are two separate lock
    acquisitions -- and a `release` landing in the window between them would
    be a lost wakeup and a permanent hang. `_generation`, bumped under the
    lock by every `release`, closes that window: `try_acquire` hands back the
    generation it decided at, and `wait_for_change` returns immediately if
    anything has happened since. This is reachable in the ordinary case (one
    recipe is refused at the instant a sibling finishes), not a theoretical
    race.

    ## Fairness

    Admission is strictly FIFO by ticket. Within one recipe, draining in
    declaration order already prevents starvation; across sibling recipes
    sharing this budget it does not -- four branches running a steady stream
    of small steps can hold the reserved total above zero indefinitely, and a
    branch needing 200 of 250 GiB would never be admitted. A refused
    scheduler takes a ticket, and no later arrival is admitted ahead of it,
    so current holders drain and the large step runs. This costs some
    utilisation, and is the same trade "block in declaration order" already
    makes, extended across recipes.

    ## Why this cannot deadlock

    - A reservation is only ever held by a step that is *running*; a parked
      scheduler holds a ticket, which reserves nothing.
    - A step holding a reservation never re-enters `_run_recipe` (the
      scheduler skips demand for nested-`Recipe` steps), so no reservation
      can be waiting on itself.
    - Every running step terminates, and `_run_recipe` releases its
      reservation on every exit path including exceptions.

    Given those, some reservation always eventually releases and bumps the
    generation, and FIFO guarantees the oldest waiter is the one served. A
    demand larger than the whole budget never waits at all -- it is granted
    immediately, since no amount of waiting could ever make it fit.
    """

    def __init__(self, total: Resources) -> None:
        """Create a budget with `total` available.

        Args:
            total: The resolved totals. An unset field means that dimension
                is unconstrained.
        """
        self.total = total
        self._cond = threading.Condition()
        self._cpus_used = 0.0
        self._memory_used = 0
        self._generation = 0
        self._next_ticket = 0
        # Waiting schedulers, by thread. A recipe's scheduler loop runs on
        # one thread for its whole life, so the thread *is* the waiter
        # identity -- no ticket needs plumbing through the call site.
        self._waiters: dict[int, int] = {}

    @property
    def reserved(self) -> Resources:
        """What is currently reserved. For tests and diagnostics."""
        with self._cond:
            return Resources(cpus=self._cpus_used, memory=self._memory_used)

    def _dimensions(self, demand: Resources) -> Iterator[tuple[float, float, float]]:
        """`(reserved, requested, total)` for each dimension that can
        actually refuse this demand.

        A dimension the budget doesn't constrain, or that the step doesn't
        declare, is skipped -- neither can ever be the reason something is
        held back, so both admission questions below reduce to one loop over
        the dimensions that count.
        """
        if self.total.cpus is not None and demand.cpus is not None:
            yield self._cpus_used, demand.cpus, self.total.cpus
        if self.total.memory is not None and demand.memory is not None:
            yield self._memory_used, demand.memory, self.total.memory

    def _fits(self, demand: Resources) -> bool:
        """Whether `demand` fits in what is currently unreserved."""
        return all(reserved + requested <= total for reserved, requested, total in self._dimensions(demand))

    def _exceeds_total(self, demand: Resources) -> bool:
        """Whether `demand` could never fit, even in a completely idle budget."""
        return any(requested > total for _, requested, total in self._dimensions(demand))

    def _reserve(self, demand: Resources) -> None:
        """Add `demand` to the reserved totals. Caller holds the lock."""
        self._cpus_used += demand.cpus or 0.0
        self._memory_used += demand.memory or 0

    def try_acquire(self, demand: Resources, label: str = "") -> tuple[bool, int]:
        """Reserve `demand` if it is admissible right now. Never blocks.

        Args:
            demand: What the step declared. An empty `Resources` always
                succeeds and reserves nothing.
            label: Step name, used only in the over-budget warning.

        Returns:
            `(granted, generation)`. On refusal the caller should reap any
            in-flight work and, if it has nothing else to do, park in
            `wait_for_change(generation)` -- passing back the generation
            returned here is what makes the wait race-free.
        """
        with self._cond:
            if demand.is_empty():
                return True, self._generation
            ident = threading.get_ident()
            # Larger than the entire budget: waiting could never make it fit,
            # so grant it now rather than hang. It is still reserved, which
            # pushes the reserved total over the budget and blocks every
            # later admission until it finishes -- it does not get to run
            # alongside anything that starts after it.
            if self._exceeds_total(demand):
                logger.warning(
                    "step %s declares %s, more than the whole budget (%s) -- running it anyway, unconstrained",
                    label or "<unnamed>",
                    demand.describe(),
                    self.total.describe(),
                )
                self._waiters.pop(ident, None)
                self._reserve(demand)
                return True, self._generation
            my_ticket = self._waiters.get(ident)
            oldest = min(self._waiters.values(), default=None)
            ahead_of_us = oldest is not None and (my_ticket is None or oldest < my_ticket)
            if not ahead_of_us and self._fits(demand):
                self._waiters.pop(ident, None)
                self._reserve(demand)
                return True, self._generation
            if my_ticket is None:
                self._waiters[ident] = self._next_ticket
                self._next_ticket += 1
            return False, self._generation

    def release(self, demand: Resources) -> None:
        """Return `demand` to the budget and wake every parked scheduler.

        Args:
            demand: Exactly what `try_acquire` was granted for this unit.
        """
        if demand.is_empty():
            return
        with self._cond:
            self._cpus_used = max(0.0, self._cpus_used - (demand.cpus or 0.0))
            self._memory_used = max(0, self._memory_used - (demand.memory or 0))
            self._generation += 1
            self._cond.notify_all()

    def wait_for_change(self, since: int, timeout: float | None = None) -> None:
        """Park until the budget changes.

        Args:
            since: The generation `try_acquire` refused at. If anything has
                been released since then this returns immediately, which is
                what makes refusal-then-park race-free.
            timeout: Optional cap on the wait, in seconds.
        """
        with self._cond:
            if self._generation != since:
                return
            self._cond.wait(timeout)

    def abandon(self) -> None:
        """Give up this thread's place in the queue.

        A scheduler that stops waiting -- because it failed, or finished --
        must not leave a ticket behind: strict FIFO means a stale ticket
        from a dead scheduler would block every later arrival forever.
        Called from `_run_recipe`'s exit path.
        """
        with self._cond:
            if self._waiters.pop(threading.get_ident(), None) is not None:
                self._generation += 1
                self._cond.notify_all()
