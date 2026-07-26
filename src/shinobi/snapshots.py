"""Tier 1: point-in-time snapshots of paths a step rewrites in place, so a
mutation chain can be rolled back to the state each step's position in the
DAG actually calls for.

Read `shinobi.cache` first -- this is built on top of it and changes none
of it. The skip cache already models in-place mutation correctly *as a
key*: a wired path is identified by the cache key of the step that
produced it, so a consumer's key names which state of a
repeatedly-rewritten MS it consumed. What it cannot do is put that state
back on disk, and it says so (`cache.py`: "if a consumer of a mid-chain
path does re-run on its own account, it reads whatever is on disk now").
That gap is two wrong-science bugs, not two slow runs:

- **Re-run with changed params.** Chain `split -> flag -> cal`, all three
  rewriting one MS. Change `flag`'s params and re-run: `flag` misses and
  re-executes -- against an MS that holds *post-`cal`* state, because
  that is what is on disk. Its output is garbage that looks finished.
- **Resume after a mid-mutation failure.** `cal` is killed halfway
  through rewriting the MS. A mutated path is content-blind to the cache
  by construction (it is dropped from the key, or the step would never
  look unchanged), so nothing can see the corruption, and the re-run of
  `cal` executes against its own half-written output.

The fix is to give each state of a mutated path a *name*, snapshot it
under that name, and restore it before a step that needs it re-runs.

**States are named `(producing step's cache key, producing output field)`**
-- see `cache.ProvenanceKey` for why the field is needed (a step with two
mutated outputs produces two states in one run and both resolve to its one
cache key) and how it is threaded without changing a byte of hashed key
material. A boundary path with no in-recipe producer gets a generation-0
name derived from its content fingerprint at first mutation.

**The journal** (`snapshots/chains.json`) is a new naming authority, and
the design admits it rather than pretending otherwise: generation 0 cannot
be recomputed at restore time, because by then only the post-chain state is
on disk. Per tracked path it holds the chain of generations, the current
head, the head's *status*, an in-flight marker, and the state each
consumer last consumed.

**A head is trusted or untrusted.** `UNTRUSTED` means a step was killed
mid-mutation, so the disk holds a partial write of a state we *can* name
(or the user ran `cache invalidate`). Forcing a restore is right there, and
is the whole mid-mutation recovery story. A write we cannot name is a
different problem and gets a different mechanism:

**Unnamed writes are recorded positionally, as a taint.** Some writer the
journal cannot name will legitimately advance the disk: an uncached
mutating step (caching is per-scope, so a chain can be partly cached), a
step whose mutated field Tier 1 had to exclude, or an out-of-band
replacement of the tree. The disk then holds *good* state under no name,
and every generation recorded before that write is missing it. Rolling
back to one of those would revert real work -- worse than the shipped
behaviour this is supposed to improve on.

So the chain records `tainted_through`: the newest generation known to
predate an unnamed write. A restore whose target is at or behind that
point refuses and runs against live disk, exactly as shipped. Generations
*after* it are unaffected, because they were snapshotted from a disk that
already contained the write.

This is deliberately a position and not a chain-wide flag, for two
reasons. A flag is all-or-nothing, so it throws away protection for states
that are perfectly restorable. And a flag has to be cleared by something:
the obvious candidate is the next successful mutator, but the unnamed
writer usually *cache-hits* on the following run and never re-raises it --
so the flag would be gone exactly when the restore that needs it comes
around. A position is cumulative and never needs clearing.

**The manifest is the success oracle.** Nothing here infers "the step
finished" from journal state; only a manifest entry recorded by *this run*
says so, which is why `CacheManifest.record` also stores the run id and
reconciliation compares it. A manifest entry alone is not enough: it
survives from earlier runs, so a step that re-ran (because some other
declared output was deleted) and was then killed mid-mutation would find
its own stale entry and conclude it had succeeded -- a durable false cache
hit over corrupt content, which is the exact bug class this module exists
to kill.

**Never worse than shipped.** Any field this module cannot name or vouch
for -- a keyless producer, a scattered or many-valued mutated field, a
space preflight refusal, a missing snapshot, a tainted generation -- degrades
to precisely the shipped behaviour (proceed against live disk) plus a
warning. It never degrades to a restore it cannot justify.

Out of scope, deliberately: offloaded (`--engine slurm`) runs, where no
shinobi process is present to see the writes; and the capacity store
(cross-workspace reuse, column stripping, cold tiers), which is Tier 2 and
is not approved for implementation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from shinobi.cache import ProvenanceKey, _JsonFileStore, _hash_path
from shinobi.clonefs import CloneTier, can_afford, clone_tree, probe, tree_size
from shinobi.steps.schema import Scope, mutated_path_fields

logger = logging.getLogger("shinobi.snapshots")


TRASH_SUFFIX = ".shinobi-trash."


class HeadStatus(str, Enum):
    """What the journal is willing to claim about the disk at a path."""

    # The head names the state on disk, and vouches for it.
    TRUSTED = "trusted"
    # The disk holds something we did not put there and cannot vouch for --
    # a partial write from an interrupted step, or a state the user
    # explicitly invalidated. Either way: force a restore before re-running.
    UNTRUSTED = "untrusted"


class _Faults:
    """Fault-injection points between the post-success stages.

    The post-success sequence is a five-stage commit and every gap in it is
    a crash window with its own correct recovery. Tests need to stop the
    sequence at each gap exactly, and doing that by patching internals
    would pin the implementation rather than the behaviour -- so the gaps
    are named and callable, and empty in production.
    """

    def __init__(self) -> None:
        self.hooks: dict[str, Callable[[], None]] = {}

    def __call__(self, stage: str) -> None:
        hook = self.hooks.get(stage)
        if hook is not None:
            hook()


faults = _Faults()


def _sanitize(name: str) -> str:
    """A field name reduced to something safe as a directory name."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def state_name(cache_key: str, producer_field: str) -> str:
    """The directory name for a produced state.

    The full 64-hex key, never truncated: these names are the only thing
    standing between a restore and the wrong data, and a birthday collision
    on a shortened key would be both silent and catastrophic.
    """
    return f"{cache_key}__{_sanitize(producer_field)}"


def gen0_name(path: Path, fingerprint: Any) -> str:
    """The name of a boundary path's pre-chain state.

    Hashed, because the fingerprint of an MS is thousands of entries and
    unusable as a filename. The path string is hashed in alongside the
    content for the same reason `compute_cache_key` keeps it: `cp -a` and
    `tar -x` preserve mtimes, so two copies of one MS fingerprint
    identically, and two chains must not share a generation 0 by accident.
    """
    blob = json.dumps([str(path), fingerprint], sort_keys=True, default=str)
    return f"gen0__{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


@dataclass
class Generation:
    """One named state of a tracked path."""

    name: str
    size: int = 0
    snapshot_present: bool = True

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "snapshot_present": self.snapshot_present}


@dataclass
class Marker:
    """The in-flight record: a step is mutating this path right now.

    Written before the mutation starts and cleared last, so a crash
    anywhere in the step or in the post-success sequence leaves it set and
    reconciliation gets to decide conservatively. `cache_key` is `None` for
    an uncached mutator, which has no name for what it is producing but
    must still be noticed if it dies.
    """

    step_path: str
    field: str
    cache_key: str | None
    run_id: str
    started_at: float

    def as_json(self) -> dict[str, Any]:
        return {"step_path": self.step_path, "field": self.field, "cache_key": self.cache_key, "run_id": self.run_id, "started_at": self.started_at}


@dataclass
class Chain:
    """Everything the journal knows about one tracked path."""

    dev: int
    ino: int
    ctime_ns: int
    path: str
    generations: list[Generation] = dataclass_field(default_factory=list)
    head: str | None = None
    status: HeadStatus = HeadStatus.TRUSTED
    marker: Marker | None = None
    consumed: dict[str, str] = dataclass_field(default_factory=dict)
    # The newest generation known to predate a write this journal could not
    # name (see the module docstring). Anything at or behind it is missing
    # that write; anything after it was snapshotted from a disk that already
    # contained it. `None` means no such write is known.
    tainted_through: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "dev": self.dev,
            "ino": self.ino,
            "ctime_ns": self.ctime_ns,
            "path": self.path,
            "generations": [g.as_json() for g in self.generations],
            "head": self.head,
            "status": self.status.value,
            "marker": self.marker.as_json() if self.marker else None,
            "consumed": dict(self.consumed),
            "tainted_through": self.tainted_through,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Chain":
        marker = data.get("marker")
        return cls(
            dev=data["dev"],
            ino=data["ino"],
            ctime_ns=data.get("ctime_ns", 0),
            path=data.get("path", ""),
            generations=[Generation(**g) for g in data.get("generations", [])],
            head=data.get("head"),
            status=HeadStatus(data.get("status", HeadStatus.TRUSTED.value)),
            marker=Marker(**marker) if marker else None,
            consumed=dict(data.get("consumed", {})),
            tainted_through=data.get("tainted_through"),
        )

    def generation(self, name: str) -> Generation | None:
        return next((g for g in self.generations if g.name == name), None)

    def taint_blocks(self, required: str) -> bool:
        """Whether restoring to `required` would discard an unnamed write.

        True when `required` is at or behind the taint. An unknown name on
        either side answers True as well: the comparison is positional, so
        if it cannot be made the honest answer is "cannot vouch for this",
        and refusing costs a re-run against live disk while allowing costs
        somebody's data.
        """
        if self.tainted_through is None:
            return False
        order = {generation.name: index for index, generation in enumerate(self.generations)}
        taint, target = order.get(self.tainted_through), order.get(required)
        if taint is None or target is None:
            return True
        return target <= taint


def chain_id(path: Path) -> str:
    """The journal's key for a tracked path: its canonical path string.

    Keying by inode instead is tempting -- it is what distinguishes two
    workspaces reaching one physical MS through a symlink or a bind mount
    -- but it cannot work here, because the central operation *replaces the
    inode*. A restore renames the live tree aside and clones a snapshot
    into its place, which is a freshly created directory with a new inode
    number. Keyed by inode, every restore would orphan the chain it was
    restoring and start an empty one, so the step after a rollback would
    find no generations, no head and no snapshots -- protection silently
    switching itself off precisely when it had just been used.

    The aliasing hazard the inode was meant to catch is handled where it
    actually belongs, by `aliased_chain`: two chains resolving to one live
    inode are refused, rather than each being left believing it owns the
    head.
    """
    return str(Path(os.path.abspath(os.path.normpath(path))))


def aliased_chain(chains: dict[str, "Chain"], cid: str, dev: int, ino: int) -> str | None:
    """Another chain naming the same live inode as `cid`, if there is one.

    Two workspaces reaching one physical MS through a symlink, a bind mount
    or a hardlinked tree canonicalize to different strings, and each chain
    would believe it owned the head -- so a restore in one would silently
    revert the other. Detected here and refused rather than papered over.
    """
    for other, chain in chains.items():
        if other == cid:
            continue
        try:
            st = os.stat(chain.path)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (dev, ino):
            return other
    return None


class ChainJournal(_JsonFileStore):
    """`snapshots/chains.json` -- the per-path chain of named states.

    Shares `CacheManifest`'s discipline exactly (one JSON file, a
    per-process lock, write-temp-then-rename), and its limitation: two
    concurrent *processes* on one cache directory are unguarded. Per-path
    serialisation against concurrent mutators is inherited from "the
    declared graph is the truth" -- two steps mutating one path with no
    edge between them are already an unprotected race at `max_workers > 1`,
    and with an edge they never run concurrently.

    The journal and the snapshot directory are one unit, created and
    destroyed together. A partial clean is worse than either alone: a
    generation-0 name is re-derived from live disk state, so a journal
    surviving without its snapshots would let the *post*-chain state be
    named under the position the pre-chain state used to hold.
    """

    def __init__(self, root: Path):
        super().__init__(root / "chains.json")
        self.root = root

    def snapshot_dir(self, name: str) -> Path:
        return self.root / "states" / name

    def _load(self) -> dict[str, Chain]:
        return {cid: Chain.from_json(data) for cid, data in self._read().items()}

    def _store(self, chains: dict[str, Chain]) -> None:
        self._write_atomic({cid: chain.as_json() for cid, chain in chains.items()})

    def all_chains(self) -> dict[str, Chain]:
        with self._lock:
            return self._load()

    def get(self, cid: str) -> Chain | None:
        with self._lock:
            return self._load().get(cid)

    def update(self, cid: str, mutate: Callable[[Chain | None], Chain | None]) -> None:
        """Read-modify-write one chain under the lock.

        Every journal write goes through here so that no caller can read a
        chain, decide something, and write back over a change made in
        between.
        """
        with self._lock:
            chains = self._load()
            result = mutate(chains.get(cid))
            if result is None:
                chains.pop(cid, None)
            else:
                chains[cid] = result
            self._store(chains)


_journals: dict[str, ChainJournal] = {}


def get_journal(cache_dir: str) -> ChainJournal:
    """One `ChainJournal` (and one lock) per cache directory, as with
    `get_cache_manifest` -- separate instances would each hold their own
    lock and defeat the thread-safety they exist to provide.
    """
    root = Path(cache_dir) / "snapshots"
    key = str(root.resolve())
    if key not in _journals:
        _journals[key] = ChainJournal(root)
    return _journals[key]


def new_run_id() -> str:
    """An identifier for one top-level dispatch.

    Not the pid: pids recycle, and the two things this names -- trash
    directories left by a crash, and "was it *this* run that recorded that
    manifest entry" -- are both cross-run questions where a recycled pid
    silently answers wrong.
    """
    return uuid.uuid4().hex[:16]


# --- eligibility ----------------------------------------------------------


@dataclass(frozen=True)
class Excluded:
    """A mutated field Tier 1 will not handle, and why. Reported once."""

    field: str
    reason: str


def _single_key(key: Any) -> ProvenanceKey | None:
    """The one `ProvenanceKey` in `key`, if there is exactly one.

    List wiring gives a list of keys, one per source. A single-element list
    names exactly one state, so it is unwrapped and treated like a scalar --
    the same allowance `eligible_fields` makes for a single-element path
    list, and for the same reason.
    """
    if isinstance(key, ProvenanceKey):
        return key
    if isinstance(key, (list, tuple)) and len(key) == 1 and isinstance(key[0], ProvenanceKey):
        return key[0]
    return None


def eligible_fields(
    scope: Scope,
    prepared: dict[str, Any],
    input_keys: dict[str, Any] | None,
    wired_fields: set[str] | None,
    is_scatter_slice: bool,
) -> tuple[dict[str, Path], list[Excluded]]:
    """Split `scope`'s mutated path fields into the ones Tier 1 protects and
    the ones it loudly declines to.

    The mutated set comes from `schema.mutated_path_fields`, the very same
    computation `compute_cache_key` uses to decide what to drop from the
    key -- shared rather than reimplemented, because a snapshotter and a
    keyer that disagreed about what "mutated" means would roll back content
    the key still vouched for.

    Exclusions, all of them loud:

    - **Scattered steps with mutated scatter fields.** Per-slice names would
      work for the produced state, but the *consumed* name needs the slice
      index threaded down into `_dispatch`, and scatter-plus-mutation is not
      the shape this is for (a caracal-style pipeline mutates one MS through
      a linear chain and parallelises by nesting recipes).
    - **Many-valued mutated fields.** One `(key, field)` name cannot stand
      for N paths. A wrong restore here is catastrophic rather than merely
      wasteful, so the shape is refused rather than misnamed. A *single*-
      element list is not that shape and is protected: `["/x/obs.ms"]` has
      exactly one path, so the name is unambiguous. This is not a corner
      case -- it is how an imager taking `List[MS]` is invoked on one
      measurement set, which is the common caracal shape.
    - **Wired but keyless.** The producer is uncached or uncacheable, so
      there is no name for the state this step is about to consume. Snapshot
      it under the head and a later restore would reinstate the wrong thing.
    """
    protected: dict[str, Path] = {}
    excluded: list[Excluded] = []
    keys = input_keys or {}
    for name in sorted(mutated_path_fields(scope)):
        value = prepared.get(name)
        if value is None:
            continue
        if is_scatter_slice:
            excluded.append(Excluded(name, "the step is scattered, and a scattered mutation has no single consumed state to name"))
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                excluded.append(Excluded(name, f"the field holds {len(value)} paths, and one (key, field) name cannot stand for several"))
                continue
            value = value[0]
        wired = name in wired_fields if wired_fields is not None else name in keys
        if wired and _single_key(keys.get(name)) is None:
            excluded.append(Excluded(name, "wired to a producer with no cache key (uncached or uncacheable), so the state it consumes has no name"))
            continue
        protected[name] = Path(value)
    return protected, excluded


# --- the per-step guard ---------------------------------------------------


@dataclass
class _FieldPlan:
    """What Tier 1 intends to do about one mutated field of one step."""

    field: str
    path: Path
    cid: str | None = None
    required: str | None = None  # R: the state name this step's position calls for
    trash: Path | None = None
    restored: bool = False
    skip: bool = False


class SnapshotGuard:
    """Tier 1 for one step: restore before, snapshot after, roll back on
    failure.

    Created by `_dispatch` after a cache miss and before the step body
    runs. A step whose fields are all excluded, or which mutates nothing,
    gets no guard at all.
    """

    def __init__(
        self,
        journal: ChainJournal,
        step_path: str,
        cache_key: str | None,
        run_id: str,
        fields: dict[str, Path],
        input_keys: dict[str, Any] | None,
        wired_fields: set[str] | None,
        force_copy: bool = False,
        tainting: dict[str, tuple[Path, ...]] | None = None,
    ):
        self.journal = journal
        self.step_path = step_path
        self.cache_key = cache_key
        self.run_id = run_id
        self.input_keys = input_keys or {}
        self.wired_fields = wired_fields
        self.force_copy = force_copy
        self.plans = [_FieldPlan(field=name, path=path) for name, path in fields.items()]
        # Mutated fields Tier 1 declined to protect. It cannot snapshot them,
        # but it must still record that they *wrote*, or a later restore
        # would roll the path back over work nothing will regenerate.
        self.tainting = dict(tainting or {})

    # -- pre-run ----------------------------------------------------------

    def before_run(self) -> None:
        """The pre-run sequence, in the one order that is safe.

        1. compute R per field, 2. restore, 3. write the marker, 4. Rule A.

        Restore *before* Rule A is load-bearing. In the window where a
        previous run's Rule B was skipped by a space preflight and that run
        then died mid-mutation, a Rule A that ran first would snapshot
        corrupt content under the head's durable name -- manufacturing
        exactly the false state this module exists to prevent. Restoring
        first makes the disk honest (or loudly warns that it cannot be)
        before anything is named.

        None of this can move the step's cache key: a mutated path
        contributes only its path string to the key, never a fingerprint
        (`compute_cache_key`), and a restore preserves the path string. That
        property is what licenses running the hook after the key has already
        been computed.
        """
        for plan in self.plans:
            self._prepare(plan)
        for plan in self.plans:
            if not plan.skip:
                self._restore(plan)
        for plan in self.plans:
            if not plan.skip:
                self._mark_in_flight(plan)
        for plan in self.plans:
            if not plan.skip:
                self._rule_a(plan)

    def _prepare(self, plan: _FieldPlan) -> None:
        """Resolve the chain and compute R, the state name this step needs."""
        try:
            st = plan.path.stat()
        except OSError:
            # Acquire-and-mutate: the path does not exist yet, so there is
            # nothing to stat, nothing to restore and no generation 0. The
            # chain starts at Rule B when the step succeeds.
            plan.skip = True
            return
        plan.cid = chain_id(plan.path)
        chains = self.journal.all_chains()
        chain = chains.get(plan.cid)

        alias = aliased_chain(chains, plan.cid, st.st_dev, st.st_ino)
        if alias is not None:
            logger.warning(
                "step %s: %s and %s are the same tree reached by two paths; snapshot protection is refused for '%s' because two chains over one inode would each roll the other back",
                self.step_path,
                plan.path,
                alias,
                plan.field,
            )
            plan.skip = True
            return

        if chain is not None and chain.ctime_ns and chain.ctime_ns != st.st_ctime_ns:
            # The tree's root metadata moved since our last recorded step:
            # entries added, removed or renamed by something that is not in
            # this journal. That is how an inode reuse looks, and how an
            # out-of-band replacement looks -- and, in practice, how a step
            # that writes a new column into an MS without declaring it looks.
            # Whatever it was, every generation up to here predates it.
            logger.warning(
                "step %s: %s was modified outside this pipeline (tree identity changed); states up to %s can no longer be restored, because they predate that write",
                self.step_path,
                plan.path,
                (chain.head or "the start of the chain"),
            )
            self._taint(plan.cid, chain.head)
            chain = self.journal.get(plan.cid)

        plan.required = self._required_state(plan, chain, st)

    def _required_state(self, plan: _FieldPlan, chain: Chain | None, st: os.stat_result) -> str | None:
        """R: the name of the state this step's DAG position calls for."""
        key = _single_key(self.input_keys.get(plan.field))
        if key is not None and key.producer_field is not None:
            return state_name(str(key), key.producer_field)
        if chain is not None:
            # An unwired boundary path. What this step consumed last time it
            # succeeded is the honest answer; failing that the head, which is
            # the mid-first-mutation recovery case -- the *live* fingerprint
            # would name the corruption itself.
            consumed = chain.consumed.get(f"{self.step_path}::{plan.field}")
            if consumed is not None:
                return consumed
            return chain.head
        return gen0_name(plan.path, _hash_path(plan.path))

    def _restore(self, plan: _FieldPlan) -> None:
        """Put the state named by R back on disk, if it is not there already."""
        assert plan.cid is not None
        chain = self.journal.get(plan.cid)
        if chain is None:
            return  # first mutation ever: nothing to roll back to
        if plan.required is None:
            return

        forced = chain.marker is not None or chain.status is HeadStatus.UNTRUSTED
        if not forced and chain.head == plan.required:
            return  # the disk already holds what this step needs

        if chain.taint_blocks(plan.required):
            if forced:
                # The disk is a partial write, so there is no "leave it
                # alone" option that is safe -- proceeding would re-run the
                # step against corruption, which is the failure this module
                # exists to prevent. Roll back and say plainly what it costs.
                logger.warning(
                    "step %s: restoring '%s' at %s to %s to recover from an interrupted run, but that state predates a write this journal could not name (an uncached or undeclared mutator). Anything that write produced is being discarded and will not be regenerated unless its step re-runs.",
                    self.step_path,
                    plan.field,
                    plan.path,
                    plan.required,
                )
            else:
                logger.warning(
                    "step %s: not restoring '%s' at %s to %s -- that state predates a write this journal could not name (an uncached or undeclared mutator), so rolling back to it would silently discard work nothing will regenerate. Running against live disk instead, exactly as an uncached run would.",
                    self.step_path,
                    plan.field,
                    plan.path,
                    plan.required,
                )
                return

        # Whether a state can be restored is a fact about the snapshot
        # directory, not about what the journal remembers recording. The two
        # come apart whenever something removed a snapshot behind the
        # journal's back -- eviction, a partial `ninja clean`, a user with
        # rm -rf -- and trusting the journal there turns a recoverable
        # "warn and proceed" into a crash mid-restore.
        if not self.journal.snapshot_dir(plan.required).exists():
            generation = chain.generation(plan.required)
            logger.warning(
                "step %s: '%s' needs state %s at %s but %s -- running against whatever is on disk, exactly as an uncached run would. %s",
                self.step_path,
                plan.field,
                plan.required,
                plan.path,
                "that snapshot was never taken (a space preflight refused it)" if generation is not None and not generation.snapshot_present else "no snapshot of it exists",
                "The previous run was interrupted, so that content may be a partial write." if forced else "The path holds a later state in its chain.",
            )
            return

        self._quarantine_and_swap(plan)

    def _quarantine_and_swap(self, plan: _FieldPlan) -> None:
        """Move the live tree aside, clone the snapshot in, keep the trash.

        The trash is retained until the *step* finishes, not until the swap
        finishes: a step that fails after a restore must leave the workspace
        exactly as it found it, or a failed re-run would destroy the
        calibrated tip and leave an under-processed MS sitting at a
        finished-looking path.
        """
        assert plan.required is not None
        source = self.journal.snapshot_dir(plan.required)
        tier = CloneTier.COPY if self.force_copy else probe(plan.path.parent)
        affordable, needed, available = can_afford(source, plan.path.parent, tier=tier)
        if not affordable:
            logger.warning(
                "step %s: refusing to restore '%s' at %s -- a full copy needs %d bytes and only %d are free. Running against live disk instead (as an uncached run would); free space or enable a clone-capable filesystem to restore.",
                self.step_path,
                plan.field,
                plan.path,
                needed,
                available,
            )
            return

        trash = plan.path.with_name(plan.path.name + TRASH_SUFFIX + self.run_id)
        os.rename(plan.path, trash)
        try:
            clone_tree(source, plan.path, tier=tier)
        except Exception:
            # Never leave the workspace without its tree because a restore
            # failed halfway.
            if plan.path.exists():
                shutil.rmtree(plan.path, ignore_errors=True)
            os.rename(trash, plan.path)
            raise
        plan.trash = trash
        plan.restored = True
        # The swap just replaced the tree, so the identity we recorded is
        # stale by construction. Refresh it now, or the *next* step would
        # read our own restore as an out-of-band replacement and detach the
        # chain.
        self._refresh_identity(plan)
        logger.info("step %s: restored '%s' at %s to state %s", self.step_path, plan.field, plan.path, plan.required)

    def _refresh_identity(self, plan: _FieldPlan) -> None:
        """Re-record the tree identity after we ourselves replaced it."""
        if plan.cid is None:
            return
        try:
            st = plan.path.stat()
        except OSError:
            return

        def mutate(chain: Chain | None) -> Chain | None:
            if chain is not None:
                chain.dev, chain.ino, chain.ctime_ns = st.st_dev, st.st_ino, st.st_ctime_ns
            return chain

        self.journal.update(plan.cid, mutate)

    def _mark_in_flight(self, plan: _FieldPlan) -> None:
        """Record that a mutation is starting, before it starts.

        Written for uncached mutators too (with a null key). They cannot
        name what they are about to produce, but a crash still has to be
        visible to whatever runs next.
        """
        assert plan.cid is not None
        marker = Marker(step_path=self.step_path, field=plan.field, cache_key=self.cache_key, run_id=self.run_id, started_at=time.time())
        try:
            st = plan.path.stat()
        except OSError:
            return

        def mutate(chain: Chain | None) -> Chain:
            if chain is None:
                chain = Chain(dev=st.st_dev, ino=st.st_ino, ctime_ns=st.st_ctime_ns, path=str(plan.path))
            chain.marker = marker
            return chain

        self.journal.update(plan.cid, mutate)

    def _rule_a(self, plan: _FieldPlan) -> None:
        """Rule A -- snapshot the state this step is *consuming*, before it
        overwrites it.

        Skip-if-the-name-already-exists, so on a clone-capable filesystem
        this usually dedups for free against the previous step's Rule B,
        which wrote the same state under the same name.
        """
        if plan.required is None or self.cache_key is None:
            return
        self._take(plan, plan.required, "A")

    # -- post-run ---------------------------------------------------------

    def after_success(self, record: Callable[[], None]) -> None:
        """The five-stage commit, in the one order that survives a crash at
        every gap.

        - **S1** Rule B: snapshot the state this step just produced.
        - **S2** journal: append the generation, move the head, mark it
          trusted, record what was consumed.
        - **S3** `manifest.record` -- the caller's callback.
        - **S4** delete the trash.
        - **S5** clear the marker.

        Two orderings are the point. S1 precedes S3, or a crash between them
        leaves a step the skip cache hits forever with no snapshot of its
        tip. And the marker clears *last*, so a crash anywhere in here
        leaves the path in-flight and reconciliation decides conservatively.
        """
        for plan in self.plans:
            if not plan.skip:
                self._rule_b(plan)
        faults("S1")
        for plan in self.plans:
            if not plan.skip:
                self._commit_generation(plan)
        self._taint_excluded()
        faults("S2")
        record()
        faults("S3")
        for plan in self.plans:
            self._discard_trash(plan)
        faults("S4")
        for plan in self.plans:
            if not plan.skip:
                self._clear_marker(plan)
        faults("S5")

    def _rule_b(self, plan: _FieldPlan) -> None:
        """Rule B -- snapshot the state this step produced.

        Without it a chain's final state exists only at the workspace path,
        and the first restore that touches that path destroys it.
        """
        if self.cache_key is None:
            return
        self._take(plan, state_name(self.cache_key, plan.field), "B")

    def _take(self, plan: _FieldPlan, name: str, rule: str) -> None:
        """Snapshot `plan.path` under `name`, unless that name already exists.

        A preflight refusal is recorded as `snapshot_present: false` on the
        generation rather than silently skipped, so a later restore can say
        precisely why it has nothing rather than reporting a generic miss.
        """
        dest = self.journal.snapshot_dir(name)
        if dest.exists():
            return
        tier = CloneTier.COPY if self.force_copy else probe(self.journal.root)
        affordable, needed, available = can_afford(plan.path, self.journal.root, tier=tier)
        if not affordable:
            logger.warning(
                "step %s: rule %s could not snapshot '%s' as %s -- needs %d bytes, %d free. The chain is incomplete at this generation; a later rollback to it will refuse rather than restore the wrong thing.",
                self.step_path,
                rule,
                plan.field,
                name,
                needed,
                available,
            )
            self._record_generation(plan, name, size=0, present=False)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(dest.name + ".partial." + self.run_id)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        try:
            clone_tree(plan.path, staging, tier=tier)
            # Rename last, so the name only ever appears over a complete
            # tree: a half-written snapshot under a durable name is
            # indistinguishable from a good one at restore time.
            os.rename(staging, dest)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if rule == "A":
            # Apparent size, recorded once at insert -- eviction must never
            # `du` a snapshot directory to decide what to drop.
            self._record_generation(plan, name, size=tree_size(dest), present=True)

    def _record_generation(self, plan: _FieldPlan, name: str, size: int, present: bool) -> None:
        if plan.cid is None:
            return

        def mutate(chain: Chain | None) -> Chain | None:
            if chain is None:
                return None
            for gen in chain.generations:
                if gen.name == name:
                    gen.snapshot_present = gen.snapshot_present or present
                    return chain
            chain.generations.append(Generation(name=name, size=size, snapshot_present=present))
            return chain

        self.journal.update(plan.cid, mutate)

    def _commit_generation(self, plan: _FieldPlan) -> None:
        """S2 -- the head now names what is on disk, and vouches for it.

        For an uncached mutator (rule 1 of the never-worse rules) there is
        no name to move the head to, so the chain is *detached* instead: it
        keeps its snapshots, gives up restore until a cached step
        re-establishes a head, and never lets a later step trust a stale
        name over work it cannot see.
        """
        assert plan.cid is not None
        try:
            st = plan.path.stat()
        except OSError:
            return
        produced = state_name(self.cache_key, plan.field) if self.cache_key else None
        consumed_key = f"{self.step_path}::{plan.field}"
        consumed_name = plan.required

        def mutate(chain: Chain | None) -> Chain:
            if chain is None:
                chain = Chain(dev=st.st_dev, ino=st.st_ino, ctime_ns=st.st_ctime_ns, path=str(plan.path))
            chain.ctime_ns = st.st_ctime_ns
            if produced is None:
                # An uncached mutator: it just advanced the disk to a state
                # it cannot name, so every generation up to here is missing
                # that work and must not be restored over it.
                if chain.head is not None and (chain.tainted_through is None or not chain.taint_blocks(chain.head)):
                    chain.tainted_through = chain.head
                return chain
            if not any(g.name == produced for g in chain.generations):
                snapshot = self.journal.snapshot_dir(produced)
                chain.generations.append(Generation(name=produced, size=tree_size(snapshot), snapshot_present=snapshot.exists()))
            chain.head = produced
            chain.status = HeadStatus.TRUSTED
            if consumed_name is not None:
                chain.consumed[consumed_key] = consumed_name
            return chain

        self.journal.update(plan.cid, mutate)

    def _taint_excluded(self) -> None:
        """Record the writes this step made through fields Tier 1 excluded.

        The taint lands at the chain's *current* head, which is the newest
        state that predates this write. It is written even though nothing
        was snapshotted, and that is the whole point: an excluded field is
        invisible to every other mechanism here, so without this a later
        restore happily rolls back over it. A path with no chain yet is
        skipped -- there is no earlier generation for the write to invalidate.
        """
        for field, paths in self.tainting.items():
            for path in paths:
                cid = chain_id(path)
                chain = self.journal.get(cid)
                if chain is None or chain.head is None:
                    continue
                logger.info(
                    "step %s: '%s' wrote to %s without snapshot protection; states up to %s can no longer be restored, because they predate that write",
                    self.step_path,
                    field,
                    path,
                    chain.head,
                )
                self._taint(cid, chain.head)

    def after_failure(self) -> None:
        """The step failed (non-zero, or an exception). Put back what we moved.

        The workspace keeps its pre-run state, and the marker is left *set*
        on purpose: the next run's restore then forces a rollback from R
        before re-executing, which is the mid-mutation recovery case. No
        reconciliation is needed for this path -- the journal is already
        honest about it.
        """
        for plan in self.plans:
            if plan.trash is None:
                continue
            try:
                if plan.path.exists():
                    shutil.rmtree(plan.path, ignore_errors=True)
                os.rename(plan.trash, plan.path)
                logger.info("step %s: rolled '%s' at %s back to its pre-run state after failure", self.step_path, plan.field, plan.path)
            except OSError:
                logger.exception("step %s: could not restore %s from %s -- the pre-run tree is still there", self.step_path, plan.path, plan.trash)
            finally:
                plan.trash = None

    def _discard_trash(self, plan: _FieldPlan) -> None:
        if plan.trash is not None:
            shutil.rmtree(plan.trash, ignore_errors=True)
            plan.trash = None

    def _clear_marker(self, plan: _FieldPlan) -> None:
        if plan.cid is None:
            return

        def mutate(chain: Chain | None) -> Chain | None:
            if chain is not None:
                chain.marker = None
            return chain

        self.journal.update(plan.cid, mutate)

    def _taint(self, cid: str, through: str | None) -> None:
        """Record that a write this journal cannot name landed after `through`.

        The taint only ever moves *forward*: two unnamed writes at different
        points in a chain both have to be honoured, and the later one
        subsumes the earlier.

        `through=None` -- an unnamed write with no generation before it --
        is a no-op, and deliberately so. Every generation this chain goes on
        to record will be snapshotted from a disk that already contains that
        write, so there is nothing for a taint to protect.
        """
        if through is None:
            return

        def mutate(chain: Chain | None) -> Chain | None:
            if chain is None:
                return None
            # `taint_blocks(through)` is true when `through` is at or behind
            # the taint already recorded, i.e. the existing one is stricter.
            if chain.tainted_through is None or not chain.taint_blocks(through):
                chain.tainted_through = through
            return chain

        self.journal.update(cid, mutate)


# --- who else is running --------------------------------------------------


class RunPresence:
    """Which shinobi processes are alive on one cache directory.

    This exists for exactly one decision: whether it is safe to *reconcile*.
    Reconciliation reads an in-flight marker with no matching manifest entry
    as "that run died" and rolls the workspace back to the quarantined tree
    -- which is right for a corpse and catastrophic for a run that is simply
    still going. A second shinobi starting while the first is mid-step would
    otherwise delete the first's work out from under it, mid-write.

    Each run announces itself by taking an exclusive `flock` on its own
    `locks/<run_id>.lock`, held for as long as the process lives. "Am I
    alone" is then answered by trying to lock everyone *else's* file: a file
    that locks is a corpse's, a file that refuses is a live run's. Per-run
    files rather than one shared lock, because a single lock would have to
    be converted between shared and exclusive to re-ask the question, and
    `flock(2)` does not promise conversion is atomic -- the gap would be
    exactly the window this is meant to close.

    **Best-effort, and deliberately so.** `flock` is unreliable on precisely
    the filesystems A1 names: it needs the `flock` mount option on Lustre and
    can otherwise fail *or silently no-op*, and on NFS it depends on the
    protocol version and a healthy lock daemon. So this never claims safety
    it cannot demonstrate -- anything it cannot establish reads as "not
    alone", and reconciliation is skipped.

    Skipping is the safe direction, which is what makes a best-effort guard
    worth having. Reconciliation is a repair, not a prerequisite: leaving a
    marker set makes the next restore *force* a rollback (branch 1), so
    crash recovery still happens through the normal path. What is deferred
    is only the tidying -- quarantined trees stay on disk, where
    `ninja cache check` reports them.

    Making concurrent runs actually *work*, rather than merely not corrupt
    each other, needs a path-scoped lock held across the mutating window and
    a probe that verifies locking functions at all on the filesystem in use.
    That is a bigger piece of work and is not this.
    """

    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.run_id = run_id
        self._path = root / "locks" / f"{run_id}.lock"
        self._fd: int | None = None
        self.announced = False

    def announce(self) -> None:
        """Register this process as live, if the filesystem lets us."""
        if self._fd is not None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return
        self._fd = fd
        self.announced = True

    def alone(self) -> bool:
        """Whether no *other* live process is using this cache directory.

        `False` whenever that cannot be established -- including when we
        could not announce ourselves, since a process invisible to everyone
        else has no business making destructive decisions about their state.
        """
        if not self.announced:
            return False
        try:
            others = [entry for entry in (self.root / "locks").iterdir() if entry != self._path]
        except OSError:
            return False
        for other in others:
            try:
                fd = os.open(other, os.O_RDWR)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False  # somebody still holds it: a live run
            finally:
                os.close(fd)
            # Lockable, so its owner is gone. Left on disk rather than
            # unlinked: a run that has just created its file but not yet
            # flocked it also looks lockable, and deleting it there would
            # make a live run invisible -- the very failure this prevents.
        return True

    def release(self) -> None:
        """Stop advertising this process, and remove its lock file."""
        if self._fd is None:
            return
        os.close(self._fd)  # releases the flock
        self._fd = None
        self.announced = False
        try:
            self._path.unlink()
        except OSError:
            pass


_presences: dict[str, RunPresence] = {}
_presences_lock = threading.Lock()


def announce_run(cache_dir: str, run_id: str) -> RunPresence:
    """The `RunPresence` for this cache directory, announced on first use.

    One per cache directory per process: a process that dispatches twice is
    the same live process both times, and re-announcing would leave it
    holding two lock files and treating its own first one as a corpse.
    """
    root = Path(cache_dir) / "snapshots"
    key = str(root.resolve())
    with _presences_lock:
        presence = _presences.get(key)
        if presence is None:
            presence = _presences[key] = RunPresence(root, run_id)
        presence.announce()
        return presence


def release_runs() -> None:
    """Drop every announcement this process holds. For tests, and for
    long-lived embedders that want to stop advertising between pipelines.
    """
    with _presences_lock:
        for presence in _presences.values():
            presence.release()
        _presences.clear()


# --- crash reconciliation -------------------------------------------------


def reconcile(cache_dir: str, manifest) -> list[str]:
    """Decide what a crashed run left behind, once per cache directory per
    process.

    A SIGKILL, an OOM or a node eviction leaves no process to roll anything
    back, so the marker survives and this is what reads it. The decision is
    made against the *manifest*, because a manifest entry is the only
    durable record that a step actually finished -- and specifically
    against an entry recorded by the same run, since an entry from an
    earlier successful run of the same step would otherwise be read as
    proof that the interrupted run had succeeded.

    That distinction is not hypothetical. Delete some unrelated declared
    output of a step that mutates an MS: the outputs-exist check misses, the
    step re-runs, and if it is killed mid-rewrite a run-blind oracle would
    find the old entry, call the step complete, drop the trash and leave the
    head trusted -- a permanent cache hit over a half-written MS.

    Returns a human-readable line per decision, for `ninja cache check`.
    """
    journal = get_journal(cache_dir)
    notes: list[str] = []
    for cid, chain in journal.all_chains().items():
        marker = chain.marker
        if marker is None:
            continue
        entry = manifest.entry(marker.step_path)
        completed = entry is not None and entry.get("cache_key") == marker.cache_key and marker.cache_key is not None and entry.get("run_id") == marker.run_id
        trash = _trash_for(Path(chain.path), marker.run_id)

        if completed:
            # Crashed in S4/S5, after the step had really finished and
            # recorded. S1-before-S3 guarantees its tip snapshot exists.
            if trash is not None:
                shutil.rmtree(trash, ignore_errors=True)
            notes.append(f"{chain.path}: run {marker.run_id} of '{marker.step_path}' completed and recorded; discarded its trash")

            def clear(c: Chain | None) -> Chain | None:
                if c is not None:
                    c.marker = None
                return c

            journal.update(cid, clear)
            continue

        # No entry from this run: the step did not finish. If the crash fell
        # between S2 and S3 this re-runs a step that actually succeeded --
        # bounded waste in a narrow window, taken deliberately over the
        # alternative, which is a false hit over content nothing verified.
        if trash is not None:
            try:
                if Path(chain.path).exists():
                    shutil.rmtree(chain.path, ignore_errors=True)
                os.rename(trash, chain.path)
                notes.append(f"{chain.path}: run {marker.run_id} of '{marker.step_path}' did not complete; rolled back to its pre-run state")
            except OSError:
                notes.append(f"{chain.path}: could not roll back from {trash} -- left in place for inspection")
        else:
            notes.append(f"{chain.path}: run {marker.run_id} of '{marker.step_path}' did not complete; the path is a partial write and will be restored before the step re-runs")

        def interrupted(c: Chain | None) -> Chain | None:
            if c is not None:
                c.marker = None
                c.status = HeadStatus.UNTRUSTED
            return c

        journal.update(cid, interrupted)
    return notes


def _trash_for(path: Path, run_id: str) -> Path | None:
    candidate = path.with_name(path.name + TRASH_SUFFIX + run_id)
    return candidate if candidate.exists() else None


def orphan_trash(cache_dir: str) -> list[Path]:
    """Trash directories with no marker to explain them.

    Never swapped back automatically: without its marker there is nothing
    to say what a quarantined tree was quarantined *for*, and reinstating it
    blindly is as likely to revert good work as to recover from a crash.
    Reported here, removed by `ninja clean --force`.
    """
    journal = get_journal(cache_dir)
    chains = journal.all_chains().values()
    live = {str(_trash_for(Path(c.path), c.marker.run_id)) for c in chains if c.marker and _trash_for(Path(c.path), c.marker.run_id)}
    found: list[Path] = []
    for chain in chains:
        path = Path(chain.path)
        if not path.parent.is_dir():
            continue
        for sibling in path.parent.glob(path.name + TRASH_SUFFIX + "*"):
            if str(sibling) not in live:
                found.append(sibling)
    return sorted(set(found))


# --- capacity -------------------------------------------------------------


def _protected_names(chains: dict[str, Chain]) -> set[str]:
    """Every state name some chain still needs.

    A live head, and every recorded consumed state. Deleting a generation
    mid-chain makes every restore through it impossible, and on a
    clone-capable filesystem reclaims almost nothing anyway (its blocks are
    shared with the workspace tree and with its neighbours), so the trade is
    bad in both directions.

    The consumed-record rule is what protects generation 0 automatically:
    the first mutator in a chain records it as what it consumed, and keeps
    doing so for as long as that step's last run consumed it.
    """
    protected: set[str] = set()
    for chain in chains.values():
        if chain.head is not None:
            protected.add(chain.head)
        if chain.tainted_through is not None:
            # The taint is a *position*, resolved by looking its name up
            # among the generations. Evict it and the comparison loses its
            # anchor, at which point every restore on the chain refuses.
            protected.add(chain.tainted_through)
        protected.update(chain.consumed.values())
    return protected


def evict(cache_dir: str, target_bytes: int) -> list[tuple[str, int]]:
    """Free up to `target_bytes` of snapshot space, safest candidates first.

    Two passes, in order:

    1. **Superseded generations** -- named by no live head and no consumed
       record -- newest first. These are states nothing can still ask for.
    2. **Dead chains only** (the workspace path no longer exists), oldest
       first. A dead chain's tip is not a tip anyone can roll back to,
       because there is nothing left at the path to roll back.

    A live chain's tip is never evicted, which is not the same rule as
    "never evict a tip": for a live chain the tip *is* the head, and the
    head is what the next run's restore is most likely to need.

    Sizes come from the journal, recorded when each snapshot was taken --
    walking the snapshot directory at eviction time to decide what to delete
    would make the cheap operation the expensive one.

    Returns the `(name, bytes)` pairs actually removed.
    """
    journal = get_journal(cache_dir)
    chains = journal.all_chains()
    live = {cid: chain for cid, chain in chains.items() if Path(chain.path).exists()}
    # Only a *live* chain can protect a name: protection means "some future
    # restore might need this", and a chain whose workspace path is gone has
    # nothing left to restore to.
    protected = _protected_names(live)
    removed: list[tuple[str, int]] = []
    freed = 0

    def age(name: str) -> float:
        snapshot = journal.snapshot_dir(name)
        return snapshot.stat().st_mtime if snapshot.exists() else 0.0

    # Pass 1: superseded generations of live chains, newest first -- the
    # oldest states in a chain are the ones a rollback is most likely to
    # reach back for. `generations` is append-ordered, so later is newer.
    superseded: list[tuple[str, int]] = []
    for chain in live.values():
        for gen in reversed(chain.generations):
            if gen.snapshot_present and gen.name not in protected:
                superseded.append((gen.name, gen.size))

    # Pass 2: dead chains entirely, oldest-accessed first.
    dead: list[tuple[float, str, int]] = []
    for cid, chain in chains.items():
        if cid in live:
            continue
        for gen in chain.generations:
            if gen.snapshot_present and gen.name not in protected:
                dead.append((age(gen.name), gen.name, gen.size))

    seen: set[str] = set()
    for name, size in superseded + [(name, size) for _age, name, size in sorted(dead)]:
        if freed >= target_bytes:
            break
        if name in seen:
            continue
        seen.add(name)
        shutil.rmtree(journal.snapshot_dir(name), ignore_errors=True)
        removed.append((name, size))
        freed += size

    dropped = {name for name, _ in removed}
    if dropped:

        def prune(chain: Chain | None) -> Chain | None:
            if chain is not None:
                for gen in chain.generations:
                    if gen.name in dropped:
                        gen.snapshot_present = False
            return chain

        for cid in chains:
            journal.update(cid, prune)
    return removed


def invalidate(cache_dir: str, step_path: str, manifest) -> list[str]:
    """Drop a step's cached result *and* its Tier 1 state, together.

    Removing only the manifest entry is not enough, and the reason is
    specific: Rule B makes whatever a zero-returncode run produced into a
    durable named state, and restore reinstates it faithfully. So a step
    that "succeeded" while writing garbage leaves that garbage named,
    snapshotted, and reachable -- and the escape hatch has to reach it.

    So this also rolls the head back to the previous generation and marks
    it untrusted. The rollback alone would not be enough either: without the
    untrusted mark the *disk* still holds the post-invalidation state while
    the head names the rolled-back one, so the next miss compares them,
    finds head == R, and no-ops -- re-executing against exactly the content
    the invalidation was meant to escape.
    """
    journal = get_journal(cache_dir)
    notes: list[str] = []

    # Read the entry before removing it: its cache key is what names the
    # generations this step produced (`state_name(key, field)`), which is the
    # only link from a step path to a state name.
    entry = manifest.entry(step_path)
    cache_key = entry.get("cache_key") if entry else None
    with manifest._lock:
        data = manifest._read()
        if data.pop(step_path, None) is not None:
            manifest._write_atomic(data)
            notes.append(f"removed the manifest entry for '{step_path}'")
    if cache_key is None:
        return notes

    for cid, chain in journal.all_chains().items():
        order = {gen.name: index for index, gen in enumerate(chain.generations)}
        produced = {gen.name for gen in chain.generations if gen.name.startswith(f"{cache_key}__")}
        if not produced:
            continue
        first = min(order[name] for name in produced)

        # Every step downstream of the invalidated one has to re-run too, and
        # the skip cache will not make it: the invalidated step re-executes
        # with the same params and so produces the *same* key, which means
        # its consumers' keys do not move either. They would cache-hit over a
        # tree that has just been rolled back, leaving the chain stopped
        # mid-way with a manifest claiming it finished. Downstream is read off
        # the consumed records -- a step that consumed the invalidated
        # generation or anything after it is downstream of it; one that
        # consumed something earlier is not, and is left alone.
        downstream = {key.split("::")[0] for key, record in chain.consumed.items() if order.get(record, -1) >= first}
        with manifest._lock:
            data = manifest._read()
            dropped_steps = [name for name in downstream if data.pop(name, None) is not None]
            if dropped_steps:
                manifest._write_atomic(data)
        for name in sorted(dropped_steps):
            notes.append(f"removed the manifest entry for '{name}', downstream of '{step_path}' on {chain.path}")

        for name in produced:
            shutil.rmtree(journal.snapshot_dir(name), ignore_errors=True)
        surviving = [gen for gen in chain.generations if gen.name not in produced]
        previous = surviving[-1].name if surviving else None

        def roll_back(c: Chain | None, previous=previous, produced=produced) -> Chain | None:
            if c is None:
                return None
            c.generations = [g for g in c.generations if g.name not in produced]
            if c.head in produced:
                c.head = previous
            c.consumed = {key: record for key, record in c.consumed.items() if record not in produced}
            c.status = HeadStatus.UNTRUSTED
            return c

        journal.update(cid, roll_back)
        notes.append(f"rolled {chain.path} back to {previous or 'no generation'} and marked it untrusted, so the next run restores it before re-running")
    return notes


def check(cache_dir: str, manifest) -> dict[str, list[str]]:
    """Everything `ninja cache check` reports, gathered in one pass.

    The point of reporting the capability-ladder decisions alongside the
    chain state is that when the space arithmetic looks wrong on some future
    Lustre or ZFS deployment, the answer should be in a report rather than
    in a re-run with debug logging.
    """
    from shinobi.clonefs import decisions

    journal = get_journal(cache_dir)
    chains = journal.all_chains()
    report: dict[str, list[str]] = {"off_tip": [], "unreconciled": [], "orphan_trash": [], "disagreements": [], "missing_snapshots": [], "unprotected": [], "capabilities": []}

    for chain in chains.values():
        if chain.marker is not None:
            report["unreconciled"].append(f"{chain.path}: '{chain.marker.step_path}' (run {chain.marker.run_id}) never finished")
        if chain.status is HeadStatus.UNTRUSTED:
            report["off_tip"].append(f"{chain.path}: head {chain.head} is not vouched for; the next run restores before re-running")
        if chain.tainted_through is not None:
            report["unprotected"].append(f"{chain.path}: states up to {chain.tainted_through} cannot be restored -- a write this journal could not name (an uncached or undeclared mutator) landed after them")
        if not Path(chain.path).exists():
            report["disagreements"].append(f"{chain.path}: tracked but no longer on disk")
        for gen in chain.generations:
            if not gen.snapshot_present:
                report["missing_snapshots"].append(f"{chain.path}: generation {gen.name} was never snapshotted (a space preflight refused it)")
            elif not journal.snapshot_dir(gen.name).exists():
                report["disagreements"].append(f"{chain.path}: generation {gen.name} is recorded present but its snapshot is gone")
        for key, record in chain.consumed.items():
            step_path = key.split("::")[0]
            if manifest.entry(step_path) is None:
                report["disagreements"].append(f"{chain.path}: '{step_path}' has a consumed record ({record}) but no manifest entry")

    report["orphan_trash"] = [str(path) for path in orphan_trash(cache_dir)]
    report["capabilities"] = [f"{d.mountpoint or '?'} ({d.fstype}): {d.tier.value} -- {d.reason}" for d in decisions()]
    return report
