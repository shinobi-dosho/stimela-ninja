"""Step-level caching: skip re-running a step whose identity (container
image + command, or -- for a `@shinobi.pystep`-style bare `Scope` -- its
own function's source) and resolved params are unchanged since a prior
successful run, and whose declared outputs still exist on disk.

Opt-in (see `Scope.cache`/`Scope.cache_dir`, same precedence chain as
`Scope.backend`: explicit call-time `cache=`/`cache_dir=` kwarg > a
Scope's own `.cache`/`.cache_dir` > the enclosing recipe's > `AppConfig.
cache`'s default, itself disabled). Applied per-leaf-step (a `Cab`, or a
bare-`Scope` step driven by a function) -- never per-`Recipe`: a
partially-changed nested recipe should still skip only its own unchanged
sub-steps, and `_run_recipe` already recurses into each sub-step's own
`_dispatch` call individually, so nothing special is needed for nested
recipes; the gate (see `steps/dispatch.py::_dispatch`) simply never fires
for a `Recipe`-shaped scope.

A path input is identified one of two ways, and which one depends on
whether the DAG knows where the file came from.

**Wired paths: by provenance.** If an input is wired (`OutputRef`, or an
`InputRef` carrying provenance in from an enclosing recipe), it is
identified by the *cache key of the step that produced it* -- Merkle-style,
so a step's key transitively covers everything upstream of it -- and its
bytes on disk are not examined at all. This is the only thing that models
in-place mutation correctly. A path that several steps rewrite in sequence
(the caracal2 shape: split an MS, then tag it, flag it, calibrate it, every
step reading and writing the *same* MS) has no single "current" content:
what a consumer actually consumed is the state of that path *at its own
point in the chain*, which is exactly what the producing step's key names
and exactly what an mtime cannot express. Concretely, mtime cannot
distinguish "I mutated this path myself last run" from "an upstream step
rebuilt it", nor "a step declared before me rewrote it" from "the file I
read changed" -- and those need opposite answers.

**Unwired paths: by content**, as `(relative_path, mtime_ns, size)` per
file -- not a full byte hash, since radio-astronomy inputs (MS directories,
FITS cubes) run to many GB and hashing them every run would defeat the
point; the same tradeoff Make accepts. These are the DAG's boundary: raw
data the user supplied, which shinobi did not produce and about whose
history it knows nothing. An unwired path that is *also* a declared output
field of the same step is excluded from the key entirely, because a step
that mutates a boundary path in place would otherwise always look
"changed" on a resumed run -- its own previous mutation moved the mtime.
For such a step "unchanged" means its params are unchanged and its declared
output still exists.

Note what the boundary regime does *not* promise. It covers a path a step
declares as an **input**; a step that merely *produces* a path it was never
given (an acquisition step resolving raw data from an identifier, say)
has nothing content-hashed, and is keyed on its params alone. That is
usually right -- the identifier denotes the data -- but it means the bytes
behind such a path can be replaced without invalidating anything. A caller
whose "identifier" is really a filename should know that about itself.

What this buys, in the two cases the previous mtime-only scheme got wrong:

- an upstream rebuild now invalidates every dependent, transitively, even
  through steps whose only path input is one they mutate in place (which
  was excluded from the key, leaving them nothing at all to notice a
  rebuild by). Skipping those produced not a stale result but a *wrong*
  one -- a chain half-updated, e.g. an MS re-split with new parameters and
  then never re-flagged;
- a resumed run no longer re-runs steps whose input merely got mutated
  *later* by some other step. Those were re-running on every pass forever,
  since each pass moved the mtime again for the next one.

What it costs: shinobi now trusts its own graph about intermediate files.
An intermediate edited out of band between runs is not noticed (its
producer's key is unchanged, so consumers hit) -- deleting it is still
caught, via `CacheManifest.check`'s outputs-exist test. And an **undeclared
dependency** -- two steps sharing a path on disk without an edge between
them -- gets no protection at all, where content hashing used to catch some
of them by accident. Both follow from the same principle the rest of
shinobi is built on: the declared graph is the truth, and an on-disk
dependency left out of it does not exist (it is already a race at
`max_workers > 1`, for the same reason).

Related hazard, which no cache can fix: if a consumer of a mid-chain path
*does* re-run on its own account, it reads whatever is on disk now, not the
state its position in the DAG says it should see. In-place mutation makes
the graph's dataflow and the filesystem's disagree; provenance keys the
former faithfully, it cannot repair the latter.

Only the wiring layer knows which inputs came from which step, so the keys
are threaded down from `_run_recipe` (see `steps/dispatch.py`) rather than
discovered here. A `Recipe` is never itself cached, so it carries a
*per-output-field* provenance map (`StepResult.output_keys`) instead of one
key: each of its declared outputs is produced by a different sub-step, and
keying all of them off "something in this recipe changed" would invalidate
most of the cache on any edit.

The cache key's image component is the image's tag string, not a resolved
container digest -- avoids an extra `docker`/`podman inspect` call and a
hard runtime dependency on the container tool being reachable at
cache-check time. Known, accepted limitation: rebuilding a mutable tag
like `:latest` without bumping the tag string won't invalidate the cache.

`CacheManifest` is one JSON file per configured cache directory, shared
by every step regardless of which top-level Recipe it belongs to --
entries are keyed by a step's full dotted path (`<top-level-recipe-name>.
<step>.<sub-step>...`), which already disambiguates unrelated pipelines
as long as their top-level Recipe names actually differ; a caller that
wants that guarantee (e.g. one assembling several distinct pipelines
that might share one `cache_dir`) is responsible for giving each
top-level `Recipe` a name that's unique to it.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import stat
import threading
from pathlib import Path
from typing import Any, Callable

from shinobi.results import StepResult
from shinobi.steps.schema import Cab, Scope, mutated_path_fields, path_fields


_path_hash_cache: dict[Path, Any] = {}
_path_hash_lock = threading.Lock()


def invalidate_path_hashes() -> None:
    """Drop every memoized `_hash_path` result. Called by `_dispatch` after
    any step actually executes -- see `_hash_path` for why that is the only
    safe invalidation point.
    """
    with _path_hash_lock:
        _path_hash_cache.clear()


# `Path.exists()`/`Path.is_file()` treat these as "not there" rather than
# raising, and the fingerprint keeps that: an optional input the caller
# didn't supply, a path whose parent isn't a directory, and a symlink loop
# resolved from the top all key as absent. Anything else (EACCES, most
# obviously) is a path that *is* there and cannot be read -- see
# `_walk_fingerprint`.
_ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def _walk_fingerprint(path: Path) -> Any:
    """`[relative_path, mtime_ns, size]` per file under `path`, sorted. A
    regular file yields the single entry `["." , mtime_ns, size]`; a
    directory (an MS, say) yields one entry per file beneath it.

    Traversal is an explicit `os.scandir` stack rather than `os.walk` plus a
    `Path` per file. That is not a style preference -- constructing a `Path`,
    calling its bound `.stat()` (which re-does `os.stat` on a freshly built
    string) and then `.relative_to()` (another `Path`, split into parts and
    compared) costs ~40 us per file against ~2.8 us here. Measured on a
    1,997-file tree: 90.4 ms before, 6.7 ms after, for byte-identical output.
    The stat syscalls were never the expensive part (~3.5 ms of that 90).

    Semantics, which `os.walk` defined only by accident and nothing pinned:

    - **Symlinked directories are neither descended nor listed.** `os.walk`
      puts them in its `dirs` list and, with `followlinks=False`, does not
      recurse -- so they never reached the old `files` loop either. Preserved
      deliberately: it is what stops a symlink cycle from hanging the walk.
    - **Symlinked files are followed**, contributing their target's mtime and
      size, because the old code stat'd through the link.
    - **A broken symlink, a symlink loop, and an unreadable subdirectory are
      skipped, not fatal.** All three used to propagate out of
      `compute_cache_key` and kill the run: `os.walk` lists a dangling name
      under `files` and the subsequent `stat` raised `FileNotFoundError`; a
      cycle raised `ELOOP`; a directory with `r` but no `x` raised
      `PermissionError`. A boundary input is data shinobi did not produce and
      does not control, so a cache fingerprint is the wrong place to fail.
    - **A file that vanishes mid-walk is skipped**, closing the same race.

    An unreadable `path` (EACCES on the root itself) is a deliberate change
    rather than an inherited one: it used to raise. It now returns a distinct
    `__unreadable__` marker, *not* `None`, so it cannot key the same as a
    genuinely missing file and quietly turn into a hit when permissions
    change. Anything `Path.exists()` treated as absent still keys as absent.
    """
    try:
        st = os.stat(path)
    except ValueError:
        return None  # embedded NUL: `Path.exists()` swallows this too
    except OSError as exc:
        if exc.errno in _ABSENT_ERRNOS:
            return None
        return [["__unreadable__", exc.errno]]

    # The sample is appended as a *fourth* element rather than folded into
    # the existing three, so with the flag off every entry is byte-identical
    # to what it always was and no existing cache entry is orphaned -- the
    # same conditional-append discipline as `__venv__`/`__upstream__`.
    sample = _content_sample_enabled

    if stat.S_ISREG(st.st_mode):
        entry = [".", st.st_mtime_ns, st.st_size]
        if sample:
            entry.append(_content_sample(path, st.st_size))
        return [entry]

    entries: list[list[Any]] = []
    stack: list[tuple[str, str]] = [(os.fspath(path), "")]
    while stack:
        dirpath, prefix = stack.pop()
        try:
            scan = os.scandir(dirpath)
        except OSError:
            continue  # unreadable subdirectory; os.walk(onerror=None) also swallows this
        with scan:
            for entry in scan:
                rel = f"{prefix}{os.sep}{entry.name}" if prefix else entry.name
                # lstat first, and branch on the mode bits rather than
                # entry.is_dir(): on filesystems that return DT_UNKNOWN in
                # d_type (Lustre, some XFS configs) is_dir() issues its own
                # lstat and a following stat() issues a second, doubling the
                # syscalls per entry.
                try:
                    est = entry.stat(follow_symlinks=False)
                except OSError:
                    continue  # vanished between scandir and stat
                if stat.S_ISDIR(est.st_mode):
                    stack.append((entry.path, rel))
                    continue
                if stat.S_ISLNK(est.st_mode):
                    try:
                        est = os.stat(entry.path)
                    except OSError:
                        continue  # dangling, or a loop
                    if stat.S_ISDIR(est.st_mode):
                        continue  # symlink to a directory: os.walk never listed these
                record = [rel, est.st_mtime_ns, est.st_size]
                if sample:
                    record.append(_content_sample(Path(entry.path), est.st_size))
                entries.append(record)
    return sorted(entries)


_SAMPLE_BYTES = 4096
_content_sample_enabled = False


def set_content_sample(enabled: bool) -> None:
    """Turn the bounded content sample on or off (see `_content_sample`).

    Process-global rather than threaded through `compute_cache_key`, because
    it has to reach inside `_hash_path`'s memo -- and because it is a
    property of the *workspace*, not of a step: two steps in one run keying
    the same boundary path by different rules would be incoherent. Flipping
    it invalidates the memo for the same reason a step execution does.
    """
    global _content_sample_enabled
    if enabled != _content_sample_enabled:
        _content_sample_enabled = enabled
        invalidate_path_hashes()


def _content_sample(path: Path, size: int) -> str | None:
    """A digest of the first and last 4 KiB of `path`.

    Aimed at exactly one gap: two *different* datasets colliding because
    they have the same layout, the same sizes and mtimes preserved by
    `cp -a`/`tar -x` inside one granularity window. Sampling the extents
    separates them for 8 KiB of reads per file.

    It emphatically does **not** cover the limitation it sits next to. An
    intermediate edited out of band -- a rewritten `FLAG` column, say --
    changes neither the file's size nor its first and last pages, and stays
    invisible. That case is undetectable by design, not by omission: the
    declared graph is the truth, and an on-disk dependency left out of it
    does not exist. Anyone reading this sample as covering both will be
    wrong in the direction that loses data.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(_SAMPLE_BYTES)
            if size > _SAMPLE_BYTES:
                handle.seek(max(size - _SAMPLE_BYTES, _SAMPLE_BYTES))
                tail = handle.read(_SAMPLE_BYTES)
            else:
                tail = b""
    except OSError:
        return None
    # blake2b from the stdlib, not blake3: a new third-party dependency is
    # not justified by 8 KiB of hashing per file.
    return hashlib.blake2b(head + tail, digest_size=16).hexdigest()


def _hash_path(path: Path) -> Any:
    """`(relative_path, mtime_ns, size)` for every file under `path` -- a
    single file yields one tuple; a directory (e.g. an MS) yields one per
    file within it, sorted for a deterministic result. `None` if `path`
    doesn't exist (e.g. an optional input the caller didn't supply). See
    `_walk_fingerprint` for the traversal and its edge-case semantics.

    Memoized on the resolved path, because an MS is a directory of many
    thousands of small files and this walk+stat runs per unwired boundary
    input, per step, per run -- a caracal-shaped recipe pointing most of its
    steps at one MS pays it over and over for an answer that has usually not
    changed.

    "Usually" is the whole difficulty, so the memo is **not** run-scoped.
    An unwired boundary path is exactly the kind a step can mutate in place,
    and a mutating step contributes no memo entry of its own (a path that is
    both input and output is dropped from the key entirely -- see
    `compute_cache_key`). So a reader before the mutation, a mutation, and a
    reader after it would otherwise serve the second reader the *pre*-mutation
    hash: an identical key, a false cache hit, and a silently skipped step.
    That is the failure the wired/unwired split exists to prevent, so the
    memo must not reintroduce it.

    The cache is therefore cleared at both points where the workspace may
    have moved underneath it (`invalidate_path_hashes`, called from
    `_dispatch`): after any step executes, and again on entry to a top-level
    dispatch, since the memo must not outlive a single run -- two runs
    sharing a process have an arbitrary gap between them that no
    step-completion hook observes.

    What survives is the sharing that is provably safe: several fields of one
    step naming the same path, steps in the same parallel wave, and
    consecutive steps that all hit the cache without running. A long chain of
    mutating steps still re-walks between each one, because there the answer
    genuinely did change.

    What remains outside the memo's guarantee is what was already outside
    mtime's: a *concurrent external writer* touching a boundary path mid-run,
    between two steps neither of which executed. The window is one run and
    the scheme is already racy against that writer.
    """
    key = path.resolve()
    with _path_hash_lock:
        if key in _path_hash_cache:
            return _path_hash_cache[key]

    result = _walk_fingerprint(path)

    with _path_hash_lock:
        _path_hash_cache[key] = result
    return result


def _identity(scope: Scope, func: Callable | None) -> Any:
    """The non-parameter part of a step's cache key: what tool/code is
    actually being run. A step with its own orchestration function
    (a `@shinobi.pystep`'s bare `Scope`, or any `@shinobi.step`-wrapped
    scope with a custom `func`) is keyed by that function's own source --
    editing the function's implementation correctly invalidates every
    cache entry that used it. A plain `Cab` (`func is None`) is keyed by
    its `command`/`flavour`.

    `@shinobi.pystep`'s own `func` is a generic adapter closure (defined
    once in `steps/pyfunc.py`) wrapping the actual decorated function --
    every pystep's adapter has identical source text, so `getsource`
    would be useless for distinguishing them without unwrapping through
    the adapter's `__wrapped__` pointer first (the standard convention,
    set by `pyfunc.py`'s own decorator).
    """
    if func is not None:
        real_func = inspect.unwrap(func)
        try:
            source = inspect.getsource(real_func)
        except (OSError, TypeError):
            source = repr(real_func)
        return ["func", source]
    if isinstance(scope, Cab):
        return ["cab", scope.command, scope.flavour]
    raise TypeError(f"no cacheable identity for a bare Scope with no func ({scope.name!r})")


class ProvenanceKey(str):
    """A provenance key that also remembers *which output field of the
    producing step* it names.

    The key alone answers "which state of this path", which is all the
    skip cache ever needed. Snapshotting needs one thing more: a step with
    two mutated outputs produces two states in one run, and both resolve
    to that step's single `cache_key` (see `StepResult.provenance_key`), so
    the key by itself cannot tell them apart. `(key, producer field)` can.

    It is a `str` subclass rather than a pair because these values are
    hashed into cache keys, in `__upstream__` and through `combine_keys`,
    and any change to their *shape* would rewrite every key in every
    existing manifest. `json.dumps` serializes a `str` subclass as the
    plain string -- as value, as list element, and as dict key, under
    `sort_keys` and `default=str` alike -- and equality, ordering and
    hashing against plain `str` are inherited unchanged. So the field
    rides along in memory and vanishes at the point of hashing, which is
    what lets this land without invalidating anything.

    `producer_field` is a *class* attribute as well as an instance one:
    `copy.deepcopy` and `pickle` reconstruct a `str` subclass through
    `str.__new__` and then restore state, so an instance can briefly exist
    without it, and reading it must not raise.
    """

    producer_field: str | None = None

    def __new__(cls, value: str, producer_field: str | None = None) -> "ProvenanceKey":
        """Build a key naming `value` as produced by output `producer_field`.

        Args:
            value: The producing step's cache key.
            producer_field: The producing step's output field this key
                names, or `None` when it isn't known.
        """
        obj = str.__new__(cls, value)
        obj.producer_field = producer_field
        return obj


def as_provenance_key(key: Any, producer_field: str | None) -> Any:
    """Attach `producer_field` to `key`, unless it already names one.

    Already-wrapped keys pass through *unchanged*, and that is the whole
    subtlety. A key crossing a recipe boundary is the producing leaf's
    key, and the field it names is the leaf's own output field -- not the
    name the enclosing recipe re-exports it under, nor the name the
    consumer wires it into. Re-wrapping at each boundary would rename the
    state at every hop and break the one-state-one-name invariant. `None`
    (no provenance) passes through as `None`.
    """
    if key is None or isinstance(key, ProvenanceKey):
        return key
    return ProvenanceKey(key, producer_field)


def combine_keys(keys: list[Any]) -> str | None:
    """One key standing for a list of them, or `None` if none of them
    carried provenance. Used to give a scattered step -- N independently
    keyed slices, gathered into one `StepResult` -- a single key its
    dependents can key off.
    """
    if not any(key is not None for key in keys):
        return None
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def compute_cache_key(scope: Scope, func: Callable | None, prepared: dict[str, Any], input_keys: dict[str, Any] | None = None) -> str:
    """Hashes `(scope.image, _identity(scope, func), canonicalized
    prepared params, upstream provenance)`.

    `input_keys` maps an input field name to the cache key of the step that
    produced it (or a list of them, for a field wired from several
    sources); a field with no known producer is simply absent. It decides
    which of the two identification regimes each path input falls into (see
    the module docstring):

    - **present** -- the field is identified by its producer's key, carried
      in the `__upstream__` part. Its bytes are never read: for a path
      several steps rewrite in turn, the producer's key is the only thing
      that says *which* state of it this step consumed.
    - **absent** -- an unwired boundary path, keyed by its path *and* its
      content (`_hash_path`, mtime+size). Unless it is also mutated in place
      by this step, in which case it is dropped from the key altogether: a
      step mutating a boundary path would otherwise never look unchanged.

    A boundary path contributes its path string as well as its content
    hash. Content alone is not an identity: `cp -a`/`rsync -a`/`tar -x` all
    preserve mtimes, so two identically-laid-out copies of one MS hashed to
    the same value and a step repointed from one to the other saw a cache
    hit; and two *different* absent paths both hash to `None`, so swapping a
    step onto a different missing file was invisible. The wired branch below
    always kept the path repr for the same reason -- this is that rule
    applied consistently.

    "Mutated in place" means either spelling: the field is also declared on
    `outputs_model`, or its input is declared `Mutability.MUTABLE`. The
    second is how a cab that rewrites its input without re-declaring it as
    an output says so (the flag/gaincal/applycal shape -- see
    `Scope.mutability_of` and `steps.schema.Mutability`), and keying such a
    step on the content it is about to overwrite made it re-run on every
    resumed run, forever.

    Provenance is one part at the end rather than per-field alongside the
    params, so a step with no wired inputs keys exactly as it did before
    provenance existed and its cache entries survive the upgrade.
    """
    input_paths = path_fields(scope.inputs_model)
    mutated_paths = mutated_path_fields(scope)
    wired = set(input_keys or ())

    parts: list[Any] = [scope.image, _identity(scope, func)]
    # Appended conditionally rather than seeded into `parts`: a venv-less
    # scope keys byte-identically to before this field existed, so its cache
    # entries survive the upgrade (same reasoning as `__upstream__` below).
    # The *declared* venv string keys the step, not its resolved freeze hash
    # -- mirroring the image-tag stance in this module's docstring.
    if scope.venv:
        parts.append(["__venv__", scope.venv])
    # `Scope.resources` is deliberately NOT keyed, even though `venv` above
    # is and the two look alike. A venv changes *which software runs*; a
    # resource declaration only changes how the scheduler and the container
    # runtime constrain the very same command -- the same category as
    # `max_workers`, which is likewise not part of a step's identity.
    # Re-running a step because someone re-tuned its memory declaration
    # would be a false invalidation, not a correctness win.
    for name in sorted(prepared):
        value = prepared[name]
        if name in input_paths and name not in mutated_paths and name not in wired and value is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            parts.append([name, repr(value), [_hash_path(Path(v)) for v in values]])
        else:
            # A wired path still contributes its *value* here (the path
            # string), which the `__upstream__` part below does not cover --
            # rewiring a step to a producer that happens to share a cache key
            # would otherwise be invisible.
            parts.append([name, repr(value)])
    if input_keys:
        parts.append(["__upstream__", [[name, input_keys[name]] for name in sorted(input_keys)]])

    blob = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class _JsonFileStore:
    """One JSON object in one file, read and written under a
    `threading.Lock`, with writes going to a temp file that is then renamed
    over the target.

    Shared by `CacheManifest` and the mutation-chain journal
    (`shinobi.snapshots`) rather than reimplemented in each: they have the
    same shape, the same concurrency story, and the same atomicity
    requirement, and two private copies of "atomic JSON store" is exactly
    the kind of drift this repo's DRY discipline exists to prevent.

    The lock is cheap and sufficient *within* a process -- `_run_recipe`'s
    concurrency is a `ThreadPoolExecutor`. Two separate *processes* sharing
    one file remain unguarded, a known limitation inherited by both users.
    """

    def __init__(self, path: Path):
        """Initialize the store, backed by a JSON file at `path`.

        Args:
            path: Path to the JSON file. Not read until first use; created
                (with parent directories) on first write.
        """
        self._path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write_atomic(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._path)


class CacheManifest(_JsonFileStore):
    """A JSON-backed `{step_path: {cache_key, outputs}}` store -- see
    `_JsonFileStore` for the locking and atomicity it inherits.
    """

    def check(self, step_path: str, cache_key: str, scope: Scope, prepared: dict[str, Any]) -> StepResult | None:
        """`None` on any kind of miss (no entry, key mismatch, or a
        declared output path that no longer exists on disk) -- otherwise
        a synthesized `StepResult(cached=True)` restored from the
        manifest's persisted outputs.
        """
        with self._lock:
            entry = self._read().get(step_path)
        if entry is None or entry["cache_key"] != cache_key:
            return None

        for field in path_fields(scope.outputs_model):
            value = entry["outputs"].get(field)
            # `path_fields` unwraps `list[Path]` too, so a declared output can
            # be a list of paths -- `Path(a_list)` would raise TypeError, and
            # this is only reached on a key *match*, which provenance keying
            # makes far more common than it used to be.
            for one in value if isinstance(value, list) else [value]:
                if one and not Path(one).exists():
                    return None

        outputs = scope.outputs_model(**entry["outputs"])
        inputs = scope.inputs_model(**prepared)
        # Restore provenance too (missing on entries written by older
        # versions -- `.get` defaults keep those readable), so a cached step
        # carries the same kind/backend/image/digest into the run manifest as
        # a freshly-run one and doesn't spuriously mark the run unpinned.
        return StepResult(
            name=scope.name,
            returncode=0,
            outputs=outputs,
            inputs=inputs,
            stdout="",
            stderr="",
            cached=True,
            kind=entry.get("kind", "cab"),
            backend=entry.get("backend"),
            image=entry.get("image"),
            image_digest=entry.get("image_digest"),
            containerized=entry.get("containerized", False),
            # Restore venv provenance too -- else a cache hit on a venv step
            # comes back with venv=None and the manifest would read it as a
            # plain native step (pinned), laundering an unverified environment.
            venv=entry.get("venv"),
            venv_digest=entry.get("venv_digest"),
            sandboxed=entry.get("sandboxed", False),
        )

    def entry(self, step_path: str) -> dict[str, Any] | None:
        """The raw persisted entry for `step_path`, or `None`.

        Read directly, without `check`'s key comparison or outputs-exist
        test, because `shinobi.snapshots` needs to ask a different question:
        not "may this step be skipped" but "did *this run* finish and record
        it". See `record` for why `run_id` is part of the answer.
        """
        with self._lock:
            return self._read().get(step_path)

    def record(self, step_path: str, cache_key: str, result, run_id: str | None = None) -> None:
        """Persist the *whole* outputs model (not just path-valued
        fields) -- a downstream step wired to a non-path (e.g. wrangled)
        output of a cached step still needs a real value on a later hit --
        plus the step's provenance (kind/backend/image/digest) so a later
        cache hit can reconstruct a manifest-complete `StepResult`.

        `run_id` identifies the top-level dispatch that recorded the entry.
        The skip cache does not read it; crash recovery does, and needs it
        to be right. There is exactly one entry per step path, so a key
        match alone says only that *some* run of this step with this key
        once succeeded -- and a step that re-ran (because another of its
        declared outputs was deleted) and was then killed mid-mutation would
        find its own entry from the earlier run and conclude it had
        finished, leaving a half-written MS under a cache key that hits
        forever. Comparing the run makes the oracle answer the question
        actually being asked. Entries written before this field existed have
        no `run_id`, so they compare unequal and recovery takes the
        conservative branch -- the safe direction.
        """
        with self._lock:
            data = self._read()
            data[step_path] = {
                "cache_key": cache_key,
                "run_id": run_id,
                "outputs": json.loads(result.outputs.model_dump_json()),
                "kind": result.kind,
                "backend": result.backend,
                "image": result.image,
                "image_digest": result.image_digest,
                "containerized": result.containerized,
                "venv": result.venv,
                "venv_digest": result.venv_digest,
                "sandboxed": result.sandboxed,
            }
            self._write_atomic(data)


_manifests: dict[str, CacheManifest] = {}
_manifests_lock = threading.Lock()


def get_cache_manifest(cache_dir: str) -> CacheManifest:
    """One `CacheManifest` instance (and its lock) per resolved
    `cache_dir`, reused across calls within a process -- distinct
    `CacheManifest` objects for the same file would each have their own
    lock, defeating the thread-safety guarantee.
    """
    path = Path(cache_dir) / "manifest.json"
    key = str(path.resolve())
    with _manifests_lock:
        if key not in _manifests:
            _manifests[key] = CacheManifest(path)
        return _manifests[key]
