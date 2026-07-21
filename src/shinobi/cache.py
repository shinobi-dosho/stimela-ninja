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

import hashlib
import inspect
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from shinobi.results import StepResult
from shinobi.steps.schema import Cab, Scope, path_fields


def _hash_path(path: Path) -> Any:
    """`(relative_path, mtime_ns, size)` for every file under `path` -- a
    single file yields one tuple; a directory (e.g. an MS) yields one per
    file within it, sorted for a deterministic result. `None` if `path`
    doesn't exist (e.g. an optional input the caller didn't supply).
    """
    if not path.exists():
        return None
    if path.is_file():
        st = path.stat()
        return [[".", st.st_mtime_ns, st.st_size]]
    entries: list[list[Any]] = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            fpath = Path(root) / fname
            st = fpath.stat()
            entries.append([str(fpath.relative_to(path)), st.st_mtime_ns, st.st_size])
    return sorted(entries)


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
    - **absent** -- an unwired boundary path, hashed by `_hash_path`
      (mtime+size). Unless it is also a declared *output* of this step, in
      which case it is dropped from the key altogether: a step mutating a
      boundary path in place would otherwise never look unchanged.

    Provenance is one part at the end rather than per-field alongside the
    params, so a step with no wired inputs keys exactly as it did before
    provenance existed and its cache entries survive the upgrade.
    """
    input_paths = path_fields(scope.inputs_model)
    output_paths = path_fields(scope.outputs_model)
    mutated_paths = input_paths & output_paths
    wired = set(input_keys or ())

    parts: list[Any] = [scope.image, _identity(scope, func)]
    for name in sorted(prepared):
        value = prepared[name]
        if name in input_paths and name not in mutated_paths and name not in wired and value is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            parts.append([name, [_hash_path(Path(v)) for v in values]])
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


class CacheManifest:
    """A JSON-backed `{step_path: {cache_key, outputs}}` store. Guards
    reads/writes with a `threading.Lock` (cheap -- one process, `_run_
    recipe`'s own concurrency is a `ThreadPoolExecutor`) and writes via
    write-temp-then-rename for atomicity. Two separate *processes*
    sharing one manifest file remain unguarded -- a known limitation, not
    solved here.
    """

    def __init__(self, path: Path):
        """Initialize the manifest, backed by a JSON file at `path`.

        Args:
            path: Path to the JSON manifest file. Not read until first use;
                created (with parent directories) on first write.
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
            sandboxed=entry.get("sandboxed", False),
        )

    def record(self, step_path: str, cache_key: str, result) -> None:
        """Persist the *whole* outputs model (not just path-valued
        fields) -- a downstream step wired to a non-path (e.g. wrangled)
        output of a cached step still needs a real value on a later hit --
        plus the step's provenance (kind/backend/image/digest) so a later
        cache hit can reconstruct a manifest-complete `StepResult`.
        """
        with self._lock:
            data = self._read()
            data[step_path] = {
                "cache_key": cache_key,
                "outputs": json.loads(result.outputs.model_dump_json()),
                "kind": result.kind,
                "backend": result.backend,
                "image": result.image,
                "image_digest": result.image_digest,
                "containerized": result.containerized,
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
