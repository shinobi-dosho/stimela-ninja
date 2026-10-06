"""Per-step sandbox execution: run a tool with its cwd inside a private
scratch directory, then move only *declared* outputs back to the workspace
and delete everything else -- so auxiliary droppings (tool logfiles,
``*.last`` files, scratch products) never land in the user's cwd.

This is an allowlist, not a blocklist: what survives is exactly the step's
declared path-typed output fields (after ``implicit`` template resolution)
plus any ``Scope.harvest`` globs (the explicit declaration for dynamically-
named output families that can't be enumerated as literal fields). An
undeclared output simply doesn't survive -- "fully-defined I/O" enforced by
construction rather than by a validator.

The same declarations drive setup: declared output parents and the literal
directory prefixes of harvest and scratch patterns are pre-created before
subprocess runs, including absolute destinations and unsandboxed execution
(`prepare_output_parents`), because tools generally don't ``mkdir -p``
their own output stems and would otherwise crash on e.g. ``plots/gain.html``.

The ones the tool never used are removed again before harvesting
(`prune_unused_parents`), preserving harvest's invariant that everything
present in the sandbox was written by the tool.

``Scope.scratch`` is the deliberate asymmetry in that pair: it declares a
write target that is *not* a product -- a cache tree, a scratch/wisdom
directory, a tool logfile -- so it is pre-created here (and bind-mounted by
the container backends) exactly like an output's directory, and then *not*
harvested. Under a sandbox it is therefore written and then swept with
everything else. Without it the two properties were welded together and a cab
had to choose: declare a cache as an output and drag it into the caller's
workspace on every run, or leave it undeclared and have the tool write into
the container (discarded on ``docker run --rm``, a hard failure on
apptainer's read-only image).

Boundaries of the mechanism, by design:

* Inputs are never copied in. Path-typed inputs are rewritten to absolute
  paths anchored at the workspace (`absolutize_path_inputs`), so the tool
  reads -- and, for MUTABLE inputs like an MS, writes -- the caller's real
  files in place. A tool that drops junk *next to an input* therefore
  writes into the workspace; the sandbox can't catch that.
* Absolute-path outputs bypass the sandbox entirely (the tool writes them
  straight to their declared destination); harvest skips them. What harvest
  gives a relative output -- the previous run's product *replaced* rather
  than written over -- these get from `clear_stale_outputs` instead, which
  runs before the tool and deletes exactly the declared destinations the
  tool is about to write directly. It is also what gives an unsandboxed run
  that guarantee at all, since there the same is true of every output.
* Harvest moves by `os.replace`/rename, so the sandbox root must live on
  the same filesystem as the workspace (`AppConfig.sandbox.dir` is
  workspace-relative for exactly this reason). Directory moves fall back
  to `shutil.move` which copies across filesystems -- correct but slow, so
  don't point the root elsewhere for huge products.
* On failure the sandbox is deliberately *kept* (and its path reported)
  for post-mortem; nothing is harvested.
* Only subprocess-backed runs can be sandboxed (the backend gets a
  per-run ``cwd``). In-process pysteps are exempt: ``os.chdir`` is
  process-global and recipes run steps on a thread pool.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import warnings
from glob import has_magic
from pathlib import Path
from typing import Any, Sequence

from shinobi.exceptions import ParameterError, StepError
from shinobi.steps.schema import (
    Scope,
    _product_pattern_directory,
    declared_output_dirs,
    declared_output_paths,
    path_fields,
    output_path_values,
    paths_overlap,
    path_input_modes,
    validate_declared_writes,
    write_path_fields,
)

logger = logging.getLogger(__name__)


def create_sandbox(root: str, label: str, *, dataset_resources: set[Path] | None = None) -> Path:
    """Create (and return, resolved absolute) a fresh per-step sandbox
    directory under `root`, named after `label` plus a unique suffix.
    `root` is created on demand; a relative `root` is anchored at the cwd,
    which keeps it on the workspace's filesystem so harvest can rename.
    """
    root_path = Path(root)
    if dataset_resources:
        validate_sandbox_location(root_path, dataset_resources, fresh=False)
    root_path.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_") or "step"
    sandbox = Path(tempfile.mkdtemp(prefix=f"{safe_label}-", dir=root_path)).resolve()
    if dataset_resources:
        validate_sandbox_location(sandbox, dataset_resources, fresh=True)
    return sandbox


def validate_sandbox_location(path: Path, resources: set[Path], *, fresh: bool) -> None:
    """Refuse sandbox filesystem work within a protected dataset closure."""
    from shinobi.exceptions import DatasetLifecycleUnavailableError

    try:
        canonical = path.resolve()
        for resource in resources:
            protected = resource.resolve()
            if canonical == protected or canonical.is_relative_to(protected) or (protected.is_relative_to(canonical) and (fresh or not canonical.is_dir())):
                raise DatasetLifecycleUnavailableError(
                    f"contained MSv2 execution refused: sandbox {'directory' if fresh else 'root'} {path} overlaps the MSv2 closure: {protected}"
                )
    except (OSError, RuntimeError) as exc:
        raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: cannot inspect sandbox location {path}: {exc}") from exc


def prepare_output_parents(scope: Scope, prepared: dict[str, Any], sandbox_dir: Path) -> list[Path]:
    """Create declared write directories relative to the execution cwd.

    ``sandbox_dir`` retains its original keyword name, but may also be the
    workspace for an unsandboxed run. Absolute declarations are prepared at
    their destinations. Products and existing parents are left untouched.
    Returns exactly the newly created directories, including intermediates;
    only entries canonically inside a sandbox should be passed to pruning.
    """
    validate_declared_writes(scope, prepared, sandbox_dir)
    dirs = {d if d.is_absolute() else sandbox_dir / d for d, _ in declared_output_dirs(scope, prepared)}
    created: list[Path] = []
    for directory in sorted(dirs):
        missing: list[Path] = []
        path = directory
        while not path.is_dir():
            missing.append(path)
            if path == path.parent:
                break
            path = path.parent
        for path in reversed(missing):
            try:
                path.mkdir()
            except FileExistsError:
                if not path.is_dir():
                    raise
            else:
                created.append(path)
    return created


def observe_product_path(path: Path, *, recursive: bool = False):
    """Filesystem write-attribution evidence, without following directory symlinks.

    Schema-only setup directories need shallow observations. Only direct
    harvest directory matches request descendants; these observations do not
    impose a recursive cache-hit existence obligation.
    """
    info = path.lstat()
    own = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if recursive and path.is_dir() and not path.is_symlink():
        return own, tuple((child.name, observe_product_path(child, recursive=True)) for child in sorted(path.iterdir()))
    return own


def observe_prepared_parents(created: Sequence[Path]) -> dict[Path, Any]:
    """Record shallow setup identities after all parents have been prepared."""
    observations = {}
    for path in created:
        try:
            observations[path] = observe_product_path(path)
        except OSError:
            # Missing cache/pruning evidence cannot revoke a tool's success.
            # Unknown identities use only best-effort empty-directory pruning.
            observations[path] = None
    return observations


def prune_unused_parents(created: list[Path], observations: dict[Path, Any] | None = None) -> None:
    """Remove unchanged empty setup directories, deepest first.

    Production launchers pass pre-launch observations so recreated or modified
    empty directories with known identities are tool products and survive.
    Unknown identities retain historical best-effort empty-directory removal,
    preventing library scaffolding from replacing workspace data. Select unchanged paths
    before deleting any child, since deletion changes its parent's timestamps.
    The one-argument form retains historical empty-directory pruning for callers
    which do not retain pre-launch observations.
    """
    unchanged = []
    for path in created:
        try:
            if observations is None or observations.get(path) is None or observe_product_path(path) == observations[path]:
                unchanged.append(path)
        except OSError:
            if observations is not None:
                observations[path] = None
            unchanged.append(path)
    for path in sorted(unchanged, key=lambda path: len(path.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _anchor(value: Any, workspace: Path) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(_anchor(item, workspace) for item in value)
    path = Path(str(value))
    return value if path.is_absolute() else workspace / path


def path_input_names(scope: Scope, prepared: dict[str, Any]) -> set[str]:
    """Present path-typed inputs, using shared schema classification."""
    return set(path_input_modes(scope, prepared))


def absolutize_path_inputs(scope: Scope, prepared: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """A copy of `prepared` with every relative path-typed input value
    (`path_input_names`) anchored at `workspace`, so the tool still finds
    (and mutates in place) the caller's real files when its cwd is the
    sandbox. Non-path values pass through untouched -- notably, a
    *string*-typed output-prefix input stays relative, so the tool writes
    that output family inside the sandbox for harvest to pick up.
    """
    anchored = dict(prepared)
    for name in path_input_names(scope, prepared):
        anchored[name] = _anchor(prepared[name], workspace)
    return anchored


def _input_paths_to_keep(scope: Scope, run_inputs: dict[str, Any]) -> list[Path]:
    """Resolved values of every path input that carries data *in* -- the
    paths `clear_stale_outputs` must never delete.

    That is every `path_input_names` value except the ones the scope
    declares as write targets (`schema.write_path_fields`). The exclusion is
    the whole point: an output field that echoes a same-named input is
    written the same way whether the tool created that path
    (``mstransform``'s ``outputvis``) or rewrote the caller's data in place
    (``flagdata``'s ``vis``), and only the declaration tells them apart.
    Unmarked means "caller's data", so a cab that says nothing keeps
    today's behaviour and its inputs.
    """
    keep: list[Path] = []
    writes = write_path_fields(scope)
    for name in path_input_names(scope, run_inputs) - writes:
        value = run_inputs[name]
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if item is not None:
                keep.append(Path(str(item)).resolve())
    return keep


def clear_stale_outputs(scope: Scope, run_inputs: dict[str, Any], workspace: Path, *, sandboxed: bool) -> list[Path]:
    """Delete the previous run's product from each declared output path the
    tool is about to write **directly**, before it starts. Returns what was
    removed.

    Re-running a step is supposed to replace the previous run's products,
    and for an output the tool writes inside a sandbox that is exactly what
    happens: fresh scratch dir, tool writes, `harvest_outputs` moves the new
    product over the destination (`_move`). An output the tool writes
    straight to its destination gets none of that -- it lands on top of
    whatever the last run left. Tools that refuse to overwrite then fail the
    step (CASA is a whole family: ``mstransform``, ``split``, ``importuvfits``
    all check the output for existence first), and tools that append or merge
    silently produce a corrupt product, which is the worse half. This closes
    that gap so both kinds of output get the same guarantee.

    "Directly" is decided from the value the *tool* receives (`run_inputs`,
    i.e. post-`absolutize_path_inputs`), not from the caller's spelling: a
    relative output under a sandbox is written inside the scratch dir and is
    left alone here, and everything else -- an absolute path, a path-typed
    input anchored into one, any output at all when unsandboxed -- resolves
    to its real destination. Note that a *path*-typed input naming a
    destination is anchored, so it is in the second group even when the
    caller spelled it relative and a sandbox is on; only a string-typed
    stem's products stay behind in the scratch dir for harvest.

    Two things are never cleared, and both are load-bearing:

    * **A path the step reads** (`_input_paths_to_keep`), compared resolved
      and by containment (`schema.paths_overlap`), so the in-place-mutation
      idiom is safe: ``flagdata(vis=...) -> vis`` declares one path on both
      models, and that "output" is the caller's MS, not this step's product.
      Deleting it would destroy data mid-pipeline. An MS *is* a directory,
      so containment matters -- an output resolving inside a declared input
      is that input's data too.
    * **`harvest`/`scratch` glob matches.** Only declared output *fields*
      are cleared (`schema.declared_output_paths`). A glob's matches are
      named by the tool at run time, so they can collide with workspace data
      this step knows nothing about -- the same reason `_move` refuses to
      replace an undeclared directory rather than rmtree it silently.

    Every removal is logged at INFO naming the path and the declaration it
    came from, because deleting a declared *directory* (an MS, an image
    tree) is a real deletion and not the ordinary file overwrite `_move`
    performs. `AppConfig.execution.clear_stale_outputs` turns the whole
    thing off.

    The cost, stated plainly: a re-run that then *fails* leaves neither the
    new product nor the old one, where a sandboxed relative output would
    still have the old one (nothing is harvested over a failure). That is
    unavoidable rather than a choice -- a tool that refuses to overwrite has
    already failed by the time anything could harvest, so the deletion has to
    happen before it starts -- and it is exactly what a per-cab `overwrite`
    flag does. With Tier 1 snapshots on (`shinobi.snapshots`) the marker left
    by the failed run restores it on the next one.
    """
    # Nothing declared, nothing to clear -- and no `resolve()`/`stat` calls
    # for a step that declares no path outputs at all, which is the common
    # case and is on the critical path of every single run.
    candidates = [(path, source) for path, source in declared_output_paths(scope, run_inputs) if path.is_absolute() or not sandboxed]
    if not candidates:
        return []
    keep = _input_paths_to_keep(scope, run_inputs)
    workspace = workspace.resolve()
    removed: list[Path] = []
    for path, source in candidates:
        dst = path if path.is_absolute() else workspace / path
        if not dst.exists() and not dst.is_symlink():
            continue
        # Resolve for comparison only -- removal below acts on `dst` itself,
        # so a symlinked destination loses the link, never the target.
        resolved = dst.resolve()
        if any(paths_overlap(resolved, kept) for kept in keep):
            continue
        if workspace.is_relative_to(resolved):
            # The declaration resolved to the workspace itself or an ancestor
            # of it (a mis-templated output, an empty stem). Clearing that is
            # never what was meant, and it would take the run's inputs with it.
            warnings.warn(
                f"'{scope.name}' {source} resolved to {resolved}, which contains the workspace -- not clearing it; the tool will see whatever the previous run left there",
                stacklevel=3,
            )
            continue
        writers, readers = live_holders(resolved)
        if writers:
            # Refuse rather than warn-and-skip. Skipping would hand the tool
            # a destination that already exists, which is the failure this
            # function was written to prevent -- and the *reason* it exists
            # here is not "stale product in the way" but "something is still
            # writing this", which no amount of retrying fixes and which the
            # user has to see to act on.
            raise StepError(
                f"'{scope.name}' would clear the stale {source} at {dst} before re-running, but "
                f"{'a process is' if len(writers) == 1 else 'processes are'} still writing to it: {', '.join(writers)}. "
                f"Deleting a path something is still writing does not stop the writer -- it keeps going into "
                f"files that no longer have names, and the product it leaves is silently incomplete. This is "
                f"usually a run that was interrupted without its container being stopped. Stop the process(es) "
                f"above, then re-run. (Set execution.clear_stale_outputs=false to skip this replacement "
                f"entirely, at the cost of the tool seeing the previous run's product.)"
            )
        if readers:
            # Not fatal: a reader keeps the old inode and is unaffected by the
            # unlink. Said out loud anyway, because "my viewer went blank" is
            # otherwise a mystery.
            logger.warning(
                "step %s: clearing %s at %s while %s reading it -- their view will not update",
                scope.name,
                source,
                dst,
                "a process is" if len(readers) == 1 else "processes are",
            )
        logger.info("step %s: clearing stale %s at %s before re-running", scope.name, source, dst)
        _remove(dst)
        removed.append(dst)
    return removed


def _describe(pid: int) -> str:
    """`<pid> (<command>)`, for naming a process in an error a human has to
    act on. Falls back to the bare pid if the process is already gone."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmdline = ""
    return f"{pid} ({cmdline[:120]})" if cmdline else str(pid)


def _opened_for_writing(fd_link: Path) -> bool:
    """Was this descriptor opened for writing?

    `/proc/<pid>/fd/<n>` is just a symlink and carries no access mode; the
    matching `fdinfo` entry carries the open flags, whose low two bits are
    `O_RDONLY`/`O_WRONLY`/`O_RDWR`. Unreadable fdinfo counts as writing --
    when we cannot tell, the cautious answer is the one that refuses to
    delete.
    """
    try:
        for line in Path(str(fd_link).replace("/fd/", "/fdinfo/")).read_text().splitlines():
            if line.startswith("flags:"):
                return (int(line.split()[1], 8) & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR)
    except (OSError, ValueError, IndexError):
        return True
    return True


def live_holders(path: Path) -> tuple[list[str], list[str]]:
    """Who has `path`, or something under it, open right now -- split into
    writers and readers.

    The guard against a deletion nobody meant to make. Its motivating case
    is real and cost 13 GB: a container orphaned by an earlier interrupt kept
    writing an MS, the next run cleared that "stale" output from under it,
    and the writer carried on into unlinked files -- 13 GB on disk that
    collapsed to 102 MB the moment its file descriptors closed. Nothing about
    that is visible from the path alone; the only evidence is that somebody
    still has it open.

    Writers and readers are separated because only one of them is a problem.
    Deleting under a *reader* is harmless -- it keeps reading the old inode,
    which is ordinary POSIX and was the behaviour before this guard existed --
    and treating the two alike would fail a run because somebody had a
    `casaviewer` on the previous image, a `tail -f` on a log, or a shell
    sitting in the output directory. A `cwd` match counts as reading for the
    same reason: being in a directory is not writing to it.

    Known blind spots, all of which mean this returns *fewer* holders than
    exist rather than more:

    * Only this machine. A step still running on a cluster node through the
      slurm or kubernetes backends holds paths on shared storage that no
      local `/proc` will ever show.
    * Only processes we can read. A rootful docker container's payload runs
      as root by default (`backend.run_as_host_user` controls the `--user`
      flag), and root's `/proc/<pid>/fd` is unreadable to us -- so the
      leaked-docker-container case this guard reads as its motivating story
      is exactly the one it can miss.
    * Only descriptors. casacore memory-maps table files, and a mapping whose
      fd has been closed lives in `/proc/<pid>/maps`, not `fd`.

    Linux-only: `/proc` is how you ask this question, and where it does not
    exist (macOS) both lists come back empty -- "nothing known to be using
    it", the pre-guard behaviour.

    Args:
        path: A resolved path (file or directory) about to be deleted.

    Returns:
        `(writers, readers)`, each `"<pid> (<command>)"`, deduplicated by pid.
        A pid appears in `writers` if *any* of its matching descriptors is
        writable.
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return [], []
    target = str(path)
    prefix = target + os.sep
    me = os.getpid()
    writers: list[str] = []
    readers: list[str] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        matched = writing = False
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            fds = []  # unreadable (another user) or exited between listing and now
        for link in fds:
            try:
                dest = os.readlink(link)
            except OSError:
                continue
            # NFS silly-renames a deleted-but-open file to `.nfsXXXX` in its
            # own directory, and Linux appends " (deleted)" to the link; both
            # still name a live writer inside the tree, which is precisely
            # the state the motivating case was found in.
            if dest == target or dest.startswith(prefix):
                matched = True
                if _opened_for_writing(link):
                    writing = True
                    break
        if not matched:
            try:
                dest = os.readlink(entry / "cwd")
            except OSError:
                continue
            matched = dest == target or dest.startswith(prefix)
        if matched:
            (writers if writing else readers).append(_describe(pid))
    return writers, readers


def _relativize(value: Any, workspace: Path) -> Any:
    """Convert an absolute path value to workspace-relative, if applicable.
    Handles single paths and lists/tuples of paths. Non-path values and
    paths outside the workspace pass through unchanged."""
    if isinstance(value, (list, tuple)):
        return type(value)(_relativize(item, workspace) for item in value)
    path = Path(str(value))
    if not path.is_absolute():
        return value
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return value
    return relative


def relativize_path_outputs(scope: Scope, outputs: Any, workspace: Path) -> Any:
    """A copy of `outputs` with absolute path-typed output values converted
    to workspace-relative paths. Inverse of `absolutize_path_inputs` --
    ensures cache entries use consistent relative paths regardless of
    whether the step ran sandboxed (where inputs were anchored absolute)
    or unsandboxed (where they stayed relative). A path outside the
    workspace (e.g. an absolute output the caller explicitly requested)
    passes through unchanged.
    """
    declared = path_fields(scope.outputs_model)
    values: dict[str, Any] = {}
    changed = False
    for name in scope.outputs_model.model_fields:
        value = getattr(outputs, name, None)
        if value is None or name not in declared:
            values[name] = value
            continue
        relativized = _relativize(value, workspace)
        values[name] = relativized
        if relativized is not value:
            changed = True
    if not changed:
        return outputs
    return scope.outputs_model(**values)


def _relative_targets(scope: Scope, outputs: Any, prepared: dict[str, Any], sandbox_dir: Path) -> dict[str, bool]:
    """The sandbox-relative paths harvest should rescue, each mapped to
    whether it is **declared**: a path-typed output field value (absolute
    ones already live at their destination and are skipped) is declared;
    a `scope.harvest` glob match is not -- its name was chosen by the tool
    at run time, not by the schema.

    That distinction is what `_move` uses to decide how destructive it is
    allowed to be at the destination. A path reached both ways counts as
    declared.
    """
    targets: dict[str, bool] = {}
    for _name, path, _required in output_path_values(scope, outputs):
        if not path.is_absolute():
            targets[str(path)] = True
    for match in _harvest_matches(scope, prepared, sandbox_dir, absolute=False, relative=True, reject_escape=True):
        targets.setdefault(str(match.relative_to(sandbox_dir)), False)
    return targets


def _harvest_matches(scope: Scope, prepared: dict[str, Any], root: Path, *, absolute: bool, relative: bool, reject_escape: bool = False, observe: bool = False):
    """Expand harvest declarations for rescue and direct-write observation."""
    for pattern in scope.harvest:
        try:
            resolved = pattern.format(**prepared)
        except KeyError as exc:
            raise ParameterError(f"'{scope.name}' harvest pattern {pattern!r} references unknown input {exc}") from exc
        path = Path(resolved)
        if reject_escape and not path.is_absolute() and ".." in path.parts:
            escaped = (root / resolved).resolve()
            warnings.warn(
                f"'{scope.name}' harvest pattern {pattern!r} resolved to {resolved!r} (-> {escaped}), "
                "which escapes the sandbox -- skipped; any matching files were left outside the sandbox",
                stacklevel=3,
            )
            continue
        if path.is_absolute():
            if absolute:
                match_root, glob = Path(path.anchor), str(path.relative_to(path.anchor))
            else:
                continue
        elif relative:
            match_root, glob = root, resolved
        else:
            continue
        if observe and not has_magic(glob):
            # Glob suppresses some stat errors even for a literal match. Let
            # direct observation test the exact path, without parent listing.
            yield match_root / glob
            continue
        if observe and not _harvest_parent_readable(match_root, glob):
            raise OSError("directory-wildcard harvest enumeration cannot prove complete cache evidence")
        yield from match_root.glob(glob)


def _harvest_parent_readable(root: Path, pattern: str) -> bool:
    """Preflight direct cache evidence before glob can hide filesystem errors.

    Basename globs need one explicit literal-parent directory enumeration.
    Directory wildcards and recursive globs conservatively remain unknown when
    their literal prefix exists; proving every branch would need another walker.
    Missing literal prefixes cannot contain matches and are trustworthy empty.
    """
    parts = Path(pattern).parts
    if not any(has_magic(part) for part in parts):
        # Literal matches can be stated directly without listing their parent.
        return True
    prefix = root / _product_pattern_directory(pattern)
    directory_wildcard = any(has_magic(part) for part in parts[:-1])
    try:
        with os.scandir(prefix) as entries:
            for _entry in entries:
                pass
    except (FileNotFoundError, NotADirectoryError):
        return True
    return not directory_wildcard and "**" not in parts


def _move(src: Path, dst: Path, declared: bool) -> None:
    """Move `src` over `dst`, replacing what's there -- the same overwrite
    the tool itself would have done had it run in the workspace directly.

    Overwriting a *file* is that ordinary overwrite, and re-running a step
    is supposed to replace the previous run's products. Replacing a
    *directory* means `rmtree`, which is not an overwrite but a deletion of
    everything underneath -- and for an undeclared (`scope.harvest`
    glob-matched) target the colliding name was chosen by the tool at run
    time, so it may name workspace data this step knows nothing about (an
    MS, a directory of unrelated products). Rather than destroy it silently,
    refuse: the run stops with both paths named, and the user either
    declares the output or moves the directory aside.

    The asymmetry is deliberate: an *undeclared* collision with an existing
    **file** still overwrites. Only directories are refused. A stray file
    (a leftover log, a previous run's plot) is cheap to lose and cheap to
    regenerate, whereas failing a long pipeline run because one such file
    happened to share a harvested name would cost far more than it protects.
    The line is drawn at "deleting a tree of data the step never mentioned",
    which is the case that is expensive and irreversible.

    Raises:
        StepError: If `dst` is an existing directory and `declared` is False.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir() and not dst.is_symlink() and not declared:
        raise StepError(
            f"harvest would replace the directory '{dst}', which this step never declared as an "
            f"output -- it matched a harvest glob, so the name came from the tool, not the schema. "
            f"Refusing to delete it. Declare it as an output field if it really is this step's "
            f"product, or move the existing directory aside."
        )
    if dst.exists() or dst.is_symlink():
        _remove(dst)
    shutil.move(str(src), str(dst))


def _remove(path: Path) -> None:
    """Delete `path`, whatever it is. A real directory goes with its whole
    tree; a symlink (even one pointing at a directory) is unlinked, so the
    link goes and its target does not. Shared by `_move`'s replace-the-
    destination step and `clear_stale_outputs`, which must agree on what
    "the previous product is gone" means for a directory-shaped product
    like an MS.
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def harvest_outputs(scope: Scope, outputs: Any, prepared: dict[str, Any], sandbox_dir: Path, workspace: Path, *, _capture: ProductCapture | None = None) -> list[Path]:
    """Move the step's declared outputs from `sandbox_dir` to `workspace`,
    preserving their relative paths, and return the workspace-side paths
    that were moved. A declared output the tool never wrote (e.g. an
    optional product, or a same-named input passthrough that already lives
    in the workspace) is silently skipped.

    Targets move parent-first: one nested inside a directory-valued target
    travels with its parent's move and is then skipped as no-longer-present.
    Child-first order would move the child, then `_move` the parent dir over
    the same destination -- rmtree-ing the just-harvested child.
    """
    moved: list[Path] = []
    targets = _relative_targets(scope, outputs, prepared, sandbox_dir)
    candidates = ()
    if _capture is not None:
        try:
            candidates = {rel for rel in targets if (sandbox_dir / rel).exists()}
        except OSError:
            _capture.complete = False
            # Actual harvest checks/moves still run and retain their failures.

    for rel in sorted(targets, key=lambda rel: Path(rel).parts):
        src = sandbox_dir / rel
        if not src.exists() and not src.is_symlink():
            continue
        dst = workspace / rel
        _move(src, dst, targets[rel])
        moved.append(dst)
    if _capture is not None:
        # Freeze selected children before parent moves, then publish the whole
        # inventory only after every move succeeded. Each candidate is visited
        # once, including children that travelled with their declared parent.
        _capture.harvested.update(workspace / candidate for candidate in candidates)
    return moved


def discard_sandbox(sandbox_dir: Path) -> None:
    """Delete the sandbox directory and whatever junk is left in it.
    Best-effort: a straggler open file must not fail the step.
    """
    shutil.rmtree(sandbox_dir, ignore_errors=True)


class ProductCapture:
    """Freeze concrete products of one successful execution, independently of outputs.

    Direct harvest matches are compared with a pre-execution observation. Only
    harvest directories recurse: declared directories require their root alone.
    Observations distinguish writes, not scientific identity or timestamp age.
    """

    def __init__(
        self,
        scope: Scope,
        prepared: dict[str, Any],
        workspace: Path,
        sandbox_dir: Path | None = None,
        *,
        created_dirs: Sequence[Path] = (),
        prepared_observations: dict[Path, Any] | None = None,
    ):
        self.scope = scope
        self.prepared = prepared
        self.workspace = workspace
        self.sandbox_dir = sandbox_dir
        self.harvested: set[Path] = set()
        self.before = self._direct_matches()
        self.prepared_dirs = prepared_observations if prepared_observations is not None else observe_prepared_parents(created_dirs)
        self.complete = self.before is not None and all(observation is not None for observation in self.prepared_dirs.values())
        self.passthroughs = {path if path.is_absolute() else workspace / path for _name, path, _required in output_path_values(scope, prepared)}

    def _direct_matches(self) -> dict[Path, Any] | None:
        matches = {}
        try:
            for match in _harvest_matches(self.scope, self.prepared, self.workspace, absolute=True, relative=self.sandbox_dir is None, observe=True):
                if match.exists():
                    matches[match] = observe_product_path(match, recursive=True)
        except OSError:
            return None
        return matches

    def finish(self, outputs: Any) -> list[str] | None:
        """Exact products, or None when incomplete evidence requires a cache miss."""
        if not self.complete or any(observation is None for observation in self.prepared_dirs.values()):
            self.complete = False
            return None
        products = set(self.harvested)
        try:
            for _name, path, _required in output_path_values(self.scope, outputs):
                actual = path if path.is_absolute() else self.workspace / path
                if actual.exists():
                    if actual in self.prepared_dirs and self.prepared_dirs[actual] == observe_product_path(actual) and actual.is_dir() and not any(actual.iterdir()):
                        # Compare shallow identity before inspecting contents:
                        # changed opaque directories may be legitimate products.
                        continue
                    # A sandbox destination leftover is not evidence the tool
                    # produced it. Same-named input passthroughs remain valid.
                    if self.sandbox_dir is None or path.is_absolute() or actual in self.harvested or actual in self.passthroughs:
                        products.add(actual)
            after = self._direct_matches()
            if after is None:
                self.complete = False
                return None
            for path, observation in after.items():
                if self.before.get(path) != observation:
                    products.add(path)
        except OSError:
            self.complete = False
            return None
        return sorted(str(path) for path in products)
