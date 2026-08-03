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

The same declarations drive setup: parent directories of relative declared
outputs (and the literal directory prefix of harvest patterns) are
pre-created inside the fresh sandbox before the run
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
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

from shinobi.exceptions import ParameterError, StepError
from shinobi.loaders._modelgen import is_file_dtype
from shinobi.steps.schema import (
    Cab,
    Scope,
    declared_output_dirs,
    declared_output_paths,
    path_fields,
    paths_overlap,
    write_path_fields,
)

logger = logging.getLogger(__name__)


def create_sandbox(root: str, label: str) -> Path:
    """Create (and return, resolved absolute) a fresh per-step sandbox
    directory under `root`, named after `label` plus a unique suffix.
    `root` is created on demand; a relative `root` is anchored at the cwd,
    which keeps it on the workspace's filesystem so harvest can rename.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_") or "step"
    return Path(tempfile.mkdtemp(prefix=f"{safe_label}-", dir=root_path)).resolve()


def prepare_output_parents(scope: Scope, prepared: dict[str, Any], sandbox_dir: Path) -> list[Path]:
    """Pre-create, inside the fresh sandbox, the parent directories of every
    *relative* declared output directory (`schema.declared_output_dirs`).
    Tools generally don't ``mkdir -p`` their own output stems (wsclean's
    ``-name``, ragavi's ``htmlname``), so a relative output like
    ``plots/gain.html`` that works in the workspace -- where the caller made
    ``plots/`` -- crashes the tool inside an empty sandbox.

    Absolute declared outputs are skipped: they bypass the sandbox entirely
    (the tool writes them straight to their destination), which is also why
    the container backend takes exactly the half this one drops -- it has to
    bind-mount them for the write to reach the host at all.

    Returns every directory it created, for `prune_unused_parents`: harvest
    assumes anything present in the sandbox was written by the tool, so the
    dirs the tool never used must be removed again before harvesting.
    """
    dirs = {d for d, _ in declared_output_dirs(scope, prepared) if not d.is_absolute()}
    created: list[Path] = []
    for rel in sorted(dirs):
        path = sandbox_dir
        for part in rel.parts:
            path = path / part
            if not path.is_dir():
                path.mkdir()
                created.append(path)
    return created


def prune_unused_parents(created: list[Path]) -> None:
    """Remove the `prepare_output_parents` directories the tool never wrote
    into, deepest first, restoring harvest's invariant that everything
    present in the sandbox was written by the tool -- otherwise a leftover
    empty dir could be rescued over real workspace content (`_move` replaces
    the destination wholesale). A dir the tool did use is non-empty and
    survives the rmdir; tool-created dirs (even empty ones) are untouched
    and harvest exactly as they would have unsandboxed.
    """
    for path in sorted(created, reverse=True):
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
    """Which of `prepared`'s keys are path-typed *inputs*. Same field
    classification as container bind-mounting
    (`backends.container.bind_dir_modes`): declared fields via `path_fields`,
    dynamically pattern-matched Cab inputs via their `ParamMeta.dtype`.

    Factored out because two callers need the identical answer for opposite
    reasons: `absolutize_path_inputs` anchors exactly these at the workspace,
    and `clear_stale_outputs` refuses to delete anything one of them points
    at. A field either side classified differently would be a path the tool
    writes for real and the deleter believes is scratch, or the reverse.
    """
    declared = path_fields(scope.inputs_model)
    match_pattern = scope.match_pattern if isinstance(scope, Cab) else None
    names: set[str] = set()
    for name, value in prepared.items():
        if value is None:
            continue
        if name not in declared:
            if match_pattern is None:
                continue
            meta = match_pattern(name)
            if meta is None or meta.dtype is None or not is_file_dtype(meta.dtype):
                continue
        names.add(name)
    return names


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
        logger.info("step %s: clearing stale %s at %s before re-running", scope.name, source, dst)
        _remove(dst)
        removed.append(dst)
    return removed


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
    for name in sorted(path_fields(scope.outputs_model)):
        value = getattr(outputs, name, None)
        if value is None:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            path = Path(str(item))
            if not path.is_absolute():
                targets[str(path)] = True
    for pattern in scope.harvest:
        try:
            resolved = pattern.format(**prepared)
        except KeyError as exc:
            raise ParameterError(f"'{scope.name}' harvest pattern {pattern!r} references unknown input {exc}") from exc
        # A pattern that *resolves* absolute (e.g. `"{prefix}-*"` with an
        # absolute prefix) is skipped, same as an absolute declared output:
        # the tool wrote those files straight to their absolute destination,
        # so there is nothing inside the sandbox to rescue -- raising here
        # would fail a successful run on ordinary input. A `..` escape can't
        # be harvested either (it points outside the sandbox), but unlike the
        # absolute case the tool's relative writes landed *next to* the
        # sandbox, not at their intended destination -- warn so the stranded
        # files can be found.
        if Path(resolved).is_absolute():
            continue
        if ".." in Path(resolved).parts:
            escaped = (sandbox_dir / resolved).resolve()
            warnings.warn(
                f"'{scope.name}' harvest pattern {pattern!r} resolved to {resolved!r} (-> {escaped}), "
                "which escapes the sandbox -- skipped; any matching files were left outside the sandbox",
                stacklevel=3,
            )
            continue
        for match in sandbox_dir.glob(resolved):
            targets.setdefault(str(match.relative_to(sandbox_dir)), False)
    return targets


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


def harvest_outputs(scope: Scope, outputs: Any, prepared: dict[str, Any], sandbox_dir: Path, workspace: Path) -> list[Path]:
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
    for rel in sorted(targets, key=lambda rel: Path(rel).parts):
        src = sandbox_dir / rel
        if not src.exists() and not src.is_symlink():
            continue
        dst = workspace / rel
        _move(src, dst, targets[rel])
        moved.append(dst)
    return moved


def discard_sandbox(sandbox_dir: Path) -> None:
    """Delete the sandbox directory and whatever junk is left in it.
    Best-effort: a straggler open file must not fail the step.
    """
    shutil.rmtree(sandbox_dir, ignore_errors=True)
