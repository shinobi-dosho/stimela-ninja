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

Boundaries of the mechanism, by design:

* Inputs are never copied in. Path-typed inputs are rewritten to absolute
  paths anchored at the workspace (`absolutize_path_inputs`), so the tool
  reads -- and, for MUTABLE inputs like an MS, writes -- the caller's real
  files in place. A tool that drops junk *next to an input* therefore
  writes into the workspace; the sandbox can't catch that.
* Absolute-path outputs bypass the sandbox entirely (the tool writes them
  straight to their declared destination); harvest skips them.
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

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

from shinobi.exceptions import ParameterError, StepError
from shinobi.loaders._modelgen import is_file_dtype
from shinobi.steps.schema import Cab, Scope, declared_output_dirs, path_fields


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


def absolutize_path_inputs(scope: Scope, prepared: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """A copy of `prepared` with every relative path-typed input value
    anchored at `workspace`, so the tool still finds (and mutates in place)
    the caller's real files when its cwd is the sandbox. Same field
    classification as container bind-mounting (`backends.container.bind_dirs`):
    declared fields via `path_fields`, dynamically pattern-matched Cab inputs
    via their `ParamMeta.dtype`. Non-path values pass through untouched --
    notably, a *string*-typed output-prefix input stays relative, so the tool
    writes that output family inside the sandbox for harvest to pick up.
    """
    declared = path_fields(scope.inputs_model)
    match_pattern = scope.match_pattern if isinstance(scope, Cab) else None
    anchored = dict(prepared)
    for name, value in prepared.items():
        if value is None:
            continue
        if name not in declared:
            if match_pattern is None:
                continue
            meta = match_pattern(name)
            if meta is None or meta.dtype is None or not is_file_dtype(meta.dtype):
                continue
        anchored[name] = _anchor(value, workspace)
    return anchored


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
    if dst.is_dir() and not dst.is_symlink():
        if not declared:
            raise StepError(
                f"harvest would replace the directory '{dst}', which this step never declared as an "
                f"output -- it matched a harvest glob, so the name came from the tool, not the schema. "
                f"Refusing to delete it. Declare it as an output field if it really is this step's "
                f"product, or move the existing directory aside."
            )
        shutil.rmtree(dst)
    elif dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.move(str(src), str(dst))


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
