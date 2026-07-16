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

from shinobi.exceptions import ParameterError
from shinobi.loaders._modelgen import is_file_dtype
from shinobi.steps.schema import Cab, Scope, path_fields


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


def _relative_targets(scope: Scope, outputs: Any, prepared: dict[str, Any], sandbox_dir: Path) -> list[str]:
    """The sandbox-relative paths harvest should rescue: declared path-typed
    output field values (absolute ones already live at their destination and
    are skipped), plus `scope.harvest` glob matches.
    """
    targets: list[str] = []
    for name in sorted(path_fields(scope.outputs_model)):
        value = getattr(outputs, name, None)
        if value is None:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            path = Path(str(item))
            if not path.is_absolute():
                targets.append(str(path))
    for pattern in scope.harvest:
        try:
            resolved = pattern.format(**prepared)
        except KeyError as exc:
            raise ParameterError(
                f"'{scope.name}' harvest pattern {pattern!r} references unknown input {exc}"
            ) from exc
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
            targets.append(str(match.relative_to(sandbox_dir)))
    return targets


def _move(src: Path, dst: Path) -> None:
    """Move `src` over `dst`, replacing what's there -- the same overwrite
    the tool itself would have done had it run in the workspace directly.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    elif dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.move(str(src), str(dst))


def harvest_outputs(
    scope: Scope, outputs: Any, prepared: dict[str, Any], sandbox_dir: Path, workspace: Path
) -> list[Path]:
    """Move the step's declared outputs from `sandbox_dir` to `workspace`,
    preserving their relative paths, and return the workspace-side paths
    that were moved. A declared output the tool never wrote (e.g. an
    optional product, or a same-named input passthrough that already lives
    in the workspace) is silently skipped.
    """
    moved: list[Path] = []
    seen: set[str] = set()
    for rel in _relative_targets(scope, outputs, prepared, sandbox_dir):
        if rel in seen:
            continue
        seen.add(rel)
        src = sandbox_dir / rel
        if not src.exists() and not src.is_symlink():
            continue
        dst = workspace / rel
        _move(src, dst)
        moved.append(dst)
    return moved


def discard_sandbox(sandbox_dir: Path) -> None:
    """Delete the sandbox directory and whatever junk is left in it.
    Best-effort: a straggler open file must not fail the step.
    """
    shutil.rmtree(sandbox_dir, ignore_errors=True)
