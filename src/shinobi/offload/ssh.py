"""Launch a `ninja run` invocation detached on a remote host over SSH.

This is a *sibling* of `shinobi.offload.slurm` (see that module's
docstring for the general "compile/hand-off, detach, poll by handle"
shape): a `ninja run TARGET --remote user@host:/path` doesn't compile
anything -- it rsyncs the target file plus its cab dependencies to the
remote host, launches a plain `ninja run` there detached, and writes back
enough state for `ninja status` to poll it later. No cluster scheduler
involved, just SSH.

Detaching over SSH without leaving a zombie session or losing the real
pid is a known-fiddly corner, so it's worth spelling out the mechanism
`launch_remote` uses:

    setsid bash -c '(<cmd>); echo $? > <exit_file>' </dev/null ><log_file> 2>&1 &
    echo $!

- `ssh host <cmd>` runs non-interactively, so the remote shell never
  turns job control on -- a `&`-backgrounded process is never made a
  process-group leader. That means a bare `setsid` (no `-f`/`--fork`)
  execs in place instead of forking, so the pid captured via `$!` is the
  *actual* long-lived pid of the detached process, not a pid that's about
  to disappear when a fork-parent exits.
- All three standard streams are redirected away from the ssh channel
  *before* anything runs, so the ssh connection can close as soon as the
  remote shell returns `$!` -- it isn't left waiting on any fd the
  background process still holds open.
- The `(<cmd>); echo $? > <exit_file>` wrapping (inside the same
  backgrounded subshell) is what lets `status_ssh` report real
  success/failure rather than just alive/dead. `ninja run TARGET
  --remote ...` never validates the target's inputs locally (see
  `cli.py`'s `run()` -- `--remote` deliberately skips `_resolve_target`,
  since the whole point is running on a host that may have dependencies
  the local machine doesn't), so a bad-input run must surface its failure
  through the handle, not just look "FINISHED".

The cab-dependency scan (`find_cab_deps`) is deliberately best-effort: it
statically walks the target file's AST for `load_file(...)` calls whose
argument is a `Path(__file__).parent / "..." ` -style expression, and
follows cult-cargo `_include:` chains from there. It cannot see
dependencies read by arbitrary orchestration code (a `StepRef`/
`@shinobi.step` function that opens some other local file itself) --
`ninja run --remote`'s `--include` option is the escape hatch for those.
"""

from __future__ import annotations

import ast
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from shinobi.exceptions import BackendError


@dataclass
class RemoteSpec:
    """A parsed `user@host:/path` (or `host:/path`) remote target.

    Attributes:
        host: The host part (optionally including `user@`).
        path: The remote filesystem path.
    """

    host: str
    path: str


def parse_remote(spec: str) -> RemoteSpec:
    """Parse 'user@host:/path' (or 'host:/path') into a RemoteSpec."""
    if ":" not in spec:
        raise ValueError(f"--remote must be 'user@host:/path' (or 'host:/path'), got {spec!r}")
    host, path = spec.split(":", 1)
    if not host or not path:
        raise ValueError(f"--remote must be 'user@host:/path' (or 'host:/path'), got {spec!r}")
    return RemoteSpec(host=host, path=path)


# ---------------------------------------------------------------------------
# Static cab-dependency scan
# ---------------------------------------------------------------------------


def _eval_path_expr(node: ast.expr, env: dict[str, ast.expr], pyfile: Path) -> Path | str | None:
    """Best-effort static evaluation of a `Path(__file__).parent / "x" /
    "y.yml"`-style expression (plus plain string literals and Name lookups
    into `env`, the module's own statically-evaluable assignments). Returns
    None if the expression isn't one of the shapes this understands.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return pyfile
        if node.id in env:
            return _eval_path_expr(env[node.id], env, pyfile)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _eval_path_expr(node.value, env, pyfile)
        return base.parent if isinstance(base, Path) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_path_expr(node.left, env, pyfile)
        right = _eval_path_expr(node.right, env, pyfile)
        if isinstance(left, Path) and isinstance(right, str):
            return left / right
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        if len(node.args) != 1:
            return None
        inner = _eval_path_expr(node.args[0], env, pyfile)
        return Path(inner) if inner is not None else None
    return None


def _collect_env(tree: ast.Module) -> dict[str, ast.expr]:
    """Module-level `Name = <expr>` assignments, kept as raw AST nodes so
    `_eval_path_expr` can evaluate them lazily (and against each other).
    """
    env: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            env[stmt.targets[0].id] = stmt.value
    return env


def _find_load_file_calls(tree: ast.Module) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "load_file":
            calls.append(node)
    return calls


def _find_include_entries(node: Any) -> list[Any]:
    """Every `_include:` entry found anywhere in a nested dict/list --
    cult-cargo's own convention lets `_include` appear at the top level
    *or* nested under `inputs:`/`outputs:` (real `cubical.yml`/
    `quartical.yml` do this; see `cultcargo.py`'s own module docstring), so
    a top-level-only scan silently misses those dependencies. Deliberately
    not `_modelgen.resolve_directive`: that resolves-and-merges each
    entry's content (needing an `entry_to_dict` callback that would have to
    already know how to load every entry kind, package-scoped ones
    included); this only needs to know *where* `_include` appears, to list
    dependency files -- not what they resolve to.
    """
    entries: list[Any] = []
    if isinstance(node, list):
        for item in node:
            entries.extend(_find_include_entries(item))
        return entries
    if not isinstance(node, dict):
        return entries
    for key, value in node.items():
        if key == "_include":
            entries.extend(value if isinstance(value, list) else [value])
        else:
            entries.extend(_find_include_entries(value))
    return entries


def _include_deps(yaml_path: Path, warnings: list[str]) -> list[Path]:
    """Follow cult-cargo `_include:` the same way
    `shinobi.loaders.cultcargo._load_raw` resolves it (relative to the
    including file, and wherever `_include` appears in the document -- not
    just at the top level), returning every included file path found (not
    the merged content -- we only need the file list to sync).
    """
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except OSError as exc:
        warnings.append(f"could not read {yaml_path} to follow its _include chain: {exc}")
        return []
    deps = []
    for inc in _find_include_entries(data):
        if not isinstance(inc, str):
            # package-scoped `{(pkg): [...]}` form -- resolves into an
            # installed package, assumed already present remotely, same
            # as cultcargo._load_raw's own warn-and-skip.
            continue
        inc_path = (yaml_path.parent / inc).resolve()
        deps.append(inc_path)
        deps.extend(_include_deps(inc_path, warnings))
    return deps


def find_cab_deps(pyfile: Path) -> tuple[list[Path], list[str]]:
    """Statically scan `pyfile` for `load_file(...)` calls (matching both
    `shinobi.loaders.cultcargo.load_file` and
    `shinobi.loaders.stimela_classic.load_file` -- same name, harmless to
    treat alike) and resolve the cab file(s) each one loads, including
    cult-cargo `_include:` chains. Returns (dep_paths, warnings); an
    unresolvable call produces a warning rather than raising, since this
    is a best-effort scan, not a full static analyzer.
    """
    tree = ast.parse(pyfile.read_text(), filename=str(pyfile))
    env = _collect_env(tree)

    deps: list[Path] = []
    warnings: list[str] = []
    for call in _find_load_file_calls(tree):
        if len(call.args) != 1:
            warnings.append(f"{pyfile}:{call.lineno}: load_file() call has an unexpected argument shape, skipping")
            continue
        resolved = _eval_path_expr(call.args[0], env, pyfile)
        if not isinstance(resolved, (Path, str)):
            warnings.append(f"{pyfile}:{call.lineno}: could not statically resolve this load_file() argument")
            continue
        dep = Path(resolved).resolve()
        if not dep.is_file():
            warnings.append(f"{pyfile}:{call.lineno}: resolved load_file() path {dep} does not exist locally")
            continue
        deps.append(dep)
        if dep.suffix in (".yml", ".yaml"):
            deps.extend(_include_deps(dep, warnings))

    # de-dupe, preserve order
    seen: set[Path] = set()
    unique_deps = []
    for d in deps:
        if d not in seen:
            seen.add(d)
            unique_deps.append(d)
    return unique_deps, warnings


# ---------------------------------------------------------------------------
# Sync + launch + status
# ---------------------------------------------------------------------------


def _ssh(host: str, command: str) -> subprocess.CompletedProcess:
    """Run `command` on `host` as a single remote shell invocation.

    OpenSSH concatenates all trailing argv elements with a plain space
    into one string before handing it to the remote login shell -- so
    `["ssh", host, "bash", "-lc", command]` does *not* make `command` the
    single argument to `-lc`; the remote shell instead sees `-lc`'s
    argument as just `command`'s first word, and everything else
    (including `command`'s own contents) becomes stray positional
    parameters. Passing exactly one trailing argument, itself already a
    complete `bash -lc '...'` string, sidesteps that join entirely -- with
    nothing else to join it with, ssh can't corrupt the quoting.
    """
    full = f"bash -lc {shlex.quote(command)}"
    return subprocess.run(["ssh", host, full], capture_output=True, text=True)


def sync_to_remote(base_dir: Path, rel_paths: list[Path], remote: RemoteSpec) -> None:
    """rsync `rel_paths` (each relative to `base_dir`) onto
    `remote.host:remote.path`, preserving their relative layout via
    `rsync -R`/`--relative`. Creates the remote directory first.
    """
    mkdir = _ssh(remote.host, f"mkdir -p {shlex.quote(remote.path)}")
    if mkdir.returncode != 0:
        raise BackendError(f"could not create {remote.path} on {remote.host}: {mkdir.stderr.strip()}")

    dest = f"{remote.host}:{remote.path}/"
    proc = subprocess.run(
        ["rsync", "-az", "-R", *[str(p) for p in rel_paths], dest],
        cwd=base_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise BackendError(f"rsync to {dest} failed: {proc.stderr.strip()}")


@dataclass
class RemoteHandle:
    """A reference to a detached, remotely-running (or completed) recipe.

    Attributes:
        host: The remote host the recipe is running on.
        path: The remote working directory the recipe runs in.
        pid: Process ID of the remote launcher process.
        log_file: Remote path to the combined stdout/stderr log.
        exit_file: Remote path to the file the process writes its exit
            code to on completion.
    """

    host: str
    path: str
    pid: str
    log_file: str
    exit_file: str


def launch_remote(remote: RemoteSpec, remote_target: str, argv: list[str], *, add_venv: bool) -> RemoteHandle:
    """Launch `ninja run <remote_target> <argv...>` detached on
    `remote.host`, under `remote.path`. See the module docstring for the
    detach mechanism. Returns a `RemoteHandle` for later `status_ssh`
    polling.
    """
    ts = int(time.time())
    log_file = f"ninja-run-{ts}.log"
    exit_file = f"ninja-run-{ts}.exit"
    # Absolute, not just cd-relative: the outer `>log_path`/`>exit_path`
    # redirects are opened by the shell that runs *before* `inner`'s own
    # `cd remote.path`, so a bare filename would land wherever the SSH
    # login shell's cwd happens to be (its home directory), not
    # remote.path.
    log_path = f"{remote.path.rstrip('/')}/{log_file}"
    exit_path = f"{remote.path.rstrip('/')}/{exit_file}"

    venv_snippet = ""
    if add_venv:
        # No wrapping parens here: `source` inside a `(...)` subshell would
        # only change *that* subshell's PATH, discarded the instant it
        # exits -- before `ninja run` (below) ever sees it. This whole
        # `inner` string already runs inside `wrapped`'s own subshell, so
        # the source just needs to stay in that same shell, not be nested
        # in another one.
        venv_snippet = "test -f venv/bin/activate && source venv/bin/activate || test -f .venv/bin/activate && source .venv/bin/activate; "

    inner = f"cd {shlex.quote(remote.path)}; {venv_snippet}ninja run {shlex.quote(remote_target)} {shlex.join(argv)}"
    wrapped = f"({inner}); echo $? > {shlex.quote(exit_path)}"
    remote_cmd = f"setsid bash -c {shlex.quote(wrapped)} </dev/null >{shlex.quote(log_path)} 2>&1 & echo $!"

    proc = _ssh(remote.host, remote_cmd)
    if proc.returncode != 0:
        raise BackendError(f"could not launch on {remote.host}: {proc.stderr.strip()}")
    pid = proc.stdout.strip()
    if not pid.isdigit():
        raise BackendError(f"unexpected launch output from {remote.host}: {proc.stdout!r} {proc.stderr!r}")

    return RemoteHandle(host=remote.host, path=remote.path, pid=pid, log_file=log_file, exit_file=exit_file)


def status_ssh(handle: dict[str, Any]) -> str:
    """Report a detached `--remote` run's progress, reconstructed fresh
    from `handle` (host/path/pid/log_file/exit_file) with a single ssh
    round-trip -- no persistent process, same contract as `status_slurm`.
    """
    host, path, pid = handle["host"], handle["path"], handle["pid"]
    log_file, exit_file = handle["log_file"], handle["exit_file"]
    exit_path = f"{path.rstrip('/')}/{exit_file}"
    check = f"if [ -f {shlex.quote(exit_path)} ]; then cat {shlex.quote(exit_path)}; else kill -0 {shlex.quote(pid)} 2>/dev/null && echo RUNNING || echo UNKNOWN; fi"
    proc = _ssh(host, check)
    if proc.returncode != 0:
        raise BackendError(f"could not query status on {host}: {proc.stderr.strip()}")
    result = proc.stdout.strip()

    if result == "RUNNING":
        return "RUNNING"
    if result.isdigit():
        code = int(result)
        return "FINISHED (success)" if code == 0 else f"FINISHED (exit {code}) -- see {path}/{log_file}"
    return f"UNKNOWN -- see {path}/{log_file}"
