from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import click

import shinobi
from shinobi.backends import registered_backend_names
from shinobi.backends._stream import TeardownIncomplete, install_signal_handlers, terminate_all
from shinobi.clickutil import build_options, unflatten_kwargs
from shinobi.config import AppConfig
from shinobi.dag import graph_nodes, render_dag
from shinobi.backends._stream import set_capture_limits
from shinobi.logsetup import setup_file_logging
from shinobi.exceptions import ShinobiError
from shinobi.graph import RecipeGraphError, RecipeNotOffloadableError
from shinobi.offload import (
    OffloadCompileError,
    WorkerSubmissionError,
    compile_slurm,
    prepare_worker_slurm,
    status_slurm,
    status_ssh,
    submit_slurm,
    submit_worker_slurm,
)
from shinobi.policies import build_argv
from shinobi.storage import SharedStorageError
from shinobi.steps.dispatch import _dispatch, _prepare_inputs
from shinobi.steps.schema import Recipe, Scope, StepRef


@click.group()
@click.option("--config", "config_file", default=None, help="Path to a config file.")
@click.option("--backend", "backend", default=None, help="Override the default backend.")
@click.option("--log-file", "log_file", default=None, help="Log filename, created under the log directory. Overrides AppConfig.log.file.")
@click.option("--log-dir", "log_dir", default=None, help="Directory log files are written to. Overrides AppConfig.log.dir.")
@click.option(
    "--log-level",
    "log_level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging verbosity. Overrides AppConfig.log.level.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_file: str | None,
    backend: str | None,
    log_file: str | None,
    log_dir: str | None,
    log_level: str | None,
) -> None:
    """ninja -- the shinobi (Stimela 3.0) CLI."""
    overrides: dict = {}
    if backend:
        # Checked here, at the door, because an unrecognised name does not
        # reliably fail later. A cab reaches `get_backend` and raises, but a
        # `@pystep` never does: its adapter matches the resolved name against
        # `CONTAINER_RUNTIMES` (and `"venv"`) and *falls through to running
        # the function in-process* on anything else. So `--backend
        # singularity` on a CASA pystep didn't say "no such backend" -- it
        # ran the containerised step on the host and surfaced as
        # `ModuleNotFoundError: No module named 'casatasks'`, which points
        # at everything except the actual mistake. A typo in the one place
        # the user typed it should be answered where they typed it.
        known = registered_backend_names()
        if backend not in known:
            raise click.BadParameter(f"unknown backend {backend!r} (available: {', '.join(known)})", param_hint="'--backend'")
        overrides["backend"] = {"default": backend}
    log_overrides = {key: value for key, value in (("file", log_file), ("dir", log_dir), ("level", log_level)) if value is not None}
    if log_overrides:
        overrides["log"] = log_overrides
    ctx.obj = AppConfig.load(config_file=config_file, **overrides)
    setup_file_logging(ctx.obj.log)
    set_capture_limits(ctx.obj.log.capture_head_lines, ctx.obj.log.capture_tail_lines)
    ctx.meta["backend_override"] = backend
    ctx.meta["config_file"] = config_file
    ctx.meta["remote_group_flags"] = _forwarded_group_flags(
        backend=backend,
        log_file=log_file,
        log_dir=log_dir,
        log_level=log_level,
    )
    # The CLI owns the process, so it is the CLI's job to make every way a run
    # ends -- not just Ctrl-C -- stop the work as well. Deliberately not done
    # on import: a library caller embedding shinobi keeps its own handlers.
    install_signal_handlers()


@main.command()
def version() -> None:
    """Print the shinobi version."""
    click.echo(shinobi.__version__)


@main.group("workspace")
def workspace_group() -> None:
    """Inspect and reconcile durable scientific-workspace ownership."""


@workspace_group.command("inspect")
@click.option("--workdir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", help="Workspace to inspect.")
@click.option("--workflow-id", help="Inspect one exact workflow when compatible readers share the workspace.")
def workspace_inspect(workdir: Path, workflow_id: str | None) -> None:
    """Read the current owner and its verified liveness without changing it."""
    from shinobi.ownership import inspect_ownership

    state = inspect_ownership(workdir, workflow_id)
    if state.owner is None:
        click.echo(f"workspace: {state.liveness} ({state.detail})")
        return
    click.echo(f"workspace: {state.liveness}")
    click.echo(f"  workflow: {state.owner.workflow_id}")
    click.echo(f"  kind: {state.owner.kind}")
    click.echo(f"  authority: {state.owner.workspace}")
    click.echo(f"  registry: {state.owner.registry}")
    if state.owner.submission:
        click.echo(f"  submission: {state.owner.submission}")
    click.echo(f"  evidence: {state.detail}")


@workspace_group.command("reconcile")
@click.option("--workdir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", help="Workspace to reconcile.")
@click.option("--workflow-id", help="Reconcile one exact workflow when compatible readers share the workspace.")
def workspace_reconcile(workdir: Path, workflow_id: str | None) -> None:
    """Release an owner only when storage/scheduler evidence proves it dead."""
    from shinobi.ownership import WorkspaceOwnershipError, reconcile_ownership

    try:
        state = reconcile_ownership(workdir, workflow_id)
    except WorkspaceOwnershipError as exc:
        raise click.ClickException(str(exc)) from None
    if state.owner is None:
        click.echo("workspace: already free")
    else:
        click.echo(f"workspace: released {state.owner.kind} workflow {state.owner.workflow_id} ({state.detail})")


@workspace_group.command("release")
@click.option("--workdir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", help="Workspace whose owner is to be released.")
@click.option("--workflow-id", required=True, help="Exact recorded workflow identity.")
@click.option("--force", is_flag=True, help="Release a live or uncertain owner after verifying its exact workflow identity; may expose live data to concurrent mutation.")
def workspace_release(workdir: Path, workflow_id: str, force: bool) -> None:
    """Explicitly release an exact owner; live/uncertain owners need --force."""
    from shinobi.ownership import WorkspaceOwnershipError, inspect_ownership, release_workspace

    try:
        state = inspect_ownership(workdir, workflow_id)
        if state.owner is None:
            if state.liveness != "free":
                raise WorkspaceOwnershipError(f"refusing to release workspace ownership: ownership is {state.liveness} ({state.detail})")
            click.echo("workspace: already free")
            return
        if state.owner.workflow_id != workflow_id:
            raise WorkspaceOwnershipError(f"workspace is owned by workflow {state.owner.workflow_id}, not {workflow_id}")
        if state.liveness != "dead" and not force:
            raise WorkspaceOwnershipError(
                f"refusing to release workflow {workflow_id}: ownership is {state.liveness} ({state.detail}); "
                "use 'workspace reconcile' for a proven-dead owner or repeat with --force after cancelling live work"
            )
        removed = release_workspace(workdir, workflow_id)
    except WorkspaceOwnershipError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo("workspace: released" if removed else "workspace: already free")


@main.command("cab")
@click.argument("cab_file")
@click.argument("cab_name")
def show_cab(cab_file: str, cab_name: str) -> None:
    """Show a cab's schema, as loaded from a scabha-dialect YAML FILE."""
    from shinobi.loaders.yaml_cab import load_file

    cabs = load_file(cab_file)
    if cab_name not in cabs:
        raise click.ClickException(f"no such cab '{cab_name}' in {cab_file}")
    click.echo(cabs[cab_name].model_dump_json(indent=2))


@main.group("cabs")
def cabs_group() -> None:
    """Look up cabs by name across installed `shinobi.cabs` providers
    (e.g. `dosho`), instead of pointing at a specific YAML file (see the
    path-based `cab` command above for that)."""


@cabs_group.command("list")
def list_cabs() -> None:
    """List every cab name, grouped by the provider that supplies it."""
    from shinobi.cabs import list_cabs as _list_cabs

    by_provider = _list_cabs()
    if not by_provider:
        raise click.ClickException("no shinobi.cabs providers installed")
    for provider, names in by_provider.items():
        click.echo(f"{provider}:")
        for name in names:
            click.echo(f"  {name}")


@cabs_group.command("show")
@click.argument("cab_name")
def show_cab_by_name(cab_name: str) -> None:
    """Show a cab's schema, resolved by name across installed providers."""
    from shinobi.cabs import get as _get_cab
    from shinobi.exceptions import CabLoadError

    try:
        cab = _get_cab(cab_name)
    except CabLoadError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(cab.model_dump_json(indent=2))


def _resolve_target(target: str):
    """Resolve 'path/to/file.py:name' or 'dotted.module.path:name' into the
    Scope or StepRef it names.
    """
    if ":" not in target:
        raise click.ClickException(f"target must be 'path:name', got {target!r}")
    location, attr = target.rsplit(":", 1)

    if os.path.isfile(location):
        name = Path(location).stem
        spec = importlib.util.spec_from_file_location(name, location)
        module = importlib.util.module_from_spec(spec)
        # Register before exec so pydantic can resolve the module's own
        # forward-ref annotations (every module here uses `from __future__
        # import annotations`, so model field types are strings that pydantic
        # resolves lazily against sys.modules[__module__]).
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)

    try:
        return getattr(module, attr)
    except AttributeError:
        raise click.ClickException(f"no '{attr}' in {location}") from None


def _forwarded_run_flags(*, quiet: bool, provenance: bool | None, sandbox: bool | None) -> list[str]:
    """Reconstruct the argv form of the `ninja run` options that mean the
    same thing on the remote host as they do locally.

    They have to be rebuilt rather than read off `ctx.args`: click consumes
    every option `run` declares into a typed parameter, so `ctx.args` (under
    `ignore_unknown_options`/`allow_extra_args`) holds only the *unknown*
    extras -- the target's own derived parameters. Anything `run` declares
    and doesn't forward is silently dropped, which is exactly what these
    three used to be.

    The tri-state pairs forward their negative form too: `--no-provenance`
    is a deliberate override of a remote `AppConfig.provenance.enabled`, not
    the same thing as saying nothing.
    """
    flags: list[str] = []
    if quiet:
        flags.append("--quiet")
    if provenance is not None:
        flags.append("--provenance" if provenance else "--no-provenance")
    if sandbox is not None:
        flags.append("--sandbox" if sandbox else "--no-sandbox")
    return flags


def _forwarded_group_flags(
    *,
    backend: str | None,
    log_file: str | None,
    log_dir: str | None,
    log_level: str | None,
) -> list[str]:
    """Reconstruct portable, explicitly supplied top-level options.

    These options belong before ``run`` on the remote command line.  Only
    explicit values travel: when an option is absent, the remote host's own
    AppConfig remains authoritative.  ``--config`` is deliberately absent;
    its path names a local file and ``_run_remote`` refuses it rather than
    silently loading one file locally and another (or none) remotely.
    """
    flags: list[str] = []
    for option, value in (
        ("--backend", backend),
        ("--log-file", log_file),
        ("--log-dir", log_dir),
        ("--log-level", log_level),
    ):
        if value is not None:
            flags.extend((option, value))
    return flags


def _venv_source(venv_lock: str | None, venv_packages: tuple[str, ...], pyfile: Path, *, provisioning: bool):
    """Decide what a `--venv` should provision from, in precedence order.

    Explicit beats discovered, and packages beat a lock only because both
    cannot be given at once -- naming a lock *and* a package list is two
    different declarations of one environment, which is refused rather than
    merged.

    The fallback differs by mode, and that asymmetry is the point:

    - **`sync`** with nothing named and no lock nearby falls back to
      `stimela-ninja==<this version>`. In `ninja run --remote` ninja *is* the
      launcher, so this is the one package the remote demonstrably needs, and
      the version is a fact this process knows rather than a guess. It is a
      version number resolved fresh by the remote's uv, not a copy of
      anything local.
    - **`use`** falls back to nothing. It has no business inventing an
      environment identity to go looking for; with no lock and no packages it
      simply does the legacy `venv/`/`.venv/` search, at no ssh cost.
    """
    from shinobi.offload.remote_venv import discover_lock, launcher_source, lock_source_for, package_source

    if venv_lock and venv_packages:
        raise ValueError("--venv-lock and --venv-package are two different declarations of one environment; pass one")
    if venv_packages:
        return package_source(venv_packages)
    if venv_lock:
        return lock_source_for(Path(venv_lock))
    discovered = discover_lock(pyfile)
    if discovered is not None:
        return discovered
    return launcher_source() if provisioning else None


def _run_remote(
    ctx: click.Context,
    target: str,
    *,
    dryrun: bool,
    cache_dir: str | None,
    no_cache: bool,
    quiet: bool,
    provenance: bool | None,
    sandbox: bool | None,
    remote: str,
    venv: str | None,
    venv_lock: str | None,
    venv_packages: tuple[str, ...],
    add_venv: bool | None,
    include_paths: tuple[str, ...],
) -> None:
    """`ninja run TARGET --remote user@host:/path`: sync TARGET's file plus
    its statically-discoverable cab deps to the remote host and launch it
    there, detached. Deliberately skips `_resolve_target` (which would exec
    the module locally) -- the whole point is running on a host that may
    have dependencies the local machine doesn't.
    """
    if dryrun:
        raise click.ClickException("--remote and --dryrun are mutually exclusive")
    if cache_dir or no_cache:
        raise click.ClickException("--cache-dir/--no-cache apply to local runs only; configure caching via the remote host's own AppConfig")
    if ctx.meta.get("config_file") is not None:
        raise click.ClickException(
            "--config cannot be used with --remote: it names a local file that is not synced; "
            "configure the remote host or use portable command-line overrides such as --backend"
        )

    # Resolved here rather than left to `launch_remote`, for two reasons: a
    # pair of flags asking for different environments should be refused
    # before anything is rsynced, and the deprecation belongs on stderr in
    # click's voice rather than as a DeprecationWarning python hides by
    # default -- an operator who never sees it never migrates.
    from shinobi.offload.remote_venv import resolve_venv_mode

    try:
        venv_mode, deprecation = resolve_venv_mode(venv, add_venv)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    if deprecation:
        click.echo(f"warning: {deprecation}", err=True)

    if ":" not in target:
        raise click.ClickException(f"target must be 'path:name', got {target!r}")
    location, attr = target.rsplit(":", 1)
    if not os.path.isfile(location):
        raise click.ClickException("--remote only supports a local file target ('path/to/file.py:name'), not a dotted module path")

    from shinobi.offload.ssh import find_cab_deps, launch_remote, parse_remote, sync_to_remote

    pyfile = Path(location).resolve()
    try:
        deps, scan_warnings = find_cab_deps(pyfile)
    except SyntaxError as exc:
        raise click.ClickException(f"cannot parse {pyfile}: {exc}") from None
    for w in scan_warnings:
        click.echo(f"warning: {w} -- pass it via --include if it must be synced", err=True)
    if not deps:
        click.echo(
            "note: only the target file and any statically-discovered cab deps are synced; use --include for anything else the recipe reads",
            err=True,
        )

    extra = [Path(p).resolve() for p in include_paths]
    all_paths = [pyfile, *deps, *extra]
    # Common root of everything being synced, not just pyfile's own directory
    # -- real recipes keep cabs in sibling/parent dirs (a shared cabs/
    # folder, an _include: ../lib/foo.yml), which a "must be under the
    # target file's directory" check would wrongly reject.
    base_dir = Path(os.path.commonpath([str(p) for p in all_paths]))
    if base_dir.is_file():
        base_dir = base_dir.parent
    if str(base_dir) == base_dir.anchor:
        raise click.ClickException("target file and its deps don't share a common directory; pass a narrower --include or restructure the recipe")

    try:
        remote_spec = parse_remote(remote)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None

    rel_paths = [p.relative_to(base_dir) for p in all_paths]
    sync_to_remote(base_dir, rel_paths, remote_spec)

    # After the target is synced, before it is launched. Provisioning can
    # take minutes and can fail, and doing it here means a failure lands
    # before a detached process exists to be confused about.
    from shinobi.offload.remote_venv import VENV_SYNC, unpinned
    from shinobi.offload.ssh import resolve_remote_venv

    try:
        source = _venv_source(venv_lock, venv_packages, pyfile, provisioning=venv_mode == VENV_SYNC)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    if source is not None:
        click.echo(f"venv: {source.describe()}", err=True)
        loose = unpinned(source.specs)
        if loose:
            # `env_id` names the spec, not what it resolves to, so an
            # unpinned one describes an environment that can differ between
            # two provisions sharing a directory. Allowed -- someone may
            # genuinely want current -- but not silently.
            click.echo(
                f"venv: {', '.join(loose)} names no exact version, so this environment's id describes what was asked for, not what was installed; pin with == for a stable id",
                err=True,
            )
    try:
        resolved = resolve_remote_venv(remote_spec, venv_mode, source)
    except ShinobiError as exc:
        # Invariant 7: a `sync` that cannot provision fails the launch. As a
        # ClickException rather than a traceback -- every one of these is a
        # condition on the remote host with something for the operator to do
        # about it, not a bug in ninja.
        raise click.ClickException(str(exc)) from None
    for note in resolved.notes:
        click.echo(f"venv: {note}", err=True)

    remote_target = f"{pyfile.relative_to(base_dir)}:{attr}"
    # `ninja run`'s own flags first, then `ctx.args` (the target's derived
    # parameters). Order is what keeps them separable: a target parameter
    # taking a value can't swallow a trailing `--sandbox` if the flags never
    # trail it.
    run_argv = [*_forwarded_run_flags(quiet=quiet, provenance=provenance, sandbox=sandbox), *ctx.args]
    launcher = ["ninja", *ctx.meta.get("remote_group_flags", ()), "run"]
    handle = launch_remote(remote_spec, remote_target, run_argv, venv=venv_mode, venv_path=resolved.path, launcher=launcher)

    handle_path = _handle_path(None, f"{pyfile.stem}.{attr}")
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle_path.write_text(json.dumps({"engine": "ssh", **handle.__dict__}, indent=2))
    name = handle_path.parent.name
    click.echo(f"launched on {remote_spec.host} (detached, pid={handle.pid})")
    click.echo(f"  handle: {handle_path}")
    click.echo(f"  watch:  ninja logs {name} --follow")
    click.echo("  list:   ninja runs")


@main.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
    no_args_is_help=True,
)
@click.argument("target")
@click.option(
    "--dryrun",
    is_flag=True,
    help="Show what would run as a graph, without actually running it.",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    default=None,
    help="Directory for step-level result caching (see shinobi.cache). Only takes effect for a "
    "step that has caching enabled some other way (its own Scope.cache, an enclosing recipe's, "
    "or AppConfig.cache.enabled) -- this option alone doesn't turn caching on.",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    help="Disable step-level caching for this run, regardless of AppConfig/Scope cache settings.",
)
@click.option(
    "--quiet",
    "quiet",
    is_flag=True,
    help="Don't live-echo running cabs' stdout/stderr (native/container backends only) -- "
    "restores the old behavior of a silent run followed by one dump of captured output at "
    "the end. Overrides AppConfig.log.stream for this invocation.",
)
@click.option(
    "--provenance/--no-provenance",
    "provenance",
    default=None,
    help="Enable reproducible-run provenance for this run: digest-pin container images before "
    "running (pin-then-run) and write a run manifest under AppConfig.provenance.dir. Off by "
    "default -- overrides AppConfig.provenance.enabled for this invocation.",
)
@click.option(
    "--sandbox/--no-sandbox",
    "sandbox",
    default=None,
    help="Run each step with its cwd inside a private scratch dir (AppConfig.sandbox.dir); on "
    "success only declared outputs are moved back to the workspace and auxiliary droppings "
    "(tool logfiles etc.) are deleted with the scratch dir. Off by default -- overrides "
    "AppConfig.sandbox.enabled for this invocation.",
)
@click.option(
    "--remote",
    "remote",
    default=None,
    help="Launch on a remote host instead of locally: 'user@host:/path'. Syncs the target file and its statically-discoverable cab deps, then runs detached -- see `ninja status`. Explicit --backend/--log-* and --provenance/--sandbox/--quiet options are forwarded; --config, --dryrun, and --cache-dir/--no-cache are refused.",
)
@click.option(
    "--venv",
    "venv",
    type=click.Choice(["off", "use", "sync"]),
    default=None,
    help="With --remote, what to do about the remote Python environment: 'use' (default) activates a provisioned environment matching the lock if there is one, else venv/bin/activate or .venv/bin/activate under the remote path, warning on stderr if there is nothing; 'sync' provisions that environment from the lock first, with uv; 'off' sources nothing.",
)
@click.option(
    "--venv-lock",
    "venv_lock",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="With --venv sync/use, the uv.lock (or requirements.txt) to provision the remote environment from. Defaults to the nearest one above the target file.",
)
@click.option(
    "--venv-package",
    "venv_packages",
    multiple=True,
    default=(),
    help="With --venv sync/use, a package to provision the remote environment with, e.g. 'caracal==2.0.1'. Repeatable. For the common case of no repository and no lockfile; defaults to the running stimela-ninja's own version. Mutually exclusive with --venv-lock.",
)
@click.option(
    # Kept for one release as the deprecated spelling of `--venv use/off`.
    # `default=None`, not `True`: a boolean default is indistinguishable from
    # a caller who typed `--add-venv`, and telling those apart is the only
    # way to notice `--venv off --add-venv` asking for two environments.
    "--add-venv/--no-add-venv",
    "add_venv",
    default=None,
    hidden=True,
    help="Deprecated alias for --venv use/--venv off.",
)
@click.option(
    "--include",
    "include_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="With --remote, additional files/dirs to sync alongside the target and its "
    "discovered cab deps -- needed for orchestration code (StepRef/@shinobi.step) that "
    "reads local files the static cab-dep scan can't see, or for cabs it can't resolve.",
)
@click.pass_context
def run(
    ctx: click.Context,
    target: str,
    dryrun: bool,
    cache_dir: str | None,
    no_cache: bool,
    quiet: bool,
    provenance: bool | None,
    sandbox: bool | None,
    remote: str | None,
    venv: str | None,
    venv_lock: str | None,
    venv_packages: tuple[str, ...],
    add_venv: bool | None,
    include_paths: tuple[str, ...],
) -> None:
    """Run a Cab, Recipe, or @shinobi.step TARGET ('path/to/file.py:name'
    or 'pkg.mod:name').

    [OPTIONS] are derived from the target's own parameters -- run
    `ninja run TARGET --help` to see them.
    """
    if target in ("-h", "--help"):
        click.echo(ctx.get_help())
        ctx.exit()

    if remote:
        _run_remote(
            ctx,
            target,
            dryrun=dryrun,
            cache_dir=cache_dir,
            no_cache=no_cache,
            quiet=quiet,
            provenance=provenance,
            sandbox=sandbox,
            remote=remote,
            venv=venv,
            venv_lock=venv_lock,
            venv_packages=venv_packages,
            add_venv=add_venv,
            include_paths=include_paths,
        )
        return

    obj = _resolve_target(target)

    if isinstance(obj, StepRef):
        scope: Scope = obj.step
        func = obj.func
        params = obj.params
    elif isinstance(obj, Scope):
        scope, func, params = obj, None, {}
    else:
        raise click.ClickException(f"{target!r} is neither a Cab, Recipe, nor a @shinobi.step function")

    def _callback(**kwargs):
        # Drop options the user didn't provide (None) so the inputs_model's
        # own defaults apply; per-step constants from a StepRef go under
        # them. `unflatten_kwargs` also re-nests any `--parent-child`
        # flattened group option (build_options' counterpart for a nested
        # BaseModel field) back into the nested dict its model expects.
        call_kwargs = {**params, **unflatten_kwargs(scope.inputs_model, kwargs)}

        if dryrun:
            if isinstance(scope, Recipe):
                try:
                    click.echo(render_dag(graph_nodes(scope, call_kwargs, workspace=Path.cwd())))
                except (RecipeGraphError, ValueError) as exc:
                    raise click.ClickException(str(exc)) from None
            else:
                prepared = _prepare_inputs(scope, {**call_kwargs})
                click.echo(" ".join(build_argv(scope, prepared)))
            return

        backend = ctx.meta.get("backend_override")
        cache = False if no_cache else None
        config = ctx.obj or AppConfig.load()
        stream = False if quiet else None
        stream_enabled = False if quiet else config.log.stream
        try:
            result = _dispatch(
                scope,
                func,
                backend=backend,
                cache=cache,
                cache_dir=cache_dir,
                stream=stream,
                provenance=provenance,
                sandbox=sandbox,
                _config=ctx.obj,
                _provenance_target=target,
                **call_kwargs,
            )
        except (ShinobiError, RecipeGraphError) as exc:
            raise click.ClickException(str(exc)) from None
        except TeardownIncomplete as exc:
            # Distinct from an ordinary interrupt, and must stay distinct: it
            # means something is *still running*. Exit 1, not 130 -- this is a
            # failure the user has to act on, not a clean cancellation.
            raise click.ClickException(str(exc)) from None
        except KeyboardInterrupt:
            # By here the step's own process tree is already down: whichever
            # of `run_streaming` or the recipe scheduler was holding the
            # interrupt tore it down on the way out. This call is the
            # backstop for a Ctrl-C that landed anywhere else in dispatch
            # (resolving wiring, hashing inputs) with a child still live --
            # it costs nothing when there is none. What the user gets is a
            # single honest line instead of a traceback through click.
            terminate_all(reason="interrupted")
            click.echo(f"'{scope.name}' interrupted", err=True)
            ctx.exit(130)  # the conventional shell code for death by SIGINT
        # When streaming happened, every line already printed live as it
        # ran -- dumping the same captured text again here would just
        # repeat it. Only fall back to the old one-shot dump when
        # streaming was off (--quiet, or config.log.stream=False), or on a
        # cache hit that never actually ran anything (so nothing was ever
        # streamed regardless of the setting).
        if not stream_enabled or result.cached:
            if result.stdout:
                click.echo(result.stdout)
            if result.stderr:
                click.echo(result.stderr, err=True)
        if not result.success:
            raise click.ClickException(f"'{scope.name}' exited with status {result.returncode}")

    inner = click.Command(
        name=target,
        params=build_options(scope.inputs_model),
        callback=_callback,
        help=scope.info,
    )
    inner.main(args=ctx.args, prog_name=f"{ctx.info_name} {target}", standalone_mode=False)


@main.command("replay")
@click.argument("run_manifest", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--target",
    "target_override",
    default=None,
    help="Override the target recorded in the manifest ('path/to/file.py:name' or "
    "'pkg.mod:name'); required for manifests that don't record one (older manifests, "
    "or runs launched programmatically rather than via `ninja run`).",
)
@click.option(
    "--allow-unpinned",
    "allow_unpinned",
    is_flag=True,
    help="Replay even when the manifest is not fully pinned (pinned: false). Unpinned steps run their original image reference or venv, so exact reproduction is not guaranteed.",
)
@click.pass_context
def replay(ctx: click.Context, run_manifest: str, target_override: str | None, allow_unpinned: bool) -> None:
    """Re-run a recorded run from its RUN_MANIFEST (a `.run.json` written by
    a `--provenance` run), forcing every containerized step to the exact
    image digest that originally ran and re-feeding the recorded inputs.

    Uses the manifest's recorded backend by default; the global `ninja
    --backend` flag overrides it. The replay is itself a provenance run and
    writes its own manifest.
    """
    from pydantic import ValidationError

    from shinobi.exceptions import ReplayError
    from shinobi.provenance import apply_manifest_pins, load_manifest, unpinned_steps

    try:
        manifest = load_manifest(Path(run_manifest))
    except (OSError, ValidationError) as exc:
        raise click.ClickException(f"cannot read run manifest {run_manifest!r}: {exc}") from None

    target = target_override or manifest.target
    if target is None:
        raise click.ClickException(
            "manifest has no 'target' (written by an older shinobi, or a programmatic run) -- pass --target 'path/to/file.py:name' to identify the recipe/cab"
        )

    if not manifest.pinned and not allow_unpinned:
        offenders = unpinned_steps(manifest.root)
        raise click.ClickException(
            f"manifest is not fully pinned (unpinned steps: {', '.join(offenders)}); replay cannot guarantee the same environment ran -- pass --allow-unpinned to proceed anyway"
        )

    obj = _resolve_target(target)
    if isinstance(obj, StepRef):
        # Per-step params are ignored: the manifest root inputs are the
        # fully-resolved values of the original run, which subsume them.
        scope, func = obj.step, obj.func
    elif isinstance(obj, Scope):
        scope, func = obj, None
    else:
        raise click.ClickException(f"{target!r} is neither a Cab, Recipe, nor a @shinobi.step function")

    try:
        scope = apply_manifest_pins(scope, manifest.root)
    except ReplayError as exc:
        raise click.ClickException(str(exc)) from None

    try:
        scope.inputs_model(**manifest.root.inputs)
    except ValidationError as exc:
        raise click.ClickException(
            f"manifest inputs no longer validate against {scope.name!r}: {exc}\n"
            "(non-serializable inputs -- e.g. MUTABLE fields -- are stored as strings "
            "in the manifest and may not be replayable)"
        ) from None

    config = ctx.obj or AppConfig.load()
    backend = ctx.meta.get("backend_override") or manifest.backend
    try:
        result = _dispatch(scope, func, backend=backend, provenance=True, _config=ctx.obj, _provenance_target=target, **manifest.root.inputs)
    except (ShinobiError, RecipeGraphError) as exc:
        raise click.ClickException(str(exc)) from None
    # Same output policy as `run`: streaming already echoed everything live;
    # only dump captured output when streaming was off or nothing ran.
    if not config.log.stream or result.cached:
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)
    if not result.success:
        raise click.ClickException(f"'{scope.name}' exited with status {result.returncode}")


@main.command("clean")
@click.option("--runs/--no-runs", "runs", default=True, help="Remove run manifests (AppConfig.provenance.dir).")
@click.option(
    "--cache/--no-cache",
    "cache",
    default=True,
    help="Clear the step cache while retaining persistent transaction locks (AppConfig.cache.dir).",
)
@click.option(
    "--sandboxes/--no-sandboxes",
    "sandboxes",
    default=True,
    help="Remove leftover step sandboxes (AppConfig.sandbox.dir) -- a failed sandboxed step keeps its scratch dir for post-mortem; this is how it eventually gets cleaned up.",
)
@click.option(
    "--launches/--no-launches",
    "launches",
    default=False,
    help=(
        "Remove detached-run launch dirs (.shinobi/<recipe>/, holding handle.json "
        "and Slurm job logs). Off by default: unlike runs/cache, deleting one can "
        "destroy `ninja status`'s only local record of a still-running detached job."
    ),
)
@click.option(
    "--workdir",
    "workdir",
    default=None,
    help="Directory to look for launch dirs under (default: cwd). Only affects --launches.",
)
@click.option("--dry-run", "dry_run", is_flag=True, help="Show what would be removed without deleting.")
@click.option(
    "--force",
    "force",
    is_flag=True,
    help=(
        "Clear the step cache even while quarantined trees are outstanding, and delete those trees too. Without this, --cache refuses rather than orphaning a tree whose only explanation is the journal it would clear."
    ),
)
@click.pass_context
def clean(ctx: click.Context, runs: bool, cache: bool, sandboxes: bool, launches: bool, workdir: str | None, dry_run: bool, force: bool) -> None:
    """Remove shinobi runtime artifacts and clear the step cache,
    leftover step sandboxes, and (opt-in) detached-run launch dirs.

    Run manifests, the step cache, and leftover sandboxes come from the
    active config (AppConfig.provenance.dir, AppConfig.cache.dir, and
    AppConfig.sandbox.dir) and are selected by default. Cache transaction
    lock inodes are retained; the other targets are removed. Launch dirs
    (.shinobi/<recipe>/, written by `ninja compile --submit` / `ninja run
    --remote`) are opt-in via --launches. Use --dry-run to preview.
    """
    config = ctx.obj or AppConfig.load()
    targets: list[tuple[str, Path]] = []
    if runs:
        targets.append(("run manifests", Path(config.provenance.dir)))
    if cache:
        targets.append(("step cache", Path(config.cache.dir)))
    if sandboxes:
        targets.append(("sandboxes", Path(config.sandbox.dir)))
    if launches:
        base = Path(workdir or os.getcwd())
        found = [p.parent for p in sorted(base.glob(".shinobi/*/handle.json"))]
        if found:
            targets.extend((f"launch ({p.name})", p) for p in found)
        else:
            click.echo(f"launches: nothing at {base / '.shinobi'}/*/handle.json")
    if not runs and not cache and not sandboxes and not launches:
        raise click.ClickException("nothing selected: pass --runs/--cache/--sandboxes/--launches")

    # Resolved before anything is cleared: the list is derived from the chain
    # journal whose contents are part of that reset.
    pending = _unreconciled_trash(config) if cache else []
    if pending and not force:
        # Clearing the journal removes the only thing that says what a quarantined tree was
        # quarantined for. Refuse rather than strand it: these trees are
        # typically the biggest things in the workspace, and nothing else will
        # ever explain them.
        listing = "\n  ".join(str(path) for path in pending)
        raise click.ClickException(
            f"refusing to clear the step cache while quarantined trees are outstanding -- clearing the journal would leave these unexplained:\n  {listing}\nRun 'ninja cache check' to see why, or pass --force to remove them too."
        )

    for label, path in targets:
        if not path.exists():
            click.echo(f"{label}: nothing at {path}")
        elif dry_run:
            action = "clear (persistent transaction locks retained)" if label == "step cache" else "remove"
            click.echo(f"{label}: would {action} {path}")
        elif label == "step cache":
            try:
                _clear_step_cache(path)
            except SharedStorageError as exc:
                raise click.ClickException(str(exc)) from exc
            click.echo(f"{label}: cleared {path} (persistent transaction locks retained)")
        else:
            shutil.rmtree(path)
            click.echo(f"{label}: removed {path}")

    if force:
        for path in pending:
            if dry_run:
                click.echo(f"trash: would remove {path}")
            else:
                shutil.rmtree(path, ignore_errors=True)
                click.echo(f"trash: removed {path}")


def _clear_step_cache(root: Path) -> None:
    """Clear cache contents without ever replacing its lock domains."""
    from shinobi.cache import CacheManifest
    from shinobi.snapshots import ChainJournal
    from shinobi.storage import sync_directory

    manifest = CacheManifest(root / "manifest.json")
    journal = ChainJournal(root / "snapshots")
    manifest.reset()

    def clear_snapshot_payloads() -> None:
        if not journal.root.exists():
            return
        for entry in journal.root.iterdir():
            # The transaction inode and run-presence locks are persistent.
            # Unlinking either while held would split a lock domain.
            if entry == journal.lock_path or entry.name == "locks":
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        sync_directory(journal.root)

    journal.reset(clear_snapshot_payloads)
    for entry in root.iterdir():
        if entry == manifest.lock_path or entry == journal.root:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
    sync_directory(root)


def _unreconciled_trash(config: AppConfig) -> list[Path]:
    """Quarantined trees the journal still has, or once had, a reason for."""
    from shinobi.snapshots import TRASH_SUFFIX, get_journal, orphan_trash

    try:
        found = list(orphan_trash(config.cache.dir))
    except Exception:  # noqa: BLE001 -- a broken journal must not block a clean
        found = []
    try:
        chains = get_journal(config.cache.dir).all_chains().values()
    except Exception:  # noqa: BLE001 -- explicit reset is corrupt-store recovery
        chains = ()
    for chain in chains:
        if chain.marker is None:
            continue
        candidate = Path(chain.path).with_name(Path(chain.path).name + TRASH_SUFFIX + chain.marker.run_id)
        if candidate.exists():
            found.append(candidate)
    return sorted(set(found))


@main.group("cache")
def cache_group() -> None:
    """Inspect and repair the step cache and its mutation-chain snapshots."""


@cache_group.command("check")
@click.pass_context
def cache_check(ctx: click.Context) -> None:
    """Report anything the cache cannot vouch for.

    Reads state, changes none: off-tip paths, unreconciled and orphaned
    quarantined trees, journal/manifest disagreements, generations whose
    snapshot was never taken, and what the clone-capability probe decided
    for each filesystem and why.
    """
    from shinobi.cache import get_cache_manifest
    from shinobi.snapshots import check

    config = ctx.obj or AppConfig.load()
    report = check(config.cache.dir, get_cache_manifest(config.cache.dir))
    labels = {
        "unreconciled": "interrupted steps (a run died here; the next run restores before re-running)",
        "off_tip": "paths whose content is not vouched for",
        "orphan_trash": "orphaned quarantined trees (no marker explains these; 'ninja clean --cache --force' removes them)",
        "disagreements": "journal/manifest disagreements",
        "missing_snapshots": "generations with no snapshot (a space preflight refused them)",
        "unprotected": "paths written by something this journal cannot name (rollback is refused past these points)",
        "capabilities": "clone capability by filesystem",
    }
    clean_report = True
    for section, label in labels.items():
        lines = report.get(section) or []
        if not lines:
            continue
        if section != "capabilities":
            clean_report = False
        click.echo(f"{label}:")
        for line in lines:
            click.echo(f"  {line}")
    if clean_report:
        click.echo("cache: nothing to report")


@cache_group.command("invalidate")
@click.argument("step_path")
@click.pass_context
def cache_invalidate(ctx: click.Context, step_path: str) -> None:
    """Forget a step's cached result and roll its snapshots back.

    STEP_PATH is the dotted path the cache records a step under
    (`<recipe>.<step>`, as shown by `ninja cache check`).

    This removes more than the manifest entry, and it has to. A step that
    exits zero while writing garbage gets that garbage snapshotted under a
    durable name, and a later restore reinstates it faithfully -- so the
    escape hatch drops the step's own generations, rolls the chain head back
    to the state before them, and marks it untrusted, which is what makes
    the next run put the earlier state back before re-running.
    """
    from shinobi.cache import get_cache_manifest
    from shinobi.snapshots import invalidate

    config = ctx.obj or AppConfig.load()
    notes = invalidate(config.cache.dir, step_path, get_cache_manifest(config.cache.dir))
    if not notes:
        click.echo(f"cache: nothing recorded for '{step_path}'")
    for note in notes:
        click.echo(f"cache: {note}")


@cache_group.command("evict")
@click.option("--bytes", "target", required=True, type=int, help="How many bytes of snapshots to try to free.")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without deleting.")
@click.pass_context
def cache_evict(ctx: click.Context, target: int, dry_run: bool) -> None:
    """Free snapshot space, safest candidates first.

    Superseded generations of live chains go before anything else, then dead
    chains (whose workspace path no longer exists) oldest first. A state some
    live chain's head or consumed record still names is never evicted: it
    would make every restore through that point impossible while reclaiming
    almost nothing on a clone-capable filesystem, since its blocks are shared.
    """
    from shinobi.snapshots import evict

    config = ctx.obj or AppConfig.load()
    if dry_run:
        click.echo("evict --dry-run is not supported yet; run 'ninja cache check' to see chain state first")
        return
    removed = evict(config.cache.dir, target)
    if not removed:
        click.echo("cache: nothing evictable (every snapshot is still named by a live chain)")
        return
    for name, size in removed:
        click.echo(f"cache: evicted {name} ({size} bytes)")
    click.echo(f"cache: freed {sum(size for _name, size in removed)} bytes across {len(removed)} snapshots")


def _handle_path(workdir: str | None, recipe: str) -> Path:
    return Path(workdir or os.getcwd()) / ".shinobi" / recipe / "handle.json"


@main.command(
    "compile",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
    no_args_is_help=True,
)
@click.argument("target")
@click.option("--engine", default="slurm", help="Workflow engine to compile to (v1: slurm).")
@click.option("--workdir", default=None, help="Working directory for the compiled jobs.")
@click.option(
    "--container-runtime",
    default="apptainer",
    help="Runtime to wrap imaged cabs in (use 'none' for bare argv).",
)
@click.option("--submit", is_flag=True, help="Submit the compiled workflow and detach.")
@click.option("--worker", is_flag=True, help="Use the experimental frozen-bundle M1 worker lifecycle (requires --submit).")
@click.option("--submission-root", type=click.Path(path_type=Path), default=None, help="Shared directory for immutable worker submissions.")
@click.option("--worker-python", type=click.Path(path_type=Path), default=None, help="Absolute compute-visible Python for the staged worker.")
@click.option("--code-root", type=click.Path(path_type=Path), multiple=True, help="Python import root for bundled pystep source (repeatable).")
@click.option(
    "--cache/--no-cache",
    "cache",
    default=None,
    help="Enable or disable runtime worker caching, overriding Scope/AppConfig settings.",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Shared runtime cache directory for worker jobs; this option alone does not enable caching.",
)
@click.pass_context
def compile_recipe(
    ctx: click.Context,
    target: str,
    engine: str,
    workdir: str | None,
    container_runtime: str,
    submit: bool,
    worker: bool,
    submission_root: Path | None,
    worker_python: Path | None,
    code_root: tuple[Path, ...],
    cache: bool | None,
    cache_dir: str | None,
) -> None:
    """Compile a Recipe TARGET ('path/to/file.py:name' or 'pkg.mod:name')
    into a cluster workflow and, with --submit, hand it off and detach.

    Only purely-declarative recipes can be offloaded; anything relying on
    live Python (orchestration functions, MUTABLE non-path inputs, non-path
    data flow) is rejected with an explanation -- run those locally via
    `ninja run`. Steps that rewrite a shared path in place are fine: the
    compiler derives the ordering they need. [OPTIONS] carry the recipe's
    own inputs; run `ninja compile TARGET --help` to see them.
    """
    if target in ("-h", "--help"):
        click.echo(ctx.get_help())
        ctx.exit()

    if engine != "slurm":
        raise click.ClickException(f"unknown engine '{engine}' (only 'slurm' in v1)")

    obj = _resolve_target(target)
    if not isinstance(obj, Recipe):
        raise click.ClickException(f"{target!r} is not a Recipe -- only recipes can be offloaded")
    recipe = obj
    runtime = None if container_runtime.lower() == "none" else container_runtime
    if worker and not submit:
        raise click.ClickException("--worker currently requires --submit because submission preparation stages immutable source and environment data")
    if not worker and (cache is not None or cache_dir is not None):
        raise click.ClickException("--cache/--no-cache/--cache-dir require --worker; the legacy argv compiler has no runtime cache lifecycle")

    def _callback(**kwargs):
        inputs = unflatten_kwargs(recipe.inputs_model, kwargs)
        if worker:
            from shinobi.offload._codec import BundleError
            from shinobi.offload.bundle import freeze_recipe

            workspace = Path(workdir or os.getcwd()).resolve()
            root = (submission_root or workspace / ".shinobi" / "submissions").resolve()
            try:
                bundle = freeze_recipe(
                    recipe,
                    inputs,
                    config=ctx.obj,
                    workspace=workspace,
                    code_roots=code_root,
                    cache=cache,
                    cache_dir=cache_dir,
                )
                workflow = prepare_worker_slurm(bundle, submission_root=root, worker_python=worker_python)
            except (BundleError, ShinobiError, RecipeNotOffloadableError, OffloadCompileError, RecipeGraphError) as exc:
                raise click.ClickException(str(exc)) from None
            try:
                launched = submit_worker_slurm(workflow)
            except WorkerSubmissionError as exc:
                raise click.ClickException(f"{exc}; accepted jobs remain detached and recoverable from {exc.handle.submission_dir / 'handle.json'}") from None
            except (ShinobiError, OffloadCompileError) as exc:
                raise click.ClickException(str(exc)) from None
            handle = workflow.submission_dir / "handle.json"
            click.echo(f"submitted {len(launched.jobs)} worker jobs (detached); handle: {handle}")
            for name, job_id in launched.jobs.items():
                click.echo(f"  {name}: {job_id}")
            if launched.finalizer_job:
                click.echo(f"  finalize: {launched.finalizer_job}")
            return
        try:
            workflow = compile_slurm(recipe, inputs, workdir=workdir, container_runtime=runtime)
        except (ShinobiError, RecipeNotOffloadableError, OffloadCompileError, RecipeGraphError) as exc:
            raise click.ClickException(str(exc)) from None

        if not submit:
            if workflow.execution_blocked_reason is not None:
                click.echo(f"# planning only: {workflow.execution_blocked_reason}")
            for job in workflow.jobs:
                dep = f"  (afterok: {', '.join(job.depends_on)})" if job.depends_on else ""
                click.echo(f"# ===== {job.name}{dep} =====")
                for reason in job.access_reasons:
                    click.echo(f"# access: {reason}")
                click.echo(job.script)
            return

        try:
            job_ids = submit_slurm(workflow, workdir=workdir)
        except ShinobiError as exc:
            raise click.ClickException(str(exc)) from None
        handle = _handle_path(workdir, workflow.recipe)
        handle.parent.mkdir(parents=True, exist_ok=True)
        handle.write_text(json.dumps({"engine": engine, "recipe": workflow.recipe, "jobs": job_ids}, indent=2))
        click.echo(f"submitted {len(job_ids)} jobs (detached); handle: {handle}")
        for name, job_id in job_ids.items():
            click.echo(f"  {name}: {job_id}")

    inner = click.Command(
        name=target,
        params=build_options(recipe.inputs_model),
        callback=_callback,
        help=recipe.info,
    )
    inner.main(args=ctx.args, prog_name=f"{ctx.info_name} {target}", standalone_mode=False)


@main.command("status")
@click.argument("handle_file")
def show_status(handle_file: str) -> None:
    """Report a detached offloaded run's progress from its HANDLE_FILE
    (written by `ninja compile --submit` or `ninja run --remote`), by
    querying the engine fresh -- no persistent process.
    """
    try:
        data = json.loads(Path(handle_file).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"cannot read handle {handle_file!r}: {exc}") from None
    engine = data.get("engine")
    # An unreachable host is a condition on the operator's network, not a
    # bug in ninja, and a traceback is the wrong way to say so.
    try:
        if engine == "slurm":
            for name, state in status_slurm(data["jobs"]).items():
                click.echo(f"{name}: {state}")
        elif engine == "slurm-worker":
            for name, state in status_slurm(data["jobs"]).items():
                click.echo(f"{name}: {state}")
            final = Path(data["submission"]) / "finalization.json"
            click.echo(f"finalization: {'published' if final.exists() else 'pending'}")
        elif engine == "ssh":
            click.echo(status_ssh(data))
        else:
            raise click.ClickException(f"unknown engine in handle: {engine!r}")
    except ShinobiError as exc:
        raise click.ClickException(str(exc)) from None


def _elapsed(launch, state=None) -> str:
    """How long the run took, as `HH:MM:SS`, or `-` when the launch time
    is unknown.

    A running job is measured to now; a finished one to when it finished,
    where the probe could establish that. Measuring a finished run to now
    would make the number grow every time the table is redrawn, reporting
    how long ago you launched it rather than how long it took.
    """
    since = launch.launched_at
    if not since:
        return "-"
    until = state.finished_at if state is not None and state.finished_at else None
    if until is None and state is not None and not state.running:
        # Finished, but the host could not say when. Better an honest
        # ceiling than a number that quietly keeps counting.
        return "-"
    seconds = int(max(0.0, (until if until is not None else time.time()) - since))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _state_style(state) -> tuple[str, str]:
    """(text, rich style) for a `RunState`. Colour carries the same
    information as the word, never replaces it -- these tables get piped
    into files and read over terminals that render no colour at all.
    """
    from shinobi.offload.tracking import FINISHED, RUNNING

    if state is None:
        return "?", "dim"
    if state.state == RUNNING:
        return "RUNNING", "cyan"
    if state.state == FINISHED:
        return ("FINISHED (0)", "green") if state.exit_code == 0 else (f"FINISHED ({state.exit_code})", "red")
    return "UNKNOWN", "yellow"


@main.command("runs")
@click.option("--workdir", "workdir", default=None, help="Directory to look for launch dirs under (default: cwd).")
@click.option("--json", "as_json", is_flag=True, help="Emit the listing as JSON instead of a table.")
@click.option("--no-probe", is_flag=True, help="List handles without querying their engines (no ssh round trips).")
def show_runs(workdir: str | None, as_json: bool, no_probe: bool) -> None:
    """List the offloaded runs this workspace has launched, with live state.

    Reads every `.shinobi/*/handle.json` -- the same handles
    `ninja status` takes one at a time and `ninja clean --launches`
    removes -- and asks each engine what became of its run. Nothing is
    cached: a listing is as current as the moment you ask for it.
    """
    from rich.console import Console
    from rich.table import Table

    from shinobi.offload.tracking import discover, probe_all

    launches = discover(workdir)
    if not launches:
        base = Path(workdir or os.getcwd())
        click.echo(f"no launches under {base / '.shinobi'}/*/handle.json")
        return
    if not no_probe:
        probe_all(launches)

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "name": launch.name,
                        "engine": launch.engine,
                        "host": launch.host,
                        "handle": str(launch.handle_path),
                        "launched_at": launch.launched_at,
                        "state": launch.state.state if launch.state else None,
                        "exit_code": launch.state.exit_code if launch.state else None,
                        "detail": launch.state.detail if launch.state else "",
                    }
                    for launch in launches
                ],
                indent=2,
            )
        )
        return

    console = Console()
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("NAME", overflow="fold")
    table.add_column("HOST")
    table.add_column("STATE")
    table.add_column("ELAPSED", justify="right")
    for launch in launches:
        text, style = _state_style(launch.state)
        table.add_row(launch.name, launch.host, f"[{style}]{text}[/{style}]", _elapsed(launch, launch.state))
    console.print(table)
    running = sum(1 for launch in launches if launch.state and launch.state.running)
    console.print(f"\n[dim]{len(launches)} launch{'es' if len(launches) != 1 else ''}, {running} running[/dim]")


@main.command("logs")
@click.argument("run")
@click.option("-f", "--follow", is_flag=True, help="Keep streaming until the run finishes.")
@click.option("-n", "--lines", default=40, show_default=True, help="How many lines of existing log to show first.")
@click.option("--workdir", "workdir", default=None, help="Directory to look for launch dirs under (default: cwd).")
def show_logs(run: str, follow: bool, lines: int, workdir: str | None) -> None:
    """Show an offloaded run's log. RUN is a name from `ninja runs`, or a
    path to a handle file.

    Replaces the `ssh <host> tail -f <path>` line a `--remote` launch
    prints: same mechanism, but it knows which host and which file, it
    shows what the run's state is while you watch, and with `--follow` it
    stops when the run does instead of waiting on a log nobody will write
    to again.
    """
    from rich.console import Console

    from shinobi.offload.tracking import find, follow as follow_log, probe

    console = Console()
    try:
        launch = find(run, workdir)
    except LookupError:
        raise click.ClickException(f"no launch named {run!r}; `ninja runs` lists what this workspace has launched") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"cannot read handle {run!r}: {exc}") from None

    try:
        state = probe(launch)
    except ShinobiError as exc:
        raise click.ClickException(str(exc)) from None

    text, style = _state_style(state)
    console.print(f"[bold]{launch.name}[/bold] [dim]·[/dim] {launch.host}   [{style}]{text}[/{style}]   [dim]elapsed {_elapsed(launch, state)}[/dim]")
    if launch.log_path:
        console.print(f"[dim]{launch.log_path}[/dim]")
    console.rule(style="dim")

    # Following a run that has already finished would block on a `tail -F`
    # of a file nothing will append to again. Print its tail and stop.
    waiting = follow and state.running
    if follow and not state.running:
        console.print("[dim]run has already finished; showing the tail of its log[/dim]")

    stop = threading.Event()

    def _watch() -> None:
        """Poll for completion so the follower can stop on its own. `tail
        -F` never will: a finished run's log just goes quiet, which is
        indistinguishable from a slow step.
        """
        while not stop.wait(5.0):
            try:
                if not probe(launch).running:
                    stop.set()
                    return
            except ShinobiError:
                continue

    watcher = threading.Thread(target=_watch, daemon=True) if waiting else None
    if watcher:
        watcher.start()

    # `stop` goes *into* the follower rather than being polled between
    # lines out here: once the run ends the log goes quiet, so there is no
    # next line to check a flag after, and the read would block forever.
    stream = follow_log(launch, lines=lines, wait=waiting, stop=stop)
    try:
        for line in stream:
            console.print(line, highlight=False, markup=False)
    except KeyboardInterrupt:
        console.print("\n[dim]detached; the run keeps going[/dim]")
        return
    except ShinobiError as exc:
        raise click.ClickException(str(exc)) from None
    finally:
        stop.set()
        stream.close()

    if waiting:
        final = probe(launch)
        text, style = _state_style(final)
        console.rule(style="dim")
        console.print(f"[{style}]{text}[/{style}] [dim]after {_elapsed(launch, final)}[/dim]")


@main.command("download")
@click.option(
    "--cult-cargo",
    is_flag=True,
    help="Download cult-cargo cab definitions from GitHub.",
)
@click.option(
    "--dest-dir",
    type=click.Path(),
    default=".shinobi/cabs/cultcargo",
    help="Destination directory for downloaded cabs (default: .shinobi/cabs/cultcargo).",
)
@click.option(
    "--version",
    default="latest",
    help="Version to download: 'latest' (highest v* tag), tag name, branch name, or commit SHA.",
)
def download(cult_cargo: bool, dest_dir: str, version: str) -> None:
    """Download cab definitions from external sources.

    Currently supports:
      --cult-cargo: Download from caracal-pipeline/cult-cargo on GitHub

    Examples:
      ninja download --cult-cargo                    # Download latest stable (v0.2.1)
      ninja download --cult-cargo --version master   # Download from master branch
      ninja download --cult-cargo --version v0.2.0   # Download specific tag
      ninja download --cult-cargo --dest-dir ./my-cabs  # Custom destination
    """
    if not cult_cargo:
        raise click.ClickException("No source specified. Use --cult-cargo to download cult-cargo cabs.")

    from shinobi.download import download_cultcargo

    try:
        result = download_cultcargo(
            dest_dir=Path(dest_dir),
            version=version,
            exclude_images=True,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from None

    click.echo(f"Downloaded cult-cargo {result['version']}")
    click.echo(f"  Files: {result['file_count']}")
    click.echo(f"  Destination: {result['dest_dir']}")


if __name__ == "__main__":
    main()
