from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import click

import shinobi
from shinobi.clickutil import build_options
from shinobi.config import AppConfig
from shinobi.dag import graph_nodes, render_dag
from shinobi.graph import RecipeGraphError, RecipeNotOffloadableError
from shinobi.offload import OffloadCompileError, compile_slurm, status_slurm, status_ssh, submit_slurm
from shinobi.policies import build_argv
from shinobi.steps.dispatch import _dispatch, _prepare_inputs
from shinobi.steps.schema import Recipe, Scope, StepRef


@click.group()
@click.option("--config", "config_file", default=None, help="Path to a config file.")
@click.option("--backend", "backend", default=None, help="Override the default backend.")
@click.pass_context
def main(ctx: click.Context, config_file: str | None, backend: str | None) -> None:
    """ninja -- the shinobi (Stimela 3.0) CLI."""
    overrides: dict = {}
    if backend:
        overrides["backend"] = {"default": backend}
    ctx.obj = AppConfig.load(config_file=config_file, **overrides)
    ctx.meta["backend_override"] = backend


@main.command()
def version() -> None:
    """Print the shinobi version."""
    click.echo(shinobi.__version__)


@main.command("cab")
@click.argument("cab_file")
@click.argument("cab_name")
def show_cab(cab_file: str, cab_name: str) -> None:
    """Show a cab's schema, as loaded from a cult-cargo style YAML FILE."""
    from shinobi.loaders.cultcargo import load_file

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


def _run_remote(
    ctx: click.Context,
    target: str,
    *,
    dryrun: bool,
    cache_dir: str | None,
    no_cache: bool,
    remote: str,
    add_venv: bool,
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
        raise click.ClickException(
            "--cache-dir/--no-cache apply to local runs only; configure caching via the "
            "remote host's own AppConfig"
        )
    if ":" not in target:
        raise click.ClickException(f"target must be 'path:name', got {target!r}")
    location, attr = target.rsplit(":", 1)
    if not os.path.isfile(location):
        raise click.ClickException(
            "--remote only supports a local file target ('path/to/file.py:name'), not a "
            "dotted module path"
        )

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
            "note: only the target file and any statically-discovered cab deps are synced; "
            "use --include for anything else the recipe reads",
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
        raise click.ClickException(
            "target file and its deps don't share a common directory; pass a narrower "
            "--include or restructure the recipe"
        )

    try:
        remote_spec = parse_remote(remote)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None

    rel_paths = [p.relative_to(base_dir) for p in all_paths]
    sync_to_remote(base_dir, rel_paths, remote_spec)

    remote_target = f"{pyfile.relative_to(base_dir)}:{attr}"
    handle = launch_remote(remote_spec, remote_target, ctx.args, add_venv=add_venv)

    handle_path = _handle_path(None, f"{pyfile.stem}.{attr}")
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle_path.write_text(json.dumps({"engine": "ssh", **handle.__dict__}, indent=2))
    click.echo(f"launched on {remote_spec.host} (detached, pid={handle.pid})")
    click.echo(f"  handle: {handle_path}")
    click.echo(f"  log: ssh {remote_spec.host} tail -f {remote_spec.path}/{handle.log_file}")


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
    "--remote",
    "remote",
    default=None,
    help="Launch on a remote host instead of locally: 'user@host:/path'. Syncs the target "
    "file and its statically-discoverable cab deps, then runs detached -- see `ninja status`.",
)
@click.option(
    "--add-venv/--no-add-venv",
    "add_venv",
    default=True,
    help="With --remote, source venv/bin/activate or .venv/bin/activate under the remote "
    "path before running, if present.",
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
    remote: str | None,
    add_venv: bool,
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
            ctx, target, dryrun=dryrun, cache_dir=cache_dir, no_cache=no_cache,
            remote=remote, add_venv=add_venv, include_paths=include_paths,
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
        raise click.ClickException(
            f"{target!r} is neither a Cab, Recipe, nor a @shinobi.step function"
        )

    def _callback(**kwargs):
        # Drop options the user didn't provide (None) so the inputs_model's
        # own defaults apply; per-step constants from a StepRef go under them.
        call_kwargs = {**params}
        for name, value in kwargs.items():
            if value is not None:
                call_kwargs[name] = value

        if dryrun:
            if isinstance(scope, Recipe):
                try:
                    click.echo(render_dag(graph_nodes(scope)))
                except RecipeGraphError as exc:
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
        result = _dispatch(
            scope, func, backend=backend, cache=cache, cache_dir=cache_dir, stream=stream,
            _config=ctx.obj, **call_kwargs
        )
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
            raise click.ClickException(
                f"'{scope.name}' exited with status {result.returncode}"
            )

    inner = click.Command(
        name=target,
        params=build_options(scope.inputs_model),
        callback=_callback,
        help=scope.info,
    )
    inner.main(args=ctx.args, prog_name=f"{ctx.info_name} {target}", standalone_mode=False)


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
@click.pass_context
def compile_recipe(
    ctx: click.Context,
    target: str,
    engine: str,
    workdir: str | None,
    container_runtime: str,
    submit: bool,
) -> None:
    """Compile a Recipe TARGET ('path/to/file.py:name' or 'pkg.mod:name')
    into a cluster workflow and, with --submit, hand it off and detach.

    Only purely-declarative recipes can be offloaded; anything relying on
    live Python (orchestration functions, MUTABLE inputs, non-path data
    flow) is rejected with an explanation -- run those locally via `ninja
    run`. [OPTIONS] carry the recipe's own inputs; run
    `ninja compile TARGET --help` to see them.
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

    def _callback(**kwargs):
        inputs = {name: value for name, value in kwargs.items() if value is not None}
        try:
            workflow = compile_slurm(recipe, inputs, workdir=workdir, container_runtime=runtime)
        except (RecipeNotOffloadableError, OffloadCompileError, RecipeGraphError) as exc:
            raise click.ClickException(str(exc)) from None

        if not submit:
            for job in workflow.jobs:
                dep = f"  (afterok: {', '.join(job.depends_on)})" if job.depends_on else ""
                click.echo(f"# ===== {job.name}{dep} =====")
                click.echo(job.script)
            return

        job_ids = submit_slurm(workflow, workdir=workdir)
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
    if engine == "slurm":
        for name, state in status_slurm(data["jobs"]).items():
            click.echo(f"{name}: {state}")
    elif engine == "ssh":
        click.echo(status_ssh(data))
    else:
        raise click.ClickException(f"unknown engine in handle: {engine!r}")


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
        raise click.ClickException(
            "No source specified. Use --cult-cargo to download cult-cargo cabs."
        )

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
