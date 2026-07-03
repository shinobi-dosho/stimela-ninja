from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import click

import shinobi
from shinobi.backends import get_backend
from shinobi.backends.trace import TraceBackend, patch_all_backends
from shinobi.config import AppConfig
from shinobi.dag import render_dag
from shinobi.recipe import call
from shinobi.schema import CabDef, ParamSchema, is_file_like_dtype


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


def _resolve_target(target: str):
    """Resolve 'path/to/file.py:name' or 'dotted.module.path:name' into
    the CabDef or @recipe-decorated function it names.
    """
    if ":" not in target:
        raise click.ClickException(f"target must be 'path:name', got {target!r}")
    location, attr = target.rsplit(":", 1)

    if os.path.isfile(location):
        spec = importlib.util.spec_from_file_location(Path(location).stem, location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)

    try:
        return getattr(module, attr)
    except AttributeError:
        raise click.ClickException(f"no '{attr}' in {location}") from None


def _click_type_for_dtype(dtype: str):
    base = dtype.split(":", 1)[-1] if dtype.startswith("list:") else dtype
    if is_file_like_dtype(base):
        return click.Path()
    return {"int": click.INT, "float": click.FLOAT, "bool": click.BOOL}.get(base, click.STRING)


def _option_flag(schema_name: str) -> str:
    # ONLY a straight "_" -> "-" replace: click derives the callback kwarg
    # name from this flag string, and it must round-trip back to the exact
    # schema key used to call() the cab / call the recipe function.
    return "--" + schema_name.replace("_", "-")


def _build_options(inputs: dict[str, ParamSchema]) -> list[click.Option]:
    options = []
    for schema_name, schema in inputs.items():
        kwargs: dict = {"required": schema.required, "help": schema.info}
        if schema.dtype == "bool":
            kwargs.update(is_flag=True, default=bool(schema.default))
        else:
            # click.Option quirk: passing default=None explicitly (rather
            # than omitting it) silently defeats required=True -- it'll
            # invoke the callback with None instead of raising
            # MissingParameter. Only set it when there's a real default.
            if schema.default is not None:
                kwargs["default"] = schema.default
            kwargs["type"] = _click_type_for_dtype(schema.dtype)
            if schema.dtype.startswith("list:"):
                kwargs["multiple"] = True
        options.append(click.Option([_option_flag(schema_name)], **kwargs))
    return options


@main.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,  # let --help fall through to ctx.args for the
    # dynamically-built inner command below to handle -- the outer
    # command doesn't know TARGET's options yet, so it can't usefully
    # show its own --help either.
)
@click.argument("target")
@click.option(
    "--dryrun",
    is_flag=True,
    help="Show what would run as a graph, without actually running it.",
)
@click.pass_context
def run(ctx: click.Context, target: str, dryrun: bool) -> None:
    """Run a @cab or @recipe TARGET ('path/to/file.py:name' or 'pkg.mod:name').

    [OPTIONS] are derived from the target's own parameters -- run
    `ninja run TARGET --help` to see them.
    """
    obj = _resolve_target(target)

    if isinstance(obj, CabDef):
        inputs, help_text = obj.inputs, obj.info
    elif hasattr(obj, "__shinobi_recipe__"):
        inputs, help_text = obj.__shinobi_recipe__.inputs, obj.__shinobi_recipe__.info
    else:
        raise click.ClickException(f"{target!r} is neither a CabDef nor a @recipe function")

    def _callback(**kwargs):
        if isinstance(obj, CabDef):
            backend = TraceBackend() if dryrun else get_backend(ctx.obj.backend.default)
            result = call(obj, backend, **kwargs)
            if dryrun:
                click.echo(render_dag(backend.steps))
                return
            click.echo(result.stdout)
            if result.stderr:
                click.echo(result.stderr, err=True)
            if not result.success:
                raise click.ClickException(f"cab '{obj.name}' exited with status {result.returncode}")
        elif dryrun:
            tracer = TraceBackend()
            with patch_all_backends(tracer):
                try:
                    obj(**kwargs)
                except Exception as exc:  # noqa: BLE001 -- deliberately broad: a
                    # dry run substitutes placeholder values for real outputs, so
                    # a recipe doing real work with them (arithmetic, real file
                    # I/O, ...) can fail for reasons that have nothing to do with
                    # whether the recipe itself is correct. Report it and still
                    # show whatever was traced up to that point, rather than
                    # crashing the CLI.
                    click.echo(
                        f"(recipe raised during dry run, showing the trace up to this point: {exc!r})",
                        err=True,
                    )
            click.echo(render_dag(tracer.steps))
        else:
            obj(**kwargs)

    inner = click.Command(name=target, params=_build_options(inputs), callback=_callback, help=help_text)
    inner.main(args=ctx.args, prog_name=f"{ctx.info_name} {target}", standalone_mode=False)


if __name__ == "__main__":
    main()
