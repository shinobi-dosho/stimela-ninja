from __future__ import annotations

import importlib
import importlib.util
import os
import types
from pathlib import Path
from typing import Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import shinobi
from shinobi.config import AppConfig
from shinobi.dag import graph_nodes, render_dag
from shinobi.graph import RecipeGraphError
from shinobi.policies import build_argv
from shinobi.steps.dispatch import _dispatch, _prepare_inputs
from shinobi.steps.schema import Recipe, Scope, StepRef, _unwrap_annotation, path_fields


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


def _resolve_target(target: str):
    """Resolve 'path/to/file.py:name' or 'dotted.module.path:name' into the
    Scope or StepRef it names.
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


def _is_list(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_is_list(arg) for arg in get_args(annotation))
    return origin in (list, tuple)


def _click_type(annotation, is_path: bool):
    if is_path:
        return click.Path()
    for leaf in _unwrap_annotation(annotation):
        if leaf in (int, float, bool, str):
            return {int: click.INT, float: click.FLOAT, bool: click.BOOL, str: click.STRING}[leaf]
    return click.STRING


def _option_flag(field_name: str) -> str:
    # ONLY a straight "_" -> "-" replace: click derives the callback kwarg
    # name from this flag string, and it must round-trip back to the exact
    # model field name used to dispatch.
    return "--" + field_name.replace("_", "-")


def _build_options(model: type[BaseModel]) -> list[click.Option]:
    paths = path_fields(model)
    options = []
    for name, field in model.model_fields.items():
        required = field.is_required()
        default = None if field.default is PydanticUndefined else field.default
        kwargs: dict = {"required": required, "help": field.description}
        leaves = _unwrap_annotation(field.annotation)
        is_list = _is_list(field.annotation)
        if bool in leaves and not is_list:
            kwargs.update(is_flag=True, default=bool(default))
        else:
            if default is not None:
                kwargs["default"] = default
            kwargs["type"] = _click_type(field.annotation, name in paths)
            if is_list:
                kwargs["multiple"] = True
        options.append(click.Option([_option_flag(name)], **kwargs))
    return options


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
@click.pass_context
def run(ctx: click.Context, target: str, dryrun: bool) -> None:
    """Run a Cab, Recipe, or @shinobi.step TARGET ('path/to/file.py:name'
    or 'pkg.mod:name').

    [OPTIONS] are derived from the target's own parameters -- run
    `ninja run TARGET --help` to see them.
    """
    if target in ("-h", "--help"):
        click.echo(ctx.get_help())
        ctx.exit()

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
        result = _dispatch(scope, func, backend=backend, _config=ctx.obj, **call_kwargs)
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
        params=_build_options(scope.inputs_model),
        callback=_callback,
        help=scope.info,
    )
    inner.main(args=ctx.args, prog_name=f"{ctx.info_name} {target}", standalone_mode=False)


if __name__ == "__main__":
    main()
