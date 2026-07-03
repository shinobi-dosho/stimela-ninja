from __future__ import annotations

import click

import shinobi
from shinobi.config import AppConfig


@click.group()
@click.option("--config", "config_file", default=None, help="Path to a config file.")
@click.option("--backend", "backend", default=None, help="Override the default backend.")
@click.pass_context
def main(ctx: click.Context, config_file: str | None, backend: str | None) -> None:
    """shinobi -- Stimela 3.0."""
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


if __name__ == "__main__":
    main()
