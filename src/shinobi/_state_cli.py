"""CLI for explicit dataset-state operations; no optional stack imports at help."""

from __future__ import annotations

import json
from pathlib import Path

import click

from shinobi.config import AppConfig
from shinobi.dataset_state import DatasetStateStore, StateError


def _run(operation, as_json):
    try:
        result = operation()
    except StateError as exc:
        raise click.ClickException(str(exc)) from exc
    rows = result if isinstance(result, list) else [result]
    if as_json:
        payload = [row.model_dump(mode="json") for row in rows]
        click.echo(json.dumps(payload if isinstance(result, list) else payload[0], indent=2))
    else:
        for row in rows:
            if hasattr(row, "phase"):
                click.echo(f"{row.attempt_id}: {row.phase}")
            else:
                click.echo(f"{row.state_id}  {row.representation_id}  exact-logical  {'verified' if row.verified else 'metadata only'}")
                if row.destination:
                    click.echo(f"Destination: {row.destination}")
                if row.attempt:
                    click.echo(f"Attempt: {row.attempt}")


@click.group()
@click.option("--store", type=click.Path(path_type=Path), help="State store directory (defaults to state.dir).")
@click.pass_context
def state(ctx, store):
    """Export, verify and materialize exact-logical local MSv2 states."""
    config = ctx.find_object(AppConfig) or AppConfig.load()
    try:
        ctx.obj = DatasetStateStore(store or config.state.dir, cache_dir=config.cache.dir)
    except StateError as exc:
        raise click.ClickException(str(exc)) from exc


@state.command("export")
@click.argument("source", type=click.Path(path_type=Path))
@click.option("--block-rows", type=click.IntRange(min=1), help="Native payload rows per chunk.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def export(store, source, block_rows, as_json):
    """Export a contained MSv2 under a shared read claim."""
    _run(lambda: store.export(source, block_rows=block_rows), as_json)


@state.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def list_states(store, as_json):
    """List committed metadata (does not verify data)."""
    _run(store.list, as_json)


@state.command("verify")
@click.argument("state_id")
@click.option("--representation", "representation_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def verify(store, state_id, representation_id, as_json):
    """Fully verify stored bytes and native/MSv4 logical identities."""
    _run(lambda: store.verify(state_id, representation_id), as_json)


@state.command("materialize")
@click.argument("state_id")
@click.argument("destination", type=click.Path(path_type=Path))
@click.option("--representation", "representation_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def materialize(store, state_id, destination, representation_id, as_json):
    """Create a fresh MSv2, independently validate, then publish."""
    _run(lambda: store.materialize(state_id, destination, representation_id=representation_id), as_json)


@state.command("recover")
@click.option("--destination", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def recover(store, destination, as_json):
    """Settle dead attempts and sweep only their recorded private staging."""
    _run(lambda: store.recover(destination), as_json)
