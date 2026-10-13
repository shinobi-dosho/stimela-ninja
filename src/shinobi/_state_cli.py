"""CLI for explicit dataset-state operations; no optional stack imports at help."""

from __future__ import annotations

import json
from pathlib import Path

import click

from shinobi.config import AppConfig
from shinobi.dataset_state import DatasetStateStore, StateError
from shinobi.state_codecs import LosslessProfile


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
                identity = getattr(row, "state_id", None) or row.derived_id
                click.echo(f"{identity}  {row.representation_id}  {row.fidelity}  {'verified' if row.verified else 'metadata only'}")
                if getattr(row, "destination", None):
                    click.echo(f"Destination: {row.destination}")
                if row.attempt:
                    click.echo(f"Attempt: {row.attempt}")


@click.group(invoke_without_command=True)
@click.option("--store", type=click.Path(path_type=Path), help="State store directory (defaults to state.dir).")
@click.pass_context
def state(ctx, store):
    """Export, verify and materialize exact-logical local MSv2 states."""
    config = ctx.obj or AppConfig.load()
    try:
        ctx.obj = DatasetStateStore(store or config.state.dir, cache_dir=config.cache.dir)
    except StateError as exc:
        raise click.ClickException(str(exc)) from exc
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@state.command("export")
@click.argument("source", type=click.Path(path_type=Path))
@click.option("--block-rows", type=click.IntRange(min=1), help="Native payload rows per chunk.")
@click.option(
    "--codec",
    type=click.Choice(["zstd", "gzip", "blosc-zstd", "packbits-zstd", "delta-zstd", "packbits-delta-zstd"]),
    help="Explicit lossless MSv4 codec; semantic choices apply PackBits to bool and Delta to integers, with Zstd for other arrays.",
)
@click.option("--level", type=click.IntRange(1, 9), default=3)
@click.option("--shuffle", type=click.Choice(["none", "byte", "bit"]), default="none")
@click.option("--chunk-rows", type=click.IntRange(1, 1_048_576))
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def export(store, source, block_rows, codec, level, shuffle, chunk_rows, as_json):
    """Export a contained MSv2 under a shared read claim."""
    if codec is None and (level != 3 or shuffle != "none" or chunk_rows is not None):
        raise click.ClickException("--codec is required with codec options")
    try:
        profile = LosslessProfile(codec=codec, level=level, shuffle=shuffle, chunk_rows=chunk_rows) if codec else None
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _run(lambda: store.export(source, block_rows=block_rows, lossless_profile=profile), as_json)


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


@state.command("derive-precision")
@click.argument("state_id")
@click.option("--representation", "representation_id")
@click.option("--array", "arrays", multiple=True, required=True, help="Explicit MSv4 array path; repeat for each selected array.")
@click.option("--keep-bits", type=click.IntRange(0, 51), required=True)
@click.option("--atol", type=float, required=True)
@click.option("--rtol", type=float, required=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def derive_precision(store, state_id, representation_id, arrays, keep_bits, atol, rtol, as_json):
    """Round selected MSv4 floating arrays under an explicit numerical policy."""
    from shinobi.state_precision import PrecisionPolicy

    try:
        policy = PrecisionPolicy(arrays=tuple(sorted(arrays)), keep_bits=keep_bits, atol=atol, rtol=rtol)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _run(lambda: store.derive_precision(state_id, policy, representation_id=representation_id), as_json)


@state.command("verify-derived")
@click.argument("derived_id")
@click.option("--representation", "representation_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def verify_derived(store, derived_id, representation_id, as_json):
    """Independently verify derived bytes and the numerical policy."""
    _run(lambda: store.verify_derived(derived_id, representation_id), as_json)


@state.command("list-derived")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def list_derived(store, as_json):
    """List approximate artifacts, including their parent exact state."""
    _run(store.list_derived, as_json)


@state.command("recover-derived")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def recover_derived(store, as_json):
    """Settle interrupted derived publications by their recorded paths."""
    _run(store.recover_derived, as_json)
