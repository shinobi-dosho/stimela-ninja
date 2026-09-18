"""Dependency-light pysteps for the physical M3 recovery workflow."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel


class MsOut(BaseModel):
    ms: Path


def _append(ms: Path, value: str) -> None:
    table = Path(ms) / "table.dat"
    table.write_text(table.read_text() + f"|{value}")


def image_v1(ctx, ms: Path) -> MsOut:
    """CASA-shaped image computation using the supported context imports."""
    join = ctx.import_callable("join", "os.path")
    from tests.slurm_physical.m3_helper import image_label

    table = Path(join(str(ms), "table.dat"))
    table.write_text(table.read_text() + f"|{image_label('v1')}")
    return MsOut(ms=ms)


def image_v2(ctx, ms: Path) -> MsOut:
    """A source change with the same declared input/output contract."""
    join = ctx.import_callable("join", "os.path")
    from tests.slurm_physical.m3_helper import image_label

    table = Path(join(str(ms), "table.dat"))
    table.write_text(table.read_text() + f"|{image_label('v2')}")
    return MsOut(ms=ms)


def image_requeue(ctx, ms: Path) -> MsOut:
    """Leave a partial mutation once, then wait to be requeued by Slurm."""
    os_module = ctx.import_module("os")
    _requeue_once(ms, "image", os_module.getpid())
    _append(ms, "image[requeued]")
    return MsOut(ms=ms)


def image_boundary_s2(ctx, ms: Path) -> MsOut:
    """Physical pre-oracle commit-boundary probe."""
    ctx.import_callable("join", "os.path")
    _append(ms, "boundary[S2]")
    return MsOut(ms=ms)


def image_boundary_w_result(ctx, ms: Path) -> MsOut:
    """Physical post-oracle/cache-index-boundary probe."""
    ctx.import_callable("join", "os.path")
    _append(ms, "boundary[W_RESULT]")
    return MsOut(ms=ms)


def venv_v1(ms: Path) -> MsOut:
    import venvonlypkg

    _append(ms, f"venv[{venvonlypkg.MAGIC}]")
    return MsOut(ms=ms)


def venv_requeue(ms: Path) -> MsOut:
    import os
    import venvonlypkg

    _requeue_once(ms, "venv", os.getpid())
    _append(ms, f"venv-requeued[{venvonlypkg.MAGIC}]")
    return MsOut(ms=ms)


def _requeue_once(ms: Path, kind: str, pid: int) -> None:
    root = Path(ms).parent
    armed = root / f".{kind}-requeue-once"
    if not armed.exists():
        return
    armed.unlink()
    _append(ms, f"PARTIAL-{kind}")
    (root / f".{kind}-requeue-ready").write_text(str(pid))
    # The physical driver issues `scontrol requeue` while this process is
    # alive. A timeout makes a failed driver obvious instead of hanging the
    # allocation forever.
    time.sleep(120)
    raise RuntimeError(f"{kind} worker was not requeued")
