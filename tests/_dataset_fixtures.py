"""Shared lifecycle fixtures with optional CASA imports deferred until use."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shinobi.dataset_lifecycle import DatasetLifecycleAttempt, DatasetLifecycleStore

ROWS = 4


QUALITY_TABLES = ("QUALITY_BASELINE_STATISTIC", "QUALITY_FREQUENCY_STATISTIC", "QUALITY_TIME_STATISTIC", "QUALITY_KIND_NAME")


def add_quality_table(ms: Path, name: str, *, value: int = 1, directory: str | None = None) -> Path:
    """Attach a small opaque AOFlagger-style table by a real MAIN keyword."""
    tables = pytest.importorskip("casacore.tables", reason="requires measurement-set group")
    path = ms / (directory or name)
    with tables.table(str(path), tables.maketabdesc([tables.makescacoldesc("VALUE", 0)]), nrow=1, readonly=False, ack=False) as table:
        table.putcell("VALUE", 0, value)
    with tables.table(str(ms), readonly=False, ack=False) as main:
        main.putkeyword(name, f"Table: {path}")
    return path


def quality_values(ms: Path) -> dict[str, int]:
    tables = pytest.importorskip("casacore.tables", reason="requires measurement-set group")
    return {name: _quality_value(ms / name, tables) for name in QUALITY_TABLES}


def _quality_value(path: Path, tables) -> int:
    with tables.table(str(path), ack=False) as table:
        return int(table.getcell("VALUE", 0))


def make_ms(path: Path, *, scan: int = 1) -> Path:
    """A minimal valid MSv2: default subtables, a DATA column and a few rows."""
    tables = pytest.importorskip("casacore.tables", reason="requires measurement-set group")
    np = pytest.importorskip("numpy", reason="requires measurement-set group")
    if not hasattr(tables, "default_ms"):  # pragma: no cover - depends on installed build
        pytest.skip("installed python-casacore has no default_ms fixture builder")

    with tables.default_ms(str(path)) as ms:
        ms.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
        ms.addrows(ROWS)
        ms.putcol("SCAN_NUMBER", np.full(ROWS, scan, dtype=np.int32))
    return path


def scans(path: Path) -> list[int]:
    tables = pytest.importorskip("casacore.tables", reason="requires measurement-set group")
    with tables.table(str(path), ack=False) as ms:
        return [int(value) for value in ms.getcol("SCAN_NUMBER")]


def set_scans(path: Path, value: int) -> None:
    tables = pytest.importorskip("casacore.tables", reason="requires measurement-set group")
    np = pytest.importorskip("numpy", reason="requires measurement-set group")
    with tables.table(str(path), readonly=False, ack=False) as ms:
        ms.putcol("SCAN_NUMBER", np.full(ms.nrows(), value, dtype=np.int32))


def attempts(workspace: Path) -> list[DatasetLifecycleAttempt]:
    paths = sorted((workspace / ".shinobi" / "dataset-attempts").glob("*.json"), key=os.path.getmtime)
    return [DatasetLifecycleStore(path).attempt() for path in paths]
