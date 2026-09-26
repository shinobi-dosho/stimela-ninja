"""Lazy boundary to the qualified MSv4 exporter and exact-native implementation.

msutils owns all scientific hashing, preservation and native writing. This
adapter only selects the qualified public APIs and records their environment.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import sys
import warnings
from pathlib import Path

MSUTILS_COMMIT = "91675064fbe466369a775286a8e35fccb591d883"
ADAPTER = "shinobi-xarray-ms-native/v1"
MSV4_SCHEMA = "xarray-ms-0.5.8-msv4/v1"
PACKAGES = ("stimela-ninja", "msutils", "xarray-ms", "xarray", "zarr", "numcodecs", "dask-ms", "arcae", "numpy", "python-casacore", "dask", "pyarrow", "pandas")


def runtime() -> dict[str, str]:
    """Refuse unqualified builds, including msutils 3.0.0 before preservation v2."""
    try:
        versions = {name: metadata.version(name) for name in PACKAGES}
        direct = json.loads(metadata.distribution("msutils").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise ImportError("state operations require the pinned stimela-ninja[state] extra") from exc
    required = {
        "msutils": "3.0.0",
        "xarray-ms": "0.5.8",
        "xarray": "2026.7.0",
        "numcodecs": "0.16.5",
        "dask-ms": "0.2.32",
        "arcae": "0.5.4",
        "zarr": "3.1.6" if sys.version_info < (3, 12) else "3.3.0",
    }
    if any(versions[name] != version for name, version in required.items()) or direct.get("vcs_info", {}).get("commit_id") != MSUTILS_COMMIT:
        raise ImportError("unqualified state stack: install the pinned stimela-ninja[state] extra (including the msutils commit)")
    import msutils

    if not all(callable(getattr(msutils, name, None)) for name in ("native_logical_id", "capture_native_preservation", "verify_native_preservation", "to_msv2")):
        raise ImportError("msutils does not supply preservation v2 public APIs")
    return {**versions, "msutils_commit": MSUTILS_COMMIT}


def native_id(source: Path) -> str:
    from msutils import native_logical_id

    return native_logical_id(source)


def export(source: Path, destination: Path) -> list[str]:
    """Export a full MS through xarray-ms; retain irregular-grid/imputation evidence."""
    import xarray as xr

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with xr.open_datatree(
            source, engine="xarray-ms:msv2", auto_corrs=True, partition_schema=["DATA_DESC_ID", "FIELD_ID"], driver="arcae", driver_kwargs={"cache_size": 64}, chunks={}
        ) as tree:
            tree.to_zarr(destination, mode="w", compute=True, consolidated=True, zarr_format=3)
    return sorted({f"{item.category.__name__}: {item.message}" for item in emitted})


def capture(source: Path, msv4: Path, bundle: Path, block_rows: int | None) -> None:
    from msutils import capture_native_preservation

    capture_native_preservation(source, msv4, bundle, block_rows=block_rows)


def verify(msv4: Path, bundle: Path):
    from msutils import verify_native_preservation

    return verify_native_preservation(msv4, bundle)


def materialize(msv4: Path, bundle: Path, destination: Path, state_id: str) -> None:
    from msutils import to_msv2

    to_msv2(msv4, destination, fidelity="exact-native-v1", preservation=bundle, expected_native_logical_id=state_id)
