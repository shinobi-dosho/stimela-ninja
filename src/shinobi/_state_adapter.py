"""Lazy boundary to the qualified MSv4 exporter and exact-native implementation.

msutils owns all scientific hashing, preservation and native writing. This
adapter only selects the qualified public APIs and records their environment.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import sys
import threading
import warnings
from pathlib import Path

from shinobi.state_codecs import MAX_TILE_BYTES, bounded_chunks

MSUTILS_COMMIT = "2856d3ae357c4c2ec70060df641fe48e32ff8288"
ADAPTER = "shinobi-xarray-ms-native/v1"
MSV4_SCHEMA = "xarray-ms-0.5.8-msv4/v1"
PACKAGES = ("stimela-ninja", "msutils", "xarray-ms", "xarray", "zarr", "numcodecs", "dask-ms", "arcae", "numpy", "python-casacore", "dask", "pyarrow", "pandas")
_EXPORT_WARNING_LOCK = threading.Lock()


def _preferred_chunks(tree) -> dict[str, int]:
    """One safe backend read-chunk map for all partitions of a lazy DataTree."""
    preferred: dict[str, int] = {}
    for node in tree.subtree:
        for variable in node.ds.variables.values():
            if variable.dtype.kind not in "biufc" or not variable.shape:
                continue
            if variable.dtype.itemsize * __import__("math").prod(variable.shape) <= MAX_TILE_BYTES:
                continue
            chunks = bounded_chunks(variable.shape, variable.shape, variable.dtype.itemsize)
            for dim, chunk in zip(variable.dims, chunks, strict=True):
                preferred[dim] = min(preferred.get(dim, chunk), chunk)
    return preferred


def _check_export_chunks(tree) -> None:
    """Refuse any numeric backend or physical Zarr chunk above the bound."""
    import math

    for node in tree.subtree:
        for name, variable in node.ds.variables.items():
            if variable.dtype.kind not in "biufc" or not variable.shape:
                continue
            chunks = variable.chunks
            if chunks is None:
                if math.prod(variable.shape) * variable.dtype.itemsize > MAX_TILE_BYTES:
                    raise ValueError(f"MSv4 variable {node.path}/{name} is not chunked")
            else:
                largest = math.prod(max(axis) for axis in chunks) * variable.dtype.itemsize
                if largest > MAX_TILE_BYTES:
                    raise ValueError(f"MSv4 backend chunk for {node.path}/{name} exceeds 16 MiB")
            encoded = variable.encoding.get("chunks")
            if encoded is not None and math.prod(encoded) * variable.dtype.itemsize > MAX_TILE_BYTES:
                # Only a physical chunk override is stale; preserve dtype,
                # units, scales and all other scientific encodings.
                variable.encoding.pop("chunks")


def runtime() -> dict[str, str]:
    """Refuse unqualified builds, including msutils 3.0.0 before preservation v2."""
    try:
        versions = {name: metadata.version(name) for name in PACKAGES if name != "dask-ms"}
        direct = json.loads(metadata.distribution("msutils").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise ImportError("state operations require the pinned measurement-set development group") from exc
    required = {
        "msutils": "3.0.0",
        "xarray-ms": "0.5.8",
        "xarray": "2026.7.0",
        "numcodecs": "0.16.5",
        "arcae": "0.5.4",
        "zarr": "3.1.6" if sys.version_info < (3, 12) else "3.3.0",
    }
    if any(versions[name] != version for name, version in required.items()) or direct.get("vcs_info", {}).get("commit_id") != MSUTILS_COMMIT:
        raise ImportError("unqualified state stack: install the pinned measurement-set development group (including the msutils commit)")
    # Keep the historical record key so old manifests remain parseable. The
    # arcae writer does not require dask-ms, but exact recorded-stack checks
    # still distinguish its presence from absence when reopening a state.
    try:
        versions["dask-ms"] = metadata.version("dask-ms")
    except metadata.PackageNotFoundError:
        versions["dask-ms"] = "not-installed"
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

    # warnings.catch_warnings mutates process-global filter state on Python
    # 3.11, so concurrent exports must not enter it together.
    with _EXPORT_WARNING_LOCK, warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        options = {
            "engine": "xarray-ms:msv2",
            "auto_corrs": True,
            "partition_schema": ["DATA_DESC_ID", "FIELD_ID"],
            "driver": "arcae",
            "driver_kwargs": {"cache_size": 64},
            "chunks": {},
        }
        with xr.open_datatree(source, **options) as metadata_tree:
            preferred = _preferred_chunks(metadata_tree)
        with xr.open_datatree(source, preferred_chunks=preferred, **options) as tree:
            _check_export_chunks(tree)
            tree.to_zarr(destination, mode="w", compute=True, consolidated=True, zarr_format=3)
    return sorted({f"{item.category.__name__}: {item.message}" for item in emitted})


def capture(source: Path, msv4: Path, bundle: Path, block_rows: int | None, *, semantic_filters: str | None = None) -> None:
    from msutils import capture_native_preservation

    options = {"semantic_filters": semantic_filters} if semantic_filters is not None else {}
    capture_native_preservation(source, msv4, bundle, block_rows=block_rows, **options)


def verify(msv4: Path, bundle: Path):
    from msutils import verify_native_preservation

    return verify_native_preservation(msv4, bundle)


def materialize(msv4: Path, bundle: Path, destination: Path, state_id: str) -> None:
    from msutils import to_msv2

    to_msv2(msv4, destination, fidelity="exact-native-v1", preservation=bundle, expected_native_logical_id=state_id)
