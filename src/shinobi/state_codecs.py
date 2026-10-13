"""Explicit, lossless MSv4 representation profiles.

This module has no optional imports at import time. Rewrites take place only
inside a caller's private staging directory; the native preservation sidecar
is captured after the rewritten MSv4 tree has passed its logical-ID check.
"""

from __future__ import annotations

import shutil
import time
from itertools import product
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TILE_BYTES = 16 * 1024 * 1024


class LosslessProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["shinobi-msv4-lossless-profile/v1"] = "shinobi-msv4-lossless-profile/v1"
    codec: Literal["zstd", "gzip", "blosc-zstd", "packbits-zstd", "delta-zstd", "packbits-delta-zstd"] = "zstd"
    level: int = Field(default=3, ge=1, le=9)
    shuffle: Literal["none", "byte", "bit"] = "none"
    chunk_rows: int | None = Field(default=None, ge=1, le=1_048_576)

    @property
    def semantic_filters(self) -> Literal["packbits", "delta", "packbits-delta"] | None:
        """Optional native-payload filter selection, absent from profile JSON."""
        return {"packbits-zstd": "packbits", "delta-zstd": "delta", "packbits-delta-zstd": "packbits-delta"}.get(self.codec)

    @model_validator(mode="after")
    def codec_options(self):
        if self.codec != "blosc-zstd" and self.shuffle != "none":
            raise ValueError("shuffle is available only with blosc-zstd")
        return self


def bounded_chunks(shape: tuple[int, ...], chunks: tuple[int, ...], itemsize: int, *, limit: int = MAX_TILE_BYTES) -> tuple[int, ...]:
    """Bound each numeric tile even when the exporter chose wide trailing axes."""
    result = [max(1, min(int(c), max(1, int(s)))) for s, c in zip(shape, chunks, strict=True)]
    if itemsize > limit:
        raise ValueError("one array element exceeds the tile limit")
    while itemsize * __import__("math").prod(result) > limit:
        largest = max(range(len(result)), key=lambda i: result[i])
        result[largest] = max(1, result[largest] // 2)
    return tuple(result)


def tile_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    """Visit all N-dimensional tiles, including arrays with zero-size axes."""
    if not shape:
        yield ()
        return
    for offsets in product(*(range(0, size, chunk) for size, chunk in zip(shape, chunks, strict=True))):
        yield tuple(slice(start, min(start + chunk, size)) for start, chunk, size in zip(offsets, chunks, shape, strict=True))


def _compressor(profile: LosslessProfile):
    from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

    if profile.codec in ("zstd", "packbits-zstd", "delta-zstd", "packbits-delta-zstd"):
        return ZstdCodec(level=profile.level)
    if profile.codec == "gzip":
        return GzipCodec(level=profile.level)
    return BloscCodec(cname="zstd", clevel=profile.level, shuffle={"none": "noshuffle", "byte": "shuffle", "bit": "bitshuffle"}[profile.shuffle])


def _filters(profile: LosslessProfile, dtype):
    if dtype.kind == "b" and profile.semantic_filters in ("packbits", "packbits-delta"):
        from zarr.codecs import PackBits

        return [PackBits()]
    if dtype.kind in "iu" and profile.semantic_filters in ("delta", "packbits-delta"):
        from zarr.codecs import Delta

        # Zarr serializes a dtype string; passing np.dtype fails on 3.3.0.
        return [Delta(dtype=dtype.name)]
    return []


def rewrite_lossless(source: Path, destination: Path, profile: LosslessProfile, *, timings: dict | None = None) -> str:
    """Copy an MSv4 tree with a selected compressor and prove exact logical equality.

    Reading and writing one leading-axis chunk at a time bounds numeric array
    memory. As with msutils, variable-length strings can still expand sharply.
    """
    import msutils
    import zarr

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not isinstance(profile, LosslessProfile):
        profile = LosslessProfile.model_validate(profile)
    before = msutils.logical_id(source)
    original = zarr.open_group(source, mode="r", use_consolidated=False)
    rewritten = zarr.open_group(destination, mode="w-", zarr_format=3)
    rewritten.attrs.update(original.attrs.asdict())

    per_array_seconds: dict[str, float] = {}
    per_array_peak_rss: dict[str, int] = {}

    def copy_group(src, dst, prefix=""):
        for name, group in src.groups():
            child = dst.create_group(name, attributes=group.attrs.asdict())
            copy_group(group, child, f"{prefix}{name}/")
        for name, array in src.arrays():
            array_started = time.perf_counter()
            shape = array.shape
            chunks = array.chunks
            if shape and profile.chunk_rows is not None:
                chunks = (min(shape[0] or 1, profile.chunk_rows), *chunks[1:])
            if shape and array.dtype.kind in "biufc":
                chunks = bounded_chunks(shape, chunks, array.dtype.itemsize)
            filters = _filters(profile, array.dtype)
            serializer = {}
            if filters and array.dtype.kind in "iu":
                import sys
                from zarr.codecs import BytesCodec

                serializer["serializer"] = BytesCodec(endian=sys.byteorder)
            created = dst.create_array(
                name,
                shape=shape,
                dtype=array.dtype,
                chunks=chunks,
                filters=filters,
                **serializer,
                compressors=[_compressor(profile)],
                config={"order": "C", "write_empty_chunks": True} if profile.semantic_filters is not None else {"write_empty_chunks": True},
                fill_value=array.fill_value,
                attributes=array.attrs.asdict(),
                dimension_names=array.metadata.dimension_names,
            )
            for region in tile_slices(shape, chunks):
                values = array[region]
                if profile.semantic_filters is not None and shape:
                    import numpy as np

                    values = np.ascontiguousarray(values)
                created[region] = values
            per_array_seconds[f"{prefix}{name}"] = time.perf_counter() - array_started
            if timings is not None:
                import resource

                per_array_peak_rss[f"{prefix}{name}"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    try:
        started = time.perf_counter()
        copy_group(original, rewritten)
        if timings is not None:
            timings["copy_encode_seconds"] = time.perf_counter() - started
            timings["per_array_copy_encode_seconds"] = per_array_seconds
            timings["per_array_worker_highwater_rss_kib"] = per_array_peak_rss
        after = msutils.logical_id(destination)
        if after != before:
            raise ValueError("lossless rewrite changed the MSv4 logical ID")
        return after
    except Exception:
        shutil.rmtree(destination)
        raise
