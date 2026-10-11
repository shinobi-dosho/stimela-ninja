"""Reproducible MS column codec measurements; never publishes a state.

Use --ms for a read-only, bounded beginning/middle/end row sample from a real
MSv2. Without --ms, generate deterministic synthetic column classes. Results
are JSON with exact input selection, sample hashes, versions, logical IDs,
encoded byte counts, throughput, peak process RSS and offline candidates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import platform
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from shinobi.state_codecs import LosslessProfile, rewrite_lossless


def _tree_bytes(path: Path) -> int:
    return sum(member.stat().st_size for member in path.rglob("*") if member.is_file())


def _root_observation(path: Path):
    digest = hashlib.sha256()
    count = 0
    for member in sorted(path.rglob("*")):
        info = member.lstat()
        digest.update(f"{member.relative_to(path)}\0{info.st_mode}\0{info.st_size}\0{info.st_mtime_ns}\0".encode())
        count += 1
    return {"entries": count, "stat_sha256": digest.hexdigest()}


def _sample_hash(value: np.ndarray) -> str:
    if value.dtype.hasobject:
        payload = json.dumps(value.tolist(), sort_keys=True).encode()
    else:
        payload = value.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _synthetic(seed: int, rows: int):
    rng = np.random.default_rng(seed)
    shape = (rows, 32, 4)
    visibility = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")
    visibility.flat[0] = complex(np.nan, np.inf)
    weight = rng.lognormal(size=shape).astype("float32")
    weight.flat[0] = np.float32(-0.0)
    flags = rng.random(shape) < 0.18
    sparse_flags = rng.random(shape) < 0.002
    dense_flags = rng.random(shape) < 0.85
    burst_flags = np.zeros(shape, dtype=bool)
    burst_flags[rows // 3 : rows // 3 + max(1, rows // 12)] = True
    return {
        "VISIBILITY": visibility,
        "WEIGHT_SPECTRUM": weight,
        "FLAG": flags,
        "FLAG_SPARSE": sparse_flags,
        "FLAG_DENSE": dense_flags,
        "FLAG_BURST": burst_flags,
        "UVW": rng.normal(size=(rows, 3)).astype("float64"),
        "TIME": np.arange(rows, dtype="float64") * 8 + 5.2e9,
        "ANTENNA1": rng.integers(0, 64, size=rows, dtype="int32"),
        "ROW_ID_RAMP": np.arange(rows, dtype="int64"),
        "FIELD_LABEL": np.asarray([f"field-{i % 7}" for i in range(rows)], dtype="U12"),
    }, {"kind": "synthetic", "seed": seed, "rows": rows}


def _real(path: Path, rows_per_region: int, flag_versions: Path | None):
    from casacore.tables import table

    columns = ("DATA", "MODEL_DATA", "CORRECTED_DATA", "WEIGHT_SPECTRUM", "WEIGHT", "SIGMA", "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2")
    with table(str(path), readonly=True, ack=False, lockoptions="usernoread") as main:
        count = main.nrows()
        starts = sorted({0, max(0, count // 2 - rows_per_region // 2), max(0, count - rows_per_region)})
        windows = [(start, min(rows_per_region, count - start)) for start in starts]
        available = set(main.colnames())
        arrays = {}
        for name in columns:
            if name not in available:
                continue
            parts = [np.asarray(main.getcol(name, startrow=start, nrow=length)) for start, length in windows if length]
            arrays[name] = np.concatenate(parts, axis=0)
    for subtable in ("ANTENNA", "FIELD", "SPECTRAL_WINDOW"):
        member = path / subtable
        if not member.is_dir():
            continue
        with table(str(member), readonly=True, ack=False, lockoptions="usernoread") as metadata:
            if "NAME" in metadata.colnames():
                labels = [str(item) for item in metadata.getcol("NAME")]
                width = max((len(item) for item in labels), default=1)
                arrays[f"{subtable}_NAME"] = np.asarray(labels, dtype=f"U{max(1, width)}")
    versions = []
    if flag_versions is not None:
        for i, member in enumerate(sorted(flag_versions.glob("flags.*"))):
            if not member.is_dir():
                continue
            with table(str(member), readonly=True, ack=False, lockoptions="usernoread") as snapshot:
                if snapshot.nrows() != count:
                    raise ValueError(f"flag version row count differs: {member}")
                for column in ("FLAG", "FLAG_ROW"):
                    if column not in snapshot.colnames():
                        continue
                    parts = [np.asarray(snapshot.getcol(column, startrow=start, nrow=length)) for start, length in windows if length]
                    arrays[f"FLAGVERSION_{i}_{column}"] = np.concatenate(parts, axis=0)
            versions.append({"index": i, "name": member.name})
    return arrays, {
        "kind": "real-msv2-sample",
        "path": str(path.resolve()),
        "total_rows": count,
        "windows": windows,
        "columns": sorted(arrays),
        "flag_versions": versions,
        "flag_versions_path": str(flag_versions.resolve()) if flag_versions else None,
    }


def _codec_result(source: Path, root: Path, arrays, profile: LosslessProfile, label: str):

    target = root / label
    started = time.perf_counter()
    timings = {}
    logical_id = rewrite_lossless(source, target, profile, timings=timings)
    rewrite_verified_seconds = time.perf_counter() - started
    import zarr

    group = zarr.open_group(target, mode="r", use_consolidated=False)
    decoded = {}
    per_column_decode = {}
    started = time.perf_counter()
    for name in arrays:
        column_started = time.perf_counter()
        decoded[name] = np.asarray(group[name][:])
        per_column_decode[name] = time.perf_counter() - column_started
    decode_seconds = time.perf_counter() - started
    for name, original in arrays.items():
        actual = decoded[name]
        if original.dtype.kind in "biufc":
            same = np.array_equal(original.view(np.uint8), actual.view(np.uint8))
        else:
            same = np.array_equal(original, actual)
        if not same:
            raise ValueError(f"lossless round trip changed {name}")
    total = sum(array.nbytes for array in arrays.values())
    return {
        "profile": profile.model_dump(mode="json"),
        "msv4_logical_id": logical_id,
        "encoded_bytes": _tree_bytes(target),
        "copy_encode_seconds": timings["copy_encode_seconds"],
        "rewrite_verified_seconds": rewrite_verified_seconds,
        "decode_seconds": decode_seconds,
        "copy_encode_mib_s": total / 1048576 / timings["copy_encode_seconds"],
        "decode_mib_s": total / 1048576 / decode_seconds,
        "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_column": {
            name: {
                "raw_bytes": value.nbytes,
                "sample_sha256": _sample_hash(value),
                "encoded_bytes": _tree_bytes(target / name),
                "copy_encode_seconds": timings["per_array_copy_encode_seconds"][name],
                "decode_seconds": per_column_decode[name],
                "copy_encode_mib_s": value.nbytes / 1048576 / timings["per_array_copy_encode_seconds"][name],
                "decode_mib_s": value.nbytes / 1048576 / per_column_decode[name],
                "worker_highwater_rss_kib_after_copy": timings["per_array_worker_highwater_rss_kib"][name],
            }
            for name, value in arrays.items()
        },
    }


def _offline_candidates(arrays):
    """Prove reversible codecs directly; never make them publishable Zarr codecs."""
    from numcodecs import Delta, PackBits, Zstd

    result = {}
    for name, value in arrays.items():
        trials = []
        if value.dtype.kind not in "biu":
            continue
        compressor = Zstd(level=3)
        raw_zstd = compressor.encode(value.tobytes())
        candidates = [("delta", Delta(dtype=value.dtype))] if value.dtype.kind in "iu" else []
        if value.dtype.kind == "b":
            candidates.append(("packbits", PackBits()))
        for label, codec in candidates:
            try:
                packed = codec.encode(value)
                restored = codec.decode(packed)
                restored = np.asarray(restored).reshape(value.shape).astype(value.dtype, copy=False)
                reversible = bool(np.array_equal(restored.view(np.uint8), value.view(np.uint8)))
                candidate_zstd = compressor.encode(memoryview(packed).tobytes())
                trials.append(
                    {
                        "candidate": label,
                        "encoded_bytes": memoryview(packed).nbytes,
                        "candidate_plus_zstd_bytes": memoryview(candidate_zstd).nbytes,
                        "raw_plus_zstd_bytes": memoryview(raw_zstd).nbytes,
                        "raw_bytes": value.nbytes,
                        "reversible": reversible,
                        "publishable": False,
                    }
                )
            except Exception as exc:
                trials.append({"candidate": label, "refusal": str(exc), "publishable": False})
        if trials:
            result[name] = trials
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", type=Path, help="Optional real MSv2 to sample read-only")
    parser.add_argument("--flag-versions", type=Path, help="Optional MS .flagversions sibling to sample read-only")
    parser.add_argument("--rows-per-region", type=int, default=128)
    parser.add_argument("--synthetic-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, help="Scratch parent; defaults to the output directory")
    args = parser.parse_args()
    if args.rows_per_region < 1 or args.synthetic_rows < 1:
        parser.error("row counts must be positive")
    if args.flag_versions is not None and args.ms is None:
        parser.error("--flag-versions requires --ms")
    before = _root_observation(args.ms) if args.ms else None
    flags_before = _root_observation(args.flag_versions) if args.flag_versions else None
    arrays, selection = _real(args.ms, args.rows_per_region, args.flag_versions) if args.ms else _synthetic(args.seed, args.synthetic_rows)
    import msutils
    import zarr
    import numcodecs

    workdir = args.workdir or args.output.parent
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shinobi-codec-benchmark-", dir=workdir) as temporary:
        root = Path(temporary)
        baseline = root / "baseline.zarr"
        group = zarr.open_group(baseline, mode="w", zarr_format=3)
        for name, array in arrays.items():
            chunks = (min(32, array.shape[0]), *array.shape[1:]) if array.ndim else ()
            group.create_array(name, data=array, chunks=chunks, dimension_names=tuple(f"axis_{i}" for i in range(array.ndim)))
        original_id = msutils.logical_id(baseline)
        profiles = [
            LosslessProfile(codec="zstd", level=3, chunk_rows=32),
            LosslessProfile(codec="zstd", level=6, chunk_rows=128),
            LosslessProfile(codec="gzip", level=3, chunk_rows=32),
            LosslessProfile(codec="blosc-zstd", level=3, shuffle="byte", chunk_rows=32),
            LosslessProfile(codec="blosc-zstd", level=3, shuffle="bit", chunk_rows=128),
        ]
        results = []
        for i, profile in enumerate(profiles):
            # A fresh process gives each profile its own peak RSS measurement.
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
                results.append(executor.submit(_codec_result, baseline, root, arrays, profile, f"candidate-{i}").result())
    after = _root_observation(args.ms) if args.ms else None
    flags_after = _root_observation(args.flag_versions) if args.flag_versions else None
    if before != after or flags_before != flags_after:
        raise RuntimeError("source MS or flagversions metadata changed during benchmark")
    report = {
        "schema_version": "shinobi-codec-benchmark/v1",
        "selection": selection,
        "source_observation_before": before,
        "source_observation_after": after,
        "flag_versions_observation_before": flags_before,
        "flag_versions_observation_after": flags_after,
        "sample_hashes": {name: _sample_hash(value) for name, value in arrays.items()},
        "baseline_msv4_logical_id": original_id,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__, "numcodecs": numcodecs.__version__},
        "results": results,
        "offline_candidates": _offline_candidates(arrays),
        "scope": "sampled MSv2 columns encoded as independent Zarr v3 arrays; no native preservation sidecar or full export measured",
        "rss_scope": "Each codec runs in a fresh worker; per-column RSS is that worker’s cumulative high-water when copying the column ends.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
