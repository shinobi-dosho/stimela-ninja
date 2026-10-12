"""Full MSv2 -> exact state -> fresh MSv2 qualification for lossless profiles.

Run once per source with --source and a new --workdir on the same filesystem.
Each stage is a separate Python process. Only the generated source copy is
archived to prove that replay needs no source at its recorded path. Failures
remain in report.json, and every error exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _write(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(pending, path)


def _report(workdir: Path) -> dict:
    return json.loads((workdir / "report.json").read_text())


def _record(workdir: Path, stage: str, details: dict, started: float) -> None:
    report = _report(workdir)
    report["stages"].append(
        {
            "stage": stage,
            "seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            **details,
        }
    )
    _write(workdir / "report.json", report)


def _versions() -> dict:
    import importlib.metadata as metadata

    packages = ("stimela-ninja", "msutils", "xarray-ms", "xarray", "zarr", "numcodecs", "arcae", "numpy", "python-casacore")
    result = {name: metadata.version(name) for name in packages} | {"python": platform.python_version()}
    result["msutils_origin"] = json.loads(metadata.distribution("msutils").read_text("direct_url.json") or "{}")
    return result


def _native_id(path: Path) -> str:
    import msutils

    return msutils.native_logical_id(path)


def _stage(workdir: Path, stage: str, block_rows: int) -> None:
    from shinobi.dataset_state import DatasetStateStore
    from shinobi.state_codecs import LosslessProfile

    started = time.perf_counter()
    report = _report(workdir)
    source = Path(report["source"])
    copied = workdir / "source-copy.ms"
    archive = workdir / "og-data" / "source-copy.ms"
    store = DatasetStateStore(workdir / "store", cache_dir=workdir / "cache")
    details: dict = {"versions": _versions(), "source_observation_before": _observation(source)}
    if report["stages"] and details["source_observation_before"] != report["stages"][0]["source_observation_before"]:
        raise ValueError("original source changed between qualification stages")
    if stage == "prepare":
        if copied.exists() or archive.exists():
            raise FileExistsError("qualification copy already exists")
        original_id = _native_id(source)
        shutil.copytree(source, copied, symlinks=False, copy_function=shutil.copy2)
        copied_id = _native_id(copied)
        if original_id != copied_id:
            raise ValueError("deep working copy changed the native logical ID")
        details.update(original_native_id=original_id, copied_native_id=copied_id)
    elif stage in ("export-baseline", "export-combined"):
        codec = "zstd" if stage == "export-baseline" else "packbits-delta-zstd"
        profile = LosslessProfile(codec=codec, level=3, chunk_rows=32)
        result = store.export(copied, block_rows=block_rows, lossless_profile=profile)
        rep, _logical, physical = store._load(result.state_id, result.representation_id)
        details.update(
            codec=codec,
            state_id=result.state_id,
            representation_id=result.representation_id,
            msv4_id=physical.msv4_id,
            payload_id=physical.payload_id,
            profile=physical.lossless_profile.model_dump(mode="json"),
            export_warnings=list(physical.export_warnings),
            representation_bytes=sum(item.stat().st_size for item in rep.rglob("*") if item.is_file()),
            actual_array_metadata={
                str(path.relative_to(rep)): json.loads(path.read_text()) for path in rep.rglob("zarr.json") if json.loads(path.read_text()).get("node_type") == "array"
            },
            attempt=str(result.attempt),
        )
        if result.state_id != report["stages"][0]["original_native_id"]:
            raise ValueError("export state ID differs from original native ID")
    elif stage == "archive":
        archive.parent.mkdir(exist_ok=False)
        copied.rename(archive)
        if copied.exists() or not archive.is_dir():
            raise ValueError("recorded source path was not removed")
        details.update(recorded_source_absent=True, archived_copy=str(archive))
    elif stage == "verify":
        if copied.exists():
            raise ValueError("recorded source path is present during fresh verification")
        for export in ("export-baseline", "export-combined"):
            prior = next(item for item in report["stages"] if item["stage"] == export)
            result = store.verify(prior["state_id"], prior["representation_id"])
            if not result.verified:
                raise ValueError(f"{export} did not verify")
            details[export] = {"verified": result.verified, "attempt": str(result.attempt)}
    elif stage == "materialize":
        if copied.exists():
            raise ValueError("recorded source path is present during fresh replay")
        for export in ("export-baseline", "export-combined"):
            prior = next(item for item in report["stages"] if item["stage"] == export)
            destination = workdir / f"{export}.ms"
            result = store.materialize(prior["state_id"], destination, representation_id=prior["representation_id"])
            actual_id = _native_id(destination)
            if actual_id != report["stages"][0]["original_native_id"]:
                raise ValueError(f"{export} replay changed native ID")
            details[export] = {"destination": str(destination), "native_id": actual_id, "attempt": str(result.attempt)}
    elif stage == "compare":
        if copied.exists():
            raise ValueError("recorded source path is present during independent comparison")
        for export in ("export-baseline", "export-combined"):
            destination = workdir / f"{export}.ms"
            prior = next(item for item in report["stages"] if item["stage"] == export)
            rep, _, _ = store._load(prior["state_id"], prior["representation_id"])
            manifest = json.loads((rep / "preservation/manifest.json").read_text())
            details[export] = _compare_values(source, destination, block_rows, manifest["native_model"])
    else:
        raise ValueError(f"unknown stage {stage}")
    details["source_observation_after"] = _observation(source)
    if details["source_observation_before"] != details["source_observation_after"]:
        raise ValueError("original source changed during qualification")
    _record(workdir, stage, details, started)


def _observation(root: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    for member in sorted(root.rglob("*")):
        info = member.lstat()
        digest.update(f"{member.relative_to(root)}\0{info.st_mode}\0{info.st_size}\0{info.st_mtime_ns}\0".encode())
        count += 1
    return {"entries": count, "stat_sha256": digest.hexdigest()}


def _normal_metadata(value, root: Path):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape, "bytes": value.tobytes().hex()} if value.dtype.kind in "biufc" else value.tolist()
    if isinstance(value, dict):
        return {key: _normal_metadata(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal_metadata(item, root) for item in value]
    if isinstance(value, str) and value.startswith("Table: "):
        return "Table: " + str(Path(value[7:]).resolve().relative_to(root))
    return value


def _metadata(tab, root: Path, model: str) -> dict:
    descriptor = tab.getdesc()
    descriptor.pop("_keywords_", None)
    descriptor.pop("_private_keywords_", None)
    managers = {}
    for key, manager in tab.getdminfo().items():
        record = dict(manager)
        spec = dict(record["SPEC"])
        if record["TYPE"] == "StandardStMan":
            spec.pop("IndexLength", None)
        if model.endswith("/v2"):
            record.pop("SEQNR", None)
            if record["TYPE"].startswith("Tiled"):
                spec.pop("HYPERCUBES", None)
                spec.pop("SEQNR", None)
            if record["TYPE"] == "TiledShapeStMan":
                spec.pop("IndexSize", None)
            key = record["NAME"]
        record["SPEC"] = spec
        managers[key] = record
    return _normal_metadata({"descriptor": descriptor, "keywords": tab.getkeywords(), "info": tab.info(), "managers": managers}, root)


def _defined_counts(tab) -> dict[str, int]:
    from casacore.tables import taql

    if not tab.nrows():
        return dict.fromkeys(tab.colnames(), 0)
    expressions = ", ".join(f"GSUM(INT(ISDEFINED(\\{column}))) AS D{i}" for i, column in enumerate(tab.colnames()))
    with taql(f"SELECT {expressions} FROM $1", tables=[tab]) as counts:
        return {column: int(counts.getcell(f"D{i}", 0)) for i, column in enumerate(tab.colnames())}


def _table_paths(root: Path) -> list[Path]:
    return sorted(path.parent.relative_to(root) for path in root.rglob("table.dat"))


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "biufc":
        return left.tobytes(order="C") == right.tobytes(order="C")
    return bool(np.array_equal(left, right))


def _comparison_batch(tab, column: str, requested_rows: int) -> tuple[int, bool]:
    """Bound numeric reads from metadata before allocating any cell values."""
    from shinobi.state_codecs import MAX_TILE_BYTES

    itemsize = {
        "boolean": 1,
        "bool": 1,
        # python-casacore widens native uchar values to uint16 on reads.
        "uchar": 2,
        "char": 1,
        "short": 2,
        "ushort": 2,
        "int": 4,
        "uint": 4,
        "int64": 8,
        "float": 4,
        "double": 8,
        "complex": 8,
        "dcomplex": 16,
    }.get(tab.getcoldesc(column)["valueType"].lower())
    if itemsize is not None:
        shape = [] if tab.isscalarcol(column) else json.loads(tab.getcolshapestring(column, 0, 1)[0])
        if not isinstance(shape, list) or any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError(f"invalid numeric cell shape metadata for {column}")
        size = math.prod(shape) * itemsize
        if size > MAX_TILE_BYTES:
            raise ValueError(f"qualification numeric cell for {column} is {size} bytes; exceeds qualification-only {MAX_TILE_BYTES}-byte limit")
        return min(requested_rows, MAX_TILE_BYTES // max(1, size)), bool(shape and 0 in shape)
    # Strings retain their existing variable-length allocation behavior.
    cell = tab.getcell(column, 0)
    shape = tuple(cell["shape"]) if isinstance(cell, dict) else np.asarray(cell).shape
    empty = bool(shape and 0 in shape)
    size = 0 if empty else np.asarray(cell).nbytes
    return min(requested_rows, max(1, MAX_TILE_BYTES // max(1, size))), empty


def _compare_values(source: Path, restored: Path, block_rows: int, model: str) -> dict:
    from casacore.tables import table

    originals = _table_paths(source)
    if originals != _table_paths(restored):
        raise ValueError("native table paths differ")
    tables = []
    total_blocks = 0
    for relative in originals:
        with (
            table(str(source / relative), readonly=True, ack=False, lockoptions="usernoread") as before,
            table(str(restored / relative), readonly=True, ack=False, lockoptions="usernoread") as after,
        ):
            if before.nrows() != after.nrows() or before.colnames() != after.colnames():
                raise ValueError(f"table rows or column names differ: {relative}")
            if _metadata(before, source, model) != _metadata(after, restored, model):
                raise ValueError(f"native metadata differs: {relative}")
            defined_before, defined_after = _defined_counts(before), _defined_counts(after)
            if defined_before != defined_after:
                raise ValueError(f"definedness counts differ: {relative}")
            count = before.nrows()
            compared = []
            undefined = []
            for column in before.colnames():
                if defined_before[column] not in (0, count):
                    raise ValueError(f"mixed-definedness outside supported profile: {relative}/{column}")
                if count and not defined_before[column]:
                    undefined.append(column)
                    continue
                rows = block_rows
                empty = False
                if count:
                    rows, empty = _comparison_batch(before, column, rows)
                for start in range(0, count, rows):
                    width = min(rows, count - start)
                    if empty:
                        if list(before.getcolshapestring(column, start, width)) != list(after.getcolshapestring(column, start, width)):
                            raise ValueError(f"empty cell shapes differ: {relative}/{column}")
                        total_blocks += 1
                        continue
                    try:
                        left = np.asarray(before.getcol(column, startrow=start, nrow=width))
                        right = np.asarray(after.getcol(column, startrow=start, nrow=width))
                        if not _same_values(left, right):
                            raise ValueError(f"value mismatch: {relative}/{column} rows {start}:{start + width}")
                    except RuntimeError as exc:
                        raise ValueError(f"bounded native read failed: {relative}/{column} rows {start}:{start + width}") from exc
                    total_blocks += 1
                compared.append(column)
            tables.append({"path": str(relative), "rows": count, "compared_columns": compared, "undefined_columns": undefined})
    return {"tables": tables, "bounded_blocks": total_blocks, "comparison": "all defined values; numeric exact bytes, other values exact equality"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--stage", choices=("prepare", "export-baseline", "export-combined", "archive", "verify", "materialize", "compare"))
    args = parser.parse_args()
    if args.block_rows < 1:
        parser.error("--block-rows must be positive")
    workdir = args.workdir.expanduser().resolve()
    if args.stage:
        import traceback

        try:
            _stage(workdir, args.stage, args.block_rows)
        except Exception:
            report = _report(workdir)
            report.update(status="failed", failed_stage=args.stage, traceback=traceback.format_exc())
            _write(workdir / "report.json", report)
            raise
        return
    if args.source is None:
        parser.error("--source is required for the full qualification")
    source = args.source.expanduser().resolve(strict=True)
    if source == workdir or source.is_relative_to(workdir) or workdir.is_relative_to(source):
        parser.error("source and workdir must be disjoint")
    workdir.mkdir(parents=True, exist_ok=False)
    _write(
        workdir / "report.json",
        {
            "schema_version": "shinobi-state-codec-qualification/v1",
            "source": str(source),
            "workdir": str(workdir),
            "stages": [],
            "status": "running",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    stages = ("prepare", "export-baseline", "export-combined", "archive", "verify", "materialize", "compare")
    try:
        for stage in stages:
            print(f"[{stage}] {source}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--workdir", str(workdir), "--block-rows", str(args.block_rows), "--stage", stage], check=True)
    except subprocess.CalledProcessError as exc:
        report = _report(workdir)
        report.update(status="failed", failed_stage=stage, exit_code=exc.returncode)
        _write(workdir / "report.json", report)
        raise
    report = _report(workdir)
    report["status"] = "passed"
    _write(workdir / "report.json", report)


if __name__ == "__main__":
    main()
