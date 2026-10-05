"""Optional real-data joint prediction and physical Slurm recovery gate.

The caller supplies isolated, bounded seed MSs and a nonzero WSClean FITS
model. Run M2 first on the exact mount. Nothing here selects or writes the
original observation. See README for prerequisites and the fresh check.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from shinobi import MSv2, Cab, DatasetAccess, DatasetColumns, Recipe
from shinobi.clonefs import clone_tree
from shinobi.dataset_backends import load_shared_storage_qualification
from shinobi.dataset_closure import ClosureStatus, resolve_dataset_closure
from shinobi.offload.records import AttemptRecord
from shinobi.offload.worker import DatasetSettlement, ExecutionPlan, Finalization, SubmittedFinalizer, SubmittedJob
from shinobi.ownership import inspect_workspace
from shinobi.snapshots import chain_id, get_journal, state_name
from shinobi.steps.schema import InputRef, OutputRef, ParamMeta, StepRef
from tests.slurm_physical.run_m4_msv2 import _submit, _wait


class MSList(BaseModel):
    ms: list[MSv2]


class Prediction(BaseModel):
    script: str
    executable: str
    options: str
    logfile: str
    model: Path
    ms: list[MSv2]


class Predicted(MSList):
    log: Path


class ScanWriter(BaseModel):
    script: str
    values: str
    ms: list[MSv2]


class Reader(BaseModel):
    script: str
    report: str
    ms: list[MSv2]


class Report(BaseModel):
    artifact: Path


REPORT_COLUMNS = ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER")


# The binary wrapper declares the actual FITS files as read-only inputs, while
# WSClean's -name takes their prefix. It runs one exec-form joint invocation.
PREDICT = """
import json,subprocess,sys
from pathlib import Path
executable,options,log,model=sys.argv[1:5]
roots=sys.argv[5:]
assert len(roots)==2 and Path(model).is_file() and model.endswith('-model.fits')
prefix=model[:-len('-model.fits')]
argv=[executable,'-predict','-name',prefix,*json.loads(options),*roots]
result=subprocess.run(argv)
Path(log).write_text(json.dumps({'argv':argv,'returncode':result.returncode},indent=2))
sys.exit(result.returncode)
"""
WRITE_SCANS = """
import json,sys
import casacore.tables as tables
import numpy as np
values=json.loads(sys.argv[1])
assert len(values)==len(sys.argv[2:])==2
for path,value in zip(sys.argv[2:],values):
    with tables.table(path,readonly=False,ack=False) as ms:
        ms.putcol('SCAN_NUMBER',np.full(ms.nrows(),value,dtype=np.int32))
"""
COPY_SEED = """
import sys
import casacore.tables as tables
import numpy as np
with tables.table(sys.argv[1],readonly=True,ack=False) as source:
    copy=source.copy(sys.argv[2],deep=True,valuecopy=True)
    copy.close()
with tables.table(sys.argv[2],readonly=False,ack=False) as ms:
    if 'MODEL_DATA' in ms.colnames(): ms.removecols('MODEL_DATA')
    if len(sys.argv)>3: ms.putcol('SCAN_NUMBER',np.full(ms.nrows(),int(sys.argv[3]),dtype=np.int32))
"""


def _column_report(paths):
    """Digest every cell, bounding memory to one row even for variable shapes."""
    import hashlib
    import socket

    import casacore.tables as tables
    import numpy as np

    result = []
    for path in paths:
        with tables.table(str(path), readonly=True, ack=False) as ms:
            columns = {}
            scans = set()
            for name in REPORT_COLUMNS:
                if name not in ms.colnames():
                    columns[name] = None
                    continue
                digest = hashlib.sha256()
                elements = nonzero = finite = undefined = complex_elements = 0
                for row in range(ms.nrows()):
                    if not ms.iscelldefined(name, row):
                        digest.update(b"undefined\n")
                        undefined += 1
                        continue
                    array = np.asarray(ms.getcell(name, row))
                    if name == "SCAN_NUMBER":
                        scans.update(int(value) for value in array.ravel())
                    digest.update(str((array.dtype.str, array.shape)).encode())
                    digest.update(array.tobytes(order="C"))
                    elements += array.size
                    complex_elements += array.size if np.iscomplexobj(array) else 0
                    finite += int(np.isfinite(array).sum())
                    nonzero += int(np.count_nonzero(array))
                columns[name] = {
                    "digest": digest.hexdigest(),
                    "elements": elements,
                    "finite": finite,
                    "nonzero": nonzero,
                    "undefined": undefined,
                    "complex_elements": complex_elements,
                }
            result.append({"root": str(path), "rows": ms.nrows(), "scans": sorted(scans), "columns": columns})
    return {"host": socket.gethostname(), "datasets": result}


READ_COLUMNS = (
    "import json,sys\nfrom pathlib import Path\n"
    f"REPORT_COLUMNS={REPORT_COLUMNS!r}\n" + inspect.getsource(_column_report) + "\nPath(sys.argv[1]).write_text(json.dumps(_column_report(sys.argv[2:]),indent=2))\n"
)


def _predict_cab(python: Path) -> Cab:
    return Cab(
        name="joint-wsclean-predict",
        command=f"{python} -c",
        inputs_model=Prediction,
        outputs_model=Predicted,
        field_meta={
            "script": ParamMeta(positional_head=True),
            **{name: ParamMeta(positional=True) for name in ("executable", "options", "logfile")},
            "model": ParamMeta(positional=True, writable=False),
            "ms": ParamMeta(positional=True, repeat_as_tokens=True),
            "log": ParamMeta(implicit="{logfile}"),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("MODEL_DATA",), create=("MODEL_DATA",)), allow_schema_change=True)],
        cache=True,
    )


def _writer_cab(python: Path) -> Cab:
    return Cab(
        name="joint-scan-writer",
        command=f"{python} -c",
        inputs_model=ScanWriter,
        outputs_model=MSList,
        field_meta={"script": ParamMeta(positional_head=True), "values": ParamMeta(positional=True), "ms": ParamMeta(positional=True, repeat_as_tokens=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))],
        cache=True,
    )


def _reader_cab(python: Path) -> Cab:
    return Cab(
        name="joint-columns-checker",
        command=f"{python} -c",
        inputs_model=Reader,
        outputs_model=Report,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "report": ParamMeta(positional=True),
            "ms": ParamMeta(positional=True, repeat_as_tokens=True),
            "artifact": ParamMeta(implicit="{report}"),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=REPORT_COLUMNS))],
        cache=True,
    )


def _recipe(args, directory, prediction=None, scans=None):
    if prediction is not None:
        steps = [StepRef(name="predict", step=_predict_cab(args.casacore_python), params=prediction, wiring={"ms": InputRef(field="ms")})]
        steps.append(
            StepRef(
                name="read",
                step=_reader_cab(args.casacore_python),
                params={"script": READ_COLUMNS, "report": str(directory / "reader.json")},
                wiring={"ms": OutputRef(step="predict", field="ms")},
            )
        )
        output = "predict"
    else:
        steps = [StepRef(name="write", step=_writer_cab(args.casacore_python), params={"script": WRITE_SCANS, "values": json.dumps(scans)}, wiring={"ms": InputRef(field="ms")})]
        output = "write"
    return Recipe(
        name="physical-joint-predict" if prediction is not None else "physical-joint-recovery",
        inputs_model=MSList,
        outputs_model=MSList,
        steps=steps,
        output_wiring={"ms": OutputRef(step=output, field="ms")},
        cache_dir=str(directory / "cache"),
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def _read_columns(args, paths, report):
    subprocess.run([str(args.casacore_python), "-c", READ_COLUMNS, str(report), *map(str, paths)], check=True)
    return json.loads(report.read_text())


def _contained(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"test resource {path} is outside {root}")
    return path


def _copies(args, label, scans=None):
    directory = args.root / "workspace" / label
    directory.mkdir(parents=True, exist_ok=False)
    paths = [directory / f"part-{index}.ms" for index in range(2)]
    for index, (seed, target) in enumerate(zip(args.seed_ms, paths)):
        command = [str(args.casacore_python), "-c", COPY_SEED, str(seed), str(target)]
        if scans is not None:
            command.append(str(scans[index]))
        subprocess.run(command, check=True)
        closure = resolve_dataset_closure(target, storage_namespace=args.root / "workspace")
        if closure.status is not ClosureStatus.VALID or any(not resource.path.is_relative_to(target) for resource in closure.resources):
            raise ValueError(f"copied MS has an uncontained closure: {closure}")
    return directory, paths


def _prediction(args):
    directory = args.root / "workspace" / f"models-{args.mode}"
    directory.mkdir(exist_ok=False)
    source = _contained(args.root, args.model_file or Path(str(args.model_prefix) + "-model.fits"))
    if not source.name.endswith("-model.fits"):
        raise ValueError("the declared model must end in -model.fits for WSClean's naming convention")
    destination = directory / source.name
    clone_tree(source, destination)
    return {
        "script": PREDICT,
        "executable": str(args.wsclean),
        "options": args.wsclean_options_json,
        "logfile": str(args.root / "workspace" / args.mode / "predict-argv.json"),
        "model": destination,
    }


def _accounting(jobs):
    deadline = time.monotonic() + 60
    while True:
        completed = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(jobs), "--format=JobIDRaw,State,ExitCode,NodeList"], check=True, capture_output=True, text=True)
        records = {}
        for line in completed.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 4 and fields[0] in jobs:
                records[fields[0]] = {"state": fields[1], "exit": fields[2], "node": fields[3]}
        if set(records) == set(jobs) and all(record["state"] in ("COMPLETED", "FAILED", "CANCELLED") for record in records.values()):
            return records
        if time.monotonic() > deadline:
            raise TimeoutError(f"sacct records are not terminal: {records}")
        time.sleep(0.5)


def _attempt_record(submission, plan, step):
    attempt = plan.attempt(step)
    return AttemptRecord.read(
        submission / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=step,
        bundle_digest=plan.bundle_digest,
    )


def _reader_artifact(args, record):
    artifact = _declared_output(args, record.result(_reader_cab(args.casacore_python)).outputs.artifact)
    return artifact, json.loads(artifact.read_text())


def _declared_output(args, path):
    """Resolve serialized output paths against the frozen fixture workspace."""
    path = Path(path)
    return _contained(args.root, path if path.is_absolute() else args.root / "workspace" / path)


def _validate_reader(report, paths, independent):
    """Check actual worker evidence against ordered roots and a fresh read."""
    assert [dataset["root"] for dataset in report["datasets"]] == list(map(str, paths))
    _check_science(report, "prediction")
    _same_columns(independent, report, ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))


def _check_access_contract(plan, step, paths, *, model=None, artifact=None):
    """Assert declared permissions in the immutable compute-side plan."""
    accesses = plan.dataset_lifecycle.step(step).accesses
    assert [access.element_index for access in accesses] == [0, 1]
    assert [str(access.root) for access in accesses] == list(map(str, paths))
    expected = {
        "predict": DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("MODEL_DATA",), create=("MODEL_DATA",)), allow_schema_change=True),
        "write": DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",))),
        "read": DatasetAccess(field="ms", mode="read", columns=DatasetColumns(read=REPORT_COLUMNS)),
    }[step]
    assert all(access.declaration == expected and access.mode == expected.mode and access.columns_known for access in accesses)
    files = {access.path: access.writes for access in plan.accesses}
    assert all(files[str(path)] is True for path in paths)
    if model is not None:
        assert files[str(model)] is False
    if artifact is not None:
        assert files[str(artifact)] is True


def _check_workspace_contract(args, plan, directory):
    assert Path(plan.dataset_lifecycle.storage_namespace) == args.root / "workspace"
    assert Path(plan.dataset_lifecycle.cache_dir) == directory / "cache"
    assert Path(plan.ownership_workspace) == directory
    assert Path(plan.ownership_registry) == args.root / "workspace-owners.json"
    assert all(Path(access.path).is_relative_to(args.root) for access in plan.accesses)
    assert inspect_workspace(directory, str(plan.workflow_id)) is None


def _phase(args, label, directory, paths, recipe, nodes, expected, fault=None, cached=False):
    handle = _submit(args, label, recipe, {"ms": paths}, nodes, fault=fault)
    finalized = _wait(handle.submission_dir, timeout=args.timeout)
    assert finalized.complete is (fault != "S2")
    report = _read_columns(args, paths, args.root / "evidence" / f"{label}-columns.json")
    journal = get_journal(str(directory / "cache"))
    heads = [journal.get(chain_id(path)).head for path in paths]
    plan = ExecutionPlan.model_validate_json((handle.submission_dir / "execution.json").read_text())
    _check_workspace_contract(args, plan, directory)
    writer = _attempt_record(handle.submission_dir, plan, "predict" if "predict" in handle.jobs else "write")
    if fault == "S2":
        expected_heads = json.loads((args.root / "evidence" / "baseline.json").read_text())["heads"]
    else:
        expected_heads = [state_name(key.cache_key, "ms", index) for index, key in enumerate(writer.output_keys["ms"])]
    assert heads == expected_heads
    assert all(journal.get(chain_id(path)).marker is None for path in paths)
    all_jobs = [*handle.jobs.values(), handle.finalizer_job]
    evidence = {
        "submission": str(handle.submission_dir),
        "paths": list(map(str, paths)),
        "cache": str(directory / "cache"),
        "heads": heads,
        "expected_heads": expected_heads,
        "expected": expected,
        "fault": fault,
        "cached": cached,
        "nodes": nodes,
        "jobs": handle.jobs,
        "finalizer_job": handle.finalizer_job,
        "accounting": _accounting(all_jobs),
        "report": report,
    }
    if "read" in handle.jobs:
        artifact, reader_report = _reader_artifact(args, _attempt_record(handle.submission_dir, plan, "read"))
        assert artifact == directory / "reader.json"
        _validate_reader(reader_report, paths, report)
        evidence.update(reader_artifact=str(artifact), reader_report=reader_report)
    _write_json(args.root / "evidence" / f"{label}.json", evidence)
    _check_science(report, expected)
    return evidence


def _check_science(report, expected):
    assert len(report["datasets"]) == 2
    for index, dataset in enumerate(report["datasets"]):
        assert dataset["rows"] > 0
        if expected == "prediction":
            model = dataset["columns"]["MODEL_DATA"]
            assert model is not None and model["elements"] > 0 and model["undefined"] == 0
            assert model["finite"] == model["elements"] == model["complex_elements"] and model["nonzero"] > 0
        else:
            assert dataset["scans"] == [expected[index]], dataset


def _same_columns(before, after, names):
    assert len(before["datasets"]) == len(after["datasets"])
    for first, second in zip(before["datasets"], after["datasets"]):
        assert first["rows"] == second["rows"]
        assert all(first["columns"][name] == second["columns"][name] for name in names)


def run(args):
    args.root = args.root.resolve()
    args.seed_ms = [_contained(args.root, path) for path in args.seed_ms]
    if args.model_prefix is not None:
        args.model_prefix = _contained(args.root, args.model_prefix)
    if len(set(args.seed_ms)) != 2:
        raise ValueError("supply two distinct isolated seed MSs")
    if len(set(args.nodes)) != 3:
        raise ValueError("supply three distinct physical nodes")
    options = json.loads(args.wsclean_options_json)
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError("--wsclean-options-json must be an argv array of strings")
    (args.root / "workspace").mkdir(exist_ok=True)
    (args.root / "evidence").mkdir(exist_ok=True)
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(args.root / "workspace-owners.json")
    args.sbatch_opts = {"cpus-per-task": args.cpus_per_task, "mem": args.mem}
    os.chdir(args.root / "workspace")
    specification = {
        "mode": args.mode,
        "root": str(args.root),
        "seed_ms": list(map(str, args.seed_ms)),
        "nodes": args.nodes,
        "casacore_python": str(args.casacore_python),
        "qualification": str(args.storage_qualification) if args.storage_qualification else None,
    }
    _write_json(args.root / f"m5-{args.mode}.json", specification)
    _read_columns(args, args.seed_ms, args.root / "evidence" / f"{args.mode}-seeds.json")
    directory, paths = _copies(args, args.mode)
    before = _read_columns(args, paths, args.root / "evidence" / f"{args.mode}-baseline.json")
    assert all(dataset["columns"]["MODEL_DATA"] is None for dataset in before["datasets"])
    prediction = _prediction(args)
    if args.mode == "native":
        cab = _predict_cab(args.casacore_python)
        for cached, label in ((False, "native-predict"), (True, "native-cached")):
            result = cab(ms=paths, **prediction, cache=True, cache_dir=str(directory / "cache"))
            assert result.success and result.cached is cached
            keys = result.provenance_key("ms")
            assert [key.element_index for key in keys] == [0, 1]
            report = _read_columns(args, paths, args.root / "evidence" / f"{label}-columns.json")
            _check_science(report, "prediction")
            _same_columns(before, report, ("DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
            _write_json(args.root / "evidence" / f"{label}.json", {"report": report, "cache_key": str(keys[0]), "cached": cached, "paths": list(map(str, paths))})
    else:
        if args.storage_qualification is None:
            raise ValueError("Slurm mode requires fresh --storage-qualification from M2")
        qualification, _ = load_shared_storage_qualification(args.storage_qualification)
        assert args.root.is_relative_to(qualification.storage_root)
        first = _phase(args, "slurm-predict", directory, paths, _recipe(args, directory, prediction=prediction), {"predict": args.nodes[0], "read": args.nodes[1]}, "prediction")
        _same_columns(before, first["report"], ("DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
        repeated = _phase(
            args, "slurm-cached", directory, paths, _recipe(args, directory, prediction=prediction), {"predict": args.nodes[2], "read": args.nodes[1]}, "prediction", cached=True
        )
        _same_columns(first["report"], repeated["report"], ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
        recovery, targets = _copies(args, "recovery", scans=[101, 202])
        for label, scans, expected, node, fault, cached in (
            ("baseline", [101, 202], [101, 202], args.nodes[0], None, False),
            ("fault-s2", [301, 402], [101, 202], args.nodes[1], "S2", False),
            ("retry", [301, 402], [301, 402], args.nodes[2], None, False),
            ("fault-s3", [501, 602], [501, 602], args.nodes[2], "S3", False),
            ("recovery-cached", [501, 602], [501, 602], args.nodes[1], None, True),
        ):
            _phase(args, label, recovery, targets, _recipe(args, recovery, scans=scans), {"write": node}, expected, fault=fault, cached=cached)
    sys.stdout.write(json.dumps({"complete": True, "mode": args.mode, "root": str(args.root)}) + "\n")
    return 0


def _check_phase(args, label, expected, fault=None, cached=False):
    evidence = json.loads((args.root / "evidence" / f"{label}.json").read_text())
    assert evidence["expected"] == expected and evidence["fault"] == fault and evidence["cached"] is cached
    _check_science(evidence["report"], expected)
    submission = _contained(args.root, Path(evidence["submission"]))
    plan = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
    finalized = Finalization.model_validate_json((submission / "finalization.json").read_text())
    settlement = DatasetSettlement.model_validate_json((submission / "dataset-settlement.json").read_text())
    assert (finalized.workflow_id, finalized.bundle_digest) == (plan.workflow_id, plan.bundle_digest)
    assert settlement.workflow_id == plan.workflow_id and settlement.bundle_digest == plan.bundle_digest
    assert settlement.attempts == tuple(attempt.attempt_id for attempt in plan.attempts)
    assert finalized.complete is (fault != "S2")
    contract = plan.dataset_lifecycle
    assert contract is not None and contract.cache_dir == evidence["cache"]
    _, pinned = load_shared_storage_qualification(args.storage_qualification)
    assert contract.qualification_path == str(args.storage_qualification) and contract.qualification_digest == pinned
    directory = args.root / "workspace" / ("slurm" if label.startswith("slurm-") else "recovery")
    _check_workspace_contract(args, plan, directory)
    accounting = _accounting([*evidence["jobs"].values(), evidence["finalizer_job"]])
    assert accounting == evidence["accounting"]
    finalizer = SubmittedFinalizer.model_validate_json((submission / "finalizer-job.json").read_text())
    assert (finalizer.workflow_id, finalizer.bundle_digest, finalizer.job_id) == (plan.workflow_id, plan.bundle_digest, evidence["finalizer_job"])
    settled_job = accounting[finalizer.job_id]
    assert settled_job["node"] in args.nodes
    assert settled_job["state"] == ("FAILED" if fault == "S2" else "COMPLETED")
    assert settled_job["exit"] == ("1:0" if fault == "S2" else "0:0")
    for step in finalized.steps:
        planned = plan.attempt(step.step_path)
        assert planned.attempt_id == step.attempt_id
        ordinal = next(index for index, item in enumerate(plan.attempts) if item.step_path == step.step_path)
        job = SubmittedJob.model_validate_json((submission / "jobs" / f"{ordinal:04d}.json").read_text())
        assert (job.workflow_id, job.bundle_digest, job.attempt_id) == (plan.workflow_id, plan.bundle_digest, step.attempt_id)
        assert job.job_id == step.job_id == evidence["jobs"][step.step_path]
        observed = accounting[job.job_id]
        assert observed["node"] == evidence["nodes"][step.step_path]
        assert observed["state"] == ("FAILED" if fault else "COMPLETED")
        assert observed["exit"] == ("86:0" if fault else "0:0")
        record = _attempt_record(submission, plan, step.step_path)
        assert record.job_id == job.job_id
        assert step.record == str(Path("attempts") / str(step.attempt_id) / "final.json")
        assert record.state == step.state == ("failed" if fault == "S2" else "cached" if cached else "succeeded")
        if step.step_path == "read":
            artifact, reader_report = _reader_artifact(args, record)
            _check_access_contract(plan, "read", evidence["paths"], artifact=artifact)
            assert str(artifact) == evidence["reader_artifact"]
            _validate_reader(reader_report, evidence["paths"], evidence["report"])
            _validate_reader(evidence["reader_report"], evidence["paths"], evidence["report"])
            _same_columns(evidence["reader_report"], reader_report, ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
        else:
            if step.step_path == "predict":
                predicted = record.result(_predict_cab(args.casacore_python))
                _check_access_contract(plan, "predict", evidence["paths"], model=predicted.inputs.model, artifact=_declared_output(args, predicted.outputs.log))
            else:
                _check_access_contract(plan, "write", evidence["paths"])
            assert record.schema_version == 3
            if record.committed:
                keys = record.output_keys["ms"]
                assert [key.element_index for key in keys] == [0, 1]
                assert all(key.producer_field == "ms" for key in keys)
                assert evidence["heads"] == [state_name(key.cache_key, "ms", index) for index, key in enumerate(keys)]
            leaves = record.dataset_lifecycle.leaves
            leaf = next(leaf for leaf in leaves if leaf.step_path.endswith("." + step.step_path))
            if not cached:
                assert [mutation.element_index for mutation in leaf.mutations] == [0, 1]
                assert [str(mutation.root) for mutation in leaf.mutations] == evidence["paths"]
                assert all(mutation.successor_state == state_name(leaf.cache.cache_key, "ms", index) for index, mutation in enumerate(leaf.mutations))
    assert evidence["heads"] == evidence["expected_heads"]
    if fault == "S2":
        baseline = json.loads((args.root / "evidence" / "baseline.json").read_text())
        assert evidence["heads"] == baseline["heads"]
    journal = get_journal(evidence["cache"])
    for index, (path, head) in enumerate(zip(evidence["paths"], evidence["heads"])):
        chain = journal.get(chain_id(Path(path)))
        if label in ("slurm-cached", "recovery-cached"):
            assert chain.head == head
        snapshot = journal.snapshot_dir(head)
        report = _read_columns(args, [snapshot], args.root / "evidence" / f"check-{label}-{index}.json")
        assert report["datasets"][0]["columns"] == evidence["report"]["datasets"][index]["columns"]
        assert journal.get(chain_id(Path(path))).marker is None
    return evidence


def check(args):
    args.root = args.root.resolve()
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(args.root / "workspace-owners.json")
    spec = json.loads((args.root / f"m5-{args.mode}.json").read_text())
    seeds = [Path(path) for path in spec["seed_ms"]]
    original = json.loads((args.root / "evidence" / f"{args.mode}-seeds.json").read_text())
    fresh = _read_columns(args, seeds, args.root / "evidence" / f"check-{args.mode}-seeds.json")
    _same_columns(original, fresh, ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
    if args.mode == "slurm":
        assert args.storage_qualification is not None
        for label, expected, fault, cached in (
            ("slurm-predict", "prediction", None, False),
            ("slurm-cached", "prediction", None, True),
            ("baseline", [101, 202], None, False),
            ("fault-s2", [101, 202], "S2", False),
            ("retry", [301, 402], None, False),
            ("fault-s3", [501, 602], "S3", False),
            ("recovery-cached", [501, 602], None, True),
        ):
            _check_phase(args, label, expected, fault=fault, cached=cached)
        scientific = args.root / "workspace" / "slurm"
        fresh_prediction = _read_columns(args, [scientific / f"part-{index}.ms" for index in range(2)], args.root / "evidence" / "check-final-prediction.json")
        _check_science(fresh_prediction, "prediction")
        predicted = json.loads((args.root / "evidence" / "slurm-predict.json").read_text())["report"]
        baseline = json.loads((args.root / "evidence" / "slurm-baseline.json").read_text())
        assert all(dataset["columns"]["MODEL_DATA"] is None for dataset in baseline["datasets"])
        _same_columns(predicted, fresh_prediction, ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
        _same_columns(baseline, fresh_prediction, ("DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
        final_paths = [args.root / "workspace" / "recovery" / f"part-{index}.ms" for index in range(2)]
        _check_science(_read_columns(args, final_paths, args.root / "evidence" / "check-final-recovery.json"), [501, 602])
    else:
        before = json.loads((args.root / "evidence" / "native-baseline.json").read_text())
        for cached, label in ((False, "native-predict"), (True, "native-cached")):
            evidence = json.loads((args.root / "evidence" / f"{label}.json").read_text())
            assert evidence["cached"] is cached
            _check_science(evidence["report"], "prediction")
            report = _read_columns(args, [Path(path) for path in evidence["paths"]], args.root / "evidence" / f"check-{label}.json")
            _same_columns(evidence["report"], report, ("DATA", "MODEL_DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
            _same_columns(before, report, ("DATA", "FLAG", "WEIGHT", "UVW", "TIME", "SCAN_NUMBER"))
            journal = get_journal(str(args.root / "workspace" / "native" / "cache"))
            for index, path in enumerate(evidence["paths"]):
                chain = journal.get(chain_id(Path(path)))
                assert chain.marker is None and chain.head == state_name(evidence["cache_key"], "ms", index)
                snapshot = journal.snapshot_dir(chain.head)
                copied = _read_columns(args, [snapshot], args.root / "evidence" / f"check-{label}-snapshot-{index}.json")
                assert copied["datasets"][0]["columns"] == report["datasets"][index]["columns"]
    sys.stdout.write(json.dumps({"complete": True, "mode": args.mode, "fresh_check": True}) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "slurm"), default="slurm")
    parser.add_argument("--seed-ms", type=Path, nargs=2)
    parser.add_argument("--model-prefix", type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--wsclean", type=Path)
    parser.add_argument("--wsclean-options-json", default='["-j","1","-no-reorder","-pol","I"]')
    parser.add_argument("--worker-python", type=Path, default=Path("/data/venv-issue18/bin/python"))
    parser.add_argument("--casacore-python", type=Path, default=Path("/data/venv-issue18/bin/python"))
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--mem", default="2G")
    parser.add_argument("--nodes", nargs=3, default=("k1", "n1", "n2"))
    parser.add_argument("--storage-qualification", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.storage_qualification is not None:
        args.storage_qualification = args.storage_qualification.resolve()
    if args.cpus_per_task < 1 or args.timeout <= 0:
        parser.error("--cpus-per-task and --timeout must be positive")
    if args.command == "run":
        if any(getattr(args, name) is None for name in ("seed_ms", "wsclean")):
            parser.error("run requires --seed-ms and --wsclean")
        if args.model_file is None and args.model_prefix is None:
            parser.error("run requires either --model-file or --model-prefix")
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
