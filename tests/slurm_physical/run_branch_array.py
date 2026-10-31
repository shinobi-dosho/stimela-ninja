"""Live branch-array matrix: concurrency, CREATE overwrite, clean Slurm requeue.

Run inside the controller with shared storage and a Python/Casacore image.
All scientific operations happen inside scheduled Apptainer tasks.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from branch_fixture import build
from shinobi.config import AppConfig
from shinobi.offload.branch import BranchCase, prepare_branch_array, submit_branch_array


def wait_for(predicate, description, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(2)
    raise TimeoutError(description)


def completed(directory, count):
    records = [json.loads(path.read_text()) for path in directory.glob("attempts/*/*/finished.json")]
    failures = [record for record in records if record["status"] != "completed"]
    if failures:
        raise AssertionError(failures)
    return records if len(records) == count else None


def manifest(workspace):
    return json.loads((workspace / ".shinobi/cache/manifest.json").read_text())


def inventory(workspace):
    root = workspace / "data/noise.ms"
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file() and path.name != "table.lock"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--partition", default="dev")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = AppConfig.model_validate({"backend": {"default": "apptainer"}, "cache": {"enabled": True}, "sandbox": {"enabled": True}, "execution": {"max_workers": 1}})
    results = {}

    def case(name, **parameters):
        workspace = root / "cases" / name
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        return BranchCase(name=name, workspace=workspace, parameters={"image": str(args.image.resolve()), **parameters})

    def launch(name, cases, parallel=1):
        directory = prepare_branch_array(
            build,
            cases,
            directory=root / name,
            code_roots=[Path(__file__).parent],
            config=config,
            # Reserve each eight-CPU worker to force distinct placements.
            sbatch_opts={"partition": args.partition, "nodes": "1", "ntasks": "1", "cpus-per-task": "8", "mem": "4G", "time": "00:15:00"},
            max_parallel=parallel,
        )
        job = submit_branch_array(directory)
        results[name] = {"job": job, "directory": str(directory)}
        (root / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        return directory, job

    cases = [case(name, first_delay=40) for name in ("a", "b", "c")]
    directory, _ = launch("concurrent", cases, parallel=3)
    results["concurrent"]["attempts"] = wait_for(lambda: completed(directory, 3), "concurrent branches")
    reports = [json.loads((item.workspace / "first.json").read_text()) for item in cases]
    assert len({report["host"] for report in reports}) == 3, reports
    assert max(report["start"] for report in reports) < min(report["end"] for report in reports), reports
    assert all(report["scans"] == [7] * 4 for report in reports)
    results["concurrent"]["overlap_reports"] = reports
    before = manifest(cases[0].workspace)
    old_bytes = inventory(cases[0].workspace)
    overwrite = case("a", scan=9, first_delay=40).model_copy(update={"overwrite_steps": ("create",)})
    directory, _ = launch("overwrite", [overwrite])
    results["overwrite"]["attempts"] = wait_for(lambda: completed(directory, 1), "CREATE overwrite")
    after = manifest(overwrite.workspace)
    leaves = ("branch.create", "branch.read[0]", "branch.read[1]")
    assert all(before[key] != after[key] for key in leaves)
    assert old_bytes != inventory(overwrite.workspace)
    assert all(json.loads((overwrite.workspace / path).read_text())["scans"] == [9] * 4 for path in ("first.json", "second.json"))
    results["overwrite"]["changed_leaves"] = list(leaves)
    interrupted = case("requeue", second_delay=120)
    directory, job = launch("requeue", [interrupted])

    def second_started():
        text = "".join(path.read_text() for path in (directory / "logs").glob("*"))
        return "BRANCH_READER_STARTED second.json" in text and (interrupted.workspace / "first.json").is_file()

    wait_for(second_started, "second slice starts")
    before = manifest(interrupted.workspace)
    old_bytes = inventory(interrupted.workspace)
    subprocess.run(["scontrol", "requeue", f"{job}_0"], check=True)
    subprocess.run(["scontrol", "update", f"JobId={job}_0", "StartTime=now"], check=True)
    results["requeue"]["attempts"] = wait_for(lambda: completed(directory, 1), "requeued task completes")
    started = [json.loads(path.read_text()) for path in directory.glob("attempts/0/*/started.json")]
    assert len(started) == 2 and sorted(record["restart_count"] for record in started) == [0, 1], started
    after = manifest(interrupted.workspace)
    assert all(before[key] == after[key] for key in ("branch.create", "branch.read[0]"))
    assert inventory(interrupted.workspace) == old_bytes
    assert json.loads((interrupted.workspace / "second.json").read_text())["scans"] == [7] * 4
    results["requeue"].update(started=started, cached_prefix_preserved=True, ms_bytes_preserved=True)
    results["passed"] = True
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
