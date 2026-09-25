"""Submit/check the physical cross-node shared-metadata acceptance probe.

Run inside the Kudu controller container; see this directory's README.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from shinobi.cache import CacheManifest
from shinobi.dataset_backends import SharedStorageQualification
from shinobi.offload.slurm import parse_sbatch_job_id, status_slurm
from shinobi.results import StepResult
from shinobi.snapshots import Chain, ChainJournal
from shinobi.storage import ofd_lock

DEFAULT_NODES = ("k1", "n1", "n2")
DEFAULT_HOSTS = {"k1": "devk1", "n1": "devn1", "n2": "devn2"}


class Empty(BaseModel):
    pass


class Value(BaseModel):
    value: int


def _verify_cross_node_lock(root: Path, node: str, participants: int, holder: str) -> None:
    """Prove every other node observes ``holder``'s OFD lock."""
    probe = root / "lock-probe" / holder
    probe.mkdir(parents=True, exist_ok=True)
    lock_path = probe / "held.lock"
    held = probe / "held"
    observed = probe / "observed"
    done = probe / "done"
    observed.mkdir(exist_ok=True)
    deadline = time.monotonic() + 60
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if node == holder:
            ofd_lock(fd, fcntl.F_WRLCK)
            held.write_text("locked")
            while len(list(observed.iterdir())) < participants - 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError("other nodes did not test the held lock")
                time.sleep(0.1)
        else:
            while not held.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("lock holder did not publish the probe barrier")
                time.sleep(0.1)
            try:
                ofd_lock(fd, fcntl.F_WRLCK, blocking=False)
            except BlockingIOError:
                (observed / node).write_text("excluded")
            else:
                raise AssertionError(f"{node} acquired a lock held on {holder}: locking is node-local")
    finally:
        os.close(fd)
    if node == holder:
        done.write_text("released")
    else:
        while not done.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{holder} did not finish its lock round")
            time.sleep(0.1)


def _barrier(root: Path, phase: str, node: str, participants: int) -> None:
    ready = root / "barriers" / phase
    ready.mkdir(parents=True, exist_ok=True)
    (ready / node).write_text("ready")
    deadline = time.monotonic() + 60
    while len(list(ready.iterdir())) < participants:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"not every node reached barrier {phase!r}")
        time.sleep(0.02)


def worker(args) -> int:
    """Contend on both mutable stores from one allocated cluster node."""
    ready = args.root / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    (ready / args.node).write_text(f"{host}:{os.getpid()}")
    deadline = time.monotonic() + 60
    while len(list(ready.iterdir())) < args.participants:
        if time.monotonic() >= deadline:
            raise TimeoutError("not every node reached the shared-storage barrier")
        time.sleep(0.1)

    for holder in args.nodes:
        _verify_cross_node_lock(args.root, args.node, args.participants, holder)

    manifest = CacheManifest(args.root / "manifest.json")
    journal = ChainJournal(args.root / "snapshots")
    for index in range(args.iterations):
        # Bound every worker to the same iteration. This forces genuine
        # cross-node contention instead of allowing one fast allocation to
        # finish before its peers leave the lock-probe polling loop.
        _barrier(args.root, f"update-{index}", args.node, args.participants)
        name = f"{args.node}.{index}"
        # A fresh store object in each Slurm process proves correctness does
        # not depend on the process-local singleton registries.
        manifest.record(
            name,
            f"key-{name}",
            StepResult(name=name, returncode=0, inputs=Empty(), outputs=Value(value=index)),
            run_id=f"{host}:{os.getpid()}",
        )
        journal.update_chain(
            name,
            lambda _old, name=name, index=index: Chain(
                dev=1,
                ino=index,
                ctime_ns=index,
                path=f"/data/{name}",
            ),
        )
    return 0


def submit(args) -> int:
    if len(set(args.nodes)) != len(args.nodes):
        raise ValueError("--nodes must not contain duplicates")
    expected_hosts = _expected_hosts(args.expected_host)
    if not expected_hosts:
        expected_hosts = DEFAULT_HOSTS if tuple(args.nodes) == DEFAULT_NODES else {node: node for node in args.nodes}
    if set(expected_hosts) != set(args.nodes):
        raise ValueError("--expected-host mappings must name every and only --nodes entry")
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root / "logs").mkdir()
    script = args.source_root / "tests" / "slurm_physical" / "run_m2.py"
    jobs = {}
    for node in args.nodes:
        command = shlex.join(
            [
                str(args.worker_python),
                str(script),
                "worker",
                "--root",
                str(args.root),
                "--node",
                node,
                "--participants",
                str(len(args.nodes)),
                "--iterations",
                str(args.iterations),
                "--nodes",
                *args.nodes,
            ]
        )
        process = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--partition={args.partition}",
                f"--nodelist={node}",
                f"--job-name=m2-store-{node}",
                f"--output={args.root}/logs/{node}.out",
                f"--error={args.root}/logs/{node}.err",
                f"--export=ALL,PYTHONPATH={args.source_root / 'src'}:{args.source_root}",
                "--wrap",
                command,
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"sbatch failed for {node}: {process.stderr.strip()}")
        jobs[node] = parse_sbatch_job_id(process.stdout)
    handle = {
        "jobs": jobs,
        "nodes": args.nodes,
        "expected_hosts": expected_hosts,
        "iterations": args.iterations,
        "root": str(args.root),
    }
    (args.root / "handle.json").write_text(json.dumps(handle, indent=2))
    sys.stdout.write(json.dumps(handle) + "\n")
    return 0


def check(args) -> int:
    handle = json.loads((args.root / "handle.json").read_text())
    states = status_slurm(handle["jobs"])
    assert set(states.values()) == {"COMPLETED"}, states
    nodes = tuple(handle["nodes"])
    expected = {f"{node}.{index}" for node in nodes for index in range(handle["iterations"])}
    manifest = CacheManifest(args.root / "manifest.json")
    assert {name for name in expected if manifest.entry(name) is not None} == expected
    assert set(ChainJournal(args.root / "snapshots").all_chains()) == expected
    assert set(json.loads((args.root / "manifest.json").read_text())) == expected
    assert set(json.loads((args.root / "snapshots" / "chains.json").read_text())) == expected
    assert not list(args.root.glob(".manifest.json.*.tmp"))
    assert not list((args.root / "snapshots").glob(".chains.json.*.tmp"))
    ready = {path.name: path.read_text() for path in (args.root / "ready").iterdir()}
    expected_hosts = handle["expected_hosts"]
    actual_hosts = {node: value.split(":", 1)[0] for node, value in ready.items()}
    assert actual_hosts == expected_hosts
    assert len(set(actual_hosts.values())) == len(nodes)
    excluded = {holder: {path.name for path in (args.root / "lock-probe" / holder / "observed").iterdir()} for holder in nodes}
    assert all(contenders == set(nodes) - {holder} for holder, contenders in excluded.items())
    result = {
        "complete": True,
        "states": states,
        "records": len(expected),
        "workers": ready,
        "excluded": {holder: sorted(contenders) for holder, contenders in excluded.items()},
    }
    storage_root = (args.storage_root or args.root.parent).resolve()
    probe_root = args.root.resolve()
    if probe_root != storage_root and not probe_root.is_relative_to(storage_root):
        raise ValueError(f"physical probe {probe_root} is outside requested storage root {storage_root}")
    qualification_path = (args.qualification or args.root / "qualification.json").resolve()
    if qualification_path != storage_root and not qualification_path.is_relative_to(storage_root):
        raise ValueError(f"qualification {qualification_path} must remain under qualified storage root {storage_root}")
    qualification = SharedStorageQualification(
        qualification_id=uuid4(),
        storage_root=storage_root,
        verified_at=time.time(),
        workers=tuple(nodes),
        evidence=f"physical M2 probe {args.root}; jobs {json.dumps(handle['jobs'], sort_keys=True)}",
    )
    qualification_path.write_text(qualification.model_dump_json(indent=2) + "\n")
    result["qualification"] = str(qualification_path)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _expected_hosts(entries: list[str] | None) -> dict[str, str]:
    result = {}
    for entry in entries or ():
        node, separator, host = entry.partition("=")
        if not separator or not node or not host:
            raise ValueError(f"--expected-host must be NODE=HOST, got {entry!r}")
        result[node] = host
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "check", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/data/src/stimela-ninja"))
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--node")
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--expected-host", action="append", help="Expected scheduler-node to runtime-hostname mapping, as NODE=HOST")
    parser.add_argument("--participants", type=int, default=len(DEFAULT_NODES))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--qualification", type=Path, help="Where check writes the site qualification (default: ROOT/qualification.json).")
    parser.add_argument("--storage-root", type=Path, help="Exact shared tree qualified by the probe (default: ROOT's parent).")
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
