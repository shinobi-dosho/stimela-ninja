"""Fresh-process v7 storage/monitoring fixture; invoked by the scaling test."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from uuid import uuid4

from pydantic import BaseModel

from shinobi.config import AppConfig
from shinobi.offload.bundle import write_new
from shinobi.offload.mixed import Invocation, MixedExecution, _declared_result, freeze_mixed, plan_mixed
from shinobi.offload.record_store import Edge, Store
from shinobi.results import StepResult
from shinobi.steps.schema import Cab, Recipe, SlurmPlacement
from shinobi.steps.dispatch import _aggregate_scatter_results


class Empty(BaseModel):
    pass


class Case(BaseModel):
    case: int


class Statistics(BaseModel):
    rms: float = 0.25
    channels: tuple[int, int] = (128, 256)


class Outputs(BaseModel):
    stats: Statistics
    products: list[str] = []
    metrics: dict[str, float] = {}


def own(name, children=None, log=""):
    return StepResult(
        name=name,
        kind="recipe" if children else "cab",
        returncode=0,
        inputs=Empty(),
        outputs=Outputs(stats=Statistics(), products=[f"product-{i:03d}.fits" for i in range(20)], metrics={f"metric-{i}": i / 10 for i in range(20)}),
        sub_results=children,
        stdout=log if not children else "\n".join(child.stdout for child in children.values() if child.stdout),
    )


def build(directory, log_size):
    directory.mkdir()
    leaf = Cab(name="leaf", command="/bin/true", inputs_model=Empty, outputs_model=Outputs)
    cycle = Recipe(name="cycle", inputs_model=Empty, outputs_model=Outputs)
    for i in range(14):
        cycle.add_step(f"leaf-{i}", leaf)
    branch = Recipe(name="branch", inputs_model=Case, outputs_model=Outputs)
    for i in range(3):
        branch.add_step(f"cycle-{i}", cycle)
    for i in range(16):
        branch.add_step(f"last-{i}", leaf)
    root = Recipe(name="campaign", inputs_model=Empty, outputs_model=Outputs)
    root.add_step("branches", branch, placement=SlurmPlacement(), scatter=["case"], case=list(range(30)))
    bundle = freeze_mixed(root, {}, config=AppConfig(backend={"default": "native"}), workspace=directory)
    directory = bundle.stage(directory)
    digest = bundle.digest
    from shinobi.offload.bundle import Submission

    submission = Submission.model_validate_json((directory / "submission.json").read_bytes())
    execution = MixedExecution(workflow_id=submission.workflow_id, bundle_digest=digest, plan=plan_mixed(bundle), worker={}, images={}, registry=str(directory / "registry"))
    write_new(directory / "mixed.json", execution)
    store = Store(directory)
    units = []
    summaries = []

    def invocation(name):
        value = Invocation(workflow_id=execution.workflow_id, bundle_digest=digest, name=name, attempt_id=uuid4())
        write_new(directory / "invocations" / name / f"{value.attempt_id}.json", value)
        return value

    def capture(value, current, edges=None, transparent=False):
        return store.capture(
            value, edges=edges, transparent=transparent, workflow_id=execution.workflow_id, attempt_id=current.attempt_id, step_path=value.name, bundle_digest=digest
        )

    def publish(record):
        write_new(directory / "results" / str(record.attempt_id) / "final.json", record)

    for branch_index in range(30):
        leaves = 0

        def leaf_result(name):
            nonlocal leaves
            label = f"branch={branch_index:03d} leaf={leaves:03d} "
            leaves += 1
            return own(name, log=label + "x" * (log_size - len(label)))

        children = {f"cycle-{cycle_index}": own(f"cycle-{cycle_index}", {f"leaf-{i}": leaf_result(f"leaf-{i}") for i in range(14)}) for cycle_index in range(3)}
        children.update({f"last-{i}": leaf_result(f"last-{i}") for i in range(16)})
        value = own(f"campaign.branches[{branch_index}]", children)
        value.inputs = Case(case=branch_index)
        unit = invocation(f"unit-0-{branch_index}")
        record = capture(value, unit)
        publish(record)
        summaries.append(_declared_result(store.result(record, branch), branch))
        units.append(unit)
        del children, value, record
    gather = invocation("gather-0")
    gathered = _aggregate_scatter_results(branch, ["case"], {"case": list(range(30))}, summaries)
    gathered.name = "campaign.branches"
    edges = tuple(Edge(name=f"[{i}]", reference=store.root_reference(unit.attempt_id)) for i, unit in enumerate(units))
    record = capture(gathered, gather, edges).model_copy(update={"source_attempts": {unit.name: unit.attempt_id for unit in units}})
    publish(record)
    final = invocation("finalize")
    edges = (Edge(name="branches", reference=store.root_reference(gather.attempt_id)),)
    record = capture(own("campaign"), final, edges).model_copy(update={"source_attempts": {gather.name: gather.attempt_id}})
    manifest = store.manifest(directory / "results" / str(final.attempt_id) / "manifest.json", record)
    publish(record.model_copy(update={"manifest": manifest}))
    paths = list((directory / "results").rglob("*.json")) + list((directory / "schemas").glob("*.json"))
    metrics = {
        "submission": str(directory),
        "terminal_bytes": (directory / "results" / str(final.attempt_id) / "final.json").stat().st_size,
        "node_schema_bytes": sum(path.stat().st_size for path in paths if path.name != "manifest.json"),
        "log_bytes": sum(path.stat().st_size for path in (directory / "artifacts/logs").iterdir()),
        "json_files": len(paths),
        "log_files": len(list((directory / "artifacts/logs").iterdir())),
        "bundle_bytes": (directory / "bundle.json").stat().st_size,
    }
    sys.stdout.write(json.dumps(metrics))


def monitor(directory):
    import shinobi.offload.mixed as mixed
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.tracking import FINISHED

    def forbidden(*args, **kwargs):
        raise AssertionError("monitor attempted eager result reconstruction")

    mixed._read_result = forbidden
    Store.result = forbidden
    AttemptRecord.result = forbidden
    mixed._mixed_scheduler_states = lambda *args: {"fixture": "COMPLETED"}
    visited = []
    original = Store.read_one

    def read(self, reference):
        visited.append((reference.owner, reference.address))
        return original(self, reference)

    Store.read_one = read
    started = time.monotonic()
    status = mixed.mixed_status({"submission": str(directory), "jobs": {}})
    assert status.state == FINISHED and status.exit_code == 0, status
    assert len(visited) == len(set(visited)), "shared evidence read repeatedly"
    count = len(visited)
    visited.clear()
    assert mixed.current_manifest(directory).exists()
    assert len(visited) == len(set(visited))
    sys.stdout.write(json.dumps({"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "seconds": time.monotonic() - started, "unique_read_nodes": count}))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        monitor(Path(sys.argv[2]))
