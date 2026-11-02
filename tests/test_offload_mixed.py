"""End-to-end contracts for one ordinary graph locally and on detached drivers."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from shinobi.config import AppConfig
from shinobi.offload._codec import BundleError
from shinobi.offload.bundle import RecipeBundle
from shinobi.offload.mixed import freeze_mixed, plan_mixed, prepare_mixed

SOURCE = """from pathlib import Path
from pydantic import BaseModel
from shinobi import Recipe, pystep, SlurmPlacement
class Empty(BaseModel):
    pass
class One(BaseModel):
    path: Path
class Many(BaseModel):
    path: list[Path]
@pystep()
def make(path: Path) -> One:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(path.read_text())+1) if path.exists() else "1")
    return One(path=path)
@pystep()
def gather(path: Path, sources: list[Path]) -> One:
    path.write_text("\\n".join(str(source) for source in sources))
    return One(path=path)
branch = Recipe(name="branch", inputs_model=One, outputs_model=One, cache=True)
branch.add_step("make", make, path=branch.inputs.path)
branch.set_output("path", branch.outputs.make.path)
recipe = Recipe(name="mixed", inputs_model=Empty, outputs_model=Many, cache=True, max_workers=2)
recipe.add_step("branches", branch, placement=SlurmPlacement(max_parallel=2), scatter=["path"], path=[Path("a/out.txt"), Path("b/out.txt")])
recipe.set_output("path", recipe.outputs.branches.path)
"""


@pytest.fixture
def graph(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "mixed_fixture.py").write_text(SOURCE)
    monkeypatch.syspath_prepend(str(source))
    sys.modules.pop("mixed_fixture", None)
    recipe = importlib.import_module("mixed_fixture").recipe
    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    yield recipe, source, workspace
    sys.modules.pop("mixed_fixture", None)


def config():
    return AppConfig(backend={"default": "native"}, cache={"enabled": True})


def test_pure_roundtrip_and_local(graph):
    recipe, source, workspace = graph
    bundle = freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    assert list(workspace.iterdir()) == []
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json())
    assert restored.digest == bundle.digest
    assert [unit.name for unit in plan_mixed(restored).units] == ["mixed.branches[0]", "mixed.branches[1]"]
    result = recipe(_config=config())
    assert result.outputs.path == [Path("a/out.txt"), Path("b/out.txt")]
    assert (workspace / "a/out.txt").read_text() == "1"
    again = recipe(_config=config())
    assert again.outputs.path == result.outputs.path
    assert (workspace / "a/out.txt").read_text() == "1"


def invoke(directory, operation, *args):
    worker = json.loads((directory / "mixed.json").read_text())["worker"]
    return subprocess.run(
        [worker["python"], "-m", "shinobi.offload.mixed", str(directory), operation, *args], env={**os.environ, "PYTHONPATH": worker["source"]}, capture_output=True, text=True
    )


def test_detached_drivers_keep_order_and_cache(graph):
    recipe, source, workspace = graph
    bundle = freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    directory = prepare_mixed(bundle, submission_root=workspace / "submissions")
    for task in (1, 0):
        process = invoke(directory, "run", "--step", "0", "--task", str(task))
        assert process.returncode == 0, process.stderr
    for operation in ("gather", "finalize"):
        process = invoke(directory, operation, "--step", "0")
        assert process.returncode == 0, process.stderr
    from shinobi.offload.mixed import current_manifest

    assert current_manifest(directory).exists()
    from shinobi.offload.mixed import _load, _read_result

    frozen, execution = _load(directory)
    result = _read_result(directory, frozen, execution, "finalize")
    assert result.outputs.path == [Path("a/out.txt"), Path("b/out.txt")]
    recipe(_config=config())
    assert (workspace / "a/out.txt").read_text() == "1"


def test_overlap_rejected_before_staging(graph):
    recipe, source, workspace = graph
    recipe.steps[0].params["path"] = [Path("same/out.txt"), Path("same/out.txt")]
    with pytest.raises(BundleError, match="overlap"):
        freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    assert list(workspace.iterdir()) == []


def test_local_placed_strict_create_then_read_scatter(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    from pydantic import BaseModel
    from shinobi import DatasetAccess, MSv2, Recipe, SlurmPlacement, pystep
    from tests._dataset_fixtures import make_ms, scans

    class Empty(BaseModel):
        pass

    class One(BaseModel):
        ms: MSv2

    class Value(BaseModel):
        value: int

    class Values(BaseModel):
        value: list[int]

    class Cases(BaseModel):
        value: list[list[int]]

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
    def create(ms: MSv2) -> One:
        make_ms(ms)
        return One(ms=ms)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MSv2, index: int) -> Value:
        return Value(value=scans(ms)[0] + index)

    branch = Recipe(name="branch", inputs_model=One, outputs_model=Values, max_workers=2)
    branch.add_step("create", create, ms=branch.inputs.ms)
    branch.add_step("read", read, ms=branch.outputs.create.ms, scatter=["index"], index=[0, 1])
    branch.set_output("value", branch.outputs.read.value)
    root = Recipe(name="outer", inputs_model=Empty, outputs_model=Cases, max_workers=2)
    root.add_step("cases", branch, placement=SlurmPlacement(max_parallel=2), scatter=["ms"], ms=[tmp_path / "a.ms", tmp_path / "b.ms"])
    root.set_output("value", root.outputs.cases.value)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    result = root(_config=config())
    assert result.outputs.value == [[1, 2], [1, 2]]


def test_zip_and_empty_contract(graph):
    from shinobi.steps.schema import ScatterSpec

    recipe, source, workspace = graph
    recipe.steps[0].params["path"] = []
    bundle = freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    assert not plan_mixed(bundle).units
    assert recipe(_config=config()).outputs.path == []
    recipe.steps[0].scatter = ScatterSpec(fields=["path", "missing"])
    with pytest.raises(ValueError):
        freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))


def test_partial_submission_is_discoverable_and_capped(graph, monkeypatch):
    from shinobi.offload.mixed import submit_mixed
    from shinobi.offload.tracking import discover

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 2 else 0, str(100 + len(calls)), "refused" if len(calls) == 2 else "")

    monkeypatch.setattr("shinobi.offload.mixed.subprocess.run", sbatch)
    with pytest.raises(BundleError, match="recoverable"):
        submit_mixed(directory)
    assert "--array=0-1%2" in (directory / "step-0.sbatch").read_text()
    assert "--kill-on-invalid-dep=yes" in (directory / "gather-0.sbatch").read_text()
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101" in calls[2]
    assert (directory / "jobs/step-0.json").exists()
    assert (directory / "handle.json").exists()
    assert len(discover(workspace)) == 1
    with pytest.raises(FileExistsError):
        submit_mixed(directory)


def test_invalid_later_placement_never_submits(graph, monkeypatch):
    from shinobi import SlurmPlacement

    recipe, source, workspace = graph
    recipe.steps[0].placement = SlurmPlacement(sbatch_opts={"dependency": "afterok:123"})
    with pytest.raises(BundleError, match="override"):
        freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    assert list(workspace.iterdir()) == []


def test_finalizer_missing_records_fails_with_manifest(graph):
    from shinobi.offload.mixed import mixed_status

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    process = invoke(directory, "finalize")
    assert process.returncode != 0
    from shinobi.offload.mixed import current_manifest

    assert current_manifest(directory).exists()
    state = mixed_status({"submission": str(directory), "jobs": {}})
    assert state.failed


def test_source_drift_refused_before_work(graph):
    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    captured = next((directory / "code").glob("*/mixed_fixture.py"))
    captured.write_text('raise RuntimeError("tampered")')
    process = invoke(directory, "run", "--step", "0", "--task", "0")
    assert process.returncode != 0 and "staged source differs" in process.stderr
    assert not (workspace / "a/out.txt").exists()


def test_requeue_current_result_and_per_attempt_logs(graph, monkeypatch):
    from shinobi.offload.mixed import _load, _read_result, follow_mixed

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    first = invoke(directory, "run", "--step", "0", "--task", "0")
    assert first.returncode == 0, first.stderr
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    second = invoke(directory, "run", "--step", "0", "--task", "0")
    assert second.returncode == 0, second.stderr
    assert len(list((directory / "logs/unit-0-0").glob("*/run.log"))) == 2
    bundle, execution = _load(directory)
    result = _read_result(directory, bundle, execution, "unit-0-0")
    assert result.sub_results["make"].cached
    assert list(follow_mixed({"submission": str(directory)}, wait=False))


def test_composite_roundtrip_keeps_synthetic_key(graph):
    from uuid import uuid4
    from shinobi.offload.records import AttemptRecord

    recipe, source, workspace = graph
    result = recipe(_config=config())
    key = result.provenance_key("path")
    assert key.producer_field is None
    record = AttemptRecord.from_result(result, workflow_id=uuid4(), attempt_id=uuid4(), step_path="mixed", bundle_digest="test")
    assert record.schema_version == 6
    restored = AttemptRecord.model_validate_json(record.model_dump_json()).result(recipe)
    assert restored.outputs.path == result.outputs.path
    assert restored.provenance_key("path").producer_field is None
    assert list(restored.sub_results["branches"].sub_results) == ["[0]", "[1]"]


@pytest.mark.parametrize("first_partitioned", [False, True])
def test_partitioned_writer_never_weakens_coarse_owner(tmp_path, monkeypatch, first_partitioned):
    from contextlib import nullcontext
    from shinobi.ownership import acquire_workspace, partitioned_claim, WorkspaceOwnershipError

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    first = [(tmp_path / "a", True)]
    second = [(tmp_path / "b", True)]
    with partitioned_claim(tuple(first)) if first_partitioned else nullcontext():
        lease = acquire_workspace(tmp_path, "first", kind="local", accesses=first)
    try:
        with partitioned_claim(tuple(second)) if not first_partitioned else nullcontext():
            with pytest.raises(WorkspaceOwnershipError, match="owned"):
                acquire_workspace(tmp_path, "second", kind="local", accesses=second)
    finally:
        lease.release()


def test_disjoint_partition_release_preserves_sibling(tmp_path, monkeypatch):
    from shinobi.ownership import acquire_workspace, partitioned_claim, WorkspaceOwnershipStore, WorkspaceOwnershipError

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    with partitioned_claim(((tmp_path / "a", True),)):
        first = acquire_workspace(tmp_path, "first", kind="local", accesses=[(tmp_path / "a", True)])
    with partitioned_claim(((tmp_path / "b", True),)):
        second = acquire_workspace(tmp_path, "second", kind="local", accesses=[(tmp_path / "b", True)])
    try:
        assert first.owner.schema_version == 2
        with partitioned_claim(((tmp_path / "a/child", False),)):
            with pytest.raises(WorkspaceOwnershipError):
                acquire_workspace(tmp_path, "collision", kind="local", accesses=[(tmp_path / "a/child", False)])
        first.release()
        assert [owner.workflow_id for owner in WorkspaceOwnershipStore(tmp_path).owners()] == ["second"]
    finally:
        if WorkspaceOwnershipStore(tmp_path).owner("first") is not None:
            first.release()
        second.release()


def test_mixed_recovery_is_exact_owned_paths(graph, monkeypatch):
    recipe, source, workspace = graph
    calls = []

    def reconcile(cache_dir, manifest, *, paths=None, exact=False):
        calls.append((paths, exact))
        return []

    monkeypatch.setattr("shinobi.steps.dispatch.reconcile", reconcile)
    recipe(_config=config())
    assert calls
    assert all(exact and paths is not None for paths, exact in calls)
    assert all(workspace / "unrelated.ms" not in paths for paths, _ in calls)


STRICT_SOURCE = """from pathlib import Path
from pydantic import BaseModel
from shinobi import Recipe, pystep, SlurmPlacement, DatasetAccess, MSv2
class Empty(BaseModel):
    pass
class One(BaseModel):
    ms: MSv2
class Value(BaseModel):
    value: int
class Values(BaseModel):
    value: list[int]
class Cases(BaseModel):
    value: list[list[int]]
@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
def create(ms: MSv2) -> One:
    import casacore.tables as tables
    import numpy as np
    with tables.default_ms(str(ms)) as table:
        table.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
        table.addrows(2)
        table.putcol("SCAN_NUMBER", np.ones(2,dtype=np.int32))
    return One(ms=ms)
@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
def read(ms: MSv2, index: int) -> Value:
    import casacore.tables as tables
    with tables.table(str(ms), ack=False) as table:
        return Value(value=int(table.getcell("SCAN_NUMBER",0))+index)
branch = Recipe(name="branch", inputs_model=One, outputs_model=Values, max_workers=2)
branch.add_step("create",create,ms=branch.inputs.ms)
branch.add_step("read",read,ms=branch.outputs.create.ms,scatter=["index"],index=[0,1])
branch.set_output("value",branch.outputs.read.value)
recipe = Recipe(name="strict",inputs_model=Empty,outputs_model=Cases,max_workers=2)
recipe.add_step("cases",branch,placement=SlurmPlacement(max_parallel=2),scatter=["ms"],ms=[Path("a/data.ms"),Path("b/data.ms")])
recipe.set_output("value",recipe.outputs.cases.value)
"""


def test_detached_strict_create_reader_and_local_reuse(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    source = tmp_path / "source"
    source.mkdir()
    (source / "strict_fixture.py").write_text(STRICT_SOURCE)
    monkeypatch.syspath_prepend(str(source))
    sys.modules.pop("strict_fixture", None)
    recipe = importlib.import_module("strict_fixture").recipe
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    bundle = freeze_mixed(recipe, {}, config=config(), workspace=work, code_roots=(source,))
    directory = prepare_mixed(bundle, submission_root=work / "submissions")
    for task in (0, 1):
        process = invoke(directory, "run", "--step", "0", "--task", str(task))
        assert process.returncode == 0, process.stderr
    for operation in ("gather", "finalize"):
        process = invoke(directory, operation, "--step", "0")
        assert process.returncode == 0, process.stderr
    assert recipe(_config=config()).outputs.value == [[1, 2], [1, 2]]
    sys.modules.pop("strict_fixture", None)


def test_ordinary_downstream_gather_uses_committed_data(graph):
    recipe, source, workspace = graph
    recipe.add_step("gather", importlib.import_module("mixed_fixture").gather, path=Path("gather.txt"), sources=recipe.outputs.branches.path)
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    for operation, step, task in (("run", 0, 1), ("run", 0, 0), ("gather", 0, 0), ("run", 1, 0), ("gather", 1, 0), ("finalize", 0, 0)):
        process = invoke(directory, operation, "--step", str(step), "--task", str(task))
        assert process.returncode == 0, process.stderr
    assert (workspace / "gather.txt").read_text().splitlines() == ["a/out.txt", "b/out.txt"]


def test_redirected_branch_output_cannot_expand_gather_claim(graph):
    _, source, workspace = graph
    path = source / "mixed_fixture.py"
    path.write_text(SOURCE.replace("return One(path=path)", 'return One(path=Path("foreign.txt"))', 1))
    sys.modules.pop("mixed_fixture", None)
    module = importlib.import_module("mixed_fixture")
    recipe = module.recipe
    recipe.add_step("gather", module.gather, path=Path("gather.txt"), sources=recipe.outputs.branches.path)
    (workspace / "foreign.txt").write_text("unclaimed")
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    for operation, task in (("run", 0), ("run", 1), ("gather", 0)):
        process = invoke(directory, operation, "--step", "0", "--task", str(task))
        assert process.returncode == 0, process.stderr
    process = invoke(directory, "run", "--step", "1", "--task", "0")
    assert process.returncode != 0 and "changed frozen filesystem access" in process.stderr
    assert not (workspace / "gather.txt").exists()


def _complete_branches(directory):
    for operation, task in (("run", 0), ("run", 1), ("gather", 0)):
        process = invoke(directory, operation, "--step", "0", "--task", str(task))
        assert process.returncode == 0, process.stderr


def test_finalizer_requeues_select_only_current_manifest(graph, monkeypatch):
    from shinobi.offload.mixed import current_manifest

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    failed = invoke(directory, "finalize")
    assert failed.returncode != 0
    failed_manifest = current_manifest(directory)
    _complete_branches(directory)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    succeeded = invoke(directory, "finalize")
    assert succeeded.returncode == 0, succeeded.stderr
    first = current_manifest(directory)
    assert first != failed_manifest
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    again = invoke(directory, "finalize")
    assert again.returncode == 0, again.stderr
    assert current_manifest(directory) != first
    assert first.exists() and failed_manifest.exists()


def test_finalizer_crash_after_manifest_can_requeue(graph, monkeypatch):
    from shinobi.offload.mixed import current_manifest

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    _complete_branches(directory)
    worker = json.loads((directory / "mixed.json").read_text())["worker"]
    script = """import os,sys
from pathlib import Path
import shinobi.offload.mixed as mixed
original=mixed.write_new
def publish(path, value):
    if path.name == "final.json":
        os._exit(79)
    return original(path,value)
mixed.write_new=publish
mixed.execute(Path(sys.argv[1]),"finalize")
"""
    process = subprocess.run([worker["python"], "-c", script, str(directory)], env={**os.environ, "PYTHONPATH": worker["source"]}, capture_output=True, text=True)
    assert process.returncode == 79, process.stderr
    assert list((directory / "results").glob("*/manifest.json"))
    with pytest.raises(FileNotFoundError):
        current_manifest(directory)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    again = invoke(directory, "finalize")
    assert again.returncode == 0, again.stderr
    assert current_manifest(directory).exists()


def test_unscattered_recipe_paths_flow_to_downstream_consumer(graph):
    recipe, source, workspace = graph
    recipe.steps[0].scatter = None
    recipe.steps[0].params["path"] = Path("a/out.txt")
    recipe.outputs_model = recipe.steps[0].step.outputs_model
    recipe.add_step("gather", importlib.import_module("mixed_fixture").gather, path=Path("gather.txt"), sources=[recipe.outputs.branches.path])
    assert recipe(_config=config()).outputs.path == Path("a/out.txt")
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    for operation, step in (("run", 0), ("gather", 0), ("run", 1), ("gather", 1), ("finalize", 0)):
        process = invoke(directory, operation, "--step", str(step), "--task", "0")
        assert process.returncode == 0, process.stderr
    assert (workspace / "gather.txt").read_text() == "a/out.txt"


def test_recipe_exports_preserve_scalar_and_selected_producer_identity():
    from types import SimpleNamespace
    from shinobi.cache import ProvenanceKey, recipe_output_keys
    from shinobi.steps.schema import OutputRef

    keys = {
        "scalar": "raw-key",
        "family": ProvenanceKey("family-key", "dirty"),
        "scatter": ProvenanceKey("combined", None),
        "absent": None,
    }
    recipe = SimpleNamespace(
        output_wiring={
            "renamed": OutputRef(step="producer", field="scalar"),
            "selected": OutputRef(step="producer", field="family", selection={"channel": 1}),
            "gathered": OutputRef(step="producer", field="scatter"),
            "missing": OutputRef(step="producer", field="absent"),
        }
    )
    result = recipe_output_keys(recipe, {"producer": SimpleNamespace(provenance_key=lambda field: keys[field])})
    assert result["renamed"].producer_field == "scalar"
    assert result["selected"].producer_field == "dirty"
    assert result["selected"].coordinates == {"channel": 1}
    assert str(result["selected"]) == "family-key"
    assert result["gathered"].producer_field is None
    assert "missing" not in result


@pytest.mark.parametrize(
    "environment,cpus,memory",
    [
        ({"SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_CPU": "1024"}, 4, 8 * 1024**3),
        ({"SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_NODE": "2048"}, 4, 2 * 1024**3),
        ({"SLURM_CPUS_ON_NODE": "2", "SLURM_MEM_PER_CPU": "1024"}, 2, 2 * 1024**3),
        ({"SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_CPU": "0"}, 4, 16 * 1024**3),
        ({"SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_NODE": "0"}, 4, 16 * 1024**3),
    ],
)
def test_allocation_budget_caps_node_and_per_cpu_memory(monkeypatch, environment, cpus, memory):
    from types import SimpleNamespace
    from shinobi.resources import Resources
    from shinobi.offload.mixed import allocation_resources

    monkeypatch.setattr("shinobi.resources.detect_budget", lambda: (Resources(cpus=16, memory="64GiB"), "test"))
    configured = SimpleNamespace(execution=SimpleNamespace(resources=SimpleNamespace(resolve=lambda: (Resources(cpus=4, memory="16GiB"), "test"))))
    actual = allocation_resources(configured, environment)
    assert actual.cpus == cpus and actual.memory == memory


@pytest.mark.parametrize("environment", [{"SLURM_NTASKS": "2"}, {"SLURM_JOB_NUM_NODES": "2"}, {"SLURM_MEM_PER_CPU": "1024"}])
def test_unsupported_allocation_shapes_refused(environment):
    from shinobi.offload.mixed import allocation_resources

    with pytest.raises(BundleError):
        allocation_resources(config(), environment)


@pytest.mark.parametrize("options", [{"nodes": "2"}, {"ntasks": "2"}, {"ntasks-per-node": "2"}])
def test_multinode_or_multitask_placement_refused_before_staging(graph, options):
    from shinobi import SlurmPlacement

    recipe, source, workspace = graph
    recipe.steps[0].placement = SlurmPlacement(sbatch_opts=options)
    with pytest.raises(BundleError, match="one task on one node"):
        freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,))
    assert not list(workspace.iterdir())


@pytest.mark.parametrize("edge", ["after", "data"])
def test_requeue_after_verified_read_cannot_replace_consumed_attempt(graph, edge):
    from shinobi.offload.mixed import _load, _latest
    from shinobi.offload.records import AttemptRecord

    recipe, source, workspace = graph
    module = importlib.import_module("mixed_fixture")
    if edge == "after":
        recipe.add_step("consumer", module.make, path=Path("consumer.txt"))
        recipe.steps[-1].after = ["branches"]
    else:
        recipe.add_step("consumer", module.gather, path=Path("consumer.txt"), sources=recipe.outputs.branches.path)
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    _complete_branches(directory)
    worker = json.loads((directory / "mixed.json").read_text())["worker"]
    script = """import sys
from pathlib import Path
from uuid import uuid4
import shinobi.offload.mixed as mixed
original=mixed._read_committed
injected=False
def read(directory,bundle,execution,name,**kwargs):
    global injected
    verified=original(directory,bundle,execution,name,**kwargs)
    if name == "gather-0" and not injected:
        injected=True
        newer=verified[0].model_copy(update={"attempt_id":uuid4(),"restart":1})
        mixed.write_new(directory/"invocations"/name/f"{newer.attempt_id}.json",newer)
    return verified
mixed._read_committed=read
mixed.execute(Path(sys.argv[1]),"run",step=1,task=0)
"""
    process = subprocess.run([worker["python"], "-c", script, str(directory)], env={**os.environ, "PYTHONPATH": worker["source"]}, capture_output=True, text=True)
    assert process.returncode != 0 and "superseded predecessor" in process.stderr
    assert not (workspace / "consumer.txt").exists()
    bundle, execution = _load(directory)
    invocation = _latest(directory, "unit-1-0")
    record = AttemptRecord.read(
        directory / "results" / str(invocation.attempt_id) / "final.json",
        workflow_id=execution.workflow_id,
        attempt_id=invocation.attempt_id,
        step_path="mixed.consumer",
        bundle_digest=bundle.digest,
    )
    assert record.state == "failed" and not record.source_attempts


def test_recursive_result_read_keeps_expected_predecessor_identity(graph, monkeypatch):
    from uuid import uuid4
    import shinobi.offload.mixed as mixed

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    _complete_branches(directory)
    bundle, execution = mixed._load(directory)
    original = mixed._latest
    injected = False

    def latest(root, name):
        nonlocal injected
        current = original(root, name)
        if name == "unit-0-0" and not injected:
            injected = True
            newer = current.model_copy(update={"attempt_id": uuid4(), "restart": 1})
            mixed.write_new(directory / "invocations" / name / f"{newer.attempt_id}.json", newer)
            from shinobi.offload.record_store import read_terminal

            record = read_terminal(directory / "results" / str(current.attempt_id) / "final.json")

            # A successful replacement still cannot be substituted for the
            # exact attempt consumed by the already committed parent gather.
            def reidentify(value):
                return value.model_copy(
                    update={"attempt_id": newer.attempt_id, "children": {key: reidentify(child) for key, child in value.children.items()} if value.children is not None else None}
                )

            mixed.write_new(directory / "results" / str(newer.attempt_id) / "final.json", reidentify(record))
        return current

    monkeypatch.setattr(mixed, "_latest", latest)
    with pytest.raises(BundleError, match="invocation changed"):
        mixed._read_result(directory, bundle, execution, "gather-0")


@pytest.mark.parametrize("edge", ["after", "data"])
@pytest.mark.parametrize("parent_state", ["current", "missing"])
def test_dependency_edges_require_current_committed_result(graph, edge, parent_state):
    from shinobi.offload.mixed import _load, _latest
    from shinobi.offload.records import AttemptRecord

    recipe, source, workspace = graph
    module = importlib.import_module("mixed_fixture")
    if edge == "after":
        recipe.add_step("consumer", module.make, path=Path("consumer.txt"))
        recipe.steps[-1].after = ["branches"]
    else:
        recipe.add_step("consumer", module.gather, path=Path("consumer.txt"), sources=recipe.outputs.branches.path)
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    if parent_state == "current":
        _complete_branches(directory)
    process = invoke(directory, "run", "--step", "1", "--task", "0")
    bundle, execution = _load(directory)
    invocation = _latest(directory, "unit-1-0")
    record = AttemptRecord.read(
        directory / "results" / str(invocation.attempt_id) / "final.json",
        workflow_id=execution.workflow_id,
        attempt_id=invocation.attempt_id,
        step_path="mixed.consumer",
        bundle_digest=bundle.digest,
    )
    if parent_state == "current":
        assert process.returncode == 0, process.stderr
        assert record.committed and record.source_attempts == {"gather-0": _latest(directory, "gather-0").attempt_id}
        assert (workspace / "consumer.txt").exists()
    else:
        assert process.returncode != 0 and record.state == "failed"
        assert not (workspace / "consumer.txt").exists()


def test_status_requeued_array_precedes_old_finalizer(graph, monkeypatch):
    from uuid import uuid4
    import shinobi.offload.mixed as mixed

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    _complete_branches(directory)
    process = invoke(directory, "finalize")
    assert process.returncode == 0, process.stderr
    handle = mixed.MixedHandle(recipe=recipe.name, submission=str(directory), jobs={"step-0": "814", "gather-0": "815", "finalize": "818"})
    mixed.write_new(directory / "handle.json", handle)
    live_state = ""
    accounting = "814_0|COMPLETED\n814_0.batch|RUNNING\n814_1|COMPLETED\n815|COMPLETED\n818|COMPLETED\n"

    def scheduler(argv, **kwargs):
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, stdout=accounting, stderr="")
        assert argv[0] == "squeue" and "--array" in argv
        return subprocess.CompletedProcess(argv, 0, stdout=live_state, stderr="")

    monkeypatch.setattr(mixed.subprocess, "run", scheduler)
    assert mixed.mixed_status(handle.model_dump()).exit_code == 0
    for state in ("PENDING", "RUNNING", "REQUEUED"):
        live_state = f"814_0|{state}\n"
        assert mixed.mixed_status(handle.model_dump()).running
    live_state = ""
    accounting = accounting.replace("814_1|COMPLETED\n", "")
    assert mixed.mixed_status(handle.model_dump()).exit_code != 0
    accounting += "814_1|COMPLETED\n"
    prior = mixed._latest(directory, "unit-0-0")
    newer = prior.model_copy(update={"attempt_id": uuid4(), "restart": 1})
    mixed.write_new(directory / "invocations" / "unit-0-0" / f"{newer.attempt_id}.json", newer)
    assert mixed.mixed_status(handle.model_dump()).exit_code != 0


def test_mixed_scheduler_matches_singleton_arrays_and_requires_live_query(monkeypatch):
    from types import SimpleNamespace
    import shinobi.offload.mixed as mixed

    execution = SimpleNamespace(plan=SimpleNamespace(units=[SimpleNamespace(step_index=0, slice_index=None)]))
    live_returncode = 0

    def scheduler(argv, **kwargs):
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, stdout="42_0|COMPLETED\n42_0.batch|RUNNING\n", stderr="")
        return subprocess.CompletedProcess(argv, live_returncode, stdout="", stderr="")

    monkeypatch.setattr(mixed.subprocess, "run", scheduler)
    assert mixed._mixed_scheduler_states({"step-0": "42"}, execution) == {"step-0_0": "COMPLETED"}
    live_returncode = 1
    assert mixed._mixed_scheduler_states({"step-0": "42"}, execution) == {"step-0_0": "UNKNOWN"}


def _legacy_campaign(graph):
    """Make genuine v6 fixtures from the explicit eager compatibility API."""
    import shinobi.offload.mixed as mixed
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.record_store import read_terminal

    recipe, source, workspace = graph
    directory = prepare_mixed(freeze_mixed(recipe, {}, config=config(), workspace=workspace, code_roots=(source,)), submission_root=workspace / "submissions")
    _complete_branches(directory)
    process = invoke(directory, "finalize")
    assert process.returncode == 0, process.stderr
    bundle, execution = mixed._load(directory)
    legacy = {}
    for name in ("unit-0-0", "unit-0-1", "gather-0", "finalize"):
        invocation = mixed._latest(directory, name)
        path = directory / "results" / str(invocation.attempt_id) / "final.json"
        terminal = read_terminal(path)
        value = mixed._read_result(directory, bundle, execution, name)
        record = AttemptRecord.from_result(
            value, workflow_id=execution.workflow_id, attempt_id=invocation.attempt_id, step_path=terminal.step_path, bundle_digest=bundle.digest, composite=True
        )
        legacy[path] = record.model_copy(update={"source_attempts": terminal.source_attempts}).model_dump_json(indent=2).encode()
    for path, data in legacy.items():
        path.write_bytes(data)
    return directory, legacy


def test_migration_closure_interruption_retry_and_bounded_monitoring(graph, monkeypatch):
    import shinobi.offload.mixed as mixed
    from shinobi.offload.record_store import Store, read_terminal
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.tracking import FINISHED

    directory, originals = _legacy_campaign(graph)
    bundle, execution = mixed._load(directory)
    with pytest.raises(BundleError, match="convert-records"):
        mixed._CurrentEvidence(directory, bundle, execution, legacy_limit=1).read("finalize")
    assert mixed._read_result(directory, bundle, execution, "finalize").success
    original = mixed.write_new
    published = 0

    def interrupt(path, value):
        nonlocal published
        if path.name == "final.v7.json":
            published += 1
            if published == 2:
                raise OSError("conversion interrupted")
        return original(path, value)

    monkeypatch.setattr(mixed, "write_new", interrupt)
    with pytest.raises(OSError, match="conversion interrupted"):
        mixed.convert_records(directory)
    assert len(list(directory.glob("results/*/final.v7.json"))) == 1
    monkeypatch.setattr(mixed, "write_new", original)
    mixed.convert_records(directory)
    sidecars = {path: path.read_bytes() for path in directory.glob("results/*/final.v7.json")}
    assert len(sidecars) == 4
    mixed.convert_records(directory)
    assert all(path.read_bytes() == data for path, data in sidecars.items())
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert all(read_terminal(path).schema_version == 7 for path in originals)
    converted = subprocess.run([sys.executable, "-m", "shinobi.offload.mixed", "convert-records", "--submission", str(directory)], capture_output=True, text=True)
    assert converted.returncode == 0, converted.stderr

    def forbidden(*args, **kwargs):
        raise AssertionError("eager reconstruction used during monitoring")

    monkeypatch.setattr(Store, "result", forbidden)
    monkeypatch.setattr(AttemptRecord, "result", forbidden)
    monkeypatch.setattr(mixed, "_read_result", forbidden)
    monkeypatch.setattr(mixed, "_mixed_scheduler_states", lambda *args: {"fixture": "COMPLETED"})
    assert mixed.current_manifest(directory).exists()
    state = mixed.mixed_status({"submission": str(directory), "jobs": {}})
    assert state.state == FINISHED and state.exit_code == 0
    final = mixed._latest(directory, "finalize")
    final_path = directory / "results" / str(final.attempt_id) / "final.json"
    sidecar_path = final_path.with_name("final.v7.json")
    saved_sidecar = sidecar_path.read_bytes()
    sidecar_path.write_text(read_terminal(final_path).model_copy(update={"manifest": None}).model_dump_json())
    with pytest.raises(BundleError, match="no bound manifest"):
        mixed.current_manifest(directory)
    state = mixed.mixed_status({"submission": str(directory), "jobs": {}})
    assert state.state != FINISHED
    sidecar_path.write_bytes(saved_sidecar)
    path = next(iter(originals))
    path.write_bytes(originals[path] + b" ")
    with pytest.raises(BundleError, match="digest"):
        mixed.current_manifest(directory)


def test_migration_refuses_conflicting_sidecar_and_requeue(graph, monkeypatch):
    from uuid import uuid4
    import shinobi.offload.mixed as mixed
    from shinobi.offload.record_store import Record

    directory, originals = _legacy_campaign(graph)
    mixed.convert_records(directory)
    path = next(iter(originals)).with_name("final.v7.json")
    before = path.read_bytes()
    record = Record.model_validate_json(before).model_copy(update={"error": "contradictory conversion"})
    path.write_text(record.model_dump_json())
    with pytest.raises(BundleError, match="contradicts"):
        mixed.convert_records(directory)
    path.write_bytes(before)
    current = mixed._latest(directory, "unit-0-0")
    newer = current.model_copy(update={"attempt_id": uuid4(), "restart": 1})
    mixed.write_new(directory / "invocations" / current.name / f"{newer.attempt_id}.json", newer)
    with pytest.raises(BundleError, match="superseded predecessor"):
        mixed.convert_records(directory)
    with pytest.raises(BundleError, match="superseded predecessor"):
        mixed.current_manifest(directory)


def test_migration_rechecks_invocations_after_publication(graph, monkeypatch):
    from uuid import uuid4
    import shinobi.offload.mixed as mixed

    directory, _ = _legacy_campaign(graph)
    original = mixed.write_new
    changed = False

    def publish(path, value):
        nonlocal changed
        result = original(path, value)
        if path.name == "final.v7.json" and not changed:
            changed = True
            current = mixed._latest(directory, "unit-0-0")
            newer = current.model_copy(update={"attempt_id": uuid4(), "restart": 1})
            original(directory / "invocations" / current.name / f"{newer.attempt_id}.json", newer)
        return result

    monkeypatch.setattr(mixed, "write_new", publish)
    with pytest.raises(BundleError, match="invocation changed"):
        mixed.convert_records(directory)
    assert not (directory / "results" / str(mixed._latest(directory, "finalize").attempt_id) / "final.v7.json").exists()
