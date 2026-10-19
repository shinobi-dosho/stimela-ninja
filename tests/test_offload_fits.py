"""Qualified FITS worker transport, exact oracle and namespace contracts."""

from pathlib import Path
import sys

import pytest
from pydantic import BaseModel

from shinobi import FITSImage, FITSAccess, Cab, Recipe, InputRef, StepRef
from shinobi.steps.schema import ParamMeta
from shinobi.config import AppConfig
from shinobi.fits_access import resolve_scope_fits_accesses
from shinobi.fits_lifecycle import plan_fits_backend, committed_generation
from shinobi.offload.bundle import freeze_recipe, write_new
from shinobi.offload.slurm import prepare_worker_slurm
from shinobi.offload.worker import ExecutionPlan, execute_step
from shinobi.ownership import acquire_workspace
from shinobi.snapshots import chain_id, get_journal, faults, reconcile
from shinobi.cache import get_cache_manifest
from shinobi.exceptions import DatasetLifecycleUnavailableError
from tests._shared_storage import qualified_storage
from tests.test_offload_dataset_lifecycle import _record

fits = pytest.importorskip("astropy.io.fits")
from tests.test_fits import image  # noqa: E402


class Empty(BaseModel):
    pass


class Inputs(BaseModel):
    file: FITSImage


class ToolInputs(Inputs):
    script: str


def recipe(workspace, *, read=False):
    script = (
        "from astropy.io import fits;import sys;f=fits.open(sys.argv[1],mode='update');f[0].data[:]=7;f.close()"
        if not read
        else "from pathlib import Path;import sys;Path(sys.argv[1]).read_bytes()"
    )
    cab = Cab(
        name="tool",
        command=f"{sys.executable} -c",
        inputs_model=ToolInputs,
        outputs_model=Empty,
        fits_accesses=[FITSAccess(field="file", mode="read" if read else "write")],
        field_meta={"script": ParamMeta(positional_head=True), "file": ParamMeta(positional=True)},
    )
    return Recipe(
        name="fits-worker",
        inputs_model=Inputs,
        outputs_model=Empty,
        steps=[StepRef(name="tool", step=cab, params={"script": script}, wiring={"file": InputRef(field="file")})],
        cache_dir=str(workspace / "cache"),
    )


def prepare(tmp_path, monkeypatch, *, read=False):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    path = image(tmp_path / "image.fits")
    bundle = freeze_recipe(recipe(tmp_path, read=read), {"file": path}, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable), dataset_storage_qualification=qualified_storage(tmp_path))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(a.path), a.writes) for a in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    return path, workflow, plan, lease


@pytest.mark.parametrize("read", [False, True])
def test_worker_fits_closed_evidence(tmp_path, monkeypatch, read):
    path, workflow, plan, lease = prepare(tmp_path, monkeypatch, read=read)
    try:
        assert plan.schema_version == 3 and plan.dataset_lifecycle is None
        assert plan.ownership_required and plan.fits_lifecycle is not None
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = _record(workflow, plan)
        assert record.schema_version == 5 and record.fits_lifecycle.outcome == "committed"
        if not read:
            assert fits.getdata(path)[0, 0] == 7
            chain = get_journal(str(tmp_path / "cache")).get(chain_id(path))
            reference, leaf = committed_generation(chain.generation(chain.head), path, workspace=tmp_path)
            assert reference.profile == "fits-commit-reference/v2"
            assert reference.oracle.name == "final.json"
            assert leaf.publication is not None
    finally:
        lease.release()


@pytest.mark.parametrize("backend", ["docker", "podman", "apptainer", "singularity"])
def test_fits_mounts_allow_replacement_protect_siblings(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("DOCKER_CONTEXT", "default")
    monkeypatch.setenv("CONTAINER_HOST", "unix:///tmp/podman.sock")
    path = image(tmp_path / "image.fits")
    sibling = image(tmp_path / "sibling.fits")
    writer = recipe(tmp_path).steps[0].step
    reader = recipe(tmp_path, read=True).steps[0].step
    write = resolve_scope_fits_accesses(writer, {"file": path}, workspace=tmp_path)
    read = resolve_scope_fits_accesses(reader, {"file": sibling}, workspace=tmp_path)
    plan = plan_fits_backend(backend, (*write, *read), workspace=tmp_path, mutation=True, leaf_accesses=write)
    assert {(one.source, one.writable) for one in plan.mounts} == {(tmp_path, True), (sibling, False)}
    assert path not in {one.source for one in plan.mounts}
    sibling.unlink()
    with pytest.raises(DatasetLifecycleUnavailableError, match="absent nonowned"):
        plan_fits_backend(backend, (*write, *read), workspace=tmp_path, mutation=True, leaf_accesses=write)


@pytest.mark.parametrize("stage", ["S1", "S2", "W_RESULT"])
def test_worker_fits_oracle_recovery(tmp_path, monkeypatch, stage):
    path, workflow, plan, lease = prepare(tmp_path, monkeypatch)
    before = path.read_bytes()
    attempt = plan.attempts[0]

    def stop():
        raise RuntimeError("injected crash")

    faults.hooks[stage] = stop
    try:
        execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id)
        faults.hooks.clear()
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
        assert (path.read_bytes() == before) == (stage != "W_RESULT")
    finally:
        faults.hooks.clear()
        lease.release()


def prepared_recipe(tmp_path, monkeypatch, scope, inputs):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    bundle = freeze_recipe(scope, inputs, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable), dataset_storage_qualification=qualified_storage(tmp_path))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    lease = acquire_workspace(
        Path(plan.ownership_workspace),
        str(plan.workflow_id),
        kind="slurm",
        submission=workflow.submission_dir,
        accesses=((Path(a.path), a.writes) for a in plan.accesses),
        registry=Path(plan.ownership_registry),
    )
    write_new(workflow.submission_dir / "ownership.json", lease.owner)
    return workflow, plan, lease


def invoke(workflow, plan):
    attempt = plan.attempts[0]
    code = execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id)
    record = _record(workflow, plan)
    assert code == 0, record.error or record.stderr
    return record


class CreateInputs(BaseModel):
    script: str
    seed: Path
    target: str


class CreateRoot(BaseModel):
    seed: Path
    target: str


def create_recipe(tmp_path):
    cab = Cab(
        name="create",
        command=f"{sys.executable} -c",
        inputs_model=CreateInputs,
        outputs_model=Inputs,
        field_meta={"script": ParamMeta(positional_head=True), "seed": ParamMeta(positional=True), "target": ParamMeta(positional=True), "file": ParamMeta(implicit="{target}")},
        fits_accesses=[FITSAccess(field="file", mode="create")],
    )
    return Recipe(
        name="create-worker",
        inputs_model=CreateRoot,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="create",
                step=cab,
                params={"script": "import shutil,sys;shutil.copyfile(sys.argv[1],sys.argv[2])"},
                wiring={"seed": InputRef(field="seed"), "target": InputRef(field="target")},
            )
        ],
        cache_dir=str(tmp_path / "cache"),
    )


def test_create_worker_and_lost_index_repair(tmp_path, monkeypatch):
    seed = image(tmp_path / "seed.fits")
    target = tmp_path / "new.fits"
    scope = create_recipe(tmp_path)
    inputs = {"seed": seed, "target": str(target)}
    workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, scope, inputs)
    try:
        record = invoke(workflow, plan)
        assert record.state == "succeeded" and target.exists()
    finally:
        lease.release()
    manifest = get_cache_manifest(str(tmp_path / "cache"))
    manifest.update(lambda data: data.clear())
    workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, scope, inputs)
    try:
        assert invoke(workflow, plan).state == "cached"
    finally:
        lease.release()


def test_create_worker_failure_restores_absence(tmp_path, monkeypatch):
    seed = image(tmp_path / "seed.fits")
    target = tmp_path / "new.fits"
    workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, create_recipe(tmp_path), {"seed": seed, "target": str(target)})
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    try:
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        assert not target.exists()
    finally:
        faults.hooks.clear()
        lease.release()


def test_detached_corrupt_oracle_retains_fits_fences(tmp_path, monkeypatch):
    path, workflow, plan, lease = prepare(tmp_path, monkeypatch)
    faults.hooks["W_RESULT"] = lambda: (_ for _ in ()).throw(RuntimeError("stop after oracle"))
    try:
        record = invoke(workflow, plan)
        faults.hooks.clear()
        oracle = workflow.submission_dir / "attempts" / str(record.attempt_id) / "final.json"
        oracle.write_text("{}")
        with pytest.raises(DatasetLifecycleUnavailableError, match="malformed"):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
        assert get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is not None
        assert fits.getdata(path)[0, 0] == 7
    finally:
        faults.hooks.clear()
        lease.release()


class ListRoot(BaseModel):
    files: list[FITSImage]


class ListTool(ListRoot):
    script: str


def test_joint_list_worker_exact_participant_recovery(tmp_path, monkeypatch):
    paths = [image(tmp_path / f"{index}.fits") for index in range(2)]
    cab = Cab(
        name="joint",
        command=f"{sys.executable} -c",
        inputs_model=ListTool,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional_head=True), "files": ParamMeta(positional=True)},
        fits_accesses=[FITSAccess(field="files", mode="write")],
    )
    scope = Recipe(
        name="joint-worker",
        inputs_model=ListRoot,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="joint",
                step=cab,
                params={"script": "import sys;from astropy.io import fits\nfor p in sys.argv[1].split(','):\n with fits.open(p,mode='update') as f: f[0].data[:]=9"},
                wiring={"files": InputRef(field="files")},
            )
        ],
        cache_dir=str(tmp_path / "cache"),
    )
    workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, scope, {"files": paths})
    faults.hooks["W_RESULT"] = lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    try:
        record = invoke(workflow, plan)
        assert len(record.fits_lifecycle.leaves[0].mutations) == 2
        faults.hooks.clear()
        with pytest.raises(DatasetLifecycleUnavailableError):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={paths[0]}, exact=True)
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
        assert all(fits.getdata(path)[0, 0] == 9 for path in paths)
    finally:
        faults.hooks.clear()
        lease.release()


@pytest.mark.parametrize("ms_authority", [False, True])
def test_worker_typed_fits_products_fresh_and_cached(tmp_path, monkeypatch, ms_authority):
    from shinobi import FamilySpec, MemberRule, MeasurementSetV2, DatasetAccess
    from shinobi.products import ProductFamily
    from tests.test_offload_dataset_lifecycle import _install_dataset

    class ProductInputs(BaseModel):
        script: str
        seed: Path

    class MSProductInputs(ProductInputs):
        ms: MeasurementSetV2

    class ProductRoot(BaseModel):
        seed: Path

    class MSProductRoot(ProductRoot):
        ms: MeasurementSetV2

    class Products(BaseModel):
        images: ProductFamily[FITSImage]

    seed = image(tmp_path / "seed.fits")
    inputs = {"seed": seed}
    if ms_authority:
        ms = tmp_path / "data.ms"
        ms.mkdir()
        (ms / "table.dat").write_text("raw")
        _install_dataset(monkeypatch, tmp_path, ms)
        inputs["ms"] = ms
    cab = Cab(
        name="products",
        command=f"{sys.executable} -c",
        inputs_model=MSProductInputs if ms_authority else ProductInputs,
        outputs_model=Products,
        dataset_accesses=[DatasetAccess(field="ms", mode="read")] if ms_authority else [],
        field_meta={
            "script": ParamMeta(positional_head=True),
            "seed": ParamMeta(positional=True),
            **({"ms": ParamMeta(positional=True)} if ms_authority else {}),
            "images": ParamMeta(family=FamilySpec(coordinates={}, rules=(MemberRule(path="made.fits", required=True),))),
        },
    )
    scope = Recipe(
        name="products-worker",
        inputs_model=MSProductRoot if ms_authority else ProductRoot,
        outputs_model=Empty,
        steps=[
            StepRef(
                name="products",
                step=cab,
                params={"script": "import shutil,sys;shutil.copyfile(sys.argv[1],'made.fits')"},
                wiring={field: InputRef(field=field) for field in inputs},
            )
        ],
        cache=True,
        cache_dir=str(tmp_path / "cache"),
    )
    for expected in ("succeeded", "cached"):
        workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, scope, inputs)
        try:
            record = invoke(workflow, plan)
            assert record.state == expected
            assert record.fits_lifecycle is None
            assert (record.dataset_lifecycle is not None) == ms_authority
        finally:
            lease.release()
    (tmp_path / "made.fits").write_bytes(b"corrupt")
    workflow, plan, lease = prepared_recipe(tmp_path, monkeypatch, scope, inputs)
    try:
        assert invoke(workflow, plan).state == "succeeded"
        assert fits.getdata(tmp_path / "made.fits").shape == (2, 3)
    finally:
        lease.release()


def test_fits_worker_requires_qualified_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = image(tmp_path / "image.fits")
    bundle = freeze_recipe(recipe(tmp_path, read=True), {"file": path}, config=AppConfig(), workspace=tmp_path)
    with pytest.raises(DatasetLifecycleUnavailableError, match="qualification"):
        prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))


def test_fits_worker_dependency_preflight(tmp_path, monkeypatch):
    import subprocess
    import shinobi.offload.slurm as slurm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    path = image(tmp_path / "image.fits")
    original = subprocess.run

    def missing(argv, **kwargs):
        if argv[-1] == "from astropy.io import fits":
            return subprocess.CompletedProcess(argv, 1, "", "No module named astropy")
        return original(argv, **kwargs)

    monkeypatch.setattr(slurm.subprocess, "run", missing)
    bundle = freeze_recipe(recipe(tmp_path, read=True), {"file": path}, config=AppConfig(), workspace=tmp_path)
    with pytest.raises(DatasetLifecycleUnavailableError, match=r"fits-contract.*astropy"):
        prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable), dataset_storage_qualification=qualified_storage(tmp_path))


def test_old_worker_records_omit_new_fits_fields(tmp_path):
    from uuid import uuid4
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.worker import WorkerEnvironment

    record = AttemptRecord(workflow_id=uuid4(), attempt_id=uuid4(), step_path="ordinary", bundle_digest="f" * 64, state="running")
    assert "fits_lifecycle" not in record.model_dump(mode="json")
    plan = ExecutionPlan(
        workflow_id=uuid4(),
        bundle_digest="f" * 64,
        worker=WorkerEnvironment(python=sys.executable, source=str(tmp_path), source_digest="f" * 64, python_version="3.12", platform="linux"),
        attempts=(),
    )
    assert "fits_lifecycle" not in plan.model_dump(mode="json")


def test_requeue_recovers_invalid_partial_fits_before_inspection(tmp_path, monkeypatch):
    import os
    import subprocess
    from shinobi.offload.worker import _latest_attempt
    from shinobi.offload.records import AttemptRecord

    path, workflow, plan, lease = prepare(tmp_path, monkeypatch)
    attempt = plan.attempts[0]
    script = """import os,sys
from pathlib import Path
from uuid import UUID
from shinobi.snapshots import faults
from shinobi.offload.worker import execute_step
def crash():
 Path(sys.argv[1]).write_bytes(b'partial invalid FITS')
 os._exit(73)
faults.hooks['P_RESTORED']=crash
execute_step(Path(sys.argv[2]),sys.argv[3],UUID(sys.argv[4]))
"""
    try:
        env = dict(os.environ, PYTHONPATH=str(workflow.submission_dir / "worker-src"))
        proc = subprocess.run(
            [sys.executable, "-c", script, str(path), str(workflow.submission_dir), attempt.step_path, str(attempt.attempt_id)], env=env, capture_output=True, text=True
        )
        assert proc.returncode == 73, proc.stderr
        assert path.read_bytes() == b"partial invalid FITS"
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        actual = _latest_attempt(workflow.submission_dir, attempt)
        assert actual != attempt.attempt_id
        record = AttemptRecord.read(
            workflow.submission_dir / "attempts" / str(actual) / "final.json",
            workflow_id=plan.workflow_id,
            attempt_id=actual,
            step_path=attempt.step_path,
            bundle_digest=plan.bundle_digest,
        )
        assert record.committed and fits.getdata(path)[0, 0] == 7
        assert get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is None
    finally:
        lease.release()


@pytest.mark.parametrize("backend", ["docker", "podman", "apptainer", "singularity"])
def test_pystep_container_launcher_uses_fits_replacement_plan(tmp_path, monkeypatch, backend):
    from types import SimpleNamespace, MethodType
    from shinobi.steps.dispatch import ExecContext
    from shinobi.steps.pyfunc import _ContainerLauncher
    from shinobi.steps.schema import Scope

    monkeypatch.setenv("DOCKER_CONTEXT", "default")
    monkeypatch.setenv("CONTAINER_HOST", "unix:///tmp/podman.sock")
    path = image(tmp_path / "image.fits")
    scope = Scope(name="python", inputs_model=Inputs, outputs_model=Empty, image="tool:latest", fits_accesses=[FITSAccess(field="file", mode="write")])
    accesses = resolve_scope_fits_accesses(scope, {"file": path}, workspace=tmp_path)
    lifecycle = SimpleNamespace(
        kind="fits", workspace=tmp_path, mutation=True, record=SimpleNamespace(planned_accesses=accesses), resolve_claimed_leaf_accesses=lambda *args: accesses
    )
    ctx = SimpleNamespace(scope=scope, _dataset_lifecycle=lifecycle, _config=AppConfig(), _pin=False)
    ctx.dataset_backend_plan = MethodType(ExecContext.dataset_backend_plan, ctx)
    captured = {}

    def build(*args, **kwargs):
        captured.update(kwargs)
        return ["container", "python"], None

    monkeypatch.setattr("shinobi.backends.container.build_container_argv", build)
    _ContainerLauncher(backend, ctx).build(tmp_path / "runner.py", str(tmp_path), [], {"file": path})
    assert captured["dataset_mounts"] == [(str(tmp_path), True)]


def test_fits_planner_rejects_missing_parent_and_mount_collisions(tmp_path):
    from shinobi.backends.container import merge_dataset_mounts
    from shinobi.exceptions import BackendError

    scope = recipe(tmp_path).steps[0].step
    path = tmp_path / "missing" / "image.fits"
    accesses = resolve_scope_fits_accesses(scope, {"file": path}, workspace=tmp_path)
    with pytest.raises(DatasetLifecycleUnavailableError, match="existing non-root parent"):
        plan_fits_backend("apptainer", accesses, workspace=tmp_path, mutation=True)
    with pytest.raises(BackendError, match="conflicts|same path"):
        merge_dataset_mounts([(str(tmp_path), False)], [(str(tmp_path), True)], scope_name="fits")


def test_worker_refuses_runtime_resolved_fits_path_before_submission(tmp_path, monkeypatch):
    from shinobi import OutputRef

    class RuntimeOutput(BaseModel):
        file: str

    producer = Cab(name="runtime", command="true", inputs_model=Empty, outputs_model=RuntimeOutput)
    reader = recipe(tmp_path, read=True).steps[0].step
    scope = Recipe(
        name="runtime-fits",
        inputs_model=Empty,
        outputs_model=Empty,
        cache_dir=str(tmp_path / "cache"),
        steps=[StepRef(name="runtime", step=producer), StepRef(name="read", step=reader, params={"script": "pass"}, wiring={"file": OutputRef(step="runtime", field="file")})],
    )
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    bundle = freeze_recipe(scope, {}, config=AppConfig(), workspace=tmp_path)
    with pytest.raises(DatasetLifecycleUnavailableError, match="statically resolved"):
        prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable), dataset_storage_qualification=qualified_storage(tmp_path))
