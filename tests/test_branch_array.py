import importlib
import json
import os
import subprocess
import sys

import pytest

from shinobi.config import AppConfig
from shinobi.offload._codec import BundleError
from shinobi.offload.branch import BranchCase, prepare_branch_array

SOURCE = """from pathlib import Path
import os
import startup_values
from pydantic import BaseModel
from shinobi import Recipe, pystep
class Empty(BaseModel):
    pass
class Product(BaseModel):
    path: Path
@pystep()
def work(filename: str, crash: bool = False) -> Product:
    if crash and filename == "b.txt" and not Path("interrupted").exists():
        Path("interrupted").write_text("once")
        os._exit(73)
    path = Path(filename).resolve()
    path.write_text(str(int(path.read_text()) + 1) if path.exists() else "1")
    return Product(path=path)
def build(backend="native", crash=False, nested=False):
    assert startup_values.STARTUP == os.environ.get("BRANCH_ARRAY_TEST"), "environment was applied after import"
    recipe = Recipe(name="branch", inputs_model=Empty, outputs_model=Empty, backend=backend, cache=True, max_workers=1)
    recipe.add_step("work", work, scatter=["filename"], filename=["a.txt", "b.txt"], crash=crash)
    if nested:
        outer = Recipe(name="outer", inputs_model=Empty, outputs_model=Empty, cache=True, backend=backend)
        outer.add_step("branch", recipe)
        return outer
    return recipe
"""


@pytest.fixture
def builder(tmp_path, monkeypatch):
    source = tmp_path / "code"
    source.mkdir()
    (source / "branch_fixture.py").write_text(SOURCE)
    (source / "startup_values.py").write_text('import os\nSTARTUP = os.environ.get("BRANCH_ARRAY_TEST")\n')
    monkeypatch.setenv("BRANCH_ARRAY_TEST", "staged")
    monkeypatch.syspath_prepend(str(source))
    sys.modules.pop("branch_fixture", None)
    sys.modules.pop("startup_values", None)
    yield importlib.import_module("branch_fixture").build, source
    sys.modules.pop("branch_fixture", None)
    sys.modules.pop("startup_values", None)


def invoke(directory, index=0):
    env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(index)}
    env.pop("BRANCH_ARRAY_TEST", None)
    return subprocess.run(["bash", str(directory / "array.sbatch")], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("nested", [False, True])
def test_array_runs_scatter_and_resumes_missing_product(builder, tmp_path, nested):
    build, source = builder
    directory = tmp_path / "submission"
    workspace = tmp_path / "case"
    prepare_branch_array(
        build,
        [BranchCase(name="one", workspace=workspace, parameters={"nested": nested})],
        directory=directory,
        code_roots=[source],
        config=AppConfig(backend={"default": "native"}),
        environment={"BRANCH_ARRAY_TEST": "staged"},
    )
    assert "--array=0-0%1" in (directory / "array.sbatch").read_text()
    assert "--open-mode=append" in (directory / "array.sbatch").read_text()
    for _ in range(2):
        result = invoke(directory)
        assert result.returncode == 0, result.stderr
    assert (workspace / "a.txt").read_text() == (workspace / "b.txt").read_text() == "1"
    (workspace / "a.txt").unlink()
    result = invoke(directory)
    assert result.returncode == 0, result.stderr
    assert (workspace / "a.txt").read_text() == (workspace / "b.txt").read_text() == "1"
    records = list((directory / "attempts/0").glob("*/finished.json"))
    assert len(records) == 3
    assert all(json.loads(path.read_text())["status"] == "completed" for path in records)
    (directory / "source/branch_fixture.py").write_text("raise RuntimeError('tampered')")
    result = invoke(directory)
    assert result.returncode != 0 and "branch source differs" in result.stderr


def test_refuses_nested_offloading_before_staging(builder, tmp_path):
    build, source = builder
    directory = tmp_path / "submission"
    with pytest.raises(BundleError, match="nested offloading"):
        prepare_branch_array(build, [BranchCase(name="one", workspace=tmp_path / "case", parameters={"backend": "slurm"})], directory=directory, code_roots=[source])
    assert not directory.exists()


def test_killed_driver_recovers_exact_prior_claim_and_cached_slice(builder, tmp_path):
    build, source = builder
    directory = tmp_path / "submission"
    workspace = tmp_path / "case"
    prepare_branch_array(
        build,
        [BranchCase(name="one", workspace=workspace, parameters={"crash": True})],
        directory=directory,
        code_roots=[source],
        config=AppConfig(backend={"default": "native"}),
        environment={"BRANCH_ARRAY_TEST": "staged"},
    )
    assert invoke(directory).returncode == 73
    assert (workspace / "a.txt").read_text() == "1"
    result = invoke(directory)
    assert result.returncode == 0, result.stderr
    assert (workspace / "a.txt").read_text() == (workspace / "b.txt").read_text() == "1"
    assert len(list((directory / "attempts/0").glob("*/started.json"))) == 2
    assert len(list((directory / "attempts/0").glob("*/finished.json"))) == 1


@pytest.mark.parametrize("tamper", [False, True])
def test_recovery_never_releases_live_or_mismatched_attempt(builder, tmp_path, tamper):
    from uuid import uuid4
    from shinobi.offload.branch import BranchArray, BranchRecord, recover_branch_claims
    from shinobi.offload.bundle import fingerprint, write_new
    from shinobi.ownership import WorkspaceOwnershipError, acquire_workspace, inspect_workspace

    build, source = builder
    directory = tmp_path / "submission"
    case = BranchCase(name="one", workspace=tmp_path / "case")
    prepare_branch_array(build, [case], directory=directory, code_roots=[source], config=AppConfig(backend={"default": "native"}), environment={"BRANCH_ARRAY_TEST": "staged"})
    bundle = BranchArray.model_validate_json((directory / "branch-array.json").read_text())
    record = BranchRecord(
        workflow_id=bundle.workflow_id,
        bundle_digest=fingerprint(bundle),
        case=case.name,
        attempt_id=str(uuid4()),
        status="running",
        workspace=str(case.workspace),
        submission=str(directory),
    )
    write_new(directory / "attempts/0" / record.attempt_id / "started.json", record)
    write_new(case.workspace / ".shinobi/branch-attempts" / f"{record.attempt_id}.json", record.model_copy(update={"status": "failed"}) if tamper else record)
    lease = acquire_workspace(case.workspace, record.attempt_id, kind="local", registry=tmp_path / "workspace-owners.json")
    try:
        with pytest.raises((WorkspaceOwnershipError, BundleError), match="started record" if tamper else "live"):
            recover_branch_claims(case, "new-attempt")
        assert inspect_workspace(case.workspace).workflow_id == record.attempt_id
    finally:
        lease.release()
