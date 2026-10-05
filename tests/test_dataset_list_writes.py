"""Joint MSv2 mutation, using distinct real tables to detect cross-root restores."""

import pytest
from pydantic import BaseModel

from shinobi import MSv2, DatasetAccess, DatasetColumns, Recipe, pystep
from shinobi.cache import ProvenanceKey, get_cache_manifest
from shinobi.dataset_lifecycle import DatasetMutationOutcome
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.snapshots import chain_id, faults, get_journal, reconcile, state_name
from shinobi.steps.schema import InputRef, OutputRef
from tests._dataset_fixtures import attempts, make_ms, scans, set_scans

WRITE = DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))


class Inputs(BaseModel):
    ms: list[MSv2]


class Empty(BaseModel):
    pass


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    faults.hooks.clear()


def kwargs(tmp_path):
    return {"cache": True, "cache_dir": str(tmp_path / "cache")}


def roots(tmp_path):
    return [make_ms(tmp_path / "a.ms", scan=2), make_ms(tmp_path / "b.ms", scan=7)]


@pystep(dataset_accesses=[WRITE])
def increment(ms: list[MSv2], by: int = 1) -> Inputs:
    for root in ms:
        set_scans(root, scans(root)[0] + by)
    return Inputs(ms=ms)


def values(paths):
    return [scans(path)[0] for path in paths]


def test_joint_success_cache_and_indexed_identity(tmp_path):
    paths = roots(tmp_path)
    result = increment(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [3, 8]
    keys = result.provenance_key("ms")
    assert [key.element_index for key in keys] == [0, 1]
    journal = get_journal(str(tmp_path / "cache"))
    for index, path in enumerate(paths):
        chain = journal.get(chain_id(path))
        assert chain.marker is None
        assert chain.head == state_name(result.cache_key, "ms", index)
    assert increment(ms=paths, **kwargs(tmp_path)).cached
    assert values(paths) == [3, 8]
    assert [mutation.element_index for mutation in attempts(tmp_path)[0].leaves[0].mutations] == [0, 1]


@pytest.mark.parametrize("size", [1, 2])
def test_list_rerun_uses_own_predecessor_after_reorder(tmp_path, size):
    paths = roots(tmp_path)
    increment(ms=paths, **kwargs(tmp_path))
    selected = paths[::-1][:size]
    increment(ms=selected, by=4, **kwargs(tmp_path))
    assert values(selected) == [11, 6][:size]


@pytest.mark.parametrize("failure", ["raise", "postcondition", "output-order", "output-size", "shared-list"])
def test_joint_failure_restores_all(tmp_path, failure):
    paths = roots(tmp_path)

    @pystep(dataset_accesses=[WRITE])
    def bad(ms: list[MSv2]) -> Inputs:
        for root in ms:
            set_scans(root, 90)
        if failure == "raise":
            raise RuntimeError("joint failure")
        if failure == "postcondition":
            import casacore.tables as tables

            with tables.table(str(ms[1]), readonly=False, ack=False) as table:
                table.addrows(1)
        if failure in ("output-order", "shared-list"):
            ms.reverse()
        if failure == "output-size":
            ms.pop()
        return Inputs(ms=ms)

    with pytest.raises((RuntimeError, DatasetLifecycleViolationError)):
        bad(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [2, 7]
    journal = get_journal(str(tmp_path / "cache"))
    assert all(journal.get(chain_id(path)).marker is None for path in paths)
    assert all(m.outcome is DatasetMutationOutcome.ROLLED_BACK for m in attempts(tmp_path)[-1].leaves[0].mutations)


@pytest.mark.parametrize("stage", ["P_MARKED", "P_RESTORED", "S1", "S2", "S3", "S4", "S5"])
def test_exception_boundaries_follow_exact_oracle(tmp_path, stage):
    paths = roots(tmp_path)

    def crash():
        raise RuntimeError(stage)

    faults.hooks[stage] = crash
    with pytest.raises((RuntimeError, DatasetLifecycleUnavailableError)):
        increment(ms=paths, **kwargs(tmp_path))
    faults.hooks.clear()
    expected = [3, 8] if stage in ("S3", "S4", "S5") else [2, 7]
    assert values(paths) == expected
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == expected
    assert all(get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is None for path in paths)


def test_later_rollback_failure_retains_every_fence_and_subset_refused(tmp_path, monkeypatch):
    import shinobi.snapshots as snapshots

    paths = roots(tmp_path)
    replace = snapshots._replace_tree_from_snapshot

    def fail_second(path, *args, **options):
        if path == paths[1]:
            raise OSError("second root blocked")
        return replace(path, *args, **options)

    monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", fail_second)

    @pystep(dataset_accesses=[WRITE])
    def bad(ms: list[MSv2]) -> None:
        for path in ms:
            set_scans(path, 99)
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        bad(ms=paths, **kwargs(tmp_path))
    journal = get_journal(str(tmp_path / "cache"))
    assert all(journal.get(chain_id(path)).marker is not None for path in paths)
    from shinobi.dataset_state import DatasetStateStore, StateError

    @pystep()
    def reader(ms: MSv2) -> None:
        pass

    for path in paths:
        with pytest.raises(DatasetLifecycleUnavailableError):
            reader(ms=path, **kwargs(tmp_path))
        with pytest.raises(StateError, match="recover it through its workflow"):
            DatasetStateStore(tmp_path / "states", cache_dir=tmp_path / "cache").export(path)
    manifest = get_cache_manifest(str(tmp_path / "cache"))
    before = values(paths)
    with pytest.raises(DatasetLifecycleUnavailableError, match="every exact participant"):
        reconcile(str(tmp_path / "cache"), manifest, paths={paths[0]}, exact=True)
    assert values(paths) == before
    with pytest.raises(DatasetLifecycleUnavailableError, match="exact participant paths"):
        reconcile(str(tmp_path / "cache"), manifest)
    monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", replace)
    reconcile(str(tmp_path / "cache"), manifest, paths=set(paths), exact=True)
    assert values(paths) == [2, 7]
    assert all(journal.get(chain_id(path)).marker is None for path in paths)


def test_whole_list_output_lineage_and_exact_rerun(tmp_path):
    paths = roots(tmp_path)
    recipe = Recipe(name="joint", inputs_model=Inputs, outputs_model=Inputs)
    recipe.add_step("first", increment, ms=InputRef(field="ms"), by=2)
    recipe.add_step("second", increment, ms=OutputRef(step="first", field="ms"), by=3)
    recipe.output_wiring = {"ms": OutputRef(step="second", field="ms")}
    result = recipe(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [7, 12]
    assert [key.element_index for key in result.provenance_key("ms")] == [0, 1]
    assert all(isinstance(key, ProvenanceKey) for key in result.provenance_key("ms"))
    assert recipe(ms=paths, **kwargs(tmp_path)).success
    assert values(paths) == [7, 12]
    recipe.steps[1].params["by"] = 4
    recipe(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [8, 13]


@pytest.mark.parametrize("both_producers", [False, True])
def test_scalar_producer_assembly_and_mixed_boundary_lineage(tmp_path, both_producers):
    paths = roots(tmp_path)

    class Boundary(BaseModel):
        first: MSv2
        second: MSv2

    class Scalar(BaseModel):
        ms: MSv2

    @pystep(dataset_accesses=[WRITE])
    def scalar(ms: MSv2) -> Scalar:
        set_scans(ms, scans(ms)[0] + 2)
        return Scalar(ms=ms)

    recipe = Recipe(name="assembled", inputs_model=Boundary, outputs_model=Inputs)
    if both_producers:
        recipe.add_step("first-producer", scalar, ms=InputRef(field="first"))
    recipe.add_step("scalar", scalar, ms=InputRef(field="second"))
    first_source = OutputRef(step="first-producer", field="ms") if both_producers else InputRef(field="first")
    recipe.add_step("joint", increment, ms=[first_source, OutputRef(step="scalar", field="ms")], by=3)
    recipe.output_wiring = {"ms": OutputRef(step="joint", field="ms")}
    assert recipe(first=paths[0], second=paths[1], **kwargs(tmp_path)).success
    assert values(paths) == ([7, 12] if both_producers else [5, 12])
    recipe.steps[-1].params["by"] = 4
    assert recipe(first=paths[0], second=paths[1], **kwargs(tmp_path)).success
    assert values(paths) == ([8, 13] if both_producers else [6, 13])


def _same_key_cab(scalar=False):
    import sys
    from pathlib import Path
    from shinobi import Cab
    from shinobi.steps.schema import ParamMeta

    class Tool(BaseModel):
        script: str = "import os,sys;from pathlib import Path;import casacore.tables as t;\nfor i,p in enumerate(sys.argv[2:]):\n with t.table(p,readonly=False,ack=False) as m: m.putcol('SCAN_NUMBER',m.getcol('SCAN_NUMBER')*0+int(os.environ['WRITE_VALUE'])+i*5)\nPath(sys.argv[1]).write_text('done')\n"
        target: str = "ticket"
        ms: MSv2 if scalar else list[MSv2]

    class Products(BaseModel):
        ms: MSv2 if scalar else list[MSv2]
        ticket: Path

    return Cab(
        name="nondeterministic",
        command=f"{sys.executable} -c",
        inputs_model=Tool,
        outputs_model=Products,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "target": ParamMeta(positional=True),
            "ms": ParamMeta(positional=True, repeat_as_tokens=not scalar),
            "ticket": ParamMeta(implicit="{target}"),
        },
        dataset_accesses=[WRITE],
    )


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4"])
def test_same_key_rerun_freezes_independent_rollback_bytes(tmp_path, monkeypatch, stage):
    paths = roots(tmp_path)
    nondeterministic = _same_key_cab()
    monkeypatch.setenv("WRITE_VALUE", "3")
    result = nondeterministic(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [3, 8]
    (tmp_path / "ticket").unlink()
    monkeypatch.setenv("WRITE_VALUE", "4")

    def crash():
        raise RuntimeError(stage)

    faults.hooks[stage] = crash
    with pytest.raises(RuntimeError):
        nondeterministic(ms=paths, **kwargs(tmp_path))
    faults.hooks.clear()
    assert values(paths) == ([4, 9] if stage in ("S3", "S4") else [3, 8])
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    journal = get_journal(str(tmp_path / "cache"))
    for index, path in enumerate(paths):
        state = state_name(result.cache_key, "ms", index)
        assert scans(journal.snapshot_dir(state))[0] == scans(path)[0]
        assert journal.get(chain_id(path)).marker is None


def test_indexed_names_cannot_collide_with_scalar_names():
    key = "0" * 64
    assert state_name(key, "ms") == key + "__ms"
    assert state_name(key, "ms[0]") == key + "__ms_0_"
    assert state_name(key, "ms", 0) != state_name(key, "ms[0]")
    assert state_name(key, "a-b", 0) != state_name(key, "a_b", 0)


def test_real_model_data_joint_creation_and_readonly_sibling(tmp_path):
    import casacore.tables as tables
    import numpy as np

    paths = roots(tmp_path)
    sibling = make_ms(tmp_path / "sibling.ms", scan=13)

    @pystep(
        dataset_accesses=[
            DatasetAccess(field="ms", mode="write", columns=DatasetColumns(create=("MODEL_DATA",)), allow_schema_change=True),
            DatasetAccess(field="reference", mode="read"),
        ]
    )
    def predict(ms: list[MSv2], reference: MSv2) -> Inputs:
        for index, path in enumerate(ms):
            with tables.table(str(path), readonly=False, ack=False) as table:
                table.addcols(tables.maketabdesc([tables.makearrcoldesc("MODEL_DATA", 0j, ndim=2)]))
                table.putcol("MODEL_DATA", np.full((table.nrows(), 1, 1), index + 1j))
        return Inputs(ms=ms)

    assert predict(ms=paths, reference=sibling, **kwargs(tmp_path)).success
    assert scans(sibling)[0] == 13
    for index, path in enumerate(paths):
        with tables.table(str(path), ack=False) as table:
            assert table.getcell("MODEL_DATA", 0)[0, 0] == index + 1j


def _worker_recipe(tmp_path, fail=False):
    import sys
    from shinobi import Cab
    from shinobi.steps.schema import ParamMeta

    class ToolInputs(BaseModel):
        script: str
        ms: list[MSv2]

    script = (
        "import sys;import casacore.tables as t;\nfor p in sys.argv[1:]:\n with t.table(p,readonly=False,ack=False) as m: m.putcol('SCAN_NUMBER',m.getcol('SCAN_NUMBER')+2)\n"
        + ("sys.exit(7)" if fail else "")
    )
    cab = Cab(
        name="joint-tool",
        command=f"{sys.executable} -c",
        inputs_model=ToolInputs,
        outputs_model=Inputs,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True, repeat_as_tokens=True)},
        dataset_accesses=[WRITE],
    )
    recipe = Recipe(name="joint-worker", inputs_model=Inputs, outputs_model=Inputs, cache=True, cache_dir=str(tmp_path / "cache"))
    recipe.add_step("first", cab, ms=InputRef(field="ms"), script=script)
    if not fail:
        recipe.add_step("second", cab, ms=OutputRef(step="first", field="ms"), script=script)
    return recipe


def test_qualified_detached_group_and_array_wire_roundtrip(tmp_path, monkeypatch):
    from tests.test_offload_dataset_lifecycle import _prepared
    from shinobi.offload.records import AttemptRecord, IndexedProducedState
    from shinobi.offload.worker import execute_step

    paths = roots(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    recipe = _worker_recipe(tmp_path)
    for cached in (False, True):
        workflow, plan, lease = _prepared(tmp_path, recipe, paths)
        try:
            for ref in recipe.steps:
                attempt = plan.attempt(ref.name)
                assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
                record = AttemptRecord.model_validate_json((workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").read_text())
                assert record.schema_version == 3
                if not cached:
                    leaf_payload = next(leaf for leaf in record.model_dump(mode="json")["dataset_lifecycle"]["leaves"] if leaf["step_path"] == f"{recipe.name}.{attempt.step_path}")
                    assert [mutation["element_index"] for mutation in leaf_payload["mutations"]] == [0, 1]
                assert all(isinstance(key, IndexedProducedState) for key in record.output_keys["ms"])
                assert [key.element_index for key in record.output_keys["ms"]] == [0, 1]
                restored = record.result(ref.step)
                assert [key.element_index for key in restored.provenance_key("ms")] == [0, 1]
                assert restored.cached == cached
            assert values(paths) == [6, 11]
        finally:
            lease.release()


@pytest.mark.parametrize("tool_failure", [False, True])
def test_detached_group_failed_tool_and_failed_immutable_publication(tmp_path, monkeypatch, tool_failure):
    from tests.test_offload_dataset_lifecycle import _prepared
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.worker import execute_step

    paths = roots(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    workflow, plan, lease = _prepared(tmp_path, _worker_recipe(tmp_path, fail=tool_failure), paths)
    original = AttemptRecord.write

    def refuse_success(self, directory):
        if self.committed:
            raise OSError("immutable publication failed")
        return original(self, directory)

    if not tool_failure:
        monkeypatch.setattr(AttemptRecord, "write", refuse_success)
    try:
        attempt = plan.attempt("first")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == (7 if tool_failure else 1)
        assert values(paths) == [2, 7]
        record = AttemptRecord.model_validate_json((workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").read_text())
        assert record.state == "failed"
        assert record.schema_version == 3
        assert record.dataset_lifecycle.leaves[0].outcome == "failed"
        assert all(get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is None for path in paths)
    finally:
        lease.release()


@pytest.mark.parametrize("stage", ["P_MARKED", "P_RESTORE", "S1", "S2", "S3", "S4"])
def test_fresh_process_death_and_exact_group_recovery(tmp_path, monkeypatch, stage):
    import os
    import subprocess
    import sys
    from pathlib import Path

    paths = roots(tmp_path)
    increment(ms=paths, **kwargs(tmp_path))
    script = """
import os, sys
from pathlib import Path
from shinobi.snapshots import faults
from tests.test_dataset_list_writes import increment
faults.hooks[sys.argv[1]] = lambda: os._exit(73)
increment(ms=[Path('a.ms'), Path('b.ms')], by=2, cache=True, cache_dir=str(Path('cache').resolve()))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run([sys.executable, "-c", script, stage], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 73, completed.stderr
    journal = get_journal(str(tmp_path / "cache"))
    assert all(journal.get(chain_id(path)).marker is not None for path in paths)
    before = values(paths)
    with pytest.raises(DatasetLifecycleUnavailableError, match="every exact participant"):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={paths[0]}, exact=True)
    assert values(paths) == before
    recover = """
from pathlib import Path
from shinobi.cache import get_cache_manifest
from shinobi.snapshots import reconcile
reconcile(str(Path('cache').resolve()), get_cache_manifest(str(Path('cache').resolve())), paths={Path('a.ms').resolve(), Path('b.ms').resolve()}, exact=True)
"""
    restored = subprocess.run([sys.executable, "-c", recover], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    assert values(paths) == ([4, 9] if stage in ("S3", "S4") else [3, 8])
    assert all(journal.get(chain_id(path)).marker is None for path in paths)


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4"])
def test_same_key_fresh_process_death_uses_frozen_bytes(tmp_path, stage):
    import os
    import subprocess
    import sys
    from pathlib import Path

    paths = roots(tmp_path)
    script_path = tmp_path / "writer.py"
    script_path.write_text("""
import os,sys
from pathlib import Path
from shinobi.snapshots import faults
from tests.test_dataset_list_writes import _same_key_cab
if len(sys.argv) > 1:
    faults.hooks[sys.argv[1]] = lambda: os._exit(73)
_same_key_cab()(ms=[Path('a.ms'),Path('b.ms')],cache=True,cache_dir=str(Path('cache').resolve()))
""")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["WRITE_VALUE"] = "3"
    initial = subprocess.run([sys.executable, str(script_path)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert initial.returncode == 0, initial.stderr
    assert values(paths) == [3, 8]
    (tmp_path / "ticket").unlink()
    env["WRITE_VALUE"] = "4"
    dead = subprocess.run([sys.executable, str(script_path), stage], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert dead.returncode == 73, dead.stderr
    journal = get_journal(str(tmp_path / "cache"))
    for path in paths:
        marker = journal.get(chain_id(path)).marker
        assert marker is not None
        assert marker.strict_rollback_state == state_name(marker.cache_key, "ms", paths.index(path))
        if stage != "S4":
            assert Path(marker.strict_rollback_source).is_dir()
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == ([4, 9] if stage in ("S3", "S4") else [3, 8])
    for path in paths:
        chain = journal.get(chain_id(path))
        assert chain.marker is None
        assert scans(journal.snapshot_dir(chain.head))[0] == scans(path)[0]


@pytest.mark.parametrize("returncode", [0, 7])
def test_yaml_binary_joint_writer_passthrough_and_nonzero(tmp_path, returncode):
    import sys
    from shinobi.loaders.yaml_cab import loads

    paths = roots(tmp_path)
    script = tmp_path / "tool.py"
    script.write_text(
        "import sys\nimport casacore.tables as t\nfor path in sys.argv[1:]:\n with t.table(path,readonly=False,ack=False) as m: m.putcol('SCAN_NUMBER',m.getcol('SCAN_NUMBER')+3)\nsys.exit("
        + str(returncode)
        + ")\n"
    )
    import yaml

    cab = loads(
        yaml.safe_dump(
            {
                "joint": {
                    "command": sys.executable,
                    "inputs": {
                        "script": {"dtype": "str", "default": str(script), "policies": {"positional_head": True}},
                        "ms": {"dtype": "List[MSv2]", "required": True, "policies": {"positional": True, "repeat": "list"}},
                    },
                    "outputs": {"ms": {"dtype": "List[MSv2]"}},
                    "dataset_accesses": [{"field": "ms", "mode": "write", "columns": {"write": ["SCAN_NUMBER"]}}],
                }
            }
        )
    )["joint"]
    result = cab(ms=paths, sandbox=True, **kwargs(tmp_path))
    assert result.returncode == returncode
    assert values(paths) == ([5, 10] if returncode == 0 else [2, 7])
    assert [path.resolve() for path in result.outputs.ms] == paths


def test_detached_dead_group_finalizer_retains_claim_until_all_recover(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import shinobi.snapshots as snapshots
    from tests.test_offload_dataset_lifecycle import _prepared, _register_jobs
    from shinobi.offload.worker import finalize_submission
    from shinobi.ownership import inspect_workspace

    paths = roots(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    workflow, plan, lease = _prepared(tmp_path, _worker_recipe(tmp_path), paths)
    _register_jobs(workflow, plan, "first", "second")
    attempt = plan.attempt("first")
    script = """
import os,sys
from pathlib import Path
from uuid import UUID
from shinobi.snapshots import faults
from shinobi.offload.worker import execute_step
faults.hooks['S2'] = lambda: os._exit(73)
execute_step(Path(sys.argv[1]), 'first', UUID(sys.argv[2]))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    dead = subprocess.run([sys.executable, "-c", script, str(workflow.submission_dir), str(attempt.attempt_id)], env=env, capture_output=True, text=True)
    assert dead.returncode == 73, dead.stderr
    assert values(paths) == [4, 9]
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {name: "FAILED" if name == "first" else "CANCELLED" for name in jobs})
    replace = snapshots._replace_tree_from_snapshot

    def fail_second(path, *args, **options):
        if path == paths[1]:
            raise OSError("restore blocked")
        return replace(path, *args, **options)

    monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", fail_second)
    try:
        assert not finalize_submission(workflow.submission_dir).complete
        assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is not None
        assert not (workflow.submission_dir / "finalization.json").exists()
        journal = get_journal(str(tmp_path / "cache"))
        assert all(journal.get(chain_id(path)).marker is not None for path in paths)
        monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", replace)
        assert not finalize_submission(workflow.submission_dir).complete
        assert values(paths) == [2, 7]
        assert all(journal.get(chain_id(path)).marker is None for path in paths)
        assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is None
    finally:
        lease.release()


@pytest.mark.parametrize("damage", ["missing-marker", "contradictory-marker", "missing-successor", "incomplete-oracle"])
def test_group_recovery_prevalidates_all_evidence_before_changes(tmp_path, damage):
    from pathlib import Path
    import shutil
    from shinobi.dataset_lifecycle import DatasetLifecycleStore

    paths = roots(tmp_path)

    def stop_cleanup():
        raise RuntimeError("durable oracle; cleanup pending")

    faults.hooks["S3"] = stop_cleanup
    with pytest.raises(RuntimeError):
        increment(ms=paths, **kwargs(tmp_path))
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    if damage in ("missing-marker", "contradictory-marker"):

        def corrupt(chain):
            if damage == "missing-marker":
                chain.marker = None
            else:
                chain.marker.run_id = "other-run"
            return chain

        journal.update_chain(chain_id(paths[1]), corrupt)
    elif damage == "missing-successor":
        chain = journal.get(chain_id(paths[1]))
        shutil.rmtree(journal.snapshot_dir(chain.head))
    else:
        marker = journal.get(chain_id(paths[0])).marker
        store = DatasetLifecycleStore(Path(marker.success_record))
        record = store.attempt()
        leaf = record.leaves[0].model_copy(update={"mutations": record.leaves[0].mutations[:1]})
        store.replace(record.model_copy(update={"leaves": (leaf,)}))
    before = values(paths)
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == before
    assert journal.get(chain_id(paths[0])).marker is not None


def test_eviction_and_invalidation_preserve_frozen_group_sources(tmp_path):
    from shinobi.snapshots import _protected_names, evict, invalidate

    paths = roots(tmp_path)
    increment(ms=paths, **kwargs(tmp_path))
    journal = get_journal(str(tmp_path / "cache"))
    original = {journal.get(chain_id(path)).head for path in paths}

    def before_oracle():
        chains = journal.all_chains()
        assert original <= _protected_names(chains)
        evict(str(tmp_path / "cache"), 10**12)
        assert all(journal.snapshot_dir(name).exists() for name in original)
        with pytest.raises(DatasetLifecycleUnavailableError, match="in-flight strict mutation group"):
            invalidate(str(tmp_path / "cache"), "increment", get_cache_manifest(str(tmp_path / "cache")))
        raise RuntimeError("oracle not published")

    faults.hooks["S2"] = before_oracle
    with pytest.raises(RuntimeError):
        increment(ms=paths, by=2, **kwargs(tmp_path))
    faults.hooks.clear()
    assert values(paths) == [3, 8]


def test_later_successor_snapshot_failure_rolls_back_schema_and_cells(tmp_path, monkeypatch):
    import shinobi.snapshots as snapshots
    import casacore.tables as tables

    paths = roots(tmp_path)
    clone = snapshots.clone_tree

    def fail_second_successor(source, dest, **options):
        if source == paths[1] and "__list." in dest.name:
            raise OSError("later successor snapshot failed")
        return clone(source, dest, **options)

    monkeypatch.setattr(snapshots, "clone_tree", fail_second_successor)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",), create=("MODEL_DATA",)), allow_schema_change=True)])
    def predict(ms: list[MSv2]) -> None:
        for path in ms:
            set_scans(path, 99)
            with tables.table(str(path), readonly=False, ack=False) as table:
                table.addcols(tables.maketabdesc([tables.makearrcoldesc("MODEL_DATA", 0j, ndim=2)]))

    with pytest.raises(OSError, match="later successor"):
        predict(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [2, 7]
    for path in paths:
        with tables.table(str(path), ack=False) as table:
            assert "MODEL_DATA" not in table.colnames()
    assert all(get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is None for path in paths)


def test_optional_omitted_write_list_and_list_plus_scalar_group(tmp_path):
    paths = roots(tmp_path)

    @pystep(dataset_accesses=[WRITE, DatasetAccess(field="other", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))])
    def combined(other: MSv2, ms: list[MSv2] | None = None) -> None:
        set_scans(other, 44)
        if ms is not None:
            for path in ms:
                set_scans(path, 55)
            raise RuntimeError("three-root joint write")

    other = make_ms(tmp_path / "other.ms", scan=13)
    with pytest.raises(RuntimeError):
        combined(ms=paths, other=other, **kwargs(tmp_path))
    assert values(paths + [other]) == [2, 7, 13]
    assert len(attempts(tmp_path)[-1].leaves[0].mutations) == 3
    assert combined(other=other, **kwargs(tmp_path)).success
    assert scans(other)[0] == 44


def test_newly_wired_predecessor_survives_eviction_after_group_markers(tmp_path):
    from shinobi.snapshots import evict
    from shinobi.steps.dispatch import _dispatch

    paths = roots(tmp_path)
    first = increment(ms=paths, by=1, **kwargs(tmp_path))
    increment(ms=paths, by=2, **kwargs(tmp_path))
    increment(ms=paths, by=3, **kwargs(tmp_path))
    assert values(paths) == [5, 10]
    states = {state_name(first.cache_key, "ms", index) for index in range(2)}
    journal = get_journal(str(tmp_path / "cache"))
    # These older sibling states are neither heads nor previously consumed.
    for path in paths:
        chain = journal.get(chain_id(path))
        assert chain.head not in states
        assert not states.intersection(chain.consumed.values())

    def evict_after_markers():
        evict(str(tmp_path / "cache"), 10**12)
        assert all(journal.snapshot_dir(name).is_dir() for name in states)

    faults.hooks["P_MARKED"] = evict_after_markers
    result = _dispatch(increment.step, increment.func, ms=paths, by=10, _input_keys={"ms": first.provenance_key("ms")}, _wired_fields={"ms"}, **kwargs(tmp_path))
    assert result.success
    assert values(paths) == [13, 18]


@pytest.mark.parametrize("gathered", [False, True])
def test_unknown_producer_states_refuse_list_writer_before_launch(tmp_path, gathered):
    from shinobi.cache import resolve_input_keys
    from shinobi.results import StepResult
    from shinobi.steps.dispatch import _aggregate_scatter_results, _dispatch
    from shinobi.steps.schema import Scope, StepRef

    paths = roots(tmp_path)

    class Scalar(BaseModel):
        ms: MSv2

    ran = []

    @pystep(dataset_accesses=[WRITE])
    def writer(ms: list[MSv2]) -> None:
        ran.append(True)
        for path in ms:
            set_scans(path, 99)

    if gathered:
        producer = Scope(name="producer", inputs_model=Empty, outputs_model=Scalar)
        slices = [StepResult(name="producer", inputs=Empty(), outputs=Scalar(ms=path), returncode=0, cache_key=str(index) * 64) for index, path in enumerate(paths)]
        aggregate = _aggregate_scatter_results(producer, [], {}, slices)
        ref = StepRef(name="writer", step=writer.step, wiring={"ms": OutputRef(step="producer", field="ms")})
        results = {"producer": aggregate}
    else:
        ref = StepRef(name="writer", step=writer.step, wiring={"ms": [OutputRef(step="left", field="ms"), OutputRef(step="right", field="ms")]})
        results = {name: StepResult(name=name, inputs=Empty(), outputs=Scalar(ms=path), returncode=0) for name, path in zip(("left", "right"), paths)}
    keys = resolve_input_keys(ref, {}, results)
    with pytest.raises(DatasetLifecycleUnavailableError, match="producer"):
        _dispatch(writer.step, writer.func, ms=paths, _input_keys=keys, _wired_fields=set(ref.wiring), **kwargs(tmp_path))
    assert ran == []
    assert values(paths) == [2, 7]


@pytest.mark.parametrize("scalar", [False, True], ids=["singleton-list", "scalar"])
@pytest.mark.parametrize("stage,fail_restore", [("S2", False), ("S2", True), ("S3", False)])
def test_single_target_same_key_recovery_cleans_only_settled_backup(tmp_path, monkeypatch, scalar, stage, fail_restore):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import shinobi.snapshots as snapshots

    path = make_ms(tmp_path / "a.ms", scan=2)
    script = tmp_path / "writer.py"
    script.write_text(f"""
import os,sys
from pathlib import Path
from shinobi.snapshots import faults
from tests.test_dataset_list_writes import _same_key_cab
if len(sys.argv)>1: faults.hooks[sys.argv[1]]=lambda:os._exit(73)
_same_key_cab(scalar={scalar!r})(ms=Path('a.ms') if {scalar!r} else [Path('a.ms')],cache=True,cache_dir=str(Path('cache').resolve()))
""")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["WRITE_VALUE"] = "3"
    first = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert scans(path)[0] == 3
    (tmp_path / "ticket").unlink()
    env["WRITE_VALUE"] = "4"
    dead = subprocess.run([sys.executable, str(script), stage], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert dead.returncode == 73, dead.stderr
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(path)).marker
    backup = Path(marker.strict_rollback_source)
    assert backup.is_dir()
    unrelated = backup.parent / "unrelated"
    unrelated.mkdir()
    original = snapshots._replace_tree_from_snapshot
    if fail_restore:
        monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("restore unavailable")))
        with pytest.raises(OSError, match="restore unavailable"):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
        assert backup.is_dir() and journal.get(chain_id(path)).marker is not None
        monkeypatch.setattr(snapshots, "_replace_tree_from_snapshot", original)
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
    assert scans(path)[0] == (4 if stage == "S3" else 3)
    assert scans(journal.snapshot_dir(journal.get(chain_id(path)).head))[0] == scans(path)[0]
    assert journal.get(chain_id(path)).marker is None
    assert not backup.exists()
    assert unrelated.is_dir()


@pytest.mark.parametrize("fault", [OSError, ValueError], ids=["unavailable", "malformed"])
def test_local_published_group_unreadable_oracle_retains_all_fences_and_backups(tmp_path, monkeypatch, fault):
    from pathlib import Path
    from shinobi.dataset_lifecycle import DatasetLifecycleStore, StrictLeaf

    paths = roots(tmp_path)
    cab = _same_key_cab()
    monkeypatch.setenv("WRITE_VALUE", "3")
    assert cab(ms=paths, **kwargs(tmp_path)).success
    (tmp_path / "ticket").unlink()
    monkeypatch.setenv("WRITE_VALUE", "4")
    original_commit = StrictLeaf.commit
    original_attempt = DatasetLifecycleStore.attempt

    def publish_then_raise(leaf):
        original_commit(leaf)
        monkeypatch.setattr(DatasetLifecycleStore, "attempt", lambda *_: (_ for _ in ()).throw(fault("temporarily unreadable oracle")))
        raise RuntimeError("publication callback interrupted")

    monkeypatch.setattr(StrictLeaf, "commit", publish_then_raise)
    with pytest.raises(RuntimeError, match="publication callback interrupted"):
        cab(ms=paths, **kwargs(tmp_path))
    assert values(paths) == [4, 9]
    journal = get_journal(str(tmp_path / "cache"))
    backups = [Path(journal.get(chain_id(path)).marker.strict_rollback_source) for path in paths]
    assert all(backup.is_dir() for backup in backups)
    with pytest.raises(DatasetLifecycleUnavailableError, match="unreadable or malformed"):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == [4, 9]
    assert all(journal.get(chain_id(path)).marker is not None for path in paths)
    assert all(backup.is_dir() for backup in backups)
    monkeypatch.setattr(DatasetLifecycleStore, "attempt", original_attempt)
    monkeypatch.setattr(StrictLeaf, "commit", original_commit)
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == [4, 9]
    assert all(journal.get(chain_id(path)).marker is None for path in paths)
    assert all(not backup.exists() for backup in backups)
    assert attempts(tmp_path)[-1].leaves[0].outcome == "committed"


def test_detached_published_group_unreadable_oracle_preserves_exact_record_and_claim(tmp_path, monkeypatch):
    from pathlib import Path
    from tests.test_offload_dataset_lifecycle import _prepared
    from shinobi.offload.records import AttemptRecord
    from shinobi.offload.worker import execute_step
    from shinobi.ownership import inspect_workspace

    paths = roots(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    workflow, plan, lease = _prepared(tmp_path, _worker_recipe(tmp_path), paths)
    original_write = AttemptRecord.write
    original_read = Path.read_text
    blocked = set()

    def unavailable(path, *args, **kwargs):
        if path in blocked:
            raise OSError("immutable oracle temporarily unreadable")
        return original_read(path, *args, **kwargs)

    def publish_then_raise(record, directory):
        path = original_write(record, directory)
        if record.committed:
            blocked.add(path)
            raise RuntimeError("immutable publication callback interrupted")
        return path

    monkeypatch.setattr(AttemptRecord, "write", publish_then_raise)
    monkeypatch.setattr(Path, "read_text", unavailable)
    try:
        attempt = plan.attempt("first")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
        assert values(paths) == [4, 9]
        journal = get_journal(str(tmp_path / "cache"))
        assert all(journal.get(chain_id(path)).marker is not None for path in paths)
        assert all(Path(journal.get(chain_id(path)).marker.strict_rollback_source).is_dir() for path in paths)
        assert inspect_workspace(Path(plan.ownership_workspace), str(plan.workflow_id)) is not None
        with pytest.raises(DatasetLifecycleUnavailableError, match="unreadable or malformed"):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
        assert values(paths) == [4, 9]
        blocked.clear()
        final = workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json"
        published = final.read_bytes()
        assert AttemptRecord.model_validate_json(published).committed
        # An unknown discriminator must not be reinterpreted as worker JSON,
        # even when that JSON is otherwise a valid exact committed result.
        for path in paths:
            journal.update_chain(chain_id(path), lambda chain: _set_oracle_kind(chain, "unknown-kind"))
        with pytest.raises(DatasetLifecycleUnavailableError, match="oracle kind is unknown"):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
        assert values(paths) == [4, 9]
        assert all(journal.get(chain_id(path)).marker is not None for path in paths)
        for path in paths:
            journal.update_chain(chain_id(path), lambda chain: _set_oracle_kind(chain, None))
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
        assert values(paths) == [4, 9]
        assert all(journal.get(chain_id(path)).marker is None for path in paths)
        assert final.read_bytes() == published
    finally:
        lease.release()


def _set_oracle_kind(chain, kind):
    chain.marker.success_kind = kind
    return chain


@pytest.mark.parametrize("scalar", [False, True], ids=["group", "scalar"])
@pytest.mark.parametrize(
    "damage", ["missing-json", "stale-json", "corrupt-json", "absent-logical-record", "malformed-logical-record", "malformed-log-root", "corrupt-log", "unreadable-log"]
)
def test_local_oracle_reads_authoritative_log_before_exact_recovery(tmp_path, monkeypatch, scalar, damage):
    import os
    from pathlib import Path
    from shinobi.dataset_lifecycle import DatasetLifecycleStore, mutation_committed

    paths = roots(tmp_path)[:1] if scalar else roots(tmp_path)
    monkeypatch.setenv("WRITE_VALUE", "3")
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("committed; cleanup interrupted"))
    with pytest.raises(RuntimeError, match="cleanup interrupted"):
        _same_key_cab(scalar=scalar)(ms=paths[0] if scalar else paths, **kwargs(tmp_path))
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(paths[0])).marker
    oracle = Path(marker.success_record)
    store = DatasetLifecycleStore(oracle)
    authoritative_bytes = store.lock_path.read_bytes()
    published = values(paths)
    assert published == ([3] if scalar else [3, 8])
    if damage == "missing-json":
        oracle.unlink()
    elif damage == "stale-json":
        oracle.write_text('{"record":null}')
    elif damage == "corrupt-json":
        oracle.write_text("{broken compatibility view")
    elif damage == "absent-logical-record":
        store.update(lambda data: data.pop("record", None))
    elif damage == "malformed-logical-record":
        store.update(lambda data: data.update({"record": None}))
    elif damage == "malformed-log-root":
        import hashlib
        import json

        payload = json.dumps({"version": 1, "snapshot": []}).encode()
        store.lock_path.write_bytes(hashlib.sha256(payload).hexdigest().encode() + b" " + payload + b"\n")
    elif damage == "corrupt-log":
        store.lock_path.write_bytes(b"invalid authoritative transaction\n")
    else:
        original_pread = os.pread
        authority = store.lock_path.stat()

        def unreadable(fd, *args):
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) == (authority.st_dev, authority.st_ino):
                raise OSError("authoritative oracle read unavailable")
            return original_pread(fd, *args)

        monkeypatch.setattr(os, "pread", unreadable)
    identity = {"attempt_id": marker.run_id, "step_path": marker.success_step_path or marker.step_path, "cache_key": marker.cache_key, "participants": marker.strict_group}
    unknown = damage in ("malformed-logical-record", "malformed-log-root", "corrupt-log", "unreadable-log")
    if unknown:
        assert not mutation_committed(oracle, **identity)
        with pytest.raises(DatasetLifecycleUnavailableError, match="unreadable or malformed"):
            mutation_committed(oracle, strict=True, **identity)
        with pytest.raises(DatasetLifecycleUnavailableError, match="unreadable or malformed"):
            reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
        assert values(paths) == published
        assert all(journal.get(chain_id(path)).marker is not None for path in paths)
        assert all(Path(journal.get(chain_id(path)).marker.strict_rollback_source).is_dir() for path in paths)
        if damage == "unreadable-log":
            monkeypatch.setattr(os, "pread", original_pread)
        else:
            store.lock_path.write_bytes(authoritative_bytes)
    else:
        expected_commit = damage != "absent-logical-record"
        assert mutation_committed(oracle, **identity) is expected_commit
        assert mutation_committed(oracle, strict=True, **identity) is expected_commit
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert values(paths) == (([2] if scalar else [2, 7]) if damage == "absent-logical-record" else published)
    assert all(journal.get(chain_id(path)).marker is None for path in paths)
