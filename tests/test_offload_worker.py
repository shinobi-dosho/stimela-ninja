from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.records import AttemptRecord
from shinobi.offload.slurm import (
    OffloadCompileError,
    WorkerSubmissionError,
    compile_slurm,
    prepare_worker_slurm,
    submit_worker_slurm,
)
from shinobi.offload.worker import (
    ExecutionPlan,
    SubmittedFinalizer,
    SubmittedJob,
    WorkerHandleRecord,
    execute_step,
    finalize_submission,
)
from shinobi.provenance import RunManifest
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef


class RootIn(BaseModel):
    first: str = "first.txt"


class WriteIn(BaseModel):
    script: str
    out: str


class CopyIn(BaseModel):
    script: str
    src: Path
    out: str


class PathOut(BaseModel):
    out: Path


def _cab(name: str, model: type[BaseModel]) -> Cab:
    return Cab(
        name=name,
        command=f"{sys.executable} -c",
        inputs_model=model,
        outputs_model=PathOut,
        field_meta={field: ParamMeta(positional=True) for field in model.model_fields},
    )


def _recipe() -> Recipe:
    write = _cab("write", WriteIn)
    copy = _cab("copy", CopyIn)
    return Recipe(
        name="worker-pipe",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[
            StepRef(
                name="write",
                step=write,
                params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('frozen');Path('junk.log').write_text('junk')"},
                wiring={"out": InputRef(field="first")},
            ),
            StepRef(
                name="copy",
                step=copy,
                params={
                    "script": "from pathlib import Path;import sys;Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text()+' worker')",
                    "out": "second.txt",
                },
                wiring={"src": OutputRef(step="write", field="out")},
            ),
        ],
        output_wiring={"out": OutputRef(step="copy", field="out")},
    )


def _prepared(tmp_path: Path):
    bundle = freeze_recipe(_recipe(), {}, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    staged = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    return workflow, staged, plan


def _prepared_cached(tmp_path: Path, recipe: Recipe | None = None, config: AppConfig | None = None):
    config = config or AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    bundle = freeze_recipe(recipe or _recipe(), {}, config=config, workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    staged = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    return workflow, staged, plan


def _execute_all(workflow, plan):
    records = []
    for attempt in plan.attempts:
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
        records.append(
            AttemptRecord.read(
                _final_record(workflow, attempt),
                workflow_id=plan.workflow_id,
                attempt_id=attempt.attempt_id,
                step_path=attempt.step_path,
                bundle_digest=bundle.digest,
            )
        )
    return records


def _final_record(workflow, attempt) -> Path:
    return workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json"


def test_worker_executes_wiring_in_shared_sandboxes_and_finalizes(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    for attempt in plan.attempts:
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    assert (tmp_path / "first.txt").read_text() == "frozen"
    assert (tmp_path / "second.txt").read_text() == "frozen worker"
    assert not (tmp_path / "junk.log").exists()
    assert list((workflow.submission_dir / "sandboxes").iterdir()) == []

    for index, attempt in enumerate(plan.attempts):
        write_new(
            workflow.submission_dir / "jobs" / f"{index:04d}.json",
            SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id=str(100 + index)),
        )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "COMPLETED"))
    finalized = finalize_submission(workflow.submission_dir)
    assert finalized.complete
    assert [step.step_path for step in finalized.steps] == ["write", "copy"]
    assert (workflow.submission_dir / "manifest.json").is_file()
    assert finalize_submission(workflow.submission_dir) == finalized


def test_second_detached_run_uses_runtime_cache_and_committed_upstream_keys(tmp_path):
    first, _bundle, first_plan = _prepared_cached(tmp_path)
    assert [record.state for record in _execute_all(first, first_plan)] == ["succeeded", "succeeded"]

    second, _bundle, second_plan = _prepared_cached(tmp_path)
    records = _execute_all(second, second_plan)
    assert [record.state for record in records] == ["cached", "cached"]
    assert all(record.cache_key for record in records)
    assert records[1].output_keys["out"].cache_key == records[1].cache_key


def test_deleted_declared_product_forces_worker_rerun(tmp_path):
    first, _bundle, first_plan = _prepared_cached(tmp_path)
    _execute_all(first, first_plan)
    (tmp_path / "second.txt").unlink()

    second, _bundle, second_plan = _prepared_cached(tmp_path)
    records = _execute_all(second, second_plan)
    assert [record.state for record in records] == ["cached", "succeeded"]
    assert (tmp_path / "second.txt").read_text() == "frozen worker"


def test_changed_upstream_worker_identity_invalidates_its_descendant(tmp_path):
    first, _bundle, first_plan = _prepared_cached(tmp_path)
    _execute_all(first, first_plan)

    changed = _recipe().model_copy(deep=True)
    changed.steps[0].params["script"] = "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('changed')"
    second, _bundle, second_plan = _prepared_cached(tmp_path, changed)
    records = _execute_all(second, second_plan)
    assert [record.state for record in records] == ["succeeded", "succeeded"]
    assert (tmp_path / "second.txt").read_text() == "changed worker"


def test_changed_worker_branch_keeps_unrelated_cache_hit(tmp_path):
    recipe = _recipe().model_copy(deep=True)
    recipe.steps.append(
        StepRef(
            name="unrelated",
            step=_cab("unrelated", WriteIn),
            params={
                "script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('unrelated')",
                "out": "unrelated.txt",
            },
        )
    )
    first, _bundle, first_plan = _prepared_cached(tmp_path, recipe)
    assert [record.state for record in _execute_all(first, first_plan)] == ["succeeded", "succeeded", "succeeded"]

    changed = recipe.model_copy(deep=True)
    changed.steps[0].params["script"] = "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('changed')"
    second, _bundle, second_plan = _prepared_cached(tmp_path, changed)
    assert [record.state for record in _execute_all(second, second_plan)] == ["succeeded", "succeeded", "cached"]


def test_recipe_cache_setting_is_not_dropped_by_worker_compilation(tmp_path):
    configured = _recipe().model_copy(update={"cache": True, "cache_dir": str(tmp_path / "scope-cache")})
    config = AppConfig()
    first, _bundle, first_plan = _prepared_cached(tmp_path, configured, config)
    _execute_all(first, first_plan)
    second, _bundle, second_plan = _prepared_cached(tmp_path, configured, config)
    assert [record.state for record in _execute_all(second, second_plan)] == ["cached", "cached"]


def test_worker_mutator_remains_uncached_until_workspace_ownership_lands(tmp_path):
    recipe = _recipe().model_copy(deep=True)
    recipe.steps[1].step.input_mutability["src"] = Mutability.MUTABLE
    first, _bundle, first_plan = _prepared_cached(tmp_path, recipe)
    assert [record.state for record in _execute_all(first, first_plan)] == ["succeeded", "succeeded"]

    second, _bundle, second_plan = _prepared_cached(tmp_path, recipe)
    records = _execute_all(second, second_plan)
    assert [record.state for record in records] == ["cached", "succeeded"]
    assert records[1].cache_key is None


def test_missing_worker_result_is_unknown_not_success(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id="42"),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"write": "COMPLETED"})
    finalized = finalize_submission(workflow.submission_dir)
    assert not finalized.complete
    assert finalized.steps[0].state == "unknown"
    assert finalized.steps[0].record is None
    assert not (workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "unknown.json").exists()


def test_staged_worker_source_is_checked_before_execution(tmp_path):
    workflow, bundle, plan = _prepared(tmp_path)
    source = workflow.submission_dir / "worker-src" / "shinobi" / "__init__.py"
    source.write_text(source.read_text() + "\n# changed\n")
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    assert record.state == "failed"
    assert "no longer matches" in record.error


def test_submission_records_each_job_and_detached_finalizer(tmp_path, monkeypatch):
    workflow, _bundle, plan = _prepared(tmp_path)
    calls = []
    ids = iter(("101\n", "102\n", "199\n"))

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=next(ids), stderr="")

    monkeypatch.setattr("shinobi.offload.slurm.subprocess.run", run)
    handle = submit_worker_slurm(workflow)
    assert handle.jobs == {"write": "101", "copy": "102"}
    assert handle.finalizer_job == "199"
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101:102" in calls[2]
    records = sorted((workflow.submission_dir / "jobs").glob("*.json"))
    assert len(records) == len(plan.attempts)
    finalizer = SubmittedFinalizer.model_validate_json((workflow.submission_dir / "finalizer-job.json").read_text())
    assert finalizer.workflow_id == plan.workflow_id
    assert finalizer.bundle_digest == plan.bundle_digest
    assert finalizer.job_id == "199"
    handle_record = WorkerHandleRecord.model_validate_json((workflow.submission_dir / "handle.json").read_text())
    assert handle_record.jobs == handle.jobs
    assert handle_record.finalizer == "199"


def test_partial_submission_keeps_recoverable_handle(tmp_path, monkeypatch):
    workflow, _bundle, _plan = _prepared(tmp_path)
    calls = 0

    def run(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(argv, 0, stdout="101\n", stderr="")
        if calls == 2:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="queue rejected")
        return subprocess.CompletedProcess(argv, 0, stdout="199\n", stderr="")

    monkeypatch.setattr("shinobi.offload.slurm.subprocess.run", run)
    with pytest.raises(WorkerSubmissionError, match="queue rejected") as caught:
        submit_worker_slurm(workflow)
    assert caught.value.handle.jobs == {"write": "101"}
    assert caught.value.handle.finalizer_job == "199"
    persisted = WorkerHandleRecord.model_validate_json((workflow.submission_dir / "handle.json").read_text())
    assert persisted.jobs == {"write": "101"}
    assert persisted.finalizer == "199"


def test_worker_compile_accepts_checked_per_step_scheduler_placement(tmp_path):
    bundle = freeze_recipe(_recipe(), {}, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(
        bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable), step_sbatch_opts={"write": {"nodelist": "k1"}, "copy": {"nodelist": "n1"}}
    )
    assert "#SBATCH --nodelist=k1" in workflow.jobs[0].script
    assert "#SBATCH --nodelist=n1" in workflow.jobs[1].script


def test_worker_and_legacy_compilers_preserve_wired_mutation_order(tmp_path):
    class MSIn(BaseModel):
        ms: Path

    class MSOut(BaseModel):
        ms: Path | None = None

    class Empty(BaseModel):
        pass

    produce = Cab(name="produce", command="true", inputs_model=MSIn, outputs_model=MSOut)
    mutate = Cab(name="mutate", command="true", inputs_model=MSIn, outputs_model=MSOut, input_mutability={"ms": Mutability.MUTABLE})
    read = Cab(name="read", command="true", inputs_model=MSIn, outputs_model=Empty)
    recipe = Recipe(
        name="mutation-worker",
        inputs_model=MSIn,
        outputs_model=Empty,
        steps=[
            StepRef(name="produce", step=produce, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="mutate", step=mutate, wiring={"ms": OutputRef(step="produce", field="ms")}),
            StepRef(name="read", step=read, wiring={"ms": OutputRef(step="produce", field="ms")}),
        ],
    )
    legacy = compile_slurm(recipe, {"ms": "input.ms"}, workdir=str(tmp_path), container_runtime=None)
    worker = prepare_worker_slurm(
        freeze_recipe(recipe, {"ms": "input.ms"}, config=AppConfig(), workspace=tmp_path),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    assert (
        {job.name: job.depends_on for job in worker.jobs}
        == {job.name: job.depends_on for job in legacy.jobs}
        == {"produce": [], "mutate": ["produce"], "read": ["produce", "mutate"]}
    )


def test_frozen_venv_pystep_runs_out_of_process(make_venv, tmp_path):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    package = make_venv(package=("venvonlypkg", "1.0.0", "MAGIC = 4242\n"))

    class NumberIn(BaseModel):
        n: int

    ref = pystep(venv=str(package), backend="venv")(funcs.use_venv_only_pkg)
    recipe = Recipe(
        name="venv-worker",
        inputs_model=NumberIn,
        outputs_model=funcs.MagicOut,
        steps=[ref.model_copy(update={"wiring": {"n": InputRef(field="n")}})],
        output_wiring={"value": OutputRef(step=ref.name, field="value")},
    )
    bundle = freeze_recipe(recipe, {"n": 8}, config=AppConfig(), workspace=tmp_path, code_roots=(Path.cwd(),))
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workflow.submission_dir / "worker-src")
    proc = subprocess.run(
        [sys.executable, "-m", "shinobi.offload.worker", "run", "--submission", str(workflow.submission_dir), "--step", attempt.step_path, "--attempt-id", str(attempt.attempt_id)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    staged = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=staged.digest,
    )
    assert record.result(staged.steps[0].scope.restore()).outputs.value == 4250
    assert record.observation.venv == str(package)
    assert record.observation.venv_digest is not None


def test_unchanged_venv_pystep_hits_cache_and_stays_unpinned(make_venv, tmp_path, monkeypatch):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    venv = make_venv(package=("venvonlypkg", "1.0.0", "MAGIC = 4242\n"))

    class NumberIn(BaseModel):
        n: int

    ref = pystep(venv=str(venv), backend="venv")(funcs.use_venv_only_pkg)
    recipe = Recipe(
        name="cached-venv-worker",
        inputs_model=NumberIn,
        outputs_model=funcs.MagicOut,
        steps=[ref.model_copy(update={"wiring": {"n": InputRef(field="n")}})],
        output_wiring={"value": OutputRef(step=ref.name, field="value")},
    )
    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})

    def prepare():
        return prepare_worker_slurm(
            freeze_recipe(recipe, {"n": 8}, config=config, workspace=tmp_path, code_roots=(Path.cwd(),)),
            submission_root=tmp_path / "runs",
            worker_python=Path(sys.executable),
        )

    first = prepare()
    first_plan = ExecutionPlan.model_validate_json((first.submission_dir / "execution.json").read_text())
    assert _execute_all(first, first_plan)[0].state == "succeeded"

    second = prepare()
    second_bundle = RecipeBundle.read(second.submission_dir / "bundle.json")
    second_plan = ExecutionPlan.model_validate_json((second.submission_dir / "execution.json").read_text())
    cached = _execute_all(second, second_plan)[0]
    assert cached.state == "cached"
    assert cached.code_digest == second_bundle.steps[0].code.digest
    assert cached.observation.venv == str(venv)
    assert cached.observation.venv_digest == second_bundle.steps[0].tool_venv_digest

    (second.submission_dir / "jobs").mkdir()
    attempt = second_plan.attempts[0]
    write_new(
        second.submission_dir / "jobs" / "0000.json",
        SubmittedJob(
            workflow_id=second_plan.workflow_id,
            bundle_digest=second_bundle.digest,
            step_path=attempt.step_path,
            attempt_id=attempt.attempt_id,
            job_id="88",
        ),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {attempt.step_path: "COMPLETED"})
    assert finalize_submission(second.submission_dir).complete
    manifest = RunManifest.model_validate_json((second.submission_dir / "manifest.json").read_text())
    assert manifest.pinned is False


def test_changed_bundled_pystep_and_helper_invalidate_only_their_branch(make_venv, tmp_path):
    from shinobi import pystep

    class EmptyInputs(BaseModel):
        pass

    venv = make_venv()
    (tmp_path / "helper_a.py").write_text("def value():\n    return 1\n")
    source_a = tmp_path / "branch_a.py"
    source_b = tmp_path / "branch_b.py"
    source_a.write_text("def run():\n    import helper_a\n    helper_a.value()\n")
    source_b.write_text("def run():\n    return None\n")

    def load(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run

    def make_recipe() -> Recipe:
        a = pystep(name="a", venv=str(venv), backend="venv")(load("branch_a", source_a))
        b = pystep(name="b", venv=str(venv), backend="venv")(load("branch_b", source_b))
        return Recipe(name="code-branches", inputs_model=EmptyInputs, outputs_model=EmptyInputs, steps=[a, b])

    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})

    def prepare():
        return prepare_worker_slurm(
            freeze_recipe(make_recipe(), {}, config=config, workspace=tmp_path, code_roots=(tmp_path,)),
            submission_root=tmp_path / "runs",
            worker_python=Path(sys.executable),
        )

    first = prepare()
    first_plan = ExecutionPlan.model_validate_json((first.submission_dir / "execution.json").read_text())
    assert [record.state for record in _execute_all(first, first_plan)] == ["succeeded", "succeeded"]

    source_a.write_text("def run():\n    import helper_a\n    helper_a.value() + 1\n")
    (tmp_path / "helper_a.py").write_text("def value():\n    return 2\n")
    second = prepare()
    second_plan = ExecutionPlan.model_validate_json((second.submission_dir / "execution.json").read_text())
    assert [record.state for record in _execute_all(second, second_plan)] == ["succeeded", "cached"]


def test_frozen_venv_pystep_imports_bundled_helper(make_venv, tmp_path):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    venv = make_venv()

    class NumberIn(BaseModel):
        n: int

    ref = pystep(venv=str(venv), backend="venv")(funcs.use_bundled_helper)
    recipe = Recipe(
        name="venv-helper",
        inputs_model=NumberIn,
        outputs_model=funcs.MagicOut,
        steps=[ref.model_copy(update={"wiring": {"n": InputRef(field="n")}})],
        output_wiring={"value": OutputRef(step=ref.name, field="value")},
    )
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {"n": 8}, config=AppConfig(), workspace=tmp_path, code_roots=(Path.cwd(),)),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
    )
    assert record.result(bundle.steps[0].scope.restore()).outputs.value == 9


def test_frozen_image_pystep_imports_helper_and_records_exact_pins(tmp_path, monkeypatch):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    digest = "sha256:" + "a" * 64

    class NumberIn(BaseModel):
        n: int

    ref = pystep(image=f"repo/tool@{digest}", backend="apptainer")(funcs.use_bundled_helper)
    recipe = Recipe(
        name="image-helper",
        inputs_model=NumberIn,
        outputs_model=funcs.MagicOut,
        steps=[ref.model_copy(update={"wiring": {"n": InputRef(field="n")}})],
        output_wiring={"value": OutputRef(step=ref.name, field="value")},
    )

    launches = []

    def fake_container_argv(runtime, scope, argv, inputs, workdir, **kwargs):
        launches.append(scope.image)
        assert runtime == "apptainer"
        assert scope.image.endswith(digest)
        assert any(part.endswith("/code/0") for part in kwargs["extra_dirs"])
        return [sys.executable, argv[-1]], digest

    monkeypatch.setattr("shinobi.backends.container.build_container_argv", fake_container_argv)
    monkeypatch.setattr("shinobi.backends.container.container_stopper", lambda *args: None)
    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {"n": 8}, config=config, workspace=tmp_path, code_roots=(Path.cwd(),)),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
    )
    assert bundle.steps[0].image_digest == digest
    assert record.code_digest == bundle.steps[0].code.digest
    assert record.observation.image_digest == digest
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id="88"),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {attempt.step_path: "COMPLETED"})
    assert finalize_submission(workflow.submission_dir).complete
    manifest = RunManifest.model_validate_json((workflow.submission_dir / "manifest.json").read_text())
    assert manifest.root.steps[0].image_digest == digest
    assert manifest.root.steps[0].code_digest == bundle.steps[0].code.digest

    second = prepare_worker_slurm(
        freeze_recipe(recipe, {"n": 8}, config=config, workspace=tmp_path, code_roots=(Path.cwd(),)),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    second_plan = ExecutionPlan.model_validate_json((second.submission_dir / "execution.json").read_text())
    cached = _execute_all(second, second_plan)[0]
    assert cached.state == "cached"
    assert cached.observation.image_digest == digest
    assert len(launches) == 1
    assert launches[0].endswith(digest)


def test_failed_worker_keeps_shared_sandbox_and_publishes_no_success(tmp_path):
    bad = _cab("bad", WriteIn)
    recipe = Recipe(
        name="bad-worker",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[
            StepRef(
                name="bad",
                step=bad,
                params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('partial');raise SystemExit(7)"},
                wiring={"out": InputRef(field="first")},
            )
        ],
        output_wiring={"out": OutputRef(step="bad", field="out")},
    )
    workflow = prepare_worker_slurm(freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path), submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 7
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    assert record.state == "failed"
    assert not record.committed
    assert record.sandbox is not None and Path(record.sandbox).is_dir()
    assert not (tmp_path / "first.txt").exists()


def test_worker_preserves_declared_loop_short_circuit(tmp_path):
    class WorkLoopIn(BaseModel):
        ms: Path

    class WorkLoopOut(BaseModel):
        ms: Path | None = None

    class AssessLoopIn(BaseModel):
        script: str
        ms: Path
        flag: Path

    class AssessLoopOut(BaseModel):
        flag: Path | None = None

    class BodyIn(BaseModel):
        ms: Path
        flag: Path

    class BodyOut(BaseModel):
        ms: Path | None = None
        converged: Path | None = None

    work = Cab(name="work", command="/bin/true", inputs_model=WorkLoopIn, outputs_model=WorkLoopOut)
    assess = Cab(
        name="assess",
        command=f"{sys.executable} -c",
        inputs_model=AssessLoopIn,
        outputs_model=AssessLoopOut,
        field_meta={"script": ParamMeta(positional_head=True), "flag": ParamMeta(positional=True)},
    )
    body = Recipe(name="body", inputs_model=BodyIn, outputs_model=BodyOut)
    body.add_step("work", work, ms=InputRef(field="ms"))
    body.add_step(
        "assess", assess, script="from pathlib import Path;import sys;Path(sys.argv[-1]).write_text('done')", ms=OutputRef(step="work", field="ms"), flag=InputRef(field="flag")
    )
    body.set_output("ms", OutputRef(step="work", field="ms"))
    body.set_output("converged", OutputRef(step="assess", field="flag"))
    outer = Recipe(name="loop-worker", inputs_model=WorkLoopIn, outputs_model=WorkLoopOut)
    outer.add_loop("cycle", body, max_iter=3, until="converged", carry={"ms": "ms"}, ms=InputRef(field="ms"), flag=tmp_path / "converged.flag")
    (tmp_path / "input.ms").write_text("input")
    workflow = prepare_worker_slurm(
        freeze_recipe(outer, {"ms": "input.ms"}, config=AppConfig(), workspace=tmp_path), submission_root=tmp_path / "runs", worker_python=Path(sys.executable)
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    states = []
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    for attempt in plan.attempts:
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = AttemptRecord.read(
            workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
            workflow_id=plan.workflow_id,
            attempt_id=attempt.attempt_id,
            step_path=attempt.step_path,
            bundle_digest=bundle.digest,
        )
        states.append(record.state)
    assert states == ["succeeded", "succeeded", "skipped", "skipped", "skipped", "skipped"]


def test_result_publication_failure_makes_worker_fail(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    original = AttemptRecord.write

    def fail_success(self, directory):
        if self.state == "succeeded":
            raise OSError("simulated result publication failure")
        return original(self, directory)

    monkeypatch.setattr(AttemptRecord, "write", fail_success)
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    assert record.state == "failed"
    assert "publication failure" in record.error
    assert not record.committed


def test_harvest_failure_retains_sandbox_and_cannot_publish_success(tmp_path):
    class ScriptIn(BaseModel):
        script: str

    class NoOutput(BaseModel):
        pass

    collision = tmp_path / "collision"
    collision.mkdir()
    (collision / "caller-owned.txt").write_text("keep")
    cab = Cab(
        name="harvest", command=f"{sys.executable} -c", inputs_model=ScriptIn, outputs_model=NoOutput, field_meta={"script": ParamMeta(positional=True)}, harvest=["collision"]
    )
    recipe = Recipe(
        name="harvest-worker",
        inputs_model=NoOutput,
        outputs_model=NoOutput,
        steps=[StepRef(name="harvest", step=cab, params={"script": "from pathlib import Path;p=Path('collision');p.mkdir();(p/'new.txt').write_text('new')"})],
    )
    workflow = prepare_worker_slurm(freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path), submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    assert record.state == "failed" and "Refusing to delete" in record.error
    assert record.sandbox is not None and (Path(record.sandbox) / "collision" / "new.txt").is_file()
    assert (collision / "caller-owned.txt").read_text() == "keep"


def test_early_finalization_is_superseded_and_completed_result_survives_accounting_purge(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    (workflow.submission_dir / "jobs").mkdir()
    for index, attempt in enumerate(plan.attempts):
        write_new(
            workflow.submission_dir / "jobs" / f"{index:04d}.json",
            SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id=str(200 + index)),
        )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "RUNNING"))
    early = finalize_submission(workflow.submission_dir)
    assert not early.complete
    assert [step.state for step in early.steps] == ["running", "running"]
    assert not (workflow.submission_dir / "finalization.json").exists()

    for attempt in plan.attempts:
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "COMPLETED"))
    complete = finalize_submission(workflow.submission_dir)
    assert complete.complete
    assert (workflow.submission_dir / "finalization.json").is_file()

    def accounting_was_purged(_jobs):
        raise AssertionError("a canonical finalization must not depend on later sacct state")

    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", accounting_was_purged)
    assert finalize_submission(workflow.submission_dir) == complete


def test_unavailable_accounting_does_not_freeze_early_unknown_status(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    (workflow.submission_dir / "jobs").mkdir()
    for index, attempt in enumerate(plan.attempts):
        write_new(
            workflow.submission_dir / "jobs" / f"{index:04d}.json",
            SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id=str(300 + index)),
        )
    write_new(
        workflow.submission_dir / "finalizer-job.json",
        SubmittedFinalizer(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, job_id="399"),
    )

    def unavailable_or_pending(jobs):
        return {name: "PENDING" if name == "__finalizer__" else "UNKNOWN" for name in jobs}

    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", unavailable_or_pending)
    observed = finalize_submission(workflow.submission_dir)
    assert not observed.complete
    assert [step.state for step in observed.steps] == ["unknown", "unknown"]
    assert not (workflow.submission_dir / "finalization.json").exists()


def test_cancelled_attempt_is_distinct_from_unknown(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest, step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id="42"),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"write": "CANCELLED by 1000"})
    finalized = finalize_submission(workflow.submission_dir)
    assert finalized.steps[0].state == "cancelled"
    assert finalized.steps[1].state == "unknown"


def test_requeued_attempt_publishes_diagnostic_record(tmp_path):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    running = AttemptRecord(workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest, state="running")
    running.write(workflow.submission_dir.parent)
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    diagnostic = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "requeue.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    assert diagnostic.state == "failed"
    assert "already started" in diagnostic.error


def test_visible_terminal_record_wins_if_directory_sync_reports_failure(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    original = AttemptRecord.write

    def publish_then_fail(self, directory):
        path = original(self, directory)
        if self.state == "succeeded":
            raise OSError("simulated post-link directory sync failure")
        return path

    monkeypatch.setattr(AttemptRecord, "write", publish_then_fail)
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    record = AttemptRecord.read(
        _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
    )
    assert record.state == "succeeded"
    assert not (_final_record(workflow, attempt).parent / "publication-error.json").exists()


def test_unresolved_image_pin_refuses_worker_submission(tmp_path, monkeypatch):
    cab = _cab("image", WriteIn).model_copy(update={"image": "repo/tool:latest", "backend": "apptainer"})
    recipe = Recipe(
        name="image-worker",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[StepRef(name="image", step=cab, params={"script": "pass"}, wiring={"out": InputRef(field="first")})],
        output_wiring={"out": OutputRef(step="image", field="out")},
    )
    monkeypatch.setattr("shinobi.backends.container._pin_image", lambda runtime, image: (image, None))
    with pytest.raises(OffloadCompileError, match="could not be pinned"):
        prepare_worker_slurm(
            freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
            submission_root=tmp_path / "runs",
            worker_python=Path(sys.executable),
        )


def test_concurrent_attempts_own_distinct_sandbox_roots(tmp_path):
    failing = _cab("fail", WriteIn)
    recipe = Recipe(
        name="parallel-failures",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[StepRef(name=name, step=failing, params={"script": "import sys,time;time.sleep(.2);raise SystemExit(7)", "out": f"{name}.txt"}) for name in ("left", "right")],
        output_wiring={"out": OutputRef(step="left", field="out")},
    )
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workflow.submission_dir / "worker-src")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shinobi.offload.worker",
                "run",
                "--submission",
                str(workflow.submission_dir),
                "--step",
                attempt.step_path,
                "--attempt-id",
                str(attempt.attempt_id),
            ],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for attempt in plan.attempts
    ]
    assert [proc.wait() for proc in processes] == [7, 7]
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    sandboxes = []
    for attempt in plan.attempts:
        record = AttemptRecord.read(
            _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
        )
        sandbox = Path(record.sandbox)
        assert sandbox.parent == workflow.submission_dir / "sandboxes" / str(attempt.attempt_id)
        sandboxes.append(sandbox)
    assert sandboxes[0] != sandboxes[1]


def test_concurrent_worker_branches_retain_every_cache_update(tmp_path):
    writer = _cab("write", WriteIn)
    names = [f"branch-{index}" for index in range(8)]
    recipe = Recipe(
        name="parallel-cache",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[
            StepRef(
                name=name,
                step=writer,
                params={
                    "script": "from pathlib import Path;import sys,time;time.sleep(.05);Path(sys.argv[1]).write_text('done')",
                    "out": f"{name}.txt",
                },
            )
            for name in names
        ],
        output_wiring={"out": OutputRef(step=names[0], field="out")},
    )
    config = AppConfig.model_validate({"cache": {"enabled": True, "dir": str(tmp_path / "cache")}})
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {}, config=config, workspace=tmp_path),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workflow.submission_dir / "worker-src")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shinobi.offload.worker",
                "run",
                "--submission",
                str(workflow.submission_dir),
                "--step",
                attempt.step_path,
                "--attempt-id",
                str(attempt.attempt_id),
            ],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for attempt in plan.attempts
    ]
    outcomes = [proc.communicate() for proc in processes]
    assert [proc.returncode for proc in processes] == [0] * len(names), outcomes
    manifest = get_cache_manifest(str(tmp_path / "cache"))
    assert all(manifest.entry(f"parallel-cache.{name}") is not None for name in names)


def test_cross_filesystem_declared_scratch_fails_before_tool(tmp_path):
    foreign = Path("/dev/shm")
    if not foreign.is_dir() or foreign.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("no second writable filesystem available")

    class ScriptIn(BaseModel):
        script: str

    class Empty(BaseModel):
        pass

    marker = tmp_path / "tool-ran"
    cab = Cab(
        name="scratch",
        command=f"{sys.executable} -c",
        inputs_model=ScriptIn,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional=True)},
        scratch=[str(foreign / f"shinobi-{tmp_path.name}" / "*")],
    )
    recipe = Recipe(
        name="scratch-worker",
        inputs_model=Empty,
        outputs_model=Empty,
        steps=[StepRef(name="scratch", step=cab, params={"script": f"from pathlib import Path;Path({str(marker)!r}).write_text('ran')"})],
    )
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
    )
    assert "different filesystem" in record.error
    assert not marker.exists()


def test_worker_rerun_replaces_absolute_declared_output(tmp_path):
    destination = tmp_path / "absolute.txt"
    cab = _cab("replace", WriteIn)

    def run_once(text: str, require_absent: bool = False):
        check = "assert not p.exists();" if require_absent else ""
        recipe = Recipe(
            name="replacement-worker",
            inputs_model=RootIn,
            outputs_model=PathOut,
            steps=[
                StepRef(
                    name="replace", step=cab, params={"script": f"from pathlib import Path;import sys;p=Path(sys.argv[1]);{check}p.write_text({text!r})", "out": str(destination)}
                )
            ],
            output_wiring={"out": OutputRef(step="replace", field="out")},
        )
        workflow = prepare_worker_slurm(
            freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
            submission_root=tmp_path / "runs",
            worker_python=Path(sys.executable),
        )
        plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0

    run_once("first")
    run_once("second", require_absent=True)
    assert destination.read_text() == "second"


@pytest.mark.parametrize("backend", ["venv", "apptainer"])
def test_failed_worker_pystep_retains_sandbox_without_clearing_in_place_input(backend, make_venv, tmp_path, monkeypatch):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    class MSIn(BaseModel):
        ms: Path

    class Empty(BaseModel):
        pass

    options = {"backend": backend}
    if backend == "venv":
        options["venv"] = str(make_venv())
    else:
        digest = "sha256:" + "b" * 64
        options["image"] = f"repo/tool@{digest}"

        def fake_container_argv(runtime, scope, argv, inputs, workdir, **kwargs):
            runner = "import os,runpy,sys;os.chdir(sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')"
            return [sys.executable, "-c", runner, workdir, argv[-1]], digest

        monkeypatch.setattr("shinobi.backends.container.build_container_argv", fake_container_argv)
        monkeypatch.setattr("shinobi.backends.container.container_stopper", lambda *args: None)

    ref = pystep(**options)(funcs.fail_after_touch)
    recipe = Recipe(name=f"failed-{backend}", inputs_model=MSIn, outputs_model=Empty, steps=[ref.model_copy(update={"wiring": {"ms": InputRef(field="ms")}})])
    data = tmp_path / "input.ms"
    data.write_text("original")
    workflow = prepare_worker_slurm(
        freeze_recipe(recipe, {"ms": data}, config=AppConfig(), workspace=tmp_path, code_roots=(Path.cwd(),)),
        submission_root=tmp_path / "runs",
        worker_python=Path(sys.executable),
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 1
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(
        _final_record(workflow, attempt), workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id, step_path=attempt.step_path, bundle_digest=bundle.digest
    )
    assert "intentional worker pystep failure" in record.error
    assert data.read_text() == "touched"
    assert record.sandbox is not None
    assert (Path(record.sandbox) / "failed-scratch.txt").read_text() == "inspect me"
