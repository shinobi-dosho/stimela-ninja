from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel

from shinobi.config import AppConfig
from shinobi.offload.bundle import RecipeBundle, freeze_recipe, write_new
from shinobi.offload.records import AttemptRecord
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import ExecutionPlan, SubmittedJob, execute_step, finalize_submission
from shinobi.steps.schema import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


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
                params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('frozen')"},
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


def test_worker_executes_wiring_in_shared_sandboxes_and_finalizes(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    for attempt in plan.attempts:
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    assert (tmp_path / "first.txt").read_text() == "frozen"
    assert (tmp_path / "second.txt").read_text() == "frozen worker"
    assert list((workflow.submission_dir / "sandboxes").iterdir()) == []

    for index, attempt in enumerate(plan.attempts):
        write_new(
            workflow.submission_dir / "jobs" / f"{index:04d}.json",
            SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest,
                         step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id=str(100 + index)),
        )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: dict.fromkeys(jobs, "COMPLETED"))
    finalized = finalize_submission(workflow.submission_dir)
    assert finalized.complete
    assert [step.step_path for step in finalized.steps] == ["write", "copy"]
    assert (workflow.submission_dir / "manifest.json").is_file()
    assert finalize_submission(workflow.submission_dir) == finalized


def test_missing_worker_result_is_unknown_not_success(tmp_path, monkeypatch):
    workflow, bundle, plan = _prepared(tmp_path)
    attempt = plan.attempts[0]
    (workflow.submission_dir / "jobs").mkdir()
    write_new(
        workflow.submission_dir / "jobs" / "0000.json",
        SubmittedJob(workflow_id=plan.workflow_id, bundle_digest=bundle.digest,
                     step_path=attempt.step_path, attempt_id=attempt.attempt_id, job_id="42"),
    )
    monkeypatch.setattr("shinobi.offload.slurm.status_slurm", lambda jobs: {"write": "COMPLETED"})
    finalized = finalize_submission(workflow.submission_dir)
    assert not finalized.complete
    assert finalized.steps[0].state == "unknown"
    unknown = AttemptRecord.read(
        workflow.submission_dir / finalized.steps[0].record,
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path="write",
        bundle_digest=bundle.digest,
    )
    assert not unknown.committed
    assert unknown.scheduler_state == "COMPLETED"


def test_staged_worker_source_is_checked_before_execution(tmp_path):
    workflow, _bundle, plan = _prepared(tmp_path)
    source = workflow.submission_dir / "worker-src" / "shinobi" / "__init__.py"
    source.write_text(source.read_text() + "\n# changed\n")
    attempt = plan.attempts[0]
    try:
        execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id)
    except Exception as exc:
        assert "no longer matches" in str(exc)
    else:
        raise AssertionError("tampered worker source was accepted")


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


def test_frozen_venv_pystep_runs_out_of_process(make_venv, tmp_path):
    from shinobi import pystep
    from tests import _venv_pystep_funcs as funcs

    package = make_venv(package=("venvonlypkg", "1.0.0", "MAGIC = 4242\n"))

    class NumberIn(BaseModel):
        n: int

    ref = pystep(venv=str(package), backend="venv")(funcs.use_venv_only_pkg)
    recipe = Recipe(name="venv-worker", inputs_model=NumberIn, outputs_model=funcs.MagicOut,
                    steps=[ref.model_copy(update={"wiring": {"n": InputRef(field="n")}})],
                    output_wiring={"value": OutputRef(step=ref.name, field="value")})
    bundle = freeze_recipe(recipe, {"n": 8}, config=AppConfig(), workspace=tmp_path, code_roots=(Path.cwd(),))
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workflow.submission_dir / "worker-src")
    proc = subprocess.run(
        [sys.executable, "-m", "shinobi.offload.worker", "run", "--submission", str(workflow.submission_dir),
         "--step", attempt.step_path, "--attempt-id", str(attempt.attempt_id)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    staged = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
                                workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id,
                                step_path=attempt.step_path, bundle_digest=staged.digest)
    assert record.result(staged.steps[0].scope.restore()).outputs.value == 4250
    assert record.observation.venv == str(package)
    assert record.observation.venv_digest is not None


def test_failed_worker_keeps_shared_sandbox_and_publishes_no_success(tmp_path):
    bad = _cab("bad", WriteIn)
    recipe = Recipe(
        name="bad-worker",
        inputs_model=RootIn,
        outputs_model=PathOut,
        steps=[StepRef(name="bad", step=bad,
                       params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('partial');raise SystemExit(7)"},
                       wiring={"out": InputRef(field="first")})],
        output_wiring={"out": OutputRef(step="bad", field="out")},
    )
    workflow = prepare_worker_slurm(freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path),
                                    submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 7
    bundle = RecipeBundle.read(workflow.submission_dir / "bundle.json")
    record = AttemptRecord.read(workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
                                workflow_id=plan.workflow_id, attempt_id=attempt.attempt_id,
                                step_path=attempt.step_path, bundle_digest=bundle.digest)
    assert record.state == "failed"
    assert not record.committed
    assert record.sandbox is not None and Path(record.sandbox).is_dir()
    assert not (tmp_path / "first.txt").exists()
