"""Run/check the physical M3 mutation-recovery acceptance workflow.

Run inside the Kudu controller container; see this directory's README.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from shinobi import pystep
from shinobi.config import AppConfig
from shinobi.exceptions import BackendError
from shinobi.offload.bundle import freeze_recipe
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import AttemptInvocation, ExecutionPlan, Finalization, finalize_submission
from shinobi.provenance import RunManifest
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, ParamMeta, Recipe, StepRef
from tests.slurm_physical import m3_funcs


class RootIn(BaseModel):
    ms: Path


class MutationIn(BaseModel):
    script: str
    ms: Path


class ReportIn(BaseModel):
    script: str
    # Harness coordination, not scientific path input: keeping this a string
    # prevents its changing marker directory from becoming cache identity.
    barrier: str
    label: str
    out: str


class ReportOut(BaseModel):
    report: Path


def _mutation_cab() -> Cab:
    return Cab(
        name="binary-mutation",
        command="python3 -c",
        inputs_model=MutationIn,
        outputs_model=m3_funcs.MsOut,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True)},
        input_mutability={"ms": Mutability.MUTABLE},
    )


def _report_cab() -> Cab:
    return Cab(
        name="independent-report",
        command="python3 -c",
        inputs_model=ReportIn,
        outputs_model=ReportOut,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "barrier": ParamMeta(positional=True),
            "label": ParamMeta(positional=True),
            "out": ParamMeta(positional=True),
            "report": ParamMeta(implicit="{out}"),
        },
    )


REPORT_SCRIPT = (
    "from pathlib import Path;import sys,time;"
    "b=Path(sys.argv[1]);label=sys.argv[2];out=Path(sys.argv[3]);"
    "b.mkdir(parents=True,exist_ok=True);(b/label).write_text('ready');"
    "deadline=time.monotonic()+60;"
    'exec("while len(list(b.iterdir())) < 2:\\n'
    " if time.monotonic() >= deadline: raise TimeoutError('peer did not reach barrier')\\n"
    ' time.sleep(.05)");'
    "out.write_text('report-'+label)"
)


def recipe(
    workspace: Path,
    image: Path,
    tool_venv: Path,
    *,
    seed: str,
    image_func: Callable = m3_funcs.image_v1,
    venv_func: Callable = m3_funcs.venv_v1,
    split_delay: int = 0,
) -> Recipe:
    # A reduction's split/transform stage replaces its derived dataset for a
    # changed selection; it must not append to its own previous product.
    split_script = f"from pathlib import Path;import sys,time;time.sleep({split_delay});p=Path(sys.argv[1])/'table.dat';p.write_text('vis[{seed}]')"
    image_ref = pystep(name="image", image=str(image), backend="apptainer")(image_func)
    venv_ref = pystep(name="venv", venv=str(tool_venv), backend="venv")(venv_func)
    report = _report_cab()
    barrier = workspace / "report-barrier"
    return Recipe(
        name="physical-m3",
        inputs_model=RootIn,
        outputs_model=m3_funcs.MsOut,
        steps=[
            StepRef(name="split", step=_mutation_cab(), params={"script": split_script}, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="report-a", step=report, params={"script": REPORT_SCRIPT, "barrier": str(barrier), "label": "a", "out": "report-a.txt"}),
            StepRef(name="report-b", step=report, params={"script": REPORT_SCRIPT, "barrier": str(barrier), "label": "b", "out": "report-b.txt"}),
            image_ref.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}}),
            venv_ref.model_copy(update={"wiring": {"ms": OutputRef(step="image", field="ms")}}),
        ],
        output_wiring={"ms": OutputRef(step="venv", field="ms")},
    )


def _config(args) -> AppConfig:
    return AppConfig.model_validate(
        {
            "cache": {"enabled": True, "dir": str(args.root / "cache")},
            "sandbox": {"enabled": True, "dir": str(args.root / "sandboxes")},
        }
    )


def _submit(args, label: str, value: Recipe, *, fault_stage: str | None = None):
    bundle = freeze_recipe(
        value,
        {"ms": args.root / "workspace" / "obs.ms"},
        config=_config(args),
        workspace=args.root / "workspace",
        code_roots=(args.source_root,),
    )
    nodes = args.nodes
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=args.root / "submissions",
        worker_python=args.worker_python,
        sbatch_opts={"partition": args.partition},
        step_sbatch_opts={
            "split": {"nodelist": nodes[0]},
            "report-a": {"nodelist": nodes[1]},
            "report-b": {"nodelist": nodes[2]},
            "image": {"nodelist": nodes[1]},
            "venv": {"nodelist": nodes[2]},
        },
    )
    if fault_stage is not None:
        _inject_fault(args, workflow, "image", fault_stage)
    handle = submit_worker_slurm(workflow)
    handles = args.root / "handles"
    handles.mkdir(exist_ok=True)
    payload = {"submission": str(handle.submission_dir), "jobs": handle.jobs, "finalizer": handle.finalizer_job}
    (handles / f"{label}.json").write_text(json.dumps(payload, indent=2))
    return handle


def _inject_fault(args, workflow, target: str, stage: str) -> None:
    """Install a one-shot worker-process crash through ``sitecustomize``."""
    hook_root = args.root / "fault-hook"
    hook_root.mkdir(exist_ok=True)
    marker = args.root / f"fault-{stage}.once"
    marker.write_text("armed")
    (hook_root / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "stage = os.environ.get('SHINOBI_PHYSICAL_FAULT_STAGE')\n"
        "marker = Path(os.environ['SHINOBI_PHYSICAL_FAULT_ONCE'])\n"
        "if stage:\n"
        "    from shinobi.snapshots import faults\n"
        "    def stop():\n"
        "        try:\n"
        "            marker.unlink()\n"
        "        except FileNotFoundError:\n"
        "            return\n"
        "        os._exit(86)\n"
        "    faults.hooks[stage] = stop\n"
    )
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    job = next(job for job in workflow.jobs if job.name == target)
    old = f"PYTHONPATH={plan.worker.source}"
    new = f"SHINOBI_PHYSICAL_FAULT_STAGE={stage} SHINOBI_PHYSICAL_FAULT_ONCE={marker} PYTHONPATH={hook_root}:{plan.worker.source}"
    assert old in job.script
    job.script = job.script.replace(old, new, 1)


def _wait(submission: Path, timeout: float = 300) -> Finalization:
    path = submission / "finalization.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"finalizer did not publish {path}")
        time.sleep(0.5)
    return Finalization.model_validate_json(path.read_text())


def _phase(args, label: str, expected: list[str], **recipe_args) -> Finalization:
    finalized = _wait(_submit(args, label, recipe(args.root / "workspace", args.image, recipe_args.pop("tool_venv", args.tool_venv), **recipe_args)).submission_dir)
    states = [step.state for step in finalized.steps]
    assert finalized.complete and states == expected, (label, states)
    return finalized


def _requeue_phase(args, label: str, target: str, ready: Path, **recipe_args) -> Finalization:
    ready.unlink(missing_ok=True)
    (ready.parent / f".{target}-requeue-once").write_text("armed")
    handle = _submit(args, label, recipe(args.root / "workspace", args.image, recipe_args.pop("tool_venv", args.tool_venv), **recipe_args))
    deadline = time.monotonic() + 120
    while not ready.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{target} did not publish its requeue barrier")
        time.sleep(0.25)
    process = subprocess.run(["scontrol", "requeue", handle.jobs[target]], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"scontrol requeue failed: {process.stderr.strip()}")
    return _wait(handle.submission_dir)


def _source_snapshot_phase(args) -> Finalization:
    """Edit the checkout after submission while the captured helper runs."""
    helper = args.source_root / "tests" / "slurm_physical" / "m3_helper.py"
    original = helper.read_text()
    edited = original.replace('return f"image[{version}]"', 'return "EDITED"')
    assert edited != original
    handle = _submit(
        args,
        "source-snapshot",
        recipe(
            args.root / "workspace",
            args.image,
            args.alt_tool_venv,
            seed="source",
            image_func=m3_funcs.image_v1,
            split_delay=8,
        ),
    )
    try:
        cancelled = subprocess.run(["scancel", handle.finalizer_job], capture_output=True, text=True)
        if cancelled.returncode:
            raise RuntimeError(f"scancel finalizer failed: {cancelled.stderr.strip()}")
        helper.write_text(edited)
        plan = ExecutionPlan.model_validate_json((handle.submission_dir / "execution.json").read_text())
        deadline = time.monotonic() + 180
        while not all((handle.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").exists() for attempt in plan.attempts):
            if time.monotonic() >= deadline:
                raise TimeoutError("scientific jobs did not finish after finalizer cancellation")
            time.sleep(0.5)
        finalized = finalize_submission(handle.submission_dir)
    finally:
        helper.write_text(original)
    content = (args.root / "workspace" / "obs.ms" / "table.dat").read_text()
    assert content == "vis[source]|image[v1]|venv[4242]", content
    return finalized


def _fault_boundary_phase(args, stage: str, image_func: Callable) -> None:
    label = f"fault-{stage.lower()}"
    value = recipe(
        args.root / "workspace",
        args.image,
        args.alt_tool_venv,
        seed="two",
        image_func=image_func,
    )
    failed = _wait(_submit(args, f"{label}-failed", value, fault_stage=stage).submission_dir)
    assert not failed.complete
    failed_states = [step.state for step in failed.steps]
    assert failed_states[-1] == "cancelled", failed_states
    assert failed_states[3] == ("succeeded" if stage == "W_RESULT" else "failed"), failed_states

    retried = _wait(_submit(args, f"{label}-retry", value).submission_dir)
    states = [step.state for step in retried.steps]
    assert retried.complete and states == ["cached", "cached", "cached", "succeeded", "succeeded"], states
    content = (args.root / "workspace" / "obs.ms" / "table.dat").read_text()
    expected = f"vis[two]|boundary[{stage}]|venv[4242]"
    assert content == expected, content


def _verify_missing_environment(args) -> None:
    missing = args.root / "does-not-exist" / "bin" / "python"
    try:
        freeze_recipe(
            recipe(args.root / "workspace", args.image, missing.parent.parent, seed="never"),
            {"ms": args.root / "workspace" / "obs.ms"},
            config=_config(args),
            workspace=args.root / "workspace",
            code_roots=(args.source_root,),
        )
    except BackendError as exc:
        message = str(exc)
        assert "does not exist" in message
        (args.root / "missing-environment.json").write_text(json.dumps({"rejected": True, "error": message}, indent=2))
    else:
        raise AssertionError("missing tool venv did not fail during preparation")


def run(args) -> int:
    if len(args.nodes) != 3 or len(set(args.nodes)) != 3:
        raise ValueError("--nodes requires three distinct nodes")
    args.root.mkdir(parents=True, exist_ok=False)
    workspace = args.root / "workspace"
    ms = workspace / "obs.ms"
    ms.mkdir(parents=True)
    (ms / "table.dat").write_text("raw")
    os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = str(args.root / "workspace-owners.json")
    _verify_missing_environment(args)

    all_succeeded = ["succeeded"] * 5
    _phase(args, "first", all_succeeded, seed="one")
    _phase(args, "unchanged", ["cached"] * 5, seed="one")
    _phase(args, "changed", ["succeeded", "cached", "cached", "succeeded", "succeeded"], seed="two")
    _phase(
        args,
        "code",
        ["cached", "cached", "cached", "succeeded", "succeeded"],
        seed="two",
        image_func=m3_funcs.image_v2,
    )
    _phase(
        args,
        "environment",
        ["cached", "cached", "cached", "cached", "succeeded"],
        seed="two",
        image_func=m3_funcs.image_v2,
        tool_venv=args.alt_tool_venv,
    )
    (workspace / "report-a.txt").unlink()
    _phase(
        args,
        "missing-output",
        ["cached", "succeeded", "cached", "cached", "cached"],
        seed="two",
        image_func=m3_funcs.image_v2,
        tool_venv=args.alt_tool_venv,
    )

    source_final = _source_snapshot_phase(args)
    assert source_final.complete
    assert [step.state for step in source_final.steps] == ["succeeded", "cached", "cached", "succeeded", "succeeded"]

    _fault_boundary_phase(args, "S2", m3_funcs.image_boundary_s2)
    _fault_boundary_phase(args, "W_RESULT", m3_funcs.image_boundary_w_result)

    image_final = _requeue_phase(
        args,
        "image-requeue",
        "image",
        workspace / ".image-requeue-ready",
        seed="two",
        image_func=m3_funcs.image_requeue,
        tool_venv=args.alt_tool_venv,
    )
    assert image_final.complete
    assert [step.state for step in image_final.steps] == ["cached", "cached", "cached", "succeeded", "succeeded"]
    assert "PARTIAL" not in (ms / "table.dat").read_text()

    venv_final = _requeue_phase(
        args,
        "venv-requeue",
        "venv",
        workspace / ".venv-requeue-ready",
        seed="two",
        image_func=m3_funcs.image_requeue,
        venv_func=m3_funcs.venv_requeue,
        tool_venv=args.alt_tool_venv,
    )
    assert venv_final.complete
    assert [step.state for step in venv_final.steps] == ["cached", "cached", "cached", "cached", "succeeded"]
    assert (ms / "table.dat").read_text() == "vis[two]|image[requeued]|venv-requeued[4242]"
    sys.stdout.write(json.dumps({"complete": True, "root": str(args.root)}) + "\n")
    return 0


def check(args) -> int:
    expected = {
        "first": ["succeeded"] * 5,
        "unchanged": ["cached"] * 5,
        "changed": ["succeeded", "cached", "cached", "succeeded", "succeeded"],
        "code": ["cached", "cached", "cached", "succeeded", "succeeded"],
        "environment": ["cached", "cached", "cached", "cached", "succeeded"],
        "missing-output": ["cached", "succeeded", "cached", "cached", "cached"],
        "source-snapshot": ["succeeded", "cached", "cached", "succeeded", "succeeded"],
        "image-requeue": ["cached", "cached", "cached", "succeeded", "succeeded"],
        "venv-requeue": ["cached", "cached", "cached", "cached", "succeeded"],
    }
    observed = {}
    for label, states in expected.items():
        handle = json.loads((args.root / "handles" / f"{label}.json").read_text())
        submission = Path(handle["submission"])
        finalized = Finalization.model_validate_json((submission / "finalization.json").read_text())
        observed[label] = [step.state for step in finalized.steps]
        assert finalized.complete and observed[label] == states
        manifest = RunManifest.model_validate_json((submission / "manifest.json").read_text())
        image_step, venv_step = manifest.root.steps[-2:]
        assert image_step.containerized and image_step.image_digest and image_step.code_digest
        assert venv_step.venv_digest and venv_step.code_digest and not manifest.pinned
        if label.endswith("requeue"):
            target = label.removesuffix("-requeue")
            plan = ExecutionPlan.model_validate_json((submission / "execution.json").read_text())
            planned = plan.attempt(target)
            invocation = AttemptInvocation.model_validate_json((submission / "attempts" / str(planned.attempt_id) / "restart-00000001.json").read_text())
            assert invocation.restart_count == 1
            assert invocation.attempt_id == next(step.attempt_id for step in finalized.steps if step.step_path == target)
    for stage in ("s2", "w_result"):
        failed_handle = json.loads((args.root / "handles" / f"fault-{stage}-failed.json").read_text())
        failed = Finalization.model_validate_json((Path(failed_handle["submission"]) / "finalization.json").read_text())
        assert not failed.complete and failed.steps[-1].state == "cancelled"
        retry_handle = json.loads((args.root / "handles" / f"fault-{stage}-retry.json").read_text())
        retry_submission = Path(retry_handle["submission"])
        retry = Finalization.model_validate_json((retry_submission / "finalization.json").read_text())
        assert retry.complete and [step.state for step in retry.steps] == ["cached", "cached", "cached", "succeeded", "succeeded"]
        assert (retry_submission / "manifest.json").is_file()
    workspace = args.root / "workspace"
    assert (workspace / "obs.ms" / "table.dat").read_text() == "vis[two]|image[requeued]|venv-requeued[4242]"
    assert (workspace / "report-a.txt").read_text() == "report-a"
    assert (workspace / "report-b.txt").read_text() == "report-b"
    assert not (workspace / ".image-requeue-once").exists()
    assert not (workspace / ".venv-requeue-once").exists()
    rejected = json.loads((args.root / "missing-environment.json").read_text())
    assert rejected["rejected"] and "does not exist" in rejected["error"]
    sys.stdout.write(json.dumps({"complete": True, "phases": observed}) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/data/src/stimela-ninja-m3-145"))
    parser.add_argument("--image", type=Path, default=Path("/data/images/python-3.12-alpine.sif"))
    parser.add_argument("--tool-venv", type=Path, default=Path("/data/m1-tool-venv"))
    parser.add_argument("--alt-tool-venv", type=Path, default=Path("/data/m3-tool-venv-alt"))
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--nodes", nargs=3, default=("k1", "n1", "n2"))
    args = parser.parse_args()
    if args.command == "run" and not args.alt_tool_venv.exists():
        shutil.copytree(args.tool_venv, args.alt_tool_venv, symlinks=True)
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
