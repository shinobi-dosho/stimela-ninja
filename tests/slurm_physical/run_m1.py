"""Submit/check the physical shared-storage M1 acceptance workflow.

Run inside the Kudu controller container; see this directory's README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from shinobi import pystep
from shinobi.config import AppConfig
from shinobi.offload.bundle import freeze_recipe
from shinobi.offload.slurm import prepare_worker_slurm, submit_worker_slurm
from shinobi.offload.worker import Finalization
from shinobi.provenance import RunManifest
from shinobi.steps.schema import Cab, OutputRef, ParamMeta, Recipe, StepRef
from tests.slurm_physical import m1_funcs


class Empty(BaseModel):
    pass


class WriteIn(BaseModel):
    script: str
    out: str


def recipe(image: Path, tool_venv: Path, *, fail: bool = False) -> Recipe:
    write = Cab(
        name="binary",
        command="python3 -c",
        inputs_model=WriteIn,
        outputs_model=m1_funcs.Product,
        field_meta={"script": ParamMeta(positional=True), "out": ParamMeta(positional=True), "product": ParamMeta(implicit="{out}")},
    )
    image_ref = pystep(name="image", image=str(image), backend="apptainer")(m1_funcs.image_step)
    venv_ref = pystep(name="venv", venv=str(tool_venv), backend="venv")(m1_funcs.venv_step)
    return Recipe(
        name="physical-m1",
        inputs_model=Empty,
        outputs_model=m1_funcs.Product,
        steps=[
            StepRef(name="binary", step=write,
                    params={"script": "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('binary')" + (";raise SystemExit(7)" if fail else ""),
                            "out": "binary-product.txt"}),
            image_ref.model_copy(update={"wiring": {"source": OutputRef(step="binary", field="product")}}),
            venv_ref.model_copy(update={"wiring": {"source": OutputRef(step="image", field="product")}}),
        ],
        output_wiring={"product": OutputRef(step="venv", field="product")},
    )


def submit(args) -> int:
    workspace = args.root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    value = recipe(args.image, args.tool_venv, fail=args.fail)
    bundle = freeze_recipe(value, {}, config=AppConfig(), workspace=workspace, code_roots=(args.source_root,))
    workflow = prepare_worker_slurm(bundle, submission_root=workspace / ".shinobi" / "submissions", worker_python=args.worker_python,
                                    sbatch_opts={"partition": "dev"},
                                    step_sbatch_opts={"binary": {"nodelist": "k1"}, "image": {"nodelist": "n1"}, "venv": {"nodelist": "k1"}})
    handle = submit_worker_slurm(workflow)
    data = {"submission": str(handle.submission_dir), "jobs": handle.jobs, "finalizer": handle.finalizer_job}
    (args.root / "handle.json").write_text(json.dumps(data, indent=2))
    sys.stdout.write(json.dumps(data) + "\n")
    return 0


def check(args) -> int:
    data = json.loads((args.root / "handle.json").read_text())
    submission = Path(data["submission"])
    finalized = Finalization.model_validate_json((submission / "finalization.json").read_text())
    if args.fail:
        assert not finalized.complete
        assert [step.state for step in finalized.steps] == ["failed", "cancelled", "cancelled"]
        assert finalized.steps[0].scheduler_state == "FAILED"
        assert finalized.steps[1].scheduler_state.startswith("CANCELLED")
        assert finalized.steps[2].scheduler_state.startswith("CANCELLED")
        assert not (submission / "manifest.json").exists()
        failed_record = json.loads((submission / finalized.steps[0].record).read_text())
        assert Path(failed_record["sandbox"]).is_dir()
        sys.stdout.write(json.dumps({"complete": False, "failure_verified": True, "submission": str(submission)}) + "\n")
        return 0
    assert finalized.complete
    assert [step.step_path for step in finalized.steps] == ["binary", "image", "venv"]
    assert all(step.state == "succeeded" for step in finalized.steps)
    manifest = RunManifest.model_validate_json((submission / "manifest.json").read_text())
    assert [step.name for step in manifest.root.steps] == ["binary", "image", "venv"]
    assert manifest.root.steps[1].containerized and manifest.root.steps[1].image_digest
    assert manifest.root.steps[1].code_digest and manifest.root.steps[1].worker_digest
    assert manifest.root.steps[1].job_id and manifest.root.steps[1].scheduler_state == "COMPLETED"
    assert manifest.root.steps[2].venv == str(args.tool_venv) and manifest.root.steps[2].venv_digest
    assert not manifest.pinned  # a venv is version parity, never an OS/image pin
    assert (submission.parent.parent.parent / "venv-product.txt").read_text() == "binary image venv=4242"
    sys.stdout.write(json.dumps({"complete": True, "submission": str(submission), "finalizer": data["finalizer"]}) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/data/src/stimela-ninja"))
    parser.add_argument("--image", type=Path, default=Path("/data/images/python-3.12-alpine.sif"))
    parser.add_argument("--tool-venv", type=Path, default=Path("/data/m1-tool-venv"))
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
