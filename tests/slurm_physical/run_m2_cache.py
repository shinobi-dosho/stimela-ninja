"""Run/check the physical M2 detached-worker cache acceptance workflow.

Run inside the Kudu controller container; see this directory's README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from shinobi import pystep
from shinobi.cache import CacheManifest
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


class CopyIn(BaseModel):
    script: str
    src: Path
    out: str


class Product(BaseModel):
    product: Path


def _cab(name: str, inputs: type[BaseModel]) -> Cab:
    return Cab(
        name=name,
        command="python3 -c",
        inputs_model=inputs,
        outputs_model=Product,
        field_meta={
            **{field: ParamMeta(positional=True) for field in inputs.model_fields},
            "product": ParamMeta(implicit="{out}"),
        },
    )


def recipe(variant: str, image: Path, tool_venv: Path) -> Recipe:
    writer = _cab("writer", WriteIn)
    copier = _cab("copier", CopyIn)
    write_script = f"from pathlib import Path;import sys;Path(sys.argv[1]).write_text({variant!r})"
    copy_script = "from pathlib import Path;import sys;Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text()+' copied')"
    unrelated_script = "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('unrelated')"
    image_ref = pystep(name="image", image=str(image), backend="apptainer")(m1_funcs.image_step)
    venv_ref = pystep(name="venv", venv=str(tool_venv), backend="venv")(m1_funcs.venv_step)
    return Recipe(
        name="physical-m2-cache",
        inputs_model=Empty,
        outputs_model=Product,
        steps=[
            StepRef(name="write", step=writer, params={"script": write_script, "out": "root.txt"}),
            StepRef(
                name="left",
                step=copier,
                params={"script": copy_script, "out": "left.txt"},
                wiring={"src": OutputRef(step="write", field="product")},
            ),
            StepRef(
                name="right",
                step=copier,
                params={"script": copy_script, "out": "right.txt"},
                wiring={"src": OutputRef(step="write", field="product")},
            ),
            StepRef(name="unrelated", step=writer, params={"script": unrelated_script, "out": "unrelated.txt"}),
            image_ref.model_copy(update={"wiring": {"source": OutputRef(step="write", field="product")}}),
            venv_ref.model_copy(update={"wiring": {"source": OutputRef(step="image", field="product")}}),
        ],
        output_wiring={"product": OutputRef(step="left", field="product")},
    )


def _submit(args, label: str, variant: str) -> Path:
    workspace = args.root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = AppConfig.model_validate(
        {
            "cache": {"enabled": True, "dir": str(args.root / "cache")},
            "sandbox": {"enabled": True, "dir": str(args.root / "sandboxes")},
        }
    )
    bundle = freeze_recipe(
        recipe(variant, args.image, args.tool_venv),
        {},
        config=config,
        workspace=workspace,
        code_roots=(args.source_root,),
    )
    workflow = prepare_worker_slurm(
        bundle,
        submission_root=args.root / "submissions",
        worker_python=args.worker_python,
        sbatch_opts={"partition": "dev"},
        step_sbatch_opts={
            "write": {"nodelist": "k1"},
            "left": {"nodelist": "n1"},
            "right": {"nodelist": "n2"},
            "unrelated": {"nodelist": "k1"},
            "image": {"nodelist": "n1"},
            "venv": {"nodelist": "n2"},
        },
    )
    handle = submit_worker_slurm(workflow)
    handles = args.root / "handles"
    handles.mkdir(exist_ok=True)
    (handles / f"{label}.json").write_text(
        json.dumps(
            {
                "submission": str(handle.submission_dir),
                "jobs": handle.jobs,
                "finalizer": handle.finalizer_job,
            },
            indent=2,
        )
    )
    return handle.submission_dir


def _wait(submission: Path, timeout: float = 180) -> Finalization:
    path = submission / "finalization.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"finalizer did not publish {path}")
        time.sleep(0.5)
    return Finalization.model_validate_json(path.read_text())


def run(args) -> int:
    args.root.mkdir(parents=True, exist_ok=False)
    phases = [
        ("first", "one", ["succeeded"] * 6),
        ("unchanged", "one", ["cached"] * 6),
        ("changed", "two", ["succeeded", "succeeded", "succeeded", "cached", "succeeded", "succeeded"]),
    ]
    summary = {}
    for label, variant, expected in phases:
        finalized = _wait(_submit(args, label, variant))
        states = [step.state for step in finalized.steps]
        assert finalized.complete and states == expected, (label, states)
        summary[label] = states

    (args.root / "workspace" / "left.txt").unlink()
    deleted = _wait(_submit(args, "deleted", "two"))
    states = [step.state for step in deleted.steps]
    assert deleted.complete and states == ["cached", "succeeded", "cached", "cached", "cached", "cached"], states
    summary["deleted"] = states
    sys.stdout.write(json.dumps(summary) + "\n")
    return 0


def check(args) -> int:
    expected = {
        "first": ["succeeded"] * 6,
        "unchanged": ["cached"] * 6,
        "changed": ["succeeded", "succeeded", "succeeded", "cached", "succeeded", "succeeded"],
        "deleted": ["cached", "succeeded", "cached", "cached", "cached", "cached"],
    }
    observed = {}
    for label, states in expected.items():
        handle = json.loads((args.root / "handles" / f"{label}.json").read_text())
        finalized = Finalization.model_validate_json((Path(handle["submission"]) / "finalization.json").read_text())
        observed[label] = [step.state for step in finalized.steps]
        assert finalized.complete and observed[label] == states
        manifest = RunManifest.model_validate_json((Path(handle["submission"]) / "manifest.json").read_text())
        image_step, venv_step = manifest.root.steps[4:]
        assert image_step.image_digest and image_step.code_digest and image_step.containerized
        assert venv_step.venv == str(args.tool_venv) and venv_step.venv_digest and venv_step.code_digest
        assert not manifest.pinned
    manifest = CacheManifest(args.root / "cache" / "manifest.json")
    assert all(manifest.entry(f"physical-m2-cache.{name}") is not None for name in ("write", "left", "right", "unrelated", "image", "venv"))
    workspace = args.root / "workspace"
    assert (workspace / "root.txt").read_text() == "two"
    assert (workspace / "left.txt").read_text() == "two copied"
    assert (workspace / "right.txt").read_text() == "two copied"
    assert (workspace / "unrelated.txt").read_text() == "unrelated"
    assert (workspace / "image-product.txt").read_text() == "two image"
    assert (workspace / "venv-product.txt").read_text() == "two image venv=4242"
    sys.stdout.write(json.dumps({"complete": True, "phases": observed}) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/data/src/stimela-ninja-m2-141"))
    parser.add_argument("--image", type=Path, default=Path("/data/images/python-3.12-alpine.sif"))
    parser.add_argument("--tool-venv", type=Path, default=Path("/data/m1-tool-venv"))
    parser.add_argument("--worker-python", type=Path, default=Path("/opt/stimela/bin/python"))
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
