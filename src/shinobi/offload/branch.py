"""Slurm arrays with one bundled local recipe driver per branch.

This transports a trusted recipe builder and explicit JSON arguments, not a
partially flattened atomic-worker bundle. Inner recipes/scatters run through
normal local dispatch. Nested scheduler/remote backends are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from pydantic import Field, JsonValue

from shinobi.config import AppConfig
from shinobi.graph import build_graph
from shinobi.offload._codec import BundleError, WireModel, pack
from shinobi.offload.bundle import ScopeSpec, fingerprint, write_new
from shinobi.offload.code import CodeBundle, capture_code, source_tree_digest
from shinobi.offload.slurm import _stage_worker, build_sbatch_script, parse_sbatch_job_id
from shinobi.offload.worker import WorkerEnvironment, verify_worker_environment
from shinobi.steps.schema import Recipe


class BranchCase(WireModel):
    name: str
    workspace: Path
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    overwrite_steps: tuple[str, ...] = ()


class BranchArray(WireModel):
    schema_version: int = 1
    workflow_id: str
    code: CodeBundle
    worker: WorkerEnvironment
    cases: tuple[BranchCase, ...]
    config: dict[str, JsonValue]
    environment: dict[str, str]
    images: dict[str, str]
    case_images: dict[str, tuple[str, ...]]
    declarations: dict[str, JsonValue]


class BranchRecord(WireModel):
    workflow_id: str
    bundle_digest: str
    case: str
    attempt_id: str
    status: str
    scheduler_job: str | None = None
    restart_count: int = 0
    error: str | None = None
    workspace: str | None = None
    submission: str | None = None


def _file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _local_images(recipe, default_backend):
    if not isinstance(recipe, Recipe):
        raise BundleError("a branch builder must return a declared Recipe")
    build_graph(recipe)
    images = set()

    def visit(scope, inherited):
        backend = scope.backend or inherited
        if backend not in {"native", "apptainer"}:
            raise BundleError(f"branch driver requires local native/apptainer execution, got {backend!r}; nested offloading is forbidden")
        if scope.image:
            path = Path(scope.image)
            if not path.is_absolute() or not path.is_file():
                raise BundleError(f"branch images must be existing absolute local files: {scope.image}")
            images.add(str(path))
        if isinstance(scope, Recipe):
            for ref in scope.steps:
                visit(ref.step, backend)

    visit(recipe, default_backend)
    return images


def branch_declaration(scope):
    """Freeze the complete declared shape without flattening nested scatter."""
    result = {"scope": ScopeSpec.capture(scope).model_dump(mode="json")}
    if isinstance(scope, Recipe):
        result["max_workers"] = scope.max_workers
        result["outputs"] = {name: ref.model_dump(mode="json") for name, ref in scope.output_wiring.items()}
        result["steps"] = []
        for ref in scope.steps:
            func = getattr(ref.func, "__wrapped__", ref.func)
            result["steps"].append(
                {
                    "name": ref.name,
                    "scope": branch_declaration(ref.step),
                    "params": pack(ref.params),
                    "wiring": {
                        name: [one.model_dump(mode="json") for one in value] if isinstance(value, list) else value.model_dump(mode="json") for name, value in ref.wiring.items()
                    },
                    "after": ref.after,
                    "scatter": ref.scatter.model_dump(mode="json") if ref.scatter else None,
                    "loop": ref.loop.model_dump(mode="json") if ref.loop else None,
                    "callable": [func.__module__, func.__qualname__] if func else None,
                }
            )
    return result


def prepare_branch_array(builder, cases, *, directory: Path, code_roots, worker_python=None, config=None, environment=None, sbatch_opts=None, max_parallel=1):
    """Stage source, driver and one sbatch array; submit nothing.

    Workspaces must be disjoint and on storage visible to all selected workers.
    The caller supplies reviewed local container images and a matching worker
    Python. Each task gets persistent per-case cache/provenance/sandbox roots.
    """
    cases = tuple(cases)
    if not cases or max_parallel < 1 or len({case.name for case in cases}) != len(cases):
        raise BundleError("branch array needs distinct cases and positive concurrency")
    roots = [case.workspace.resolve() for case in cases]
    if any(not case.workspace.is_absolute() for case in cases) or any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1 :]):
        raise BundleError("branch workspaces must be absolute and disjoint")
    config = config or AppConfig()
    images = set()
    case_images = {}
    declarations = {}
    for case in cases:
        recipe = builder(**case.parameters)
        case_images[case.name] = tuple(sorted(_local_images(recipe, config.backend.default)))
        declarations[case.name] = branch_declaration(recipe)
        images.update(case_images[case.name])
    code = capture_code(builder, roots=tuple(code_roots))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    code.write(directory / "source")
    worker = _stage_worker(directory, Path(worker_python or sys.executable).absolute())
    environment = dict(environment or {})
    environment.setdefault("SHINOBI_OWNERSHIP_REGISTRY", str(directory.parent / "workspace-owners.json"))
    bundle = BranchArray(
        workflow_id=str(uuid4()),
        code=code,
        worker=worker,
        cases=cases,
        config=config.model_dump(mode="json"),
        environment=environment,
        images={path: _file_digest(path) for path in sorted(images)},
        case_images=case_images,
        declarations=declarations,
    )
    write_new(directory / "branch-array.json", bundle)
    (directory / "logs").mkdir()
    options = dict(sbatch_opts or {})
    if "array" in options:
        raise BundleError("array selection comes from cases and max_parallel")
    options["array"] = f"0-{len(cases) - 1}%{max_parallel}"
    options.setdefault("open-mode", "append")
    script = build_sbatch_script(
        job_name="shinobi-branches",
        chdir=str(directory),
        stdout_path=directory / "logs/%A_%a.out",
        stderr_path=directory / "logs/%A_%a.err",
        sbatch_opts=options,
        argv=[
            "env",
            *[f"{key}={value}" for key, value in bundle.environment.items()],
            f"PYTHONPATH={worker.source}",
            worker.python,
            "-m",
            "shinobi.offload.branch",
            str(directory),
        ],
        error=BundleError,
    )
    (directory / "array.sbatch").write_text(script)
    return directory


def submit_branch_array(directory: Path):
    """Submit once; preserve an immutable claim if scheduler acceptance is uncertain."""
    directory = directory.resolve()
    bundle = BranchArray.model_validate_json((directory / "branch-array.json").read_text())
    claim = BranchRecord(workflow_id=bundle.workflow_id, bundle_digest=fingerprint(bundle), case="array", attempt_id=str(uuid4()), status="submitting")
    write_new(directory / "submission-claim.json", claim)
    result = subprocess.run(["sbatch", "--parsable", str(directory / "array.sbatch")], capture_output=True, text=True, check=True)
    job = parse_sbatch_job_id(result.stdout)
    write_new(directory / "submitted.json", claim.model_copy(update={"status": "submitted", "scheduler_job": job}))
    return job


def recover_branch_claims(case, current_attempt):
    """Reconcile only exact previous branch attempts, using durable liveness."""
    from shinobi.ownership import WorkspaceRegistryStore, reconcile_ownership

    for path in sorted((case.workspace / ".shinobi/branch-attempts").glob("*.json")):
        previous = BranchRecord.model_validate_json(path.read_text())
        if previous.attempt_id == current_attempt or previous.case != case.name or previous.workspace != str(case.workspace.resolve()):
            continue
        if previous.submission is None:
            raise BundleError("previous branch attempt has no recorded submission")
        bundle = BranchArray.model_validate_json((Path(previous.submission) / "branch-array.json").read_text())
        if bundle.workflow_id != previous.workflow_id or fingerprint(bundle) != previous.bundle_digest:
            raise BundleError("previous branch attempt disagrees with its immutable bundle")
        indices = [index for index, item in enumerate(bundle.cases) if item.name == previous.case and item.workspace.resolve() == case.workspace.resolve()]
        if len(indices) != 1 or path.stem != previous.attempt_id:
            raise BundleError("previous branch attempt has inconsistent case identity")
        started = Path(previous.submission) / "attempts" / str(indices[0]) / previous.attempt_id / "started.json"
        if BranchRecord.model_validate_json(started.read_text()) != previous:
            raise BundleError("previous branch attempt disagrees with its started record")
        registry = Path(bundle.environment["SHINOBI_OWNERSHIP_REGISTRY"])
        owner = WorkspaceRegistryStore(registry).owner(previous.attempt_id)
        if owner is None:
            continue
        if owner.kind != "local" or not Path(owner.workspace).is_relative_to(case.workspace.resolve()):
            raise BundleError("previous branch claim is outside its recorded workspace")
        reconcile_ownership(Path(owner.workspace), previous.attempt_id)


def run_branch(directory: Path, index: int):
    """Execute one scheduler invocation with durable, separate attempt records."""
    directory = directory.resolve()
    bundle = BranchArray.model_validate_json((directory / "branch-array.json").read_text())
    if not 0 <= index < len(bundle.cases):
        raise BundleError("branch index is out of range")
    case = bundle.cases[index]
    record = BranchRecord(
        workflow_id=bundle.workflow_id,
        bundle_digest=fingerprint(bundle),
        case=case.name,
        attempt_id=str(uuid4()),
        status="running",
        scheduler_job=os.environ.get("SLURM_JOB_ID"),
        restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        workspace=str(case.workspace.resolve()),
        submission=str(directory),
    )
    attempt = directory / "attempts" / str(index) / record.attempt_id
    write_new(attempt / "started.json", record)
    try:
        verify_worker_environment(bundle.worker)
        if source_tree_digest(directory / "source") != bundle.code.source_digest:
            raise BundleError("branch source differs from the staged bundle")
        for path, digest in bundle.images.items():
            if _file_digest(path) != digest:
                raise BundleError(f"branch image changed: {path}")
        os.environ.update(bundle.environment)
        case.workspace.mkdir(parents=True, exist_ok=True)
        write_new(case.workspace / ".shinobi/branch-attempts" / f"{record.attempt_id}.json", record)
        recover_branch_claims(case, record.attempt_id)
        os.chdir(case.workspace)
        scratch = case.workspace / ".shinobi/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ.update(TMPDIR=str(scratch), APPTAINER_TMPDIR=str(scratch), APPTAINER_CACHEDIR=str(scratch / "apptainer-cache"))
        import tempfile

        tempfile.tempdir = str(scratch)
        sys.path.insert(0, str(directory / "source"))
        builder = importlib.import_module(bundle.code.module)
        if not Path(builder.__file__).resolve().is_relative_to(directory / "source"):
            raise BundleError("branch builder resolved outside the staged source")
        for part in bundle.code.qualname.split("."):
            builder = getattr(builder, part)
        config = AppConfig.model_validate(bundle.config)
        config.cache.enabled = True
        config.cache.dir = str(case.workspace / ".shinobi/cache")
        config.sandbox.dir = str(case.workspace / ".shinobi/sandboxes")
        config.provenance.enabled = True
        config.provenance.dir = str(case.workspace / ".shinobi/provenance")
        recipe = builder(**case.parameters)
        if _local_images(recipe, config.backend.default) != set(bundle.case_images[case.name]):
            raise BundleError("branch image selection differs from the staged plan")
        if branch_declaration(recipe) != bundle.declarations[case.name]:
            raise BundleError("branch declaration differs from the staged DAG")
        result = recipe(**case.inputs, _config=config, _run_id=record.attempt_id, overwrite_steps=case.overwrite_steps)
        if not result.success:
            raise RuntimeError(result.stderr or "branch recipe failed")
    except BaseException as exc:
        write_new(attempt / "finished.json", record.model_copy(update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        raise
    write_new(attempt / "finished.json", record.model_copy(update={"status": "completed"}))
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    run_branch(args.directory, index)


if __name__ == "__main__":
    main()
