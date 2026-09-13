"""Compile a declared Recipe into a chain of dependency-linked Slurm jobs.

The whole graph is handed to Slurm as one dependency DAG
(`sbatch --dependency=afterok:<parents>`), so the cluster runs it with no
babysitting shinobi process -- the point of offload (survive a client
disconnect on a long HPC run). shinobi is a *compiler* here: it turns the
graph into sbatch scripts and submits them, then detaches.

Two halves, deliberately split by testability:

- `compile_slurm(...)` is **pure** -- recipe + inputs in, a `SlurmWorkflow`
  (scripts + declared dependencies) out. No cluster, no side effects; the
  golden-testable core.
- `submit_slurm(...)` shells out to `sbatch` and returns the job ids. It is
  **live-verified**: `tests/test_slurm_live.py` submits a dependency-chained
  workflow to a real `sbatch`/`sacct` on a disposable single-node cluster
  (`tests/slurm_live/README.md`), skipped unless that cluster is up. Real
  submission, `afterok` gating and `sacct` parsing are covered; multi-node
  scheduling and cross-node shared storage are not.

Only recipes that pass `check_offloadable` get here (no orchestration funcs,
inter-step data flow via shared-filesystem paths only), so every value the
compiler needs is statically knowable: an inter-step `OutputRef` path is
resolved from the producing step's same-named input or its output-field
default, mirroring `_fill_outputs` minus the backend run.

Having those resolved values is also what lets this module order **in-place
mutation**, which the declared graph cannot express: several steps taking
the same MS as a plain input and rewriting it are, to `build_graph`,
independent. `MutationOrder` derives the edges they actually need from the
resolved paths and merges them into each job's `afterok` dependencies --
see its docstring for the rules, and `graph.check_offloadable` for why the
whole MUTABLE class no longer has to be refused.
"""

from __future__ import annotations

import os
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_core import PydanticUndefined

from shinobi.backends.container import build_container_argv
from shinobi.backends.slurm_script import (
    build_sbatch_script,
    parse_sbatch_job_id,
    sacct_job_fields,
    safe_slurm_name,
    sbatch_resource_opts,
)
from shinobi.exceptions import BackendError
from shinobi.graph import check_offloadable
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, Recipe, Scope, path_fields, paths_overlap


class OffloadCompileError(ValueError):
    """A recipe passed `check_offloadable` but still can't be compiled to a
    concrete Slurm workflow -- e.g. an inter-step path can't be statically
    resolved, or a name isn't safe to write into a script.
    """


@dataclass
class SlurmJob:
    """One compiled step: the sbatch script to run it, and the step names
    it must run `afterok` of. `depends_on` is by step name; `submit_slurm`
    maps those to concrete job ids at submission time.
    """

    name: str
    script: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SlurmWorkflow:
    """A recipe compiled to a set of dependent sbatch jobs, ready to submit.

    Attributes:
        recipe: Name of the source recipe.
        jobs: Compiled `SlurmJob`s, in topological order.
        log_dir: Directory where each job's `--output`/`--error` land;
            created by `submit_slurm`.
    """

    recipe: str
    jobs: list[SlurmJob]  # in topological order
    log_dir: Path  # where each job's --output/--error land; created by submit


def _touched_paths(cab: Scope, resolved: dict[str, Any]) -> list[tuple[Path, bool]]:
    """Every path-typed input the step touches, as `(canonical_path,
    mutates)` -- `mutates` being whether the cab declares that field
    `Mutability.MUTABLE`, i.e. the tool rewrites the file in place rather
    than only reading it.

    Paths are canonicalised with `Path.resolve()` so `./obs.ms`,
    `obs.ms` and `/data/obs.ms` are recognised as one file rather than
    three, and so a symlink and its target don't look independent.
    List-valued inputs (e.g. `gaintable=[a, b]`) contribute each element.
    """
    touched: list[tuple[Path, bool]] = []
    for name in sorted(path_fields(cab.inputs_model)):
        value = resolved.get(name)
        if value is None:
            continue
        mutates = cab.mutability_of(name) is Mutability.MUTABLE
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if item is None:
                continue
            touched.append((Path(str(item)).resolve(), mutates))
    return touched


@dataclass
class _PathState:
    """Who last wrote a path, and who has read it since. Exactly what is
    needed to emit the *minimal* ordering edges rather than linking every
    pair that touches it: a chain of four steps mutating one MS becomes
    1->2->3->4, not all six pairwise edges.
    """

    last_writer: str | None = None
    readers_since_write: list[str] = field(default_factory=list)


class MutationOrder:
    """Tracks who has touched which paths, and reports the ordering a step
    needs because of it. Fed one step at a time, in declaration order, as
    the compiler resolves them.

    A recipe's declared graph only has an edge where one step *wires* an
    input from another's output. In-place mutation leaves no such trace:
    `flag`, `gaincal` and `applycal` all take the same MS as a plain input
    and rewrite it, so the DAG sees three independent steps. Locally the
    default `max_workers: 1` hides that behind declaration order; handed to
    Slurm as an unordered DAG it is data corruption. This is what makes the
    mutation order the recipe already relied on explicit.

    Ordering is emitted for a pair only when **at least one** of them
    declares the shared path MUTABLE -- so read-after-write and
    write-after-read, not just write-after-write. That breadth is the
    point: `applycal` mutates the MS while `wsclean` merely reads it, so
    restricting this to mutator-vs-mutator pairs would leave exactly the
    caracal-shaped case racing. Two steps that only read the same path need
    no ordering and get none.
    """

    def __init__(self) -> None:
        self._accesses: dict[Path, _PathState] = {}

    def order_after(self, name: str, cab: Scope, resolved: dict[str, Any]) -> set[str]:
        """Record `name`'s path accesses and return the already-seen steps
        it must run after.

        Args:
            name: The step's name.
            cab: Its cab, consulted for which inputs are paths and which of
                those are declared MUTABLE.
            resolved: Its fully-resolved inputs (defaults filled in), so
                the comparison is on real path values rather than on how
                each step happened to spell them.

        Returns:
            Names of previously-recorded steps this one must follow.
        """
        required: set[str] = set()
        for path, mutates in _touched_paths(cab, resolved):
            overlapping = [state for known, state in self._accesses.items() if paths_overlap(path, known)]
            for state in overlapping:
                if mutates and state.readers_since_write:
                    # Write-after-read: follow everyone who read the current
                    # contents, or this rewrites the file out from under a
                    # reader that is still running. Those readers already
                    # order after `last_writer` (that is how they were
                    # recorded), so depending on them covers it transitively
                    # -- naming the writer too would only add a redundant
                    # edge to every job's `--dependency` list.
                    required |= set(state.readers_since_write)
                elif state.last_writer is not None:
                    required.add(state.last_writer)

            state = self._accesses.setdefault(path, _PathState())
            if mutates:
                # This step is now the last writer of `path` *and* of every
                # overlapping path, so a later toucher of either orders
                # after it. Reads recorded before the write are satisfied.
                for s in [*overlapping, state]:
                    s.last_writer = name
                    s.readers_since_write = []
            else:
                state.readers_since_write.append(name)

        required.discard(name)  # a step reading and mutating the same path
        return required


def _static_outputs(cab: Scope, resolved_inputs: dict[str, Any]) -> dict[str, Any]:
    """The cab's output values knowable without running it: a same-named
    input passthrough, else the output field's declared default. (Wrangler-
    derived outputs are excluded by `check_offloadable`, so they never need
    to be resolved here.)
    """
    out: dict[str, Any] = {}
    for name, model_field in cab.outputs_model.model_fields.items():
        if name in resolved_inputs:
            out[name] = resolved_inputs[name]
        else:
            out[name] = None if model_field.default is PydanticUndefined else model_field.default
    return out


def _static_inputs(
    name: str,
    scope: Scope,
    ref,
    recipe_inputs: dict[str, Any],
    resolved_outputs: dict[str, dict[str, Any]],
    *,
    allow_runtime_values: bool = False,
) -> dict[str, Any]:
    """Resolve the compile-time subset of one step's effective inputs.

    The legacy argv compiler needs every value. The worker compiler permits
    ordinary runtime data flow, but a MUTABLE path still has to be known so
    the same in-place ordering edges can be derived before submission.
    """
    unresolved: set[str] = set()

    def one(step_field: str, source: InputRef | OutputRef) -> Any:
        if isinstance(source, InputRef):
            return recipe_inputs[source.field]
        value = resolved_outputs.get(source.step, {}).get(source.field)
        if value is None:
            if not allow_runtime_values or (step_field in path_fields(scope.inputs_model) and scope.mutability_of(step_field) is Mutability.MUTABLE):
                raise OffloadCompileError(
                    f"step '{name}' input '{step_field}' reads '{source.step}.{source.field}', "
                    "whose path isn't statically known at compile time -- supply it as an "
                    "input to the producing step"
                )
            unresolved.add(step_field)
        return value

    kwargs: dict[str, Any] = dict(ref.params)
    for step_field, source in ref.wiring.items():
        if isinstance(source, list):
            values = [one(step_field, item) for item in source]
            if step_field not in unresolved:
                kwargs[step_field] = values
        else:
            value = one(step_field, source)
            if step_field not in unresolved:
                kwargs[step_field] = value

    if not unresolved:
        validated = scope.inputs_model(**kwargs)
        return {field: getattr(validated, field) for field in scope.inputs_model.model_fields}

    # Fill only ordinary defaults; executing a default factory at compile
    # time would turn preparation into user-code execution.
    for field_name, model_field in scope.inputs_model.model_fields.items():
        if field_name not in kwargs and field_name not in unresolved and model_field.default is not PydanticUndefined:
            kwargs[field_name] = model_field.default
    return kwargs


def _script(
    cab: Cab,
    step_name: str,
    argv: list[str],
    workdir: str,
    sbatch_opts: dict[str, str],
    log_dir: Path,
    *,
    skip_if_exists: str | None = None,
) -> str:
    """Compile one step to an sbatch script.

    The job is named after the **step**, not its cab: a recipe may bind one
    cab to several steps (an unrolled loop always does), and a per-cab job
    name would point them all at the same `--output`/`--error` file to
    overwrite. The cab name is still charset-validated even though it is no
    longer interpolated -- it arrives from an untrusted YAML cab definition (see
    SECURITY.md), and that guarantee should not quietly lapse just because
    this particular use of it moved.
    """
    safe_slurm_name(cab.name, "cab name", error=OffloadCompileError)
    job_name = safe_slurm_name(step_name, "step name", error=OffloadCompileError)
    return build_sbatch_script(
        job_name=job_name,
        chdir=workdir,
        stdout_path=log_dir / f"{job_name}.out",
        stderr_path=log_dir / f"{job_name}.err",
        # `compile_slurm` passes one workflow-global `sbatch_opts` to every
        # job; merging the step's own declaration in here is what makes the
        # emitted allocation per-step. Explicit options still win.
        sbatch_opts={**sbatch_resource_opts(cab.resources), **sbatch_opts},
        argv=argv,
        error=OffloadCompileError,
        skip_if_exists=skip_if_exists,
    )


def compile_slurm(
    recipe: Recipe,
    inputs: dict[str, Any],
    *,
    workdir: str | None = None,
    container_runtime: str | None = "apptainer",
    sbatch_opts: dict[str, str] | None = None,
) -> SlurmWorkflow:
    """Compile `recipe` (with top-level `inputs`) into a `SlurmWorkflow`.

    Raises `RecipeNotOffloadableError` if the recipe isn't purely
    declarative, `ValidationError` if `inputs` (or any statically-resolved
    step inputs) don't validate, and `OffloadCompileError` if an inter-step
    path can't be resolved statically.
    """
    graph = check_offloadable(recipe)  # raises RecipeNotOffloadableError / RecipeGraphError
    workdir = workdir or os.getcwd()
    log_dir = Path(workdir) / ".shinobi" / safe_slurm_name(recipe.name, "recipe name", error=OffloadCompileError)
    sbatch_opts = sbatch_opts or {}

    validated_recipe = recipe.inputs_model(**inputs)
    recipe_inputs = {n: getattr(validated_recipe, n) for n in recipe.inputs_model.model_fields}

    resolved_outputs: dict[str, dict[str, Any]] = {}
    jobs: list[SlurmJob] = []
    # In-place mutation of a shared path is invisible to `build_graph` (it
    # only sees wiring), so the edges it implies are derived here, from
    # resolved values, and merged into each job's `depends_on`.
    mutation_order = MutationOrder()
    step_index = {n: idx for idx, n in enumerate(graph.names)}

    for i, name in enumerate(graph.names):
        ref = recipe.steps[i]
        cab = ref.step
        assert isinstance(cab, Cab)  # guaranteed by check_offloadable

        # Validate + fill defaults exactly as dispatch would, so the argv
        # matches a local run (and bad inputs fail here, before submission).
        resolved = _static_inputs(name, cab, ref, recipe_inputs, resolved_outputs)

        argv = build_argv(cab, resolved)  # inherits the non-"binary" flavour guard
        if cab.image and container_runtime:
            # Digest is discarded here -- offloaded-Slurm provenance is a follow-up.
            # runs_here=False: compiled here, executed on a compute node whose
            # cgroup delegation this host cannot see (see `_resource_flags`).
            argv, _ = build_container_argv(container_runtime, cab, argv, resolved, workdir, runs_here=False)

        own_outputs = _static_outputs(cab, resolved)

        # An unrolled loop iteration (Recipe.add_loop) short-circuits on the
        # previous iteration's sentinel, which is statically resolved above --
        # so the whole decision compiles into the script and shinobi is not in
        # the loop per step. Nothing else needs materialising on the skip
        # path: every carried path resolves identically in every iteration
        # (a body naming outputs per cycle can't be resolved statically at
        # all, and `resolve_one` rejects it), so a downstream job's compiled
        # argv already names the converged iteration's files.
        skip_if_exists: str | None = None
        if ref.loop is not None and ref.loop.sentinel_step is not None:
            sentinel = resolved_outputs.get(ref.loop.sentinel_step, {}).get(ref.loop.sentinel_field)
            if sentinel is None:
                raise OffloadCompileError(
                    f"step '{name}' belongs to loop '{ref.loop.loop}', whose sentinel "
                    f"'{ref.loop.sentinel_step}.{ref.loop.sentinel_field}' has no statically-known "
                    "path -- an offloaded loop's convergence signal must be a path the compiler can resolve"
                )
            skip_if_exists = str(sentinel)

        # Wiring edges from the declared graph, plus the ones implied by
        # steps sharing a path at least one of them mutates. Both are by
        # step name, and both become `--dependency=afterok` links below.
        # Kept in declaration order (not name order) so the emitted
        # dependency list stays stable and readable.
        #
        # `order_after` only ever returns steps it has already recorded, so
        # every edge it adds points backwards -- which is what lets
        # `submit_slurm` resolve each parent to a job id it has already
        # submitted. Checked rather than assumed, since a forward edge would
        # otherwise surface as a confusing KeyError at submission time.
        mutation_deps = mutation_order.order_after(name, cab, resolved)
        forward = [dep for dep in mutation_deps if step_index[dep] >= i]
        if forward:
            raise OffloadCompileError(f"internal: step '{name}' derived a forward mutation dependency on {forward} -- offloaded dependencies must point at earlier steps")
        depends_on = sorted({graph.names[d] for d in graph.deps[i]} | mutation_deps, key=lambda dep: step_index[dep])
        jobs.append(
            SlurmJob(
                name=name,
                script=_script(cab, name, argv, workdir, sbatch_opts, log_dir, skip_if_exists=skip_if_exists),
                depends_on=depends_on,
            )
        )
        resolved_outputs[name] = own_outputs

    return SlurmWorkflow(recipe=recipe.name, jobs=jobs, log_dir=log_dir)


def submit_slurm(workflow: SlurmWorkflow, *, workdir: str | None = None) -> dict[str, str]:
    """Submit a compiled workflow to Slurm and return {step name -> job id},
    then detach. Jobs are submitted in topological order with
    `--dependency=afterok` linking each to its parents' job ids.

    Live-verified against a real cluster by `tests/test_slurm_live.py`
    (see module docstring).
    """
    workdir = workdir or os.getcwd()
    # The compiled scripts write stdout/stderr into log_dir; Slurm fails a
    # job outright if it can't open those files, so the directory must exist
    # before submission (a live cluster caught this -- golden tests don't).
    workflow.log_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(tempfile.mkdtemp(prefix="shinobi-slurm-", dir=workdir))
    job_ids: dict[str, str] = {}
    try:
        for job in workflow.jobs:
            script_path = script_dir / f"{job.name}.sh"
            script_path.write_text(job.script)
            args = ["sbatch", "--parsable"]
            if job.depends_on:
                parents = ":".join(job_ids[dep] for dep in job.depends_on)
                args.append(f"--dependency=afterok:{parents}")
            proc = subprocess.run([*args, str(script_path)], capture_output=True, text=True)
            if proc.returncode != 0:
                raise BackendError(f"sbatch failed for step '{job.name}': {proc.stderr.strip()}")
            job_ids[job.name] = parse_sbatch_job_id(proc.stdout)
    finally:
        # sbatch reads each script synchronously during submission (the
        # subprocess.run call above blocks until it returns), so nothing
        # needs script_dir once this function is done -- remove it here
        # rather than leaking a `shinobi-slurm-*` tempdir into workdir forever.
        shutil.rmtree(script_dir, ignore_errors=True)
    return job_ids


def status_slurm(job_ids: dict[str, str]) -> dict[str, str]:
    """Query Slurm (`sacct`) once for each submitted job and return
    {step name -> state}. This is how a fresh `ninja status` invocation
    reconstructs a detached run's progress without any persistent process.

    Live-verified against a real cluster by `tests/test_slurm_live.py`
    (see module docstring).
    """
    states: dict[str, str] = {}
    for name, job_id in job_ids.items():
        proc = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
        )
        fields = sacct_job_fields(proc.stdout, job_id)
        states[name] = fields[1].strip() if fields and len(fields) >= 2 else "UNKNOWN"
    return states


# ---------------------------------------------------------------------------
# Experimental M1 worker workflow.  The legacy argv compiler above deliberately
# remains unchanged; callers opt into the frozen-bundle lifecycle explicitly.
# ---------------------------------------------------------------------------


@dataclass
class WorkerSlurmWorkflow:
    submission_dir: Path
    jobs: list[SlurmJob]
    finalizer: SlurmJob


@dataclass
class WorkerSlurmHandle:
    submission_dir: Path
    jobs: dict[str, str]
    finalizer_job: str | None


class WorkerSubmissionError(BackendError):
    """Slurm accepted only part of a worker workflow; ``handle`` is durable."""

    def __init__(self, message: str, handle: WorkerSlurmHandle):
        super().__init__(message)
        self.handle = handle


def _stage_worker(submission_dir: Path, worker_python: Path):
    """Freeze the installed Shinobi package that prepares this submission."""
    import shinobi

    from shinobi.backends.venv import digest_of_dists, freeze_dists
    from shinobi.offload.code import source_tree_digest
    from shinobi.offload.worker import WorkerEnvironment, worker_platform

    if not worker_python.is_absolute() or not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise OffloadCompileError(f"worker Python must be an existing executable absolute path, got {worker_python}")
    package = Path(shinobi.__file__).resolve().parent
    source_root = submission_dir / "worker-src"
    shutil.copytree(package, source_root / "shinobi", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    distributions = freeze_dists(worker_python)
    return WorkerEnvironment(
        python=str(worker_python),
        source=str(source_root),
        source_digest=source_tree_digest(source_root),
        python_version=platform.python_version(),
        platform=worker_platform(),
        distributions_digest=digest_of_dists(distributions) if distributions is not None else None,
    )


def provision_worker_venv(shared_root: Path, *, project_root: Path | None = None, python: Path | None = None) -> Path:
    """Provision an immutable, lock-derived worker dependency environment.

    Downloads/builds happen here on the submission host.  Compute jobs only
    execute the finished shared venv plus the separately digested staged
    Shinobi source tree.
    """
    project_root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    lock = project_root / "uv.lock"
    if not lock.is_file():
        raise OffloadCompileError(f"cannot provision a worker without {lock}")
    if shutil.which("uv") is None:
        raise OffloadCompileError("worker provisioning requires the 'uv' executable on the submission host")
    python = (python or Path(sys.executable)).absolute()
    identity = hashlib.sha256(lock.read_bytes() + f"\0{platform.python_version()}\0{sys.platform}\0{platform.machine()}".encode()).hexdigest()[:24]
    shared_root = shared_root.resolve()
    shared_root.mkdir(parents=True, exist_ok=True)
    final = shared_root / identity
    if (final / "bin" / "python").is_file():
        return final
    staging = Path(tempfile.mkdtemp(prefix=f".{identity}-", dir=shared_root))
    requirements = staging.parent / f".{identity}-{uuid4().hex}.requirements.txt"
    try:
        commands = [
            ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--output-file", str(requirements), "--directory", str(project_root)],
            ["uv", "venv", "--python", str(python), "--no-python-downloads", str(staging)],
            ["uv", "pip", "install", "--python", str(staging / "bin" / "python"), "--require-hashes", "-r", str(requirements)],
        ]
        for command in commands:
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode:
                raise OffloadCompileError(f"worker environment provisioning failed: {proc.stderr.strip()}")
        try:
            staging.rename(final)
        except FileExistsError:
            if not (final / "bin" / "python").is_file():
                raise
        return final
    finally:
        requirements.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _pin_worker_bundle(bundle):
    """Resolve execution environments before staging; workers only verify."""
    from shinobi.backends.container import CONTAINER_RUNTIMES, _pin_image, clear_image_pin_cache
    from shinobi.backends.venv import inspect_venv_digest
    from shinobi.offload._codec import pack, unpack

    # A programmatic caller may submit several bundles from one process. Keep
    # mutable tag resolution shared within this bundle, never across submits.
    clear_image_pin_cache()
    steps = []
    for step in bundle.steps:
        settings = dict(step.scope.settings)
        image = unpack(settings["image"]) if "image" in settings else None
        if image and step.backend in CONTAINER_RUNTIMES:
            pinned, digest = _pin_image(step.backend, image)
            if digest is None:
                raise OffloadCompileError(
                    f"step {step.name!r}: image {image!r} could not be pinned on the submission host; worker jobs never resolve mutable image tags on compute nodes"
                )
            settings["image"] = pack(pinned)
            step = step.model_copy(update={"scope": step.scope.model_copy(update={"settings": settings}), "image_digest": digest})
        if step.backend == "venv":
            # Submission is a new run boundary: inspect afresh instead of
            # reusing the within-run memo, so a long-lived compiler process
            # notices a provisioned environment update.
            digest = inspect_venv_digest(Path(step.tool_venv))
            if digest is None:
                raise OffloadCompileError(f"step {step.name!r}: tool venv {step.tool_venv!r} could not be fingerprinted on the submission host")
            step = step.model_copy(update={"tool_venv_digest": digest})
        steps.append(step)
    return bundle.model_copy(update={"steps": tuple(steps)})


def prepare_worker_slurm(
    bundle,
    *,
    submission_root: Path,
    worker_python: Path | None = None,
    sbatch_opts: dict[str, str] | None = None,
    step_sbatch_opts: dict[str, dict[str, str]] | None = None,
) -> WorkerSlurmWorkflow:
    """Stage a frozen bundle and compile one short-lived worker job per step.

    This is the side-effecting submission-preparation half: image pins and the
    worker source/environment are resolved here, never during ``freeze_recipe``
    and never on a compute node.
    """
    from shinobi.graph import build_graph
    from shinobi.offload._codec import unpack
    from shinobi.offload.bundle import RecipeBundle, write_new
    from shinobi.offload.worker import ExecutionPlan, PlannedAttempt

    if not isinstance(bundle, RecipeBundle):
        raise TypeError("prepare_worker_slurm expects a RecipeBundle")
    pinned = _pin_worker_bundle(bundle)
    submission_dir = pinned.stage(submission_root)
    worker = _stage_worker(submission_dir, (worker_python or Path(sys.executable)).absolute())
    submission = json.loads((submission_dir / "submission.json").read_text())
    attempts = tuple(PlannedAttempt(step_path=step.name, attempt_id=uuid4()) for step in pinned.steps)
    plan = ExecutionPlan(workflow_id=submission["workflow_id"], bundle_digest=pinned.digest, worker=worker, attempts=attempts)
    write_new(submission_dir / "execution.json", plan)

    recipe = pinned.declaration()
    graph = build_graph(recipe)
    step_index = {name: i for i, name in enumerate(graph.names)}
    recipe_inputs = unpack(pinned.inputs)
    mutation = MutationOrder()
    resolved_outputs: dict[str, dict[str, Any]] = {}
    options = dict(sbatch_opts or {})
    options.setdefault("kill-on-invalid-dep", "yes")
    per_step = step_sbatch_opts or {}
    unknown_steps = per_step.keys() - {step.name for step in pinned.steps}
    if unknown_steps:
        raise OffloadCompileError(f"step-specific sbatch options name unknown steps: {sorted(unknown_steps)}")
    log_dir = submission_dir / "logs"
    jobs: list[SlurmJob] = []
    for index, (frozen, attempt) in enumerate(zip(pinned.steps, attempts)):
        scope = frozen.scope.restore()
        known = _static_inputs(
            frozen.name,
            scope,
            frozen.declaration(),
            recipe_inputs,
            resolved_outputs,
            allow_runtime_values=True,
        )
        mutation_deps = mutation.order_after(frozen.name, scope, known)
        depends_on = sorted({graph.names[item] for item in graph.deps[index]} | mutation_deps, key=step_index.get)
        argv = [
            "env",
            f"PYTHONPATH={worker.source}",
            worker.python,
            "-m",
            "shinobi.offload.worker",
            "run",
            "--submission",
            str(submission_dir),
            "--step",
            frozen.name,
            "--attempt-id",
            str(attempt.attempt_id),
        ]
        script = build_sbatch_script(
            job_name=safe_slurm_name(frozen.name, "step name", error=OffloadCompileError),
            chdir=pinned.workspace,
            stdout_path=log_dir / f"{frozen.name}.out",
            stderr_path=log_dir / f"{frozen.name}.err",
            sbatch_opts={**sbatch_resource_opts(scope.resources), **options, **per_step.get(frozen.name, {})},
            argv=argv,
            error=OffloadCompileError,
        )
        jobs.append(SlurmJob(name=frozen.name, script=script, depends_on=depends_on))
        resolved_outputs[frozen.name] = _static_outputs(scope, known)

    final_argv = ["env", f"PYTHONPATH={worker.source}", worker.python, "-m", "shinobi.offload.worker", "finalize", "--submission", str(submission_dir)]
    finalizer = SlurmJob(
        name="finalize",
        script=build_sbatch_script(
            job_name=safe_slurm_name(f"{recipe.name}.finalize", "job name", error=OffloadCompileError),
            chdir=pinned.workspace,
            stdout_path=log_dir / "finalize.out",
            stderr_path=log_dir / "finalize.err",
            sbatch_opts=options,
            argv=final_argv,
            error=OffloadCompileError,
        ),
        depends_on=[step.name for step in pinned.steps],
    )
    return WorkerSlurmWorkflow(submission_dir=submission_dir, jobs=jobs, finalizer=finalizer)


def submit_worker_slurm(workflow: WorkerSlurmWorkflow) -> WorkerSlurmHandle:
    """Submit a worker workflow, durably recording every accepted job id."""
    from shinobi.offload.bundle import RecipeBundle, Submission, write_new
    from shinobi.offload.worker import ExecutionPlan, SubmittedFinalizer, SubmittedJob, WorkerHandleRecord

    directory = workflow.submission_dir
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    (directory / "jobs").mkdir(parents=True, exist_ok=True)
    submission = Submission.model_validate_json((directory / "submission.json").read_text())
    bundle = RecipeBundle.read(directory / "bundle.json")
    plan = ExecutionPlan.model_validate_json((directory / "execution.json").read_text())
    script_dir = Path(tempfile.mkdtemp(prefix="scripts-", dir=directory))
    job_ids: dict[str, str] = {}
    finalizer_id = None
    failure: BackendError | None = None
    try:
        for index, job in enumerate(workflow.jobs):
            script = script_dir / f"{index:04d}-{job.name}.sh"
            script.write_text(job.script)
            args = ["sbatch", "--parsable"]
            if job.depends_on:
                args.append(f"--dependency=afterok:{':'.join(job_ids[parent] for parent in job.depends_on)}")
            proc = subprocess.run([*args, str(script)], capture_output=True, text=True)
            if proc.returncode:
                failure = BackendError(f"sbatch failed for step '{job.name}': {proc.stderr.strip()}")
                break
            job_id = parse_sbatch_job_id(proc.stdout)
            job_ids[job.name] = job_id
            attempt = plan.attempt(job.name)
            write_new(
                directory / "jobs" / f"{index:04d}.json",
                SubmittedJob(workflow_id=submission.workflow_id, bundle_digest=bundle.digest, step_path=job.name, attempt_id=attempt.attempt_id, job_id=job_id),
            )

        # Always schedule recovery/finalization for whatever Slurm accepted.
        final_script = script_dir / "finalize.sh"
        final_script.write_text(workflow.finalizer.script)
        args = ["sbatch", "--parsable"]
        if job_ids:
            args.append(f"--dependency=afterany:{':'.join(job_ids.values())}")
        proc = subprocess.run([*args, str(final_script)], capture_output=True, text=True)
        if proc.returncode:
            if failure is None:
                failure = BackendError(f"sbatch failed for finalizer: {proc.stderr.strip()}")
        else:
            finalizer_id = parse_sbatch_job_id(proc.stdout)
            write_new(
                directory / "finalizer-job.json",
                SubmittedFinalizer(
                    workflow_id=submission.workflow_id,
                    bundle_digest=bundle.digest,
                    job_id=finalizer_id,
                ),
            )
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)
    handle = WorkerSlurmHandle(submission_dir=directory, jobs=job_ids, finalizer_job=finalizer_id)
    write_new(
        directory / "handle.json",
        WorkerHandleRecord(
            recipe=bundle.declaration().name,
            submission=str(directory),
            jobs=job_ids,
            finalizer=finalizer_id,
        ),
    )
    if failure is not None:
        raise WorkerSubmissionError(str(failure), handle)
    return handle
