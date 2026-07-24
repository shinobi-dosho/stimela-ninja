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
- `submit_slurm(...)` shells out to `sbatch` and returns the job ids. Like
  `shinobi.backends.slurm`, it is **not verified against a real cluster**
  (none in the dev env) -- reviewed by construction; verify before relying.

Only recipes that pass `check_offloadable` get here (no orchestration funcs,
no MUTABLE inputs, inter-step data flow via shared-filesystem paths only),
so every value the compiler needs is statically knowable: an inter-step
`OutputRef` path is resolved from the producing step's same-named input or
its output-field default, mirroring `_fill_outputs` minus the backend run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from shinobi.steps.schema import Cab, InputRef, OutputRef, Recipe


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


def _static_outputs(cab: Cab, resolved_inputs: dict[str, Any]) -> dict[str, Any]:
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
    longer interpolated -- it arrives from untrusted cult-cargo YAML (see
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

    for i, name in enumerate(graph.names):
        ref = recipe.steps[i]
        cab = ref.step
        assert isinstance(cab, Cab)  # guaranteed by check_offloadable

        def resolve_one(step_field: str, source: InputRef | OutputRef) -> Any:
            """Resolve one step input to its statically-known value.

            Args:
                step_field: Name of the input field being resolved, used
                    in the error message if resolution fails.
                source: Where the value comes from -- either the recipe's
                    own inputs (`InputRef`) or a prior step's output
                    (`OutputRef`).

            Returns:
                The resolved value.

            Raises:
                OffloadCompileError: If `source` is an `OutputRef` whose
                    value isn't statically known at compile time.
            """
            if isinstance(source, InputRef):
                return recipe_inputs[source.field]
            value = resolved_outputs[source.step][source.field]
            if value is None:
                raise OffloadCompileError(
                    f"step '{name}' input '{step_field}' reads "
                    f"'{source.step}.{source.field}', whose path isn't statically "
                    "known at compile time (offloaded steps can't discover it at "
                    "run time) -- supply it as an input to the producing step"
                )
            return value

        kwargs: dict[str, Any] = dict(ref.params)
        for step_field, source in ref.wiring.items():
            if isinstance(source, list):
                kwargs[step_field] = [resolve_one(step_field, s) for s in source]
            else:
                kwargs[step_field] = resolve_one(step_field, source)

        # Validate + fill defaults exactly as dispatch would, so the argv
        # matches a local run (and bad inputs fail here, before submission).
        validated_step = cab.inputs_model(**kwargs)
        resolved = {n: getattr(validated_step, n) for n in cab.inputs_model.model_fields}

        argv = build_argv(cab, resolved)  # inherits the non-"binary" flavour guard
        if cab.image and container_runtime:
            # Digest is discarded here -- offloaded-Slurm provenance is a follow-up.
            argv, _ = build_container_argv(container_runtime, cab, argv, resolved, workdir)

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

        depends_on = [graph.names[d] for d in sorted(graph.deps[i])]
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

    NOT verified against a real cluster (see module docstring).
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

    NOT verified against a real cluster (see module docstring).
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
