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
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_core import PydanticUndefined

from shinobi.backends.container import build_container_argv
from shinobi.exceptions import BackendError
from shinobi.graph import build_graph, check_offloadable
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab, InputRef, OutputRef, Recipe

# A name that gets interpolated into a generated #SBATCH line or a job-name
# must not be able to smuggle in a newline (which would inject further
# directives/commands). Keep it to a strict, obviously-safe charset.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


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
    recipe: str
    jobs: list[SlurmJob]  # in topological order
    log_dir: Path  # where each job's --output/--error land; created by submit


def _safe(name: str, kind: str) -> str:
    if not _SAFE_NAME.match(name):
        raise OffloadCompileError(
            f"{kind} {name!r} contains characters unsafe for a Slurm script "
            f"(allowed: letters, digits, '.', '_', '-')"
        )
    return name


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


def _script(cab: Cab, argv: list[str], workdir: str, sbatch_opts: dict[str, str], log_dir: Path) -> str:
    job_name = _safe(cab.name, "cab name")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --chdir={workdir}",
        f"#SBATCH --output={log_dir / f'{job_name}.out'}",
        f"#SBATCH --error={log_dir / f'{job_name}.err'}",
    ]
    for key, value in sbatch_opts.items():
        lines.append(f"#SBATCH --{_safe(key, 'sbatch option')}={value}")
    lines.append("")
    lines.append(shlex.join(argv))  # exec-form argv; never a shell template
    return "\n".join(lines) + "\n"


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
    check_offloadable(recipe)  # raises RecipeNotOffloadableError / RecipeGraphError
    graph = build_graph(recipe)
    workdir = workdir or os.getcwd()
    log_dir = Path(workdir) / ".shinobi" / _safe(recipe.name, "recipe name")
    sbatch_opts = sbatch_opts or {}

    validated_recipe = recipe.inputs_model(**inputs)
    recipe_inputs = {n: getattr(validated_recipe, n) for n in recipe.inputs_model.model_fields}

    resolved_outputs: dict[str, dict[str, Any]] = {}
    jobs: list[SlurmJob] = []

    for i, name in enumerate(graph.names):
        ref = recipe.steps[i]
        cab = ref.step
        assert isinstance(cab, Cab)  # guaranteed by check_offloadable

        kwargs: dict[str, Any] = dict(ref.params)
        for step_field, source in ref.wiring.items():
            if isinstance(source, InputRef):
                kwargs[step_field] = recipe_inputs[source.field]
            elif isinstance(source, OutputRef):
                value = resolved_outputs[source.step][source.field]
                if value is None:
                    raise OffloadCompileError(
                        f"step '{name}' input '{step_field}' reads "
                        f"'{source.step}.{source.field}', whose path isn't statically "
                        "known at compile time (offloaded steps can't discover it at "
                        "run time) -- supply it as an input to the producing step"
                    )
                kwargs[step_field] = value

        # Validate + fill defaults exactly as dispatch would, so the argv
        # matches a local run (and bad inputs fail here, before submission).
        validated_step = cab.inputs_model(**kwargs)
        resolved = {n: getattr(validated_step, n) for n in cab.inputs_model.model_fields}

        argv = build_argv(cab, resolved)  # inherits the non-"binary" flavour guard
        if cab.image and container_runtime:
            argv = build_container_argv(container_runtime, cab, argv, resolved, workdir)

        depends_on = [graph.names[d] for d in sorted(graph.deps[i])]
        jobs.append(SlurmJob(name=name, script=_script(cab, argv, workdir, sbatch_opts, log_dir), depends_on=depends_on))
        resolved_outputs[name] = _static_outputs(cab, resolved)

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
        job_ids[job.name] = proc.stdout.strip().split(";")[0]
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
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.strip().splitlines()
        # first line is the job's overall state (later lines are steps like .batch)
        states[name] = lines[0].split("|")[0].strip() if lines else "UNKNOWN"
    return states
