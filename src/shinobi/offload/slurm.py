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
from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, Recipe, path_fields


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


def _touched_paths(cab: Cab, resolved: dict[str, Any]) -> list[tuple[Path, bool]]:
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


def _overlaps(a: Path, b: Path) -> bool:
    """Whether two canonical paths can name the same bytes: equal, or one
    inside the other. Containment is not a nicety here -- a Measurement Set
    *is* a directory, so a step rewriting `/data/obs.ms` and a step reading
    `/data/obs.ms/CORRECTED` touch the same data while comparing unequal.
    """
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


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

    def order_after(self, name: str, cab: Cab, resolved: dict[str, Any]) -> set[str]:
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
            overlapping = [state for known, state in self._accesses.items() if _overlaps(path, known)]
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
    # In-place mutation of a shared path is invisible to `build_graph` (it
    # only sees wiring), so the edges it implies are derived here, from
    # resolved values, and merged into each job's `depends_on`.
    mutation_order = MutationOrder()
    step_index = {n: idx for idx, n in enumerate(graph.names)}

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
