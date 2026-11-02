"""Frozen ordinary Recipe graphs with allocation boundaries and local sub-DAGs.

Planning is pure. Submission stages immutable declarations, then exits. Each
allocation runs normal dispatch; scheduler dependencies carry the outer DAG.
Pending allocations deliberately do not reserve scientific paths.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, ValidationError

from shinobi.config import AppConfig
from shinobi.cache import recipe_output_keys
from shinobi.graph import build_graph
from shinobi.offload._codec import BundleError, WireModel, pack, unpack
from shinobi.offload.bundle import Binding, FrozenStep, RecipeBundle, ScopeSpec, Submission, walk_frozen, write_new
from shinobi.offload.code import capture_code
from shinobi.offload.records import AttemptRecord
from shinobi.steps.dispatch import _aggregate_scatter_results, _build_scatter_slices, _dispatch, _prepare_inputs
from shinobi.steps.pyfunc import PystepCallable
from shinobi.steps.schema import Cab, InputRef, OutputRef, Recipe, SlurmPlacement


def has_placement(recipe):
    return any(ref.placement is not None or isinstance(ref.step, Recipe) and has_placement(ref.step) for ref in recipe.steps)


def _reject_mixed_loops(recipe):
    for ref in recipe.steps:
        if ref.loop is not None:
            raise BundleError("mixed allocation loop boundaries are unsupported for local execution and compilation")
        if isinstance(ref.step, Recipe):
            _reject_mixed_loops(ref.step)


def freeze_mixed(recipe, inputs, *, config, workspace, code_roots=(), include_modules=(), cache=None, cache_dir=None):
    """Capture a recursively constructed graph without rebuilding it on nodes."""
    from shinobi.fits_access import scope_tree_has_fits_contract
    from shinobi.dataset_access import scope_tree_has_dataset_contract
    from shinobi.policies import EXECUTABLE_FLAVOURS

    _reject_mixed_loops(recipe)
    build_graph(recipe)
    if scope_tree_has_fits_contract(recipe):
        raise BundleError("authoritative FITS is unsupported in mixed allocation graphs")
    root = ScopeSpec.capture(recipe)

    def capture(ref, inherited_backend, inside=False):
        scope = ref.step
        spec = ScopeSpec.capture(scope)
        backend = scope.backend or inherited_backend
        if backend not in ("native", "apptainer"):
            raise BundleError(f"{ref.name}: mixed allocation leaves require native or Apptainer; nested scheduler/remote execution is forbidden")
        if inside and ref.placement is not None:
            raise BundleError(f"{ref.name}: nested Slurm placement is unsupported")
        if ref.loop:
            raise BundleError("mixed allocation loop boundaries are not yet supported")
        if ref.func is not None and not isinstance(ref.func, PystepCallable):
            raise BundleError(f"{ref.name}: arbitrary orchestration cannot be frozen")
        if isinstance(scope, Cab) and scope.flavour not in EXECUTABLE_FLAVOURS:
            raise BundleError(f"{ref.name}: unsupported cab flavour {scope.flavour!r}")
        if not isinstance(scope, (Cab, Recipe)) and not isinstance(ref.func, PystepCallable):
            raise BundleError(f"{ref.name}: opaque scope has no supported executable")
        fn = ref.func if isinstance(ref.func, PystepCallable) else None
        # Freeze inherited defaults so an independently dispatched subtree sees
        # exactly the enclosing recipe's effective policy.
        spec.settings["backend"] = pack(backend)
        children = tuple(capture(child, backend, True) for child in scope.steps) if isinstance(scope, Recipe) else None
        return FrozenStep(
            name=ref.name,
            scope=spec,
            params=pack(ref.params),
            wiring={k: tuple(Binding.capture(x) for x in v) if isinstance(v, list) else Binding.capture(v) for k, v in ref.wiring.items()},
            after=tuple(ref.after),
            backend=backend,
            scatter=ref.scatter,
            placement=ref.placement,
            children=children,
            output_wiring={k: Binding.capture(v) for k, v in scope.output_wiring.items()} if isinstance(scope, Recipe) else None,
            max_workers=scope.max_workers if isinstance(scope, Recipe) else None,
            code=capture_code(fn.__wrapped__, roots=tuple(code_roots), include=tuple(include_modules)) if fn else None,
            pystep_is_empty=fn.is_empty if fn else None,
            pystep_wants_ctx=fn.wants_ctx if fn else None,
        )

    bundle = RecipeBundle(
        schema_version=4,
        workspace=str(Path(workspace).resolve()),
        recipe=root,
        inputs=pack(_prepare_inputs(recipe, inputs)),
        config={k: pack(v) for k, v in config.model_dump(mode="python").items()},
        cache_override=cache,
        cache_dir_override=cache_dir,
        steps=tuple(capture(ref, recipe.backend or config.backend.default) for ref in recipe.steps),
        output_wiring={k: Binding.capture(v) for k, v in recipe.output_wiring.items()},
        max_workers=recipe.max_workers,
        execution_blocked_reason="MSv2 compute lifecycle preparation" if scope_tree_has_dataset_contract(recipe) else None,
    )
    plan_mixed(bundle)
    return bundle


class Unit(WireModel):
    step_index: int
    slice_index: int | None = None
    name: str
    known: JsonValue
    accesses: tuple[tuple[str, bool], ...]


class MixedPlan(WireModel):
    schema_version: int = 1
    bundle_digest: str
    units: tuple[Unit, ...]
    dependencies: tuple[tuple[int, ...], ...]


def plan_mixed(bundle):
    """Resolve bounded scatter cardinality and reject independent write overlap."""

    return plan_declared(
        bundle.declaration(),
        unpack(bundle.inputs),
        Path(bundle.workspace),
        bundle.digest,
        config=AppConfig.model_validate({k: unpack(v) for k, v in bundle.config.items()}),
        cache_dir=bundle.cache_dir_override,
    )


def plan_declared(recipe, values, workspace, digest="local", *, config=None, cache_dir=None):
    from shinobi.ownership import scope_path_accesses, _resolved_leaf_inputs
    from shinobi.offload.slurm import _require_static_write_declarations
    from shinobi.steps.schema import static_wiring_values, gathered_output_values, planned_output_values

    _reject_mixed_loops(recipe)
    config = config or AppConfig()
    graph = build_graph(recipe)
    outputs = {}
    units = []
    by_step = {}
    for index in graph.topological_indices():
        ref = recipe.steps[index]
        if isinstance(ref.step, Recipe) and has_placement(ref.step):
            raise BundleError("nested allocation placement is unsupported")
        if ref.placement:
            for key in ("nodes", "ntasks", "ntasks-per-node", "ntasks-per-core", "ntasks-per-socket", "N", "n"):
                if key in ref.placement.sbatch_opts and ref.placement.sbatch_opts[key] != "1":
                    raise BundleError("mixed drivers require exactly one task on one node")
            from shinobi.backends.slurm_script import build_sbatch_script

            build_sbatch_script(
                job_name="shinobi-plan",
                argv=["true"],
                sbatch_opts=ref.placement.sbatch_opts,
                stdout_path=Path("plan.out"),
                stderr_path=Path("plan.err"),
                chdir=str(workspace),
                error=BundleError,
            )
        if ref.placement and any(key in ref.placement.sbatch_opts for key in ("array", "dependency", "chdir", "output", "error", "job-name", "wrap")):
            raise BundleError("allocation placement cannot override generated array, dependency, workspace or log paths")
        known, deferred = static_wiring_values(ref, values, outputs)
        if ref.scatter:
            if set(ref.scatter.fields) & deferred:
                raise BundleError(f"{ref.name}: dynamic scatter cardinality is unsupported")
            slices = _build_scatter_slices(ref, known)
        else:
            slices = [known]
        by_step[index] = []
        for number, one in enumerate(slices):
            for leaf, leaf_inputs, unresolved in _resolved_leaf_inputs(ref.step, one, step_inputs={}, workspace=workspace, unresolved_inputs=deferred):
                try:
                    leaf.inputs_model(**leaf_inputs)
                except ValidationError as exc:
                    if any(error["type"] != "missing" or len(error["loc"]) != 1 or error["loc"][0] not in unresolved for error in exc.errors()):
                        raise BundleError(f"{ref.name}: invalid known leaf inputs: {exc}") from exc
                _require_static_write_declarations(ref.name, leaf, leaf_inputs, workspace=workspace)
            # Every inner scatter is also bounded, including empty scatters.
            _validate_static_scatter(ref.step, one)
            name = f"{recipe.name}.{ref.name}" + (f"[{number}]" if ref.scatter else "")
            from shinobi._required_cache import RequiredCacheAuthority

            unit_scope = ref.step.model_copy(update={"name": name})
            authority = RequiredCacheAuthority(unit_scope, {}, cache_dir or ref.step.cache_dir or recipe.cache_dir or config.cache.dir, workspace, resume_creates=True)
            with authority.activate():
                accesses, _ = scope_path_accesses(unit_scope, one, workspace=workspace)
            by_step[index].append(len(units))
            units.append(
                Unit(step_index=index, slice_index=number if ref.scatter else None, name=name, known=pack(one), accesses=tuple((str(path), write) for path, write in accesses))
            )
        if ref.scatter:
            outputs[ref.name] = gathered_output_values(ref.step, slices, unresolved_inputs=deferred, workspace=workspace)
        else:
            outputs[ref.name] = planned_output_values(ref.step, known, workspace=workspace, unresolved_inputs=deferred)
    ancestors = {}
    for index in graph.topological_indices():
        ancestors[index] = set(graph.deps[index])
        for parent in graph.deps[index]:
            ancestors[index].update(ancestors[parent])
    # Canonical paths sorted lexicographically form contiguous subtrees.
    # Compare only active ancestors, rather than every pair of the many
    # product-family members across every pair of allocations.
    active = []
    entries = sorted((Path(path).parts, index, writes) for index, unit in enumerate(units) for path, writes in unit.accesses)
    for parts, index, writes in entries:
        while active and parts[: len(active[-1][0])] != active[-1][0]:
            active.pop()
        second = units[index]
        for prior, other, prior_writes in active:
            first = units[other]
            if other == index or not (writes or prior_writes):
                continue
            if first.step_index in ancestors[second.step_index] or second.step_index in ancestors[first.step_index]:
                continue
            raise BundleError(f"independent allocations {first.name!r} and {second.name!r} overlap writable paths: {Path(*prior)}, {Path(*parts)}")
        active.append((parts, index, writes))
    return MixedPlan(bundle_digest=digest, units=tuple(units), dependencies=tuple(tuple(sorted(x)) for x in graph.deps))


def _validate_static_scatter(scope, inputs):
    from shinobi.steps.schema import static_wiring_values, planned_output_values, gathered_output_values

    if not isinstance(scope, Recipe):
        return
    values = dict(inputs)
    try:
        values = _prepare_inputs(scope, values)
    except Exception:
        pass  # deferred producer inputs are validated at execution
    outputs = {}
    for index in build_graph(scope).topological_indices():
        ref = scope.steps[index]
        known, deferred = static_wiring_values(ref, values, outputs)
        if ref.scatter:
            if set(ref.scatter.fields) & deferred:
                raise BundleError(f"{ref.name}: dynamic local scatter cardinality is unsupported")
            slices = _build_scatter_slices(ref, known)
            for one in slices:
                _validate_static_scatter(ref.step, one)
            outputs[ref.name] = gathered_output_values(ref.step, slices, unresolved_inputs=deferred)
        else:
            _validate_static_scatter(ref.step, known)
            outputs[ref.name] = planned_output_values(ref.step, known, workspace=Path.cwd(), unresolved_inputs=deferred)


class MixedExecution(WireModel):
    schema_version: int = 1
    workflow_id: UUID
    bundle_digest: str
    plan: MixedPlan
    worker: JsonValue
    images: dict[str, str]
    registry: str


class Invocation(WireModel):
    workflow_id: UUID
    bundle_digest: str
    name: str
    attempt_id: UUID = Field(default_factory=uuid4)
    restart: int = 0
    job_id: str | None = None
    submission: str | None = None
    logical_name: str | None = None


class MixedHandle(WireModel):
    engine: str = "slurm"
    mixed: bool = True
    recipe: str
    submission: str
    jobs: dict[str, str] = Field(default_factory=dict)
    finalizer: str | None = None
    launched_at: float = Field(default_factory=time.time)
    error: str | None = None


def prepare_mixed(bundle, *, submission_root, worker_python=None):
    from shinobi.offload.branch import _local_images, _file_digest
    from shinobi.offload.slurm import _stage_worker
    from shinobi.ownership import ownership_registry

    plan = plan_mixed(bundle)
    config = AppConfig.model_validate({k: unpack(v) for k, v in bundle.config.items()})
    images = _local_images(bundle.declaration(), config.backend.default)
    directory = bundle.stage(Path(submission_root))
    worker = _stage_worker(directory, Path(worker_python or sys.executable).absolute())
    submission = Submission.model_validate_json((directory / "submission.json").read_text())
    execution = MixedExecution(
        workflow_id=submission.workflow_id,
        bundle_digest=bundle.digest,
        plan=plan,
        worker=worker.model_dump(mode="json"),
        images={p: _file_digest(p) for p in images},
        registry=str(ownership_registry()),
    )
    write_new(directory / "mixed.json", execution)
    return directory


def _load(directory):
    bundle = RecipeBundle.read(directory / "bundle.json")
    execution = MixedExecution.model_validate_json((directory / "mixed.json").read_text())
    submission = Submission.model_validate_json((directory / "submission.json").read_text())
    if (
        execution.bundle_digest != bundle.digest
        or execution.plan.bundle_digest != bundle.digest
        or submission.bundle_digest != bundle.digest
        or submission.workflow_id != execution.workflow_id
    ):
        raise BundleError("mixed execution and immutable bundle identities disagree")
    return bundle, execution


def submit_mixed(directory):
    """One-shot submission with durable acceptance evidence and partial handles."""
    from shinobi.offload.slurm import build_sbatch_script, parse_sbatch_job_id

    directory = Path(directory).resolve()
    bundle, execution = _load(directory)
    write_new(directory / "submission-claim.json", Invocation(workflow_id=execution.workflow_id, bundle_digest=bundle.digest, name="submit"))
    worker = execution.worker
    handle = MixedHandle(recipe=bundle.declaration().name, submission=str(directory))
    (directory / "logs").mkdir(exist_ok=True)
    # A discoverable immutable pointer precedes the first scheduler request.
    # Monitoring reconstructs acceptance from jobs/*.json even if this process
    # dies before writing the final handle.
    write_new(directory / "launch.json", handle)
    write_new(Path(bundle.workspace) / ".shinobi" / str(execution.workflow_id) / "handle.json", handle)
    failure = None

    def submit(name, argv, options, parents, condition="afterok"):
        script = build_sbatch_script(
            job_name="shinobi-mixed",
            chdir=bundle.workspace,
            stdout_path=directory / f"logs/{name}-%A_%a.out",
            stderr_path=directory / f"logs/{name}-%A_%a.err",
            sbatch_opts={**options, "open-mode": "append", "kill-on-invalid-dep": "yes"},
            argv=["env", f"PYTHONPATH={worker['source']}", worker["python"], "-m", "shinobi.offload.mixed", str(directory), *argv],
            error=BundleError,
        )
        path = directory / f"{name}.sbatch"
        path.write_text(script)
        args = ["sbatch", "--parsable"]
        if parents:
            args.append(f"--dependency={condition}:" + ":".join(parents))
        # Persist intent before crossing the scheduler boundary. Ambiguous
        # acceptance is never automatically retried.
        write_new(directory / "jobs" / f"{name}-intent.json", Invocation(workflow_id=execution.workflow_id, bundle_digest=bundle.digest, name=name))
        process = subprocess.run([*args, str(path)], capture_output=True, text=True)
        if process.returncode:
            write_new(directory / "jobs" / f"{name}-rejected.json", Invocation(workflow_id=execution.workflow_id, bundle_digest=bundle.digest, name=name))
            raise BundleError(f"sbatch failed for {name}: {process.stderr.strip()}")
        job = parse_sbatch_job_id(process.stdout)
        write_new(directory / "jobs" / f"{name}.json", Invocation(workflow_id=execution.workflow_id, bundle_digest=bundle.digest, name=name, job_id=job))
        handle.jobs[name] = job
        return job

    completed = {}
    try:
        for index in build_graph(bundle.declaration()).topological_indices():
            ref = bundle.steps[index]
            placement = ref.placement or SlurmPlacement()
            options = dict(placement.sbatch_opts)
            if any(key in options for key in ("array", "dependency", "chdir", "output", "error")):
                raise BundleError("allocation placement cannot override generated array, dependency, workspace or log paths")
            indices = [i for i, unit in enumerate(execution.plan.units) if unit.step_index == index]
            parents = [completed[p] for p in execution.plan.dependencies[index]]
            if indices:
                options["array"] = f"0-{len(indices) - 1}%{placement.max_parallel}"
                job = submit(f"step-{index}", ["run", "--step", str(index)], options, parents)
                parents = [job]
            completed[index] = submit(f"gather-{index}", ["gather", "--step", str(index)], {}, parents)
    except BaseException as exc:
        failure = exc
        handle = handle.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})
    try:
        finalizer = submit("finalize", ["finalize"], {}, list(handle.jobs.values()), "afterany")
        handle = handle.model_copy(update={"finalizer": finalizer})
    except BaseException as exc:
        failure = failure or exc
        handle = handle.model_copy(update={"error": handle.error or f"{type(exc).__name__}: {exc}"})
    write_new(directory / "handle.json", handle)
    if failure is not None:
        raise BundleError(f"{handle.error}; accepted or uncertain jobs remain recoverable at {directory / 'handle.json'}") from failure
    return handle


def _latest(directory, name):
    records = []
    for path in (directory / "invocations" / name).glob("*.json"):
        record = Invocation.model_validate_json(path.read_text())
        if record.name != name or path.stem != str(record.attempt_id):
            raise BundleError("invocation identity mismatch")
        records.append(record)
    if not records:
        raise BundleError(f"{name}: no invocation record")
    records.sort(key=lambda record: (record.restart, str(record.attempt_id)))
    if len(records) > 1 and records[-1].restart == records[-2].restart:
        raise BundleError(f"{name}: ambiguous duplicate scheduler invocation")
    return records[-1]


def _parents(execution, bundle, name):
    if name == "finalize":
        return [f"gather-{index}" for index in range(len(bundle.steps))]
    if name.startswith("gather-"):
        step = int(name.split("-")[1])
        units = [f"unit-{step}-{index}" for index, _ in enumerate(unit for unit in execution.plan.units if unit.step_index == step)]
        return units or [f"gather-{index}" for index in execution.plan.dependencies[step]]
    step = int(name.split("-")[1])
    return [f"gather-{index}" for index in execution.plan.dependencies[step]]


def _logical_name(bundle, execution, name):
    if name == "finalize":
        return bundle.declaration().name
    if name.startswith("gather-"):
        return bundle.declaration().name + "." + bundle.steps[int(name.split("-")[1])].name
    _, step, task = name.split("-")
    return [unit for unit in execution.plan.units if unit.step_index == int(step)][int(task)].name


def _read_committed(directory, bundle, execution, name, *, expected_attempt_id=None):
    """Return the verified invocation together with exactly its result.

    Recursive dependencies are pinned to the IDs in the record, never rebound
    to whichever requeue happens to be current during another filesystem read.
    """
    invocation = _latest(directory, name)
    if expected_attempt_id is not None and invocation.attempt_id != expected_attempt_id:
        raise BundleError(f"{name}: committed result refers to a superseded predecessor")
    record = AttemptRecord.read(
        directory / "results" / str(invocation.attempt_id) / "final.json",
        workflow_id=execution.workflow_id,
        attempt_id=invocation.attempt_id,
        step_path=_logical_name(bundle, execution, name),
        bundle_digest=bundle.digest,
    )
    if not record.committed:
        raise BundleError(f"{name}: no successful current result")
    if set(record.source_attempts) != set(_parents(execution, bundle, name)):
        raise BundleError("result predecessor identities disagree with the frozen dependency graph")
    for parent, attempt_id in record.source_attempts.items():
        _read_committed(directory, bundle, execution, parent, expected_attempt_id=attempt_id)
    if _latest(directory, name) != invocation:
        raise BundleError(f"{name}: invocation changed while reading its committed result")
    return invocation, record.result(bundle.declaration())


def _read_result(directory, bundle, execution, name):
    return _read_committed(directory, bundle, execution, name)[1]


def _inputs(bundle, index, verified):
    from shinobi.cache import resolve_input_keys
    from shinobi.products import resolve_reference

    frozen = bundle.steps[index]
    ref = frozen.declaration()
    loaded = {}
    values = dict(unpack(frozen.params))
    root = unpack(bundle.inputs)

    def one(source):
        if isinstance(source, InputRef):
            return root[source.field]
        parent = next(i for i, step in enumerate(bundle.steps) if step.name == source.step)
        if source.step not in loaded:
            loaded[source.step] = verified[f"gather-{parent}"][1]
        return resolve_reference(getattr(loaded[source.step].outputs, source.field), source.selection)

    for field, source in ref.wiring.items():
        values[field] = [one(item) for item in source] if isinstance(source, list) else one(source)
    keys = resolve_input_keys(ref, {}, loaded)
    return values, keys if loaded else None


def _restore(directory, bundle, index):
    from shinobi.offload.worker import _callable

    # The existing callable loader takes a flat table. Its immutable source
    # indices are preorder indices used by RecipeBundle.stage.
    table = tuple(walk_frozen(bundle.steps))
    flat = bundle.model_copy(update={"steps": table})
    positions = {id(step): i for i, step in enumerate(table)}

    def restore(frozen):
        ref = frozen.declaration()
        if ref.step.image:
            from shinobi.offload.branch import _file_digest

            ref.step._mixed_image_digest = "sha256:" + _file_digest(ref.step.image)
        if frozen.code:
            ref.func = _callable(directory, positions[id(frozen)], flat)
            from dataclasses import replace

            ref.func = replace(ref.func, mixed_code_digest=frozen.code.digest)
        if frozen.children is not None:
            ref.step.steps = [restore(child) for child in frozen.children]
        return ref

    return restore(bundle.steps[index])


def _verify(execution, directory=None, bundle=None):
    from shinobi.offload.worker import WorkerEnvironment, verify_worker_environment
    from shinobi.offload.branch import _file_digest

    verify_worker_environment(WorkerEnvironment.model_validate(execution.worker))
    if directory is not None:
        from shinobi.offload.code import source_tree_digest

        for index, frozen in enumerate(walk_frozen(bundle.steps)):
            if frozen.code and source_tree_digest(directory / "code" / str(index)) != frozen.code.source_digest:
                raise BundleError("staged source differs from the immutable graph")
    for path, digest in execution.images.items():
        if _file_digest(path) != digest:
            raise BundleError(f"frozen allocation image changed: {path}")


def _recover(directory, bundle, execution, name, current):
    """Recover exact persisted branch invocations, including prior submissions."""
    import hashlib
    from shinobi.ownership import WorkspaceRegistryStore, reconcile_ownership

    logical = _logical_name(bundle, execution, name)
    root = Path(bundle.workspace) / ".shinobi/mixed-attempts" / hashlib.sha256(logical.encode()).hexdigest()
    for path in root.glob("*.json"):
        previous = Invocation.model_validate_json(path.read_text())
        if previous.attempt_id == current.attempt_id:
            continue
        if previous.logical_name != logical or not previous.submission or path.stem != str(previous.attempt_id):
            raise BundleError("previous invocation does not match its exact branch identity")
        prior_dir = Path(previous.submission)
        prior_bundle, prior_execution = _load(prior_dir)
        if previous.workflow_id != prior_execution.workflow_id or previous.bundle_digest != prior_bundle.digest or prior_bundle.workspace != bundle.workspace:
            raise BundleError("previous invocation does not match its immutable submission")
        started = Invocation.model_validate_json((prior_dir / "invocations" / previous.name / path.name).read_text())
        if started != previous or _logical_name(prior_bundle, prior_execution, previous.name) != logical:
            raise BundleError("previous invocation disagrees with its started record")
        owner = WorkspaceRegistryStore(Path(prior_execution.registry)).owner(str(previous.attempt_id))
        if owner is not None:
            if owner.kind != "local" or not owner.partitioned:
                raise BundleError("unexpected prior allocation owner kind")
            reconcile_ownership(Path(owner.workspace), str(previous.attempt_id))
    write_new(root / f"{current.attempt_id}.json", current)


def allocation_resources(config, environment=None):
    """Intersect local/configured limits with a single-task Slurm allocation.

    Slurm memory values are MiB; zero means unlimited and adds no cap.
    Per-CPU memory uses allocated CPUs, before any tighter local admission cap.
    """
    from shinobi.resources import Resources, detect_budget

    environment = os.environ if environment is None else environment
    for key in ("SLURM_NTASKS", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        if key in environment and int(environment[key]) != 1:
            raise BundleError("mixed drivers require exactly one task on one node")
    detected, _ = detect_budget()
    configured, _ = config.execution.resources.resolve()
    cpus = [value for value in (detected.cpus, configured.cpus) if value is not None]
    memory = [value for value in (detected.memory, configured.memory) if value is not None]
    allocated = environment.get("SLURM_CPUS_PER_TASK") or environment.get("SLURM_CPUS_ON_NODE")
    allocated = int(allocated) if allocated else None
    if allocated is not None:
        if allocated <= 0:
            raise BundleError("Slurm allocated CPU count must be positive")
        cpus.append(allocated)
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        if key in environment:
            amount = int(environment[key])
            if amount < 0:
                raise BundleError("Slurm allocated memory cannot be negative")
            if amount:
                if key == "SLURM_MEM_PER_CPU" and allocated is None:
                    raise BundleError("per-CPU Slurm memory needs an allocated CPU count")
                memory.append(amount * 1024**2 * (allocated if key == "SLURM_MEM_PER_CPU" else 1))
    return Resources(cpus=min(cpus) if cpus else None, memory=min(memory) if memory else None)


def current_manifest(directory):
    """Select only a current terminal finalizer's immutable attempt manifest.

    A newer started invocation invalidates old success immediately. A crash
    between manifest and terminal-record publication leaves no selectable
    manifest; a requeue publishes its own independent pair.
    """
    directory = Path(directory)
    bundle, execution = _load(directory)
    invocation = _latest(directory, "finalize")
    record = AttemptRecord.read(
        directory / "results" / str(invocation.attempt_id) / "final.json",
        workflow_id=execution.workflow_id,
        attempt_id=invocation.attempt_id,
        step_path=bundle.declaration().name,
        bundle_digest=bundle.digest,
    )
    if record.committed:
        _read_result(directory, bundle, execution, "finalize")
    elif record.state != "failed":
        raise BundleError("finalizer has no terminal result")
    from shinobi.provenance import load_manifest

    path = directory / "results" / str(invocation.attempt_id) / "manifest.json"
    manifest = load_manifest(path)
    if (manifest.returncode == 0) != record.committed:
        raise BundleError("finalizer manifest disagrees with its committed terminal record")
    if _latest(directory, "finalize") != invocation:
        raise BundleError("finalizer changed while selecting its manifest")
    return path


def execute(directory, operation, *, step=None, task=None):
    """One requeue-fenced invocation. No live outer orchestrator is required."""
    directory = Path(directory).resolve()
    bundle, execution = _load(directory)
    name = "finalize" if operation == "finalize" else f"gather-{step}" if operation == "gather" else f"unit-{step}-{task}"
    invocation = Invocation(
        workflow_id=execution.workflow_id,
        bundle_digest=bundle.digest,
        name=name,
        restart=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        job_id=os.environ.get("SLURM_JOB_ID"),
        submission=str(directory),
        logical_name=_logical_name(bundle, execution, name),
    )
    write_new(directory / "invocations" / name / f"{invocation.attempt_id}.json", invocation)
    lockpath = directory / "locks" / name
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_attempts = {}

        def fence():
            if _latest(directory, name) != invocation:
                raise BundleError("allocation invocation has been superseded")
            for parent, attempt_id in source_attempts.items():
                _read_committed(directory, bundle, execution, parent, expected_attempt_id=attempt_id)
            _verify(execution, directory, bundle)

        try:
            fence()
            os.environ["SHINOBI_OWNERSHIP_REGISTRY"] = execution.registry
            os.chdir(bundle.workspace)
            if operation == "run":
                _recover(directory, bundle, execution, name, invocation)
            parents = _parents(execution, bundle, name)
            verified = {parent: _read_committed(directory, bundle, execution, parent) for parent in parents}
            source_attempts = {parent: item[0].attempt_id for parent, item in verified.items()}
            fence()
            config = AppConfig.model_validate({k: unpack(v) for k, v in bundle.config.items()})
            from shinobi.logsetup import setup_file_logging

            log_directory = directory / "logs" / name / str(invocation.attempt_id)
            setup_file_logging(config.log.model_copy(update={"dir": str(log_directory), "file": "run.log"}))
            if operation == "run":
                values, keys = _inputs(bundle, step, verified)
                ref = _restore(directory, bundle, step)
                units = [unit for unit in execution.plan.units if unit.step_index == step]
                unit = units[task]
                if ref.scatter:
                    values = _build_scatter_slices(ref, values)[task]
                scope = ref.step.model_copy(update={"name": unit.name})
                from shinobi.resources import Budget

                total = allocation_resources(config)

                def precommit(result):
                    fence()

                result = _dispatch_unit(
                    unit,
                    scope,
                    ref.func,
                    _config=config,
                    _run_id=str(invocation.attempt_id),
                    _input_keys=keys,
                    _budget=Budget(total),
                    _result_precommit=precommit,
                    _recipe_cache=bundle.declaration().cache,
                    _recipe_cache_dir=bundle.declaration().cache_dir,
                    _recipe_sandbox=bundle.declaration().sandbox,
                    cache=bundle.cache_override,
                    cache_dir=bundle.cache_dir_override,
                    **values,
                )
            elif operation == "gather":
                ref = bundle.steps[step].declaration()
                units = [unit for unit in execution.plan.units if unit.step_index == step]
                children = [verified[f"unit-{step}-{i}"][1] for i in range(len(units))]
                if ref.scatter:
                    if children:
                        values = {
                            field: [getattr(child.inputs, field) for child in children] if field in ref.scatter.fields else getattr(children[0].inputs, field)
                            for field in ref.step.inputs_model.model_fields
                        }
                    else:
                        values, _ = _inputs(bundle, step, verified)
                    result = _aggregate_scatter_results(ref.step, ref.scatter.fields, values, children)
                else:
                    result = children[0]
            else:
                from shinobi.results import StepResult
                from shinobi.products import resolve_reference

                results = {ref.name: verified[f"gather-{i}"][1] for i, ref in enumerate(bundle.steps)}
                root = bundle.declaration()
                result = StepResult(
                    name=root.name,
                    returncode=0,
                    kind="recipe",
                    inputs=root.inputs_model(**unpack(bundle.inputs)),
                    outputs=root.outputs_model(
                        **{field: resolve_reference(getattr(results[source.step].outputs, source.field), source.selection) for field, source in root.output_wiring.items()}
                    ),
                    output_keys=recipe_output_keys(root, results),
                    sub_results=results,
                )
                from shinobi.provenance import build_manifest

                write_new(directory / "results" / str(invocation.attempt_id) / "manifest.json", build_manifest(result, backend="slurm"))
            fence()
            lifecycle_path = Path(bundle.workspace) / ".shinobi/dataset-attempts" / f"{invocation.attempt_id}.json"
            lifecycle = None
            if operation == "run" and lifecycle_path.exists():
                from shinobi.dataset_lifecycle import DatasetLifecycleStore

                lifecycle = DatasetLifecycleStore(lifecycle_path).attempt()
            record = AttemptRecord.from_result(
                result,
                workflow_id=execution.workflow_id,
                attempt_id=invocation.attempt_id,
                step_path=_logical_name(bundle, execution, name),
                bundle_digest=bundle.digest,
                job_id=invocation.job_id,
                composite=True,
                dataset_lifecycle=lifecycle,
            )
            (log_directory / "stdout.log").write_text(result.stdout)
            (log_directory / "stderr.log").write_text(result.stderr)
            record = record.model_copy(update={"source_attempts": source_attempts})
            for parent, attempt_id in source_attempts.items():
                if _latest(directory, parent).attempt_id != attempt_id:
                    raise BundleError("predecessor requeued before result publication")
            write_new(directory / "results" / str(invocation.attempt_id) / "final.json", record)
            if not result.success:
                raise BundleError(f"{name}: execution failed")
            return result
        except BaseException as exc:
            if operation == "finalize" and not (directory / "results" / str(invocation.attempt_id) / "manifest.json").exists():
                from shinobi.results import StepResult
                from shinobi.provenance import build_manifest

                root = bundle.declaration()
                partial = {}
                for index, ref in enumerate(bundle.steps):
                    try:
                        partial[ref.name] = _read_result(directory, bundle, execution, f"gather-{index}")
                    except (OSError, ValueError):
                        pass
                failed = StepResult(
                    name=root.name,
                    kind="recipe",
                    returncode=1,
                    inputs=root.inputs_model(**unpack(bundle.inputs)),
                    outputs=root.outputs_model.model_construct(),
                    sub_results=partial,
                    stderr=f"{type(exc).__name__}: {exc}",
                )
                write_new(directory / "results" / str(invocation.attempt_id) / "manifest.json", build_manifest(failed, backend="slurm"))
            path = directory / "results" / str(invocation.attempt_id) / "final.json"
            if not path.exists():
                write_new(
                    path,
                    AttemptRecord(
                        workflow_id=execution.workflow_id,
                        attempt_id=invocation.attempt_id,
                        step_path=_logical_name(bundle, execution, name),
                        bundle_digest=bundle.digest,
                        state="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("operation", choices=("run", "gather", "finalize"))
    parser.add_argument("--step", type=int)
    parser.add_argument("--task", type=int, default=None)
    args = parser.parse_args()
    task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    execute(args.directory, args.operation, step=args.step, task=task)


def _dispatch_unit(unit, scope, func, **kwargs):
    from shinobi.ownership import partitioned_claim

    with partitioned_claim(tuple((Path(path), writes) for path, writes in unit.accesses)):
        return _dispatch(scope, func, **kwargs)


def _local_code_identity(recipe):
    """Use the same frozen source identity on both sides of mixed dispatch."""
    from dataclasses import replace
    import inspect

    result = recipe.model_copy(deep=False)
    result.steps = []
    for original in recipe.steps:
        ref = original.model_copy(deep=False)
        ref.step = ref.step.model_copy(deep=False)
        if ref.step.image and Path(ref.step.image).is_file():
            from shinobi.offload.branch import _file_digest

            ref.step._mixed_image_digest = "sha256:" + _file_digest(ref.step.image)
        if isinstance(ref.step, Recipe):
            ref.step = _local_code_identity(ref.step)
        elif isinstance(ref.func, PystepCallable):
            function = ref.func.__wrapped__
            if "<locals>" not in function.__qualname__ and function.__module__ != "__main__":
                filename = Path(inspect.getsourcefile(function)).resolve()
                parts = function.__module__.split(".")
                root = filename.parent
                for _ in range(len(parts) if filename.name == "__init__.py" else len(parts) - 1):
                    root = root.parent
                ref.func = replace(ref.func, mixed_code_digest=capture_code(function, roots=(root,)).digest)
        result.steps.append(ref)
    return result


def run_local_mixed(recipe, inputs, *, config, budget=None, cache=None, cache_dir=None, backend=None, stream=None, provenance=None, sandbox=None, input_keys=None, target=None):
    """Run explicitly placed branches locally with independent lifecycle roots.

    The parent is only a scheduler/aggregator. Every branch uses ordinary root
    dispatch and one shared budget. Threads never change cwd or environment.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from shinobi.resources import Budget
    from shinobi.steps.dispatch import _resolve_wiring, _admission_demand
    from contextlib import ExitStack
    import heapq
    from shinobi.cache import resolve_input_keys
    from shinobi.results import StepResult
    from shinobi.products import resolve_reference
    from shinobi.provenance import build_manifest, run_manifest_path

    _reject_mixed_loops(recipe)
    recipe = _local_code_identity(recipe)
    prepared = _prepare_inputs(recipe, inputs)
    plan = plan_declared(recipe, prepared, Path.cwd(), config=config, cache_dir=cache_dir)
    budget = budget or Budget(config.execution.resources.resolve()[0])
    graph = build_graph(recipe)
    results = {}
    pending = set(range(len(recipe.steps)))
    workers = recipe.max_workers or config.execution.max_workers
    failure = None
    held = {}

    def drain_reservations():
        # Run before joining the pool, including scheduler-side exceptions.
        # A completed direct leaf may hold the budget a nested scheduler
        # needs in order to finish; joining first would deadlock them.
        budget.abandon()
        for future in list(held):
            if future.cancel():
                budget.release(held.pop(future))
        while held:
            done, _ = wait(held, return_when=FIRST_COMPLETED)
            for future in done:
                budget.release(held.pop(future))

    with ExitStack() as cleanup:
        pool = cleanup.enter_context(ThreadPoolExecutor(max_workers=workers))
        cleanup.callback(drain_reservations)
        running = {}
        groups = {}
        queued = []
        payloads = {}
        while pending or queued or running:
            for index in sorted(pending):
                if failure is not None or not all(recipe.steps[parent].name in results for parent in graph.deps[index]):
                    continue
                ref = recipe.steps[index]
                values = _resolve_wiring(ref, prepared, results)
                keys = resolve_input_keys(ref, input_keys or {}, results)
                output_wired = any(isinstance(source, OutputRef) for sources in ref.wiring.values() for source in (sources if isinstance(sources, list) else [sources]))
                if not output_wired and input_keys is None:
                    keys = None
                slices = _build_scatter_slices(ref, values) if ref.scatter else [values]
                groups[index] = (values, [None] * len(slices))
                for number, one in enumerate(slices):
                    name = f"{recipe.name}.{ref.name}" + (f"[{number}]" if ref.scatter else "")
                    scope = ref.step.model_copy(update={"name": name})
                    unit = next(unit for unit in plan.units if unit.step_index == index and unit.slice_index == (number if ref.scatter else None))
                    payloads[index, number] = (
                        unit,
                        scope,
                        ref.func,
                        dict(
                            _config=config.model_copy(deep=True),
                            _budget=budget,
                            _run_id=str(uuid4()),
                            _input_keys=keys,
                            _wired_fields=set(ref.wiring),
                            _recipe_backend=backend or recipe.backend,
                            _recipe_cache=recipe.cache,
                            _recipe_cache_dir=recipe.cache_dir,
                            _recipe_sandbox=recipe.sandbox,
                            cache=cache,
                            cache_dir=cache_dir,
                            stream=stream,
                            provenance=provenance,
                            sandbox=sandbox,
                            **one,
                        ),
                    )
                    heapq.heappush(queued, (index, number))
                pending.remove(index)
                if not slices:
                    results[ref.name] = _aggregate_scatter_results(ref.step, ref.scatter.fields, values, [])
            parked = False
            generation = 0
            while queued and len(running) < workers and failure is None:
                index, number = queued[0]
                unit, scope, func, kwargs = payloads[index, number]
                demand = _admission_demand(scope)
                granted, generation = budget.try_acquire(demand, scope.name)
                if not granted:
                    parked = True
                    break
                try:
                    future = pool.submit(_dispatch_unit, unit, scope, func, **kwargs)
                except BaseException:
                    budget.release(demand)
                    raise
                held[future] = demand
                running[future] = (index, number)
                heapq.heappop(queued)
                del payloads[index, number]
            if not parked:
                # Empty composite demands do not consume a refused atomic
                # unit's FIFO ticket. If composites fill the worker slots,
                # retaining that ticket blocks their leaves while we wait
                # for the composites themselves to finish.
                budget.abandon()
            if not running:
                if parked:
                    budget.wait_for_change(generation)
                    continue
                break
            # A nested scheduler may release budget before its composite
            # future finishes. Retry admission then instead of waiting for
            # that whole subtree while holding this scheduler's FIFO ticket.
            done, _ = wait(running, timeout=0.05 if parked else None, return_when=FIRST_COMPLETED)
            for future in done:
                index, number = running.pop(future)
                budget.release(held.pop(future))
                try:
                    result = future.result()
                    groups[index][1][number] = result
                    if not result.success:
                        failure = failure or BundleError(f"{result.name}: failed")
                except BaseException as exc:
                    failure = failure or exc
                values, children = groups[index]
                if all(child is not None for child in children):
                    ref = recipe.steps[index]
                    results[ref.name] = _aggregate_scatter_results(ref.step, ref.scatter.fields, values, children) if ref.scatter else children[0]
            if failure is not None:
                for future in list(running):
                    if future.cancel():
                        running.pop(future)
                        budget.release(held.pop(future))
                budget.abandon()
    if failure is not None:
        raise failure
    result = StepResult(
        name=recipe.name,
        kind="recipe",
        returncode=0,
        inputs=recipe.inputs_model(**prepared),
        outputs=recipe.outputs_model(**{field: resolve_reference(getattr(results[ref.step].outputs, ref.field), ref.selection) for field, ref in recipe.output_wiring.items()}),
        output_keys=recipe_output_keys(recipe, results),
        sub_results={ref.name: results[ref.name] for ref in recipe.steps},
        stdout="\n".join(results[ref.name].stdout for ref in recipe.steps if results[ref.name].stdout),
        stderr="\n".join(results[ref.name].stderr for ref in recipe.steps if results[ref.name].stderr),
    )
    if provenance if provenance is not None else config.provenance.enabled:
        build_manifest(result, backend=backend or recipe.backend or config.backend.default, target=target).write(run_manifest_path(config, recipe.name))
    return result


def _mixed_scheduler_states(jobs, execution):
    """Match logical array tasks, with live queue state ahead of accounting.

    A requeue can be pending before its new worker invocation is published.
    Accounting may still contain the previous completed attempt at that point.
    """
    expected = {}
    for name, job_id in jobs.items():
        if name.startswith("step-"):
            index = int(name.removeprefix("step-"))
            units = [unit for unit in execution.plan.units if unit.step_index == index]
            expected.update({f"{name}_{number}": f"{job_id}_{number}" for number, _ in enumerate(units)})
        else:
            expected[name] = job_id
    states = dict.fromkeys(expected, "UNKNOWN")
    if not jobs:
        return states
    ids = ",".join(dict.fromkeys(jobs.values()))
    try:
        accounting = subprocess.run(
            ["sacct", "-j", ids, "--format=JobID,State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
        )
        live = subprocess.run(
            ["squeue", "--jobs", ids, "--array", "--noheader", "--format=%i|%T"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return states
    # Failure to query the live controller cannot certify an old success.
    if live.returncode != 0:
        return states
    by_id = {job_id: name for name, job_id in expected.items()}
    for process in (accounting, live):
        if process.returncode != 0:
            continue
        for line in process.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0] in by_id:
                states[by_id[fields[0]]] = fields[1].strip()
    return states


def mixed_status(handle):
    from shinobi.offload.tracking import RunState, RUNNING, FINISHED, UNKNOWN, _SLURM_PENDING

    directory = Path(handle["submission"])
    states = {}
    detail = ""
    try:
        bundle, execution = _load(directory)
        if (directory / "handle.json").exists():
            handle = MixedHandle.model_validate_json((directory / "handle.json").read_text()).model_dump(mode="json")
        else:
            handle = dict(handle)
            handle["jobs"] = {
                path.stem: Invocation.model_validate_json(path.read_text()).job_id
                for path in (directory / "jobs").glob("*.json")
                if not path.stem.endswith(("-intent", "-rejected"))
            }
        uncertain = [
            path
            for path in (directory / "jobs").glob("*-intent.json")
            if not path.with_name(path.name.replace("-intent.json", ".json")).exists() and not path.with_name(path.name.replace("-intent.json", "-rejected.json")).exists()
        ]
        if uncertain:
            return RunState(UNKNOWN, detail="scheduler acceptance is uncertain; inspect " + ", ".join(str(path) for path in uncertain))
        states = _mixed_scheduler_states(handle.get("jobs", {}), execution)
        detail = ", ".join(f"{name}={state}" for name, state in states.items())
        if any(state.split()[0] in _SLURM_PENDING for state in states.values()):
            return RunState(RUNNING, detail=detail)
        invocation = _latest(directory, "finalize")
        record = AttemptRecord.read(
            directory / "results" / str(invocation.attempt_id) / "final.json",
            workflow_id=execution.workflow_id,
            attempt_id=invocation.attempt_id,
            step_path=bundle.declaration().name,
            bundle_digest=bundle.digest,
        )
        if record.committed:
            if any(state != "COMPLETED" for state in states.values()):
                return RunState(UNKNOWN, detail=detail + "; scheduler does not confirm completed execution")
            _read_result(directory, bundle, execution, "finalize")
        return RunState(FINISHED, exit_code=0 if record.committed else 1, detail=record.error or "immutable finalizer result")
    except (OSError, ValueError):
        pass
    return RunState(UNKNOWN, detail=detail + "; missing committed finalizer record")


def follow_mixed(handle, *, lines=40, wait=True, stop=None):
    """Follow immutable-attempt and scheduler logs on the shared filesystem."""
    directory = Path(handle["submission"]) / "logs"
    offsets = {}
    while True:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            with path.open(errors="replace") as stream:
                if path not in offsets:
                    content = stream.readlines()
                    emitted = content[-lines:] if lines else []
                else:
                    stream.seek(offsets[path])
                    emitted = stream.readlines()
                offsets[path] = stream.tell()
            for line in emitted:
                yield f"[{path.relative_to(directory)}] {line}"
        if not wait or stop is not None and stop.is_set():
            return
        if not mixed_status(handle).running:
            return
        if stop is not None:
            stop.wait(0.5)
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
