"""Short-circuit semantics for an unrolled loop (see `Recipe.add_loop`).

A loop is declared, not interpreted: `add_loop` flattens its body into the
recipe `max_iter` times and chains the copies with real wiring, so the graph
is fully inspectable before anything runs. What stays a run-time decision is
narrow -- whether an already-declared step does any work. This module is that
decision, and nothing else.

The rule lives here rather than in the scheduler because **two tiers evaluate
it**: `_run_recipe` calls `should_skip` in-process, and the Slurm offload
compiler emits the same predicate as a shell `[ -f ... ]` test at the top of
each iteration's script. A convergence signal that is a *path* is what makes
that possible -- a bool would have no way to cross a node boundary, and the
two tiers would need separate definitions that could drift into running a
different number of cycles for the same recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shinobi.results import StepResult
from shinobi.steps.schema import StepRef


def sentinel_exists(value: Any) -> bool:
    """Whether a convergence sentinel has actually been produced.

    Args:
        value: The sentinel output's value -- a path, or None if the
            producing step never set it.

    Returns:
        True if `value` names an existing file or directory.
    """
    return value is not None and Path(value).exists()


def sentinel_value(ref: StepRef, results: dict[str, StepResult]) -> Any:
    """The sentinel value this step's skip decision reads, or None if it
    cannot skip (the first iteration, or a step outside any loop).

    Args:
        ref: The step about to be scheduled.
        results: Completed steps by name. The sentinel producer is
            guaranteed to be present: `add_loop` gives every iteration an
            edge to it, so the scheduler cannot reach this step first.

    Returns:
        The previous iteration's sentinel output value, or None.
    """
    spec = ref.loop
    if spec is None or spec.sentinel_step is None:
        return None
    producer = results.get(spec.sentinel_step)
    if producer is None:
        return None
    return getattr(producer.outputs, spec.sentinel_field, None)


def should_skip(ref: StepRef, results: dict[str, StepResult]) -> bool:
    """Whether `ref` should pass its predecessor's outputs through instead
    of running, because an earlier iteration already converged.

    Args:
        ref: The step about to be scheduled.
        results: Completed steps by name.

    Returns:
        True if the previous iteration's sentinel exists on disk.
    """
    return sentinel_exists(sentinel_value(ref, results))


def passthrough_result(ref: StepRef, prev: StepResult, inputs: Any) -> StepResult:
    """The result of a step that was skipped: the *same body step's* outputs
    from one iteration earlier, handed on unchanged.

    Passing the previous `outputs` object through (rather than re-deriving
    field by field) is what makes convergence sticky without extra
    bookkeeping: the sentinel is itself one of those outputs, so every later
    iteration sees it and skips in turn.

    `cache_key`/`output_keys` are carried over too. A skipped step produced
    no new data, so contributing nothing would drop the upstream term from
    every downstream cache key (see `shinobi.cache`) and needlessly
    invalidate work that is genuinely unchanged.

    `kind` deliberately keeps the scope's real kind -- `skipped` is a
    separate flag. `shinobi.provenance.apply_manifest_pins` asserts that a
    record's `kind` still matches the scope's type, so inventing a
    "skipped" kind would make any early-converging run unreplayable.

    Args:
        ref: The skipped step.
        prev: The corresponding step's result from the previous iteration.
        inputs: The validated inputs this step would have run with.

    Returns:
        A successful `StepResult` marked `skipped`.
    """
    return StepResult(
        name=ref.name,
        returncode=0,
        outputs=prev.outputs,
        inputs=inputs,
        kind=prev.kind,
        skipped=True,
        cache_key=prev.cache_key,
        output_keys=prev.output_keys,
    )
