"""The shared dependency graph of a Recipe's declared steps.

A `Recipe` is a declared DAG (see AGENTS.md): its `steps` list plus the
`OutputRef` wiring between them *is* the graph. `build_graph` reads that
graph once -- validating every wiring reference and rejecting cycles -- and
returns a `RecipeGraph` that BOTH consumers depend on:

- the executor (`shinobi.steps.dispatch._run_recipe`) schedules a
  topological wavefront over these *true* dependency edges;
- the `--dryrun` renderer (`shinobi.dag.graph_nodes`) builds its display
  view on top of the same edges (re-adding a display-only sequential chain
  between otherwise-independent steps).

Because both go through `build_graph`, they can never disagree on what
depends on what, or on whether the graph is even valid. Validation runs at
run time and at dryrun time -- not at build time, since a `Recipe` is
deliberately mutable through its builder methods and a forward `OutputRef`
is legitimate mid-construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shinobi.steps.schema import Cab, InputRef, Mutability, OutputRef, path_fields
from shinobi.wranglers import parse_output_action

if TYPE_CHECKING:
    from shinobi.steps.schema import Recipe


class RecipeGraphError(ValueError):
    """A recipe's declared graph is invalid: a wiring reference points at a
    field/step that doesn't exist, a step name is duplicated, or the
    dependency edges form a cycle.
    """


class RecipeNotOffloadableError(ValueError):
    """A recipe is a valid graph but cannot be compiled to an external
    workflow engine and detached: it relies on shinobi's live, in-process
    execution semantics. See `check_offloadable` for the exact rules and
    AGENTS.md's "DAG offload" section for why. Such a recipe should run
    locally (the default `ctx.run()` path) instead -- it is never silently
    degraded.
    """


@dataclass(frozen=True)
class RecipeGraph:
    """The validated dependency graph. Node ids are indices into the
    recipe's `steps` list (so id == declaration order).

    `deps[i]` are the ids step `i` *truly* depends on (via `OutputRef`
    wiring) -- no artificial sequential chaining; independent steps have
    empty `deps`. `dependents` is the reverse edge set, precomputed for the
    scheduler's in-degree bookkeeping.
    """

    names: list[str]
    deps: list[set[int]]
    dependents: list[set[int]]


def build_graph(recipe: "Recipe") -> RecipeGraph:
    """Validate a Recipe's wiring and build its dependency graph.

    Raises `RecipeGraphError` on: a duplicate step name; an `InputRef` to a
    field not on the recipe's `inputs_model`; an `OutputRef` (in a step's
    wiring or in `output_wiring`) to an unknown step; or a dependency cycle.
    """
    names = [ref.name for ref in recipe.steps]
    index: dict[str, int] = {}
    for i, name in enumerate(names):
        if name in index:
            raise RecipeGraphError(
                f"recipe '{recipe.name}' has more than one step named '{name}'"
            )
        index[name] = i

    input_fields = set(recipe.inputs_model.model_fields)
    deps: list[set[int]] = [set() for _ in names]
    dependents: list[set[int]] = [set() for _ in names]

    for i, ref in enumerate(recipe.steps):
        if ref.scatter is not None:
            step_inputs = set(ref.step.inputs_model.model_fields)
            for field in ref.scatter.fields:
                if field not in step_inputs:
                    raise RecipeGraphError(
                        f"step '{ref.name}' declares scatter over '{field}', which is not a "
                        f"field of {ref.step.inputs_model.__name__}"
                    )
        for field, source in ref.wiring.items():
            sources = source if isinstance(source, list) else [source]
            for one_source in sources:
                if isinstance(one_source, InputRef):
                    if one_source.field not in input_fields:
                        raise RecipeGraphError(
                            f"step '{ref.name}' wires input '{field}' from recipe "
                            f"input '{one_source.field}', which is not a field of "
                            f"{recipe.inputs_model.__name__}"
                        )
                elif isinstance(one_source, OutputRef):
                    if one_source.step not in index:
                        raise RecipeGraphError(
                            f"step '{ref.name}' wires input '{field}' from the "
                            f"output of step '{one_source.step}', which does not "
                            f"exist in recipe '{recipe.name}'"
                        )
                    dep = index[one_source.step]
                    deps[i].add(dep)
                    dependents[dep].add(i)

    for field, source in recipe.output_wiring.items():
        if source.step not in index:
            raise RecipeGraphError(
                f"recipe '{recipe.name}' output '{field}' is wired from step "
                f"'{source.step}', which does not exist"
            )

    _check_acyclic(recipe.name, names, deps, dependents)
    return RecipeGraph(names=names, deps=deps, dependents=dependents)


def _check_acyclic(
    recipe_name: str, names: list[str], deps: list[set[int]], dependents: list[set[int]]
) -> None:
    """Kahn's algorithm: drain zero-in-degree nodes; any that never drain
    are part of (or downstream of) a cycle.
    """
    indeg = [len(d) for d in deps]
    ready = [i for i, d in enumerate(indeg) if d == 0]
    drained = 0
    while ready:
        i = ready.pop()
        drained += 1
        for d in dependents[i]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    if drained != len(names):
        stuck = sorted(names[i] for i, d in enumerate(indeg) if d > 0)
        raise RecipeGraphError(
            f"recipe '{recipe_name}' has a dependency cycle involving: "
            f"{', '.join(stuck)}"
        )


def _wrangler_output_fields(cab: Cab) -> set[str]:
    """Output field names a cab fills by parsing stdout/stderr at run time
    (its `PARSE_OUTPUT` wranglers). An offloaded step's console output is
    never seen by shinobi, so a downstream consumer of such a field would
    get nothing -- hence these are not offloadable when consumed.
    """
    fields: set[str] = set()
    for actions in cab.wranglers.values():
        for action in actions:
            parsed = parse_output_action(action)
            if parsed is not None:
                fields.add(parsed[0])
    return fields


def check_offloadable(recipe: "Recipe") -> RecipeGraph:
    """Raise `RecipeNotOffloadableError` (with *all* disqualifying reasons)
    unless `recipe` is a purely declarative DAG that can be compiled to an
    external engine and detached. A valid graph is a precondition, so this
    calls `build_graph` first (which may raise `RecipeGraphError`) and
    returns it, so a caller that needs the graph right after checking
    eligibility (e.g. `offload.slurm.compile_slurm`) doesn't have to call
    `build_graph` a second time.

    The rules follow directly from "the cluster runs the graph, shinobi is
    not in the loop per step" (see AGENTS.md / the design note):

    - **No orchestration function** on any step -- arbitrary Python run
      against live upstream outputs can't be statically compiled, and
      shipping it would be a code-execution hazard.
    - **Every step is a `Cab`** -- a bare `Scope`/nested `Recipe` step has
      no single argv to emit.
    - **No MUTABLE inputs** -- pass-by-reference is a single-heap concept;
      there is no shared memory across cluster nodes.
    - **Every inter-step `OutputRef`** (one step's input wired from another
      step's output) carries a **path** output that is **not**
      wrangler-derived -- across nodes, data flows only as shared-filesystem
      paths, and a wrangler output is never populated when shinobi doesn't
      see the step's stdout. `output_wiring` (the recipe's own external
      outputs) is deliberately *not* checked: it is a reporting boundary,
      not inter-node data flow -- under detach the caller gets a handle, and
      a downstream `ninja status` reconstructs what it can (paths yes,
      dynamic values best-effort).
    """
    graph = build_graph(recipe)  # valid graph is a precondition
    reasons: list[str] = []
    by_name = {ref.name: ref for ref in recipe.steps}

    for ref in recipe.steps:
        scope = ref.step
        if ref.scatter is not None:
            reasons.append(
                f"step '{ref.name}' declares scatter over {ref.scatter.fields} -- "
                "scatter is not supported by offloaded engines in this version"
            )
        if ref.func is not None:
            reasons.append(
                f"step '{ref.name}' has an orchestration function -- run it locally"
            )
        if not isinstance(scope, Cab):
            reasons.append(
                f"step '{ref.name}' is a {type(scope).__name__}, not a Cab -- only "
                "atomic Cab steps can be compiled to an external workflow"
            )
            continue
        mutable = sorted(
            name
            for name, m in scope.input_mutability.items()
            if m is Mutability.MUTABLE
        )
        if mutable:
            reasons.append(
                f"step '{ref.name}' has MUTABLE input(s) {mutable} -- pass-by-"
                "reference cannot cross node boundaries"
            )

    def check_output_ref(label: str, src: OutputRef) -> None:
        """Append a reason to `reasons` if `src` can't cross a node boundary.

        Args:
            label: Human-readable description of what references `src`,
                used in the reported reason.
            src: The output reference to check (wrangler-derived and
                non-path outputs are ineligible for offload).
        """
        producer = by_name.get(src.step)
        if producer is None or not isinstance(producer.step, Cab):
            return  # unknown/non-Cab producer already reported elsewhere
        cab = producer.step
        if src.field in _wrangler_output_fields(cab):
            reasons.append(
                f"{label} reads wrangler-derived output '{src.step}.{src.field}' -- "
                "not populated when the step runs offloaded"
            )
        elif src.field not in path_fields(cab.outputs_model):
            reasons.append(
                f"{label} reads non-path output '{src.step}.{src.field}' -- only "
                "filesystem paths can flow between offloaded steps"
            )

    for ref in recipe.steps:
        for field, source in ref.wiring.items():
            sources = source if isinstance(source, list) else [source]
            for one_source in sources:
                if isinstance(one_source, OutputRef):
                    check_output_ref(f"step '{ref.name}' input '{field}'", one_source)

    if reasons:
        raise RecipeNotOffloadableError(
            f"recipe '{recipe.name}' cannot be offloaded to an external engine:\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )
    return graph
