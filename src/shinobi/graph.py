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

from shinobi.steps.schema import InputRef, OutputRef

if TYPE_CHECKING:
    from shinobi.steps.schema import Recipe


class RecipeGraphError(ValueError):
    """A recipe's declared graph is invalid: a wiring reference points at a
    field/step that doesn't exist, a step name is duplicated, or the
    dependency edges form a cycle.
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
        for field, source in ref.wiring.items():
            if isinstance(source, InputRef):
                if source.field not in input_fields:
                    raise RecipeGraphError(
                        f"step '{ref.name}' wires input '{field}' from recipe "
                        f"input '{source.field}', which is not a field of "
                        f"{recipe.inputs_model.__name__}"
                    )
            elif isinstance(source, OutputRef):
                if source.step not in index:
                    raise RecipeGraphError(
                        f"step '{ref.name}' wires input '{field}' from the "
                        f"output of step '{source.step}', which does not exist "
                        f"in recipe '{recipe.name}'"
                    )
                dep = index[source.step]
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
