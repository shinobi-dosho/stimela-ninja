"""Static execution-graph *renderer* for `ninja run --dryrun`.

Recipes are declared DAGs (see AGENTS.md): a `Recipe`'s `steps` list plus
its wiring already *is* the graph. The graph itself -- nodes, true
dependency edges, cycle/wiring validation -- is built by
`shinobi.graph.build_graph`, shared with the executor so the two can never
disagree. `graph_nodes(recipe)` adapts that graph into `TraceStep` nodes for
*display*, and `render_dag` draws them as an indented tree:

    make
    ├── sub  (recipe)
    │   ├── sub.a
    │   └── sub.b
    └── after
        └── merge  (also waits on: sub)

A step sits under the step it waits on, so every branch is a real fan-out.
A step that waits on several is drawn once, under the last of them in
declaration order, and names the others in an "also waits on" note -- a DAG
is not a tree, and the note is what keeps the drawing honest about that. A
nested `Recipe` step is a branch of its own: its steps hang beneath it under
dotted names (`sub.a`), and whatever waits on the whole recipe hangs beneath
it too, without the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shinobi.steps.schema import Recipe


@dataclass
class TraceStep:
    """A recipe step node for display purposes.

    Attributes:
        id: Index of the node in `graph_nodes`'s output -- declaration
            order, with a nested recipe's own steps right after it.
        name: The step's name; dotted (`sub.a`) inside a nested recipe.
        depends_on: IDs of the steps, at the same nesting level, this one
            waits on.
        recipe: Whether the step is a nested `Recipe`.
        group: ID of the nested-recipe node this step belongs to, or None at
            the top level.
    """

    id: int
    name: str
    depends_on: set[int] = field(default_factory=set)
    # The step's declared resource footprint, rendered beside its name.
    # `--dryrun` shows what is *declared*, and a footprint is one of the few
    # declarations that changes the shape of a run rather than its content:
    # it is what turns parallel branches into a queue. Seeing that before
    # running, rather than inferring it from timings afterwards, is the
    # whole point of the dry run.
    resources: str = ""
    access_reasons: tuple[str, ...] = ()
    recipe: bool = False
    group: int | None = None


def graph_nodes(recipe: "Recipe", inputs: dict[str, Any] | None = None, *, workspace: Path | None = None) -> list[TraceStep]:
    """Build the display graph from a Recipe's validated dependency graph.

    Uses the shared `build_graph`, so a cyclic or mis-wired recipe raises
    here exactly as it would at execution time, nested recipes included.
    A nested `Recipe` step is followed by its own steps, each in that
    recipe's `group`, with edges among themselves only: the executor runs
    the sub-recipe as one node, so its steps wait on whatever the node waits
    on, and its dependents wait on the node.
    """
    from shinobi.dataset_access import plan_recipe_accesses
    from shinobi.graph import build_graph

    plan = plan_recipe_accesses(recipe, inputs, workspace=workspace) if inputs is not None else None
    graph = plan.graph if plan is not None else build_graph(recipe)
    nodes: list[TraceStep] = []
    _place(recipe, graph, "", None, nodes, plan)
    return nodes


def _place(recipe: "Recipe", graph, prefix: str, group: int | None, nodes: list[TraceStep], plan=None) -> None:
    """Append `recipe`'s steps to `nodes` in declaration order, each nested
    recipe followed by its own steps, then add the edges between them.

    `plan` applies to the top level only, which is all `plan_recipe_accesses`
    resolves.
    """
    from shinobi.graph import build_graph
    from shinobi.steps.schema import Recipe

    ids: list[int] = []
    for i, name in enumerate(graph.names):
        scope = recipe.steps[i].step
        access_reasons: tuple[str, ...] = ()
        if plan is not None:
            access_reasons = tuple(
                reason for parent in sorted((graph.names[item] for item in graph.deps[i]), key=graph.names.index) for reason in plan.reasons.get((name, parent), ())
            )
        declared = scope.resources
        node = TraceStep(
            id=len(nodes),
            name=f"{prefix}{name}",
            resources=declared.describe() if declared else "",
            access_reasons=access_reasons,
            recipe=isinstance(scope, Recipe),
            group=group,
        )
        nodes.append(node)
        ids.append(node.id)
        if node.recipe and scope.steps:
            _place(scope, build_graph(scope), f"{node.name}.", node.id, nodes)
    for i, deps in enumerate(graph.deps):
        nodes[ids[i]].depends_on = {ids[dep] for dep in deps}


def _label(step: TraceStep, by_id: dict[int, TraceStep], parent: int | None) -> str:
    """A step's line: its name, then anything that changes how it runs."""
    others = sorted(dep for dep in step.depends_on if dep != parent)
    details = [
        *(["recipe"] if step.recipe else []),
        *([step.resources] if step.resources else []),
        *step.access_reasons,
        *([f"also waits on: {', '.join(by_id[dep].name for dep in others)}"] if others else []),
    ]
    return step.name if not details else f"{step.name}  ({'; '.join(details)})"


def render_dag(steps: list[TraceStep]) -> str:
    """Draw a recipe's execution graph as an indented tree.

    Args:
        steps: Display nodes, as produced by `graph_nodes`.

    Returns:
        A multi-line string with the rendered tree, or a placeholder
        message if `steps` is empty.
    """
    if not steps:
        return "(no steps traced)"

    by_id = {step.id: step for step in steps}
    parent_of: dict[int, int | None] = {}
    children: dict[int, list[TraceStep]] = {step.id: [] for step in steps}
    roots: list[TraceStep] = []
    for step in steps:
        # The last thing it waits on, in declaration order: by the time that
        # one is drawn, every earlier dependency already has been.
        parent = max(step.depends_on) if step.depends_on else step.group
        parent_of[step.id] = parent
        if parent is None:
            roots.append(step)
        else:
            children[parent].append(step)

    lines: list[str] = []

    def draw(step: TraceStep, indent: str) -> None:
        kids = sorted(children[step.id], key=lambda kid: kid.id)
        for position, kid in enumerate(kids):
            last = position == len(kids) - 1
            lines.append(f"{indent}{'└── ' if last else '├── '}{_label(kid, by_id, step.id)}")
            draw(kid, indent + ("    " if last else "│   "))

    for root in roots:
        lines.append(_label(root, by_id, None))
        draw(root, "")
    return "\n".join(lines)
