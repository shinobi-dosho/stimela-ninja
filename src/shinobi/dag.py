"""Static execution-graph *renderer* for `ninja run --dryrun`.

Recipes are declared DAGs (see AGENTS.md): a `Recipe`'s `steps` list plus
its wiring already *is* the graph. The graph itself -- nodes, true
dependency edges, cycle/wiring validation -- is built by
`shinobi.graph.build_graph`, shared with the executor so the two can never
disagree. `graph_nodes(recipe)` adapts that graph into `TraceStep` nodes for
*display*, and `render_dag` draws them as an indented tree:

    make
    ├── sub  (recipe)
    │   ├╌╌ a
    │   └╌╌ b
    └── after
        └── merge  (also waits on: sub)

A step sits under the step it waits on (`──`), so every branch is a real
fan-out. A step that waits on several is drawn once, under the last of them
in declaration order, and names the others in an "also waits on" note -- a
DAG is not a tree, and the note is what keeps the drawing honest about that.
A nested `Recipe` step is a branch of its own: its first steps hang beneath
it on a dashed connector (`╌╌`, "is a step of"), and whatever waits on the
whole recipe hangs beneath it on the ordinary one.

Every name is the step's own, unqualified. `build_graph` guarantees those
are unique among siblings, and an edge or note only ever connects siblings,
so the drawing is unambiguous. A qualified name (`sub.a`) would not be: step
names are free strings, and `Recipe.add_loop` itself declares dotted ones
(`selfcal.3.image`) that a nested path could spell identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import reprlib
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
        name: The step's own name, unqualified even inside a nested recipe
            (see the module docstring for why).
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
    scatter_label: str = ""
    scatter_details: tuple[str, ...] = ()


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
    _place(recipe, graph, None, nodes, plan, inputs=inputs)
    return nodes


def _place(recipe: "Recipe", graph, group: int | None, nodes: list[TraceStep], plan=None, *, inputs: dict[str, Any] | None = None) -> None:
    """Append `recipe`'s steps to `nodes` in declaration order, each nested
    recipe followed by its own steps, then add the edges between them.

    `plan` applies to the top level only, which is all `plan_recipe_accesses`
    resolves.
    """
    from shinobi.graph import build_graph
    from shinobi.steps.schema import Recipe, static_wiring_values

    ids: list[int] = []
    for i, name in enumerate(graph.names):
        ref = recipe.steps[i]
        scope = ref.step
        known, _unresolved = static_wiring_values(ref, inputs or {}, {})
        scatter_label, scatter_details = _scatter_display(ref, known)
        access_reasons: tuple[str, ...] = ()
        if plan is not None:
            access_reasons = tuple(
                reason for parent in sorted((graph.names[item] for item in graph.deps[i]), key=graph.names.index) for reason in plan.reasons.get((name, parent), ())
            )
        declared = scope.resources
        node = TraceStep(
            id=len(nodes),
            name=name,
            resources=declared.describe() if declared else "",
            access_reasons=access_reasons,
            recipe=isinstance(scope, Recipe),
            group=group,
            scatter_label=scatter_label,
            scatter_details=scatter_details,
        )
        nodes.append(node)
        ids.append(node.id)
        if node.recipe and scope.steps:
            _place(scope, build_graph(scope), node.id, nodes, inputs=known if ref.scatter is None else None)
    for i, deps in enumerate(graph.deps):
        nodes[ids[i]].depends_on = {ids[dep] for dep in deps}


def _display_value(value: Any) -> str:
    """Bound static previews without instantiating an input model."""
    from shinobi.steps.schema import InputRef, OutputRef

    if isinstance(value, InputRef):
        return f"InputRef(field={_display_value(value.field)})"
    if isinstance(value, OutputRef):
        selection = f", selection={_display_value(value.selection)}" if value.selection is not None else ""
        return f"OutputRef(step={_display_value(value.step)}, field={_display_value(value.field)}{selection})"
    preview = reprlib.Repr()
    preview.maxlist = 5
    preview.maxtuple = 5
    preview.maxstring = 80
    preview.maxother = 80
    return preview.repr(value).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _scatter_display(ref, known: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Display declarations only; outputs stay symbolic and nodes stay atomic."""
    if ref.scatter is None:
        return "", ()
    fields = ref.scatter.fields
    lengths = [len(known[name]) for name in fields if isinstance(known.get(name), list)]
    count = f"; {lengths[0]} slices" if len(lengths) == len(fields) and len(set(lengths)) == 1 else ""
    values = tuple(f"{name} = {_display_value(known[name] if name in known else ref.wiring.get(name, '<unresolved>'))}" for name in fields)
    if len(fields) == 1:
        return f": {values[0]} [scatter{count}]", ()
    return f" [scatter zip({', '.join(fields)}){count}]", values


def _label(step: TraceStep, by_id: dict[int, TraceStep], parent: int | None) -> str:
    """A step's line: its name, then anything that changes how it runs."""
    others = sorted(dep for dep in step.depends_on if dep != parent)
    details = [
        *(["recipe"] if step.recipe else []),
        *([step.resources] if step.resources else []),
        *step.access_reasons,
        *([f"also waits on: {', '.join(by_id[dep].name for dep in others)}"] if others else []),
    ]
    label = step.name + step.scatter_label
    return label if not details else f"{label}  ({'; '.join(details)})"


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
    children: dict[int, list[TraceStep]] = {step.id: [] for step in steps}
    roots: list[TraceStep] = []
    for step in steps:
        # The last thing it waits on, in declaration order: by the time that
        # one is drawn, every earlier dependency already has been. A step
        # that waits on nothing in its own recipe hangs from the recipe.
        parent = max(step.depends_on) if step.depends_on else step.group
        if parent is None:
            roots.append(step)
        else:
            children[parent].append(step)

    lines: list[str] = []

    def draw(step: TraceStep, indent: str) -> None:
        kids = sorted(children[step.id], key=lambda kid: kid.id)
        for position, kid in enumerate(kids):
            last = position == len(kids) - 1
            # Dashed: `kid` is one of `step`'s own steps. Solid: it waits on `step`.
            line = "╌╌" if kid.group == step.id and not kid.depends_on else "──"
            lines.append(f"{indent}{'└' if last else '├'}{line} {_label(kid, by_id, step.id)}")
            child_indent = indent + ("    " if last else "│   ")
            lines.extend(f"{child_indent}{detail}" for detail in kid.scatter_details)
            draw(kid, child_indent)

    for root in roots:
        lines.append(_label(root, by_id, None))
        lines.extend(f"    {detail}" for detail in root.scatter_details)
        draw(root, "")
    return "\n".join(lines)
