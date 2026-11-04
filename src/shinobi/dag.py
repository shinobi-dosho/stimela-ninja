"""Static execution-graph *renderer* for `ninja run --dryrun`.

Recipes are declared DAGs: a `Recipe`'s `steps` list plus
its wiring already *is* the graph. The graph itself -- nodes, true
dependency edges, cycle/wiring validation -- is built by
`shinobi.graph.build_graph`, shared with the executor so the two can never
disagree. `graph_nodes(recipe)` adapts that graph into `TraceStep` nodes for
*display*. `render_dag` centers topological layers, places parallel steps side
by side and draws every dependency with an arrow. Independent components wrap
as complete blocks. Nested recipes remain atomic nodes in their parent's DAG;
their contents are drawn separately with explicit recipe headings.

Every name is the step's own, unqualified. `build_graph` guarantees those
are unique among siblings, and an edge or note only ever connects siblings,
so each scope's drawing is unambiguous. A qualified name (`sub.a`) would not be: step
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
    # The step's declared resource footprint, rendered below its name.
    # `--dryrun` shows what is *declared*, and a footprint is one of the few
    # declarations that changes the shape of a run rather than its content:
    # it is what turns parallel branches into a queue. Seeing that before
    # running, rather than inferring it from timings afterwards, is the
    # whole point of the dry run.
    resources: str = ""
    access_reasons: tuple[str, ...] = ()
    recipe: bool = False
    group: int | None = None
    placement: str = ""
    scatter_label: str = ""
    scatter_details: tuple[str, ...] = ()
    scatter_count: int | None = None
    wiring_details: tuple[str, ...] = ()
    wiring_previews: tuple[str, ...] = ()


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

    from shinobi.offload.mixed import has_placement, plan_declared

    if inputs is not None and has_placement(recipe):
        plan_declared(recipe, inputs, workspace or Path.cwd())
        plan = None
    else:
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
        scatter_label, scatter_details, scatter_count = _scatter_display(ref, known)
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
            placement=f"Slurm allocation; max_parallel={ref.placement.max_parallel}" if ref.placement else "",
            scatter_label=scatter_label,
            scatter_details=scatter_details,
            scatter_count=scatter_count,
            wiring_details=_wired_products(ref),
            wiring_previews=_wired_products(ref, details=True),
        )
        nodes.append(node)
        ids.append(node.id)
        if isinstance(scope, Recipe) and scope.steps:
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


def _wired_products(ref, *, details: bool = False) -> tuple[str, ...]:
    """Show producer fields; include selection values only in detailed views."""
    from shinobi.steps.schema import InputRef, OutputRef

    def label(source):
        if isinstance(source, InputRef):
            return f"inputs.{source.field}"
        selected = ""
        if source.selection is not None:
            selected = f" select {_display_value(source.selection)}" if details else f" select({', '.join(source.selection)})"
        return f"{source.step}.{source.field}{selected}"

    products = []
    for name, source in ref.wiring.items():
        sources = source if isinstance(source, list) else [source]
        if not any(isinstance(item, OutputRef) for item in sources):
            continue
        value = f"[{', '.join(label(item) for item in sources)}]" if isinstance(source, list) else label(source)
        products.append(f"{name} <- {value}")
    return tuple(products)


def _scatter_display(ref, known: dict[str, Any]) -> tuple[str, tuple[str, ...], int | None]:
    """Display declarations only; outputs stay symbolic and nodes stay atomic."""
    if ref.scatter is None:
        return "", (), None
    fields = ref.scatter.fields
    lengths = [len(known[name]) for name in fields if isinstance(known.get(name), list)]
    slice_count = lengths[0] if len(lengths) == len(fields) and len(set(lengths)) == 1 else None
    count = f"; {slice_count} slices" if slice_count is not None else ""
    values = tuple(f"{name} = {_display_value(known[name] if name in known else ref.wiring.get(name, '<unresolved>'))}" for name in fields)
    if len(fields) == 1:
        return f": {values[0]} [scatter{count}]", (), slice_count
    return f" [scatter zip({', '.join(fields)}){count}]", values, slice_count


def render_dag(steps: list[TraceStep], *, details: bool = False, width: int = 120) -> str:
    """Draw centered dependency layers with parallel branches side by side.

    Args:
        steps: Display nodes, as produced by `graph_nodes`.
        details: Include parameter previews and filesystem access reasons.
            The default shows only graph structure and wired product names.
        width: Preferred terminal width. Complete components wrap as blocks;
            exceptionally wide layers retain a minimum readable label width.

    Returns:
        A multi-line string with the rendered graph, or a placeholder
        message if `steps` is empty.
    """
    if not steps:
        return "(no steps traced)"

    from shinobi._dag_layout import render_scope

    lines = []

    def draw(group: int | None, context: str = "") -> None:
        members = [step for step in steps if step.group == group]
        labels = {}
        for step in members:
            label = [step.name]
            if step.recipe:
                label.append("(recipe)")
            if step.scatter_label:
                count = f"; {step.scatter_count} slices" if step.scatter_count is not None else ""
                label.append(f"[scatter{count}]")
            label.extend(value for value in (step.resources, step.placement) if value)
            labels[step.id] = label
        wiring = {step.id: [f"  {step.name}: {value}" for value in (step.wiring_previews if details and step.wiring_previews else step.wiring_details)] for step in members}
        lines.extend(render_scope(labels, {step.id: step.depends_on for step in members}, max(40, width), wiring=wiring))
        if details:
            diagnostics = [step for step in members if step.scatter_label or step.access_reasons]
            if diagnostics:
                lines.extend(["", "Details:"])
            for step in diagnostics:
                lines.append(f"  {step.name}{step.scatter_label}")
                lines.extend(f"    {value}" for value in (*step.scatter_details, *step.access_reasons))
        for step in members:
            if step.recipe and any(child.group == step.id for child in steps):
                child_context = f"{context} > {step.name!r}" if context else repr(step.name)
                lines.extend(["", f"Contents of recipe {child_context}:"])
                draw(step.id, child_context)

    draw(None)
    return "\n".join(lines)
