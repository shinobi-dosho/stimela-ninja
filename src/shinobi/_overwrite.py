"""Declared step selection for explicit cache invalidation at run entry."""

import re

from shinobi.cache import get_cache_manifest
from shinobi.graph import build_graph
from shinobi.snapshots import invalidate
from shinobi.steps.schema import Recipe


def overwrite_cached_steps(scope, requested, cache_dir, *, validate_only=False, root_graph=None):
    """Invalidate selected steps and dependents, including nested recipe edges.

    Paths are relative to the invoked recipe, as in ``branch.image``. Selecting
    a recipe selects all its children; selecting a scatter selects all slices.
    Use the same snapshot invalidation as ``ninja cache invalidate``. Process
    consumers first so their cache keys remain available to identify snapshots.
    """
    nodes = {}
    dependents = {}
    descendants = {}
    order = []

    def visit(node, name, pattern, directory):
        nodes[name] = (pattern, directory)
        dependents[name] = set()
        children = set()
        if isinstance(node, Recipe):
            graph = root_graph if node is scope and root_graph is not None else build_graph(node)
            names = {}
            for index in graph.topological_indices():
                ref = node.steps[index]
                child = f"{name}.{ref.name}" if name else ref.name
                names[index] = child
                suffix = r"\[\d+\]" if ref.scatter is not None else ""
                child_pattern = pattern + r"\." + re.escape(ref.name) + suffix
                visit(ref.step, child, child_pattern, ref.step.cache_dir or directory)
                children.update({child, *descendants[child]})
                dependents[child].add(name)
            for index, sources in enumerate(graph.deps):
                for source in sources:
                    dependents[names[source]].update({names[index], *descendants[names[index]]})
        descendants[name] = children
        order.append(name)

    root = "" if isinstance(scope, Recipe) else scope.name
    visit(scope, root, re.escape(scope.name), cache_dir)
    unknown = set(requested) - nodes.keys()
    if unknown or "" in requested:
        raise ValueError("overwrite names unknown step(s): " + ", ".join(sorted(unknown or {"<empty>"})))
    if validate_only:
        return []
    affected = set()
    pending = [name for selected in requested for name in {selected, *descendants[selected]}]
    while pending:
        name = pending.pop()
        if name not in affected:
            affected.add(name)
            pending.extend(dependents[name])
    # Snapshot the selection before any invalidation changes a manifest.
    entries = {}
    selected = []
    for name in reversed(order):
        if name not in affected:
            continue
        pattern, directory = nodes[name]
        manifest = get_cache_manifest(directory)
        if directory not in entries:
            entries[directory] = manifest.read()
        recorded = entries[directory]
        selected.extend((directory, path) for path in recorded if re.fullmatch(pattern, path))
    notes = []
    for directory, path in selected:
        notes.extend(invalidate(directory, path, get_cache_manifest(directory)))
    return notes
