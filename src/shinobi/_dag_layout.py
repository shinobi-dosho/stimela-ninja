"""Layered terminal diagrams for one scope's declared dependency graph."""

from __future__ import annotations

from collections import defaultdict
from graphlib import TopologicalSorter

from rich.cells import cell_len
from rich.console import Console
from rich.text import Text


def render_component(nodes: dict[int, list[str]], parents: dict[int, set[int]], width: int) -> list[str]:
    """Center topological layers and route every edge around node labels.

    Each horizontal track belongs to one source or one destination. Independent
    tracks cross at ╳, never at a junction. Long edges use outside tracks so
    they cannot pass through an intermediate node.
    """
    ranks: dict[int, int] = {}
    for key in TopologicalSorter(parents).static_order():
        ranks[key] = max((ranks[parent] + 1 for parent in parents[key]), default=0)
    layers: dict[int, list[int]] = defaultdict(list)
    for key in nodes:
        layers[ranks[key]].append(key)
    layers = dict(sorted(layers.items()))
    edges = [(parent, key) for key in nodes for parent in sorted(parents[key])]
    long_edges = [edge for edge in edges if ranks[edge[1]] > ranks[edge[0]] + 1]
    long_set = set(long_edges)
    columns = max(map(len, layers.values()))
    gap = 6
    children = defaultdict(list)
    for source, target in edges:
        children[source].append(target)
    degree = max(max(map(len, parents.values())), max((len(value) for value in children.values()), default=0))
    cell = max(16, 2 * degree + 4, min(max(cell_len(line) for lines in nodes.values() for line in lines), (width - 3 * len(long_edges) - gap * (columns - 1)) // columns))
    # Even centers give source and destination tracks distinct columns.
    cell += cell % 2
    span = columns * (cell + gap) - gap
    labels = {}
    centers = {}
    console = Console(width=cell)
    for keys in layers.values():
        start = (span - (len(keys) * (cell + gap) - gap)) // 2
        for index, key in enumerate(keys):
            centers[key] = 2 * ((start + index * (cell + gap) + cell // 2 + 1) // 2)
            labels[key] = [part.plain for line in nodes[key] for part in Text(line).wrap(console, cell, overflow="fold")]
        if len(keys) == 1 and parents[keys[0]]:
            key = keys[0]
            centers[key] = 2 * round(sum(centers[parent] for parent in parents[key]) / len(parents[key]) / 2)

    left_edges = [edge for edge in long_edges if centers[edge[0]] <= span // 2]
    left_set = set(left_edges)
    right_edges = [edge for edge in long_edges if edge not in left_set]
    padding = 3 * len(left_edges)
    centers = {key: center + padding for key, center in centers.items()}
    span += padding

    # Only a pure fan-in or fan-out can share a horizontal track. Mixed bands
    # use unique edge ports, so an unrelated wire cannot cross a corner/join.
    bands = {}
    source_ports, target_ports = {}, {}
    segments_by_rank = defaultdict(list)
    for edge in edges:
        segments_by_rank[ranks[edge[0]]].append((edge, "out"))
        if edge in long_set:
            segments_by_rank[ranks[edge[1]] - 1].append((edge, "in"))
    for rank in range(len(layers) - 1):
        segments = segments_by_rank[rank]
        simple = len({edge[0] for edge, _end in segments}) == 1 or len({edge[1] for edge, _end in segments}) == 1
        incoming, outgoing = defaultdict(list), defaultdict(list)
        for edge, end in segments:
            if end == "out":
                outgoing[edge[0]].append(edge)
            if end == "in" or edge not in long_set:
                incoming[edge[1]].append(edge)
        for key, outgoing_edges in outgoing.items():
            for index, edge in enumerate(outgoing_edges):
                source_ports[edge] = centers[key] if simple else centers[key] + 2 * (index - len(outgoing_edges) // 2)
        for key, incoming_edges in incoming.items():
            for index, edge in enumerate(incoming_edges):
                target_ports[edge] = centers[key] if simple else centers[key] + 2 * (index - len(incoming_edges) // 2) + 1
        bands[rank] = [segments] if simple else [[segment] for segment in segments]

    tops, bottoms, band_top = {}, {}, {}
    cursor = 0
    for rank, keys in layers.items():
        height = max(len(labels[key]) for key in keys)
        for key in keys:
            tops[key] = cursor
            bottoms[key] = cursor + len(labels[key]) - 1
        cursor += height
        if rank in bands:
            band_top[rank] = cursor
            cursor += len(bands[rank]) + (2 if len(bands[rank]) == 1 else 4)

    tracks = {edge: padding - 2 - 3 * index for index, edge in enumerate(left_edges)}
    tracks.update({edge: span + 2 + 3 * index for index, edge in enumerate(right_edges)})
    lanes = {}
    shared_lanes = {}
    for rank, groups in bands.items():
        for index, group in enumerate(groups):
            for edge, end in group:
                lanes[edge, end] = band_top[rank] + index + (1 if len(groups) == 1 else 2)
        if len(groups) == 1:
            shared_lanes[band_top[rank] + 1] = {edge for edge, _end in groups[0]}
    canvas: dict[tuple[int, int], dict[tuple[int, int], set[str]]] = defaultdict(lambda: defaultdict(set))
    arrows = set()

    def segment(edge, x1, y1, x2, y2):
        dx, dy = (x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1)
        while (x1, y1) != (x2, y2):
            nx, ny = x1 + dx, y1 + dy
            canvas[x1, y1][edge].add("R" if dx > 0 else "L" if dx < 0 else "D" if dy > 0 else "U")
            canvas[nx, ny][edge].add("L" if dx > 0 else "R" if dx < 0 else "U" if dy > 0 else "D")
            x1, y1 = nx, ny

    for edge in edges:
        source, target = edge
        rank = ranks[source]
        sx, tx = source_ports[edge], target_ports[edge]
        sy, ty = bottoms[source] + 1, tops[target] - 1
        lane = lanes[edge, "out"]
        points = [(centers[source], sy), (sx, sy), (sx, lane)]
        if edge in tracks:
            track = tracks[edge]
            end = lanes[edge, "in"]
            points += [(track, lane), (track, end), (tx, end)]
        else:
            points += [(tx, lane)]
        points += [(tx, ty - 1), (centers[target], ty - 1), (centers[target], ty)]
        for start, end in zip(points, points[1:]):
            segment(edge, *start, *end)
        canvas[centers[source], sy][edge].add("U")
        arrows.add((centers[target], ty))

    glyphs = {
        "D": "│",
        "U": "│",
        "DU": "│",
        "L": "─",
        "R": "─",
        "LR": "─",
        "DR": "┌",
        "DL": "┐",
        "RU": "└",
        "LU": "┘",
        "DLR": "┬",
        "LRU": "┴",
        "DRU": "├",
        "DLU": "┤",
        "DLRU": "┼",
    }
    rows = [{} for _ in range(cursor)]
    crossed = False
    for (x, y), owners in canvas.items():
        directions = set.union(*owners.values())
        sources, targets = {edge[0] for edge in owners}, {edge[1] for edge in owners}
        source_join = len(sources) == 1 and y == bottoms[next(iter(sources))] + 1
        target_join = len(targets) == 1 and y == tops[next(iter(targets))] - 2
        shared_join = y in shared_lanes and owners.keys() <= shared_lanes[y]
        # Readers with just one common successor can merge early without
        # creating a route to any other node.
        terminal_join = len(targets) == 1 and all(set(children[source]) == targets for source in sources)
        crossing = len(owners) > 1 and not (source_join or target_join or shared_join or terminal_join) and bool(directions & {"U", "D"}) and bool(directions & {"L", "R"})
        crossed |= crossing
        rows[y][x] = "╳" if crossing else glyphs.get("".join(sorted(directions)), "│")
    for x, y in arrows:
        rows[y][x] = "▼"
    for key in nodes:
        for index, line in enumerate(labels[key]):
            left = centers[key] - cell_len(line) // 2
            # Keep each label intact: wide glyphs occupy multiple cells and
            # combining marks belong to the preceding character.
            rows[tops[key] + index][left] = line
    rendered = []
    for row in rows:
        parts, position = [], 0
        for x, text in sorted(row.items()):
            parts.extend((" " * (x - position), text))
            position = x + cell_len(text)
        rendered.append("".join(parts).rstrip())
    if crossed:
        rendered += ["", "╳ = crossing; no connection"]
    return rendered


def render_scope(nodes: dict[int, list[str]], parents: dict[int, set[int]], width: int, *, wiring: dict[int, list[str]] | None = None) -> list[str]:
    """Lay disconnected components side by side when they fit; wrap by block."""
    neighbors = {key: set(parents[key]) for key in nodes}
    for key in nodes:
        for parent in parents[key]:
            neighbors[parent].add(key)
    components, remaining = [], set(nodes)
    order = {key: index for index, key in enumerate(nodes)}
    for key in nodes:
        if key not in remaining:
            continue
        component, pending = set(), [key]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        remaining -= component
        keys = sorted(component, key=order.__getitem__)
        block = render_component({key: nodes[key] for key in keys}, {key: parents[key] for key in component}, width)
        notes = [line for key in keys for line in (wiring or {}).get(key, ())]
        if notes:
            block += ["", "Wiring:", *notes]
        components.append(block)
    rendered, row, used = [], [], 0

    def flush():
        if rendered:
            rendered.append("")
        for index in range(max(len(block) for block, _width in row)):
            parts = []
            for block, block_width in row:
                line = block[index] if index < len(block) else ""
                parts.append(line + " " * (block_width - cell_len(line)))
            rendered.append("      ".join(parts).rstrip())

    for component in components:
        block_width = max(map(cell_len, component))
        if row and used + 6 + block_width > width:
            flush()
            row, used = [], 0
        row.append((component, block_width))
        used += block_width + (6 if len(row) > 1 else 0)
    if row:
        flush()
    return rendered
