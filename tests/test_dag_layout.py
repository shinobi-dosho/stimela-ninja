"""Read the drawn arrows to verify that crossings preserve the exact DAG."""

from random import Random

import pytest
from rich.cells import cell_len

from shinobi._dag_layout import render_component
from shinobi.dag import TraceStep, render_dag


def _drawn_edges(lines, names, heights=None):
    positions = {}
    for key, name in names.items():
        row = next(index for index, line in enumerate(lines) if name in line)
        positions[key] = (cell_len(lines[row][: lines[row].index(name)]) + cell_len(name) // 2, row)
    destinations = {(x, y - 1): key for key, (x, y) in positions.items()}
    directions = {
        "│": {(0, -1), (0, 1)},
        "─": {(-1, 0), (1, 0)},
        "┌": {(0, 1), (1, 0)},
        "┐": {(0, 1), (-1, 0)},
        "└": {(0, -1), (1, 0)},
        "┘": {(0, -1), (-1, 0)},
        "┬": {(-1, 0), (1, 0), (0, 1)},
        "┴": {(-1, 0), (1, 0), (0, -1)},
        "├": {(0, -1), (0, 1), (1, 0)},
        "┤": {(0, -1), (0, 1), (-1, 0)},
        "┼": {(-1, 0), (1, 0), (0, -1), (0, 1)},
    }

    cells = []
    for line in lines:
        row, offset = {}, 0
        for symbol in line:
            size = cell_len(symbol)
            if size:
                row[offset] = symbol
            offset += size
        cells.append(row)

    def char(x, y):
        return cells[y].get(x, " ") if 0 <= y < len(cells) else " "

    edges = set()
    for source, (x, y) in positions.items():
        bottom = y + (heights or {}).get(source, 1)
        pending = [(x, bottom, 0, 1), (x + 1, bottom, 0, 1)]
        seen = set()
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            x, y, dx, dy = state
            symbol = char(x, y)
            if symbol == "▼":
                assert (x, y) in destinations
                edges.add((source, destinations[x, y]))
                continue
            options = {(dx, dy)} if symbol == "╳" else directions.get(symbol, set())
            for nx, ny in options:
                if ny < 0:
                    continue  # Arrow flow goes down; a fan-in is not a reverse fork.
                next_symbol = char(x + nx, y + ny)
                if next_symbol in ("╳", "▼") or (-nx, -ny) in directions.get(next_symbol, set()):
                    pending.append((x + nx, y + ny, nx, ny))
    return edges


@pytest.mark.parametrize("seed", range(12))
def test_drawn_arrows_match_all_and_only_declared_edges(seed):
    random = Random(seed)
    names = {key: f"node-{key:02d}" for key in range(12)}
    parents = {key: {parent for parent in range(key) if random.random() < 0.23} for key in names}
    # Include skipped layers and several partially overlapping fan-ins.
    parents[9].update({0, 4})
    parents[11].update({1, 8, 10})
    lines = render_component({key: [name] for key, name in names.items()}, parents, 120)
    assert _drawn_edges(lines, names) == {(parent, key) for key in names for parent in parents[key]}


def test_components_wrap_without_splitting_their_diagrams():
    steps = [
        TraceStep(0, "first-source-long-name"),
        TraceStep(1, "first-target-long-name", {0}),
        TraceStep(2, "second-source-long-name"),
        TraceStep(3, "second-target-long-name", {2}),
    ]
    wide = render_dag(steps, width=120).splitlines()
    assert any("first-source" in line and "second-source" in line for line in wide)
    narrow = render_dag(steps, width=40).splitlines()
    assert next(i for i, line in enumerate(narrow) if "first-target" in line) < next(i for i, line in enumerate(narrow) if "second-source" in line)


def test_topological_layout_does_not_require_declaration_order():
    nodes = {2: ["node-02"], 0: ["node-00"], 1: ["node-01"]}
    parents = {2: {1}, 0: set(), 1: {0}}
    lines = render_component(nodes, parents, 80)
    assert _drawn_edges(lines, {key: value[0] for key, value in nodes.items()}) == {(0, 1), (1, 2)}


def test_long_identifier_wraps_without_losing_characters():
    name = "long_step_identifier_" * 8
    lines = render_dag([TraceStep(0, name)], width=40).splitlines()
    assert "".join(line.strip() for line in lines) == name
    assert max(map(len, lines)) <= 40


def test_component_wiring_stays_with_its_own_diagram():
    steps = [
        TraceStep(0, "first-root"),
        TraceStep(1, "first-use", {0}, wiring_details=("value <- first-root.result",)),
        TraceStep(2, "second-root"),
        TraceStep(3, "second-use", {2}),
    ]
    rendered = render_dag(steps, width=40)
    assert rendered.index("first-use: value <- first-root.result") < rendered.index("second-root")


def test_arrows_attach_to_labels_of_different_heights():
    nodes = {0: ["node-00"], 1: ["node-01", "metadata"], 2: ["node-02"], 3: ["node-03"]}
    parents = {0: set(), 1: set(), 2: {0}, 3: {1}}
    lines = render_component(nodes, parents, 80)
    assert _drawn_edges(lines, {key: value[0] for key, value in nodes.items()}, {key: len(value) for key, value in nodes.items()}) == {(0, 2), (1, 3)}


@pytest.mark.parametrize("name", ["測試" * 8, "e\u0301" * 16, "測e\u0301" * 6])
def test_unicode_labels_preserve_arrow_connections(name):
    names = {0: "root", 1: name, 2: "other", 3: "merge"}
    parents = {0: set(), 1: {0}, 2: {0}, 3: {0, 1, 2}}
    lines = render_component({key: [value] for key, value in names.items()}, parents, 80)
    assert name in "\n".join(lines)
    assert _drawn_edges(lines, names) == {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}


@pytest.mark.parametrize("name", ["測試" * 40, "e\u0301" * 80, "測e\u0301" * 40, "👩\u200d💻" * 40])
def test_unicode_labels_wrap_without_losing_characters(name):
    lines = render_dag([TraceStep(0, name)], width=40).splitlines()
    assert "".join(line.strip() for line in lines) == name
    assert max(map(cell_len, lines)) <= 40
    assert all(not line.lstrip().startswith("\u0301") for line in lines)


@pytest.mark.parametrize("name", ["測" * 12, "e\u0301" * 24])
def test_unicode_components_use_display_width_for_packing_and_padding(name):
    steps = [TraceStep(0, name), TraceStep(1, "second-component")]
    wide = render_dag(steps, width=60).splitlines()
    assert len(wide) == 1
    assert cell_len(wide[0][: wide[0].index("second-component")]) == 30
    assert cell_len(wide[0]) <= 60
    narrow = render_dag(steps, width=40).splitlines()
    assert any(name in line for line in narrow)
    assert not any(name in line and "second-component" in line for line in narrow)
