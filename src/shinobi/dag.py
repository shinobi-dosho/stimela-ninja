"""A simple execution-graph tracer + renderer for `ninja run --dryrun`.

Since recipes are plain Python (see AGENTS.md), there's no declared graph
anywhere to inspect -- the only honest way to show one is to actually run
the recipe's real code, with cabs replaced by a no-op TraceBackend
(shinobi.backends.trace) that records each call instead of executing it.
This traces exactly the one path the recipe takes for the given inputs;
a branch not taken (or a different number of loop iterations for
different inputs) never appears -- that's an inherent property of
tracing plain synchronous Python, not a bug.

Dependency edges are detected, not assumed: each traced call is given a
unique placeholder string per declared output (embedding its call id), so
if a later call's params contain that placeholder, that's a genuine data
dependency -- not just "happened after." When a call has no detected
dependency, it's chained after the immediately preceding call instead, so
unrelated calls still render in the order they actually ran rather than
as a meaningless flat list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PLACEHOLDER_RE = re.compile(r"<<trace:(\d+):[^>]*>>")


@dataclass
class TraceStep:
    id: int
    name: str
    depends_on: set[int] = field(default_factory=set)


def placeholder(call_id: int, output_name: str) -> str:
    return f"<<trace:{call_id}:{output_name}>>"


def find_dependencies(params: dict) -> set[int]:
    """Scan resolved params for placeholders left by earlier traced calls."""
    deps: set[int] = set()

    def _scan(value: object) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                _scan(item)
            return
        for match in _PLACEHOLDER_RE.finditer(str(value)):
            deps.add(int(match.group(1)))

    for value in params.values():
        _scan(value)
    return deps


def _group_into_batches(steps: list[TraceStep]) -> list[list[TraceStep]]:
    """Consecutive (in call order) steps that share the exact same
    depends_on set are "parallel" siblings and rendered as one row.
    """
    batches: list[list[TraceStep]] = []
    for step in steps:
        if batches and batches[-1][0].depends_on == step.depends_on:
            batches[-1].append(step)
        else:
            batches.append([step])
    return batches


def _box(name: str) -> str:
    return f"[ {name} ]"


def _row_layout(batch: list[TraceStep], gap: int = 3) -> tuple[str, list[int]]:
    """A batch's row text, and the column-center of each box within it."""
    boxes = [_box(s.name) for s in batch]
    row = (" " * gap).join(boxes)
    centers = []
    col = 0
    for box in boxes:
        centers.append(col + len(box) // 2)
        col += len(box) + gap
    return row, centers


def _blank(width: int) -> list[str]:
    return [" "] * width


def _ticks(width: int, cols: list[str] | list[int], ch: str = "|") -> str:
    line = _blank(width)
    for c in cols:
        line[c] = ch
    return "".join(line)


def _bracket(
    width: int, cols: list[int], junction: str, left_corner: str, right_corner: str, junction_col: int | None = None
) -> str:
    """A horizontal bar spanning cols, with `junction` at the given
    column (or the bar's own midpoint if not given) and the corner
    characters at the two ends (a no-op, single '|', if there's only one
    column).
    """
    if len(cols) == 1:
        return _ticks(width, cols)
    lo, hi = min(cols), max(cols)
    mid = junction_col if junction_col is not None else (lo + hi) // 2
    line = _blank(width)
    for col in range(lo, hi + 1):
        line[col] = "-"
    line[lo], line[hi] = left_corner, right_corner
    line[mid] = junction
    return "".join(line)


def _arrows(width: int, cols: list[int]) -> str:
    line = _blank(width)
    for c in cols:
        line[c] = "v"
    return "".join(line)


def _connector(
    prev_batch: list[TraceStep], prev_centers: list[int], batch: list[TraceStep], centers: list[int], width: int
) -> list[str]:
    prev_ids = {s.id for s in prev_batch}
    shared_deps = batch[0].depends_on
    clean = shared_deps == prev_ids and bool(shared_deps)

    if not clean:
        # No exact match between what this batch depends on and the
        # previous batch's ids -- render a plain sequential connector
        # rather than implying a fan structure we can't back up.
        mid_prev = prev_centers[len(prev_centers) // 2]
        mid_next = centers[len(centers) // 2]
        return [_ticks(width, [mid_prev]), _ticks(width, [mid_next], "v")]

    lines = [_ticks(width, prev_centers)]

    if len(prev_centers) > 1:
        # fan-in bracket: lines come UP from parents, and a single line
        # continues DOWN from the junction -- corners curve up-to-across,
        # junction points down.
        lines.append(_bracket(width, prev_centers, junction="+", left_corner="+", right_corner="+"))
        spine = (min(prev_centers) + max(prev_centers)) // 2
    else:
        spine = prev_centers[0]

    if len(centers) > 1:
        # then, if fanning out again: a single line comes DOWN into the
        # junction, and spreads back out to each child -- corners curve
        # across-to-down, junction points up. The junction must sit at
        # `spine`'s column (where the incoming line actually is), not the
        # bracket's own geometric midpoint over `centers`. Only need a
        # fresh tick line here if a fan-in bracket was drawn above --
        # otherwise the very first tick line (under the lone parent) is
        # already at this exact column.
        if len(prev_centers) > 1:
            lines.append(_ticks(width, [spine]))
        lines.append(
            _bracket(width, centers, junction="+", left_corner="+", right_corner="+", junction_col=spine)
        )
        lines.append(_arrows(width, centers))
    else:
        lines.append(_ticks(width, [spine], "v"))

    return lines


def render_dag(steps: list[TraceStep]) -> str:
    if not steps:
        return "(no steps traced)"

    batches = _group_into_batches(steps)
    row_texts, row_centers = zip(*(_row_layout(batch) for batch in batches), strict=True)
    width = max(len(r) for r in row_texts)
    offsets = [(width - len(r)) // 2 for r in row_texts]

    out: list[str] = []
    for i, batch in enumerate(batches):
        centers_abs = [c + offsets[i] for c in row_centers[i]]
        if i > 0:
            prev_centers_abs = [c + offsets[i - 1] for c in row_centers[i - 1]]
            out.extend(_connector(batches[i - 1], prev_centers_abs, batch, centers_abs, width))
        out.append(" " * offsets[i] + row_texts[i])

    return "\n".join(out)
