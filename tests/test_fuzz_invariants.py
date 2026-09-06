"""Deterministic fuzz-style invariants for parser and graph boundaries.

These are deliberately standard-library randomized tests rather than a new
test dependency: each fixed seed produces a broad, reproducible corpus that
exercises combinations cumbersome to enumerate by hand.  Keep the seeds
stable; a failing generated case can then be reproduced exactly.
"""

from __future__ import annotations

import random
import string
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shinobi.graph import build_graph
from shinobi.loaders._modelgen import dtype_to_type
from shinobi.steps.schema import Cab, OutputRef, ParamMeta, ParamPattern, ParamSegment, Recipe, StepRef


def _dtype_case(rng: random.Random, depth: int = 0) -> tuple[str, Any]:
    """Build a supported dtype spelling alongside its expected Python type."""
    atoms = [("str", str), ("int", int), ("float", float), ("File", Path)]
    if depth >= 3 or rng.random() < 0.45:
        return rng.choice(atoms)

    kind = rng.choice(("list", "tuple", "union"))
    if kind == "list":
        inner, expected = _dtype_case(rng, depth + 1)
        return f"List[{inner}]", list[expected]

    items = [_dtype_case(rng, depth + 1) for _ in range(rng.randint(1, 4))]
    spelling = ", ".join(item[0] for item in items)
    types = tuple(item[1] for item in items)
    if kind == "tuple":
        return f"Tuple[{spelling}]", tuple[types]

    expected = types[0]
    for item in types[1:]:
        expected = expected | item
    return f"Union[{spelling}]", expected


def test_dtype_parser_fuzzes_nested_supported_grammar():
    """Nested brackets must never make comma splitting lose type structure."""
    rng = random.Random(0xD7A7E)
    for _ in range(500):
        spelling, expected = _dtype_case(rng)
        assert dtype_to_type(spelling) == expected, spelling


def _name(rng: random.Random, *, separator: str) -> str:
    alphabet = string.ascii_letters + string.digits + "_-"
    while True:
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        if separator not in value:
            return value


def test_param_pattern_fuzzes_exact_segment_boundaries():
    """Attrs are literal segments, even when one is a suffix of another."""
    rng = random.Random(0xFA77E)
    attrs = {
        "int": ParamMeta(dtype="int"),
        "time-int": ParamMeta(dtype="float"),
        "very-long-time-int": ParamMeta(dtype="File"),
    }
    pattern = ParamPattern(
        separator="-",
        segments=[ParamSegment(regex=r"[A-Za-z0-9_]+?"), ParamSegment(attrs=attrs)],
    )

    for _ in range(500):
        prefix = _name(rng, separator="-")
        attr = rng.choice(list(attrs))
        assert pattern.matches(f"{prefix}-{attr}") is attrs[attr]

        # Extra/missing segments must not be accepted through a partial match.
        assert pattern.matches(f"{prefix}-{attr}-extra") is None
        assert pattern.matches("int") is None


class _GraphInput(BaseModel):
    value: str = "value"


class _GraphOutput(BaseModel):
    value: str = "value"


def test_recipe_graph_fuzzes_random_declared_dags():
    """Graph reverse edges and topological validity agree for varied wiring."""
    rng = random.Random(0xDA6)
    cab = Cab(name="leaf", command="leaf", inputs_model=_GraphInput, outputs_model=_GraphOutput)

    for _ in range(200):
        count = rng.randint(1, 30)
        expected_deps: list[set[int]] = []
        refs: list[StepRef] = []
        for index in range(count):
            deps = {candidate for candidate in range(index) if rng.random() < 0.18}
            expected_deps.append(deps)
            sources = [OutputRef(step=f"step{dep}", field="value") for dep in sorted(deps)]
            wiring: dict[str, OutputRef | list[OutputRef]] = {}
            if len(sources) == 1:
                wiring["value"] = sources[0]
            elif sources:
                wiring["value"] = sources
            refs.append(StepRef(name=f"step{index}", step=cab, wiring=wiring))

        graph = build_graph(Recipe(name="generated", inputs_model=_GraphInput, outputs_model=_GraphOutput, steps=refs))
        assert graph.deps == expected_deps
        assert graph.dependents == [{i for i, dependencies in enumerate(expected_deps) if node in dependencies} for node in range(count)]
        assert set(TopologicalSorter(dict(enumerate(graph.deps))).static_order()) == set(range(count))
