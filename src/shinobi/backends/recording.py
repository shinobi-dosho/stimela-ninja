"""A backend that records calls instead of executing -- a test double.

Registered as "recording" so tests can select it via a cab/recipe
backend or ``register_step_backend``. Records ``(cab, argv, inputs)`` per
call and returns an empty, successful ``BackendRun`` (so downstream
output-filling falls back to defaults / same-named inputs).
"""

from __future__ import annotations

from typing import Any

from shinobi.backends import Backend, register
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab


@register
class RecordingBackend(Backend):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[Cab, list[str], dict[str, Any]]] = []

    def run(self, cab: Cab, argv: list[str], inputs: dict[str, Any]) -> BackendRun:
        self.calls.append((cab, argv, inputs))
        return BackendRun(returncode=0, stdout="", stderr="")
