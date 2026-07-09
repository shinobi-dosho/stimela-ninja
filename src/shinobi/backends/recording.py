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
    """Test-double backend that records calls instead of executing them."""

    name = "recording"

    def __init__(self) -> None:
        """Initialize the backend with an empty call log."""
        self.calls: list[tuple[Cab, list[str], dict[str, Any]]] = []

    def run(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, label: str = "", stream: bool = True
    ) -> BackendRun:
        """Record the call and return an empty, successful `BackendRun`.

        Args:
            cab: The cab being "executed".
            argv: Resolved command-line arguments that would have been run.
            inputs: Prepared inputs dict.
            label: Unused.
            stream: Unused.

        Returns:
            A `BackendRun` with returncode 0 and empty stdout/stderr.
        """
        self.calls.append((cab, argv, inputs))
        return BackendRun(returncode=0, stdout="", stderr="")
