from __future__ import annotations

from typing import Any

from shinobi.backends import Backend, register
from shinobi.backends._stream import run_streaming
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab


@register
class NativeBackend(Backend):
    """Runs the cab's command directly on the host, via subprocess."""

    name = "native"

    def run(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, label: str = "", stream: bool = True
    ) -> BackendRun:
        return run_streaming(argv, label=label or cab.name, stream=stream)
