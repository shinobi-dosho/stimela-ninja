from __future__ import annotations

import subprocess
from typing import Any

from shinobi.backends import Backend, register
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab


@register
class NativeBackend(Backend):
    """Runs the cab's command directly on the host, via subprocess."""

    name = "native"

    def run(self, cab: Cab, argv: list[str], inputs: dict[str, Any]) -> BackendRun:
        proc = subprocess.run(argv, capture_output=True, text=True)
        return BackendRun(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
