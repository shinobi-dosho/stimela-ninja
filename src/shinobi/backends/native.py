from __future__ import annotations

import subprocess
from typing import Any

from shinobi.backends import Backend, register
from shinobi.results import Result
from shinobi.schema import CabDef
from shinobi.wranglers import apply_wranglers


@register
class NativeBackend(Backend):
    """Runs the cab's command directly on the host, via subprocess."""

    name = "native"

    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        proc = subprocess.run(argv, capture_output=True, text=True)
        lines = proc.stdout.splitlines() + proc.stderr.splitlines()
        outputs = apply_wranglers(cab.wranglers, lines)
        return Result(
            cab_name=cab.name,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            outputs=outputs,
        )
