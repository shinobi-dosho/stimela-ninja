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
        self,
        cab: Cab,
        argv: list[str],
        inputs: dict[str, Any],
        *,
        label: str = "",
        stream: bool = True,
        pin: bool = False,  # accepted for the Backend protocol; native runs no container to pin
        cwd: str | None = None,
    ) -> BackendRun:
        """Run a cab's argv directly on the host.

        Args:
            cab: The cab being executed.
            argv: Resolved command-line arguments to run.
            inputs: Prepared inputs dict; unused by this backend.
            label: Label used for streamed output lines. Defaults to `cab.name`.
            stream: Whether to stream stdout/stderr live as the process runs.
            cwd: Working directory for the subprocess (e.g. a step sandbox);
                `None` runs in the process cwd, as before.

        Returns:
            The completed `BackendRun` (never raises on non-zero exit).
        """
        return run_streaming(argv, label=label or cab.name, stream=stream, cwd=cwd)
