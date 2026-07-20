"""Shared subprocess-running helper for backends that want live stdout/
stderr echo (`native`, `container`, and `steps.pyfunc`'s own inline
container-subprocess call for pysteps) without changing the
`Backend.run()` contract every caller depends on: a blocking call that
returns a complete `BackendRun(returncode, stdout, stderr)`.

`stream=True` adds a side channel (each line echoed to the terminal, via
`click.echo`, as it arrives, prefixed with a caller-supplied label) on top
of that same contract -- it does not change what's captured or returned.
"""

from __future__ import annotations

import subprocess
import threading
from typing import Any

import click

from shinobi.results import BackendRun


def _pump(stream, sink: list[str], *, label: str, err: bool) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        click.echo(f"[{label}] {line.rstrip()}", err=err)
    stream.close()


def run_streaming(argv: list[str], *, label: str, stream: bool, **popen_kwargs: Any) -> BackendRun:
    """Run `argv`, returning a `BackendRun` with the complete captured
    stdout/stderr either way.

    `stream=False`: identical to `subprocess.run(argv, capture_output=True,
    text=True)` -- today's behavior, unchanged.

    `stream=True`: runs via `subprocess.Popen`, with one reader thread per
    stream echoing each line (prefixed `"[{label}] "`) to the terminal as
    it arrives while also accumulating it, so the returned `BackendRun` is
    byte-for-byte the same text a non-streaming run would have captured.
    """
    if not stream:
        proc = subprocess.run(argv, capture_output=True, text=True, **popen_kwargs)
        return BackendRun(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, stdout_lines), kwargs={"label": label, "err": False}),
        threading.Thread(target=_pump, args=(proc.stderr, stderr_lines), kwargs={"label": label, "err": True}),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    returncode = proc.wait()
    return BackendRun(returncode=returncode, stdout="".join(stdout_lines), stderr="".join(stderr_lines))
