"""Pystep functions loaded by the venv's own interpreter in
`test_steps_pyfunc_venv.py`. Kept in their own module with only framework
(stubbed in-runner) and stdlib imports -- the runner execs this whole file
under the venv, so a stray `import pytest` here would break the run, exactly
as it would in a real pipeline's pystep module.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class MagicOut(BaseModel):
    value: int


def use_venv_only_pkg(n: int) -> MagicOut:
    import venvonlypkg  # only exists in the test venv, never on the host

    return MagicOut(value=venvonlypkg.MAGIC + n)


class PathOut(BaseModel):
    report: Path


def write_report(n: int) -> PathOut:
    import venvonlypkg

    # A relative, output-only path: under a sandbox the cwd is the scratch
    # dir, so this lands there and is harvested back to the workspace.
    report = Path("report.txt")
    report.write_text(f"magic={venvonlypkg.MAGIC + n}\n")
    return PathOut(report=report)


def plain_double(n: int) -> MagicOut:
    return MagicOut(value=n * 2)
