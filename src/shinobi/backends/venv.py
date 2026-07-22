"""Venv backend: runs a cab's command inside an existing Python virtualenv.

Weaker than the container backends by design -- no filesystem namespace, no
UID mapping, no OS-level image pin -- and a *complement* to them, not a
replacement: it covers the pip-installable half of a pipeline (quartical,
tricolour, breizorro, ...) without requiring a container runtime on every
host. The heterogeneous half (wsclean, casa, aoflagger) still needs images.

Execution is `native` plus an environment: the venv's `bin` is prepended to
`PATH`, `VIRTUAL_ENV` is set, and `PYTHONHOME`/`PYTHONPATH` are cleared --
exactly what `bin/activate` does, minus the shell. `argv[0]` is rewritten to
the venv's own copy of the tool when one exists, so a missing tool fails
loudly instead of silently falling through to a host binary of the same name.

Provenance (only under `pin=True`, mirroring the container pin-then-run
contract) is a `venv_digest`: a sha256 of the venv's sorted `name==version`
distribution list. This is a *version-parity* record, not an OS-level pin --
identical version lists can sit on different compiled C-extensions -- so a
venv step is always reported *unpinned* in the run manifest (see
`shinobi.provenance.RunManifest.pinned`); the digest is informational.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, get_backend, register
from shinobi.backends._stream import run_streaming
from shinobi.config import AppConfig
from shinobi.exceptions import BackendError
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab

# Code run *by the venv's own interpreter* to list its installed
# distributions. Not `pip freeze`: `uv venv` does not install pip by default,
# so `python -m pip` fails on exactly the venvs this feature most often meets.
# `importlib.metadata` is stdlib and always present.
_FREEZE_CODE = "import importlib.metadata as m, json;print(json.dumps(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))"


def resolve_venv(scope_venv: str | None, config: AppConfig | None = None) -> Path | None:
    """Resolve a step's declared venv (or the config default) to a validated
    venv directory, or `None` when nothing is declared anywhere.

    Resolution order: the scope's own value, then `backend.venv.default`. A
    value that is a key in `backend.venv.envs` maps to its path; otherwise it
    is treated as a filesystem path (``~`` expanded, made absolute).

    Raises:
        BackendError: If a venv is declared but `<venv>/bin/python` is missing.
    """
    cfg = config or AppConfig.load()
    declared = scope_venv or cfg.backend.venv.default
    if not declared:
        return None
    target = cfg.backend.venv.envs.get(declared, declared)
    venv = Path(target).expanduser().resolve()
    if not (venv / "bin" / "python").exists():
        raise BackendError(
            f"venv {declared!r} resolves to {venv}, but {venv / 'bin' / 'python'} "
            "does not exist -- is it a real virtualenv? (this backend does not "
            "create venvs; provision it first)"
        )
    return venv


def venv_env(venv: Path) -> dict[str, str]:
    """The process environment for running inside `venv`: `bin` prepended to
    `PATH`, `VIRTUAL_ENV` set, `PYTHONHOME`/`PYTHONPATH` cleared. Mirrors
    `bin/activate` without the shell; dropping the two `PYTHON*` vars is what
    stops the host interpreter's paths leaking into the venv.
    """
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def resolve_command(venv: Path, argv0: str) -> str:
    """Rewrite a bare command name to the venv's own copy when it has one.

    A command containing a ``/`` is an explicit path and is left untouched.
    Otherwise, if `<venv>/bin/<argv0>` exists, return that absolute path so a
    missing tool is a hard failure rather than a silent fall-through to a host
    binary of the same name; if it doesn't, return `argv0` unchanged (the
    venv's `PATH` still applies, so a tool that lives elsewhere on it resolves
    normally).
    """
    if "/" in argv0:
        return argv0
    candidate = venv / "bin" / argv0
    return str(candidate) if candidate.exists() else argv0


@lru_cache(maxsize=None)
def venv_digest(venv: Path) -> str | None:
    """`sha256` of the venv's sorted `name==version` distribution list, or
    `None` on any failure (an honest null -- never a fabricated digest).

    Cached per resolved path: a venv does not change mid-run, mirroring the
    container backend's `lru_cache` on image pinning. Under `max_workers > 1`
    several threads may redundantly shell out on a first miss, but the result
    is deterministic.
    """
    python = venv / "bin" / "python"
    try:
        proc = subprocess.run(
            [str(python), "-c", _FREEZE_CODE],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        dists = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    blob = "\n".join(dists)
    return hashlib.sha256(blob.encode()).hexdigest()


@register
class VenvBackend(Backend):
    """Runs the cab's command inside an existing Python virtualenv.

    When no venv is declared (on the scope or in config) the run degrades to
    the native backend with a warning -- selecting `venv` is an opt-in to
    isolation, so a silent no-op would be a surprise, but a hard error would
    make `backend.default: venv` unusable for recipes that mix container and
    pure-Python steps.
    """

    name = "venv"

    def run(
        self,
        cab: Cab,
        argv: list[str],
        inputs: dict[str, Any],
        *,
        label: str = "",
        stream: bool = True,
        pin: bool = False,
        cwd: str | None = None,
    ) -> BackendRun:
        """Run `argv` inside the resolved venv (or natively if none)."""
        venv = resolve_venv(cab.venv)
        if venv is None:
            import warnings

            warnings.warn(
                f"step '{label or cab.name}' selected the venv backend but no venv is "
                "declared (on the step or via backend.venv.default) -- running natively, "
                "with no environment isolation",
                stacklevel=2,
            )
            return get_backend("native").run(cab, argv, inputs, label=label, stream=stream, pin=pin, cwd=cwd)

        run_argv = [resolve_command(venv, argv[0]), *argv[1:]] if argv else argv
        run = run_streaming(run_argv, label=label or cab.name, stream=stream, cwd=cwd, env=venv_env(venv))
        run.venv = str(venv)
        if pin:
            run.venv_digest = venv_digest(venv)
        return run
