"""Container backend: wraps a cab's argv in a container-runtime invocation.

Shells out to the runtime binary (docker/podman/apptainer) rather than
using each runtime's Python SDK -- one code path for all of them, no extra
heavyweight client dependencies, and it works uniformly for runtimes (like
apptainer) that don't have a good Python API to begin with.

Volume mounts are derived from the cab's own schema: any resolved
parameter whose declared dtype looks file-like (``File``, ``MS``,
``list:File``, ...) has its parent directory bind-mounted at the same
path inside the container, so the tool sees the same paths the caller
used. This is why Backend.run() is handed the resolved params dict, not
just argv -- argv is just strings by that point, with no memory of which
of them are paths.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, register
from shinobi.exceptions import BackendError
from shinobi.results import Result
from shinobi.schema import CabDef, is_file_like_dtype
from shinobi.wranglers import apply_wranglers

_DOCKER_LIKE = {"docker", "podman"}
_APPTAINER_LIKE = {"apptainer"}


def bind_dirs(cab: CabDef, params: dict[str, Any], workdir: str) -> list[str]:
    """Parent directories of every File/MS-valued resolved param, plus the
    working directory itself. Order-preserving, de-duplicated.
    """
    dirs = [workdir]
    seen = {workdir}

    for name, value in params.items():
        schema = cab.inputs.get(name)
        if schema is None or value is None or not is_file_like_dtype(schema.dtype):
            continue

        for item in value if isinstance(value, (list, tuple)) else [value]:
            path = Path(str(item))
            if not path.is_absolute():
                path = Path(workdir) / path
            parent = str(path.parent)
            if parent not in seen:
                seen.add(parent)
                dirs.append(parent)

    return dirs


def build_container_argv(
    runtime: str, cab: CabDef, argv: list[str], params: dict[str, Any], workdir: str
) -> list[str]:
    """Wrap argv in a container-runtime invocation. Shared with the Slurm
    backend, which runs cabs under apptainer the same way a plain
    ContainerBackend would, just inside a batch job.
    """
    if not cab.image:
        raise BackendError(f"cab '{cab.name}' has no image, cannot run under {runtime}")

    dirs = bind_dirs(cab, params, workdir)

    if runtime in _DOCKER_LIKE:
        mounts = [flag for d in dirs for flag in ("-v", f"{d}:{d}")]
        return [runtime, "run", "--rm", *mounts, "-w", workdir, cab.image, *argv]

    # apptainer
    binds = [flag for d in dirs for flag in ("--bind", f"{d}:{d}")]
    return [runtime, "exec", *binds, "--pwd", workdir, cab.image, *argv]


class ContainerBackend(Backend):
    def __init__(self, runtime: str, workdir: str | None = None):
        if runtime not in _DOCKER_LIKE | _APPTAINER_LIKE:
            raise ValueError(f"unsupported container runtime '{runtime}'")
        self.runtime = runtime
        self.workdir = workdir or os.getcwd()

    def _wrap(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> list[str]:
        return build_container_argv(self.runtime, cab, argv, params, self.workdir)

    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        full_argv = self._wrap(cab, argv, params)
        proc = subprocess.run(full_argv, capture_output=True, text=True)
        lines = proc.stdout.splitlines() + proc.stderr.splitlines()
        outputs = apply_wranglers(cab.wranglers, lines)
        return Result(
            cab_name=cab.name,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            outputs=outputs,
        )


@register
class DockerBackend(ContainerBackend):
    name = "docker"

    def __init__(self, workdir: str | None = None):
        super().__init__("docker", workdir)


@register
class PodmanBackend(ContainerBackend):
    name = "podman"

    def __init__(self, workdir: str | None = None):
        super().__init__("podman", workdir)


@register
class ApptainerBackend(ContainerBackend):
    name = "apptainer"

    def __init__(self, workdir: str | None = None):
        super().__init__("apptainer", workdir)
