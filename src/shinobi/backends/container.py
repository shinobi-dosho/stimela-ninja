"""Container backend: wraps a cab's argv in a container-runtime invocation.

Shells out to the runtime binary (docker/podman/apptainer/singularity)
rather than using each runtime's Python SDK -- one code path for all of
them, no extra heavyweight client dependencies, and it works uniformly for
runtimes (like apptainer) that don't have a good Python API to begin with.

This is a first-pass scaffold: it mounts the current working directory and
nothing else. Real volume-mount policy (input/output dirs, MS dirs, etc.)
is a follow-up -- deliberately not designed here until a real pipeline
exercises the requirements.
"""

from __future__ import annotations

import os
import subprocess

from shinobi.backends import Backend, register
from shinobi.exceptions import BackendError
from shinobi.results import Result
from shinobi.schema import CabDef
from shinobi.wranglers import apply_wranglers

_DOCKER_LIKE = {"docker", "podman"}
_SINGULARITY_LIKE = {"apptainer", "singularity"}


class ContainerBackend(Backend):
    def __init__(self, runtime: str, workdir: str | None = None):
        if runtime not in _DOCKER_LIKE | _SINGULARITY_LIKE:
            raise ValueError(f"unsupported container runtime '{runtime}'")
        self.runtime = runtime
        self.workdir = workdir or os.getcwd()

    def _wrap(self, cab: CabDef, argv: list[str]) -> list[str]:
        if not cab.image:
            raise BackendError(f"cab '{cab.name}' has no image, cannot run under {self.runtime}")

        if self.runtime in _DOCKER_LIKE:
            return [
                self.runtime,
                "run",
                "--rm",
                "-v",
                f"{self.workdir}:{self.workdir}",
                "-w",
                self.workdir,
                cab.image,
                *argv,
            ]

        # apptainer/singularity
        return [
            self.runtime,
            "exec",
            "--bind",
            f"{self.workdir}:{self.workdir}",
            "--pwd",
            self.workdir,
            cab.image,
            *argv,
        ]

    def run(self, cab: CabDef, argv: list[str]) -> Result:
        full_argv = self._wrap(cab, argv)
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
