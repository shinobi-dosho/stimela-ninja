"""Container backend: wraps a cab's argv in a container-runtime invocation.

Shells out to the runtime binary (docker/podman/apptainer) rather than
using each runtime's Python SDK -- one code path for all of them, no extra
heavyweight client dependencies, and it works uniformly for runtimes (like
apptainer) that don't have a good Python API to begin with.

Volume mounts are derived from the cab's own schema: any input field whose
type is file-like (``pathlib.Path`` -- see ``path_fields``) has its parent
directory bind-mounted at the same path inside the container, so the tool
sees the same paths the caller used. This is why Backend.run() is handed
the validated inputs model, not just argv -- argv is just strings by that
point, with no memory of which of them are paths.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, register
from shinobi.exceptions import BackendError
from shinobi.loaders._modelgen import is_file_dtype
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab, Scope, path_fields

_DOCKER_LIKE = {"docker", "podman"}
_APPTAINER_LIKE = {"apptainer"}

# The one authoritative set of container-runtime backend names; also
# consumed by the pystep adapter (steps/pyfunc.py) to decide whether a
# resolved backend means "run this function in a container".
CONTAINER_RUNTIMES = frozenset(_DOCKER_LIKE | _APPTAINER_LIKE)


def bind_dirs(scope: Scope, inputs: dict[str, Any], workdir: str) -> list[str]:
    """Parent directories of every File/MS-valued input, plus the working
    directory itself. Order-preserving, de-duplicated.

    Covers both declared fields (via `path_fields`, which inspects the
    type annotation) and dynamically-named `ParamPattern` inputs (which
    have no declared field/annotation -- `cab.match_pattern` and
    `ParamMeta.dtype` are the only way to tell those are file-like).
    Pattern-matched inputs are only checked for `Cab` scopes (which have
    `match_pattern`); bare `Scope` instances (e.g. from `@shinobi.pystep`)
    have fully-typed signatures so all inputs are declared.
    """
    dirs = [workdir]
    seen = {workdir}
    declared = path_fields(scope.inputs_model)
    # Only Cabs carry dynamically-named `ParamPattern` inputs; bare Scopes
    # (e.g. from `@shinobi.pystep`) are fully typed, so every input is
    # already in `declared`.
    match_pattern = scope.match_pattern if isinstance(scope, Cab) else None

    for name, value in inputs.items():
        if name not in declared:
            if match_pattern is None:
                continue
            meta = match_pattern(name)
            if meta is None or meta.dtype is None or not is_file_dtype(meta.dtype):
                continue
        if value is None:
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
    runtime: str,
    scope: Scope,
    argv: list[str],
    inputs: dict[str, Any],
    workdir: str,
    *,
    extra_dirs: list[str] | None = None,
) -> list[str]:
    """Wrap argv in a container-runtime invocation. Shared with the Slurm
    backend, which runs cabs under apptainer the same way a plain
    ContainerBackend would, just inside a batch job.

    `scope` is the Cab or Scope being executed. For Cabs, bind mounts are
    derived from path-typed inputs. For bare Scopes (e.g. pysteps), all
    inputs are declared so no pattern matching is needed. The container
    image comes from `scope.image` in both cases.

    `extra_dirs` adds additional bind-mount directories beyond those
    derived from the scope's path-typed inputs (e.g. a pystep's runner
    script directory and source module directory).
    """
    image = scope.image
    if not image:
        name = getattr(scope, "name", "<scope>")
        raise BackendError(f"'{name}' has no image, cannot run under {runtime}")

    dirs = bind_dirs(scope, inputs, workdir)
    if extra_dirs:
        seen = set(dirs)
        for d in extra_dirs:
            if d not in seen:
                seen.add(d)
                dirs.append(d)

    if runtime in _DOCKER_LIKE:
        mounts = [flag for d in dirs for flag in ("-v", f"{d}:{d}")]
        return [runtime, "run", "--rm", *mounts, "-w", workdir, image, *argv]

    # apptainer
    binds = [flag for d in dirs for flag in ("--bind", f"{d}:{d}")]
    return [runtime, "exec", *binds, "--pwd", workdir, image, *argv]


class ContainerBackend(Backend):
    def __init__(self, runtime: str, workdir: str | None = None):
        if runtime not in CONTAINER_RUNTIMES:
            raise ValueError(f"unsupported container runtime '{runtime}'")
        self.runtime = runtime
        self.workdir = workdir or os.getcwd()

    def _wrap(self, cab: Cab, argv: list[str], inputs: dict[str, Any]) -> list[str]:
        return build_container_argv(self.runtime, cab, argv, inputs, self.workdir)

    def run(self, cab: Cab, argv: list[str], inputs: dict[str, Any]) -> BackendRun:
        full_argv = self._wrap(cab, argv, inputs)
        proc = subprocess.run(full_argv, capture_output=True, text=True)
        return BackendRun(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


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
