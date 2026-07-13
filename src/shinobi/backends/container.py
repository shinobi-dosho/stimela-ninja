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
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, register
from shinobi.backends._stream import run_streaming
from shinobi.config import AppConfig
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


def _apptainer_image_uri(image: str) -> str:
    """Resolve a cab's `image` string to something `apptainer exec` accepts.

    docker/podman auto-pull a bare `quay.io/org/img:tag` registry ref, but
    apptainer treats an unschemed string as a local filesystem path and
    fails (`could not open image /cwd/quay.io/...`). Prepend `docker://` so
    the registry ref is pulled (and cached as a SIF) on first use -- unless
    the image is already a URI (`docker://`, `oras://`, `library://`, ...)
    or a local path / `.sif` file the caller supplied deliberately.
    """
    if "://" in image or image.endswith(".sif") or image.startswith((".", "/")):
        return image
    return f"docker://{image}"


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
    run_as_host_user: bool = False,
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

    `run_as_host_user`, for docker/podman only, adds `--user uid:gid` plus
    `HOME=<workdir>` so bind-mounted outputs come out owned by the invoking
    host user instead of root -- the modern equivalent of stimela-classic's
    `/etc/passwd`-bind-mount trick. Setting `HOME` to the (writable,
    bind-mounted) workdir covers the common case of tools that fall back to
    `getpwuid()` only when `$HOME` is unset; tools that need a real
    passwd/nss entry (e.g. `getpwnam`, some MPI stacks) can still fail --
    no bind-mount fully replaces `/etc/passwd` in Linux's user model. No-op
    for apptainer, which already runs as the host user.
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
        user_flags: list[str] = []
        if run_as_host_user:
            user_flags = ["--user", f"{os.getuid()}:{os.getgid()}", "-e", f"HOME={workdir}"]
        return [runtime, "run", "--rm", *user_flags, *mounts, "-w", workdir, image, *argv]

    # apptainer
    binds = [flag for d in dirs for flag in ("--bind", f"{d}:{d}")]
    return [runtime, "exec", *binds, "--pwd", workdir, _apptainer_image_uri(image), *argv]


class ContainerBackend(Backend):
    """Backend that runs cabs via a container runtime (docker/podman/apptainer)."""

    def __init__(
        self,
        runtime: str,
        workdir: str | None = None,
        run_as_host_user: bool | None = None,
    ):
        """Initialize the backend for a given container runtime.

        Args:
            runtime: Runtime binary name, one of `CONTAINER_RUNTIMES`.
            workdir: Working directory to bind-mount and run inside. Defaults
                to the current working directory.
            run_as_host_user: Whether to run docker/podman containers as the
                invoking host user (see `build_container_argv`). Defaults to
                `AppConfig.backend.run_as_host_user` when not given.

        Raises:
            ValueError: If `runtime` is not a supported container runtime.
        """
        if runtime not in CONTAINER_RUNTIMES:
            raise ValueError(f"unsupported container runtime '{runtime}'")
        self.runtime = runtime
        self.workdir = workdir or os.getcwd()
        self.run_as_host_user = (
            AppConfig.load().backend.run_as_host_user
            if run_as_host_user is None
            else run_as_host_user
        )

    def _wrap(self, cab: Cab, argv: list[str], inputs: dict[str, Any]) -> list[str]:
        return build_container_argv(
            self.runtime,
            cab,
            argv,
            inputs,
            self.workdir,
            run_as_host_user=self.run_as_host_user,
        )

    def run(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, label: str = "", stream: bool = True
    ) -> BackendRun:
        """Run a cab's argv inside the configured container runtime.

        Args:
            cab: The cab being executed.
            argv: Resolved command-line arguments to run inside the container.
            inputs: Prepared inputs dict used to derive bind mounts.
            label: Label used for streamed output lines. Defaults to `cab.name`.
            stream: Whether to stream stdout/stderr live as the process runs.

        Returns:
            The completed `BackendRun` (never raises on non-zero exit).
        """
        full_argv = self._wrap(cab, argv, inputs)
        return run_streaming(full_argv, label=label or cab.name, stream=stream)


@register
class DockerBackend(ContainerBackend):
    """Container backend that shells out to `docker`."""

    name = "docker"

    def __init__(self, workdir: str | None = None, run_as_host_user: bool | None = None):
        """Initialize a Docker-backed container backend.

        Args:
            workdir: Working directory to bind-mount and run inside.
            run_as_host_user: Whether to run as the invoking host user.
        """
        super().__init__("docker", workdir, run_as_host_user)


@register
class PodmanBackend(ContainerBackend):
    """Container backend that shells out to `podman`."""

    name = "podman"

    def __init__(self, workdir: str | None = None, run_as_host_user: bool | None = None):
        """Initialize a Podman-backed container backend.

        Args:
            workdir: Working directory to bind-mount and run inside.
            run_as_host_user: Whether to run as the invoking host user.
        """
        super().__init__("podman", workdir, run_as_host_user)


@register
class ApptainerBackend(ContainerBackend):
    """Container backend that shells out to `apptainer`."""

    name = "apptainer"

    def __init__(self, workdir: str | None = None, run_as_host_user: bool | None = None):
        """Initialize an Apptainer-backed container backend.

        Args:
            workdir: Working directory to bind-mount and run inside.
            run_as_host_user: Ignored for apptainer (always runs as host user).
        """
        super().__init__("apptainer", workdir, run_as_host_user)
