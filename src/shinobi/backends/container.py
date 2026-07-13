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

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
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


def _sha256_file(path: str) -> str:
    """Streaming sha256 of a file's contents, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_tag(name: str) -> str:
    """Drop a trailing `:tag` from a repo reference, leaving a registry
    `host:port` intact (a tag is a `:` in the *last* path segment).
    """
    last = name.rsplit("/", 1)[-1]
    return name.rsplit(":", 1)[0] if ":" in last else name


def _with_digest(ref: str, digest: str) -> str:
    """Rewrite a `[scheme://]repo:tag` reference to `[scheme://]repo@digest`
    so the container runtime executes exactly that digest.
    """
    scheme, sep, rest = ref.partition("://")
    if not sep:
        scheme, rest = "", ref
    rest = rest.split("@", 1)[0]  # drop any existing @digest
    return f"{scheme}{sep}{_strip_tag(rest)}@{digest}"


_DOCKER_HUB = "registry-1.docker.io"

# Manifest media types we accept, preferring multi-arch index/list types so
# the digest we pin is the portable top-level one (matching `docker buildx
# imagetools inspect`), not a single platform's manifest.
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def _split_ref(ref: str) -> tuple[str, str, str] | None:
    """Split an image reference into `(registry_host, repository, reference)`,
    or `None` if it isn't a plain docker-registry ref we can resolve over
    HTTP (e.g. an `oras://`/`library://` scheme, or a local path).

    `reference` is a `:tag` (defaulting to `latest`) or an `@sha256:...`
    digest. Bare names resolve to Docker Hub, with the `library/` prefix
    added for official single-name images (`alpine` -> `library/alpine`).
    """
    if "://" in ref:
        scheme, _, ref = ref.partition("://")
        if scheme != "docker":
            return None
    name, reference = ref, "latest"
    if "@" in name:
        name, _, reference = name.partition("@")
    else:
        last = name.rsplit("/", 1)[-1]
        if ":" in last:  # a tag, not a registry :port
            name, _, reference = name.rpartition(":")
    first = name.split("/", 1)[0]
    if "/" in name and ("." in first or ":" in first or first == "localhost"):
        registry, _, repo = name.partition("/")
    else:
        registry = _DOCKER_HUB
        repo = name if "/" in name else f"library/{name}"
    return registry, repo, reference


def _auth_keys(registry: str) -> list[str]:
    """Candidate keys under which `registry`'s credentials might be stored in
    a Docker `config.json` `auths` map (Docker Hub is stored under its legacy
    `index.docker.io/v1/` key).
    """
    keys = [registry, f"https://{registry}", f"https://{registry}/v1/"]
    if registry == _DOCKER_HUB or registry.endswith("docker.io"):
        keys += ["https://index.docker.io/v1/", "index.docker.io", "docker.io"]
    return keys


def _cred_helper_get(helper: str, registry: str) -> tuple[str, str] | None:
    """Query a Docker credential helper (`docker-credential-<helper> get`)
    for `registry`. `None` if the helper is absent or returns no usable
    username/secret. Never raises.
    """
    exe = f"docker-credential-{helper}"
    if not shutil.which(exe):
        return None
    try:
        out = subprocess.run([exe, "get"], input=registry, capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    user, secret = data.get("Username"), data.get("Secret")
    return (user, secret) if user and secret else None


@lru_cache(maxsize=None)
def _docker_config_auth(registry: str) -> tuple[str, str] | None:
    """`(username, password)` for `registry` from the Docker config
    (`$DOCKER_CONFIG/config.json` or `~/.docker/config.json`) -- via a
    per-registry/global credential helper first, then a static base64 `auth`
    entry. `None` if nothing matches. Best-effort and never raises.
    """
    try:
        cfg_dir = os.environ.get("DOCKER_CONFIG") or os.path.join(os.path.expanduser("~"), ".docker")
        path = os.path.join(cfg_dir, "config.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            cfg = json.load(f)
        helper = (cfg.get("credHelpers") or {}).get(registry) or cfg.get("credsStore")
        if helper:
            creds = _cred_helper_get(helper, registry)
            if creds:
                return creds
        auths = cfg.get("auths") or {}
        for key in _auth_keys(registry):
            entry = auths.get(key) or {}
            if entry.get("auth"):
                user, _, pw = base64.b64decode(entry["auth"]).decode().partition(":")
                if user:
                    return user, pw
    except Exception:  # noqa: BLE001 -- credential lookup is best-effort
        return None
    return None


def _basic_auth(creds: tuple[str, str]) -> str:
    """`Authorization: Basic ...` header value for `(username, password)`."""
    return "Basic " + base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()


def _authorize(challenge: str, creds: tuple[str, str] | None) -> str | None:
    """Turn a `WWW-Authenticate` challenge into an `Authorization` header
    value: fetch a bearer token (using `creds` for the token request when the
    repo is private), or send Basic directly. `None` if unsatisfiable.
    """
    scheme = challenge.split(" ", 1)[0].lower()
    if scheme == "bearer":
        token = _bearer_token(challenge, creds)
        return f"Bearer {token}" if token else None
    if scheme == "basic" and creds:
        return _basic_auth(creds)
    return None


def _bearer_token(challenge: str, creds: tuple[str, str] | None = None) -> str | None:
    """Fetch a pull token from a registry's `WWW-Authenticate: Bearer ...`
    challenge (realm + service + scope). When `creds` are given, authenticate
    the token request with Basic so private-repo scopes are granted. `None`
    if the challenge isn't bearer or the request fails.
    """
    if not challenge.lower().startswith("bearer "):
        return None
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = params.get("realm")
    if not realm:
        return None
    query = {k: params[k] for k in ("service", "scope") if k in params}
    url = f"{realm}?{urllib.parse.urlencode(query)}"
    headers = {"Authorization": _basic_auth(creds)} if creds else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- https realm
        data = json.load(resp)
    return data.get("token") or data.get("access_token")


def _manifest_digest_header(url: str, authorization: str | None, method: str) -> tuple[str | None, str | None]:
    """One manifest request. Returns `(digest, challenge)`: the
    `Docker-Content-Digest` header on success, or the `WWW-Authenticate`
    challenge on a 401 so the caller can authorize and retry.
    """
    headers = {"Accept": _MANIFEST_ACCEPT}
    if authorization:
        headers["Authorization"] = authorization
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- https registry
            return resp.headers.get("Docker-Content-Digest"), None
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return None, exc.headers.get("WWW-Authenticate")
        return None, None


@lru_cache(maxsize=None)
def _registry_api_digest(ref: str) -> str | None:
    """Resolve a registry reference to its manifest digest by querying the
    registry's HTTP v2 API directly -- no external binary, no image pull.

    This is the primary resolver: it reads the `Docker-Content-Digest`
    header (the same value skopeo/buildx report), doing the token/Basic
    dance -- with credentials from the Docker config for private repos -- when
    the registry challenges. Best-effort and never raises: returns `None` on
    any parse/network/auth failure, and the caller falls back to skopeo, then
    a runtime-native query.
    """
    try:
        parts = _split_ref(ref)
        if parts is None:
            return None
        registry, repo, reference = parts
        if reference.startswith("sha256:"):
            return reference  # already digest-pinned; no network needed
        creds = _docker_config_auth(registry)
        url = f"https://{registry}/v2/{repo}/manifests/{reference}"
        for method in ("HEAD", "GET"):  # a few registries omit the digest on HEAD
            digest, challenge = _manifest_digest_header(url, None, method)
            if digest is None and challenge:
                authorization = _authorize(challenge, creds)
                if authorization:
                    digest, _ = _manifest_digest_header(url, authorization, method)
            if digest:
                return digest
        return None
    except Exception:  # noqa: BLE001 -- resolution is best-effort
        return None


@lru_cache(maxsize=None)
def _registry_digest(ref: str) -> str | None:
    """`sha256:...` digest of a registry reference, resolved via `skopeo
    inspect` without pulling. `None` if skopeo is unavailable, the ref
    isn't a registry image, or the lookup fails (offline, auth, no such
    tag). Best-effort and never raises. Memoized so a many-step recipe on
    one image does a single registry round-trip.
    """
    if not shutil.which("skopeo"):
        return None
    url = ref if "://" in ref else f"docker://{ref}"
    try:
        out = subprocess.run(
            ["skopeo", "inspect", "--format", "{{.Digest}}", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    digest = out.stdout.strip()
    return digest if out.returncode == 0 and digest.startswith("sha256:") else None


@lru_cache(maxsize=None)
def _docker_digest(runtime: str, image: str) -> str | None:
    """`sha256:...` digest of `image` via `<runtime> buildx imagetools
    inspect --raw` -- the content hash of the manifest bytes, fetched from
    the registry without pulling. This is the docker-native fallback used
    when skopeo isn't installed (buildx ships with modern Docker). `None` on
    any failure. Best-effort and never raises.
    """
    if not shutil.which(runtime):
        return None
    try:
        # No text=True: hash the exact manifest bytes the registry returned.
        out = subprocess.run(
            [runtime, "buildx", "imagetools", "inspect", "--raw", image],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    return "sha256:" + hashlib.sha256(out.stdout).hexdigest()


def _pin_image(runtime: str, image: str) -> tuple[str, str | None]:
    """Resolve `image` to `(ref_to_run, digest)` for pin-then-run.

    `ref_to_run` is what goes into the container argv -- digest-pinned when
    we could resolve one, so what executes is exactly what the manifest
    records. `digest` is the `sha256:...` that ran, or `None` when the image
    can't be pinned (local-built/untagged, offline, no skopeo) -- in which
    case the run proceeds unpinned and the manifest honestly reports it.

    Three cases:
      * a local `.sif` file -- the file itself is the executed artifact, so
        its content hash is a true, reproducible digest;
      * a registry reference (docker/podman, or apptainer's `docker://`
        form) -- resolved to its registry digest and rewritten to run pinned;
      * anything else -- run as-is, digest `None`.
    """
    if image.endswith(".sif") and os.path.isfile(image):
        return image, "sha256:" + _sha256_file(image)
    docker_like = runtime in _DOCKER_LIKE
    ref = image if docker_like else _apptainer_image_uri(image)
    if docker_like or ref.startswith("docker://"):
        # Primary: a pure-Python registry API query (no external tool, works
        # for every runtime incl. apptainer). Fall back to skopeo, then a
        # docker-native manifest query, before giving up (honestly unpinned).
        digest = _registry_api_digest(ref)
        if digest is None:
            digest = _registry_digest(ref)
        if digest is None and docker_like:
            digest = _docker_digest(runtime, image)
        if digest is not None:
            return _with_digest(ref, digest), digest
    return ref, None


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
    pin: bool = False,
) -> tuple[list[str], str | None]:
    """Wrap argv in a container-runtime invocation. Shared with the Slurm
    backend, which runs cabs under apptainer the same way a plain
    ContainerBackend would, just inside a batch job.

    Returns `(argv, image_digest)`. When `pin` is set (provenance enabled),
    the image reference in the argv is digest-pinned via `_pin_image`
    (pin-then-run: what executes is exactly what gets recorded) and
    `image_digest` is that `sha256:...` (or `None` when it couldn't be
    resolved). When `pin` is False (the default), the image runs by its
    original tag and `image_digest` is `None` -- no registry round-trip, no
    behaviour change from a plain container run.

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

    # With provenance on, resolve (and digest-pin) the image before building
    # the argv, so the reference that runs is the one recorded. Otherwise run
    # the image by its original ref -- no resolution, no behaviour change.
    if pin:
        run_ref, digest = _pin_image(runtime, image)
    else:
        run_ref = image if runtime in _DOCKER_LIKE else _apptainer_image_uri(image)
        digest = None

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
        return [runtime, "run", "--rm", *user_flags, *mounts, "-w", workdir, run_ref, *argv], digest

    # apptainer
    binds = [flag for d in dirs for flag in ("--bind", f"{d}:{d}")]
    return [runtime, "exec", *binds, "--pwd", workdir, run_ref, *argv], digest


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

    def _wrap(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, pin: bool = False
    ) -> tuple[list[str], str | None]:
        return build_container_argv(
            self.runtime,
            cab,
            argv,
            inputs,
            self.workdir,
            run_as_host_user=self.run_as_host_user,
            pin=pin,
        )

    def run(
        self,
        cab: Cab,
        argv: list[str],
        inputs: dict[str, Any],
        *,
        label: str = "",
        stream: bool = True,
        pin: bool = False,
    ) -> BackendRun:
        """Run a cab's argv inside the configured container runtime.

        Args:
            cab: The cab being executed.
            argv: Resolved command-line arguments to run inside the container.
            inputs: Prepared inputs dict used to derive bind mounts.
            label: Label used for streamed output lines. Defaults to `cab.name`.
            stream: Whether to stream stdout/stderr live as the process runs.
            pin: Digest-pin the image before running (provenance enabled).

        Returns:
            The completed `BackendRun` (never raises on non-zero exit).
        """
        full_argv, image_digest = self._wrap(cab, argv, inputs, pin=pin)
        run = run_streaming(full_argv, label=label or cab.name, stream=stream)
        run.image_digest = image_digest
        run.containerized = True
        return run


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
