"""Backend capability and namespace plans for strict dataset execution.

The dataset lifecycle owns validation, claims, observations and recovery.  A
backend adapter answers the narrower question the lifecycle cannot infer from
a backend name alone: does this execution route see the same storage
namespace, and which closure paths must be made visible there?

Only routes with a ``tested`` result may execute a strict dataset contract.
An ``unavailable`` route is a real backend whose namespace/recovery contract
has not been supplied; ``unsupported`` means the route cannot provide the
contract in its present form.  These are durable capability observations, not
feature flags, and a route never degrades a :class:`MeasurementSetV2` to an
opaque path.
"""

from __future__ import annotations

import json
import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from shinobi.dataset_access import DatasetMode, ResolvedDatasetAccess
from shinobi.exceptions import DatasetLifecycleUnavailableError


DATASET_BACKEND_CAPABILITY_PROFILE = "msv2-backend-route/v1"
DATASET_READ_CAPABILITY = "contained-native-msv2-read/v1"
DATASET_MUTATION_CAPABILITY = "contained-native-msv2-mutation/v1"
_LOCAL_ROUTES = frozenset({"native", "venv"})
_IDENTITY_BIND_ROUTES = frozenset({"docker", "podman", "apptainer", "singularity"})


def _container_namespace_issue(backend: str) -> str | None:
    """Why a daemon-backed runtime is not proven local, if applicable.

    Bind sources are resolved by the daemon, not by the CLI process.  A
    remote Docker context/host or Podman connection can therefore give the
    container an identically spelled but different tree from the one the
    lifecycle claimed and inspected.  Local Unix sockets are admissible;
    every other explicit endpoint fails closed.
    """

    if backend == "docker":
        host = os.environ.get("DOCKER_HOST", "").strip()
        if host and not host.startswith("unix://"):
            return f"DOCKER_HOST={host!r} does not identify a local Unix-socket daemon"
        context = os.environ.get("DOCKER_CONTEXT", "").strip()
        if context and context != "default":
            return f"DOCKER_CONTEXT={context!r} is not the local default context"
        if host or context == "default":
            return None
        config_dir = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker"))
        config = config_dir / "config.json"
        if config.is_file():
            try:
                current = json.loads(config.read_text()).get("currentContext", "")
            except (OSError, ValueError, AttributeError) as exc:
                return f"Docker context configuration {config} could not be validated: {exc}"
            if current and current != "default":
                return f"Docker current context {current!r} is not the local default context"
    elif backend == "podman":
        connection = (os.environ.get("CONTAINER_CONNECTION") or os.environ.get("PODMAN_CONNECTION") or "").strip()
        if connection:
            return f"Podman connection {connection!r} has no proven local filesystem namespace"
        host = os.environ.get("CONTAINER_HOST", "").strip()
        if host and not host.startswith("unix://"):
            return f"CONTAINER_HOST={host!r} does not identify a local Unix-socket service"
        if host:
            return None
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        connections_path = Path(os.environ.get("PODMAN_CONNECTIONS_CONF", config_home / "containers" / "podman-connections.json"))
        default_connection = ""
        destinations: dict[str, object] = {}
        if connections_path.is_file():
            try:
                connection_data = json.loads(connections_path.read_text()).get("Connection", {})
                default_connection = str(connection_data.get("Default", ""))
                destinations.update(connection_data.get("Connections", {}))
            except (OSError, ValueError, AttributeError, TypeError) as exc:
                return f"Podman connection configuration {connections_path} could not be validated: {exc}"

        explicit_conf = os.environ.get("CONTAINERS_CONF")
        if explicit_conf:
            config_paths = [Path(explicit_conf)]
        else:
            user_dir = config_home / "containers"
            config_paths = [
                Path("/usr/share/containers/containers.conf"),
                Path("/etc/containers/containers.conf"),
                *sorted(Path("/etc/containers/containers.conf.d").glob("*.conf")),
                Path("/etc/containers/containers.rootless.conf"),
                user_dir / "containers.conf",
                *sorted((user_dir / "containers.conf.d").glob("*.conf")),
            ]
        active_service = ""
        for path in config_paths:
            if not path.is_file():
                continue
            try:
                engine = tomllib.loads(path.read_text()).get("engine", {})
                if "active_service" in engine:
                    active_service = str(engine["active_service"])
                configured = engine.get("service_destinations", {})
                if not isinstance(configured, dict):
                    return f"Podman service destinations in {path} are not a mapping"
                destinations.update(configured)
            except (OSError, tomllib.TOMLDecodeError, AttributeError, TypeError) as exc:
                return f"Podman engine configuration {path} could not be validated: {exc}"
        selected = default_connection or active_service
        if selected:
            destination = destinations.get(selected)
            uri = (destination.get("URI") or destination.get("uri") or "") if isinstance(destination, dict) else ""
            if not isinstance(uri, str) or not uri.startswith("unix://"):
                return f"Podman default connection {selected!r} does not identify a local Unix-socket service"
    return None


class DatasetBackendStatus(str, Enum):
    """Evidence state for one concrete backend/lifecycle route."""

    TESTED = "tested"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class DatasetNamespaceMode(str, Enum):
    """How host paths relate to paths observed by the executor."""

    LOCAL = "local"
    IDENTITY_BIND = "identity-bind"
    SHARED_IDENTITY = "shared-identity"
    REMOTE_DELEGATED = "remote-delegated"
    UNPROVEN = "unproven"


class DatasetBackendCapability(BaseModel):
    """Versioned capability decision for one backend and lifecycle kind.

    ``inspection_location`` and ``recovery_location`` are explicit because a
    scheduler accepting a job on one host does not prove that its compute
    node sees the same path or that restoring the submit host repairs what the
    task changed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profile: Literal["msv2-backend-route/v1"] = DATASET_BACKEND_CAPABILITY_PROFILE
    backend: str
    lifecycle: Literal["contained-native-msv2-read/v1", "contained-native-msv2-mutation/v1"]
    status: DatasetBackendStatus
    namespace_mode: DatasetNamespaceMode
    execution_location: str
    inspection_location: str
    recovery_location: str
    reason: str


class DatasetBackendMount(BaseModel):
    """One identity bind required in addition to schema-derived mounts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Path
    target: Path
    writable: bool
    reason: str

    @model_validator(mode="after")
    def _identity_mapping(self) -> "DatasetBackendMount":
        if not self.source.is_absolute() or not self.target.is_absolute():
            raise ValueError("strict dataset mount endpoints must be absolute")
        if self.source != self.target:
            raise ValueError("the local container adapter accepts identity mounts only")
        return self


class DatasetBackendPlan(BaseModel):
    """Closed, data-only plan handed to a local backend adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    capability: DatasetBackendCapability
    storage_namespace: Path
    mounts: tuple[DatasetBackendMount, ...] = ()


def dataset_backend_capability(backend: str, *, mutation: bool) -> DatasetBackendCapability:
    """Return the explicit capability decision for ``backend``.

    Local processes and virtualenvs share the lifecycle controller's
    namespace directly. The four local container runtimes use identity bind
    mounts augmented from the resolved closure plan; daemon-backed runtimes
    are unavailable when their configuration does not prove a local daemon.
    Slurm and Kubernetes deliberately remain unavailable: neither carries a
    validated submit-host-to-executor namespace mapping or runs observation
    and recovery at the execution authority.  ``remote`` is a launch surface,
    not a local backend; the remote ``ninja run`` must make its own decision.
    """

    lifecycle = DATASET_MUTATION_CAPABILITY if mutation else DATASET_READ_CAPABILITY
    if backend in _LOCAL_ROUTES:
        location = "local host" if backend == "native" else "local host virtualenv"
        return DatasetBackendCapability(
            backend=backend,
            lifecycle=lifecycle,
            status=DatasetBackendStatus.TESTED,
            namespace_mode=DatasetNamespaceMode.LOCAL,
            execution_location=location,
            inspection_location="local lifecycle controller",
            recovery_location="local lifecycle controller",
            reason="execution, under-claim inspection and exact recovery use one local filesystem namespace",
        )
    if backend in _IDENTITY_BIND_ROUTES:
        namespace_issue = _container_namespace_issue(backend)
        if namespace_issue is not None:
            return DatasetBackendCapability(
                backend=backend,
                lifecycle=lifecycle,
                status=DatasetBackendStatus.UNAVAILABLE,
                namespace_mode=DatasetNamespaceMode.UNPROVEN,
                execution_location=f"configured {backend} executor",
                inspection_location="local lifecycle controller",
                recovery_location="local lifecycle controller",
                reason=f"container bind sources may resolve outside the lifecycle controller's storage namespace: {namespace_issue}",
            )
        return DatasetBackendCapability(
            backend=backend,
            lifecycle=lifecycle,
            status=DatasetBackendStatus.TESTED,
            namespace_mode=DatasetNamespaceMode.IDENTITY_BIND,
            execution_location=f"local {backend} container",
            inspection_location="local lifecycle controller",
            recovery_location="local lifecycle controller",
            reason="the resolved contained closure is identity-bind-mounted at its canonical host paths; inspection and recovery remain on the host",
        )
    if backend == "slurm":
        return DatasetBackendCapability(
            backend=backend,
            lifecycle=lifecycle,
            status=DatasetBackendStatus.UNAVAILABLE,
            namespace_mode=DatasetNamespaceMode.UNPROVEN,
            execution_location="Slurm compute node",
            inspection_location="submission host",
            recovery_location="submission host",
            reason="no validated submission-host to compute-node storage namespace mapping or compute-side lifecycle adapter is configured",
        )
    if backend == "kubernetes":
        return DatasetBackendCapability(
            backend=backend,
            lifecycle=lifecycle,
            status=DatasetBackendStatus.UNAVAILABLE,
            namespace_mode=DatasetNamespaceMode.UNPROVEN,
            execution_location="Kubernetes pod node",
            inspection_location="client host",
            recovery_location="client host",
            reason="hostPath spelling does not prove that the scheduled node exposes the claimed storage namespace, and pod-side lifecycle validation is not implemented",
        )
    if backend == "remote":
        return DatasetBackendCapability(
            backend=backend,
            lifecycle=lifecycle,
            status=DatasetBackendStatus.TESTED,
            namespace_mode=DatasetNamespaceMode.REMOTE_DELEGATED,
            execution_location="remote ninja process",
            inspection_location="remote lifecycle controller",
            recovery_location="remote lifecycle controller",
            reason="the launcher makes no local dataset claim; the remote ninja process replans paths and applies the capability of its selected backend in the remote namespace",
        )
    return DatasetBackendCapability(
        backend=backend,
        lifecycle=lifecycle,
        status=DatasetBackendStatus.UNSUPPORTED,
        namespace_mode=DatasetNamespaceMode.UNPROVEN,
        execution_location="unknown",
        inspection_location="unknown",
        recovery_location="unknown",
        reason="the backend has no strict dataset namespace/lifecycle adapter",
    )


def worker_dataset_backend_capability(*, mutation: bool) -> DatasetBackendCapability:
    """Capability supplied by a short-lived shared-storage Slurm worker.

    This is deliberately distinct from the blocking ``slurm`` step backend.
    The submission host does not inspect or recover a worker's dataset.  The
    worker re-resolves the frozen identity mapping, observes, executes and
    recovers in one compute-node namespace while the detached workflow's
    durable shared-storage claim remains active.
    """

    return DatasetBackendCapability(
        backend="slurm-worker",
        lifecycle=DATASET_MUTATION_CAPABILITY if mutation else DATASET_READ_CAPABILITY,
        status=DatasetBackendStatus.TESTED,
        namespace_mode=DatasetNamespaceMode.SHARED_IDENTITY,
        execution_location="Slurm compute-node worker",
        inspection_location="same Slurm compute-node worker",
        recovery_location="same Slurm compute-node worker or detached finalizer",
        reason=(
            "the immutable plan records an identity mapping for shared storage; "
            "the compute worker re-resolves it under the workflow claim and performs lifecycle observation and recovery locally"
        ),
    )


def plan_dataset_backend(
    backend: str,
    accesses: tuple[ResolvedDatasetAccess, ...],
    *,
    workspace: Path,
    mutation: bool,
    leaf_accesses: tuple[ResolvedDatasetAccess, ...] | None = None,
) -> DatasetBackendPlan:
    """Build the backend-specific namespace plan for resolved accesses.

    ``accesses`` is the shared workflow closure plan. ``leaf_accesses`` is the
    current leaf's exact declared subset (and defaults to ``accesses`` for
    direct callers). Every existing workflow root is mounted read-only, then
    only roots written or created by this leaf are upgraded. The lifecycle
    has already restricted execution to contained single-root closures, but
    this function validates that invariant again at the adapter boundary:
    mounting an incomplete closure would turn a policy bug into a successful
    tool run against the wrong dataset.
    """

    capability = dataset_backend_capability(backend, mutation=mutation)
    if capability.status is not DatasetBackendStatus.TESTED:
        raise DatasetLifecycleUnavailableError(
            f"contained MSv2 execution refused: backend {backend!r} route is {capability.status.value} under {capability.profile}: {capability.reason}"
        )

    namespace = workspace.resolve()
    if capability.namespace_mode is DatasetNamespaceMode.REMOTE_DELEGATED:
        raise DatasetLifecycleUnavailableError(
            "contained MSv2 execution refused: 'remote' is a CLI launch surface, not a local step backend; the remote ninja process must select and prove its own backend"
        )
    if capability.namespace_mode is not DatasetNamespaceMode.IDENTITY_BIND:
        return DatasetBackendPlan(capability=capability, storage_namespace=namespace)

    leaf_accesses = accesses if leaf_accesses is None else leaf_accesses
    roots: dict[Path, set[str]] = {}
    modes: dict[Path, bool] = {}
    reasons: dict[Path, set[str]] = {}
    for access in accesses:
        if not access.path_known or access.root is None:
            raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: backend {backend!r} cannot mount dataset field {access.field!r} without a resolved root")
        root = access.root.resolve()
        resources = tuple(Path(resource).resolve() for resource in access.resources)
        if any(resource != root and not resource.is_relative_to(root) for resource in resources):
            external = sorted((resource for resource in resources if resource != root and not resource.is_relative_to(root)), key=str)
            raise DatasetLifecycleUnavailableError(
                f"contained MSv2 execution refused: backend {backend!r} will not identity-mount external closure resources: " + ", ".join(map(str, external))
            )
        roots.setdefault(root, set()).add(f"{access.mode.value} {access.field}")

    writable_roots: set[Path] = set()
    for access in leaf_accesses:
        if not access.path_known or access.root is None:
            raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: backend {backend!r} cannot grant leaf access for unresolved dataset field {access.field!r}")
        root = access.root.resolve()
        if root not in roots:
            raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: backend {backend!r} leaf dataset root {root} is outside the shared workflow closure plan")
        if access.mode is not DatasetMode.READ:
            writable_roots.add(root)

    for root, declarations in roots.items():
        writable = root in writable_roots
        if root.exists():
            mount = root
        elif writable:
            mount = root.parent
            if not mount.is_dir():
                raise DatasetLifecycleUnavailableError(
                    f"contained MSv2 execution refused: backend {backend!r} CREATE target {root} has no existing directory parent to identity-mount"
                )
        else:
            raise DatasetLifecycleUnavailableError(
                f"contained MSv2 execution refused: backend {backend!r} cannot protect absent planned dataset root {root} from this leaf; only its declaring CREATE leaf may receive the writable parent mount"
            )
        if mount.parent == mount:
            raise DatasetLifecycleUnavailableError(f"contained MSv2 execution refused: backend {backend!r} would need to mount filesystem root for dataset closure {root}")
        modes[mount] = modes.get(mount, False) or writable
        permission = "leaf write/create" if writable else "leaf read-only"
        reasons.setdefault(mount, set()).add(f"{permission} closure at {root} ({', '.join(sorted(declarations))})")

    mounts = tuple(
        DatasetBackendMount(source=path, target=path, writable=modes[path], reason="; ".join(sorted(reasons[path])))
        for path in sorted(modes, key=lambda item: (len(item.parts), str(item)))
    )
    return DatasetBackendPlan(capability=capability, storage_namespace=namespace, mounts=mounts)


__all__ = [
    "DATASET_BACKEND_CAPABILITY_PROFILE",
    "DATASET_MUTATION_CAPABILITY",
    "DATASET_READ_CAPABILITY",
    "DatasetBackendCapability",
    "DatasetBackendMount",
    "DatasetBackendPlan",
    "DatasetBackendStatus",
    "DatasetNamespaceMode",
    "dataset_backend_capability",
    "plan_dataset_backend",
    "worker_dataset_backend_capability",
]
