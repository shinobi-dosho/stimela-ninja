from __future__ import annotations

from pathlib import Path

import pytest

from shinobi.dataset_access import DatasetAccess, DatasetFallback, DatasetMode, ResolvedDatasetAccess
from shinobi.dataset_backends import (
    DATASET_BACKEND_CAPABILITY_PROFILE,
    DatasetBackendStatus,
    DatasetBackendMount,
    DatasetNamespaceMode,
    dataset_backend_capability,
    plan_dataset_backend,
)
from shinobi.dataset_closure import ClosureStatus
from shinobi.exceptions import DatasetLifecycleUnavailableError


@pytest.fixture(autouse=True)
def _local_container_configuration(monkeypatch, tmp_path):
    """Keep route capability tests independent of the developer's contexts."""

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "CONTAINER_HOST", "CONTAINER_CONNECTION", "PODMAN_CONNECTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "empty-docker-config"))
    monkeypatch.setenv("PODMAN_CONNECTIONS_CONF", str(tmp_path / "empty-podman-connections.json"))
    monkeypatch.setenv("CONTAINERS_CONF", str(tmp_path / "empty-containers.conf"))


def _access(root: Path, mode: DatasetMode, *, requested: Path | None = None, resources: tuple[Path, ...] | None = None) -> ResolvedDatasetAccess:
    root = root.resolve()
    declaration = DatasetAccess(field="ms", mode=mode)
    return ResolvedDatasetAccess(
        field="ms",
        declaration=declaration,
        requested_path=(requested or root).absolute(),
        root=root,
        resources=resources or (root,),
        mode=mode,
        whole_dataset=True,
        path_known=True,
        columns_known=False,
        fallback=DatasetFallback.UNKNOWN_COLUMNS,
        closure_status=ClosureStatus.VALID,
        reason=f"{mode.value} whole dataset",
    )


@pytest.mark.parametrize("backend", ["native", "venv", "docker", "podman", "apptainer", "singularity"])
@pytest.mark.parametrize("mutation", [False, True])
def test_local_route_capabilities_are_versioned_and_tested(backend, mutation):
    capability = dataset_backend_capability(backend, mutation=mutation)

    assert capability.profile == DATASET_BACKEND_CAPABILITY_PROFILE
    assert capability.status is DatasetBackendStatus.TESTED
    assert capability.namespace_mode is (DatasetNamespaceMode.IDENTITY_BIND if backend in {"docker", "podman", "apptainer", "singularity"} else DatasetNamespaceMode.LOCAL)
    assert capability.lifecycle.endswith("mutation/v1" if mutation else "read/v1")


@pytest.mark.parametrize("backend", ["slurm", "kubernetes"])
def test_distributed_routes_are_explicitly_unavailable_without_a_namespace_mapping(backend):
    capability = dataset_backend_capability(backend, mutation=True)

    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert capability.namespace_mode is DatasetNamespaceMode.UNPROVEN
    assert "namespace" in capability.reason


def test_unknown_route_is_explicitly_unsupported():
    capability = dataset_backend_capability("recording", mutation=False)

    assert capability.status is DatasetBackendStatus.UNSUPPORTED
    assert "no strict dataset" in capability.reason


def test_remote_launch_delegates_identity_to_the_remote_lifecycle():
    capability = dataset_backend_capability("remote", mutation=False)

    assert capability.status is DatasetBackendStatus.TESTED
    assert capability.namespace_mode is DatasetNamespaceMode.REMOTE_DELEGATED
    assert "remote ninja process" in capability.reason


@pytest.mark.parametrize(
    ("backend", "variable", "value"),
    [
        ("docker", "DOCKER_HOST", "ssh://worker.example"),
        ("docker", "DOCKER_CONTEXT", "desktop-linux"),
        ("podman", "CONTAINER_HOST", "ssh://worker.example/run/podman.sock"),
        ("podman", "CONTAINER_CONNECTION", "cluster"),
    ],
)
def test_remote_container_configuration_is_unavailable(backend, variable, value, monkeypatch):
    monkeypatch.setenv(variable, value)

    capability = dataset_backend_capability(backend, mutation=True)

    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert capability.namespace_mode is DatasetNamespaceMode.UNPROVEN
    assert "storage namespace" in capability.reason


def test_nondefault_docker_config_context_is_unavailable(tmp_path, monkeypatch):
    config_dir = tmp_path / "docker"
    config_dir.mkdir()
    (config_dir / "config.json").write_text('{"currentContext": "remote-cluster"}')
    monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))

    capability = dataset_backend_capability("docker", mutation=False)

    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert "remote-cluster" in capability.reason


def test_default_podman_connection_file_is_unavailable(tmp_path, monkeypatch):
    config = tmp_path / "podman-connections.json"
    config.write_text('{"Connection":{"Default":"cluster","Connections":{"cluster":{"URI":"ssh://worker.example/run/podman.sock"}}}}')
    monkeypatch.setenv("PODMAN_CONNECTIONS_CONF", str(config))

    capability = dataset_backend_capability("podman", mutation=False)

    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert "cluster" in capability.reason


def test_default_podman_service_destination_is_unavailable(tmp_path, monkeypatch):
    config = tmp_path / "containers.conf"
    config.write_text('[engine]\nactive_service="cluster"\n[engine.service_destinations.cluster]\nuri="ssh://worker.example/run/podman.sock"\n')
    monkeypatch.setenv("CONTAINERS_CONF", str(config))

    capability = dataset_backend_capability("podman", mutation=False)

    assert capability.status is DatasetBackendStatus.UNAVAILABLE
    assert "cluster" in capability.reason


def test_local_route_needs_no_mount_plan(tmp_path):
    root = tmp_path / "obs.ms"
    root.mkdir()

    plan = plan_dataset_backend("venv", (_access(root, DatasetMode.READ),), workspace=tmp_path, mutation=False)

    assert plan.storage_namespace == tmp_path.resolve()
    assert plan.mounts == ()


def test_container_mapping_is_explicit_and_identity_only(tmp_path):
    source = (tmp_path / "source.ms").resolve()
    target = (tmp_path / "target.ms").resolve()

    with pytest.raises(ValueError, match="identity mounts only"):
        DatasetBackendMount(source=source, target=target, writable=False, reason="mismatch")


def test_container_reader_mounts_the_canonical_root_read_only(tmp_path):
    real = tmp_path / "storage" / "obs.ms"
    real.mkdir(parents=True)
    alias = tmp_path / "alias.ms"
    alias.symlink_to(real)
    subtable = real / "ANTENNA"
    subtable.mkdir()
    access = _access(real, DatasetMode.READ, requested=alias, resources=(real.resolve(), subtable.resolve()))

    plan = plan_dataset_backend("docker", (access,), workspace=tmp_path, mutation=False)

    assert [(mount.source, mount.target, mount.writable) for mount in plan.mounts] == [(real.resolve(), real.resolve(), False)]
    assert "alias.ms" not in str(plan.mounts[0].source)


def test_container_writer_mounts_existing_root_read_write(tmp_path):
    root = tmp_path / "obs.ms"
    root.mkdir()

    plan = plan_dataset_backend("apptainer", (_access(root, DatasetMode.WRITE),), workspace=tmp_path, mutation=True)

    assert [(mount.source, mount.target, mount.writable) for mount in plan.mounts] == [(root.resolve(), root.resolve(), True)]


def test_container_leaf_gets_only_its_declared_mutation_root_read_write(tmp_path):
    left = tmp_path / "left.ms"
    right = tmp_path / "right.ms"
    left.mkdir()
    right.mkdir()
    left_write = _access(left, DatasetMode.WRITE)
    right_write = _access(right, DatasetMode.WRITE)

    plan = plan_dataset_backend(
        "docker",
        (left_write, right_write),
        workspace=tmp_path,
        mutation=True,
        leaf_accesses=(left_write,),
    )

    assert [(mount.source, mount.writable) for mount in plan.mounts] == [(left.resolve(), True), (right.resolve(), False)]


def test_unannotated_container_leaf_gets_every_existing_root_read_only(tmp_path):
    left = tmp_path / "left.ms"
    right = tmp_path / "right.ms"
    left.mkdir()
    right.mkdir()
    workflow = (_access(left, DatasetMode.WRITE), _access(right, DatasetMode.WRITE))

    plan = plan_dataset_backend("podman", workflow, workspace=tmp_path, mutation=True, leaf_accesses=())

    assert [(mount.source, mount.writable) for mount in plan.mounts] == [(left.resolve(), False), (right.resolve(), False)]


def test_container_create_mounts_the_existing_parent_read_write(tmp_path):
    target = tmp_path / "new.ms"

    plan = plan_dataset_backend("podman", (_access(target, DatasetMode.CREATE),), workspace=tmp_path, mutation=True)

    assert [(mount.source, mount.target, mount.writable) for mount in plan.mounts] == [(tmp_path.resolve(), tmp_path.resolve(), True)]


def test_container_create_refuses_a_missing_parent(tmp_path):
    target = tmp_path / "missing" / "new.ms"
    access = _access(target, DatasetMode.CREATE)

    with pytest.raises(DatasetLifecycleUnavailableError, match="no existing directory parent"):
        plan_dataset_backend("docker", (access,), workspace=tmp_path, mutation=True)


def test_non_owner_leaf_cannot_receive_an_absent_create_root(tmp_path):
    target = tmp_path / "new.ms"
    access = _access(target, DatasetMode.CREATE)

    with pytest.raises(DatasetLifecycleUnavailableError, match="only its declaring CREATE leaf"):
        plan_dataset_backend("docker", (access,), workspace=tmp_path, mutation=True, leaf_accesses=())


def test_container_plan_refuses_external_closure_resources(tmp_path):
    root = tmp_path / "obs.ms"
    external = tmp_path / "shared" / "ANTENNA"
    root.mkdir()
    external.mkdir(parents=True)
    access = _access(root, DatasetMode.READ, resources=(root.resolve(), external.resolve()))

    with pytest.raises(DatasetLifecycleUnavailableError, match="external closure resources"):
        plan_dataset_backend("docker", (access,), workspace=tmp_path, mutation=False)


def test_unavailable_route_refuses_before_a_plan_is_returned(tmp_path):
    root = tmp_path / "obs.ms"
    root.mkdir()

    with pytest.raises(DatasetLifecycleUnavailableError, match="route is unavailable"):
        plan_dataset_backend("slurm", (_access(root, DatasetMode.READ),), workspace=tmp_path, mutation=False)


def test_remote_launch_cannot_be_used_as_a_local_backend_plan(tmp_path):
    root = tmp_path / "obs.ms"
    root.mkdir()

    with pytest.raises(DatasetLifecycleUnavailableError, match="CLI launch surface"):
        plan_dataset_backend("remote", (_access(root, DatasetMode.READ),), workspace=tmp_path, mutation=False)
