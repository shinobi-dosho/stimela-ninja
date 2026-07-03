"""Live integration tests against a real Kubernetes cluster and a real
radio-astronomy tool image. Skipped unless a cluster is reachable via
kubectl and the image is cached locally -- this is not meant to spin up a
cluster or pull a multi-GB image on its own.

To run these locally: `kind create cluster` (with an extraMounts entry
covering wherever your test paths live, so hostPath volumes actually
resolve), then `kind load docker-image quay.io/stimela/wsclean:1.8.0`.
These were verified this way against a real kind cluster during
development; see AGENTS.md.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from shinobi.backends.kubernetes import KubernetesBackend
from shinobi.schema import CabDef, ParamSchema

WSCLEAN_IMAGE = "quay.io/stimela/wsclean:1.8.0"


def _cluster_reachable() -> bool:
    if not shutil.which("kubectl"):
        return False
    return subprocess.run(["kubectl", "cluster-info"], capture_output=True).returncode == 0


def _image_available_locally(image: str) -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


requires_k8s_and_wsclean_image = pytest.mark.skipif(
    not (_cluster_reachable() and _image_available_locally(WSCLEAN_IMAGE)),
    reason=f"no reachable kubectl cluster, or {WSCLEAN_IMAGE} not cached/loaded",
)


@requires_k8s_and_wsclean_image
def test_real_tool_runs_as_a_job():
    cab = CabDef(
        name="wsclean",
        command="wsclean",
        image=WSCLEAN_IMAGE,
        inputs={"version": ParamSchema(dtype="bool")},
    )
    backend = KubernetesBackend()
    result = backend.run(cab, ["wsclean", "--version"], {"version": True})

    assert result.success
    assert "WSClean" in result.stdout


@requires_k8s_and_wsclean_image
def test_host_file_visible_via_hostpath_mount(tmp_path):
    host_file = tmp_path / "hello.txt"
    host_file.write_text("hello from the host, via kind\n")

    cab = CabDef(
        name="probe",
        command="/bin/cat",
        image=WSCLEAN_IMAGE,
        inputs={"path": ParamSchema(dtype="File", required=True)},
    )
    backend = KubernetesBackend()
    result = backend.run(cab, ["/bin/cat", str(host_file)], {"path": str(host_file)})

    assert result.success
    assert result.stdout == "hello from the host, via kind\n"


@requires_k8s_and_wsclean_image
def test_failing_job_reports_real_container_exit_code():
    cab = CabDef(name="fail", command="/bin/sh", image=WSCLEAN_IMAGE)
    backend = KubernetesBackend()
    result = backend.run(cab, ["/bin/sh", "-c", "exit 17"], {})

    assert not result.success
    assert result.returncode == 17


@requires_k8s_and_wsclean_image
def test_job_cleaned_up_after_run():
    cab = CabDef(name="cleanup-check", command="/bin/echo", image=WSCLEAN_IMAGE)
    backend = KubernetesBackend(namespace="default")
    backend.run(cab, ["/bin/echo", "hi"], {})

    proc = subprocess.run(
        ["kubectl", "get", "jobs", "-n", "default", "-o", "name"], capture_output=True, text=True
    )
    assert proc.stdout.strip() == ""
