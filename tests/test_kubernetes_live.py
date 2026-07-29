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

from pathlib import Path
from typing import Optional

from pydantic import Field, create_model

from shinobi.backends.kubernetes import KubernetesBackend
from shinobi.exceptions import BackendError
from shinobi.loaders import build_model
from shinobi.steps.schema import Cab, ParamMeta

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
    cab = Cab(
        name="wsclean",
        command="wsclean",
        image=WSCLEAN_IMAGE,
        inputs_model=build_model("In", {"version": ("bool", False, None)}),
        outputs_model=build_model("Out", {}),
    )
    backend = KubernetesBackend(namespace="default")
    result = backend.run(cab, ["wsclean", "--version"], {"version": True})

    assert result.success
    assert "WSClean" in result.stdout


@requires_k8s_and_wsclean_image
def test_host_file_visible_via_hostpath_mount(tmp_path):
    host_file = tmp_path / "hello.txt"
    host_file.write_text("hello from the host, via kind\n")

    cab = Cab(
        name="probe",
        command="/bin/cat",
        image=WSCLEAN_IMAGE,
        inputs_model=build_model("In", {"path": ("File", True, None)}),
        outputs_model=build_model("Out", {}),
    )
    backend = KubernetesBackend(namespace="default")
    result = backend.run(cab, ["/bin/cat", str(host_file)], {"path": str(host_file)})

    assert result.success
    assert result.stdout == "hello from the host, via kind\n"


@requires_k8s_and_wsclean_image
def test_failing_job_reports_real_container_exit_code():
    cab = Cab(name="fail", command="/bin/sh", image=WSCLEAN_IMAGE, inputs_model=build_model("In", {}), outputs_model=build_model("Out", {}))
    backend = KubernetesBackend(namespace="default")
    result = backend.run(cab, ["/bin/sh", "-c", "exit 17"], {})

    assert not result.success
    assert result.returncode == 17


@requires_k8s_and_wsclean_image
def test_job_cleaned_up_after_run():
    cab = Cab(name="cleanup-check", command="/bin/echo", image=WSCLEAN_IMAGE, inputs_model=build_model("In", {}), outputs_model=build_model("Out", {}))
    backend = KubernetesBackend(namespace="default")
    backend.run(cab, ["/bin/echo", "hi"], {})

    proc = subprocess.run(["kubectl", "get", "jobs", "-n", "default", "-o", "name"], capture_output=True, text=True)
    assert proc.stdout.strip() == ""


# -- nested volumeMount shadowing --------------------------------------------
#
# The container backends resolve a write target colliding with a
# `writable: false` input by nesting: the directory is mounted read-write for
# the product, the input re-asserted `readOnly` at its own path inside it.
# That only holds if a kubelet shadows nested volumeMounts the way docker,
# podman and apptainer shadow nested binds. This is the test that says it does
# -- run it before trusting the k8s backend with a cab of that shape.

BUSYBOX = "docker.io/library/busybox:latest"

requires_k8s = pytest.mark.skipif(not _cluster_reachable(), reason="no reachable kubectl cluster")


def _node_can_see(directory) -> bool:
    """Whether the node running the pods actually has `directory` -- a
    `hostPath` volume resolves on the *node*, and one whose path is missing
    there is silently created as an empty directory rather than failing. So
    read a sentinel back through a pod instead of assuming."""
    sentinel = directory / ".node-visibility-probe"
    sentinel.write_text("visible")
    probe = Cab(name="visibility-probe", command="/bin/cat", image=BUSYBOX, inputs_model=build_model("In", {"path": ("File", True, None)}), outputs_model=build_model("Out", {}))
    try:
        result = KubernetesBackend(namespace="default").run(probe, ["/bin/cat", str(sentinel)], {"path": str(sentinel)})
    except BackendError:
        return False
    finally:
        sentinel.unlink(missing_ok=True)
    return result.stdout.strip() == "visible"


@requires_k8s
def test_nested_readonly_volumemount_shadows_its_writable_parent(tmp_path):
    """A pod gets `<dir>` read-write and `<dir>/in.txt` `readOnly` nested
    inside it. The read-only file must stay readable, refuse the write, and
    survive on the host, while a product elsewhere in the directory lands.

    Needs the cluster's node to see `tmp_path` (kind: `extraMounts`), and
    pulls a ~4MB busybox rather than the multi-GB tool image the rest of this
    file uses. kind refuses to `extraMounts` `/tmp` itself (its node image
    already mounts one), so point pytest at a mountable root instead:

        kind create cluster --config <extraMounts /tmp/kind-nested>
        pytest tests/test_kubernetes_live.py --basetemp=/tmp/kind-nested/pytest

    Verified passing this way on kind (kubelet shadows the nested mount, as
    docker/podman/apptainer do). If the node cannot see the path, the test
    *skips* rather than fails: a `hostPath` with no `type` silently becomes an
    empty directory the kubelet creates on the node, which would otherwise
    read as this test disproving the shadowing it exists to confirm.
    """
    protected = tmp_path / "in.txt"
    protected.write_text("original")
    if not _node_can_see(tmp_path):
        pytest.skip(f"the cluster's node cannot see {tmp_path} -- needs a kind extraMounts entry covering it (see the docstring)")

    cab = Cab(
        name="nested-probe",
        command="/bin/sh",
        image=BUSYBOX,
        inputs_model=create_model(
            "In",
            prefix=(Optional[str], None),
            ref=(Optional[Path], Field(None, json_schema_extra={"writable": False})),
        ),
        outputs_model=build_model("Out", {"img": ("File", False, None)}),
        field_meta={"img": ParamMeta(implicit="{prefix}-image.fits")},
    )
    inputs = {"ref": str(protected), "prefix": f"{tmp_path}/img"}
    script = f'cat {protected}; echo tampered > {protected} 2>/dev/null && echo "RO-BROKEN" || echo "ro-enforced"; echo product > {tmp_path}/out.txt && echo "product-ok"'
    result = KubernetesBackend(namespace="default").run(cab, ["/bin/sh", "-c", script], inputs)

    assert "original" in result.stdout  # read-through works
    assert "ro-enforced" in result.stdout and "RO-BROKEN" not in result.stdout
    assert "product-ok" in result.stdout
    assert protected.read_text() == "original"  # and the host file is untouched
