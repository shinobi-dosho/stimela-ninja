import json
import subprocess

import pytest

from shinobi.backends.kubernetes import KubernetesBackend
from shinobi.exceptions import BackendError
from shinobi.loaders import build_model
from shinobi.steps.schema import Cab


def make_cab(fields=None, image="tool:latest") -> Cab:
    return Cab(
        name="tool",
        command="tool",
        image=image,
        inputs_model=build_model("In", fields or {}),
        outputs_model=build_model("Out", {}),
    )


def cp(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_manifest_is_a_job_with_image_and_command():
    backend = KubernetesBackend(namespace="ns", workdir="/work")
    manifest = backend._manifest(make_cab(), ["tool", "--x", "1"], {}, "job-abc")
    assert manifest["kind"] == "Job"
    assert manifest["metadata"] == {"name": "job-abc", "namespace": "ns"}
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "tool:latest"
    assert container["command"] == ["tool", "--x", "1"]
    assert container["workingDir"] == "/work"


def test_manifest_no_image_raises():
    with pytest.raises(BackendError):
        KubernetesBackend()._manifest(make_cab(image=None), ["tool"], {}, "job-abc")


def test_manifest_mounts_file_like_params_as_hostpath_volumes():
    backend = KubernetesBackend(workdir="/work")
    cab = make_cab({"ms": ("MS", False, None)})
    manifest = backend._manifest(cab, ["tool"], {"ms": "/data/foo.ms"}, "job-abc")
    spec = manifest["spec"]["template"]["spec"]
    hostpaths = {v["hostPath"]["path"] for v in spec["volumes"]}
    assert hostpaths == {"/work", "/data"}
    mountpaths = {m["mountPath"] for m in spec["containers"][0]["volumeMounts"]}
    assert mountpaths == {"/work", "/data"}


def test_run_end_to_end_returns_backendrun(monkeypatch):
    backend = KubernetesBackend(poll_interval=0)
    calls = []

    def fake_kubectl(self, *args, input=None):
        calls.append(args)
        if args[0] == "apply":
            return cp(returncode=0)
        if args[:2] == ("get", "job"):
            return cp(json.dumps({"status": {"conditions": [{"type": "Complete", "status": "True"}]}}))
        if args[:2] == ("get", "pods"):
            return cp("pod-xyz")
        if args[0] == "logs":
            return cp("answer=42\n")
        if args[0] == "delete":
            return cp(returncode=0)
        raise AssertionError(f"unexpected kubectl args {args}")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)
    run = backend.run(make_cab(), ["tool"], {})
    assert run.success
    assert run.stdout == "answer=42\n"
    assert calls[-1][0] == "delete"


def test_run_failed_job_reports_container_exit_code(monkeypatch):
    backend = KubernetesBackend(poll_interval=0)

    def fake_kubectl(self, *args, input=None):
        if args[0] == "apply":
            return cp(returncode=0)
        if args[:2] == ("get", "job"):
            return cp(json.dumps({"status": {"conditions": [{"type": "Failed", "status": "True"}]}}))
        if args[:2] == ("get", "pods"):
            return cp("pod-xyz")
        if args[0] == "logs":
            return cp("boom\n")
        if args[:2] == ("get", "pod"):
            return cp("17")
        if args[0] == "delete":
            return cp(returncode=0)
        raise AssertionError(f"unexpected kubectl args {args}")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)
    run = backend.run(make_cab(), ["tool"], {})
    assert not run.success
    assert run.returncode == 17


def test_apply_failure_raises_backend_error(monkeypatch):
    def fake_kubectl(self, *args, input=None):
        if args[0] == "apply":
            return cp(returncode=1)
        raise AssertionError("should not reach further calls")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)
    with pytest.raises(BackendError):
        KubernetesBackend().run(make_cab(), ["tool"], {})


def test_job_is_cleaned_up_even_when_wait_raises(monkeypatch):
    calls = []

    def fake_kubectl(self, *args, input=None):
        calls.append(args)
        if args[0] == "apply":
            return cp(returncode=0)
        if args[:2] == ("get", "job"):
            return cp(returncode=1, stdout="")
        if args[0] == "delete":
            return cp(returncode=0)
        raise AssertionError(f"unexpected kubectl args {args}")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)
    with pytest.raises(BackendError):
        KubernetesBackend().run(make_cab(), ["tool"], {})
    assert calls[-1][0] == "delete"
