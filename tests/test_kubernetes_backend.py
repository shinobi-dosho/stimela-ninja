import json
import subprocess

import pytest

from shinobi.backends.kubernetes import KubernetesBackend
from shinobi.exceptions import BackendError
from shinobi.schema import CabDef, ParamSchema


def make_cab(**kwargs) -> CabDef:
    kwargs.setdefault("name", "tool")
    kwargs.setdefault("command", "tool")
    kwargs.setdefault("image", "tool:latest")
    return CabDef(**kwargs)


def cp(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_manifest_is_a_job_with_image_and_command():
    backend = KubernetesBackend(namespace="ns", workdir="/work")
    cab = make_cab()
    manifest = backend._manifest(cab, ["tool", "--x", "1"], {}, "job-abc")

    assert manifest["kind"] == "Job"
    assert manifest["metadata"] == {"name": "job-abc", "namespace": "ns"}
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "tool:latest"
    assert container["command"] == ["tool", "--x", "1"]
    assert container["workingDir"] == "/work"


def test_manifest_no_image_raises():
    backend = KubernetesBackend()
    cab = make_cab(image=None)
    with pytest.raises(BackendError):
        backend._manifest(cab, ["tool"], {}, "job-abc")


def test_manifest_mounts_file_like_params_as_hostpath_volumes():
    backend = KubernetesBackend(workdir="/work")
    cab = make_cab(inputs={"ms": ParamSchema(dtype="MS")})
    manifest = backend._manifest(cab, ["tool"], {"ms": "/data/foo.ms"}, "job-abc")

    spec = manifest["spec"]["template"]["spec"]
    hostpaths = {v["hostPath"]["path"] for v in spec["volumes"]}
    assert hostpaths == {"/work", "/data"}
    mountpaths = {m["mountPath"] for m in spec["containers"][0]["volumeMounts"]}
    assert mountpaths == {"/work", "/data"}


def test_run_end_to_end(monkeypatch):
    backend = KubernetesBackend(poll_interval=0)
    cab = make_cab(wranglers={r"answer=(?P<n>\d+)": ["PARSE_OUTPUT:n:int"]})

    calls = []

    def fake_kubectl(self, *args, input=None):
        calls.append(args)
        if args[0] == "apply":
            return cp(returncode=0)
        if args[:2] == ("get", "job"):
            status = {"conditions": [{"type": "Complete", "status": "True"}]}
            return cp(json.dumps({"status": status}))
        if args[:2] == ("get", "pods"):
            return cp("pod-xyz")
        if args[0] == "logs":
            return cp("answer=42\n")
        if args[0] == "delete":
            return cp(returncode=0)
        raise AssertionError(f"unexpected kubectl args {args}")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)

    result = backend.run(cab, ["tool"], {})

    assert result.success
    assert result.outputs["n"] == 42
    assert result.stdout == "answer=42\n"
    # the Job is always cleaned up, even on the success path
    assert calls[-1][0] == "delete"


def test_run_failed_job_reports_container_exit_code(monkeypatch):
    backend = KubernetesBackend(poll_interval=0)
    cab = make_cab()

    def fake_kubectl(self, *args, input=None):
        if args[0] == "apply":
            return cp(returncode=0)
        if args[:2] == ("get", "job"):
            status = {"conditions": [{"type": "Failed", "status": "True"}]}
            return cp(json.dumps({"status": status}))
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

    result = backend.run(cab, ["tool"], {})
    assert not result.success
    assert result.returncode == 17


def test_apply_failure_raises_backend_error(monkeypatch):
    backend = KubernetesBackend()
    cab = make_cab()

    def fake_kubectl(self, *args, input=None):
        if args[0] == "apply":
            return cp(returncode=1)
        raise AssertionError("should not reach further calls")

    monkeypatch.setattr(KubernetesBackend, "_kubectl", fake_kubectl)

    with pytest.raises(BackendError):
        backend.run(cab, ["tool"], {})


def test_job_is_cleaned_up_even_when_wait_raises(monkeypatch):
    backend = KubernetesBackend()
    cab = make_cab()
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
        backend.run(cab, ["tool"], {})

    assert calls[-1][0] == "delete"
