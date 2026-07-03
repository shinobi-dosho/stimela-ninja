"""Kubernetes backend: submits a cab as a batch Job via `kubectl apply`,
blocks until it completes (the same synchronous contract every other
backend has), fetches its logs, then deletes the Job. Shells out to
kubectl -- matching shinobi's "runtime CLI, not an SDK" convention -- so
there's no Kubernetes client library dependency.

Volume mounts reuse the container backend's File/MS-dtype-derived
directory list, mounted into the pod as hostPath volumes. That's the
simplest thing that actually works on a single-node dev cluster (kind,
minikube) or a cluster where every worker node shares the same storage
(common on radio-astronomy k8s deployments, e.g. via NFS mounted
identically everywhere). A production multi-node cluster without shared
node storage needs PersistentVolumeClaims instead -- that's a deliberate
scope boundary, not an oversight; see AGENTS.md.

Not live-verified against a real cluster: none was available in the dev
environment this was built in, unlike the container backend, which was
checked against a real docker daemon and a real wsclean image. Treat this
as reviewed-by-construction -- verify it against a real cluster before
depending on it.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from typing import Any

from shinobi.backends import Backend, register
from shinobi.backends.container import bind_dirs
from shinobi.exceptions import BackendError
from shinobi.results import Result
from shinobi.schema import CabDef
from shinobi.wranglers import apply_wranglers

_TERMINAL_CONDITIONS = {"Complete", "Failed"}


@register
class KubernetesBackend(Backend):
    name = "kubernetes"

    def __init__(
        self,
        *,
        namespace: str = "default",
        workdir: str | None = None,
        poll_interval: float = 5.0,
    ):
        self.namespace = namespace
        self.workdir = workdir or os.getcwd()
        self.poll_interval = poll_interval

    def _manifest(
        self, cab: CabDef, argv: list[str], params: dict[str, Any], job_name: str
    ) -> dict[str, Any]:
        if not cab.image:
            raise BackendError(f"cab '{cab.name}' has no image, cannot run on kubernetes")

        dirs = bind_dirs(cab, params, self.workdir)
        volumes = [{"name": f"vol{i}", "hostPath": {"path": d}} for i, d in enumerate(dirs)]
        mounts = [{"name": f"vol{i}", "mountPath": d} for i, d in enumerate(dirs)]

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self.namespace},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": cab.name,
                                "image": cab.image,
                                "command": argv,
                                "workingDir": self.workdir,
                                "volumeMounts": mounts,
                            }
                        ],
                        "volumes": volumes,
                    }
                },
            },
        }

    def _kubectl(self, *args: str, input: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", *args, "-n", self.namespace], input=input, capture_output=True, text=True
        )

    def _wait(self, job_name: str) -> str:
        while True:
            proc = self._kubectl("get", "job", job_name, "-o", "json")
            if proc.returncode != 0:
                raise BackendError(f"kubectl get job failed: {proc.stderr.strip()}")
            status = json.loads(proc.stdout).get("status", {})
            for condition in status.get("conditions", []):
                if condition.get("status") == "True" and condition.get("type") in _TERMINAL_CONDITIONS:
                    return condition["type"]
            time.sleep(self.poll_interval)

    def _pod_name(self, job_name: str) -> str:
        proc = self._kubectl(
            "get", "pods", "-l", f"job-name={job_name}", "-o", "jsonpath={.items[0].metadata.name}"
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise BackendError(f"could not find pod for job '{job_name}': {proc.stderr.strip()}")
        return proc.stdout.strip()

    def _exit_code(self, pod_name: str) -> int:
        proc = self._kubectl(
            "get",
            "pod",
            pod_name,
            "-o",
            "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}",
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.strip())
        return 1

    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        job_name = f"shinobi-{cab.name}-{uuid.uuid4().hex[:8]}"
        manifest = self._manifest(cab, argv, params, job_name)

        apply = self._kubectl("apply", "-f", "-", input=json.dumps(manifest))
        if apply.returncode != 0:
            raise BackendError(f"kubectl apply failed: {apply.stderr.strip()}")

        try:
            condition = self._wait(job_name)
            pod_name = self._pod_name(job_name)
            logs = self._kubectl("logs", pod_name).stdout
            returncode = 0 if condition == "Complete" else self._exit_code(pod_name)
        finally:
            self._kubectl("delete", "job", job_name, "--ignore-not-found")

        outputs = apply_wranglers(cab.wranglers, logs.splitlines())
        return Result(cab_name=cab.name, returncode=returncode, stdout=logs, stderr="", outputs=outputs)
