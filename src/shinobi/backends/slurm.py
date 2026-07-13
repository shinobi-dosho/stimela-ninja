"""Slurm backend: submits a cab as a batch job via sbatch, blocks until it
finishes, then returns a Result exactly like every other backend does.
Blocking (rather than fire-and-forget) matches every other backend here,
because recipes are plain Python -- the next line of a recipe usually
needs this step's Result before it can run.

Shells out to sbatch/sacct, matching the "shell out to the runtime CLI,
not an SDK" convention used by the container backends. On a real cluster,
a cab with an image normally still needs a container runtime to actually
execute it -- Slurm schedules compute, it doesn't run containers itself --
so this reuses the container backend's own argv-wrapping (apptainer by
default, the common choice on HPC clusters that also run Slurm).

Not live-verified against a real cluster: none was available in the dev
environment this was built in, unlike the container backend, which was
checked against a real docker daemon and a real wsclean image. Treat this
as reviewed-by-construction -- verify it against a real cluster before
depending on it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, register
from shinobi.backends.container import build_container_argv
from shinobi.backends.slurm_script import build_sbatch_script, parse_sbatch_job_id, sacct_job_fields
from shinobi.exceptions import BackendError
from shinobi.results import BackendRun
from shinobi.steps.schema import Cab

# sacct job states that mean the job is done and won't change again.
# ExitCode is only meaningful once a job reaches one of these.
_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
}


@register
class SlurmBackend(Backend):
    """Backend that runs cabs as blocking Slurm batch jobs via `sbatch`."""

    name = "slurm"

    def __init__(
        self,
        *,
        container_runtime: str | None = "apptainer",
        workdir: str | None = None,
        sbatch_opts: dict[str, str] | None = None,
        poll_interval: float = 5.0,
    ):
        """Initialize the backend.

        Args:
            container_runtime: Container runtime used to wrap cabs that
                declare an image (e.g. `"apptainer"`). Set to `None` to run
                the cab's argv directly, without a container.
            workdir: Working directory the job runs in. Defaults to the
                current working directory.
            sbatch_opts: Extra `#SBATCH` options to include in the job script.
            poll_interval: Seconds to wait between `sacct` status polls.
        """
        self.container_runtime = container_runtime
        self.workdir = workdir or os.getcwd()
        self.sbatch_opts = sbatch_opts or {}
        self.poll_interval = poll_interval

    def _inner_argv(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, pin: bool = False
    ) -> tuple[list[str], str | None]:
        """The argv the job runs, and the pinned image digest (or `None` for
        a non-container or unpinned run). Digest resolution is memoized, so
        resolving it here and again in `run` is a single registry round-trip.
        """
        if cab.image and self.container_runtime:
            return build_container_argv(self.container_runtime, cab, argv, inputs, self.workdir, pin=pin)
        return argv, None

    def _script(
        self,
        cab: Cab,
        argv: list[str],
        inputs: dict[str, Any],
        stdout_path: Path,
        stderr_path: Path,
        *,
        pin: bool = False,
    ) -> str:
        return build_sbatch_script(
            job_name=cab.name,
            chdir=self.workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            sbatch_opts=self.sbatch_opts,
            argv=self._inner_argv(cab, argv, inputs, pin=pin)[0],
            error=BackendError,
        )

    def _submit(self, script_path: Path) -> str:
        proc = subprocess.run(["sbatch", "--parsable", str(script_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise BackendError(f"sbatch failed: {proc.stderr.strip()}")
        return parse_sbatch_job_id(proc.stdout)

    def _wait(self, job_id: str) -> int:
        while True:
            proc = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
            )
            fields = sacct_job_fields(proc.stdout, job_id)
            if fields and len(fields) >= 3:
                state = fields[1].strip().split()[0]  # strip suffixes e.g. "CANCELLED by 1000"
                if state in _TERMINAL_STATES:
                    return int(fields[2].split(":")[0])
            time.sleep(self.poll_interval)

    def run(
        self, cab: Cab, argv: list[str], inputs: dict[str, Any], *, label: str = "", stream: bool = True,
        pin: bool = False,
    ) -> BackendRun:
        """Submit a cab as a Slurm job and block until it terminates.

        Args:
            cab: The cab being executed.
            argv: Resolved command-line arguments to run in the job.
            inputs: Prepared inputs dict used to derive container bind mounts,
                if `container_runtime` and `cab.image` are both set.
            label: Unused; slurm has no log-tailing/streaming support.
            stream: Unused; slurm has no log-tailing/streaming support.
            pin: Digest-pin the container image before submitting (provenance).

        Returns:
            A `BackendRun` with the job's exit code and captured stdout/stderr.

        Raises:
            BackendError: If `sbatch` submission fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "job.sh"
            stdout_path = tmp_path / "stdout.log"
            stderr_path = tmp_path / "stderr.log"
            script_path.write_text(self._script(cab, argv, inputs, stdout_path, stderr_path, pin=pin))

            job_id = self._submit(script_path)
            returncode = self._wait(job_id)

            stdout = stdout_path.read_text() if stdout_path.exists() else ""
            stderr = stderr_path.read_text() if stderr_path.exists() else ""

        # Memoized, so this doesn't re-hit the registry after _script's call.
        _, image_digest = self._inner_argv(cab, argv, inputs, pin=pin)
        containerized = bool(cab.image and self.container_runtime)
        return BackendRun(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            image_digest=image_digest,
            containerized=containerized,
        )
