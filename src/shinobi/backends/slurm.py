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
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from shinobi.backends import Backend, register
from shinobi.backends.container import build_container_argv
from shinobi.exceptions import BackendError
from shinobi.results import Result
from shinobi.schema import CabDef
from shinobi.wranglers import apply_wranglers

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
    name = "slurm"

    def __init__(
        self,
        *,
        container_runtime: str | None = "apptainer",
        workdir: str | None = None,
        sbatch_opts: dict[str, str] | None = None,
        poll_interval: float = 5.0,
    ):
        self.container_runtime = container_runtime
        self.workdir = workdir or os.getcwd()
        self.sbatch_opts = sbatch_opts or {}
        self.poll_interval = poll_interval

    def _inner_argv(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> list[str]:
        if cab.image and self.container_runtime:
            return build_container_argv(self.container_runtime, cab, argv, params, self.workdir)
        return argv

    def _script(
        self,
        cab: CabDef,
        argv: list[str],
        params: dict[str, Any],
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={cab.name}",
            f"#SBATCH --chdir={self.workdir}",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
        ]
        for key, value in self.sbatch_opts.items():
            lines.append(f"#SBATCH --{key}={value}")
        lines.append("")
        lines.append(shlex.join(self._inner_argv(cab, argv, params)))
        return "\n".join(lines) + "\n"

    def _submit(self, script_path: Path) -> str:
        proc = subprocess.run(["sbatch", "--parsable", str(script_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise BackendError(f"sbatch failed: {proc.stderr.strip()}")
        return proc.stdout.strip().split(";")[0]

    def _wait(self, job_id: str) -> int:
        while True:
            proc = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.strip().splitlines():
                fields = line.split("|")
                if len(fields) < 3:
                    continue
                job_field, state, exit_code = fields[0], fields[1], fields[2]
                # sacct also reports steps like "<id>.batch"/"<id>.extern" --
                # only the bare job id line reflects the job's overall state
                if job_field != job_id:
                    continue
                state = state.strip().split()[0]  # strip suffixes e.g. "CANCELLED by 1000"
                if state in _TERMINAL_STATES:
                    return int(exit_code.split(":")[0])
            time.sleep(self.poll_interval)

    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "job.sh"
            stdout_path = tmp_path / "stdout.log"
            stderr_path = tmp_path / "stderr.log"
            script_path.write_text(self._script(cab, argv, params, stdout_path, stderr_path))

            job_id = self._submit(script_path)
            returncode = self._wait(job_id)

            stdout = stdout_path.read_text() if stdout_path.exists() else ""
            stderr = stderr_path.read_text() if stderr_path.exists() else ""

        lines = stdout.splitlines() + stderr.splitlines()
        outputs = apply_wranglers(cab.wranglers, lines)
        return Result(cab_name=cab.name, returncode=returncode, stdout=stdout, stderr=stderr, outputs=outputs)
