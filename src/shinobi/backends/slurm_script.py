"""Shared sbatch-script grammar for `shinobi.backends.slurm` (blocking, one
job at a time) and `shinobi.offload.slurm` (compiles a whole recipe graph
into a workflow of scripts and detaches). Both submit the exact same kind
of script; unifying the grammar here means the security-hardening around
what may be interpolated into it can't drift between the two the way it
had -- the offload compiler charset-validated `cab.name`/sbatch-option
keys before writing them into a `#SBATCH` line, the backend didn't, so a
cab name from untrusted cult-cargo YAML (see SECURITY.md's "never eval()/
exec()" note) could smuggle a newline into a real Slurm submission that
never goes through the offload compiler.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_slurm_name(name: str, kind: str, *, error: type[Exception] = ValueError) -> str:
    """A name interpolated into a generated `#SBATCH` line or job-name must
    not be able to smuggle in a newline (which would inject further
    directives) -- keep it to a strict, obviously-safe charset. `error` lets
    each caller surface its own exception type (`BackendError` for the
    blocking backend, `OffloadCompileError` for the offload compiler) around
    this one shared rule.
    """
    if not _SAFE_NAME.match(name):
        raise error(
            f"{kind} {name!r} contains characters unsafe for a Slurm script "
            f"(allowed: letters, digits, '.', '_', '-')"
        )
    return name


def build_sbatch_script(
    *,
    job_name: str,
    chdir: str,
    stdout_path: Path,
    stderr_path: Path,
    sbatch_opts: dict[str, str],
    argv: list[str],
    error: type[Exception] = ValueError,
) -> str:
    """The one sbatch script grammar shared by the blocking backend and the
    offload compiler: a `#!/bin/bash` header, `#SBATCH` directives (job
    name/chdir/output/error, then any extra `sbatch_opts`), then the
    exec-form argv (`shlex.join`, never a shell template).
    """
    job_name = safe_slurm_name(job_name, "job name", error=error)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --chdir={chdir}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
    ]
    for key, value in sbatch_opts.items():
        lines.append(f"#SBATCH --{safe_slurm_name(key, 'sbatch option', error=error)}={value}")
    lines.append("")
    lines.append(shlex.join(argv))  # exec-form argv; never a shell template
    return "\n".join(lines) + "\n"


def parse_sbatch_job_id(stdout: str) -> str:
    """`sbatch --parsable`'s stdout is `<jobid>` or `<jobid>;<cluster>` --
    the job id is always the first `;`-separated field.
    """
    return stdout.strip().split(";")[0]


def sacct_job_fields(stdout: str, job_id: str) -> list[str] | None:
    """Parse `sacct --noheader --parsable2` output (whose first requested
    `--format` column must be `JobID`) and return the pipe-split fields of
    the row for `job_id` itself.

    `sacct -j <job_id>` also reports sub-step rows (`<job_id>.batch`,
    `<job_id>.extern`) alongside the bare job id row -- only the bare row
    reflects the job's own overall state, so this matches on it by exact
    field equality rather than assuming it's positionally first (a real
    backend query and the offload compiler's status check used to disagree
    on this: one filtered by exact match, the other just took `lines[0]`).
    Returns `None` if no row for `job_id` is present yet (e.g. the job
    hasn't reached the accounting database).
    """
    for line in stdout.strip().splitlines():
        fields = line.split("|")
        if fields and fields[0] == job_id:
            return fields
    return None
