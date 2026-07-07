"""Compile a declared Recipe DAG to an external workflow engine and detach.

This is the "compile-and-offload" orchestration tier (see AGENTS.md's "DAG
offload" section and the design note): a *sibling* of the live executor in
`shinobi.steps.dispatch`, not part of it. It consumes the same
`shinobi.graph.build_graph` the local executor and the dryrun renderer use,
and only accepts recipes that pass `shinobi.graph.check_offloadable`.

v1 targets Slurm dependency chains (`sbatch --dependency=afterok`). Argo
and HyperQueue are documented future targets, not built.

`shinobi.offload.ssh` is a simpler sibling: no DAG compilation, just
"rsync the target file + its cab deps, SSH in, launch `ninja run`
detached" for `ninja run TARGET --remote user@host:/path`.
"""

from __future__ import annotations

from shinobi.offload.slurm import (
    OffloadCompileError,
    SlurmJob,
    SlurmWorkflow,
    compile_slurm,
    status_slurm,
    submit_slurm,
)
from shinobi.offload.ssh import (
    RemoteHandle,
    RemoteSpec,
    find_cab_deps,
    launch_remote,
    parse_remote,
    status_ssh,
    sync_to_remote,
)

__all__ = [
    "OffloadCompileError",
    "RemoteHandle",
    "RemoteSpec",
    "SlurmJob",
    "SlurmWorkflow",
    "compile_slurm",
    "find_cab_deps",
    "launch_remote",
    "parse_remote",
    "status_slurm",
    "status_ssh",
    "submit_slurm",
    "sync_to_remote",
]
