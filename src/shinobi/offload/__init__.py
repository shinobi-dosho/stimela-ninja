"""Compile a declared Recipe DAG to an external workflow engine and detach.

This is the "compile-and-offload" orchestration tier (see AGENTS.md's "DAG
offload" section and the design note): a *sibling* of the live executor in
`shinobi.steps.dispatch`, not part of it. It consumes the same
`shinobi.graph.build_graph` the local executor and the dryrun renderer use,
and only accepts recipes that pass `shinobi.graph.check_offloadable`.

v1 targets Slurm dependency chains (`sbatch --dependency=afterok`). Argo
and HyperQueue are documented future targets, not built.
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

__all__ = [
    "OffloadCompileError",
    "SlurmJob",
    "SlurmWorkflow",
    "compile_slurm",
    "status_slurm",
    "submit_slurm",
]
