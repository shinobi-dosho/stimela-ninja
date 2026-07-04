# Live Slurm test cluster

A throwaway, single-node Slurm controller for `tests/test_slurm_live.py` —
the only test that exercises the offload path (`shinobi.offload.slurm`)
against a **real** `sbatch`/`sacct`. It's skipped unless this cluster is up,
so `uv run pytest` stays green without it.

## Why a shim, not our code inside the container

`submit_slurm`/`status_slurm` shell out to `sbatch`/`sacct` directly. Rather
than install our Python stack into the Slurm image, we run our code on the
host and forward those two commands into the container via the PATH-shims in
`bin/` (`docker exec`). For that to work the script/log/data paths must be
identical on both sides, so the host workdir is bind-mounted at the same
absolute path in the container (see `docker-compose.yml`).

## What it proves (and doesn't)

Proves: real submission, `--dependency=afterok` actually gating a downstream
step, `sacct` state/exit-code parsing, and a file written by one step being
visible to the next through the shared workdir. Single-node only — it does
**not** prove multi-node scheduling or cross-node shared storage.

## Run it

```sh
# from the repo root
export SHINOBI_SLURM_WORKDIR="$PWD/tests/.slurmwork"
mkdir -p "$SHINOBI_SLURM_WORKDIR"

docker compose -f tests/slurm_live/docker-compose.yml up -d
# wait a few seconds for slurmctld/slurmdbd to accept connections

SHINOBI_SLURM_WORKDIR="$SHINOBI_SLURM_WORKDIR" uv run pytest tests/test_slurm_live.py -v

docker compose -f tests/slurm_live/docker-compose.yml down
```

The test auto-detects the cluster (via `docker exec shinobi-slurm sinfo`)
and puts `bin/` on `PATH` itself; it skips cleanly if `SHINOBI_SLURM_WORKDIR`
is unset or the container isn't running. Override the container name with
`SHINOBI_SLURM_CONTAINER` if you change it.
