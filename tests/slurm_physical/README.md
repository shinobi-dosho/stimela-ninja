# Physical M1 acceptance probe

This probe targets the Kudu/Nyala cluster documented in
`shinobi-dosho/shinobi-test-cluster`. It is intentionally separate from the
ordinary pytest suite: it submits real detached jobs and requires shared
`/data`, Slurm, Apptainer, and a pre-provisioned tool venv.

Inside the controller, prepare `/data/images/python-3.12-alpine.sif` and a
shared `/data/m1-tool-venv` containing a `venvonlypkg` module with
`MAGIC = 4242`. Then run `run_m1.py submit`, let that process exit, wait for
the reported finalizer job, and invoke `run_m1.py check` from a fresh process.

The workflow covers a binary cab, image-backed pystep and venv-backed pystep;
runtime output wiring; shared sandboxes and harvesting; immutable attempt/job
records; image, code, tool-environment and worker provenance; declaration-order
finalization; and submitter disconnect.
