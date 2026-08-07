Offloading to a cluster
=======================

A recipe that is *purely declarative* can be compiled to a cluster workflow and
handed off, so the pipeline runs without a live ``ninja`` process babysitting
it. This is what ``ninja compile`` does.

When a recipe can be offloaded
------------------------------

Offloading requires that the whole recipe be statically knowable -- the
compiler must be able to determine every job and every dependency without
running any Python. A recipe is offload-eligible only when:

* it has **no orchestration functions** (nothing whose behaviour depends on
  live Python control flow),
* every step is a ``binary``-flavour ``Cab``,
* any **MUTABLE input is a path** (see below), and
* only **paths** cross between steps (an output wired into a later input must be
  a filesystem path knowable at compile time, not a wrangler-derived value).

Anything relying on live Python is rejected with an explanation. That is not
the end of the road for a cluster: see :ref:`offload-remote` below, which runs
any recipe on a remote host and has none of these restrictions.

.. _offload-remote:

The other path: ``ninja run --remote``
--------------------------------------

``ninja compile`` is not the only way onto a cluster, and it is the more
demanding one. ``ninja run TARGET --remote user@host:/path`` rsyncs the target
and its cab dependencies to one host, starts ``ninja run`` there detached, and
gives you a handle to poll -- no scheduler in between.

Which to reach for:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``ninja compile --submit``
     - ``ninja run --remote``
   * - Runs on
     - Many nodes, as a scheduler-managed DAG
     - One host, start to finish
   * - Recipe must be
     - Purely declarative (see above)
     - Anything ``ninja run`` can run
   * - Steps are ordered by
     - Slurm ``afterok`` dependencies
     - ninja's own scheduler, on that host

So a recipe with orchestration functions -- the kind offload rejects -- is
still perfectly runnable on the cluster's big-memory node over ``--remote``.
The cost is that one process babysits the whole pipeline there, which is
precisely what ``compile`` exists to avoid for long multi-node runs.

The environment on the far side
-------------------------------

Both paths need a Python environment on the remote host, and neither
inherits yours. ``--remote`` can build one:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --remote user@cluster:/scratch/run1 \
        --venv sync --venv-package 'caracal==2.0.1'

``--venv sync`` provisions with ``uv``, installing uv itself first if the host
has none. The environment is named by a hash of what was asked for plus the
host's architecture, libc and Python, so it is built once and reused by every
later launch, and two different hosts sharing one ``/scratch`` cannot activate
each other's. See :ref:`the --venv options <cli-remote>` for the full flag set.

Two things specific to clusters:

* **Provision from a login node.** Compute nodes commonly have no route to
  PyPI, and ``sync`` needs one. Run ``--venv sync`` once where there is egress;
  every subsequent launch can use the default ``--venv use``, which needs no
  network and never writes to the remote.
* **Provisioning executes build backends** for any source distribution
  involved, under your account. That is a real difference from a container
  image, which executes nothing when it is pulled.

``ninja compile`` has no equivalent: its jobs run whatever ``sbatch`` finds on
the node, so a container runtime (``--container-runtime``) is the reproducible
option there.

.. _offload-mutation-ordering:

In-place mutation is offloadable
--------------------------------

Self-cal pipelines rewrite one Measurement Set in place: ``flag``,
``gaincal`` and ``applycal`` each take the same MS as a plain input and
modify it. Nothing wires them together, so the declared graph sees three
*independent* steps -- run locally that is harmless, because the default
``max_workers: 1`` executes them in declaration order anyway, but handed to
a cluster as an unordered DAG they would run concurrently against the same
files.

``ninja compile`` therefore derives the missing edges itself. As it resolves
each step's inputs it records which paths that step touches and whether it
declares them ``MUTABLE``, then orders any two steps that share a path when
**at least one** of them mutates it:

* mutate-then-mutate -- the second waits for the first;
* mutate-then-read -- a reader sees the finished result;
* read-then-mutate -- the writer waits for readers of the old contents.

Two steps that only *read* the same path are left parallel, which is the
whole point of offloading them.

Because this works on **resolved values**, it does not care how each step
spells the path. A step wiring the MS from a recipe input and a step naming
the same file as a literal are recognised as touching one file, as are
``./obs.ms`` and ``/data/obs.ms``, a path neither step mentions because both
take a schema default, and ``/data/obs.ms`` versus ``/data/obs.ms/CORRECTED``
-- a Measurement Set is a directory, so containment counts.

A MUTABLE input that is *not* a path is still refused: that is a live Python
object, and no shared filesystem can carry one across a node boundary.

.. warning::

   Canonicalisation is ``Path.resolve()``, and it runs on the machine where
   ``ninja compile`` runs. Two steps reaching one MS by paths that are only
   equal *on the compute node* are therefore not recognised as sharing it,
   and no ordering edge is emitted. In practice that means a cluster where
   the submitting host and the compute nodes disagree about the filesystem:
   ``/scratch`` against ``/mnt/scratch`` under a different automount layout,
   or a symlink that resolves one way on the login node and another way on
   the node that runs the job.

   The container boundary is *not* affected -- every container backend
   identity-mounts (``-v {d}:{d}``, ``--bind {d}:{d}``, and Kubernetes
   ``mountPath == hostPath.path``), so a container-side path equals its
   host-side path by construction and comparison holds straight through.

   Closing the cross-node case needs a canonical naming the cluster itself
   agrees to, which shinobi cannot derive. Until then: give steps that share
   an MS the same spelling of its path, and prefer paths that resolve
   identically on both sides.

A :ref:`declared loop <declared-loops>` satisfies all of this: unrolling
leaves a plain dependency chain of ``Cab`` steps, and its convergence test
becomes a guard at the top of each job's script --

.. code-block:: bash

    if [ -e /scratch/converged.flag ]; then
      exit 0
    fi

-- so an iteration that runs after the loop has converged exits successfully
without doing any work, satisfying the ``afterok`` dependency so the rest of
the chain proceeds. It needs to create nothing on the way out: every path a
loop carries resolves to the same name in every iteration. A body that instead
names its outputs *per cycle* is not statically knowable and is rejected, like
anything else the compiler cannot resolve.

A minimal offloadable recipe
----------------------------

This mirrors ``examples/offload_demo.py``: two steps wired by a single
filesystem path -- ``make`` touches a file, ``use`` reads it.

.. code-block:: python

    from pathlib import Path

    from pydantic import BaseModel

    from shinobi.steps import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


    class PipeInputs(BaseModel):
        target: Path = Path("made.ms")


    class TouchInputs(BaseModel):
        out: Path


    class PathOutputs(BaseModel):
        out: Path | None = None


    class CatInputs(BaseModel):
        f: Path | None = None


    class OkOutputs(BaseModel):
        ok: bool = True


    make = Cab(name="make", command="/bin/touch", inputs_model=TouchInputs,
               outputs_model=PathOutputs, field_meta={"out": ParamMeta(positional=True)})
    use = Cab(name="use", command="/bin/cat", inputs_model=CatInputs,
              outputs_model=OkOutputs, field_meta={"f": ParamMeta(positional=True)})

    pipe = Recipe(
        name="pipe",
        inputs_model=PipeInputs,
        outputs_model=OkOutputs,
        steps=[
            StepRef(name="make", step=make, wiring={"out": InputRef(field="target")}),
            StepRef(name="use", step=use, wiring={"f": OutputRef(step="make", field="out")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )

Because the only thing crossing between steps is a path (``make``'s ``out``
output is a passthrough of its ``out`` input, so it is known statically), the
recipe is offload-eligible.

Compile it
----------

Preview the compiled Slurm workflow without submitting anything -- no cluster
needed:

.. code-block:: console

    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms --container-runtime none

This prints two ``sbatch`` scripts linked by ``--dependency=afterok``: ``make``
first, then ``use`` once ``make`` succeeds.

Or run the same recipe locally instead, driven in-process:

.. code-block:: console

    $ ninja run myrecipe.py:pipe --target /tmp/made.ms

Submit and detach
-----------------

Add ``--submit`` to hand the workflow to a real Slurm cluster and detach. A
handle file is written under ``<workdir>/.shinobi/<recipe>/handle.json``:

.. code-block:: console

    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms \
        --container-runtime none --submit

Check on it later
-----------------

``ninja status`` queries the engine fresh from the handle file -- there is no
persistent process to keep alive:

.. code-block:: console

    $ ninja status /scratch/.shinobi/pipe/handle.json

``ninja runs`` does the same for *every* launch this workspace has made, in
one table, and ``ninja logs <name> --follow`` streams a ``--remote`` run's
output until it finishes. Both reconstruct state the same way and keep
nothing running locally. See :ref:`cli-runs` and :ref:`cli-logs`.

Once a run is done, remove its handle file and Slurm job logs with
``ninja clean --launches --workdir <workdir>`` (or run it from ``<workdir>``;
see :ref:`ninja-clean`) -- unlike run manifests and the step cache, this is
opt-in, since deleting a handle for a still-running detached job doesn't
stop it, but does destroy ``ninja status``'s only local record of it.

.. note::

   ``compile``/``submit_slurm``/``status_slurm`` are live-verified
   single-node against a real Slurm controller (``tests/test_slurm_live.py``,
   plus the throwaway all-in-one cluster under ``tests/slurm_live/``) -- not
   proven multi-node, since only a single controller+node was available. The
   plain ``slurm`` *step* backend used by ``ninja run`` (as opposed to
   ``ninja compile``) is a separate code path with no live test yet; see
   :doc:`concepts/backends`.
