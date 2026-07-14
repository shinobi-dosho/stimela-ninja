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
* it has **no MUTABLE inputs**, and
* only **paths** cross between steps (an output wired into a later input must be
  a filesystem path knowable at compile time, not a wrangler-derived value).

Anything relying on live Python is rejected with an explanation -- run those
recipes locally with ``ninja run`` instead.

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

Once a run is done, remove its handle file and Slurm job logs with
``ninja clean --launches`` (see :ref:`ninja-clean`) -- unlike run manifests
and the step cache, this is opt-in, since deleting a handle for a
still-running job only loses your local way to check on it.

.. note::

   The Slurm engine was **not** live-verified against a real cluster in the
   development environment. See ``AGENTS.md`` in the repository for what that
   means in practice.
