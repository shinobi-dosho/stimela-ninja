Recipes
=======

A :class:`~shinobi.Recipe` composes steps into a pipeline. It is itself a
scope, so a recipe can be a step of a larger recipe, and any recipe is a valid
``ninja run`` target.

There are two ways to express the data flow between steps. Both are ordinary
Python and produce the same wired recipe.

Declarative wiring with the input/output proxies
-------------------------------------------------

``add_step`` appends a step; the ``recipe.inputs`` and ``recipe.outputs``
proxies produce references you pass as parameter values. Each reference is
resolved by the engine at run time.

.. code-block:: python

    from shinobi import Recipe
    from shinobi.loaders import build_model

    selfcal = Recipe(
        name="selfcal",
        inputs_model=ImageInputs,
        outputs_model=build_model("Out", {"mask": ("File", False, None)}),
    )
    selfcal.add_step("image", wsclean, ms=selfcal.inputs.ms, prefix=selfcal.inputs.prefix)
    selfcal.add_step("mask", breizorro, restored_image=selfcal.outputs.image.restored)
    selfcal.set_output("mask", selfcal.outputs.mask.mask)

* ``selfcal.inputs.ms`` -- the recipe's own ``ms`` input, threaded into the
  first step.
* ``selfcal.outputs.image.restored`` -- the ``restored`` output of the step
  named ``image``. Passing it as the ``mask`` step's ``restored_image`` input
  is what creates the dependency edge.
* ``set_output`` exposes a step's output as one of the recipe's own outputs.

Explicit wiring with ``StepRef`` / ``InputRef`` / ``OutputRef``
---------------------------------------------------------------

The same recipe can be built by constructing the steps and wiring directly.
This is the lower-level form the proxies desugar to, and it is convenient when
you build a recipe programmatically:

.. code-block:: python

    from shinobi.steps import InputRef, OutputRef, Recipe, StepRef

    pipe = Recipe(
        name="pipe",
        inputs_model=PipeInputs,
        outputs_model=OkOutputs,
        steps=[
            # make.out  <- the recipe's own `target` input
            StepRef(name="make", step=make, wiring={"out": InputRef(field="target")}),
            # use.f     <- make.out   (this OutputRef is the dependency edge)
            StepRef(name="use", step=use, wiring={"f": OutputRef(step="make", field="out")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )

* :class:`~shinobi.InputRef` -- a step input sourced from one of the recipe's
  own inputs.
* :class:`~shinobi.OutputRef` -- a step input (or a recipe output, in
  ``output_wiring``) sourced from another step's output. This is the edge the
  execution graph is built from.

The dependency graph
--------------------

A recipe's ``steps`` list plus the ``OutputRef`` wiring between them *is* a
directed acyclic graph (DAG). Every ``OutputRef`` is a **data-dependency
edge**: a step that consumes another step's output cannot start until that
producer has finished. Steps with no edge between them are independent, no
matter what order they were declared in.

So **declaration order is not execution order**. The edges are the data flow;
declaration order only breaks ties between steps that become ready at the same
moment (see below). Two steps that both read only the recipe's own inputs have
no edge between them and may run in either order -- or at the same time.

The graph is built and validated by ``shinobi.graph.build_graph``, which
raises ``RecipeGraphError`` on:

* a duplicated step name;
* an ``InputRef`` to a field the recipe's ``inputs_model`` does not have;
* an ``OutputRef`` (in a step's wiring or in ``output_wiring``) naming a step
  that does not exist;
* a **cycle** in the dependency edges.

Validation runs at **run time and dry-run time**, not when you call
``add_step``: a recipe is deliberately mutable while you build it, so a forward
reference to a step you have not added yet is legitimate mid-construction. The
executor and the ``--dryrun`` renderer both go through ``build_graph``, so they
can never disagree about what depends on what, or on whether the recipe is even
valid.

Because a recipe is itself a scope, a recipe used as a step of a larger recipe
is simply one node in the parent's graph; its own internal graph runs when that
node is scheduled.

Execution and concurrency
-------------------------

The executor schedules a **topological wavefront** over the true dependency
edges. A step becomes *ready* the moment every step it depends on has
completed; ready steps run on a pool of ``max_workers`` worker threads. When
more steps are ready than there are free workers, the lowest-declaration-index
step goes first -- so at ``max_workers = 1`` the recipe runs in exact
declaration order, and raising it lets independent branches run concurrently.

.. code-block:: python

    Recipe(name="pipe", inputs_model=..., outputs_model=..., max_workers=4)

``max_workers`` defaults to ``1`` (per recipe, falling back to
``AppConfig.execution.max_workers``); concurrency is opt-in. The reason is data
safety: at ``1`` no ``MUTABLE`` input can be shared between two steps running
at once. With ``max_workers > 1`` you must ensure two concurrently-running
steps never share a mutable object -- see :doc:`config`.

Declaring what a step costs
---------------------------

``max_workers`` counts slots, not cost. That is fine when steps are cheap and
interchangeable, and badly wrong when they are not: a ``wsclean`` or DDFacet
step is often sized to use most of a machine on its own, so five of them
running because five slots were free will oversubscribe the CPU and blow
through the memory the machine actually has.

A step can therefore declare a footprint, and the scheduler admits work against
a budget as well as a slot count:

.. code-block:: python

    from shinobi.resources import Resources

    recipe.add_step("selfcal", selfcal_cab, ms=recipe.inputs.ms,
                    resources=Resources(cpus=16, memory="200GiB"))

``resources=`` is also accepted by ``@recipe.step`` and ``@shinobi.step``, and
can be set directly on a ``Cab``. Sizes accept ``GB``/``GiB``/``T`` suffixes
(SI is decimal, IEC binary: ``200GB`` is 200×1000³, ``200GiB`` is 200×1024³).

The rules are deliberately few:

* **Undeclared means free.** A step that declares nothing is admitted on
  ``max_workers`` alone, exactly as before. A budget only constrains what you
  have actually described, so it is worth only as much as your declarations.
* **Admission blocks in declaration order.** If the next step does not fit, the
  scheduler waits rather than skipping ahead to a smaller one, so a large step
  cannot be starved by a stream of small ones. Across sibling recipes sharing
  one budget, the same guarantee is kept by serving waiters first-come.
* **One budget covers the whole run.** Nested recipes share the *parent's*
  budget rather than each creating their own -- otherwise a pipeline that nests
  each parallel branch as its own ``Recipe`` (a common shape) would give every
  branch its own scheduler and no shared limit at all, which is the exact
  situation this is meant to prevent. Declare footprints on the leaf steps that
  do the work; declaring them on a nested ``Recipe`` is rejected by
  ``build_graph``, since a recipe is not a unit of execution.
* **A step bigger than the whole budget still runs.** Waiting could never make
  it fit, so it is admitted immediately with a warning, and it holds everything
  else back until it finishes.

``ninja run --dryrun`` shows declared footprints in each step's box, so you can
see before running which branches will actually overlap.

Declaration is not enforcement -- locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scheduler decides *whether to start* a step; it cannot hold a running
process to its word. What it does is stop the machine from being oversubscribed
in the first place.

Actual enforcement depends on where the step runs, and the difference is worth
knowing (see :doc:`backends`): container backends and cluster backends really
do constrain the process, so a runaway dies in its own cgroup instead of taking
its siblings' memory with it. A ``native`` or ``venv`` step has **no backstop at
all** -- if its declaration is wrong, nothing catches it.

One residual gap: the "a reservation is never held by something waiting on the
budget" rule is enforced by skipping nested ``Recipe`` steps. A ``@shinobi.pystep``
or bare ``Scope`` step that declares resources *and* internally dispatches a
sub-recipe defeats that, and can deadlock. Don't do that; declare resources on
steps that do work, not on steps that orchestrate other steps.

When a step is killed
~~~~~~~~~~~~~~~~~~~~~

A process killed by a signal reports a negative exit status, and a bare ``-9``
is the least informative thing a failed step can say -- it is also exactly what
exceeding a memory limit produces. Shinobi names the signal in the failure
message, and calls out ``SIGKILL`` as the usual out-of-memory culprit. With
``provenance`` enabled the run manifest also records what each step declared,
so a post-mortem can compare the declaration against what happened.

However steps interleave, results are **deterministic**: stdout, stderr, the
aggregated outputs, and the recipe's exit code are all assembled in declaration
order, not completion order. On the first failure -- a non-zero exit or a
raised exception -- no further steps are submitted; steps already running are
allowed to finish (a launched job cannot be honestly cancelled), and the first
failure by declaration order is the one reported.

Scatter / fan-out
-----------------

A step can declare that one or more of its inputs are lists at the *recipe*
level and should be fanned out into multiple independent executions. Each slice
receives the element at the same index from every scattered field. The step's
own ``inputs_model``/``outputs_model`` describe a single slice; downstream
steps see the scattered step's outputs gathered into lists.

Use the ``scatter=`` argument of ``add_step`` or the decorators:

.. code-block:: python

    from pydantic import BaseModel
    from shinobi import Recipe

    class Inputs(BaseModel):
        mss: list[str]

    class Outputs(BaseModel):
        images: list[str]

    recipe = Recipe(name="image_all", inputs_model=Inputs, outputs_model=Outputs)
    recipe.add_step(
        "image",
        wsclean,            # wsclean.inputs_model has ms: str
        scatter=["ms"],
        ms=recipe.inputs.mss,
    )
    recipe.set_output("images", recipe.outputs.image.image)

Rules:

* every scattered field must be a list at runtime;
* all scattered fields for one step must have the same length;
* the step's schema describes one slice (e.g. ``ms: str``), not the list;
* a downstream step can scatter over a gathered output in turn, or consume the
  whole list as a single input.

Scattered recipes are not offloadable in v1.

.. _declared-loops:

Loops
-----

Some pipelines repeat a block until a result is good enough -- self-calibration
runs a calibrate/image/assess cycle until the image fidelity stops improving.
The number of cycles is a run-time decision, but a recipe is a graph declared
*before* anything runs.

``add_loop`` resolves that by **unrolling**: the body is flattened into the
recipe ``max_iter`` times and the copies are chained together, so the result is
an ordinary graph you can render and validate. Once an iteration reports
convergence, the remaining iterations pass the converged outputs straight
through without doing any work.

.. code-block:: python

    loop = recipe.add_loop(
        "selfcal",
        cycle,                              # a Recipe: calibrate -> image -> assess
        max_iter=10,
        until="converged",                  # a path output; existing == converged
        carry={"ms": "ms"},                 # this cycle's output -> next cycle's input
        index_input="cycle",                # binds the 1-based iteration number
        ms=recipe.inputs.ms,                # iteration 1's inputs
    )
    recipe.add_step("publish", pub, image=loop.outputs.image)

That declares steps ``selfcal.1.calibrate`` … ``selfcal.10.assess``.
``loop.outputs`` resolves to the final iteration, so nothing downstream has to
mention ``max_iter``.

The body does not have to be a ``Recipe``. A single ``Cab`` (or a ``StepRef``
from ``@shinobi.pystep``) is one step per iteration -- ``selfcal.1``,
``selfcal.2``, … -- with no name to flatten, and behaves identically in every
other respect, ``index_input`` included.

The convergence signal is a **path**, not a boolean: the body writes that file
when it has converged, and its existence is the test. That is what lets an
offloaded run apply the identical rule -- a boolean has no way to travel
between two Slurm jobs, but a file on shared storage does.

Rules:

* the body must be a **fixed point** -- whatever ``carry`` names must be both an
  output and an input of the body, so one iteration can feed the next;
* ``carry`` is explicit; it is not inferred from matching field names, because
  those pairs are the actual edges between iterations;
* ``index_input`` is the one input that differs between iterations; without it
  every iteration is identical, so a body cannot name its outputs per cycle;
* a body containing ``scatter`` or a further nested ``Recipe`` is rejected.

.. note::

   ``--dryrun`` shows what is **declared**, so all ``max_iter`` iterations are
   drawn even though fewer may run. What actually happened is in the run
   manifest, where short-circuited steps are recorded with ``skipped: true``.

A loop *is* offloadable -- it compiles to a plain dependency chain, with the
convergence test as a guard at the top of each job's script. The exception is a
body that names its outputs per cycle (via ``index_input`` and an ``implicit``
template): those paths cannot be known at compile time, so ``ninja compile``
rejects the recipe rather than emitting a path no job will write. Such a loop
still runs locally.

Orchestration functions
------------------------

A step's own body can still be arbitrary Python. ``@shinobi.step``/
``@recipe.step`` bind a plain function that receives ``ctx`` and decides how
(or how many times) to call ``ctx.run()`` -- to retry, post-process a
result, or auto-run with ``None``. That function only runs when the step
actually dispatches, though: it has no effect on the recipe's *declared*
graph, which is fixed by the ``InputRef``/``OutputRef`` wiring passed to
``add_step``/``@recipe.step`` when the recipe is built, and is exactly what
``--dryrun`` renders -- nothing is executed to produce it.

.. note::

   A Python ``for`` loop inside such a function is the one case worth
   resisting. It works, but the repetition is invisible: the graph shows one
   node, ``--dryrun`` cannot tell you how many cycles there might be, and the
   step cannot be offloaded at all (an orchestration function is never
   compiled to a cluster workflow). Use :ref:`add_loop <declared-loops>`
   instead -- it expresses the same thing as declared steps.

The rendered graph groups independent steps that share the same set of
dependencies onto a single row (a **fan-out**), and shows steps that several
others feed into as a **fan-in** -- the same wavefront structure the executor
runs. See :doc:`../cli` for the ``--dryrun`` output.

Offloading
----------

A recipe that is *purely declarative* -- no orchestration functions, no
MUTABLE inputs, and only paths crossing between steps -- can be compiled to a
cluster workflow and detached. See :doc:`../offloading`.
