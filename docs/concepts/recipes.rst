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

Orchestration functions
------------------------

Wiring does not have to be fully declarative. Because a recipe is plain
Python, its orchestration can be a function whose body uses ordinary ``if`` and
``for``. The graph a ``--dryrun`` shows is then the *one* path actually taken
for the given inputs -- the dry run executes the real control flow with each
cab swapped for a no-op that records the call, so it never shows an untaken
branch.

The rendered graph groups independent steps that share the same set of
dependencies onto a single row (a **fan-out**), and shows steps that several
others feed into as a **fan-in** -- the same wavefront structure the executor
runs. See :doc:`../cli` for the ``--dryrun`` output.

Offloading
----------

A recipe that is *purely declarative* -- no orchestration functions, no
MUTABLE inputs, and only paths crossing between steps -- can be compiled to a
cluster workflow and detached. See :doc:`../offloading`.
