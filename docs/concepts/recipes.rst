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

Orchestration functions
------------------------

Wiring does not have to be fully declarative. Because a recipe is plain
Python, its orchestration can be a function whose body uses ordinary ``if`` and
``for`` -- the graph a ``--dryrun`` shows is the *one* path actually taken for
the given inputs, not a static declaration of all possible branches. See
:doc:`../cli` for how ``--dryrun`` renders the graph, and ``AGENTS.md`` in the
repository for how fan-out/fan-in is detected.

Offloading
----------

A recipe that is *purely declarative* -- no orchestration functions, no
MUTABLE inputs, and only paths crossing between steps -- can be compiled to a
cluster workflow and detached. See :doc:`../offloading`.
