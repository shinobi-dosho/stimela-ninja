Quickstart
==========

This walkthrough builds a tiny two-step pipeline: image a measurement set with
`WSClean <https://wsclean.readthedocs.io/>`_, then make a mask from the
resulting image with `breizorro
<https://github.com/ratt-ru/breizorro>`_. It mirrors
``examples/simple_selfcal.py`` in the repository.

Define some cabs
----------------

A :class:`~shinobi.Cab` is a typed, backend-agnostic description of an atomic
task. Give it a ``name``, the ``command`` to run, an optional container
``image``, and pydantic models for its inputs and outputs.

.. code-block:: python

    from pydantic import BaseModel

    from shinobi import Cab, Recipe, step
    from shinobi.loaders import build_model


    class ImageInputs(BaseModel):
        ms: str = "obs.ms"
        prefix: str = "img"


    class ImageOutputs(BaseModel):
        restored: str | None = None


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=ImageInputs,
        outputs_model=ImageOutputs,
    )

    breizorro = Cab(
        name="breizorro",
        command="breizorro",
        image="breizorro:latest",
        inputs_model=build_model("MaskInputs", {"restored_image": ("File", True, None)}),
        outputs_model=build_model("MaskOutputs", {"mask": ("File", False, None)}),
    )

You can hand-write the pydantic models, or build them from a compact
``{name: (dtype, required, default)}`` spec with
:func:`shinobi.loaders.build_model` (the same helper the YAML loaders use).

Run a single cab
----------------

Wrap a cab in a :func:`@shinobi.step <shinobi.step>` function to make it
runnable. The body receives an :class:`~shinobi.ExecContext` (``ctx``); calling
``ctx.run()`` executes the cab on the chosen backend and returns its result.

.. code-block:: python

    @step(wsclean, backend="native")
    def image(ctx):
        """Image the visibilities. A near-empty body auto-runs the cab."""
        return ctx.run()

Save the file as ``myrecipe.py`` and run it straight from the command line --
the cab's input schema becomes the CLI options:

.. code-block:: console

    $ ninja run myrecipe.py:image --ms data.ms --prefix out

Compose a recipe
----------------

A :class:`~shinobi.Recipe` wires steps together. Use ``add_step`` with the
``recipe.inputs`` / ``recipe.outputs`` proxies to declare the data flow: each
proxy attribute is a reference that the engine resolves at run time.

.. code-block:: python

    selfcal = Recipe(
        name="selfcal",
        inputs_model=ImageInputs,
        outputs_model=build_model("Out", {"mask": ("File", False, None)}),
    )
    selfcal.add_step("image", wsclean, ms=selfcal.inputs.ms, prefix=selfcal.inputs.prefix)
    selfcal.add_step("mask", breizorro, restored_image=selfcal.outputs.image.restored)
    selfcal.set_output("mask", selfcal.outputs.mask.mask)

Here ``selfcal.outputs.image.restored`` is the ``restored`` output of the step
named ``image`` -- wiring it into the ``mask`` step's ``restored_image`` input
creates the dependency edge between the two.

Preview the graph
-----------------

Before running anything, use ``--dryrun`` to see the execution graph the recipe
produces for a given set of inputs:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --dryrun
    [ image ]
        |
        v
    [ mask ]

Run the recipe
--------------

Drop ``--dryrun`` to execute it for real:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms

Where to next
-------------

* :doc:`concepts/cabs` -- defining cabs in Python or loading them from YAML.
* :doc:`concepts/steps` -- ``@shinobi.step`` vs ``@shinobi.pystep``.
* :doc:`concepts/recipes` -- declarative vs orchestration-function wiring.
* :doc:`concepts/backends` -- running natively, in containers, or on a cluster.
* :doc:`offloading` -- compiling a recipe to Slurm and detaching.
