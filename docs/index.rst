stimela-ninja
=============

**Stimela 3.0** -- a simple but flexible framework for
reproducible radio astronomy pipelines.

A spiritual successor to `Stimela classic
<https://github.com/ratt-ru/Stimela-classic>`_, built around the same core
philosophy (see :doc:`design` for the full rationale). Recipes are declared
directed acyclic graphs built in Python: typed references wire a step's inputs
to recipe inputs or earlier step outputs, and the resulting graph can be
validated and rendered before execution. There is no YAML
expression/substitution language, alias-propagation system, or second set of
control-flow semantics hidden in configuration.

The execution layer is usable end to end: native, virtualenv, container,
Slurm, and Kubernetes backends; concurrent and resource-aware scheduling;
scatter and bounded declared loops; sandboxed execution; caching and mutation
snapshots; provenance/replay; and remote or compiled cluster runs. Backend
limitations and live-verification status are stated in :doc:`concepts/backends`
and :doc:`offloading`.

.. code-block:: python

    from pathlib import Path

    from pydantic import BaseModel

    from shinobi import Cab, step
    from shinobi.steps import ParamMeta


    class ImageInputs(BaseModel):
        ms: Path = Path("obs.ms")
        prefix: str = "img"


    class ImageOutputs(BaseModel):
        restored: Path | None = None


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=ImageInputs,
        outputs_model=ImageOutputs,
        field_meta={
            "restored": ParamMeta(implicit="{prefix}-MFS-image.fits")
        },
    )


    @step(wsclean, backend="native")
    def image(ctx):
        return ctx.run()

.. code-block:: console

    $ ninja run myrecipe.py:image --ms data.ms --prefix out


.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   migration

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/cabs
   concepts/steps
   concepts/recipes
   concepts/backends
   concepts/loaders
   concepts/config
   concepts/results
   concepts/provenance
   concepts/sandbox

.. toctree::
   :maxdepth: 2
   :caption: Using ninja

   cli
   offloading

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   design
   security
   contributing


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
