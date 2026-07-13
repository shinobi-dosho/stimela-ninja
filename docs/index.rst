stimela-ninja
=============

**Stimela 3.0** -- a simple but flexible framework for
reproducible radio astronomy pipelines.

A spiritual successor to `Stimela classic
<https://github.com/ratt-ru/Stimela-classic>`_, built around the same core
philosophy. Recipes are plain Python: a step is a function call, and a step's
output is a Python value you wire into the next call. There is no YAML
expression/substitution language, no alias-propagation system, and no stacked
config libraries -- control flow is just Python, and it doesn't need
reinventing.

.. note::

   Early scaffolding. The interfaces documented here are real and tested
   (``pytest``), but the project is not yet ready to run real pipelines.

.. code-block:: python

    from pydantic import BaseModel

    from shinobi import Cab, step


    class ImageInputs(BaseModel):
        ms: str = "obs.ms"
        prefix: str = "img"


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=ImageInputs,
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

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/cabs
   concepts/steps
   concepts/recipes
   concepts/backends
   concepts/loaders
   concepts/config
   concepts/provenance

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

   contributing


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
