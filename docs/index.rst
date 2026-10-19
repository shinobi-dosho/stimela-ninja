stimela-ninja
=============


**stimela ninja (shinobi)** is the spiritual successor to the original `stimela <https://github.com/ratt-ru/Stimela-classic>`_.
Shinobi recipes are declared directed acyclic graphs built in Python: typed references wire a step's inputs
to recipe inputs or earlier step outputs, and the resulting graph can be validated and rendered before execution.
Python builders select which nodes to declare, and typed wiring records their data dependencies.

Like its sibling, `stimela2 <https://stimela.readthedocs.io>`_, it adds native, virtualenv, and Slurm backends
to the original's container backend.
Further, it introduces robust workflow support in the form of caching and mutation snapshots, provenance/replay,
and offloading to a remote server or cluster.
Backends, including live-verification status, are discussed in :doc:`concepts/backends` and :doc:`offloading`.

``stimela-ninja`` is the installable Python distribution and source repository, ``shinobi`` is its importable package,
and ``ninja`` is the command-line tool.
`dosho <https://dosho.readthedocs.io>`_ is the companion collection of native task definitions (cabs).

.. code-block:: python

    from pathlib import Path

    from dosho.cabs import breizorro
    from pydantic import BaseModel

    from shinobi import Cab, FITSAccess, FITSImage, Recipe
    from shinobi.steps import ParamMeta


    class ImageInputs(BaseModel):
        ms: Path = Path("obs.ms")
        prefix: str = "img"


    class PipelineInputs(ImageInputs):
        mask: Path = Path("mask.fits")


    class ImageOutputs(BaseModel):
        restored: FITSImage | None = None


    class PipelineOutputs(BaseModel):
        mask: FITSImage | None = None


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        backend="native",
        inputs_model=ImageInputs,
        outputs_model=ImageOutputs,
        field_meta={
            "restored": ParamMeta(implicit="{prefix}-MFS-image.fits")
        },
    )


    class MaskInputs(breizorro.inputs_model):
        restored_image: FITSImage | None = None


    class MaskOutputs(breizorro.outputs_model):
        outfile: FITSImage | None = None


    breizorro = breizorro.model_copy(update={
        "inputs_model": MaskInputs,
        "outputs_model": MaskOutputs,
        "fits_accesses": [
            FITSAccess(field="restored_image", mode="read"),
            FITSAccess(field="outfile", mode="create"),
        ],
        "backend": "native",
    })


    pipeline = Recipe(
        name="pipeline",
        inputs_model=PipelineInputs,
        outputs_model=PipelineOutputs,
    )
    pipeline.add_step("image", wsclean, ms=pipeline.inputs.ms, prefix=pipeline.inputs.prefix)
    pipeline.add_step(
        "mask",
        breizorro,
        restored_image=pipeline.outputs.image.restored,
        outfile=pipeline.inputs.mask,
    )
    pipeline.set_output("mask", pipeline.outputs.mask.outfile)

The WSClean cab is defined above, while ``breizorro`` comes from ``dosho``.
Its two FITS fields are specialized for this recipe without redefining the cab.
Passing ``pipeline.outputs.image.restored`` to ``restored_image`` wires the
masking step after imaging and makes the image available to it.
``FITSImage`` declares the image contract; ``FITSAccess`` marks Breizorro's
read and create paths. Install ``dosho`` and ``stimela-ninja[fits]`` and use
native WSClean and Breizorro executables to run this example.

.. code-block:: console

    $ ninja run myrecipe.py:pipeline --ms data.ms --prefix out --mask out-mask.fits


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
   concepts/products
   concepts/datasets
   concepts/states
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

   security
   contributing


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
