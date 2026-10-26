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
`dosho <https://dosho.readthedocs.io>`_ provides task definitions (cabs),
parameter validation and argv formatting. It is installed automatically with
Shinobi. Shinobi owns workflow execution, filesystem enforcement and dataset
lifecycle; see :doc:`design_cab_boundary` for the ownership contract.

.. code-block:: python

    from pathlib import Path

    from dosho.cabs import breizorro
    from pydantic import BaseModel

    from shinobi import Cab, FITSImage, Recipe
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


    breizorro = breizorro.model_copy(update={"backend": "native"})

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
Its FITS annotations and read/create declarations come from dosho.
Passing ``pipeline.outputs.image.restored`` to ``restored_image`` wires the
masking step after imaging and makes the image available to it.
Install Shinobi with the ``fits`` extra as described in
:doc:`installation` in the environment performing inspection. Its Dosho
dependency includes the Breizorro FITS declarations. This example selects ``native`` explicitly and requires
host-installed WSClean and Breizorro executables. Cabs without an explicit
backend use Apptainer by default.

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
   design_cab_boundary


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
