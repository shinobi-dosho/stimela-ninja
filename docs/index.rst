Stimela Ninja
=============

**Stimela Ninja** helps you turn radio astronomy processing tasks into
repeatable pipelines. Connect tools such as WSClean, CASA and QuartiCal in a
Python recipe, then run the recipe from the command line.

A recipe describes the steps to run and how their inputs and outputs connect.
For example, an imaging step can pass its image to a masking step. Shinobi
works out the order, checks the inputs, and runs the tools.

What you can do
---------------

* Run a single tool or combine several tools in a recipe.
* Preview the steps before starting a run.
* Run tools on your computer, in containers, or on a computing cluster.
* Reuse cached results and keep a record of how a run was made.

Ready-made tool definitions come from `Dosho
<https://dosho.readthedocs.io/>`_, which is included with Stimela Ninja.
Each definition is called a **cab**: it describes a tool's inputs, outputs
and how to call it.

Start here
----------

1. Follow :doc:`installation` to install Stimela Ninja and check it works.
2. Use :doc:`quickstart` to build an imaging and masking recipe.
3. See :doc:`concepts/backends` when choosing where to run your tools.

Names you will see
------------------

* **Stimela Ninja** is the project; install it as ``stimela-ninja``.
* **Shinobi** is its Python package: ``from shinobi import Recipe``.
* **ninja** is the command you use in a terminal.

Stimela Ninja builds on `Stimela classic
<https://github.com/ratt-ru/Stimela-classic>`_. If you already use Stimela 2
or CARACal, start with :doc:`migration`.

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
